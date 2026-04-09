#include "ctranslate2/hot_words.h"

#include <numeric>

namespace ctranslate2 {

  HotWordsBiaser::HotWordsBiaser(
      std::vector<std::string> vocab_map,
      const std::vector<std::pair<std::string, float>>& hot_words)
    : _vocab_map(std::move(vocab_map))
  {
    for (const auto& [word, weight] : hot_words) {
      if (!word.empty()) {
        _trie.insert(" " + word, weight);
        _trie.insert(word, weight);
      }
    }
  }

  std::vector<const HotWordsTrieNode*>
  HotWordsBiaser::find_active_nodes(const std::string& text) const {
    std::vector<const HotWordsTrieNode*> active;

    // Check all suffixes of the text — we want to find every position
    // where we're partway through a hot word.
    // Limit suffix length to avoid excessive work (hot words are typically < 40 chars).
    const size_t max_suffix = std::min(text.size(), size_t(50));
    for (size_t start = text.size() - max_suffix; start < text.size(); ++start) {
      const HotWordsTrieNode* node = &_trie;
      bool matched = true;
      for (size_t i = start; i < text.size(); ++i) {
        node = node->advance(text[i]);
        if (!node) {
          matched = false;
          break;
        }
      }
      if (matched && node)
        active.push_back(node);
    }

    // Also add the root — tokens that START a new hot word match.
    active.push_back(&_trie);

    return active;
  }

  void HotWordsBiaser::apply(
      dim_t /*step*/,
      StorageView& logits,
      DisableTokens& /*disable_tokens*/,
      BiasTokens& /*bias_tokens*/,
      const StorageView& sequences,
      const std::vector<dim_t>& /*batch_offset*/,
      const std::vector<std::vector<size_t>>* /*prefix*/)
  {
    if (!sequences || _vocab_map.empty())
      return;

    // We need direct additive access to logits (BiasTokens uses multiplication
    // which doesn't work for boosting low-probability tokens).
    float* logits_data = logits.data<float>();
    if (!logits_data)
      return;  // GPU not supported for now

    const dim_t batch_size = sequences.dim(0);
    const dim_t seq_length = sequences.dim(1);
    const dim_t vocab_size = logits.dim(-1);

    for (dim_t batch_id = 0; batch_id < batch_size; ++batch_id) {
      // Decode the token sequence to text.
      std::string decoded_text;
      decoded_text.reserve(seq_length * 4);
      for (dim_t t = 0; t < seq_length; ++t) {
        const auto token_id = static_cast<size_t>(sequences.at<int32_t>({batch_id, t}));
        if (token_id < _vocab_map.size())
          decoded_text += _vocab_map[token_id];
      }

      // Find all active trie positions for the current text.
      auto active_nodes = find_active_nodes(decoded_text);

      // Check if the text ENDS with a completed hot word — if so, we want to
      // boost word-boundary tokens to prevent the model from extending
      // the word (e.g. "Kynetec" -> "Kynetech").
      bool just_completed_hot_word = false;
      float completion_weight = 0.0f;
      for (size_t start = (decoded_text.size() > 50 ? decoded_text.size() - 50 : 0);
           start < decoded_text.size(); ++start) {
        const HotWordsTrieNode* node = &_trie;
        for (size_t i = start; i < decoded_text.size(); ++i) {
          node = node->advance(decoded_text[i]);
          if (!node) break;
        }
        if (node && node->is_terminal) {
          just_completed_hot_word = true;
          completion_weight = node->bias_weight;
          break;
        }
      }

      // For each token in the vocabulary, check if it would continue
      // or complete any active hot word match.
      const dim_t logit_offset = batch_id * vocab_size;

      for (size_t token_id = 0; token_id < static_cast<size_t>(vocab_size) && token_id < _vocab_map.size(); ++token_id) {
        const auto& token_text = _vocab_map[token_id];
        if (token_text.empty())
          continue;

        float best_bias = 0.0f;

        for (const auto* node : active_nodes) {
          const HotWordsTrieNode* current = node;
          bool valid = true;
          bool completed_at_end = false;
          size_t char_idx = 0;
          for (char c : token_text) {
            current = current->advance(c);
            if (!current) {
              valid = false;
              break;
            }
            char_idx++;
            if (current->is_terminal) {
              if (char_idx == token_text.size()) {
                // Token exactly ends at hot word completion — strong bias.
                // Use 2x weight for exact completion to overcome acoustic
                // evidence for similar-sounding alternatives.
                best_bias = std::max(best_bias, current->bias_weight * 2.0f);
                completed_at_end = true;
              }
              // Token extends PAST the hot word (e.g. "ech" for "ec") —
              // suppress with negative bias to prevent wrong endings.
              else {
                best_bias = std::min(best_bias, -current->bias_weight);
              }
            }
          }

          // Still on a valid trie path — apply continuation bias.
          // Scale by a fraction of the hot word weight so stronger-weighted
          // hot words get stronger continuation boosts.
          if (valid && current && !current->children.empty()) {
            // Use the deepest terminal weight reachable from this trie path.
            // Walk down to find the max weight among descendants.
            float max_weight = 0.0f;
            std::function<void(const HotWordsTrieNode*)> find_max;
            find_max = [&](const HotWordsTrieNode* n) {
              if (n->is_terminal)
                max_weight = std::max(max_weight, n->bias_weight);
              for (const auto& [_, child] : n->children)
                find_max(child.get());
            };
            find_max(current);
            // Scale: full weight for continuation, half for starting a new match
            float continuation = (node == &_trie) ? max_weight * 0.8f : max_weight;
            best_bias = std::max(best_bias, continuation);
          }
        }

        if (best_bias > 0.0f)
          logits_data[logit_offset + token_id] += best_bias;

        // After a completed hot word, boost word-boundary tokens and
        // suppress letter-continuation tokens to prevent extensions.
        if (just_completed_hot_word && !token_text.empty()) {
          char first = token_text[0];
          if (first == ' ' || first == '.' || first == ',' || first == '?' ||
              first == '!' || first == ':' || first == ';') {
            logits_data[logit_offset + token_id] += completion_weight;
          }
        }
      }
    }
  }

}
