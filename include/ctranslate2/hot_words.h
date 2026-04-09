#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include <memory>

#include "decoding_utils.h"

namespace ctranslate2 {

  // Character-level trie for hot words.
  // Supports lookup of all hot words that have a given string as a prefix.
  struct HotWordsTrieNode {
    std::unordered_map<char, std::unique_ptr<HotWordsTrieNode>> children;
    bool is_terminal = false;    // true if a complete hot word ends here
    float bias_weight = 0.0f;    // bias for completing this hot word

    void insert(const std::string& word, float weight, size_t pos = 0) {
      if (pos >= word.size()) {
        is_terminal = true;
        bias_weight = weight;
        return;
      }
      char c = word[pos];
      if (!children.count(c))
        children[c] = std::make_unique<HotWordsTrieNode>();
      children[c]->insert(word, weight, pos + 1);
    }

    // Advance through the trie by one character. Returns nullptr if no match.
    const HotWordsTrieNode* advance(char c) const {
      auto it = children.find(c);
      return it != children.end() ? it->second.get() : nullptr;
    }
  };

  // LogitsProcessor that biases decoding toward hot words using character-level matching.
  //
  // At each decoding step, for each beam hypothesis:
  //   1. Decode the generated token sequence to text (using the vocabulary map)
  //   2. Find all suffixes of the text that match a prefix in the hot words trie
  //   3. For each candidate next token, check if appending its text would
  //      continue or complete a hot word match
  //   4. Boost matching tokens' logits
  //
  // This allows biasing toward novel words (e.g. "Kynetec") even when the model
  // has never seen them, because the matching operates at the character level
  // rather than the token level.
  class HotWordsBiaser : public LogitsProcessor {
  public:
    // vocab_map: token ID -> decoded text string for each token in the vocabulary
    // hot_words: list of (word, bias_weight) pairs
    HotWordsBiaser(std::vector<std::string> vocab_map,
                   const std::vector<std::pair<std::string, float>>& hot_words);

    void apply(dim_t step,
               StorageView& logits,
               DisableTokens& disable_tokens,
               BiasTokens& bias_tokens,
               const StorageView& sequences,
               const std::vector<dim_t>& batch_offset,
               const std::vector<std::vector<size_t>>* prefix) override;

  private:
    HotWordsTrieNode _trie;
    std::vector<std::string> _vocab_map;  // token ID -> text

    // Given the decoded text so far, find all active trie positions
    // (i.e., suffixes of the text that are prefixes of hot words)
    std::vector<const HotWordsTrieNode*> find_active_nodes(const std::string& text) const;
  };

}
