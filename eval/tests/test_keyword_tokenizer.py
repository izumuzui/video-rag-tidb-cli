from __future__ import annotations

import unittest

from video_rag_cli import tokenize_keyword_query


class KeywordTokenizerTest(unittest.TestCase):
    def test_removes_stopwords_and_lowercases(self) -> None:
        self.assertEqual(
            tokenize_keyword_query("Chef makes pizza and cuts it up."),
            ["chef", "makes", "pizza", "cuts"],
        )

    def test_deduplicates_tokens(self) -> None:
        self.assertEqual(
            tokenize_keyword_query("Pizza pizza PIZZA with chef"),
            ["pizza", "chef"],
        )

    def test_fallback_for_non_empty_query(self) -> None:
        self.assertEqual(tokenize_keyword_query(" 料理 "), ["料理"])


if __name__ == "__main__":
    unittest.main()
