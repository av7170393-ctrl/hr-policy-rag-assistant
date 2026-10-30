import unittest

from app import config
from app.rag import indexer
from app.rag.citations import best_sentence, highlight
from app.rag.textutils import rrf, split_text, tokenize


class TextUtilsTests(unittest.TestCase):
    def test_tokenize_drops_stopwords_and_plurals(self):
        self.assertEqual(tokenize("How many leaves do the employees get?"), ["leave", "employee"])

    def test_split_text_respects_size_and_overlap(self):
        text = " ".join(f"Sentence number {i} is here." for i in range(40))
        chunks = split_text(text, size=200)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(c) <= 260 for c in chunks))
        self.assertIn("Sentence number 0", chunks[0])

    def test_rrf_prefers_items_ranked_high_in_both_lists(self):
        self.assertEqual(rrf([["a", "b", "c"], ["b", "a", "d"]])[:2], ["a", "b"])
        self.assertEqual(rrf([["x", "y"], ["y", "z"]])[0], "y")

    def test_best_sentence_and_highlight(self):
        chunk = "Casual leave is 12 days per year. It cannot be carried forward. Sick leave is 10 days."
        s = best_sentence(chunk, "can casual leave be carried forward")
        self.assertEqual(s, "It cannot be carried forward.")
        self.assertIn("**It cannot be carried forward.**", highlight(chunk, s))


class AccessLevelTests(unittest.TestCase):
    def test_chunks_carry_access_levels(self):
        chunks = indexer.make_chunks(config.DATA_DIR, indexer.load_access())
        levels = {c["source"]: c["level"] for c in chunks}
        self.assertEqual(levels["leave_policy.md"], 1)
        self.assertEqual(levels["manager_guidelines.md"], 2)
        self.assertEqual(levels["salary_bands_confidential.md"], 3)
        self.assertTrue(all(c["text"].strip() for c in chunks))


if __name__ == "__main__":
    unittest.main()
