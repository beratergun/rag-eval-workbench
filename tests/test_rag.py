import unittest

from rag_eval import *


class T(unittest.TestCase):
    def setUp(self):
        self.c = RetrievalCase("q", frozenset({"a", "c"}), ("a", "b", "c"))

    def test_precision(self): self.assertEqual(precision_at_k(self.c, 2), .5)
    def test_recall(self): self.assertEqual(recall_at_k(self.c, 3), 1)
    def test_rr(self): self.assertEqual(reciprocal_rank(self.c), 1)
    def test_ndcg(self): self.assertTrue(0 <= ndcg_at_k(self.c, 3) <= 1)

    def test_duplicate_retrieval_does_not_receive_repeated_credit(self):
        case = RetrievalCase("q", frozenset({"a"}), ("a", "a", "b"))
        self.assertLessEqual(ndcg_at_k(case, 3), 1)
        self.assertEqual(recall_at_k(case, 3), 1)

    def test_empty_relevance_is_bounded(self):
        case = RetrievalCase("q", frozenset(), ("a",))
        self.assertEqual(recall_at_k(case, 1), 0)
        self.assertEqual(ndcg_at_k(case, 1), 0)

    def test_bad_k(self):
        with self.assertRaises(ValueError): precision_at_k(self.c, 0)

    def test_empty_evaluation_rejected(self):
        with self.assertRaises(ValueError): evaluate([], 3)

    def test_ground(self):
        self.assertGreater(lexical_groundedness("alpha beta gamma", ["alpha beta"]), .5)

    def test_groundedness_is_lexical_not_factuality(self):
        self.assertEqual(lexical_groundedness("unseen tokens", ["different context"]), 0)

    def test_invalid_context_type(self):
        with self.assertRaises(TypeError): lexical_groundedness("answer", "not-a-list")


if __name__ == "__main__":
    unittest.main()
