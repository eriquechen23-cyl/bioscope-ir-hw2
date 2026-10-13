"""Hand-calculated checks usable before the scientific stack is installed."""
import unittest


class EmbeddingCoreTests(unittest.TestCase):
    def test_context_samples_have_correct_direction(self):
        from bioir.embedding import training_samples
        cbow, sg = training_samples(["insulin", "lowers", "glucose"], 1)
        self.assertEqual(cbow[1], {"context": ["insulin", "glucose"], "target": "lowers"})
        self.assertEqual(len(cbow), 3)
        self.assertEqual(sg, [{"input": "insulin", "target": "lowers"}, {"input": "lowers", "target": "insulin"}, {"input": "lowers", "target": "glucose"}, {"input": "glucose", "target": "lowers"}])

    def test_vocabulary_min_count_and_sentence_boundary(self):
        from bioir.embedding import vocabulary_statistics
        actual = vocabulary_statistics([{"abstract": "Insulin insulin GLP-1. Rare."}], 2)
        self.assertEqual({k: actual[k] for k in ["documents", "sentences", "tokens", "vocabulary", "retained_vocabulary"]},
                         {"documents": 1, "sentences": 2, "tokens": 4, "vocabulary": 3, "retained_vocabulary": 1})


if __name__ == "__main__":
    unittest.main()
