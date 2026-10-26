import unittest
from scholarly_insight.discovery import rank_related, tokenize


class DiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.papers = [
            {"id": "a", "title": "Graph neural networks", "abstract": "Graph learning and node embeddings"},
            {"id": "b", "title": "Graph representation learning", "abstract": "Graph neural embeddings for nodes"},
            {"id": "c", "title": "Ocean temperatures", "abstract": "Sea temperature and marine ecosystems"},
        ]

    def test_tokenize(self):
        self.assertEqual(tokenize("Graphs, graphs! AND nodes."), ["graphs", "graphs", "nodes"])

    def test_related_ranking(self):
        result = rank_related(self.papers, "a")
        self.assertEqual(result[0]["id"], "b")
        self.assertNotIn("a", [paper["id"] for paper in result])
        self.assertGreater(result[0]["similarity"], result[1]["similarity"])

    def test_missing_paper(self):
        with self.assertRaises(ValueError):
            rank_related(self.papers, "missing")


if __name__ == "__main__":
    unittest.main()
