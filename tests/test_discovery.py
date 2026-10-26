import unittest
from urllib.parse import parse_qs, urlparse
from unittest.mock import patch
from io import BytesIO
from scholarly_insight.discovery import build_search_query, fetch_papers, rank_related, tokenize


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

    def test_phrase_query(self):
        self.assertEqual(
            build_search_query("  graph   neural networks "),
            '(ti:"graph neural networks" OR abs:"graph neural networks")',
        )
        with self.assertRaises(ValueError):
            build_search_query("   ")

    def test_fetch_uses_phrase_query(self):
        atom = b'<feed xmlns="http://www.w3.org/2005/Atom"></feed>'
        class FakeResponse:
            def __enter__(self):
                return BytesIO(atom)
            def __exit__(self, *args):
                return False
        with patch("scholarly_insight.discovery.urllib.request.urlopen", return_value=FakeResponse()) as mocked:
            self.assertEqual(fetch_papers("graph neural networks"), [])
        request = mocked.call_args.args[0]
        params = parse_qs(urlparse(request.full_url).query)
        self.assertEqual(params["search_query"][0], '(ti:"graph neural networks" OR abs:"graph neural networks")')

    def test_missing_paper(self):
        with self.assertRaises(ValueError):
            rank_related(self.papers, "missing")


if __name__ == "__main__":
    unittest.main()
