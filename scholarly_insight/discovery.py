"""Retrieve arXiv papers and rank related abstracts with a simple TF-IDF baseline."""
from __future__ import annotations

import math
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter

ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV_API = "https://export.arxiv.org/api/query"
TOKEN = re.compile(r"[a-z][a-z0-9]{1,}")
STOPWORDS = set("the and for with from this that into over under using use are was were have has how why what when where who which about between their they them our your can not but also via based study paper approach method methods results".split())


def build_search_query(query: str) -> str:
    """Search a phrase in titles or abstracts, rather than loosely matching words.

    Quotes inside user input are stripped to keep this simple query builder safe.
    """
    phrase = " ".join(query.replace('"', " ").split())
    if not phrase:
        raise ValueError("Search query must not be empty")
    return f'(ti:"{phrase}" OR abs:"{phrase}")'


def fetch_papers(query: str, max_results: int = 10) -> list[dict]:
    """Fetch and normalize public arXiv metadata; no API key required."""
    if not 1 <= max_results <= 100:
        raise ValueError("max_results must be between 1 and 100")
    search_query = build_search_query(query)
    params = urllib.parse.urlencode({
        "search_query": search_query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    })
    request = urllib.request.Request(
        f"{ARXIV_API}?{params}",
        headers={"User-Agent": "ScholarlyInsightStudentPrototype/0.1 (educational project)"},
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        root = ET.fromstring(response.read())
    papers = []
    for entry in root.findall(f"{ATOM}entry"):
        url = (entry.findtext(f"{ATOM}id") or "").strip()
        papers.append({
            "id": url.rsplit("/", 1)[-1],
            "url": url,
            "title": " ".join((entry.findtext(f"{ATOM}title") or "").split()),
            "abstract": " ".join((entry.findtext(f"{ATOM}summary") or "").split()),
            "authors": [
                (author.findtext(f"{ATOM}name") or "").strip()
                for author in entry.findall(f"{ATOM}author")
            ],
            "published": (entry.findtext(f"{ATOM}published") or "")[:10],
        })
    return papers


def tokenize(text: str) -> list[str]:
    return [word for word in TOKEN.findall(text.lower()) if word not in STOPWORDS]


def rank_related(papers: list[dict], selected_id: str, limit: int = 5) -> list[dict]:
    """Rank other papers by cosine similarity of TF-IDF abstract vectors.

    Scores are corpus-relative baseline similarities, not validated relevance metrics.
    """
    if limit < 1:
        raise ValueError("limit must be positive")
    matches = [i for i, paper in enumerate(papers) if paper["id"] == selected_id]
    if not matches:
        raise ValueError(f"Paper not found: {selected_id}")
    docs = [Counter(tokenize(p.get("title", "") + " " + p.get("abstract", ""))) for p in papers]
    df = Counter(term for doc in docs for term in doc)
    n = len(docs)
    vectors = []
    for doc in docs:
        vector = {term: (1 + math.log(freq)) * (math.log((1 + n) / (1 + df[term])) + 1)
                  for term, freq in doc.items()}
        norm = math.sqrt(sum(value * value for value in vector.values()))
        vectors.append({term: value / norm for term, value in vector.items()} if norm else {})
    selected = vectors[matches[0]]
    ranked = []
    for i, paper in enumerate(papers):
        if paper["id"] == selected_id:
            continue
        score = sum(weight * vectors[i].get(term, 0) for term, weight in selected.items())
        ranked.append({**paper, "similarity": round(score, 4)})
    return sorted(ranked, key=lambda p: (-p["similarity"], p["id"]))[:limit]
