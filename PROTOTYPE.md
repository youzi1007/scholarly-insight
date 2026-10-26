# Scholarly Insight — Initial Prototype

**Status:** Early proof of concept for the October 27, 2026 progress presentation. Not a completed application.

## Problem
Keyword-only paper search can miss related work that uses different terminology.

## Proposed solution
1. Retrieve research papers from the public arXiv API.
2. Normalize titles, authors, publication dates, abstracts, and IDs.
3. Rank related papers using a transparent TF-IDF cosine-similarity baseline.
4. Later: add a web UI, persistent storage, improved semantic embeddings, and a paper exploration graph.

## Run locally
Requires Python 3.10+; no third-party dependencies.

```bash
python demo.py --query "graph neural networks" --max-results 10
python -m unittest discover -s tests -v
```

The demo needs internet access to query arXiv. It saves a JSON file to `output/papers.json` and prints related-paper rankings. Similarity scores are relative to the retrieved sample; they are **not** a validated recommendation-quality measurement.

## Proposed architecture
Web UI (planned) → backend API (planned) → arXiv ingestion + database (planned) → ranking service (TF-IDF baseline implemented).

## Suggested November milestones
- Week 1: paper retrieval, tests, data model, frontend mockup
- Week 2: API and database integration, search interface
- Week 3: semantic embeddings and relationship visualization
- Week 4: integration, usability, demo, and deployment

## Limitations
- Retrieval now searches an exact phrase in arXiv titles or abstracts. This improves candidate precision but can miss relevant papers using different terminology; an expanded-query strategy is future work.
- Live relevance of the updated results still needs to be verified against the arXiv service.
- TF-IDF similarity is lexical, not truly semantic. It may miss conceptually related papers using different vocabulary.
- No citation graph, database, web UI, deployment, or user accounts yet.
- Avoid presenting proposed components as implemented.
