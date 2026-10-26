"""Run: python demo.py --query 'graph neural networks' --max-results 10"""
import argparse
import json
from pathlib import Path
from scholarly_insight.discovery import fetch_papers, rank_related


def main():
    parser = argparse.ArgumentParser(description="arXiv discovery prototype")
    parser.add_argument("--query", default="graph neural networks")
    parser.add_argument("--max-results", type=int, default=10)
    parser.add_argument("--output", default="output/papers.json")
    args = parser.parse_args()

    papers = fetch_papers(args.query, args.max_results)
    if not papers:
        print("No papers found.")
        return
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(papers, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {len(papers)} papers to {output}")
    selected = papers[0]
    print(f"\nSelected paper: {selected['title']}")
    for paper in rank_related(papers, selected["id"], limit=5):
        print(f"  {paper['similarity']:.3f}  {paper['title']}  ({paper['url']})")


if __name__ == "__main__":
    main()
