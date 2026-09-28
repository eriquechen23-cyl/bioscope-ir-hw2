"""Run from repo root: python -m scripts.prepare_corpus --count 1000."""
import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from bioir.pubmed import DEFAULT_QUERY, PubMedClient, corpus_bytes, parse_pmids, validate_records


def main():
    parser = argparse.ArgumentParser(description="Prepare a real PubMed abstract corpus.")
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--pmids", type=Path, help="Optional text file of PMIDs (must all have abstracts).")
    parser.add_argument("--output", type=Path, default=Path("data/glp1_1000.jsonl"))
    args = parser.parse_args()
    client = PubMedClient(os.getenv("NCBI_EMAIL", ""), os.getenv("NCBI_API_KEY", ""))
    def progress(done, total):
        print(f"Fetched {done}/{total}", flush=True)
    if args.pmids:
        ids = parse_pmids(args.pmids.read_text(encoding="utf-8-sig"))
        records = client.fetch_pmids(ids, progress)
        missing = sorted(set(ids) - {r["pmid"] for r in records})
        if missing:
            raise SystemExit(f"No abstract or unavailable PMID: {', '.join(missing)}. Existing output unchanged.")
        metadata = {"source": "NCBI EFetch / PMID list", "actual_count": len(records),
                    "retrieved_at_utc": datetime.now(timezone.utc).isoformat()}
    else:
        records, metadata = client.search(args.query, args.count, progress)
    validate_records(records)
    content = corpus_bytes(records)
    metadata["sha256"] = hashlib.sha256(content).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(".tmp")
    temp.write_bytes(content)
    temp.replace(args.output)
    args.output.with_suffix(".manifest.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    args.output.with_suffix(".pmids.txt").write_text("\n".join(r["pmid"] for r in records) + "\n", encoding="utf-8")
    print(f"Saved {len(records)} abstracts to {args.output}", flush=True)


if __name__ == "__main__":
    main()
