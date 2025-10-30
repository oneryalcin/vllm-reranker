import argparse
import json
import os
import random
from typing import Dict, List, Set

from beir.datasets.data_loader import GenericDataLoader
from beir.util import download_and_unzip
from tqdm import tqdm


def sample_negative_ids(corpus_ids: List[str], positive_ids: Set[str], num_negatives: int) -> List[str]:
    pool = [cid for cid in corpus_ids if cid not in positive_ids]
    if num_negatives >= len(pool):
        return pool
    return random.sample(pool, num_negatives)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a rerank workload from a BEIR dataset")
    parser.add_argument("--dataset", required=True, help="BEIR dataset name, e.g. scifact, nq, trec-covid")
    parser.add_argument("--split", default="test", help="Dataset split (default: test)")
    parser.add_argument("--limit", type=int, default=100, help="Number of queries to include")
    parser.add_argument("--top-k", type=int, default=20, help="Number of positive docs per query")
    parser.add_argument(
        "--negatives",
        type=int,
        default=20,
        help="Number of random negative docs per query",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default="data/bench/rerank_payload.json", help="Output JSON path")
    args = parser.parse_args()

    random.seed(args.seed)

    base_dir = "beir"
    dataset_path = os.path.join(base_dir, args.dataset)
    corpus_file = os.path.join(dataset_path, "corpus.jsonl")

    if not os.path.exists(corpus_file):
        os.makedirs(base_dir, exist_ok=True)
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{args.dataset}.zip"
        print(f"Dataset not found locally. Downloading from {url} ...")
        download_and_unzip(url, base_dir)
    corpus, queries, qrels = GenericDataLoader(dataset_path).load(split=args.split)
    corpus_ids = list(corpus.keys())

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    cases = []
    for qid, rels in tqdm(list(qrels.items())[: args.limit], desc="queries"):
        query = queries.get(qid)
        if not query:
            continue

        positive_ids = list(rels.keys())[: args.top_k]
        positive_docs = [corpus[pid]["text"] for pid in positive_ids if pid in corpus]

        negative_ids = sample_negative_ids(corpus_ids, set(positive_ids), args.negatives)
        negative_docs = [corpus[nid]["text"] for nid in negative_ids if nid in corpus]

        docs = positive_docs + negative_docs
        if not docs:
            continue

        cases.append({
            "query": query,
            "documents": docs,
            "positives": len(positive_docs),
            "negatives": len(negative_docs),
            "source": {
                "dataset": args.dataset,
                "split": args.split,
                "qid": qid,
            },
        })

    payload: Dict[str, List[dict]] = {"cases": cases}

    with open(args.out, "w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(cases)} cases to {args.out}")


if __name__ == "__main__":
    main()
