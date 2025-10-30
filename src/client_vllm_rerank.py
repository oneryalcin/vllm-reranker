import argparse
import json
import requests


def call_rerank(base_url: str, query: str, docs, top_k: int, model_id: str):
    # OpenAI-compatible rerank endpoint expected from vLLM when --task score is enabled
    url = base_url.rstrip("/") + "/v1/rerank"
    payload = {
        "model": model_id,
        "query": query,
        "documents": docs,
        "top_n": top_k,
    }
    r = requests.post(url, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--query", required=True)
    ap.add_argument("--docs-file", required=True)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--model-id", default="mixedbread-ai/mxbai-rerank-base-v2")
    args = ap.parse_args()

    with open(args.docs_file, "r") as f:
        docs = json.load(f)["documents"]

    resp = call_rerank(args.url, args.query, docs, args.top_k, args.model_id)
    print(json.dumps(resp, indent=2))


if __name__ == "__main__":
    main()
