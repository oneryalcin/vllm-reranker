import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.onnx_reranker import OnnxReranker, RerankRequest, load_documents


def main() -> None:
    ap = argparse.ArgumentParser(description="Run ONNX CPU reranker")
    ap.add_argument("--query", required=True)
    ap.add_argument("--docs-file", required=True)
    ap.add_argument("--model-dir", default="onnx/mxbai-base")
    ap.add_argument("--model-file", default="model-int8.onnx")
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()

    docs = load_documents(args.docs_file)
    reranker = OnnxReranker(Path(args.model_dir), args.model_file)
    results = reranker.rerank(
        RerankRequest(
            query=args.query,
            documents=docs,
            top_n=args.top_k,
        )
    )

    print(json.dumps({"query": args.query, "results": results}, indent=2))


if __name__ == "__main__":
    main()
