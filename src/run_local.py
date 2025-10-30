import argparse
import json
import os
import warnings
from typing import List

try:
    from transformers.utils import logging as hf_logging
except Exception:  # pragma: no cover
    hf_logging = None


def maybe_quiet(quiet: bool) -> None:
    if not quiet:
        return
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    if hf_logging:
        hf_logging.set_verbosity_error()
    # Silence specific advisory/deprecation warnings we expect from the stack.
    warnings.filterwarnings("ignore", category=FutureWarning, message=r".*torch_dtype.*deprecated.*")
    warnings.filterwarnings("ignore", category=UserWarning, message=r".*Qwen2TokenizerFast.*__call__.*")


def rerank_locally(query: str, docs: List[str], top_k: int, model_id: str, device: str) -> List[dict]:
    try:
        # Prefer the official wrapper for correct scoring behavior.
        from mxbai_rerank import MxbaiRerankV2
    except Exception as e:
        raise RuntimeError(
            "Missing dependency mxbai-rerank. Either run with uv one-off: \n"
            "  uv run --with mxbai-rerank --with transformers --with torch "
            "python src/run_local.py --query ... --docs-file ... --top-k ... --model-id ...\n"
            "or install permanently: pip install mxbai-rerank transformers torch"
        ) from e

    # Create model; allow explicit device or auto.
    if device and device.lower() != "auto":
        try:
            model = MxbaiRerankV2(model_id, device=device)
        except TypeError:
            # Older versions may not accept device kw; fall back to default behavior.
            model = MxbaiRerankV2(model_id)
    else:
        model = MxbaiRerankV2(model_id)
    # Use the high-level rank API to mirror HF docs.
    results = model.rank(query, docs, return_documents=True, top_k=top_k)
    # results may be objects (e.g., RankResult) or dicts; normalize.
    normalized = []
    for r in results:
        doc_text = None
        score_val = None

        if isinstance(r, dict):
            doc_text = r.get("document") or r.get("text") or r.get("doc")
            score_val = r.get("score")
            if doc_text is None and "index" in r:
                try:
                    doc_text = docs[int(r["index"])]
                except Exception:
                    pass
        else:
            # Try common attribute names
            for attr in ("document", "text", "doc", "content", "value"):
                if hasattr(r, attr):
                    doc_text = getattr(r, attr)
                    break
            if hasattr(r, "index") and doc_text is None:
                try:
                    doc_text = docs[int(getattr(r, "index"))]
                except Exception:
                    pass
            if hasattr(r, "score"):
                score_val = getattr(r, "score")

        normalized.append({
            "document": doc_text,
            "score": float(score_val) if score_val is not None else None,
        })
    return normalized


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True)
    ap.add_argument("--docs-file", required=True)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--model-id", default="mixedbread-ai/mxbai-rerank-base-v2")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"], help="force device selection")
    ap.add_argument("--quiet", action="store_true", help="suppress advisory warnings")
    args = ap.parse_args()

    maybe_quiet(args.quiet)

    with open(args.docs_file, "r") as f:
        data = json.load(f)
    docs = data["documents"]

    results = rerank_locally(args.query, docs, args.top_k, args.model_id, args.device)
    print(json.dumps({"query": args.query, "results": results}, indent=2))


if __name__ == "__main__":
    main()
