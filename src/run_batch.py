import argparse
import json
import os
import warnings
from typing import List, Dict, Any

try:
    from transformers.utils import logging as hf_logging
except Exception:
    hf_logging = None


def maybe_quiet(quiet: bool) -> None:
    if not quiet:
        return
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    if hf_logging:
        hf_logging.set_verbosity_error()
    warnings.filterwarnings("ignore", category=FutureWarning, message=r".*torch_dtype.*deprecated.*")
    warnings.filterwarnings("ignore", category=UserWarning, message=r".*Qwen2TokenizerFast.*__call__.*")


def run_case(model_id: str, query: str, documents: List[str], top_k: int, device: str) -> Dict[str, Any]:
    from mxbai_rerank import MxbaiRerankV2

    if device and device.lower() != "auto":
        try:
            model = MxbaiRerankV2(model_id, device=device)
        except TypeError:
            model = MxbaiRerankV2(model_id)
    else:
        model = MxbaiRerankV2(model_id)
    results = model.rank(query, documents, return_documents=True, top_k=top_k)

    normalized = []
    for r in results:
        if isinstance(r, dict):
            doc = r.get("document") or r.get("text") or r.get("doc")
            score = r.get("score")
        else:
            doc = getattr(r, "document", None) or getattr(r, "text", None) or getattr(r, "doc", None)
            if doc is None and hasattr(r, "index"):
                try:
                    doc = documents[int(getattr(r, "index"))]
                except Exception:
                    pass
            score = getattr(r, "score", None)
        normalized.append({"document": doc, "score": float(score) if score is not None else None})

    return {"query": query, "results": normalized}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases-file", required=True)
    ap.add_argument("--model-id", default="mixedbread-ai/mxbai-rerank-base-v2")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"], help="force device selection")
    ap.add_argument("--quiet", action="store_true", help="suppress advisory warnings")
    args = ap.parse_args()

    maybe_quiet(args.quiet)

    with open(args.cases_file, "r") as f:
        payload = json.load(f)

    outputs = []
    for case in payload["cases"]:
        outputs.append(
            {
                "name": case.get("name"),
                **run_case(args.model_id, case["query"], case["documents"], case.get("top_k", 3), args.device),
            }
        )

    print(json.dumps({"model": args.model_id, "cases": outputs}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
