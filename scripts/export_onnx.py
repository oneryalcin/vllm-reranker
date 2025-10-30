"""Export Mixedbread reranker to ONNX for CPU inference.

Usage:
  uv run --with optimum[onnxruntime] --with transformers --with torch \
    python scripts/export_onnx.py \
      --model-id mixedbread-ai/mxbai-rerank-base-v2 \
      --out-dir onnx/mxbai-base \
      --opset 17

The script downloads the original model from Hugging Face, exports it to
ONNX (sequence classification), and stores tokenizer assets alongside the
generated `model.onnx` file. Set HF_TOKEN if the repository is gated.
"""

import argparse
import pathlib

from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoTokenizer


def export(model_id: str, out_dir: pathlib.Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)
    tokenizer.save_pretrained(out_dir)

    model = ORTModelForCausalLM.from_pretrained(
        model_id,
        export=True,
        provider="CPUExecutionProvider",
        trust_remote_code=True,
        cache_dir=str(out_dir),
        force_download=True,
        use_io_binding=False,
        use_cache=False,
    )

    model.save_pretrained(out_dir)

    print(f"Export complete -> {out_dir / 'model.onnx'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="mixedbread-ai/mxbai-rerank-base-v2")
    ap.add_argument("--out-dir", default="onnx/mxbai-base")
    args = ap.parse_args()

    export(args.model_id, pathlib.Path(args.out_dir))


if __name__ == "__main__":
    main()
