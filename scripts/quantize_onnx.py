"""Apply dynamic quantization to an ONNX reranker model.

Usage:
  uv run --with onnxruntime --with onnxruntime-tools \
    python scripts/quantize_onnx.py \
      --model-path onnx/mxbai-base/model.onnx \
      --out-path onnx/mxbai-base/model-int8.onnx

Dynamic int8 quantization reduces model size and improves CPU throughput.
"""

import argparse
import pathlib

from onnxruntime.quantization import QuantType, quantize_dynamic


def quantize(model_path: pathlib.Path, out_path: pathlib.Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    quantize_dynamic(
        model_input=str(model_path),
        model_output=str(out_path),
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul", "Gemm"],
        per_channel=False,
        reduce_range=False,
        extra_options={"OptimizeModel": True},
    )

    print(f"Quantized model written to {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--out-path", required=True)
    args = ap.parse_args()

    quantize(pathlib.Path(args.model_path), pathlib.Path(args.out_path))


if __name__ == "__main__":
    main()
