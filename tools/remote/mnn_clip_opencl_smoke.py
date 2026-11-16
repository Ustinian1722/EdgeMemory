#!/usr/bin/env python3
"""Run one CLIP image/text pair with MNN OpenCL image + CPU ONNX text."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path


def cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("embedding_shape_mismatch")
    dot = sum(a * b for a, b in zip(left, right))
    lhs = math.sqrt(sum(a * a for a in left))
    rhs = math.sqrt(sum(b * b for b in right))
    return dot / (lhs * rhs) if lhs and rhs else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--ort-python-path", type=Path, required=True)
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--mnn-model", type=Path, required=True)
    parser.add_argument("--image-model", type=Path, required=True)
    parser.add_argument("--text-model", type=Path, required=True)
    parser.add_argument("--vocab", type=Path, required=True)
    parser.add_argument("--merges", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--text", required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.ort_python_path))
    sys.path.insert(0, str(args.source_root))
    from edgememory.visual_embeddings import MnnOpenCLClipBackend

    backend = MnnOpenCLClipBackend(
        image_model_path=args.image_model,
        mnn_model_path=args.mnn_model,
        text_model_path=args.text_model,
        vocab_path=args.vocab,
        merges_path=args.merges,
        bridge_path=args.bridge,
        model_id="openai-clip-vit-b-32-onnx",
        threads=2,
    )
    started = time.perf_counter()
    image_vector = backend.embed_image(args.image)
    image_seconds = time.perf_counter() - started
    text_vector = backend.embed_text(args.text)
    snapshot = backend.snapshot()
    result = {
        "execution_target": "IQ-9075",
        "image_encoder": snapshot.get("active_image_backend"),
        "text_encoder": snapshot.get("text_execution_provider"),
        "device_execution_trace": snapshot.get("device_execution_trace"),
        "image_dimensions": len(image_vector),
        "text_dimensions": len(text_vector),
        "image_l2_norm": math.sqrt(sum(value * value for value in image_vector)),
        "text_l2_norm": math.sqrt(sum(value * value for value in text_vector)),
        "image_text_cosine": cosine(image_vector, text_vector),
        "image_seconds": image_seconds,
        "calls": snapshot.get("calls"),
        "fallback_errors": snapshot.get("fallback_errors"),
    }
    backend.close()
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if (result["image_encoder"] == "mnn-opencl" and
                 result["text_encoder"] == "CPUExecutionProvider" and
                 "MNN_FORWARD_OPENCL" in result["device_execution_trace"] and
                 result["image_dimensions"] == 512 and result["text_dimensions"] == 512) else 2


if __name__ == "__main__":
    raise SystemExit(main())
