#!/usr/bin/env python3
"""Adapt CLIP's two ONNX integer inputs to QIM's INT32 tensor contract.

The pinned model remains an ONNX graph executed by the installed QIM ONNX
Runtime plugin. Only graph input type declarations are changed; ONNX Gather
accepts INT32 indices. The original Hugging Face SHA256 is checked before the
atomic rewrite and both source and adapted hashes are recorded in the receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from datetime import datetime, timezone


INPUTS = {"input_ids", "attention_mask"}
ONNX_INT32 = 6
ONNX_INT64 = 7


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _varint(data: bytes, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while offset < len(data):
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, offset
        shift += 7
        if shift > 63:
            break
    raise ValueError("invalid ONNX protobuf varint")


def _encode_varint(value: int) -> bytes:
    encoded = bytearray()
    while value > 0x7F:
        encoded.append((value & 0x7F) | 0x80)
        value >>= 7
    encoded.append(value)
    return bytes(encoded)


def _fields(data: bytes):
    offset = 0
    while offset < len(data):
        start = offset
        key, offset = _varint(data, offset)
        number, wire = key >> 3, key & 7
        if number == 0:
            raise ValueError("invalid ONNX protobuf field number")
        if wire == 0:
            value, offset = _varint(data, offset)
        elif wire == 1:
            end = offset + 8
            if end > len(data):
                raise ValueError("truncated ONNX protobuf fixed64 field")
            value, offset = data[offset:end], end
        elif wire == 2:
            size, offset = _varint(data, offset)
            end = offset + size
            if end > len(data):
                raise ValueError("truncated ONNX protobuf bytes field")
            value, offset = data[offset:end], end
        elif wire == 5:
            end = offset + 4
            if end > len(data):
                raise ValueError("truncated ONNX protobuf fixed32 field")
            value, offset = data[offset:end], end
        else:
            raise ValueError("unsupported ONNX protobuf wire type")
        yield number, wire, value, data[start:offset]


def _length_field(number: int, value: bytes) -> bytes:
    return _encode_varint((number << 3) | 2) + _encode_varint(len(value)) + value


def _varint_field(number: int, value: int) -> bytes:
    return _encode_varint(number << 3) + _encode_varint(value)


def _patch_tensor_type(data: bytes) -> tuple[bytes, bool]:
    changed = False
    output = []
    saw_element_type = False
    for number, wire, value, raw in _fields(data):
        if number == 1 and wire == 0:
            saw_element_type = True
            if value == ONNX_INT64:
                output.append(_varint_field(1, ONNX_INT32))
                changed = True
                continue
            if value != ONNX_INT32:
                raise ValueError(f"CLIP tensor input type {value} is not INT32 or INT64")
        output.append(raw)
    if not saw_element_type:
        raise ValueError("ONNX tensor input does not declare an element type")
    return b"".join(output), changed


def _patch_type_proto(data: bytes) -> tuple[bytes, bool]:
    output = []
    changed = False
    found_tensor_type = False
    for number, wire, value, raw in _fields(data):
        # TypeProto.tensor_type is field 1 and TensorTypeProto is length-delimited.
        if number == 1 and wire == 2:
            found_tensor_type = True
            tensor, tensor_changed = _patch_tensor_type(value)
            output.append(_length_field(number, tensor))
            changed = changed or tensor_changed
        else:
            output.append(raw)
    if not found_tensor_type:
        raise ValueError("CLIP graph input is not a tensor")
    return b"".join(output), changed


def _patch_value_info(data: bytes) -> tuple[bytes, str | None, bool]:
    name = None
    type_bytes = None
    fields = list(_fields(data))
    for number, wire, value, _raw in fields:
        if number == 1 and wire == 2:
            try:
                name = value.decode("utf-8")
            except UnicodeDecodeError as error:
                raise ValueError("ONNX graph input name is not UTF-8") from error
        elif number == 2 and wire == 2:
            type_bytes = value
    if name not in INPUTS:
        return data, name, False
    if type_bytes is None:
        raise ValueError(f"ONNX input {name} has no tensor type")
    patched_type, changed = _patch_type_proto(type_bytes)
    if not changed:
        return data, name, False
    output = []
    replaced = False
    for number, wire, value, raw in fields:
        if number == 2 and wire == 2 and not replaced:
            output.append(_length_field(2, patched_type))
            replaced = True
        else:
            output.append(raw)
    return b"".join(output), name, True


def _patch_graph(data: bytes) -> tuple[bytes, set[str], bool]:
    output = []
    changed_names: set[str] = set()
    found_inputs: set[str] = set()
    for number, wire, value, raw in _fields(data):
        # GraphProto.input=11, value_info=13; update only the two named inputs.
        if number in (11, 13) and wire == 2:
            patched, name, changed = _patch_value_info(value)
            if number == 11 and name in INPUTS:
                found_inputs.add(name)
            if changed:
                changed_names.add(str(name))
                output.append(_length_field(number, patched))
            else:
                output.append(raw)
        else:
            output.append(raw)
    if found_inputs != INPUTS:
        raise ValueError("ONNX text graph must expose input_ids and attention_mask")
    if changed_names and changed_names != INPUTS:
        raise ValueError("both CLIP text inputs must be adapted together")
    return b"".join(output), changed_names, bool(changed_names)


def _patch_model(data: bytes) -> tuple[bytes, set[str], bool]:
    output = []
    graph_count = 0
    changed_names: set[str] = set()
    changed = False
    for number, wire, value, raw in _fields(data):
        # ModelProto.graph = field 7.
        if number == 7 and wire == 2:
            graph_count += 1
            graph, names, graph_changed = _patch_graph(value)
            changed_names.update(names)
            changed = changed or graph_changed
            output.append(_length_field(number, graph) if graph_changed else raw)
        else:
            output.append(raw)
    if graph_count != 1:
        raise ValueError("expected exactly one ONNX ModelProto graph")
    return b"".join(output), changed_names, changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    path = args.model.resolve(strict=True)
    source_hash = sha256(path)
    if source_hash != args.expected_source_sha256.lower():
        raise ValueError("downloaded CLIP text model does not match the pinned source SHA256")
    data = path.read_bytes()
    patched, changed_names, changed = _patch_model(data)
    if not changed:
        raise ValueError("CLIP text model already declares INT32 inputs; refusing unexpected rewrite")
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp",
                                                   dir=str(path.parent))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(patched)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    receipt = {
        "schema_version": 1,
        "model_id": "openai-clip-vit-b-32-onnx",
        "source_sha256": source_hash,
        "adapted_sha256": sha256(path),
        "inputs_changed": sorted(changed_names),
        "adaptation": "ONNX graph input declarations INT64 to INT32 for the QIM GstML tensor interface",
        "onnx_int32_enum": ONNX_INT32,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    receipt_tmp = args.receipt.with_name(args.receipt.name + ".tmp")
    receipt_tmp.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    os.replace(receipt_tmp, args.receipt)
    print(json.dumps({"status": "adapted", **receipt}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
