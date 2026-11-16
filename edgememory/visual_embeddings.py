from __future__ import annotations

"""Resident SigLIP2 image/text embedding adapter for Qualcomm QIM + QNN.

The adapter uses QIM's qtimlqnn element for model execution. It deliberately
does not implement QNN kernels. Image preprocessing uses qtimlvconverter's GLES
path when available; a CPU tensor-preparation fallback keeps the model usable
when that optional plugin is absent.
"""

import math
import ctypes
import ctypes.util
import html
import json
import os
from pathlib import Path
import re
import threading
import time
from typing import Protocol, Sequence


class VisualEmbeddingProvider(Protocol):
    @property
    def model_id(self) -> str: ...

    def embed_image(self, image_path: Path) -> list[float]: ...

    def embed_text(self, text: str) -> list[float]: ...

    def snapshot(self) -> dict: ...


def _unit(vector: Sequence[float]) -> list[float]:
    values = [float(value) for value in vector]
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("visual embedding output must contain finite values")
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0:
        raise ValueError("visual embedding output has zero norm")
    return [value / norm for value in values]


class QimQnnSigLIP2Backend:
    """Keep image and text SigLIP2 QNN contexts resident behind GStreamer."""

    def __init__(self, *, image_model_path: Path, text_model_path: Path,
                 tokenizer_path: Path, backend_library: str,
                 model_id: str, output_dimensions: int = 768,
                 timeout_s: float = 60.0, backend_device_id: int = 0):
        self.image_model_path = Path(image_model_path).resolve(strict=True)
        self.text_model_path = Path(text_model_path).resolve(strict=True)
        self.tokenizer_path = Path(tokenizer_path).resolve(strict=True)
        self.backend_library = backend_library
        self.backend_device_id = backend_device_id
        self._model_id = model_id
        self.output_dimensions = output_dimensions
        self.timeout_s = timeout_s
        if output_dimensions < 1 or timeout_s <= 0:
            raise ValueError("visual embedding dimensions and timeout must be positive")

        try:
            import cv2
            import numpy as np
            import gi
            from tokenizers import Tokenizer
        except Exception as error:
            raise RuntimeError("QIM SigLIP2 requires target OpenCV, NumPy, PyGObject and tokenizers") from error
        try:
            gi.require_version("Gst", "1.0")
            from gi.repository import Gst
        except Exception as error:
            raise RuntimeError("GStreamer Python bindings are unavailable") from error

        self._cv2 = cv2
        self._np = np
        self._Gst = Gst
        Gst.init(None)
        self._tokenizer = Tokenizer.from_file(str(self.tokenizer_path))
        self._pad_id = self._tokenizer.token_to_id("<pad>")
        if self._pad_id is None:
            raise RuntimeError("SigLIP2 tokenizer has no <pad> token")
        self._asset_hashes = {
            "image_model_sha256": _sha256(self.image_model_path),
            "text_model_sha256": _sha256(self.text_model_path),
            "tokenizer_sha256": _sha256(self.tokenizer_path),
        }
        self._lock = threading.RLock()
        self._image_pipeline, self._image_source, self._image_sink, self._image_preprocess = (
            self._create_pipeline(self.image_model_path, "image_embeds", image=True))
        try:
            self._text_pipeline, self._text_source, self._text_sink, _ = (
                self._create_pipeline(self.text_model_path, "text_embeds", image=False))
        except Exception:
            self._image_pipeline.set_state(Gst.State.NULL)
            raise
        self._calls = {"image": 0, "text": 0}
        self._last_latency_us = {"image": None, "text": None}

    @property
    def model_id(self) -> str:
        return self._model_id

    def _create_pipeline(self, model_path: Path, output_name: str, *, image: bool):
        Gst = self._Gst
        pipeline = Gst.Pipeline.new("hexagonflow-siglip2-image" if image
                                    else "hexagonflow-siglip2-text")
        source = Gst.ElementFactory.make("appsrc", "input")
        inference = Gst.ElementFactory.make("qtimlqnn", "qnn-inference")
        sink = Gst.ElementFactory.make("appsink", "embedding-output")
        if pipeline is None or source is None or inference is None or sink is None:
            raise RuntimeError("QIM qtimlqnn is unavailable")
        converter = None
        capsfilter = None
        if image and Gst.ElementFactory.find("qtimlvconverter") is not None:
            converter = Gst.ElementFactory.make("qtimlvconverter", "gpu-preprocess")
            capsfilter = Gst.ElementFactory.make("capsfilter", "image-tensor-caps")
            if converter is not None and capsfilter is not None:
                converter.set_property("engine", "gles")
                converter.set_property("image-disposition", "stretch")
                if converter.find_property("subpixel-layout") is not None:
                    # GStreamer input is BGR; the model contract is RGB.
                    converter.set_property("subpixel-layout", 1)
                capsfilter.set_property("caps", Gst.Caps.from_string(
                    'neural-network/tensors,type=FLOAT32,dimensions="<<1,3,224,224>>"'))
            else:
                converter = None
                capsfilter = None

        source.set_property("is-live", False)
        source.set_property("block", True)
        source.set_property("format", Gst.Format.TIME)
        if image and converter is not None:
            source.set_property("caps", Gst.Caps.from_string(
                "video/x-raw,format=BGR,width=224,height=224,framerate=1/1"))
        elif image:
            source.set_property("caps", Gst.Caps.from_string(
                'neural-network/tensors,type=FLOAT32,dimensions="<<1,3,224,224>>"'))
        else:
            source.set_property("caps", Gst.Caps.from_string(
                'neural-network/tensors,type=INT32,dimensions="<<1,64>>"'))
        inference.set_property("model", str(model_path))
        inference.set_property("backend", self.backend_library)
        if inference.find_property("backend-device-id") is not None:
            inference.set_property("backend-device-id", int(self.backend_device_id))
        if inference.find_property("tensors") is not None:
            try:
                inference.set_property("tensors", [output_name])
            except (TypeError, ValueError):
                # Some QIM builds expose GstValueArray only through their parser.
                # Omitting this filter is safe for the single-output SigLIP2 graph.
                pass
        sink.set_property("emit-signals", False)
        sink.set_property("sync", False)
        sink.set_property("max-buffers", 1)
        sink.set_property("drop", False)
        for element in (source, *(([converter, capsfilter] if converter else [])), inference, sink):
            pipeline.add(element)
        chain = ([source, converter, capsfilter, inference, sink]
                 if converter else [source, inference, sink])
        for left, right in zip(chain, chain[1:]):
            if not left.link(right):
                raise RuntimeError("QIM SigLIP2 pipeline caps negotiation failed")
        result = pipeline.set_state(Gst.State.PLAYING)
        if result == Gst.StateChangeReturn.FAILURE:
            pipeline.set_state(Gst.State.NULL)
            raise RuntimeError("QIM SigLIP2 pipeline failed to start")
        return pipeline, source, sink, "qtimlvconverter_gles" if converter else (
            "cpu_tensor_preprocess" if image else "tokenizer_int32")

    def _infer(self, source, sink, payload: bytes, *, channel: str) -> list[float]:
        Gst = self._Gst
        buffer = Gst.Buffer.new_allocate(None, len(payload), None)
        buffer.fill(0, payload)
        buffer.pts = self._calls[channel] * Gst.SECOND
        buffer.duration = Gst.SECOND
        started = time.perf_counter_ns()
        flow = source.emit("push-buffer", buffer)
        if flow != Gst.FlowReturn.OK:
            raise RuntimeError(f"QIM {channel} input buffer was rejected")
        sample = sink.emit("try-pull-sample", int(self.timeout_s * Gst.SECOND))
        if sample is None:
            raise TimeoutError(f"QIM {channel} embedding timed out")
        output = sample.get_buffer()
        ok, mapped = output.map(Gst.MapFlags.READ)
        if not ok:
            raise RuntimeError(f"QIM {channel} output tensor could not be mapped")
        try:
            caps = sample.get_caps()
            structure = caps.get_structure(0) if caps and caps.get_size() else None
            type_name = "FLOAT32"
            if structure is not None:
                type_name = str(structure.get_value("type") or
                                structure.get_value("format") or type_name).upper()
            dtype = self._np.float16 if "16" in type_name else self._np.float32
            vector = self._np.frombuffer(mapped.data, dtype=dtype).astype(self._np.float32)
            if vector.size != self.output_dimensions:
                raise RuntimeError("QIM SigLIP2 output dimension does not match the pinned model contract")
            result = _unit(vector.tolist())
        finally:
            output.unmap(mapped)
        self._calls[channel] += 1
        self._last_latency_us[channel] = (time.perf_counter_ns() - started) // 1000
        return result

    def embed_image(self, image_path: Path) -> list[float]:
        with self._lock:
            frame = self._cv2.imread(str(image_path), self._cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError("visual embedding image could not be decoded")
            if self._image_preprocess == "qtimlvconverter_gles":
                height, width = frame.shape[:2]
                # The converter performs the 224x224 resize and float scaling on
                # its configured GLES path. Input memory remains valid through push-buffer.
                self._image_source.set_property("caps", self._Gst.Caps.from_string(
                    f"video/x-raw,format=BGR,width={width},height={height},framerate=1/1"))
                payload = frame.tobytes()
            else:
                rgb = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)
                resized = self._cv2.resize(rgb, (224, 224), interpolation=self._cv2.INTER_LINEAR)
                tensor = self._np.transpose(resized.astype(self._np.float32) / 255.0,
                                            (2, 0, 1))[self._np.newaxis, ...]
                payload = tensor.astype("<f4", copy=False).tobytes()
            return self._infer(self._image_source, self._image_sink, payload, channel="image")

    def embed_text(self, text: str) -> list[float]:
        if not text.strip():
            raise ValueError("cannot embed empty visual query")
        with self._lock:
            ids = self._tokenizer.encode(text, add_special_tokens=True).ids[:64]
            ids.extend([int(self._pad_id)] * (64 - len(ids)))
            payload = self._np.asarray(ids, dtype="<i4").tobytes()
            return self._infer(self._text_source, self._text_sink, payload, channel="text")

    def snapshot(self) -> dict:
        return {
            "status": "resident",
            "backend_id": f"qim-qtimlqnn:{self.backend_library}",
            "model_id": self.model_id,
            **self._asset_hashes,
            "output_dimensions": self.output_dimensions,
            "image_preprocess_path": self._image_preprocess,
            "requested_qnn_backend_library": self.backend_library,
            "requested_qnn_backend_device_id": self.backend_device_id,
            "device_execution_trace": "not_exposed_by_qtimlqnn_api",
            "calls": dict(self._calls),
            "last_latency_us": dict(self._last_latency_us),
        }

    def close(self) -> None:
        for pipeline in (getattr(self, "_image_pipeline", None),
                         getattr(self, "_text_pipeline", None)):
            if pipeline is not None:
                pipeline.set_state(self._Gst.State.NULL)


class _ClipBpeTokenizer:
    """Small dependency-free OpenAI CLIP byte-level BPE tokenizer."""

    _pattern = re.compile(
        r"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|"
        r"[^\W\d_]+|\d|[^\s\w]+|_+",
        re.IGNORECASE,
    )

    def __init__(self, vocab_path: Path, merges_path: Path):
        self.vocab_path = Path(vocab_path).resolve(strict=True)
        self.merges_path = Path(merges_path).resolve(strict=True)
        self.encoder = json.loads(self.vocab_path.read_text(encoding="utf-8"))
        self.encoder = {str(token): int(index) for token, index in self.encoder.items()}
        self.decoder = {index: token for token, index in self.encoder.items()}
        with self.merges_path.open("r", encoding="utf-8") as stream:
            merge_lines = [line.strip() for line in stream if line.strip() and
                           not line.startswith("#")]
        self.ranks = {
            tuple(line.split()): rank for rank, line in enumerate(merge_lines)
            if len(line.split()) == 2
        }
        self.start_id = self.encoder.get("<|startoftext|>")
        self.end_id = self.encoder.get("<|endoftext|>")
        if self.start_id is None or self.end_id is None or len(self.encoder) != 49408:
            raise ValueError("CLIP tokenizer assets do not match the pinned 49,408-token vocabulary")
        self.pad_id = 0
        self._byte_encoder = self._bytes_to_unicode()
        self._cache: dict[str, tuple[str, ...]] = {}

    @staticmethod
    def _bytes_to_unicode() -> dict[int, str]:
        visible = list(range(ord("!"), ord("~") + 1))
        visible += list(range(ord("¡"), ord("¬") + 1))
        visible += list(range(ord("®"), ord("ÿ") + 1))
        codepoints = visible[:]
        extra = 0
        for value in range(256):
            if value not in visible:
                visible.append(value)
                codepoints.append(256 + extra)
                extra += 1
        return {value: chr(codepoint) for value, codepoint in zip(visible, codepoints)}

    @staticmethod
    def _pairs(word: tuple[str, ...]) -> set[tuple[str, str]]:
        return set(zip(word, word[1:]))

    def _bpe(self, token: str) -> tuple[str, ...]:
        cached = self._cache.get(token)
        if cached is not None:
            return cached
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = self._pairs(word)
        while pairs:
            bigram = min(pairs, key=lambda pair: self.ranks.get(pair, float("inf")))
            if bigram not in self.ranks:
                break
            first, second = bigram
            merged: list[str] = []
            index = 0
            while index < len(word):
                try:
                    next_index = word.index(first, index)
                except ValueError:
                    merged.extend(word[index:])
                    break
                merged.extend(word[index:next_index])
                index = next_index
                if index + 1 < len(word) and word[index + 1] == second:
                    merged.append(first + second)
                    index += 2
                else:
                    merged.append(word[index])
                    index += 1
            word = tuple(merged)
            if len(word) == 1:
                break
            pairs = self._pairs(word)
        self._cache[token] = word
        return word

    @staticmethod
    def _clean_text(text: str) -> str:
        text = html.unescape(html.unescape(text))
        text = " ".join(text.strip().split())
        return text.lower()

    def encode(self, text: str, *, context_length: int = 77) -> tuple[list[int], list[int]]:
        if context_length != 77:
            raise ValueError("the pinned CLIP ViT-B/32 text encoder requires 77 tokens")
        token_ids: list[int] = [int(self.start_id)]
        normalized = self._clean_text(text)
        for token in self._pattern.findall(normalized):
            byte_token = "".join(self._byte_encoder[value] for value in token.encode("utf-8"))
            for piece in self._bpe(byte_token):
                token_id = self.encoder.get(piece)
                if token_id is None:
                    raise ValueError("CLIP tokenizer vocabulary is missing a BPE token")
                token_ids.append(token_id)
        token_ids = token_ids[:context_length - 1]
        token_ids.append(int(self.end_id))
        mask = [1] * len(token_ids)
        padding = context_length - len(token_ids)
        if padding > 0:
            token_ids.extend([self.pad_id] * padding)
            mask.extend([0] * padding)
        return token_ids, mask


class _QimTensorMetaBridge:
    """Attach tensor metadata to multi-memory buffers through the public QIM C API."""

    class _MetaHeader(ctypes.Structure):
        _fields_ = [
            ("flags", ctypes.c_uint32),
            ("_alignment", ctypes.c_uint32),
            ("info", ctypes.c_void_p),
            ("id", ctypes.c_uint32),
        ]

    def __init__(self):
        library = ctypes.util.find_library("gstqtimlbase")
        if not library:
            for candidate in ("/usr/lib/libgstqtimlbase.so.1", "/usr/lib/libgstqtimlbase.so"):
                if Path(candidate).is_file():
                    library = candidate
                    break
        if not library:
            raise RuntimeError("QIM multi-tensor metadata library is unavailable")
        self._library = ctypes.CDLL(library)
        type_from_string = self._library.gst_ml_type_from_string
        type_from_string.argtypes = [ctypes.c_char_p]
        type_from_string.restype = ctypes.c_int
        self._int32 = int(type_from_string(b"INT32"))
        self._add_meta = self._library.gst_buffer_add_ml_tensor_meta
        self._add_meta.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_uint,
                                   ctypes.POINTER(ctypes.c_uint)]
        self._add_meta.restype = ctypes.c_void_p
        if self._int32 < 0:
            raise RuntimeError("QIM does not expose the INT32 tensor type")

    def add(self, buffer, *, tensor_id: int, dimensions: Sequence[int]) -> None:
        shape = tuple(int(value) for value in dimensions)
        if not shape or any(value < 1 for value in shape):
            raise ValueError("QIM tensor dimensions must be positive")
        dims = (ctypes.c_uint * len(shape))(*shape)
        meta = self._add_meta(ctypes.c_void_p(hash(buffer)), self._int32, len(shape), dims)
        if not meta:
            raise RuntimeError("QIM rejected CLIP multi-input tensor metadata")
        # The QIM API assigns id 0 when adding metadata; the public layout defines
        # id as the GstMemory index. Set the second tensor's documented index.
        ctypes.cast(meta, ctypes.POINTER(self._MetaHeader)).contents.id = int(tensor_id)


class QimOnnxClipBackend:
    """Resident paired CLIP ONNX encoders through Qualcomm's QIM ONNX element.

    QNN GPU is requested first. If its first inference cannot execute, the same
    pinned model falls back to QIM's CPU execution provider. The QIM interface
    does not expose actual per-node placement, so this class never reports the
    requested provider as verified hardware utilization.
    """

    _IMAGE_CAPS = 'neural-network/tensors,type=FLOAT32,dimensions="<<1,3,224,224>>"'
    _TEXT_CAPS = 'neural-network/tensors,type=INT32,dimensions="<<1,77>,<1,77>>"'

    def __init__(self, *, image_model_path: Path, text_model_path: Path,
                 vocab_path: Path, merges_path: Path, backend_library: str,
                 model_id: str, output_dimensions: int = 512, timeout_s: float = 45.0,
                 execution_provider: str = "qnn", threads: int = 2):
        self.image_model_path = Path(image_model_path).resolve(strict=True)
        self.text_model_path = Path(text_model_path).resolve(strict=True)
        self.vocab_path = Path(vocab_path).resolve(strict=True)
        self.merges_path = Path(merges_path).resolve(strict=True)
        self.backend_library = str(backend_library)
        self._model_id = model_id
        self.output_dimensions = int(output_dimensions)
        self.timeout_s = float(timeout_s)
        self.threads = int(threads)
        if self.output_dimensions != 512 or self.timeout_s <= 0 or self.threads < 1:
            raise ValueError("invalid CLIP embedding dimension, timeout, or thread budget")
        if execution_provider not in {"qnn", "cpu"}:
            raise ValueError("QIM ONNX execution provider must be qnn or cpu")

        try:
            import cv2
            import numpy as np
            import gi
        except Exception as error:
            raise RuntimeError("QIM CLIP requires target OpenCV, NumPy and PyGObject") from error
        try:
            gi.require_version("Gst", "1.0")
            from gi.repository import Gst
        except Exception as error:
            raise RuntimeError("GStreamer Python bindings are unavailable") from error

        self._cv2 = cv2
        self._np = np
        self._Gst = Gst
        Gst.init(None)
        if Gst.ElementFactory.find("qtimlonnx") is None:
            raise RuntimeError("QIM qtimlonnx is unavailable")
        self._tokenizer = _ClipBpeTokenizer(self.vocab_path, self.merges_path)
        self._tensor_meta = _QimTensorMetaBridge()
        self._requested_provider = execution_provider
        self._active_provider = ""
        self._provider_failures: list[str] = []
        self._lock = threading.RLock()
        self._calls = {"image": 0, "text": 0}
        self._last_latency_us = {"image": None, "text": None}
        self._pipelines: dict[str, tuple[object, object, object]] = {}
        try:
            self._activate_provider(execution_provider)
        except Exception as error:
            if execution_provider != "qnn":
                raise
            self._provider_failures.append(type(error).__name__)
            self._activate_provider("cpu")
        self._asset_hashes = {
            "image_model_sha256": _sha256(self.image_model_path),
            "patched_text_model_sha256": _sha256(self.text_model_path),
            "vocab_sha256": _sha256(self.vocab_path),
            "merges_sha256": _sha256(self.merges_path),
        }

    @property
    def model_id(self) -> str:
        return self._model_id

    def _create_pipeline(self, channel: str, provider: str):
        Gst = self._Gst
        pipeline = Gst.Pipeline.new(f"hexagonflow-clip-{channel}-{provider}")
        source = Gst.ElementFactory.make("appsrc", f"clip-{channel}-input")
        inference = Gst.ElementFactory.make("qtimlonnx", f"clip-{channel}-onnx")
        sink = Gst.ElementFactory.make("appsink", f"clip-{channel}-output")
        if pipeline is None or source is None or inference is None or sink is None:
            raise RuntimeError("QIM CLIP pipeline elements could not be created")
        source.set_property("is-live", False)
        source.set_property("block", True)
        source.set_property("format", Gst.Format.TIME)
        source.set_property("caps", Gst.Caps.from_string(
            self._IMAGE_CAPS if channel == "image" else self._TEXT_CAPS))
        inference.set_property("model", str(
            self.image_model_path if channel == "image" else self.text_model_path))
        inference.set_property("execution-provider", 1 if provider == "qnn" else 0)
        if inference.find_property("threads") is not None:
            inference.set_property("threads", self.threads)
        if provider == "qnn":
            if inference.find_property("backend-path") is None:
                raise RuntimeError("QIM qtimlonnx does not expose backend-path")
            inference.set_property("backend-path", self.backend_library)
        sink.set_property("emit-signals", False)
        sink.set_property("sync", False)
        sink.set_property("max-buffers", 1)
        sink.set_property("drop", False)
        for element in (source, inference, sink):
            pipeline.add(element)
        if not source.link(inference) or not inference.link(sink):
            raise RuntimeError("QIM CLIP ONNX caps negotiation failed")
        result = pipeline.set_state(Gst.State.PLAYING)
        if result == Gst.StateChangeReturn.FAILURE:
            pipeline.set_state(Gst.State.NULL)
            raise RuntimeError(f"QIM CLIP {channel} pipeline failed to start")
        return pipeline, source, sink

    def _activate_provider(self, provider: str) -> None:
        created: dict[str, tuple[object, object, object]] = {}
        try:
            created["image"] = self._create_pipeline("image", provider)
            created["text"] = self._create_pipeline("text", provider)
        except Exception:
            for pipeline, _source, _sink in created.values():
                pipeline.set_state(self._Gst.State.NULL)
            raise
        for pipeline, _source, _sink in self._pipelines.values():
            pipeline.set_state(self._Gst.State.NULL)
        self._pipelines = created
        self._active_provider = provider

    def _text_buffer(self, input_ids: Sequence[int], attention_mask: Sequence[int]):
        Gst = self._Gst
        ids_payload = self._np.asarray(input_ids, dtype="<i4").tobytes()
        mask_payload = self._np.asarray(attention_mask, dtype="<i4").tobytes()
        buffer = Gst.Buffer.new_wrapped(ids_payload)
        mask_buffer = Gst.Buffer.new_wrapped(mask_payload)
        buffer.append_memory(mask_buffer.get_memory(0))
        if buffer.n_memory() != 2:
            raise RuntimeError("QIM CLIP text input must contain exactly two tensor memories")
        self._tensor_meta.add(buffer, tensor_id=0, dimensions=(1, 77))
        self._tensor_meta.add(buffer, tensor_id=1, dimensions=(1, 77))
        return buffer

    def _infer_once(self, channel: str, payload, *, multi_tensor: bool) -> list[float]:
        Gst = self._Gst
        _pipeline, source, sink = self._pipelines[channel]
        buffer = self._text_buffer(*payload) if multi_tensor else Gst.Buffer.new_wrapped(payload)
        buffer.pts = self._calls[channel] * Gst.SECOND
        buffer.duration = Gst.SECOND
        started = time.perf_counter_ns()
        flow = source.emit("push-buffer", buffer)
        if flow != Gst.FlowReturn.OK:
            raise RuntimeError(f"QIM CLIP {channel} input buffer was rejected")
        sample = sink.emit("try-pull-sample", int(self.timeout_s * Gst.SECOND))
        if sample is None:
            raise TimeoutError(f"QIM CLIP {channel} embedding timed out")
        output = sample.get_buffer()
        ok, mapped = output.map(Gst.MapFlags.READ)
        if not ok:
            raise RuntimeError(f"QIM CLIP {channel} output tensor could not be mapped")
        try:
            values = self._np.frombuffer(mapped.data, dtype="<f4")
            if values.size != self.output_dimensions:
                raise RuntimeError("QIM CLIP output dimension differs from the pinned 512D contract")
            result = _unit(values.tolist())
        finally:
            output.unmap(mapped)
        self._calls[channel] += 1
        self._last_latency_us[channel] = (time.perf_counter_ns() - started) // 1000
        return result

    def _infer(self, channel: str, payload, *, multi_tensor: bool = False) -> list[float]:
        try:
            return self._infer_once(channel, payload, multi_tensor=multi_tensor)
        except Exception as error:
            if self._active_provider != "qnn" or self._requested_provider != "qnn":
                raise
            self._provider_failures.append(type(error).__name__)
            self._activate_provider("cpu")
            return self._infer_once(channel, payload, multi_tensor=multi_tensor)

    def embed_image(self, image_path: Path) -> list[float]:
        with self._lock:
            frame = self._cv2.imread(str(image_path), self._cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError("CLIP image could not be decoded")
            rgb = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)
            resized = self._cv2.resize(rgb, (224, 224), interpolation=self._cv2.INTER_CUBIC)
            tensor = resized.astype(self._np.float32) / 255.0
            mean = self._np.asarray([0.48145466, 0.4578275, 0.40821073], dtype=self._np.float32)
            std = self._np.asarray([0.26862954, 0.26130258, 0.27577711], dtype=self._np.float32)
            tensor = (tensor - mean) / std
            payload = self._np.transpose(tensor, (2, 0, 1))[self._np.newaxis, ...]
            return self._infer("image", payload.astype("<f4", copy=False).tobytes(),
                               multi_tensor=False)

    def embed_text(self, text: str) -> list[float]:
        if not text.strip():
            raise ValueError("cannot embed empty visual query")
        with self._lock:
            ids, mask = self._tokenizer.encode(text)
            return self._infer("text", (ids, mask), multi_tensor=True)

    def snapshot(self) -> dict:
        return {
            "status": "resident" if any(self._calls.values()) else "pipeline_ready",
            "backend_id": f"qim-qtimlonnx:{self._active_provider}",
            "model_id": self.model_id,
            **self._asset_hashes,
            "output_dimensions": self.output_dimensions,
            "requested_execution_provider": self._requested_provider,
            "active_execution_provider": self._active_provider,
            "requested_qnn_backend_library": (self.backend_library
                                               if self._requested_provider == "qnn" else None),
            "provider_fallback_errors": list(self._provider_failures),
            "device_execution_trace": "not_exposed_by_qtimlonnx_api",
            "calls": dict(self._calls),
            "last_latency_us": dict(self._last_latency_us),
        }

    def close(self) -> None:
        for pipeline, _source, _sink in self._pipelines.values():
            pipeline.set_state(self._Gst.State.NULL)
        self._pipelines = {}


def _sha256(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class CpuClipBackend:
 def __init__(self, *, image_model_path, text_model_path, vocab_path, merges_path, model_id, threads=2):
  import cv2, numpy as np, onnxruntime as ort
  self.cv2=cv2; self.np=np; self.ort=ort
  opts=ort.SessionOptions(); opts.intra_op_num_threads=threads
  self.image=ort.InferenceSession(str(image_model_path),opts,providers=["CPUExecutionProvider"])
  self.text=ort.InferenceSession(str(text_model_path),opts,providers=["CPUExecutionProvider"])
  self.tok=_ClipBpeTokenizer(Path(vocab_path),Path(merges_path)); self.model_id=model_id
 def embed_text(self,text):
  np=self.np
  ids,mask=self.tok.encode(text); y=self.text.run(None,{"input_ids":np.array([ids],dtype=np.int64),"attention_mask":np.array([mask],dtype=np.int64)})[0]; return _unit(y.flatten().tolist())
 def embed_image(self,path):
  cv2=self.cv2; np=self.np
  im=cv2.imread(str(path));
  if im is None: raise ValueError("image unreadable")
  h,w=im.shape[:2]; scale=224/min(h,w); im=cv2.resize(im,(round(w*scale),round(h*scale)),interpolation=cv2.INTER_CUBIC); h,w=im.shape[:2]; im=im[(h-224)//2:(h-224)//2+224,(w-224)//2:(w-224)//2+224]; im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB).astype(np.float32)/255
  im=(im-np.array([.48145466,.4578275,.40821073],np.float32))/np.array([.26862954,.26130258,.27577711],np.float32); x=np.transpose(im,(2,0,1))[None,...]; return _unit(self.image.run(None,{self.image.get_inputs()[0].name:x})[0].flatten().tolist())
 def snapshot(self): return {"status":"resident","backend_id":"onnxruntime-cpu-clip","model_id":self.model_id,"execution_provider":"CPUExecutionProvider"}
 def close(self): self.image=None; self.text=None


class MnnOpenCLClipBackend:
    """CLIP image encoder on MNN OpenCL, paired with CPU ONNX text encoder.

    The bridge reports the backend selected by the MNN session. If it cannot
    confirm MNN_FORWARD_OPENCL or an inference call fails, image encoding
    switches to the pinned ONNX Runtime CPU path. Text always stays on CPU.
    """

    def __init__(self, *, image_model_path, mnn_model_path, text_model_path,
                 vocab_path, merges_path, bridge_path, model_id, threads=2):
        import ctypes
        import cv2
        import numpy as np
        import onnxruntime as ort

        self.ctypes = ctypes
        self.cv2 = cv2
        self.np = np
        self.model_id = model_id
        self._lock = threading.RLock()
        self._handle = None
        self._image_cpu = None
        self._failures = []
        self._calls = {"image_opencl": 0, "image_cpu_fallback": 0, "text_cpu": 0}
        self._last_latency_us = {}
        self._opencl_device_name = ""
        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, int(threads))
        self._text = ort.InferenceSession(
            str(text_model_path), options, providers=["CPUExecutionProvider"])
        self._tokenizer = _ClipBpeTokenizer(Path(vocab_path), Path(merges_path))
        self._image_model_path = Path(image_model_path)
        self._mnn_model_path = Path(mnn_model_path)
        self._active_image_backend = "onnxruntime-cpu"
        self._requested_image_backend = "mnn-opencl"
        try:
            library = ctypes.CDLL(str(bridge_path))
            library.hexagonflow_mnn_clip_opencl_create.argtypes = [
                ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t]
            library.hexagonflow_mnn_clip_opencl_create.restype = ctypes.c_void_p
            library.hexagonflow_mnn_clip_opencl_infer.argtypes = [
                ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_size_t), ctypes.c_char_p, ctypes.c_size_t]
            library.hexagonflow_mnn_clip_opencl_infer.restype = ctypes.c_int
            library.hexagonflow_mnn_clip_opencl_device_name.argtypes = [
                ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]
            library.hexagonflow_mnn_clip_opencl_device_name.restype = ctypes.c_int
            library.hexagonflow_mnn_clip_opencl_destroy.argtypes = [ctypes.c_void_p]
            library.hexagonflow_mnn_clip_opencl_destroy.restype = None
            self._library = library
            error = ctypes.create_string_buffer(256)
            self._handle = library.hexagonflow_mnn_clip_opencl_create(
                os.fsencode(self._mnn_model_path), error, len(error))
            if not self._handle:
                raise RuntimeError(error.value.decode("utf-8", "replace") or
                                   "mnn_opencl_initialization_failed")
            device = ctypes.create_string_buffer(512)
            library.hexagonflow_mnn_clip_opencl_device_name(
                self._handle, device, len(device))
            self._opencl_device_name = device.value.decode("utf-8", "replace")
            self._active_image_backend = "mnn-opencl"
        except Exception as error:
            self._failures.append(type(error).__name__)
            self._enable_cpu_fallback()

    def _enable_cpu_fallback(self):
        import onnxruntime as ort
        options = ort.SessionOptions()
        self._image_cpu = ort.InferenceSession(
            str(self._image_model_path), options, providers=["CPUExecutionProvider"])
        self._active_image_backend = "onnxruntime-cpu-fallback"

    def _preprocess(self, image_path):
        np = self.np
        image = self.cv2.imread(str(image_path), self.cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("CLIP image could not be decoded")
        height, width = image.shape[:2]
        scale = 224 / min(height, width)
        image = self.cv2.resize(
            image, (round(width * scale), round(height * scale)),
            interpolation=self.cv2.INTER_CUBIC)
        height, width = image.shape[:2]
        image = image[(height - 224) // 2:(height - 224) // 2 + 224,
                      (width - 224) // 2:(width - 224) // 2 + 224]
        image = self.cv2.cvtColor(image, self.cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.asarray([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
        std = np.asarray([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
        image = (image - mean) / std
        return np.ascontiguousarray(np.transpose(image, (2, 0, 1))[None, ...], dtype="<f4")

    def embed_image(self, image_path: Path) -> list[float]:
        with self._lock:
            started = time.perf_counter_ns()
            tensor = self._preprocess(image_path)
            if self._handle is not None:
                output = (self.ctypes.c_float * 512)()
                count = self.ctypes.c_size_t()
                error = self.ctypes.create_string_buffer(256)
                code = self._library.hexagonflow_mnn_clip_opencl_infer(
                    self._handle,
                    tensor.ctypes.data_as(self.ctypes.POINTER(self.ctypes.c_float)),
                    tensor.size, output, len(output), self.ctypes.byref(count),
                    error, len(error))
                if code == 3 and count.value == 512:
                    self._calls["image_opencl"] += 1
                    self._last_latency_us["image_opencl"] = (
                        time.perf_counter_ns() - started) // 1000
                    return _unit(list(output[:count.value]))
                reason = error.value.decode("utf-8", "replace") or f"backend_code_{code}"
                self._failures.append(reason)
                self._library.hexagonflow_mnn_clip_opencl_destroy(self._handle)
                self._handle = None
                self._enable_cpu_fallback()
            values = self._image_cpu.run(
                None, {self._image_cpu.get_inputs()[0].name: tensor})[0].reshape(-1).tolist()
            self._calls["image_cpu_fallback"] += 1
            self._last_latency_us["image_cpu_fallback"] = (
                time.perf_counter_ns() - started) // 1000
            return _unit(values)

    def embed_text(self, text: str) -> list[float]:
        if not text.strip():
            raise ValueError("cannot embed empty visual query")
        with self._lock:
            ids, mask = self._tokenizer.encode(text)
            np = self.np
            values = self._text.run(None, {
                "input_ids": np.asarray([ids], dtype=np.int64),
                "attention_mask": np.asarray([mask], dtype=np.int64),
            })[0].reshape(-1).tolist()
            self._calls["text_cpu"] += 1
            return _unit(values)

    def snapshot(self):
        if self._handle is not None:
            device_trace = ("MNN_FORWARD_OPENCL" +
                            (f"; OpenCL GPU={self._opencl_device_name}"
                             if self._opencl_device_name else "; OpenCL GPU name unavailable"))
        else:
            device_trace = "CPUExecutionProvider"
        return {
            "status": "resident",
            "backend_id": "mnn-opencl-image-with-cpu-fallback+onnxruntime-cpu-text",
            "model_id": self.model_id,
            "requested_image_backend": self._requested_image_backend,
            "active_image_backend": self._active_image_backend,
            "text_execution_provider": "CPUExecutionProvider",
            "device_execution_trace": device_trace,
            "fallback_errors": list(self._failures), "calls": dict(self._calls),
            "last_latency_us": dict(self._last_latency_us), "output_dimensions": 512,
        }

    def close(self):
        with self._lock:
            if self._handle is not None:
                self._library.hexagonflow_mnn_clip_opencl_destroy(self._handle)
                self._handle = None
            self._text = None
            self._image_cpu = None
