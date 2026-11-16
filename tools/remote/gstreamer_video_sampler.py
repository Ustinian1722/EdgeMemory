#!/usr/bin/env python3
"""Decode one video and encode JPEG only for frames that pass the scene gate.

This helper intentionally has no model or SDK dependency. Its JSON output is
consumed by the local HexagonFlow process and must not contain credentials.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", required=True, type=int)
    parser.add_argument("--interval-ms", required=True, type=int)
    parser.add_argument("--max-frame-side", required=True, type=int)
    parser.add_argument("--jpeg-quality", required=True, type=int)
    parser.add_argument("--scene-threshold", type=float, default=0.16)
    parser.add_argument("--max-interval-ms", type=int, default=12000)
    parser.add_argument("--preprocess-path", choices=("auto", "software", "qim"),
                        default="auto")
    parser.add_argument("--decoder-io-mode", choices=("auto", "dmabuf"), default="auto")
    parser.add_argument("--stream-jsonl", action="store_true",
                        help="emit each encoded frame as soon as appsink receives it")
    parser.add_argument("--decoder-factory",
                        help="require this exact GStreamer video decoder factory inside decodebin")
    return parser.parse_args()


def _caps_trace(caps) -> dict | None:
    if caps is None:
        return None
    features = []
    try:
        for index in range(caps.get_size()):
            item = caps.get_features(index)
            features.append(item.to_string() if item is not None else "")
    except Exception:
        features = []
    return {"caps": caps.to_string(), "features": features}


def _buffer_trace(buffer, caps) -> dict:
    memory_count = int(buffer.n_memory())
    allocator_types: list[str] = []
    dmabuf_detected = None
    try:
        import gi
        gi.require_version("GstAllocators", "1.0")
        from gi.repository import GstAllocators
        dmabuf_detected = False
        for index in range(memory_count):
            memory = buffer.peek_memory(index)
            allocator = getattr(memory, "allocator", None)
            memory_type = getattr(allocator, "mem_type", "") if allocator else ""
            if memory_type:
                allocator_types.append(str(memory_type))
            is_dmabuf = bool(GstAllocators.is_dmabuf_memory(memory))
            dmabuf_detected = dmabuf_detected or is_dmabuf
    except Exception:
        for index in range(memory_count):
            try:
                memory = buffer.peek_memory(index)
                allocator = getattr(memory, "allocator", None)
                memory_type = getattr(allocator, "mem_type", "") if allocator else ""
                if memory_type:
                    allocator_types.append(str(memory_type))
            except Exception:
                break
    caps_info = _caps_trace(caps)
    if dmabuf_detected is None and caps_info is not None:
        dmabuf_detected = any("dmabuf" in item.casefold()
                              for item in caps_info["features"])
    return {
        **(caps_info or {}),
        "memory_count": memory_count,
        "allocator_types": sorted(set(allocator_types)),
        "dmabuf_detected": dmabuf_detected,
    }


def _property_trace(element, name: str):
    try:
        value = element.get_property(name)
    except Exception:
        return None
    nick = getattr(value, "value_nick", None)
    return str(nick if nick is not None else value)


_NATIVE_STDOUT = None


def _separate_native_output() -> None:
    """Keep native GStreamer diagnostics off the machine-readable JSONL pipe."""
    global _NATIVE_STDOUT
    if _NATIVE_STDOUT is not None:
        return
    original = sys.stdout
    try:
        original.flush()
        output_fd = original.fileno()
        error_fd = sys.stderr.fileno()
        json_fd = os.dup(output_fd)
        os.dup2(error_fd, output_fd)
        sys.stdout = os.fdopen(json_fd, "w", buffering=1)
        _NATIVE_STDOUT = original
    except (AttributeError, OSError, ValueError):
        return


def main() -> int:
    args = _arguments()
    _separate_native_output()
    source = Path(args.video)
    output = Path(args.output)
    if (not source.is_file() or args.max_samples < 0 or
            (args.max_samples == 0 and not args.stream_jsonl) or args.interval_ms < 1 or
            args.max_interval_ms < args.interval_ms or
            not 0.0 <= args.scene_threshold <= 1.0 or
            not 1 <= args.jpeg_quality <= 100):
        return 2
    output.mkdir(mode=0o700, parents=True, exist_ok=True)

    try:
        import gi
        gi.require_version("Gst", "1.0")
        gi.require_version("GstVideo", "1.0")
        from gi.repository import Gst, GstVideo
        import cv2
        import numpy as np
    except Exception:
        return 3
    Gst.init(None)
    if (not args.decoder_factory and Path(args.video).suffix.casefold() == ".mp4"
            and Gst.ElementFactory.find("v4l2h264dec") is not None):
        args.decoder_factory = "v4l2h264dec"
    if args.decoder_factory and Gst.ElementFactory.find(args.decoder_factory) is None:
        return 9

    explicit_h264 = args.decoder_factory == "v4l2h264dec"
    pipeline = Gst.Pipeline.new("hexagonflow-video-sampler")
    source_element = Gst.ElementFactory.make("filesrc", "source")
    demux = Gst.ElementFactory.make("qtdemux", "demux") if explicit_h264 else None
    parser = Gst.ElementFactory.make("h264parse", "h264-parser") if explicit_h264 else None
    decoder = Gst.ElementFactory.make(
        args.decoder_factory if explicit_h264 else "decodebin", "decoder")
    decoded_queue = Gst.ElementFactory.make("queue", "bounded-decoded-frames")
    rate = Gst.ElementFactory.make("videorate", "rate")
    rate_capsfilter = Gst.ElementFactory.make("capsfilter", "sample-cadence")
    selected_preprocess = args.preprocess_path
    if selected_preprocess == "auto":
        # The QCS9075 image's QIM path is enabled when the factory exists.
        # The explicit software option remains available as a safe fallback.
        selected_preprocess = ("qim" if Gst.ElementFactory.find("qtivtransform") is not None
                               else "software")
    qim_transform = (Gst.ElementFactory.make("qtivtransform", "qim-transform")
                     if selected_preprocess == "qim" else None)
    scale = (Gst.ElementFactory.make("videoscale", "scale")
             if selected_preprocess == "software" else None)
    convert = (Gst.ElementFactory.make("videoconvert", "convert")
               if selected_preprocess == "software" else None)
    capsfilter = Gst.ElementFactory.make("capsfilter", "sample-format-size")
    sink = Gst.ElementFactory.make("appsink", "evidence")
    required = [pipeline, source_element, decoder, decoded_queue, rate, rate_capsfilter,
                capsfilter, sink]
    required.extend((qim_transform,) if selected_preprocess == "qim"
                    else (scale, convert))
    if explicit_h264:
        required.extend((demux, parser))
    if any(item is None for item in required):
        return 4
    effective_io_mode = args.decoder_io_mode
    if (effective_io_mode == "auto" and explicit_h264 and
            all(decoder.find_property(name) is not None
                for name in ("capture-io-mode", "output-io-mode"))):
        # QCS9075's msm_vidc_v4l2 path was exercised with mode 4 on target.
        # The later buffer probes report whether DMABUF actually survived caps.
        effective_io_mode = "dmabuf"
    if effective_io_mode == "dmabuf":
        if not explicit_h264:
            return 11
        for property_name in ("capture-io-mode", "output-io-mode"):
            if decoder.find_property(property_name) is None:
                if args.decoder_io_mode == "auto":
                    effective_io_mode = "auto"
                    break
                return 11
            decoder.set_property(property_name, 4)

    source_element.set_property("location", str(source))
    rate.set_property("drop-only", True)
    rate.set_property("silent", True)
    sink.set_property("emit-signals", True)
    sink.set_property("sync", True)
    sink.set_property("max-buffers", 2)
    sink.set_property("drop", False)
    decoded_queue.set_property("max-size-buffers", 2)
    decoded_queue.set_property("max-size-bytes", 0)
    decoded_queue.set_property("max-size-time", 0)
    decoded_queue.set_property("leaky", 0)

    # videorate's rational framerate avoids accumulated float timestamp drift.
    interval_s = args.interval_ms / 1000.0
    denominator = max(1, round(interval_s * 1000))
    numerator = 1000
    divisor = math.gcd(numerator, denominator)
    numerator //= divisor
    denominator //= divisor
    requested_caps = Gst.Caps.from_string(
        f"video/x-raw,framerate={numerator}/{denominator}")
    rate_capsfilter.set_property("caps", requested_caps)

    elements = [source_element]
    if explicit_h264:
        elements.extend((demux, parser))
    elements.extend((decoder, decoded_queue, rate, rate_capsfilter))
    if selected_preprocess == "qim":
        elements.extend((qim_transform, capsfilter, sink))
    else:
        elements.extend((scale, convert, capsfilter, sink))
    for element in elements:
        pipeline.add(element)
    if explicit_h264:
        if not source_element.link(demux) or not parser.link(decoder) or not decoder.link(decoded_queue):
            return 5
        decoder_factories_seed = {args.decoder_factory}
    else:
        if not source_element.link(decoder):
            return 5
        decoder_factories_seed = set()
    chain = ([decoded_queue, rate, rate_capsfilter, qim_transform, capsfilter, sink]
             if selected_preprocess == "qim" else
             [decoded_queue, rate, rate_capsfilter, scale, convert, capsfilter, sink])
    for upstream, downstream in zip(chain, chain[1:]):
        if not upstream.link(downstream):
            return 6

    dimensions = {"width": 0, "height": 0, "sample_width": 0,
                  "sample_height": 0, "fps_num": 0, "fps_den": 1}
    buffer_observations: dict[str, dict | None] = {
        "decoder_output": None, "preprocess_output": None, "appsink_output": None,
    }
    decoder_factories: set[str] = set(decoder_factories_seed)
    frame_rows: list[dict[str, int | float | str]] = []
    sequence = 0
    candidate_count = 0
    previous_gray = None
    last_selected_ms: int | None = None
    first_frame_ns = time.monotonic_ns()
    reached_limit = False

    def deep_added(_bin, _sub_bin, element):
        factory = element.get_factory()
        if factory is not None:
            name = factory.get_name()
            if "dec" in name.casefold():
                decoder_factories.add(name)

    if not explicit_h264:
        decoder.connect("deep-element-added", deep_added)

    # GstAutoplugSelectResult is missing from the PyGI namespace on the
    # Qualcomm Linux image we target. The public C enum values are TRY=0,
    # EXPOSE=1, SKIP=2; use the ABI values so the optional decoder constraint
    # remains usable on that image.
    def autoplug_select(_bin, _pad, _caps, factory):
        if not args.decoder_factory:
            return 0
        klass = (factory.get_klass() or "").casefold()
        if "decoder" in klass and "video" in klass:
            return 0 if factory.get_name() == args.decoder_factory else 2
        return 0

    if args.decoder_factory and not explicit_h264:
        decoder.connect("autoplug-select", autoplug_select)

    def pad_added(_decoder, pad):
        caps = pad.get_current_caps() or pad.query_caps(None)
        if not caps or caps.get_size() < 1:
            return
        structure = caps.get_structure(0)
        if not structure.get_name().startswith("video/x-raw"):
            return
        width = int(structure.get_value("width"))
        height = int(structure.get_value("height"))
        fps_num = 0
        fps_den = 1
        try:
            fps = structure.get_value("framerate")
            fps_num, fps_den = int(fps.num), max(1, int(fps.denom))
        except Exception:
            pass
        factor = min(1.0, args.max_frame_side / max(width, height))
        target_width = max(4, round(width * factor) // 4 * 4)
        target_height = max(1, round(height * factor))
        pixel_format = "BGR"
        capsfilter.set_property("caps", Gst.Caps.from_string(
            f"video/x-raw,format={pixel_format},width={target_width},height={target_height}"))
        dimensions.update(width=width, height=height,
                          sample_width=target_width, sample_height=target_height,
                          fps_num=fps_num, fps_den=fps_den)
        target = decoded_queue.get_static_pad("sink")
        if not target.is_linked():
            pad.link(target)

    def demux_pad_added(_demux, pad):
        caps = pad.get_current_caps() or pad.query_caps(None)
        if not caps or caps.get_size() < 1:
            return
        structure = caps.get_structure(0)
        if not structure.get_name().startswith("video/x-h264"):
            return
        try:
            width = int(structure.get_value("width"))
            height = int(structure.get_value("height"))
        except (TypeError, ValueError):
            width = height = 0
        if width > 0 and height > 0:
            factor = min(1.0, args.max_frame_side / max(width, height))
            target_width = max(4, round(width * factor) // 4 * 4)
            target_height = max(1, round(height * factor))
            capsfilter.set_property("caps", Gst.Caps.from_string(
                f"video/x-raw,format=BGR,width={target_width},height={target_height}"))
            dimensions.update(width=width, height=height,
                              sample_width=target_width, sample_height=target_height)
        target = parser.get_static_pad("sink")
        if not target.is_linked():
            pad.link(target)

    if explicit_h264:
        demux.connect("pad-added", demux_pad_added)
    else:
        decoder.connect("pad-added", pad_added)

    def observe_buffer(stage: str):
        def probe(pad, info):
            if buffer_observations[stage] is None:
                buffer = info.get_buffer()
                if buffer is not None:
                    buffer_observations[stage] = _buffer_trace(
                        buffer, pad.get_current_caps())
            return Gst.PadProbeReturn.OK
        return probe

    decoded_src = decoded_queue.get_static_pad("src")
    decoded_src.add_probe(Gst.PadProbeType.BUFFER, observe_buffer("decoder_output"))
    if qim_transform is not None:
        qim_src = qim_transform.get_static_pad("src")
        qim_src.add_probe(Gst.PadProbeType.BUFFER, observe_buffer("preprocess_output"))

    def new_sample(appsink):
        nonlocal sequence, candidate_count, previous_gray, last_selected_ms, reached_limit
        sample = appsink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.ERROR
        buffer = sample.get_buffer()
        if buffer_observations["appsink_output"] is None:
            buffer_observations["appsink_output"] = _buffer_trace(
                buffer, sample.get_caps())
        pts = buffer.pts
        if pts == Gst.CLOCK_TIME_NONE:
            return Gst.FlowReturn.OK
        success, mapped = buffer.map(Gst.MapFlags.READ)
        if not success:
            return Gst.FlowReturn.ERROR
        try:
            width = int(dimensions["sample_width"])
            height = int(dimensions["sample_height"])
            info = GstVideo.VideoInfo.new_from_caps(sample.get_caps())
            if info is None:
                print("gstreamer_sampler_video_info_unavailable", file=sys.stderr, flush=True)
                return Gst.FlowReturn.ERROR
            meta = GstVideo.buffer_get_video_meta(buffer)
            if meta is not None:
                if int(meta.width) != width or int(meta.height) != height:
                    print("gstreamer_sampler_video_meta_dimensions_mismatch", file=sys.stderr, flush=True)
                    return Gst.FlowReturn.ERROR
                stride = int(meta.stride[0])
                offset = int(meta.offset[0])
            else:
                stride = int(info.stride[0])
                offset = int(info.offset[0])
            required = offset + stride * max(0, height - 1) + width * 3
            if stride < width * 3 or len(mapped.data) < required:
                print("gstreamer_sampler_invalid_bgr_plane", file=sys.stderr, flush=True)
                return Gst.FlowReturn.ERROR
            frame = np.ndarray((height, width, 3), dtype=np.uint8,
                               buffer=mapped.data, offset=offset,
                               strides=(stride, 3, 1))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
            delta = (0.0 if previous_gray is None else
                     float(cv2.absdiff(gray, previous_gray).mean()) / 255.0)
            timestamp_ms = int(pts // Gst.MSECOND)
            candidate_count += 1
            elapsed_ms = None if last_selected_ms is None else timestamp_ms - last_selected_ms
            reason = "first_frame" if last_selected_ms is None else ""
            if last_selected_ms is not None and elapsed_ms is not None:
                if elapsed_ms >= args.max_interval_ms:
                    reason = "max_interval"
                elif elapsed_ms >= args.interval_ms and delta >= args.scene_threshold:
                    reason = "scene_change"
            if not reason:
                if previous_gray is None or delta >= args.scene_threshold / 4:
                    previous_gray = gray.copy()
                payload = None
            else:
                encoded_ok, encoded = cv2.imencode(
                    ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
                if not encoded_ok:
                    return Gst.FlowReturn.ERROR
                payload = encoded.tobytes()
        finally:
            buffer.unmap(mapped)
        if payload is None:
            return Gst.FlowReturn.OK
        frame_path = output / f"frame_{sequence:09d}_{timestamp_ms:012d}.jpg"
        frame_path.write_bytes(payload)
        frame_row = {"frame_index": sequence, "timestamp_ms": timestamp_ms,
                     "path": str(frame_path), "size_bytes": len(payload),
                     "sample_reason": reason, "scene_delta": delta}
        if not args.stream_jsonl:
            frame_rows.append(frame_row)
        previous_gray = gray.copy()
        last_selected_ms = timestamp_ms
        if args.stream_jsonl:
            print(json.dumps({
                "type": "frame", "schema_version": 2,
                "frame": frame_row,
                "metadata": {**dimensions,
                             "fps": (dimensions["fps_num"] / dimensions["fps_den"]
                                     if dimensions["fps_num"] else 0)},
            }, separators=(",", ":")), flush=True)
        sequence += 1
        if args.max_samples and sequence >= args.max_samples:
            reached_limit = True
            pipeline.send_event(Gst.Event.new_eos())
        return Gst.FlowReturn.OK

    sink.connect("new-sample", new_sample)
    if pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
        return 7
    bus = pipeline.get_bus()
    message = bus.timed_pop_filtered(
        Gst.CLOCK_TIME_NONE, Gst.MessageType.ERROR | Gst.MessageType.EOS)
    elapsed_ms = (time.monotonic_ns() - first_frame_ns) // 1_000_000
    error = None
    if message is not None and message.type == Gst.MessageType.ERROR:
        error, debug = message.parse_error()
        print(f"gstreamer_decode_error: {error}; debug={debug or ''}",
              file=sys.stderr, flush=True)
        error = "gstreamer_decode_error"
    duration_ns = -1
    try:
        ok, value = pipeline.query_duration(Gst.Format.TIME)
        if ok:
            duration_ns = int(value)
    except Exception:
        pass
    pipeline.set_state(Gst.State.NULL)
    if error:
        for path in output.glob("frame_*.jpg"):
            try:
                path.unlink()
            except OSError:
                pass
        return 8

    if args.decoder_factory and args.decoder_factory not in decoder_factories:
        for path in output.glob("frame_*.jpg"):
            try:
                path.unlink()
            except OSError:
                pass
        return 10

    summary = {
        "schema_version": 2,
        "duration_ms": max(0, duration_ns // Gst.MSECOND),
        "source_frame_count": 0,
        "fps": dimensions["fps_num"] / dimensions["fps_den"] if dimensions["fps_num"] else 0,
        "width": dimensions["width"],
        "height": dimensions["height"],
        "sample_width": dimensions["sample_width"],
        "sample_height": dimensions["sample_height"],
        "decoder_factories": sorted(decoder_factories),
        "selected_decoder_factory": args.decoder_factory or "auto",
        "decoder_io_mode_requested": args.decoder_io_mode,
        "decoder_io_mode": effective_io_mode,
        "decoder_properties": {
            "device": _property_trace(decoder, "device"),
            "capture_io_mode": _property_trace(decoder, "capture-io-mode"),
            "output_io_mode": _property_trace(decoder, "output-io-mode"),
        },
        "preprocess_path": selected_preprocess,
        "preprocess_path_requested": args.preprocess_path,
        "preprocess_element": "qtivtransform" if selected_preprocess == "qim" else "videoscale+videoconvert",
        "preprocess_engine": (_property_trace(qim_transform, "engine")
                              if qim_transform is not None else None),
        "buffer_observations": buffer_observations,
        "copy_boundary": "appsink maps BGR buffers; selected frames are JPEG-encoded and written for the GenieX path",
        "scene_gate": {"threshold": args.scene_threshold,
                        "minimum_interval_ms": args.interval_ms,
                        "maximum_interval_ms": args.max_interval_ms},
        "candidate_frames": candidate_count,
        "jpeg_encoded_frames": sequence,
        "selected_frame_count": sequence,
        "decode_elapsed_ms": elapsed_ms,
        "playback_mode": "realtime_sync",
        "decode_wall_to_duration_ratio": (round(elapsed_ms / max(1, duration_ns // Gst.MSECOND), 4) if duration_ns > 0 else None),
        "max_samples_reached": reached_limit,
        "bounded_decoder_queue_buffers": 2,
        "decoder_queue_leaky": False,
        "appsink_max_buffers": 2,
        "frames": frame_rows if not args.stream_jsonl else [],
    }
    if args.stream_jsonl:
        summary["type"] = "summary"
    print(json.dumps(summary, separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
