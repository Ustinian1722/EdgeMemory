from __future__ import annotations

"""Best-effort target telemetry with explicit source and availability labels."""

from pathlib import Path
import os
import re
import threading
import time


class HardwareProfiler:
    GPU_CANDIDATES = (
        Path("/sys/class/kgsl/kgsl-3d0/gpu_busy_percentage"),
        Path("/sys/class/kgsl/kgsl-3d0/devfreq/cur_freq"),
        Path("/sys/class/kgsl/kgsl-3d0/devfreq/max_freq"),
    )
    FASTRPC_CANDIDATES = (
        Path("/dev/fastrpc-cdsp"), Path("/dev/fastrpc-adsp"),
        Path("/dev/fastrpc-sdsp"), Path("/dev/fastrpc-mdsp"),
    )

    def __init__(self):
        self._lock = threading.Lock()
        self._cpu_previous: tuple[int, int] | None = None
        self._sampled_at_ns = 0

    @staticmethod
    def _read_int(path: Path) -> int | None:
        try:
            raw = path.read_text(encoding="ascii").strip()
            match = re.search(r"-?\d+", raw)
            return int(match.group(0)) if match else None
        except (OSError, ValueError):
            return None

    @staticmethod
    def _cpu_counters() -> tuple[int, int] | None:
        try:
            first = Path("/proc/stat").read_text(encoding="ascii").splitlines()[0]
            fields = [int(value) for value in first.split()[1:]]
            if len(fields) < 4:
                return None
            idle = fields[3] + (fields[4] if len(fields) > 4 else 0)
            return sum(fields), idle
        except (OSError, ValueError, IndexError):
            return None

    @staticmethod
    def _memory() -> dict:
        values = {}
        try:
            for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
                key, separator, rest = line.partition(":")
                if separator and key in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}:
                    values[key] = int(rest.split()[0]) * 1024
        except (OSError, ValueError, IndexError):
            pass
        return {key: {"value": value, "unit": "bytes", "source": "/proc/meminfo",
                      "status": "available"} for key, value in values.items()}

    def snapshot(self) -> dict:
        with self._lock:
            now = time.time_ns()
            current = self._cpu_counters()
            cpu = {"value": None, "unit": "percent", "source": "/proc/stat",
                   "status": "unavailable"}
            if current is not None and self._cpu_previous is not None:
                total_delta = current[0] - self._cpu_previous[0]
                idle_delta = current[1] - self._cpu_previous[1]
                if total_delta > 0 and 0 <= idle_delta <= total_delta:
                    cpu = {"value": round(100.0 * (total_delta - idle_delta) / total_delta, 2),
                           "unit": "percent", "source": "/proc/stat delta",
                           "status": "available"}
            self._cpu_previous = current
            self._sampled_at_ns = now

        gpu = {}
        for path in self.GPU_CANDIDATES:
            value = self._read_int(path)
            key = "busy_percent" if "busy_percentage" in str(path) else (
                "current_frequency_hz" if "cur_freq" in str(path) else "maximum_frequency_hz")
            gpu[key] = {"value": value, "unit": "percent" if key == "busy_percent" else "Hz",
                        "source": str(path),
                        "status": "available" if value is not None else "unavailable"}

        threads = rss = None
        process_status = Path("/proc/self/status")
        try:
            for line in process_status.read_text(encoding="ascii").splitlines():
                if line.startswith("Threads:"):
                    threads = int(line.split()[1])
                elif line.startswith("VmRSS:"):
                    rss = int(line.split()[1]) * 1024
        except (OSError, ValueError, IndexError):
            pass
        process = {
            "threads": {"value": threads, "unit": "threads", "source": str(process_status),
                        "status": "available" if threads is not None else "unavailable"},
            "rss_bytes": {"value": rss, "unit": "bytes", "source": str(process_status),
                          "status": "available" if rss is not None else "unavailable"},
        }
        fastrpc = []
        for path in self.FASTRPC_CANDIDATES:
            if path.exists():
                fastrpc.append({"path": str(path), "readable_by_process": os.access(path, os.R_OK),
                                "writable_by_process": os.access(path, os.W_OK),
                                "status": "present"})
        return {
            "collected_at_ns": now,
            "cpu_busy": cpu,
            "memory": self._memory(),
            "device_process": process,
            "gpu_sysfs": gpu,
            "fastrpc_nodes": fastrpc,
            "limitations": "Unavailable counters are not inferred from backend selection; process RSS is not accelerator memory.",
        }
