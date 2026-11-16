from contextlib import contextmanager
import threading
import time
import uuid

class DirectExecution:
    def __init__(self, resources, wait_timeout_s):
        self.resources = resources
        self.wait_timeout_s = wait_timeout_s
        self._htp_lock = threading.Lock()
        self._locks = {r["resource_id"]: (self._htp_lock if r["family"] == 1 else threading.Lock()) for r in resources}
        self._local = threading.local()
        self._history = []
        self._guard = threading.Lock()
        self._closed = False

    @contextmanager
    def reserve(self, model_id, *, allowed_backend_ids=None, **kwargs):
        allowed = set(allowed_backend_ids or (r["backend_id"] for r in self.resources))
        candidates = [r for r in self.resources if r["backend_id"] in allowed]
        candidates.sort(key=lambda r: (0 if r["family"] == 1 else 1, r["resource_id"]))
        if not candidates:
            raise RuntimeError("direct execution has no allowed backend")
        start = time.monotonic_ns()
        deadline = time.monotonic() + self.wait_timeout_s
        preferred_until = time.monotonic() + min(5.0, self.wait_timeout_s / 2)
        has_accelerator = any(r["family"] == 1 for r in candidates)
        chosen = None
        while chosen is None:
            if self._closed:
                raise RuntimeError("direct execution is closed")
            for r in candidates:
                if has_accelerator and r["family"] != 1 and time.monotonic() < preferred_until:
                    continue
                if self._locks[r["resource_id"]].acquire(blocking=False):
                    chosen = r
                    break
            if chosen is None:
                if time.monotonic() >= deadline:
                    raise TimeoutError("direct execution admission timed out")
                time.sleep(0.02)
        decision = {"task_id": uuid.uuid4().hex, "resource_id": chosen["resource_id"],
                    "reason": "direct-backend-admission", "dispatched": True,
                    "recorded_ns": time.time_ns()}
        self._local.decision = decision
        with self._guard:
            self._history.append(decision)
            del self._history[:-256]
        try:
            yield chosen["resource_id"], (time.monotonic_ns() - start) // 1000
        finally:
            self._locks[chosen["resource_id"]].release()

    def decision_for_current_thread(self):
        return getattr(self._local, "decision", None)

    @property
    def decision_history(self):
        with self._guard:
            return list(self._history)

    def observe_service_time(self, *args, **kwargs):
        return None

    def close(self):
        self._closed = True
