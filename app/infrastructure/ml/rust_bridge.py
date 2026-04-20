from __future__ import annotations

import json
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class RustBridgeResult:
    ok: bool
    data: Any = None
    error: str | None = None


class RustBridge:
    """
    Bridge to the Rust core binary.

    Uses a **persistent subprocess** (JSONL over stdin/stdout) so that process-spawn
    overhead is paid only once per RustBridge lifetime rather than once per call.
    The subprocess is started lazily on the first call and restarted automatically
    if it exits unexpectedly.

    Thread-safe: a single threading.Lock serialises all stdin/stdout exchanges.
    """

    def __init__(self, binary_path: Path | None = None) -> None:
        self.binary_path = binary_path or self._default_binary_path()
        self._process: subprocess.Popen | None = None
        self._lock = threading.Lock()
        # None = not yet checked; True/False = cached result
        self._available: bool | None = None

    @staticmethod
    def _default_binary_path() -> Path:
        return (
            Path(__file__).resolve().parents[4]
            / "rust"
            / "target"
            / "release"
            / "huali_garbage_core.exe"
        )

    def available(self) -> bool:
        """Return True if the Rust binary exists on disk (result cached after first check)."""
        if self._available is None:
            self._available = self.binary_path.exists()
        return self._available

    # ------------------------------------------------------------------
    # Persistent subprocess management
    # ------------------------------------------------------------------

    def _ensure_process(self) -> bool:
        """
        Ensure the persistent Rust subprocess is running.
        Must be called with self._lock held.
        Returns True when a live process is available.
        """
        if self._process is not None and self._process.poll() is None:
            return True  # already running
        if not self.binary_path.exists():
            self._available = False
            return False
        try:
            self._process = subprocess.Popen(
                [str(self.binary_path)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                bufsize=1,  # line-buffered (text mode)
            )
            return True
        except Exception:
            self._process = None
            return False

    def call(self, payload: dict) -> RustBridgeResult:
        """Send one JSON request and receive one JSON response via the persistent process."""
        with self._lock:
            if not self._ensure_process():
                return RustBridgeResult(
                    ok=False, error=f"Rust binary not found: {self.binary_path}"
                )
            try:
                line = json.dumps(payload, ensure_ascii=False) + "\n"
                self._process.stdin.write(line)
                self._process.stdin.flush()
                response_line = self._process.stdout.readline()
            except Exception as exc:
                self._process = None
                return RustBridgeResult(ok=False, error=str(exc))

        if not response_line:
            with self._lock:
                self._process = None
            return RustBridgeResult(ok=False, error="Rust process closed unexpectedly")

        try:
            parsed = json.loads(response_line)
        except Exception as exc:
            return RustBridgeResult(ok=False, error=f"Invalid Rust output: {exc}")

        if parsed.get("status") == "err":
            return RustBridgeResult(ok=False, error=parsed.get("message", "unknown rust error"))

        return RustBridgeResult(ok=True, data=parsed)

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    def health_check(self) -> dict:
        """
        Verify the Rust binary exists and produces a correct result.

        Runs a known-answer IoU test:
          Box A [0,0,100,100] ∩ Box B [50,50,150,150]
          intersection = 50×50 = 2500
          union = 10000 + 10000 − 2500 = 17500
          expected IoU ≈ 0.142857

        Returns a dict with keys:
          available (bool) — binary exists on disk
          healthy   (bool) — binary ran and returned the correct answer
          error     (str|None) — human-readable error if not healthy
          latency_ms (float|None) — round-trip time for the test call
        """
        if not self.available():
            return {
                "available": False,
                "healthy": False,
                "error": f"Rust 二进制不存在: {self.binary_path}",
                "latency_ms": None,
            }

        payload = {
            "action": "compute_iou",
            "a": {"x1": 0, "y1": 0, "x2": 100, "y2": 100},
            "b": {"x1": 50, "y1": 50, "x2": 150, "y2": 150},
        }
        t0 = time.perf_counter()
        result = self.call(payload)
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        if not result.ok:
            return {
                "available": True,
                "healthy": False,
                "error": result.error,
                "latency_ms": latency_ms,
            }

        expected = 2500.0 / 17500.0
        actual = float(result.data.get("value", -1.0)) if result.data else -1.0
        if abs(actual - expected) > 0.001:
            return {
                "available": True,
                "healthy": False,
                "error": f"计算结果不匹配：期望 {expected:.6f}，实际 {actual:.6f}",
                "latency_ms": latency_ms,
            }

        return {
            "available": True,
            "healthy": True,
            "error": None,
            "latency_ms": latency_ms,
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Terminate the persistent subprocess if it is running."""
        with self._lock:
            if self._process is not None:
                try:
                    self._process.stdin.close()
                    self._process.wait(timeout=2)
                except Exception:
                    try:
                        self._process.kill()
                    except Exception:
                        pass
                finally:
                    self._process = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Typed high-level helpers
    # ------------------------------------------------------------------

    def filter_boxes(self, boxes: list[list[int]], threshold: float) -> list[list[int]] | None:
        """Call Rust filter_overlapping_boxes. Returns filtered boxes or None on failure."""
        payload = {
            "action": "filter_overlapping_boxes",
            "boxes": [{"x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3]} for b in boxes],
            "threshold": threshold,
        }
        result = self.call(payload)
        if not result.ok or not result.data:
            return None
        return [[b["x1"], b["y1"], b["x2"], b["y2"]] for b in result.data.get("boxes", [])]

    def dedupe_events(
        self,
        events: list[dict],
        cooldown_ms: int,
        iou_threshold: float,
    ) -> list[dict] | None:
        """
        Call Rust dedupe_track_events.

        Each event dict must have keys: class_id (int), bbox ([x1,y1,x2,y2]), timestamp_ms (int).
        Returns deduplicated events with the same shape, or None on failure.
        """
        payload = {
            "action": "dedupe_track_events",
            "events": [
                {
                    "class_id": e["class_id"],
                    "bbox": {
                        "x1": e["bbox"][0],
                        "y1": e["bbox"][1],
                        "x2": e["bbox"][2],
                        "y2": e["bbox"][3],
                    },
                    "timestamp_ms": e["timestamp_ms"],
                }
                for e in events
            ],
            "cooldown_ms": cooldown_ms,
            "iou_threshold": iou_threshold,
        }
        result = self.call(payload)
        if not result.ok or not result.data:
            return None
        out = []
        for e in result.data.get("events", []):
            b = e["bbox"]
            out.append({
                "class_id": e["class_id"],
                "bbox": [b["x1"], b["y1"], b["x2"], b["y2"]],
                "timestamp_ms": e["timestamp_ms"],
            })
        return out
