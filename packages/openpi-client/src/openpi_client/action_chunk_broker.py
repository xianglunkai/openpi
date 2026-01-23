from typing import Dict

import threading
import time
import numpy as np
import tree
from typing_extensions import override

from openpi_client import base_policy as _base_policy


class ActionChunkBroker(_base_policy.BasePolicy):
    """Wraps a policy to return action chunks one-at-a-time with optional RTC (Real-Time Compute).

    Args:
        policy: The inner policy to wrap
        action_horizon: Number of steps in each action chunk
        is_rtc: Enable real-time compute with background inference
        s: Steps before triggering background inference (default: 25)
        d: Tolerance steps for background inference (default: 10)
        fps: Frame rate for timing calculations (default: 50)
    """

    def __init__(self, policy: _base_policy.BasePolicy, action_horizon: int,
                 is_rtc: bool = False, s: int = 25, d: int = 10, fps: int = 50):
        self._policy = policy
        self._action_horizon = action_horizon
        self._cur_step = 0
        self._last_results: Dict[str, np.ndarray] | None = None
        self._last_origin_actions: np.ndarray | None = None
        self._obs: Dict[str, np.ndarray] | None = None

        # RTC parameters
        self._is_rtc = is_rtc
        self._s = s  # steps before triggering background inference
        self._d = d  # tolerance steps
        self.fps = fps
        self._deadline = d / fps  # max inference time in seconds

        # Thread state
        self._background_results: Dict[str, np.ndarray] | None = None
        self._background_running = False

        if self._is_rtc:
            self._lock = threading.Lock()
            self._stop_event = threading.Event()
            self._infer_thread = threading.Thread(
                target=self._background_infer, daemon=True, name="BackgroundInferThread"
            )
            self._infer_thread.start()

            # Simple warmup with dummy data in background
            threading.Thread(target=self._warmup_inference, daemon=True, name="WarmupThread").start()

    def _background_infer(self):
        """Background thread that pre-computes next action chunks."""
        while not self._stop_event.is_set():
            try:
                should_run = False
                with self._lock:
                    if (not self._background_running and
                        self._background_results is None and
                        self._cur_step >= self._s):
                        self._background_running = True
                        should_run = True
                        obs_snapshot = self._obs
                        prev_snapshot = self._last_origin_actions

                if should_run:
                    ts = time.monotonic()
                    bg_res = self._policy.infer(obs=obs_snapshot, prev_action=prev_snapshot, use_rtc=True)
                    infer_time = time.monotonic() - ts

                    with self._lock:
                        # Discard stale results that exceed deadline
                        if infer_time > self._deadline:
                            print(f"Warning: Inference took {infer_time:.3f}s > deadline {self._deadline:.3f}s")
                        else:
                            self._background_results = bg_res
                        self._background_running = False
                else:
                    self._stop_event.wait(0.01)
            except Exception:
                if not self._stop_event.is_set():
                    import traceback
                    traceback.print_exc()
                with self._lock:
                    self._background_running = False

    def _warmup_inference(self):
        """Perform warmup inference to avoid cold-start latency on first call."""
        try:
            dummy_example = self._policy.make_example()
            if dummy_example is not None:
                print("ActionChunkBroker: Performing warmup inference...")
                ts = time.monotonic()
                r = self._policy.infer(obs=dummy_example, prev_action=None, use_rtc=True)
                actions = r["origin_actions"]
                _ = self._policy.infer(obs=dummy_example, prev_action=actions, use_rtc=True)
                warmup_time = time.monotonic() - ts
                print(f"ActionChunkBroker: Warmup completed in {warmup_time:.3f}s")
        except Exception as e:
            print(f"ActionChunkBroker: Warmup failed - {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """Cleanly shutdown background thread."""
        if not self._is_rtc:
            return

        self._stop_event.set()
        if self._infer_thread.is_alive():
            self._infer_thread.join(timeout=1.0)
            if self._infer_thread.is_alive():
                print("Warning: Background thread did not exit gracefully")

    @override
    def infer(self, obs: Dict) -> Dict:
        """Get action for current observation, using RTC if enabled."""
        if self._is_rtc:
            return self._infer_rtc(obs)
        else:
            return self._infer_normal(obs)

    def _infer_rtc(self, obs: Dict) -> Dict:
        """RTC mode: Use background inference to reduce latency."""
        # First call: perform inference (warmup should have reduced latency)
        if self._last_results is None:
            self._last_results = self._policy.infer(obs=obs, prev_action=None, use_rtc=self._is_rtc)
            self._last_origin_actions = self._last_results["origin_actions"]
            self._last_results = {"actions": self._last_results["actions"]}
            self._cur_step = 0

        # Get current action chunk
        results = self._slice_results(self._last_results, self._cur_step)

        # Update state under lock
        with self._lock:
            self._obs = obs
            self._cur_step += 1
            
            # Swap in background results when ready
            ready = (self._background_results is not None and
                     self._cur_step >= self._s + self._d and
                     not self._background_running)
            if ready:
                self._last_origin_actions = self._background_results["origin_actions"]
                self._last_results = {"actions": self._background_results["actions"]}
                # Roll back step count
                self._cur_step = max(0, self._cur_step - self._s)
                self._background_results = None

        return results

    def _infer_normal(self, obs: Dict) -> Dict:
        """Normal mode: Simple action chunking without RTC."""
        if self._last_results is None:
            self._last_results = self._policy.infer(obs=obs, prev_action=None, use_rtc=False)
            self._cur_step = 0

        results = self._slice_results(self._last_results, self._cur_step)
        self._cur_step += 1

        if self._cur_step >= self._action_horizon:
            self._last_results = None

        return results

    def _slice_results(self, results: Dict, step: int) -> Dict:
        """Slice action at given step, with fallback for index errors."""
        try:
            return tree.map_structure(lambda x: x[step, ...], results)
        except Exception:
            return tree.map_structure(
                lambda x: x[-1, ...] if isinstance(x, np.ndarray) else x,
                results
            )


    @override
    def reset(self) -> None:
        """Reset broker and policy state."""
        self._policy.reset()
        self._last_results = None
        self._last_origin_actions = None
        self._background_results = None
        self._cur_step = 0
        print()

    @override
    def make_example(self) -> Dict:
        return None
