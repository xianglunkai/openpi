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
    """

    def __init__(self, policy: _base_policy.BasePolicy, action_horizon: int,
                 is_rtc: bool = False, s: int = 25, d: int = 10):
        self._policy = policy
        self._action_horizon = action_horizon
        self._cur_step = 0
        self._last_results: Dict[str, np.ndarray] | None = None
        self._last_origin_actions: np.ndarray | None = None
        self._obs: Dict[str, np.ndarray] | None = None

        # RTC parameters
        self._is_rtc = is_rtc
        self._smin = s # minimal inference delay time steps
        self._s = s  #  real time number of actions executed since last inference started
        self._d = d  #  real time number of steps for inference
        self._deadline = d  # max inference time steps
     
        # warmup
        self._warmup = False

        if self._is_rtc:
            self._lock = threading.Lock()
            self._stop_event = threading.Event()
            self._infer_thread = threading.Thread(
                target=self._background_infer, daemon=True, name="BackgroundInferThread"
            )
            self._infer_thread.start()

    def _background_infer(self):
        """Background thread that pre-computes next action chunks."""
        while not self._stop_event.is_set():
            try:
                should_run = False
                with self._lock:
                    if (self._cur_step >= self._smin):
                        should_run = True
                        # s is the number of actions executed since last inference started
                        self._s = self._cur_step
                        # Store the current observation
                        obs_snapshot = self._obs.copy()
                        # Remove the actions that have already been executed
                        prev_snapshot = self._last_origin_actions.copy()
                        print(f"1. self._cur_step: {self._cur_step}, self._s: {self._s }, self._d: {self._d }")
           
                if should_run:
                    new_actions = self._policy.infer(obs=obs_snapshot, prev_action=prev_snapshot, use_rtc=True)
            
                    with self._lock:
                        # Swap to the new chunk as soon as it is available
                        self._last_origin_actions = new_actions["origin_actions"].copy()
                        self._last_results = new_actions.copy()
                        # Reset t so that it indexes into A new
                        self._cur_step = max(0, self._cur_step - self._s)
                        self._d = self._cur_step
                        print(f"2. self._cur_step: {self._cur_step}, self._s: {self._s }, self._d: {self._d }")

                    
                     # Discard stale results that exceed deadline
                    if self._d > self._deadline:
                        print(f"Warning: Inference took {self._d} steps > deadline {self._deadline} steps")
                else:
                    time.sleep(0.01)        
           
            except Exception:
                if not self._stop_event.is_set():
                    import traceback
                    traceback.print_exc()


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
        if not self._warmup:
            init_actions = self._policy.infer(obs=obs, prev_action=None, use_rtc=self._is_rtc)
            self._last_results = self._policy.infer(obs=obs, prev_action= init_actions["origin_actions"], use_rtc=self._is_rtc)
            self._last_origin_actions = self._last_results["origin_actions"]
            # self._last_results = {"actions": self._last_results["actions"]}
            self._cur_step = 0
            self._warmup = True


       
        
        # Update state under lock
        with self._lock:
            # Get current action chunk
            def slicer(x):
                if isinstance(x, np.ndarray):
                    return x[self._cur_step, ...]
                else:
                    return x
        
            results = tree.map_structure(slicer, self._last_results)
            self._obs = obs
            self._cur_step += 1
        return results

    def _infer_normal(self, obs: Dict) -> Dict:
        """Normal mode: Simple action chunking without RTC."""
        if self._last_results is None:
            self._last_results = self._policy.infer(obs=obs, prev_action=None, use_rtc=False)
            self._cur_step = 0
         
        def slicer(x):
            if isinstance(x, np.ndarray):
                return x[self._cur_step, ...]
            else:
                return x

        results = tree.map_structure(slicer, self._last_results)
        self._cur_step += 1

        if self._cur_step >= self._action_horizon:
            self._last_results = None

        return results


    @override
    def reset(self) -> None:
        """Reset broker and policy state."""
        self._policy.reset()
        self._last_results = None
        self._last_origin_actions = None
        self._background_results = None
        self._cur_step = 0

    @override
    def make_example(self) -> Dict:
        return None
