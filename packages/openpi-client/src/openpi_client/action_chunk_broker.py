from typing import Dict

import threading
import time
import numpy as np
import tree
from typing_extensions import override

from openpi_client import base_policy as _base_policy


class ActionChunkBroker(_base_policy.BasePolicy):
    """Wraps a policy to return action chunks one-at-a-time.

    Assumes that the first dimension of all action fields is the chunk size.

    A new inference call to the inner policy is only made when the current
    list of chunks is exhausted.
    """

    def __init__(self, policy: _base_policy.BasePolicy, action_horizon: int, is_rtc: bool = False, s: int = 25, d: int = 10, fps: int = 50,):
        self._policy = policy
        self._action_horizon = action_horizon
        self._cur_step: int = 0

        self._last_results: Dict[str, np.ndarray] | None = None
        self._last_origin_actions: np.ndarray | None = None
        self._background_results: Dict[str, np.ndarray] | None = None
        self._background_running: bool = False
       
        
        self._obs: Dict[str, np.ndarray] | None = None
        self._s = s  # 25
        self._d = d  # 10
        self.fps = fps
        
        self._infer_delay:int  = d
        self._is_rtc = is_rtc
        # lock to protect shared state between main thread and background thread
        self._lock = threading.Lock()

        if self._is_rtc:
            self._stop_event = threading.Event()
            self._infer_thread = None
            self._start_background_thread()
        
        self._infer_last_time = 0
            
    def _start_background_thread(self):
        if self._infer_thread is None or not self._infer_thread.is_alive():
            self._stop_event.clear()
            self._infer_thread = threading.Thread(
                target=self._background_infer,
                daemon=True,
                name="BackgroundInferThread"
            )
            self._infer_thread.start()
    
    def _background_infer(self):
        while not self._stop_event.is_set():
            try:
                # Trigger if we've reached or passed the s threshold and a background
                # result is not already available. Use >= to avoid missing the exact
                # equality due to thread scheduling. Acquire a lock to snapshot
                # the inputs and set the running flag, but do the heavy infer
                # call outside the lock.
                should_run = False
                with self._lock:
                    if (not self._background_running) and (self._background_results is None) and (self._cur_step >= self._s):
                        # mark as running and snapshot inputs
                        self._background_running = True
                        should_run = True
                        obs_snapshot = self._obs
                        prev_snapshot = self._last_origin_actions

                if should_run:
                    try:
                        # perform infer outside the lock (heavy operation)
                        # ts = time.monotonic()
                        bg_res = self._policy.infer(obs=obs_snapshot, prev_action=prev_snapshot, use_rtc=self._is_rtc)
                        # tf = time.monotonic()
                        # self._infer_delay = np.ceil((tf - ts) * self.fps)
                    except Exception:
                        # ensure we clear running flag on exception
                        with self._lock:
                            self._background_running = False
                        raise
                    with self._lock:
                        self._background_results = bg_res
                        self._background_running = False
                else:
                    # sleep a short while to avoid busy loop
                    self._stop_event.wait(0.005)
            except Exception as e:
                 if not self._stop_event.is_set():
                    import traceback
                    traceback.print_exc()
    
    def __enter__(self):
        """enter context"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """exit context"""
        self.close()
    
    def close(self):
        """close safely"""
        self._stop_event.set()
        if self._infer_thread and self._infer_thread.is_alive():
            self._infer_thread.join(timeout=1.0)
            if self._infer_thread.is_alive():
                print("Warning: Background thread did not exit gracefully")      


    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        if self._is_rtc:
            # first call infer for cot start
            if self._last_results is None:
                ts = time.monotonic()
                self._last_results = self._policy.infer(obs=obs, prev_action=None, use_rtc=True)
                tf = time.monotonic()
                self._last_origin_actions = self._last_results["origin_actions"]
                self._last_results = {"actions": self._last_results["actions"]}
                self._infer_delay = np.ceil((tf - ts) * self.fps)
                self._cur_step = 0

            # slice current step results (guard against index errors)
            try:
                results = tree.map_structure(lambda x: x[self._cur_step, ...], self._last_results)
            except Exception:
                # if out-of-bounds for some reason, fallback to last available step
                def safe_take(x):
                    if isinstance(x, np.ndarray):
                        return x[-1, ...]
                    return x

                results = tree.map_structure(safe_take, self._last_results)

            # update observation and advance step (do this under lock to keep background thread consistent)
            with self._lock:
                self._obs = obs
                self._cur_step += 1

            # if we've reached or passed s+d, swap in background results if ready
            do_swap = False
            with self._lock:
                if (self._background_results is not None) and (self._cur_step >= self._s + self._d) and (not self._background_running):
                    do_swap = True
                    bg_res = self._background_results
                    # consume background result so it won't be reused
                    self._background_results = None

            if do_swap:
                # apply swap (we already snapped bg_res under lock)
                try:
                    self._last_origin_actions = bg_res["origin_actions"]
                    self._last_results = {"actions": bg_res["actions"]}
                except Exception:
                    # malformed background result; ignore swap
                    pass
                # roll back cur_step by s (do under lock)
                with self._lock:
                    self._cur_step = max(0, self._cur_step - self._s)
                
                
            now = time.perf_counter()  
            if self._infer_last_time != 0:
                infer_interval = (now - self._infer_last_time)
                
            self._infer_last_time = now
            return results

        else:
            if self._last_results is None:
                ts = time.monotonic()
                self._last_results = self._policy.infer(obs=obs, prev_action=None, use_rtc=False)
                tf = time.monotonic()
                self._infer_delay = np.ceil((tf - ts) * self.fps)
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
        
            now = time.perf_counter()  
            if self._infer_last_time != 0:
                infer_interval = (now - self._infer_last_time)
                print(f"No RTC infer-delay: {self._infer_delay}step, infer_interval: {infer_interval}s")
                
            self._infer_last_time = now
    
            return results

    @override
    def reset(self) -> None:
        self._policy.reset()
        self._last_results = None
        self._last_origin_actions = None
        self._background_results = None
        self._cur_step = 0
        print()

    @override
    def make_example(self) -> Dict:
        return None