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
                if self._cur_step == self._s:
                    self._background_running = True
                    # ts = time.monotonic()
                    self._background_results = self._policy.infer(
                        obs = self._obs, 
                        prev_action=self._last_origin_actions, 
                        use_rtc=self._is_rtc)
                    # tf = time.monotonic()
                    # update infer_delay
                    # self._infer_delay = np.ceil((tf - ts) * self.fps)
                    self._background_running = False
                else:
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
                print("_last_results is None!")

            results = tree.map_structure(lambda x: x[self._cur_step, ...], self._last_results)
            self._obs = obs
            self._cur_step += 1

            # if current step equals s+d, wait for background inference to complete
            if self._cur_step == self._s + self._d:
                while self._background_running:
                    time.sleep(0.005)
                self._last_origin_actions = self._background_results["origin_actions"]
                self._last_results = {"actions": self._background_results["actions"]}
                self._cur_step -= self._s
                
                
            now = time.perf_counter()  
            if self._infer_last_time != 0:
                infer_interval = (now - self._infer_last_time)
                print(f"RTC infer-delay: {self._infer_delay}step, infer_interval: {infer_interval}s")
                
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

            # self._last_results = {"actions": self._last_results["actions"]}
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