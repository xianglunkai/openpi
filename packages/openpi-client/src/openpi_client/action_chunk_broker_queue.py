from typing import Dict, Optional
import threading
import time
import logging

import numpy as np
import torch
from typing_extensions import override

from openpi_client import base_policy as _base_policy
from lerobot.policies.rtc.action_queue import ActionQueue
from lerobot.policies.rtc.configuration_rtc import RTCConfig

logger = logging.getLogger(__name__)


class ActionChunkBrokerQueue(_base_policy.BasePolicy):
    """Action chunk broker backed by an ActionQueue (event-driven).

    Behavior:
    - Maintains an ActionQueue instance to store processed and original actions.
    - Runs a background thread which requests new chunks from the wrapped policy
      whenever the queue size falls below a threshold.
    - The main thread `infer()` returns the next action (numpy array) popped from the queue.

    This implementation follows the `get_actions` pattern used by the demo: it
    snapshots the queue index and leftover, times the inference call to compute
    the real delay in steps, then merges the result into the ActionQueue.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        action_horizon: int,
        is_rtc: bool = False,
        queue_threshold: Optional[int] = None,
        fps: int = 50,
    ) -> None:
        super().__init__()
        self._policy = policy
        self._action_horizon = action_horizon
        self._is_rtc = is_rtc
        self._fps = fps

        # Create ActionQueue with a minimal RTCConfig
        rtc_cfg = RTCConfig(execution_horizon=action_horizon, enabled=is_rtc)
        self._action_queue = ActionQueue(rtc_cfg)

        # threshold for requesting new chunks (default: half horizon or provided)
        if queue_threshold is None:
            self._queue_threshold = max(1, action_horizon // 2)
        else:
            self._queue_threshold = queue_threshold

        # background thread control
        self._stop_event = threading.Event()
        self._bg_thread = threading.Thread(target=self._bg_worker, daemon=True, name="ACBQueue-Requester")
        self._bg_thread_lock = threading.Lock()
        self._bg_thread.start()

    def close(self) -> None:
        self._stop_event.set()
        if self._bg_thread.is_alive():
            self._bg_thread.join(timeout=1.0)

    def reset(self) -> None:
        # reset internal queue and policy
        with self._bg_thread_lock:
            self._action_queue = ActionQueue(RTCConfig(execution_horizon=self._action_horizon, enabled=self._is_rtc))
        try:
            self._policy.reset()
        except Exception:
            pass

    def _bg_worker(self) -> None:
        """Background requester: when queue low, request a new chunk via policy.infer()."""
        time_per_chunk = 1.0 / float(self._fps)
        while not self._stop_event.is_set():
            try:
                qsize = self._action_queue.qsize()
                if (qsize <= self._queue_threshold) and (not self._stop_event.is_set()):
                    # snapshot index and leftover
                    action_index_before_inference = self._action_queue.get_action_index()
                    prev_leftover = self._action_queue.get_left_over()

                    # if prev_leftover is torch.Tensor, keep as is; policy.infer expects numpy-style obs/prev_action per repo
                    # We just pass prev_leftover through; policies should accept None or tensor-like.

                    ts = time.perf_counter()
                    try:
                        # call policy.infer; expect dict with keys "origin_actions" and "actions"
                        res = self._policy.infer(obs=self._snapshot_obs(), prev_action=prev_leftover, use_rtc=self._is_rtc)
                    except TypeError:
                        # Some policies expect a slightly different signature
                        res = self._policy.infer(obs=None, prev_action=prev_leftover, use_rtc=self._is_rtc)
                    tf = time.perf_counter()

                    if res is None:
                        # nothing returned; sleep a bit and continue
                        self._stop_event.wait(0.01)
                        continue

                    # origin_actions and actions should be arrays/tensors with leading time dim
                    origin_actions = res.get("origin_actions")
                    processed_actions = res.get("actions")

                    # compute inferred delay in steps
                    new_latency = tf - ts
                    new_delay = int(np.ceil(new_latency / time_per_chunk))

                    # merge into action queue
                    try:
                        # convert numpy to torch if necessary
                        if isinstance(origin_actions, np.ndarray):
                            origin_actions_t = torch.from_numpy(origin_actions)
                        else:
                            origin_actions_t = origin_actions

                        if isinstance(processed_actions, np.ndarray):
                            processed_actions_t = torch.from_numpy(processed_actions)
                        else:
                            processed_actions_t = processed_actions

                        with self._bg_thread_lock:
                            self._action_queue.merge(
                                original_actions=origin_actions_t,
                                processed_actions=processed_actions_t,
                                real_delay=new_delay,
                                action_index_before_inference=action_index_before_inference,
                            )
                    except Exception as e:
                        logger.exception("Failed to merge actions into ActionQueue: %s", e)
                else:
                    # sleep a bit to avoid busy loop
                    self._stop_event.wait(0.01)
            except Exception:
                logger.exception("Background actor encountered exception")
                self._stop_event.wait(0.1)

    def _snapshot_obs(self):
        """Placeholder to snapshot observation for policy.infer.

        The original ActionChunkBroker expects the caller to pass observations
        into `infer(obs)`. In this queue-backed broker we don't have a global
        stored observation; policies that need an observation should be called
        directly via `policy.infer` by external code or the broker can be
        extended to receive and store the latest obs via a `feed_observation`
        method. For now return None and rely on policies that don't strictly
        require obs or accept None.
        """
        return None

    @override
    def infer(self, obs: Dict) -> Dict:
        """Return the next action (single step) as numpy arrays in a dict.

        If the queue is empty, attempt a synchronous policy.infer call to
        replenish the queue immediately and return the first action.
        """
        # If obs is provided, let background worker know: policies often need the latest obs.
        # For now we call policy.infer synchronously if queue empty.
        action = self._action_queue.get()
        if action is None:
            # request synchronously
            try:
                res = self._policy.infer(obs=obs, prev_action=None, use_rtc=self._is_rtc)
            except TypeError:
                res = self._policy.infer(obs=None, prev_action=None, use_rtc=self._is_rtc)

            if res is None:
                return {"actions": None}

            origin_actions = res.get("origin_actions")
            processed_actions = res.get("actions")

            # convert to torch and merge
            if isinstance(origin_actions, np.ndarray):
                origin_actions_t = torch.from_numpy(origin_actions)
            else:
                origin_actions_t = origin_actions

            if isinstance(processed_actions, np.ndarray):
                processed_actions_t = torch.from_numpy(processed_actions)
            else:
                processed_actions_t = processed_actions

            # merge with zero delay (we just computed it)
            try:
                self._action_queue.merge(original_actions=origin_actions_t, processed_actions=processed_actions_t, real_delay=0, action_index_before_inference=0)
            except Exception:
                logger.exception("Failed to merge synchronously returned actions")

            action = self._action_queue.get()

        if action is None:
            return {"actions": None}

        # return numpy arrays for compatibility with older callers
        if isinstance(action, torch.Tensor):
            arr = action.cpu().numpy()
        elif isinstance(action, np.ndarray):
            arr = action
        else:
            # unknown type, try to coerce
            try:
                arr = np.array(action)
            except Exception:
                arr = None

        return {"actions": arr}
