#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Action queue management for Real-Time Chunking (RTC).

This module provides ActionQueue, a thread-safe queue for managing action chunks
in real-time control scenarios. It supports both RTC-enabled and non-RTC modes,
handling action merging and leftover tracking.
"""

import logging
from threading import Lock
import numpy as np
from openpi_client.rtc_config import RTCConfig

logger = logging.getLogger(__name__)


def pad_rtc_prev_actions_hold_last(prev_actions: np.ndarray, target_steps: int) -> np.ndarray:
    """Pad/truncate leftover to execution horizon ``s`` by holding the last action.

    Matches LeRobot HIL ``_normalize_prev_actions_length``: pad before the policy so
    leftover length is already ``s`` and RTC weights on ``[T, s)`` lock the last
    command instead of zeros. Truncation past ``s`` is unused (``W=0``).
    """
    prev = np.asarray(prev_actions, dtype=np.float32)
    if prev.ndim != 2:
        raise ValueError(f"prev_actions must be rank-2 (T, D), got {prev.shape}")
    target_steps = max(1, int(target_steps))
    t = int(prev.shape[0])
    if t == 0:
        raise ValueError("prev_actions must be non-empty to hold-pad")
    if t == target_steps:
        return prev
    if t > target_steps:
        return prev[:target_steps]
    hold = np.repeat(prev[-1:], target_steps - t, axis=0)
    return np.concatenate([prev, hold], axis=0)


class ActionQueue:
    """Thread-safe queue for managing action chunks in real-time control.

    This queue handles two types of action sequences:
    - Original actions: Used for RTC to compute leftovers from previous chunks
    - Processed actions: Post-processed actions ready for robot execution

    The queue operates in two modes:
    1. RTC-enabled: Replaces the entire queue with new actions, accounting for inference delay
    2. RTC-disabled: Appends new actions to the queue, maintaining continuity

    Args:
        cfg (RTCConfig): Configuration for Real-Time Chunking behavior.

    Attributes:
        queue (Tensor | None): Processed actions for robot rollout (time_steps, action_dim).
        original_queue (Tensor | None): Original actions for RTC computation (time_steps, action_dim).
        last_index (int): Current consumption index in the queue.
    """

    def __init__(self, enabled: bool):
        """Initialize the action queue.

        Args:
            cfg: RTC configuration controlling queue behavior.
        """
        self.queue = None  # Processed actions for robot rollout
        self.original_queue = None  # Original actions for RTC
        self.lock = Lock()
        self.last_index = 0
        self.enabled = enabled

    def get(self) ->  np.ndarray | None:
        """Get the next action from the queue.

        Returns:
            Tensor | None: The next action (action_dim,) or None if queue is empty.
                          Returns a clone to prevent external modifications.
        """
        with self.lock:
            if self.queue is None or self.last_index >= len(self.queue):
                return None

            action = self.queue[self.last_index]
            self.last_index += 1
            return action.copy()  # Return a copy to prevent external modifications


    def clear(self) -> None:
        """Clear queued actions and reset consumption index."""
        with self.lock:
            self.queue = None
            self.original_queue = None
            self.last_index = 0
            
    def qsize(self) -> int:
        """Get the number of remaining actions in the queue.

        Returns:
            int: Number of unconsumed actions.
        """
        if self.queue is None:
            return 0
        length = len(self.queue)
        return length - self.last_index

    def empty(self) -> bool:
        """Check if the queue is empty.

        Returns:
            bool: True if no actions remain, False otherwise.
        """
        if self.queue is None:
            return True

        length = len(self.queue)
        return length - self.last_index <= 0

    def get_action_index(self) -> int:
        """Get the current action consumption index.

        Returns:
            int: Index of the next action to be consumed.
        """
        return self.last_index

    def get_left_over(self) -> np.ndarray | None:
        """Get leftover original actions for RTC prev_chunk_left_over.

        These are the unconsumed actions from the current chunk, which will be
        used by RTC to compute corrections for the next chunk.

        Returns:
            np.ndarray | None: Remaining original actions (remaining_steps, action_dim),
                          or None if no original queue exists.
        """
        with self.lock:
            if self.original_queue is None:
                return None
            leftover = self.original_queue[self.last_index :]
            if len(leftover) == 0:
                return None
            return leftover
    
    def get_processed_left_over(self) -> np.ndarray | None:
        """Get leftover processed actions (the actions currently executed by the robot).

        Returns:
            Tensor | None: Remaining processed actions (remaining_steps, action_dim),
                or None if no processed queue exists.
        """
        with self.lock:
            if self.queue is None:
                return None
            return self.queue[self.last_index :]

    def merge(
        self,
        original_actions: np.ndarray,
        processed_actions: np.ndarray,
        real_delay: int,
        action_index_before_inference: int | None = 0,
    ):
        """Merge new actions into the queue.

        This method operates differently based on RTC mode:
        - RTC enabled: Replaces the queue, accounting for inference delay
        - RTC disabled: Appends to the queue, maintaining continuity

        Args:
            original_actions: Unprocessed actions from policy (time_steps, action_dim).
            processed_actions: Post-processed actions for robot (time_steps, action_dim).
            real_delay: Number of time steps of inference delay.
            action_index_before_inference: Index before inference started, for validation.
        """
        with self.lock:
            effective_delay = self._check_delays(real_delay, action_index_before_inference)

            if self.enabled:
                self._replace_actions_queue(original_actions, processed_actions, effective_delay)
                return

            self._append_actions_queue(original_actions, processed_actions)

    def _replace_actions_queue(self, original_actions: np.ndarray, processed_actions: np.ndarray, real_delay: int):
        """Replace the queue with new actions (RTC mode).

        Discards the first `real_delay` actions since they correspond to the time
        spent during inference, when the robot was executing previous actions.

        Args:
            original_actions: Unprocessed actions from policy.
            processed_actions: Post-processed actions for robot.
            real_delay: Number of time steps to skip due to inference delay.
        """
        clamped_delay = max(0, min(real_delay, len(original_actions), len(processed_actions)))
        self.original_queue = original_actions[clamped_delay:].copy()
        self.queue = processed_actions[clamped_delay:].copy()

        logger.debug(f"original_actions shape: {self.original_queue.shape}")
        logger.debug(f"processed_actions shape: {self.queue.shape}")
        logger.debug(f"real_delay: {real_delay}, clamped_delay: {clamped_delay}")

        self.last_index = 0

    def _append_actions_queue(self, original_actions: np.ndarray, processed_actions: np.ndarray):
        """Append new actions to the queue (non-RTC mode).

        Removes already-consumed actions and appends new ones, maintaining
        queue continuity without replacement.

        Args:
            original_actions: Unprocessed actions from policy.
            processed_actions: Post-processed actions for robot.
        """
        if self.queue is None:
            self.original_queue = original_actions.copy()
            self.queue = processed_actions.copy()
            return

        self.original_queue = np.concatenate([self.original_queue, original_actions.copy()])
        self.original_queue = self.original_queue[self.last_index :]

        self.queue = np.concatenate([self.queue, processed_actions.copy()])
        self.queue = self.queue[self.last_index :]

        self.last_index = 0

    def _check_delays(
            self, real_delay: int, action_index_before_inference: int | None = None
        ) -> int:
            """Validate that computed delays match expectations.

            Compares the delay computed from inference latency with the actual
            number of actions consumed during inference.

            Args:
                real_delay: Delay computed from inference latency.
                action_index_before_inference: Action index when inference started.

            Returns:
                int: Delay to use.
            """
            effective_delay = max(0, real_delay)

            if action_index_before_inference is not None:
                indexes_diff = max(0, self.last_index - action_index_before_inference)
                if indexes_diff != real_delay:
                    logger.info(
                        "Indexes diff is not equal to real delay. indexes_diff=%d, real_delay=%d",
                        indexes_diff,
                        real_delay,
                    )
                    # Never discard more than was actually consumed, or the queue splices ahead
                    # of the physical pose.
                    return min(real_delay, indexes_diff)

            return effective_delay

