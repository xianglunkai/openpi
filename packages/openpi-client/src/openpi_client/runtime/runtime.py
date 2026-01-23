import logging
import threading
import time

import numpy as np

from openpi_client.runtime import agent as _agent
from openpi_client.runtime import environment as _environment
from openpi_client.runtime import subscriber as _subscriber


class Runtime:
    """The core module orchestrating interactions between key components of the system."""

    def __init__(
        self,
        environment: _environment.Environment,
        agent: _agent.Agent,
        subscribers: list[_subscriber.Subscriber],
        max_hz: float = 0,
        num_episodes: int = 1,
        max_episode_steps: int = 0,
        use_action_interpolation: bool = False,
        robot_fps: int = 50,
    ) -> None:
        self._environment = environment
        self._agent = agent
        self._subscribers = subscribers
        self._max_hz = max_hz
        self._num_episodes = num_episodes
        self._max_episode_steps = max_episode_steps
        self._use_action_interpolation = use_action_interpolation

        self._in_episode = False
        self._episode_steps = 0
        self._agent_steps = 0
        self._interpolator = ActionInterpolator(robot_fps) if use_action_interpolation else None

    def run(self) -> None:
        """Runs the runtime loop continuously until stop() is called or the environment is done."""
        for _ in range(self._num_episodes):
            self._run_episode()

        # Final reset, this is important for real environments to move the robot to its home position.
        self._environment.reset()

    def run_in_new_thread(self) -> threading.Thread:
        """Runs the runtime loop in a new thread."""
        thread = threading.Thread(target=self.run)
        thread.start()
        return thread

    def mark_episode_complete(self) -> None:
        """Marks the end of an episode."""
        self._in_episode = False

    def _run_episode(self) -> None:
        """Runs a single episode."""
        logging.info("Starting episode...")
        self._environment.reset()
        self._agent.reset()
        if self._interpolator:
            self._interpolator.reset()
        for subscriber in self._subscribers:
            subscriber.on_episode_start()

        self._in_episode = True
        # self._episode_steps = 0
        self._agent_steps = 0  # Track agent inference steps
        agent_step_time = 1 / self._max_hz if self._max_hz > 0 else 0
        robot_step_time = 1 / self._interpolator.robot_fps if self._interpolator else agent_step_time
        last_step_time = time.time()
        last_agent_time = time.time()

        while self._in_episode:
            # Determine if we need to get new action from agent
            current_time = time.time()
            time_since_last_agent = current_time - last_agent_time

            # Get new action at agent frequency
            action = None
            if time_since_last_agent >= agent_step_time:
                observation = self._environment.get_observation()
                action = self._agent.get_action(observation)

                # Update interpolator with new action
                if self._interpolator:
                    self._interpolator.update(action["actions"])

                last_agent_time = current_time
                self._agent_steps += 1

                # Notify subscribers with agent action
                for subscriber in self._subscribers:
                    subscriber.on_step(observation, action)

            # Apply interpolated action at robot frequency
            if self._interpolator:
                interpolated_action, _ = self._interpolator.get_interpolated_action()
                if interpolated_action is not None:
                    action_to_apply = {"actions": interpolated_action}
                    self._environment.apply_action(action_to_apply)
            else:
                if action is not None:
                    self._environment.apply_action(action)

            # self._episode_steps += 1

            # Sleep to maintain robot control frequency
            now = time.time()
            dt = now - last_step_time
            if dt < robot_step_time:
                time.sleep(robot_step_time - dt)
                last_step_time = time.time()
            else:
                last_step_time = now

            # Check episode completion
            if self._environment.is_episode_complete() or (
                self._max_episode_steps > 0 and self._agent_steps >= self._max_episode_steps
            ):
                self.mark_episode_complete()

        logging.info("Episode completed.")
        for subscriber in self._subscribers:
            subscriber.on_episode_end()

    def _step(self) -> None:
        """A single step of the runtime loop. Deprecated: logic moved to _run_episode."""
        raise NotImplementedError("_step is deprecated, see _run_episode")


class ActionInterpolator:
    """Interpolate between RTC actions for smoother robot control with velocity estimation."""
    
    def __init__(self, robot_fps: int):
        self.robot_fps = robot_fps
        self.prev_action = None
        self.curr_action = None
        self.prev_time: float = 0
        self.curr_time: float = 0
        self.last_interpolated = None
        
    def update(self, new_action) -> None:
        self.prev_action = self.curr_action
        self.prev_time = self.curr_time
        self.curr_action = new_action
        self.curr_time = time.perf_counter()
        
    def get_interpolated_action(self) -> tuple:
        """Returns (interpolated_position, estimated_velocity)"""
        if self.curr_action is None:
            return None, None
        if self.prev_action is None:
            self.last_interpolated = self.curr_action.copy()
            return self.curr_action, np.zeros_like(self.curr_action)

        # Time-based interpolation
        current_time = time.perf_counter()
        dt_actions = self.curr_time - self.prev_time
        if dt_actions <= 0:
            dt_actions = 1.0 / self.robot_fps  # Fallback

        t = (current_time - self.prev_time) / dt_actions
        t = max(0.0, min(t, 1.25))  # Allow slight extrapolation

        interpolated = self.prev_action + t * (self.curr_action - self.prev_action)

        # Estimate velocity
        dt_robot = 1.0 / self.robot_fps
        if self.last_interpolated is not None:
            velocity = (interpolated - self.last_interpolated) / dt_robot
        else:
            velocity = (self.curr_action - self.prev_action) / dt_actions

        self.last_interpolated = interpolated.copy()
        return interpolated, velocity
    
    def reset(self):
        self.prev_action = None
        self.curr_action = None
        self.prev_time = 0
        self.curr_time = 0
        self.last_interpolated = None