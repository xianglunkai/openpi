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
        multiplier: int = 1,
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
        self._interpolator = ActionInterpolator(multiplier) if use_action_interpolation else None

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
      
        if self._interpolator:
            control_interval = self._interpolator.get_control_interval(self._max_hz)
        else:
            control_interval = 1 / self._max_hz if self._max_hz > 0 else 0
        
        last_step_time = time.time()

        while self._in_episode:
       
            # Get new action at agent frequency
            action = None
                
            # Update interpolator with new action
            if self._interpolator:
                if self._interpolator.needs_new_action():
                    observation = self._environment.get_observation()
                    action = self._agent.get_action(observation)
                    self._interpolator.add(action["actions"])
                    self._agent_steps += 1
            else:
                observation = self._environment.get_observation()
                action = self._agent.get_action(observation)
                self._agent_steps += 1

           
            # Apply interpolated action at robot frequency
            if self._interpolator:
                interpolated_action = self._interpolator.get()
                if interpolated_action is not None:
                    action_to_apply = {"actions": interpolated_action}
                    self._environment.apply_action(action_to_apply)
                  
            else:
                if action is not None:
                    self._environment.apply_action(action)
            
            # Notify subscribers with agent action
            for subscriber in self._subscribers:
                subscriber.on_step(observation, action)

            # self._episode_steps += 1

            # Sleep to maintain robot control frequency
            now = time.time()
            dt = now - last_step_time
            
            if dt < control_interval:
                time.sleep(control_interval - dt)
                last_step_time = time.time()
            else:
                last_step_time = now
                print(f"inference takes time {dt * 1000}ms > control_interval :{control_interval * 1000}ms")
              
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
    """Interpolates between consecutive actions for smoother control.

    When enabled with multiplier N, produces N actions per policy action
    by linearly interpolating between the previous and current action.

    Example with multiplier=3:
        prev_action -> [1/3 interpolated, 2/3 interpolated, current_action]

    This effectively multiplies the control rate for smoother motion.

    Usage:
        interpolator = ActionInterpolator(multiplier=2)  # 2x control rate

        # In control loop:
        if interpolator.needs_new_action():
            new_action = queue.get()
            if new_action:
                interpolator.add(new_action.cpu())

        action = interpolator.get()
        if action:
            robot.send_action(action)
    """

    def __init__(self, multiplier: int = 1):
        """Initialize the interpolator.

        Args:
            multiplier: Control rate multiplier (1 = no interpolation, 2 = 2x, 3 = 3x, etc.)
        """
        if multiplier < 1:
            raise ValueError(f"multiplier must be >= 1, got {multiplier}")
        self.multiplier = multiplier
        self._prev = None
        self._buffer: list = []
        self._idx = 0

    @property
    def enabled(self) -> bool:
        """Whether interpolation is active (multiplier > 1)."""
        return self.multiplier > 1

    def reset(self):
        """Reset interpolation state (call between episodes)."""
        self._prev = None
        self._buffer = []
        self._idx = 0

    def needs_new_action(self) -> bool:
        """Check if a new action is needed from the queue."""
        return self._idx >= len(self._buffer)

    def add(self, action) -> None:
        """Add a new action and compute interpolated sequence.

        Args:
            action: New action tensor from policy/queue (already on CPU).
        """
        if self.multiplier > 1 and self._prev is not None:
            self._buffer = []
            for i in range(1, self.multiplier + 1):
                t = i / self.multiplier
                interp = self._prev + t * (action - self._prev)
                self._buffer.append(interp)
        else:
            self._buffer = [action]
        self._prev = action
        self._idx = 0

    def get(self):
        """Get the next interpolated action.

        Returns:
            Next action tensor, or None if buffer is exhausted.
        """
        if self._idx >= len(self._buffer):
            return None
        action = self._buffer[self._idx]
        self._idx += 1
        return action

    def get_control_interval(self, fps: float) -> float:
        """Get the control interval based on interpolation multiplier.

        Args:
            fps: Base frames per second.

        Returns:
            Control interval in seconds (divided by multiplier).
        """
        return 1.0 / (fps * self.multiplier)