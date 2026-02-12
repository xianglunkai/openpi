import logging
import threading
import time

import numpy as np

from openpi_client.runtime import agent as _agent
from openpi_client.runtime import environment as _environment
from openpi_client.runtime import subscriber as _subscriber
from openpi_client.action_interpolator import ActionInterpolator


class Runtime:
    """The core module orchestrating interactions between key components of the system."""

    def __init__(
        self,
        environment: _environment.Environment,
        agent: _agent.Agent,
        subscribers: list[_subscriber.Subscriber],
        max_hz: float = 0,
        num_episodes: int = 1,
        max_episode_time_s: int = 0,
        use_action_interpolation: bool = False,
        multiplier: int = 1,
    ) -> None:
        self._environment = environment
        self._agent = agent
        self._subscribers = subscribers
        self._max_hz = max_hz
        self._num_episodes = num_episodes
        self._max_episode_time_s = max_episode_time_s
        self._use_action_interpolation = use_action_interpolation

        self._in_episode = False
        self._episode_steps = 0
 
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
        self._episode_steps = 0
    
        if self._interpolator:
            control_interval = self._interpolator.get_control_interval(self._max_hz)
        else:
            control_interval = 1 / self._max_hz if self._max_hz > 0 else 0
        
        last_step_time = time.time()
        eposide_start_time = time.time()
        while self._in_episode:
       
            # Get new action at agent frequency
            action = None
                
            # Update interpolator with new action
            if self._interpolator:
                if self._interpolator.needs_new_action():
                    observation = self._environment.get_observation()
                    action = self._agent.get_action(observation)
                    if action is not None:
                        self._interpolator.add(action["actions"])
            else:
                observation = self._environment.get_observation()
                action = self._agent.get_action(observation)
              
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

            # Sleep to maintain robot control frequency
            now = time.time()
            dt = now - last_step_time
            
            print(f"runtime: dt= {dt * 1000}ms")
        
            if dt < control_interval:
                time.sleep(control_interval - dt)
                last_step_time = time.time()
            else:
                last_step_time = now

            # Check episode completion
            loop_time_eplased = time.time() - eposide_start_time
            if self._environment.is_episode_complete() or (loop_time_eplased >= self._max_episode_time_s
            ):
                self.mark_episode_complete()

        logging.info("Episode completed.")
        for subscriber in self._subscribers:
            subscriber.on_episode_end()

    def _step(self) -> None:
        """A single step of the runtime loop. Deprecated: logic moved to _run_episode."""
        raise NotImplementedError("_step is deprecated, see _run_episode")