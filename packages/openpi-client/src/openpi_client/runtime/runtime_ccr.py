import logging
import threading
import time
import math
from typing import Dict, Optional, List
import numpy as np

from openpi_client.runtime import environment as _environment
from openpi_client.runtime import subscriber as _subscriber
from openpi_client import base_policy as _base_policy
from openpi_client.rtc_config import RTCConfig, RTCAttentionSchedule
from openpi_client.action_queue import ActionQueue
from openpi_client.latency_tracker import LatencyTracker
from openpi_client.action_interpolator import ActionInterpolator
from openpi_client.time_utils import precise_sleep
from openpi_client.bspline import optimize_actions_with_ccr
from openpi_client.process import ProcessSignalHandler

class RuntimeCCR:
    """Runtime with integrated RTC functionality using dual-thread architecture.
    
    Key features:
    1. Dual-thread design: get_action_worker and actor_control_worker
    2. Unified infer interface: policy.infer(obs, prev_action, use_rtc)
    3. Uses provided RTC infrastructure: RTCConfig, ActionQueue, LatencyTracker
    4. No background inference thread - all inference is done in get_action_worker
    """
    
    def __init__(
        self,
        environment: _environment.Environment,
        policy: _base_policy.BasePolicy,
        subscribers: List[_subscriber.Subscriber],
        fps: float = 50.0,
        num_episodes: int = 1,
        max_episode_time_s: int = 0,
        use_action_interpolation: bool = False,
        multiplier: int = 1,
        action_queue_size_to_get_new_actions: int = 30,
    ) -> None:
        # Basic parameters
        self._environment = environment
        self._policy = policy
        self._subscribers = subscribers
        self._fps = fps
        self._num_episodes = num_episodes
        self._max_episode_time_s = max_episode_time_s
        self._use_action_interpolation = use_action_interpolation
        self._interpolation_multiplier = multiplier
        self._action_queue_size_to_get_new_actions = action_queue_size_to_get_new_actions
        
        # Thread synchronization
        # Setup signal handler for graceful shutdown
        self.signal_handler = ProcessSignalHandler(use_threads=True, display_pid=False)
        self._shutdown_event = self.signal_handler.shutdown_event
        
        # Infrastructure components
        self._action_queue = ActionQueue(enabled=True)
        self._latency_tracker = LatencyTracker(maxlen=100)
        
        # Interpolator for smooth control
        self._interpolator = ActionInterpolator(multiplier) if use_action_interpolation else None
        
        # Statistics
        self._episode_steps = 0
        self._total_inference_time = 0
        self._total_control_time = 0
        self._inference_count = 0
        
        
        # State tracking
        self._inference_warmup_steps = 0
    
    def run(self) -> None:
        """Run the runtime with RTC."""
        for episode_idx in range(self._num_episodes):
            self._run_episode(episode_idx)
        
        # Final reset
        self._environment.reset()
        
        # Log statistics
        self._log_statistics()
    
    def run_in_new_thread(self) -> threading.Thread:
        """Run the runtime in a new thread."""
        thread = threading.Thread(target=self.run)
        thread.start()
        return thread
    
    def stop(self) -> None:
        """Stop the runtime gracefully."""
        self._shutdown_event.set()
       
    
    def _run_episode(self, episode_idx: int) -> None:
        """Run a single episode."""
        print(f"Starting episode {episode_idx + 1}...")
        
        # Reset environment and policy
        self._environment.reset()
        self._policy.reset()
        
        # Reset infrastructure
        self._action_queue = ActionQueue(enabled=True)  # Create new queue
        self._latency_tracker.reset()
        
        # Reset interpolator
        if self._interpolator:
            self._interpolator.reset()
        
        # Reset statistics and state
        self._episode_steps = 0
        self._total_inference_time = 0
        self._total_control_time = 0
        self._inference_count = 0
        self._inference_warmup_steps = 0
      
        
        # Notify subscribers
        for subscriber in self._subscribers:
            subscriber.on_episode_start()
        
        # Start worker threads
        self._shutdown_event.clear()
        
        get_action_thread = threading.Thread(
            target=self._get_action_worker,
            daemon=True,
            name="GetActionThread"
        )
        
        actor_control_thread = threading.Thread(
            target=self._actor_control_worker,
            daemon=True,
            name="ActorControlThread"
        )
        
        get_action_thread.start()
        
        time.sleep(3)
        
        actor_control_thread.start()
        
        # Main thread monitors episode completion
        start_time = time.perf_counter()
        while not self._shutdown_event.is_set():
            time.sleep(10)
            # Check episode completion conditions
            if self._environment.is_episode_complete():
                print("Environment marked episode as complete")
                break
            
            now = time.perf_counter()
            if (now - start_time)  > self._max_episode_time_s:
                self._episode_steps += 1
                break
                
        print(f"Episode {episode_idx + 1} duration reached or shutdown requested")

        # Signal shutdown
        self._shutdown_event.set()
        
        # Wait for threads to finish
        if actor_control_thread and actor_control_thread.is_alive():
            print("Waiting for action executor thread to finish...")
            actor_control_thread.join()

        if get_action_thread and get_action_thread.is_alive():
            print("Waiting for chunk requester thread to finish...")
            get_action_thread.join()
        
        
        # Notify subscribers of episode end
        for subscriber in self._subscribers:
            subscriber.on_episode_end()
        
        print(f"Episode {episode_idx + 1} completed. Steps: {self._episode_steps}")
    
    def _get_action_worker(self):
        """Worker thread that gets observations and runs policy inference.
        
        This implements the core RTC logic:
        1. Monitors action queue size and triggers inference when needed
        2. Tracks inference latency and calculates real_delay
        3. Uses unified policy.infer() interface with use_rtc parameter
        4. Merges new actions into queue using ActionQueue.merge()
        """
        print("[GetActionThread] Starting...")
        
        # Calculate time per step
        time_per_step = 1.0 / self._fps
        
        # Use threshold from configuration
        get_actions_threshold = self._action_queue_size_to_get_new_actions
  
        
        while not self._shutdown_event.is_set():
            try:
                # Check if we need to get new actions
                if self._action_queue.qsize() <= get_actions_threshold:
                    current_time = time.perf_counter()
                    
                    # Record state before inference
                    action_index_before_inference = self._action_queue.get_action_index()
                    
                    # Get leftover actions for RTC continuity
                    prev_actions = self._action_queue.get_left_over()
                    
                    # Calculate expected inference delay based on latency history
                    inference_latency = self._latency_tracker.max()
                    inference_delay = math.ceil(inference_latency / time_per_step)
                  
                    # Get observation from environment
                    obs = self._environment.get_observation()
                    
                
                    # Unified infer interface - policy handles RTC internally
                    result = self._policy.infer(
                        obs=obs, 
                        use_rtc=False,
                    )
                    
                    if self._inference_warmup_steps <= 3:
                        self._inference_warmup_steps += 1
                        continue
                    
                    # Extract actions from result
                    # Note: policy should return dict with at least "actions"
                    if "actions" not in result:
                        ValueError(f"Policy inference result missing 'actions' key, result:{result}")
                       
                    
                    actions = result["actions"]
              
                    
                    # Ensure actions are 2D (time_steps, action_dim)
                    if len(actions.shape) == 1:
                        print(f"[GetActionThread] Inference result received. Actions shape: {actions.shape}")
                        actions = actions[np.newaxis, ...]
                        
                        
                    # use CCR to optimize actions if enabled
                    if prev_actions is not None and inference_delay > 0:
                        actions = optimize_actions_with_ccr(
                            actions=actions.copy(),
                            executed_actions=prev_actions.copy(),
                            k=3,
                            n_ctrl=10, # total number of control points (including fixed and free)
                            n_prefix= inference_delay+1, # number of fixed control points at the start (execution horizon + 1)
                            n_free=4, # number of free control points (not fixed by execution horizon)
                            last_pt_weight=0.05,
                        )  
                    
                    # Update statistics
                    new_latency = time.perf_counter() - current_time
                    new_delay = math.ceil(new_latency / time_per_step)
                    
                    self._total_inference_time += new_latency
                    self._inference_count += 1
                    
                    self._latency_tracker.add(new_latency)
                
                    # Merge new actions into queue
                    self._action_queue.merge(
                        original_actions = actions,
                        processed_actions = actions,
                        real_delay=new_delay,
                        action_index_before_inference=action_index_before_inference
                    )
                    
                    # Log inference details (debug level)
                    if self._inference_count % 10 == 0:
                        print(
                            f"[GetActionThread] Inference {self._inference_count}: "
                            f"time={new_latency*1000:.1f}ms, "
                            f"delay={new_delay} steps, "
                            f"queue_size={self._action_queue.qsize()}",
                        )
                else:
                    # Small sleep to prevent busy waiting
                    precise_sleep(time_per_step)
                
            except Exception as e:
                logging.exception(f"[GetActionThread] Error: {e}")
                if not self._shutdown_event.is_set():
                    precise_sleep(0.1)
        
        print("[GetActionThread] Exiting...")
    
    def _actor_control_worker(self):
        """Worker thread that gets actions from queue and sends to environment."""
        print("[ActorControlThread] Starting...")
        
        # Calculate control interval
        if self._interpolator:
            control_interval = self._interpolator.get_control_interval(self._fps)
        else:
            control_interval = 1.0 / self._fps
        
    
        step_count = 0
      
        while not self._shutdown_event.is_set():
            try:
                start_time = time.perf_counter()
                # Get action for interpolation/execution
                action_to_apply = None
                
                if self._interpolator:
                    if self._interpolator.needs_new_action():
                        # Get new action from queue
                        action = self._action_queue.get()
                        if action is not None:
                            self._interpolator.add(action)
                    
                    # Get interpolated action
                    interpolated_action = self._interpolator.get()
                    if interpolated_action is not None:
                        action_to_apply = {"actions": interpolated_action}
                else:
                    # Get action directly from queue
                    action = self._action_queue.get()
                    if action is not None:
                        action_to_apply = {"actions": action}
                
                # Apply action to environment
                if action_to_apply is not None:
                    self._environment.apply_action(action_to_apply)
                    
                    # Notify subscribers
                    for subscriber in self._subscribers:
                         # Get observation for subscribers (optional)
                        observation = self._environment.get_observation()
                        subscriber.on_step(observation, action_to_apply)
                    
                    step_count += 1
               
                dt = time.perf_counter() - start_time
                self._total_control_time += dt
        
                # Sleep to maintain control frequency
                sleep_time = control_interval - dt
                if sleep_time > 0:
                    precise_sleep(sleep_time)
                else:
                    # If sleep_time is negative or zero, we don't sleep to avoid over-compensating
                    print(f"[ActorControlThread] Warning: control loop is taking longer ({dt*1000:.1f}ms) than control interval ({control_interval*1000:.1f}ms). Skipping sleep.")
         
                
            except Exception as e:
                logging.exception(f"[ActorControlThread] Error: {e}")
                if not self._shutdown_event.is_set():
                    precise_sleep(0.1)
        
        print(f"[ActorControlThread] Exiting. Executed {step_count} steps.")
    
    def _log_statistics(self):
        """Log runtime statistics."""
        print("=" * 50)
        print("Runtime Statistics")
        print("=" * 50)
        
        # Inference statistics
        if self._inference_count > 0:
            avg_inference_time = self._total_inference_time / self._inference_count
            print(f"Inference Statistics:")
            print(f"  Average inference time: {avg_inference_time*1000:.2f}ms")
            print(f"  Total inferences: {self._inference_count}")
            
            p95_latency = self._latency_tracker.p95()
            max_latency = self._latency_tracker.max()
            print(f"  P95 inference latency: {p95_latency*1000:.2f}ms")
            print(f"  Max inference latency: {max_latency*1000:.2f}ms")
        
        # Control statistics
        if self._episode_steps > 0:
            avg_control_time = self._total_control_time / self._episode_steps
            effective_fps = self._episode_steps / self._total_control_time if self._total_control_time > 0 else 0
            
            print(f"Control Statistics:")
            print(f"  Average control loop time: {avg_control_time*1000:.2f}ms")
            print(f"  Total steps executed: {self._episode_steps}")
            print(f"  Effective FPS: {effective_fps:.1f} (target: {self._fps})")
        
        print("=" * 50)
    
    @property
    def episode_steps(self) -> int:
        """Get current episode step count."""
        return self._episode_steps
    
    @property
    def action_queue_size(self) -> int:
        """Get current action queue size."""
        return self._action_queue.qsize()