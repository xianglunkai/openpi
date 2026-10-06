from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path

import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Play exported joint chunks on the robot. Inside each chunk, steps are sent at step_hz. "
            "Across chunk boundaries, the interval is chunk_boundary_interval_ms instead of the normal step interval."
        )
    )
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory produced by export_episode_joint_playback.py.")
    parser.add_argument("--mode", choices=("ref", "actor"), required=True, help="Which exported chunk sequence to play.")
    parser.add_argument("--topic", type=str, default="/joint_states_gripper", help="JointState topic to publish to.")
    parser.add_argument(
        "--current-state-topic",
        type=str,
        default="/joint_states_single_gripper",
        help="JointState topic used to read the robot's current joint state before startup ramp.",
    )
    parser.add_argument(
        "--joint-names",
        type=str,
        default="joint1,joint2,joint3,joint4,joint5,joint6,gripper",
        help="Comma-separated JointState.name entries.",
    )
    parser.add_argument(
        "--gripper-effort",
        type=float,
        default=1.0,
        help="Gripper effort written to JointState.effort[6], aligned with the mainline ROSCommandPublisher behavior.",
    )
    parser.add_argument("--max-gripper-m", type=float, default=0.097, help="Clamp playback gripper distance to this maximum.")
    parser.add_argument(
        "--disable-gripper-stream",
        action="store_true",
        help="Disable continuous data_msgs/Gripper streaming and publish only JointState commands.",
    )
    parser.add_argument(
        "--gripper-ctrl-topic",
        type=str,
        default="/gripper/gripper/ctrl",
        help="data_msgs/Gripper control topic, aligned with pika_sync_ros.py.",
    )
    parser.add_argument(
        "--gripper-ctrl-rate-hz",
        type=float,
        default=20.0,
        help="Continuous data_msgs/Gripper stream rate, aligned with pika_sync_ros.py.",
    )
    parser.add_argument("--step-hz", type=float, default=20.0, help="Per-step send rate inside a chunk.")
    parser.add_argument(
        "--chunk-boundary-interval-ms",
        type=float,
        default=90.0,
        help="Interval between the last step of chunk N and the first step of chunk N+1. This replaces the normal step interval.",
    )
    parser.add_argument("--start-chunk", type=int, default=0, help="First chunk index to play.")
    parser.add_argument("--max-chunks", type=int, default=None, help="Optional number of chunks to play.")
    parser.add_argument("--initial-wait-ms", type=float, default=0.0, help="Optional wait before the first publish.")
    parser.add_argument(
        "--startup-ramp-duration-sec",
        type=float,
        default=2.0,
        help="Linearly interpolate from current robot joint state to the first playback frame as startup_reset before replay. Set 0 to disable.",
    )
    parser.add_argument(
        "--post-reset-hold-sec",
        type=float,
        default=1.5,
        help="After startup_reset reaches the first playback frame, hold that frame for this duration before replay starts. Set 0 to disable.",
    )
    parser.add_argument(
        "--current-state-timeout-sec",
        type=float,
        default=5.0,
        help="How long to wait for one current JointState message before startup ramp.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print summary only; do not start ROS or publish.")
    return parser.parse_args()


def _load_input(input_dir: Path) -> tuple[dict, np.lib.npyio.NpzFile]:
    meta_path = input_dir / "meta.json"
    npz_path = input_dir / "playback_data.npz"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing {meta_path}")
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing {npz_path}")
    meta = json.loads(meta_path.read_text())
    arrays = np.load(npz_path)
    return meta, arrays


def _select_chunks(arrays: np.lib.npyio.NpzFile, mode: str) -> np.ndarray:
    key = "ref_chunks" if mode == "ref" else "actor_chunks"
    if key not in arrays:
        raise KeyError(f"Missing {key} in exported npz")
    chunks = np.asarray(arrays[key], dtype=np.float64)
    if chunks.ndim != 3:
        raise ValueError(f"{key} must be [num_chunks, chunk_len, action_dim], got shape={chunks.shape}")
    if chunks.shape[2] < 7:
        raise ValueError(f"{key} action_dim must be >= 7 for 6 joints + gripper, got shape={chunks.shape}")
    return chunks


def _slice_chunks(chunks: np.ndarray, start_chunk: int, max_chunks: int | None) -> np.ndarray:
    if start_chunk < 0 or start_chunk >= chunks.shape[0]:
        raise ValueError(f"start_chunk={start_chunk} out of range for {chunks.shape[0]} chunks")
    end = chunks.shape[0] if max_chunks is None else min(chunks.shape[0], start_chunk + max_chunks)
    return chunks[start_chunk:end]


def _print_plan(meta: dict, chunks: np.ndarray, args: argparse.Namespace) -> None:
    num_chunks, chunk_len, action_dim = chunks.shape
    print(f"Input dir: {args.input_dir}")
    print(f"Mode: {args.mode}")
    print(f"Episode: {meta['episode_id']}")
    print(f"Chunks to play: {num_chunks}")
    print(f"Replay start chunk: {args.start_chunk}")
    print(f"Chunk length: {chunk_len}")
    print(f"Action dim: {action_dim}")
    print(f"Step Hz: {args.step_hz}")
    print(f"Step interval: {1000.0 / args.step_hz:.1f} ms")
    print(f"Chunk boundary interval: {args.chunk_boundary_interval_ms:.1f} ms")
    print(f"Joint topic: {args.topic}")
    print(f"Current state topic: {args.current_state_topic}")
    print(f"Joint names: {args.joint_names}")
    print(f"Gripper effort: {args.gripper_effort:.3f}")
    print(f"Max gripper: {args.max_gripper_m:.3f} m")
    print(f"Gripper stream enabled: {not args.disable_gripper_stream}")
    if not args.disable_gripper_stream:
        print(f"Gripper ctrl topic: {args.gripper_ctrl_topic}")
        print(f"Gripper ctrl rate: {args.gripper_ctrl_rate_hz:.1f} Hz")
    print(f"Startup reset duration: {args.startup_ramp_duration_sec:.2f} s")
    print(f"Post-reset hold duration: {args.post_reset_hold_sec:.2f} s")
    print(
        "Timing semantics: inside a chunk, steps are spaced by the normal step interval; "
        "between chunks, the interval is the chunk-boundary interval instead of the normal step interval."
    )
    print("Phase semantics:")
    print("  1. startup_reset: current robot state -> first playback frame, not part of replay")
    print("  2. post_reset_hold: hold the first playback frame, not part of replay")
    print("  3. replay: publish exported chunks starting at chunk[0][0]")


def _extract_joint_positions(msg, action_dim: int) -> np.ndarray:
    positions = np.asarray(msg.position[:action_dim], dtype=np.float64)
    if positions.shape[0] != action_dim:
        raise RuntimeError(f"Expected at least {action_dim} joint positions, got {positions.shape[0]}")
    return positions


def _normalize_gripper_position(gripper_m: float, max_gripper_m: float) -> float:
    gripper_m = float(np.clip(gripper_m, 0.0, max_gripper_m))
    if gripper_m < 0.003:
        gripper_m = 0.0
    return gripper_m


def _publish_chunks(chunks: np.ndarray, args: argparse.Namespace) -> None:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    try:
        from data_msgs.msg import Gripper as GripperMsg
    except Exception:  # noqa: BLE001
        GripperMsg = None

    class PublisherNode(Node):
        def __init__(
            self,
            cmd_topic: str,
            current_state_topic: str,
            *,
            gripper_ctrl_topic: str,
            gripper_ctrl_rate_hz: float,
            enable_gripper_stream: bool,
        ):
            super().__init__("joint_chunk_playback_publisher")
            self.publisher = self.create_publisher(JointState, cmd_topic, 10)
            self.current_joint_msg: JointState | None = None
            self.create_subscription(JointState, current_state_topic, self._on_current_joint, 1)
            self.gripper_publisher = None
            self._gripper_lock = threading.Lock()
            self._gripper_enabled = True
            self._gripper_distance = 0.08
            self._gripper_velocity = 0.0
            self._gripper_timer = None

            if enable_gripper_stream:
                if GripperMsg is None:
                    raise RuntimeError(
                        "data_msgs.msg.Gripper is unavailable, but gripper stream is enabled by default. "
                        "Install the message package or pass --disable-gripper-stream."
                    )
                self.gripper_publisher = self.create_publisher(GripperMsg, gripper_ctrl_topic, 10)
                period = 1.0 / max(float(gripper_ctrl_rate_hz), 1e-6)
                self._gripper_timer = self.create_timer(period, self._on_gripper_timer)

        def _on_current_joint(self, msg: JointState) -> None:
            self.current_joint_msg = msg

        def set_gripper_target(self, distance_m: float, *, velocity: float = 0.0, enable: bool = True) -> None:
            with self._gripper_lock:
                self._gripper_distance = float(distance_m)
                self._gripper_velocity = float(velocity)
                self._gripper_enabled = bool(enable)

        def _on_gripper_timer(self) -> None:
            if self.gripper_publisher is None:
                return
            with self._gripper_lock:
                distance = self._gripper_distance
                velocity = self._gripper_velocity
                enable = self._gripper_enabled

            msg = GripperMsg()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.enable = enable
            msg.set_zero = False
            msg.distance = distance
            msg.velocity = velocity
            self.gripper_publisher.publish(msg)

        def publish_positions(
            self,
            positions: np.ndarray,
            joint_names: list[str],
            gripper_effort: float,
            *,
            max_gripper_m: float,
        ) -> None:
            positions = np.asarray(positions, dtype=np.float64).reshape(-1).copy()
            if positions.shape[0] < 7:
                raise ValueError(f"Expected action dim >= 7, got {positions.shape}")
            positions[6] = _normalize_gripper_position(positions[6], max_gripper_m)
            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name = joint_names
            msg.position = positions.astype(float).tolist()
            msg.effort = [0.0] * 6 + [float(gripper_effort)]
            self.publisher.publish(msg)
            if self.gripper_publisher is not None:
                self.set_gripper_target(positions[6], velocity=0.0, enable=True)

    joint_names = [name.strip() for name in args.joint_names.split(",") if name.strip()]
    step_interval_s = 1.0 / args.step_hz
    chunk_boundary_interval_s = args.chunk_boundary_interval_ms / 1000.0
    initial_wait_s = args.initial_wait_ms / 1000.0
    startup_ramp_duration_s = args.startup_ramp_duration_sec
    post_reset_hold_s = args.post_reset_hold_sec
    current_state_timeout_s = args.current_state_timeout_sec

    def sleep_with_spin(node: PublisherNode, duration_s: float) -> None:
        deadline = time.time() + max(float(duration_s), 0.0)
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            rclpy.spin_once(node, timeout_sec=min(0.05, remaining))

    rclpy.init()
    node = PublisherNode(
        args.topic,
        args.current_state_topic,
        gripper_ctrl_topic=args.gripper_ctrl_topic,
        gripper_ctrl_rate_hz=args.gripper_ctrl_rate_hz,
        enable_gripper_stream=not args.disable_gripper_stream,
    )
    try:
        if initial_wait_s > 0:
            sleep_with_spin(node, initial_wait_s)

        if startup_ramp_duration_s > 0:
            print(f"Phase 1/3: startup_reset ({startup_ramp_duration_s:.2f}s)")
        else:
            print("Phase 1/3: startup_reset skipped")

        if startup_ramp_duration_s > 0:
            deadline = time.time() + current_state_timeout_s
            while node.current_joint_msg is None and time.time() < deadline:
                rclpy.spin_once(node, timeout_sec=0.1)
            if node.current_joint_msg is None:
                raise RuntimeError(
                    f"Timed out after {current_state_timeout_s:.1f}s waiting for current joint state on {args.current_state_topic}"
                )
            current_positions = _extract_joint_positions(node.current_joint_msg, chunks.shape[2])
            target_positions = np.asarray(chunks[0, 0], dtype=np.float64)
            ramp_steps = max(2, int(round(startup_ramp_duration_s * args.step_hz)))
            for ramp_idx in range(1, ramp_steps + 1):
                alpha = ramp_idx / ramp_steps
                interp = (1.0 - alpha) * current_positions + alpha * target_positions
                node.publish_positions(interp, joint_names, args.gripper_effort, max_gripper_m=args.max_gripper_m)
                if ramp_idx < ramp_steps:
                    sleep_with_spin(node, step_interval_s)

        if post_reset_hold_s > 0:
            print(f"Phase 2/3: post_reset_hold ({post_reset_hold_s:.2f}s)")
            node.publish_positions(chunks[0, 0], joint_names, args.gripper_effort, max_gripper_m=args.max_gripper_m)
            sleep_with_spin(node, post_reset_hold_s)
        else:
            print("Phase 2/3: post_reset_hold skipped")

        print("Phase 3/3: replay")
        print(f"Replay started at chunk={args.start_chunk} step=0")
        num_chunks, chunk_len, _ = chunks.shape
        for chunk_idx in range(num_chunks):
            chunk = chunks[chunk_idx]
            for step_idx in range(chunk_len):
                node.publish_positions(chunk[step_idx], joint_names, args.gripper_effort, max_gripper_m=args.max_gripper_m)

                is_last_step_in_chunk = step_idx == chunk_len - 1
                is_last_chunk = chunk_idx == num_chunks - 1
                if is_last_step_in_chunk:
                    if not is_last_chunk:
                        sleep_with_spin(node, chunk_boundary_interval_s)
                else:
                    sleep_with_spin(node, step_interval_s)
    finally:
        node.destroy_node()
        rclpy.shutdown()


def main() -> None:
    args = _parse_args()
    meta, arrays = _load_input(args.input_dir)
    chunks = _select_chunks(arrays, args.mode)
    chunks = _slice_chunks(chunks, args.start_chunk, args.max_chunks)
    _print_plan(meta, chunks, args)
    if args.dry_run:
        return
    _publish_chunks(chunks, args)


if __name__ == "__main__":
    main()
