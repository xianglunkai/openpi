import gymnasium
import numpy as np
import einops
from openpi_client import image_tools
from openpi_client.runtime import environment as _environment
from typing_extensions import override
import logging
import time


class AlohaSimEnvironment(_environment.Environment):
    """An environment for an Aloha robot in simulation that mirrors the real env I/O.

    This exposes the same observation dict shape as `AlohaRealEnvironment` so the
    rest of the code can use the same transforms and subscribers.
    """

    def __init__(
        self,
        task: str,
        obs_type: str = "pixels_agent_pos",
        seed: int = 0,
        render_height: int = 224,
        render_width: int = 224,
        control_freq_hz: int = 50,
        zmq_monitor: bool = False,
        zmq_endpoint: str = "tcp://127.0.0.1:5556",
        use_physical_images: bool = True,
    ) -> None:
        np.random.seed(seed)
        self._rng = np.random.default_rng(seed)

        # If physical images are disabled we operate in a fake-mode that
        # does not instantiate the simulator (avoids any GL/context work).
        # In fake-mode we synthesize deterministic state and images below.
        self._fake_mode = not bool(use_physical_images)
        if not self._fake_mode:
            # normal simulator-backed environment
            self._gym = gymnasium.make(task, obs_type=obs_type)
        else:
            self._gym = None
        self._render_height = render_height
        self._render_width = render_width
        self._control_freq_hz = control_freq_hz

        self._last_obs = None
        self._done = True
        self._episode_reward = 0.0

        # Whether to use images produced by the physics engine. When False,
        # images are synthesized deterministically from the RNG to avoid GPU/
        # GL context work on non-main threads.
        self._use_physical_images = bool(use_physical_images)

        # ZeroMQ publisher (optional). Created lazily if requested.
        self._zmq_ctx = None
        self._zmq_pub = None
        self._zmq = None
        if zmq_monitor:
            try:
                import zmq

                self._zmq_ctx = zmq.Context()
                self._zmq_pub = self._zmq_ctx.socket(zmq.PUB)
                # bind so external viewers can connect
                self._zmq_pub.bind(zmq_endpoint)
                # make sends non-blocking at caller side
                self._zmq = zmq
            except Exception as e:
                logging.warning(f"Failed to start ZeroMQ publisher in sim env: {e}")
                self._zmq_ctx = None
                self._zmq_pub = None
                self._zmq = None

    @override
    def reset(self) -> None:
        if not self._fake_mode:
            gym_obs, _ = self._gym.reset(seed=int(self._rng.integers(2**32 - 1)))
            self._last_obs = self._convert_observation(gym_obs)  # type: ignore
            self._done = False
            self._episode_reward = 0.0
        else:
            # Synthesize a deterministic initial observation
            self._episode_step = 0
            synth = self._synthesize_observation()
            self._last_obs = synth
            self._done = False
            self._episode_reward = 0.0

    @override
    def is_episode_complete(self) -> bool:
        return self._done

    @override
    def get_observation(self) -> dict:
        if self._last_obs is None:
            raise RuntimeError("Observation is not set. Call reset() first.")

        return self._last_obs  # type: ignore

    @override
    def apply_action(self, action: dict) -> None:
        # Accept action dict with key "actions" to mirror real env interface.
        if not self._fake_mode:
            gym_obs, reward, terminated, truncated, info = self._gym.step(action["actions"])
            self._last_obs = self._convert_observation(gym_obs)  # type: ignore
            self._done = terminated or truncated
            # keep track of cumulative or peak reward similar to other env implementations
            self._episode_reward = max(self._episode_reward, reward)
        else:
            # In fake mode, advance deterministic state and synthesize images.
            self._episode_step = getattr(self, "_episode_step", 0) + 1
            self._last_obs = self._synthesize_observation()
            self._done = False
            self._episode_reward = 0.0
        # Publish lightweight telemetry (timestamp, real state, desired actions)
        try:
            if self._zmq_pub is not None and self._zmq is not None:
                real = self._last_obs.get("state") if self._last_obs is not None else None
                try:
                    real_list = np.asarray(real).ravel().tolist()
                except Exception:
                    real_list = []

                desired = action.get("actions")
                try:
                    desired_list = np.asarray(desired).ravel().tolist()
                except Exception:
                    desired_list = []

                msg = {"ts": time.time(), "real": real_list, "desired": desired_list}
                try:
                    # non-blocking send
                    self._zmq_pub.send_json(msg, flags=self._zmq.NOBLOCK)
                except Exception:
                    # drop if cannot send
                    pass
        except Exception:
            pass

    def close(self) -> None:
        """Close any optional resources (ZeroMQ sockets/contexts)."""
        try:
            if self._zmq_pub is not None:
                try:
                    self._zmq_pub.close(linger=0)
                except Exception:
                    try:
                        self._zmq_pub.close()
                    except Exception:
                        pass
                self._zmq_pub = None
        except Exception:
            pass
        try:
            if self._zmq_ctx is not None:
                try:
                    self._zmq_ctx.term()
                except Exception:
                    pass
                self._zmq_ctx = None
        except Exception:
            pass

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _convert_observation(self, gym_obs: dict) -> dict:
        # If physical images are enabled and the simulator provided pixels,
        # use and normalize them. Otherwise synthesize deterministic images
        # (so downstream pipelines receive the same dict shape).
        pixels = {k: v for k, v in gym_obs.get("pixels", {}).items() if "_depth" not in k}

        images = {}
        if self._use_physical_images and len(pixels) > 0:
            # Map known keys to standard cam names and convert to CHW uint8
            for k, img in pixels.items():
                lk = k.lower()
                if lk in ("top", "top_rgb", "camera_top") or "top" in lk:
                    name = "cam_high"
                elif "mid" in lk or lk in ("camera_mid", "mid_rgb"):
                    name = "cam_left_wrist"
                elif "low" in lk or "wrist" in lk or lk in ("camera_bottom", "bottom", "cam_low"):
                    name = "cam_right_wrist"
                elif lk in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
                    name = lk
                else:
                    # Unrecognized camera key; skip
                    continue

                img = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(img, self._render_height, self._render_width)
                )
                images[name] = einops.rearrange(img, "h w c -> c h w")

        # Ensure presence of three cameras. Use deterministic RNG to fill missing ones.
        desired = ["cam_high", "cam_left_wrist", "cam_right_wrist"]
        for cam in desired:
            if cam not in images:
                rnd = self._rng.integers(0, 256, size=(self._render_height, self._render_width, 3), dtype=np.uint8)
                images[cam] = einops.rearrange(rnd, "h w c -> c h w")

        # Provide a state key compatible with real env (use qpos if available else agent_pos)
        state = gym_obs.get("qpos", gym_obs.get("agent_pos"))

        return {"state": state, "images": images}

    def _synthesize_observation(self) -> dict:
        """Create a deterministic synthetic observation (state + 3 images).

        The images are generated from the RNG seeded in __init__ so they are
        reproducible between runs when the same seed is used.
        """
        # deterministic synthetic observation (no RNG used)
        step = int(getattr(self, "_episode_step", 0))

        # deterministic state: 14 joints for standard Aloha robot.
        # Use a small ramp plus a smooth periodic component so values evolve
        # with step but remain fully deterministic and repeatable.
        base = np.linspace(-0.1, 0.1, num=14, dtype=np.float32)
        phase = 0.1 * step
        cyc = 0.01 * np.sin(np.arange(14, dtype=np.float32) * 0.2 + phase)
        state = base + cyc

        H = int(self._render_height)
        W = int(self._render_width)

        # create coordinate grids (0..255) deterministically
        y = np.linspace(0, 255, H, dtype=np.uint8)[:, None]
        x = np.linspace(0, 255, W, dtype=np.uint8)[None, :]
        xv = np.broadcast_to(x, (H, W))
        yv = np.broadcast_to(y, (H, W))

        images = {}

        # cam_high: smooth horizontal + vertical gradients with step offset
        r = ((xv + step) % 256).astype(np.uint8)
        g = ((yv + step) % 256).astype(np.uint8)
        b = (((xv.astype(np.uint16) + yv.astype(np.uint16)) // 2 + step) % 256).astype(np.uint8)
        img = np.stack([r, g, b], axis=-1)
        images["cam_high"] = einops.rearrange(img, "h w c -> c h w")

        # cam_left_wrist: vertical-heavy pattern (stripes)
        r = (((xv.astype(np.uint16) * 2) + (step * 3)) % 256).astype(np.uint8)
        g = (((yv.astype(np.uint16) * 3) + (step * 5)) % 256).astype(np.uint8)
        b = (((xv.astype(np.uint16) + (yv.astype(np.uint16) * 2)) + (step * 7)) % 256).astype(np.uint8)
        img = np.stack([r, g, b], axis=-1)
        images["cam_left_wrist"] = einops.rearrange(img, "h w c -> c h w")

        # cam_right_wrist: coarse checkerboard-like pattern for contrast
        r = ((((xv // 8).astype(np.uint16) * 16) + ((yv // 8).astype(np.uint16) * 32) + (step * 11)) % 256).astype(np.uint8)
        g = ((((xv // 16).astype(np.uint16) * 24) + ((yv // 16).astype(np.uint16) * 8) + (step * 13)) % 256).astype(np.uint8)
        b = ((((xv // 10).astype(np.uint16) * 12) + ((yv // 12).astype(np.uint16) * 18) + (step * 17)) % 256).astype(np.uint8)
        img = np.stack([r, g, b], axis=-1)
        images["cam_right_wrist"] = einops.rearrange(img, "h w c -> c h w")

        return {"state": state, "images": images}
