"""Lightweight RealTimeMonitor thread for visualizing real vs desired joint angles.

Place this file under `examples/mobile_aloha_AgileX/` and start it from your
simulation main (e.g. `main_sim.py`) by creating a Queue and calling
`monitor = RealTimeMonitor(q, joint_names=..., interval=0.1); monitor.start()`.

The monitor will attempt to use matplotlib (Agg backend). In headless setups it
will fall back to printing a concise one-line summary and save a PNG
`realtime_angles.png` periodically when matplotlib is available.
"""
from queue import Queue, Empty
import threading
import time
from typing import Dict, List

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False


class RealTimeMonitor(threading.Thread):
    def __init__(self, queue: Queue, joint_names: List[str], interval: float = 0.1, title: str = "Real vs Desired"):
        super().__init__(daemon=True)
        self.queue = queue
        self.joint_names = list(joint_names)
        self.interval = float(interval)
        self._stop = threading.Event()

        # latest readings
        self.latest_real: Dict[str, float] = {n: 0.0 for n in self.joint_names}
        self.latest_desired: Dict[str, float] = {n: 0.0 for n in self.joint_names}

        # plotting state
        self._fig = None
        self._ax = None
        if HAS_MPL:
            try:
                self._fig, self._ax = plt.subplots(1, 1, figsize=(8, 3))
                self._fig.suptitle(title)
            except Exception:
                self._fig = None
                self._ax = None

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        last_draw = 0.0
        while not self._stop.is_set():
            # Drain queue to get most recent sample
            try:
                while True:
                    ts, real, desired = self.queue.get_nowait()
                    for k in self.joint_names:
                        if k in real:
                            self.latest_real[k] = float(real[k])
                        if k in desired:
                            self.latest_desired[k] = float(desired[k])
            except Empty:
                pass

            now = time.time()
            if now - last_draw >= self.interval:
                self._render()
                last_draw = now

            time.sleep(min(self.interval / 2.0, 0.01))

        # final render
        self._render()

    def _render(self) -> None:
        if not HAS_MPL or self._ax is None:
            # Fallback: print concise text summary
            pairs = [f"{n}: r={self.latest_real[n]:.3f}, d={self.latest_desired[n]:.3f}" for n in self.joint_names]
            print(" | ".join(pairs))
            return

        ax = self._ax
        ax.clear()
        x = range(len(self.joint_names))
        real_vals = [self.latest_real[n] for n in self.joint_names]
        desired_vals = [self.latest_desired[n] for n in self.joint_names]
        width = 0.35
        ax.bar([i - width / 2 for i in x], real_vals, width=width, label="real")
        ax.bar([i + width / 2 for i in x], desired_vals, width=width, label="desired")
        ax.set_xticks(list(x))
        ax.set_xticklabels(self.joint_names, rotation=45, ha="right")
        ax.set_ylabel("angle")
        ax.legend()
        try:
            self._fig.tight_layout()
            # save a snapshot to disk (useful in headless CI)
            self._fig.canvas.draw()
            self._fig.savefig("realtime_angles.png")
        except Exception:
            # If saving fails, degrade to printing
            pairs = [f"{n}: r={self.latest_real[n]:.3f}, d={self.latest_desired[n]:.3f}" for n in self.joint_names]
            print(" | ".join(pairs))
