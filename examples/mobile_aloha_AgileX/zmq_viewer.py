"""ZeroMQ subscriber visualizer for real vs desired joint angles.

Run as a separate process. It connects to the publisher endpoint and subscribes
to all messages. Each message is expected to be a JSON object with keys:
  {"ts": float, "real": {joint: float, ...}, "desired": {joint: float, ...}}

The viewer will attempt to use matplotlib (Agg) to save a snapshot `realtime_angles_zmq.png`.
If matplotlib is unavailable, it will fallback to printing summaries.
"""
import time
import argparse
from typing import Dict, List
import math
from collections import deque
import numpy as np

try:
    import zmq
except Exception:
    zmq = None

try:
    import os
    import matplotlib
    # If a display is available, prefer an interactive backend; otherwise use Agg
    INTERACTIVE = bool(os.environ.get("DISPLAY"))
    if not INTERACTIVE:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False
    INTERACTIVE = False


def run_viewer(endpoint: str, joint_names: List[str], interval: float = 0.1):
    if zmq is None:
        raise RuntimeError("pyzmq is required to run zmq_viewer.py")

    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt_string(zmq.SUBSCRIBE, "")
    sock.connect(endpoint)
    sock.RCVTIMEO = int(interval * 1000)

    latest_real: Dict[str, float] = {n: 0.0 for n in joint_names}
    latest_desired: Dict[str, float] = {n: 0.0 for n in joint_names}

    if HAS_MPL:
        # create a grid of subplots (one per joint)
        n = len(joint_names)
        cols = int(math.ceil(math.sqrt(n)))
        rows = int(math.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(max(8, cols * 3), max(3, rows * 2)), squeeze=False)
        fig.suptitle("ZeroMQ: Real vs Desired (per-joint time series)")
        # buffers for time-series history
        # keep a few seconds of history; avoid pre-filling buffers so lengths
        # stay consistent with `times` (pre-filling caused shape mismatches).
        buf_len = max(100, int(2.0 / interval) * 50)  # heuristic
        times = deque(maxlen=buf_len)
        real_bufs = {n: deque(maxlen=buf_len) for n in joint_names}
        des_bufs = {n: deque(maxlen=buf_len) for n in joint_names}

        # flatten axes and create lines per joint
        flat_axes = [axes[i // cols][i % cols] for i in range(rows * cols)]
        joint_axes = flat_axes[:n]
        lines_real = {}
        lines_des = {}
        for ax, jn in zip(joint_axes, joint_names):
            ln_real, = ax.plot([], [], label="real", color="C0")
            ln_des, = ax.plot([], [], label="desired", color="C1")
            ax.set_title(jn, fontsize=8)
            ax.grid(True, linestyle="--", linewidth=0.5)
            ax.legend(fontsize=6)
            lines_real[jn] = ln_real
            lines_des[jn] = ln_des
        # hide unused axes
        for ax in flat_axes[n:]:
            ax.axis("off")

        # interactive mode if DISPLAY available
        if INTERACTIVE:
            try:
                plt.ion()
                fig.show()
            except Exception:
                pass

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    else:
        fig = None
        joint_axes = []
        lines_real = {}
        lines_des = {}

    try:
        while True:
            try:
                msg = sock.recv_json(flags=0)
            except Exception:
                # timeout or interrupted; still render periodically
                msg = None

            if msg is not None:
                real = msg.get("real", {})
                desired = msg.get("desired", {})

                # Support both dict (by joint name) and list (ordered) formats
                if isinstance(real, list):
                    for i, n in enumerate(joint_names):
                        if i < len(real):
                            try:
                                latest_real[n] = float(real[i])
                            except Exception:
                                pass
                elif isinstance(real, dict):
                    for n in joint_names:
                        if n in real:
                            try:
                                latest_real[n] = float(real[n])
                            except Exception:
                                pass

                if isinstance(desired, list):
                    for i, n in enumerate(joint_names):
                        if i < len(desired):
                            try:
                                latest_desired[n] = float(desired[i])
                            except Exception:
                                pass
                elif isinstance(desired, dict):
                    for n in joint_names:
                        if n in desired:
                            try:
                                latest_desired[n] = float(desired[n])
                            except Exception:
                                pass

            # render
            if not HAS_MPL or fig is None:
                print(" | ".join(f"{n}: r={latest_real[n]:.3f}, d={latest_desired[n]:.3f}" for n in joint_names))
            else:
                # append time and values to buffers
                t = time.time()
                if len(times) == 0:
                    last_t = t
                else:
                    last_t = times[-1]
                # if no new time advance (recv timeout), increment by interval for a smooth x axis
                if msg is None:
                    t = last_t + interval
                times.append(t)
                for jn in joint_names:
                    real_bufs[jn].append(latest_real[jn])
                    des_bufs[jn].append(latest_desired[jn])

                xs = list(range(-len(times) + 1, 1))
                for jn, ax in zip(joint_names, joint_axes):
                    y_real = list(real_bufs[jn])
                    y_des = list(des_bufs[jn])
                    try:
                        lines_real[jn].set_data(xs, y_real)
                        lines_des[jn].set_data(xs, y_des)
                        ax.relim()
                        ax.autoscale_view()
                    except Exception:
                        # on very first draw the data lengths might mismatch; ignore
                        pass

                try:
                    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
                    if INTERACTIVE:
                        # try several methods to update the interactive canvas
                        try:
                            fig.canvas.draw_idle()
                        except Exception:
                            try:
                                fig.canvas.draw()
                            except Exception:
                                pass
                        try:
                            # some backends require flush + pause to update window
                            fig.canvas.flush_events()
                        except Exception:
                            pass
                        try:
                            plt.pause(0.001)
                        except Exception:
                            pass
                    else:
                        # headless: save snapshot for inspection
                        try:
                            fig.canvas.draw()
                        except Exception:
                            pass
                        fig.savefig("realtime_angles_zmq.png")
                except Exception as e:
                    import traceback

                    print("Failed to update figure; printing text instead")
                    traceback.print_exc()
                    print(" | ".join(f"{n}: r={latest_real[n]:.3f}, d={latest_desired[n]:.3f}" for n in joint_names))

            time.sleep(interval)
    except KeyboardInterrupt:
        pass
    finally:
        sock.close()
        ctx.term()


def cli():
    p = argparse.ArgumentParser()
    p.add_argument("--endpoint", default="tcp://127.0.0.1:5556")
    p.add_argument("--joint-names", default="l_joint_0,l_joint_1,l_joint_2,l_joint_3,l_joint_4,l_joint_5,l_gripper_6, r_joint_0,r_joint_1,r_joint_2,r_joint_3,r_joint_4,r_joint_5,r_gripper_6")
    p.add_argument("--interval", type=float, default=0.1)
    args = p.parse_args()
    joint_names = [s.strip() for s in args.joint_names.split(",") if s.strip()]
    run_viewer(args.endpoint, joint_names, args.interval)


if __name__ == "__main__":
    cli()
