#!/usr/bin/env python3
"""Example/test for OptimizeActionsQP transform.

Usage:
    python examples/test_optimize_actions_qp.py

If osqp or scipy is not installed you'll see an ImportError with guidance.
"""
import sys
import numpy as np

try:
    from openpi.transforms import OptimizeActionsQP
except Exception as e:
    print("Failed to import OptimizeActionsQP:", e)
    print("Ensure you are running from the project root and that dependencies are installed:")
    print("    pip install osqp scipy")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
except Exception as e:
    print("Failed to import matplotlib:", e)
    print("Install it to enable plotting:")
    print("    pip install matplotlib")
    # We don't exit here because user might still want to run the numeric part
    plt = None


def make_noisy_trajs(B=2, T=50, A=3, noise_scale=0.5):
    t = np.linspace(0, 2 * np.pi, T)
    base = np.stack([np.sin(t), np.cos(t), 0.5 * np.sin(2 * t)], axis=-1)  # (T, A)
    base = np.broadcast_to(base[None, ...], (B, T, A))
    noise = np.random.randn(B, T, A) * noise_scale
    return base + noise


def show_stats(name, arr):
    print(f"{name}: shape={arr.shape}, min={arr.min():.4f}, max={arr.max():.4f}, mean_abs={np.mean(np.abs(np.diff(arr, axis=1))):.4f}")


if __name__ == "__main__":
    np.random.seed(0)
    B, T, A = 2, 50, 3
    actions = make_noisy_trajs(B=B, T=T, A=A, noise_scale=0.6)

    print("Before smoothing (first batch, first channel):")
    print(actions[0, :8, 0])
    show_stats("Before (all)", actions)

    transform = OptimizeActionsQP(dt=1.0, vel_limits=(-2.0, 2.0), acc_limits=None, w_data=1.0, w_acc=50.0, fix_ends=True, verbose=True)

    data = {"actions": actions}
    out = transform(data)

    smoothed = out["actions"]
    print("\nAfter smoothing (first batch, first channel):")
    print(smoothed[0, :8, 0])
    show_stats("After (all)", smoothed)

    # Print a small diff summary
    diff = np.abs(smoothed - actions)
    print(f"\nMean absolute change: {diff.mean():.6f}, max change: {diff.max():.6f}")
    print("Done.")

    # If matplotlib is available, make and save plots comparing before/after
    if plt is not None:
        B, T, A = actions.shape
        # Plot first batch per-channel before/after
        fig, axes = plt.subplots(A, 1, figsize=(10, 3 * max(1, A)), sharex=True)
        if A == 1:
            axes = [axes]
        for a in range(A):
            ax = axes[a]
            ax.plot(actions[0, :, a], label="before", alpha=0.6)
            ax.plot(smoothed[0, :, a], label="after", alpha=0.9)
            ax.set_ylabel(f"ch{a}")
            ax.legend()
        axes[-1].set_xlabel("time")
        fig.suptitle("OptimizeActionsQP: before vs after (first batch)")
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        out_path1 = "examples/optimize_actions_qp_before_after.png"
        fig.savefig(out_path1, dpi=150)
        print(f"Saved before/after plot: {out_path1}")

        # Plot mean absolute change over time (averaged over batch and channels)
        time_mean = np.mean(np.abs(smoothed - actions), axis=(0, 2))  # (T,)
        fig2, ax2 = plt.subplots(1, 1, figsize=(10, 3))
        ax2.plot(time_mean)
        ax2.set_title("Mean absolute change over time (avg over batch & channels)")
        ax2.set_xlabel("time")
        ax2.set_ylabel("mean abs change")
        out_path2 = "examples/optimize_actions_qp_diff.png"
        fig2.tight_layout()
        fig2.savefig(out_path2, dpi=150)
        print(f"Saved diff-over-time plot: {out_path2}")
