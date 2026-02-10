# -*- coding: utf-8 -*-

"""
对比两种模型加载方式的推理结果
1. converted_pkl (triton优化)
2. 29999_converted_float16 (标准方式)
"""

from openpi.training import config as _config
from openpi.policies import policy_config
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import torch

model_name = "pi05_fold_clothes40"

# 固定随机种子以确保使用相同的输入
np.random.seed(42)

def _fixed_observation_droid() -> dict:
    """生成固定的观察数据"""
    return {
        "state": np.ones((14,)),
        "images": {
            "cam_high": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_left_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_right_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": "do something",
    }

print('='*80)
print('生成固定的输入观察数据...')
example = _fixed_observation_droid()

# ===== 加载方式1: converted_pkl + triton优化 =====
print('='*80)
# print('方式1: 加载 converted_pkl 模型 (triton优化)...')
# config1 = _config.get_config(model_name)
# checkpoint_dir1 = "/home/xlk/work/openpi/checkpoints/pi05_cobot/pi05_fold_clothes40/pi05_fold_clothes40_triton"

# start_t1 = time.time()
# policy1 = policy_config.create_trained_policy(config1, checkpoint_dir1, default_prompt=example["prompt"])
# load_time1 = time.time() - start_t1
# print(f'模型1加载完成，耗时: {load_time1*1000:.2f} ms')

# # 预热并计时
# start_w1 = time.time()
# _ = policy1.infer(example)
# warmup_time1 = time.time() - start_w1

# # 预热后多次推理计时
num_runs = 10
# times1 = []
# result1 = None
# for _ in range(num_runs):
#     start_t1 = time.time()
#     result1 = policy1.infer(example)
#     times1.append(time.time() - start_t1)
# inference_time1 = float(np.mean(times1))
# actions1 = result1["actions"]
# print(f'模型1推理完成，平均耗时(排除预热，{num_runs}次): {inference_time1*1000:.2f} ms')
# print(f'输出形状: {actions1.shape}')
# # print(f'输出统计: min={actions1.min():.6f}, max={actions1.max():.6f}, mean={actions1.mean():.6f}, std={actions1.std():.6f}')

torch.cuda.empty_cache()
time.sleep(3)


# ===== 加载方式2: 29999_converted_float16 标准方式 =====
print('='*80)
print('方式2: 加载 29999_converted_float16 模型 (标准方式)...')
config2 = _config.get_config(model_name)
# checkpoint_dir2 = "./checkpoints/pi0_mobile_cobot_navigation_demo/pi0_mobile_cobot_navigation_demo/converted_torch"
checkpoint_dir2 = "/home/xlk/work/openpi/checkpoints/pi05_cobot/pi05_fold_clothes40/pi05_fold_clothes40/25000"

start_t2 = time.time()
policy2 = policy_config.create_trained_policy(config2, checkpoint_dir2, default_prompt=example["prompt"])
load_time2 = time.time() - start_t2
print(f'模型2加载完成，耗时: {load_time2*1000:.2f} ms')

# 预热并计时
start_w2 = time.time()
_0 = policy2.infer(example)
_ = policy2.infer(example)
warmup_time2 = time.time() - start_w2

# 预热后多次推理计时
times2 = []
result2 = None
for _ in range(num_runs):
    start_t2 = time.time()
    result2 = policy2.infer(example)
    times2.append(time.time() - start_t2)
inference_time2 = float(np.mean(times2))
actions2 = result2["actions"]
print(f'模型2推理完成，平均耗时(排除预热，{num_runs}次): {inference_time2*1000:.2f} ms')
print(f'输出形状: {actions2.shape}')
# print(f'输出统计: min={actions2.min():.6f}, max={actions2.max():.6f}, mean={actions2.mean():.6f}, std={actions2.std():.6f}')

# ===== 结果对比 =====
# print('='*80)
# print('结果对比:')
# print(f'加载时间对比: 方式1={load_time1*1000:.2f}ms, 方式2={load_time2*1000:.2f}ms')
# print(f'推理时间对比: 方式1={inference_time1*1000:.2f}ms, 方式2={inference_time2*1000:.2f}ms')

# 计算差异
# diff = np.abs(actions1 - actions2)
# print(f'\n输出差异统计:')
# print(f'  绝对差异: min={diff.min():.6f}, max={diff.max():.6f}, mean={diff.mean():.6f}')
# print(f'  相对差异 (MSE): {np.mean((actions1 - actions2)**2):.6e}')
# print(f'  最大相对误差: {(diff / (np.abs(actions2) + 1e-8)).max():.6f}')

# ===== 可视化对比 =====
# print('='*80)
# print('生成可视化对比图...')

os.makedirs("./figs", exist_ok=True)

# 转换为numpy数组（如果是tensor），形状应为 [T, D]
def to_numpy(actions):
    if hasattr(actions, "detach"):
        actions = actions.detach()
    if hasattr(actions, "cpu"):
        return actions.cpu().numpy()
    return np.array(actions)

actions1_np = to_numpy(actions1)
actions2_np = to_numpy(actions2)

if actions1_np.ndim == 1:
    # 假设是 [D]，扩展一个时间维
    actions1_np = actions1_np[None, :]
    actions2_np = actions2_np[None, :]

# 检查时间步和维度
T1, D1 = actions1_np.shape[0], actions1_np.shape[-1]
T2, D2 = actions2_np.shape[0], actions2_np.shape[-1]
assert D1 == D2, f"action dim mismatch: {D1} vs {D2}"
assert T1 == T2, f"horizon mismatch: {T1} vs {T2}"

T, D = T1, D1
print(f"动作张量形状: T={T}, D={D}")

# 期望 D 至少包含前 16 维
if D < 16:
    raise ValueError(f"action dim D={D} < 16, 无法按关节/状态划分")

time_steps = np.arange(T)
def plot_segment(start, end, name, filename_prefix):
    seg1 = actions1_np[:, start:end]
    seg2 = actions2_np[:, start:end]
    num_dims = end - start

    # 每个维度两行：上方值曲线，下方误差曲线
    fig, axes = plt.subplots(num_dims, 2, figsize=(14, 3 * num_dims), sharex=True)
    if num_dims == 1:
        axes = np.array([axes])

    for i in range(num_dims):
        ax_val = axes[i, 0]
        ax_err = axes[i, 1]
        # 根据数据范围自适应量程（添加10%边距）
        vmin = min(seg1[:, i].min(), seg2[:, i].min())
        vmax = max(seg1[:, i].max(), seg2[:, i].max())
        if vmin == vmax:
            # 处理常量序列，给出一个微小范围避免显示问题
            eps = 1e-3 if vmin == 0 else abs(vmin) * 0.1
            vmin -= eps
            vmax += eps
        margin = (vmax - vmin) * 0.1
        ax_val.set_ylim(vmin - margin, vmax + margin)
        # 值曲线
        ax_val.plot(time_steps, seg1[:, i], "b-", label="Triton", alpha=0.8)
        ax_val.plot(time_steps, seg2[:, i], "r--", label="Standard", alpha=0.8)
        ax_val.set_ylabel(f"dim {start + i}")
        ax_val.grid(True, alpha=0.3)
        if i == 0:
            ax_val.set_title(f"{name} Values (dims {start}-{end-1})")
            ax_val.legend()

        # 误差曲线
        diff = seg1[:, i] - seg2[:, i]
        ax_err.plot(time_steps, diff, "g-", alpha=0.8)
        ax_err.axhline(0.0, color="k", linestyle="--", alpha=0.4)
        # 误差量程设置为对称范围
        max_abs = max(abs(diff.min()), abs(diff.max()))
        if max_abs == 0:
            max_abs = 1e-3
        ax_err.set_ylim(-max_abs * 1.2, max_abs * 1.2)
        ax_err.set_ylabel("error")
        ax_err.grid(True, alpha=0.3)
        if i == 0:
            ax_err.set_title(f"{name} Error")

        if i == num_dims - 1:
            ax_val.set_xlabel("time step")
            ax_err.set_xlabel("time step")

    plt.tight_layout()
    out_path = f"./src/figs/{filename_prefix}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"{name} 对比图已保存至: {out_path}")


# 左臂 7 关节: 0-6
plot_segment(0, 7, "Left Arm Joints", "left_arm_joints")

# 右臂 7 关节: 7-13
if D >= 14:
    plot_segment(7, 14, "Right Arm Joints", "right_arm_joints")

# cmd_vel 和 odom: 14-15 (注意索引从 0 开始)
if D >= 16:
    plot_segment(14, 16, "cmd_vel & odom", "cmd_vel_odom")
def plot_time_comparison(load1, load2, warmup1, warmup2, infer_series1, infer_series2):
    """绘制加载时间、第一次预热时间与后续推理时间对比图。
    推理时间使用预热后的多次测量序列的均值与标准差。
    """
    methods = ["Triton", "Standard"]
    load_times = [load1 * 1000.0, load2 * 1000.0]
    warmup_times = [warmup1 * 1000.0, warmup2 * 1000.0]
    # 计算推理均值与标准差（毫秒）
    infer_mean_1 = float(np.mean(infer_series1)) * 1000.0
    infer_std_1 = float(np.std(infer_series1, ddof=1)) * 1000.0 if len(infer_series1) > 1 else 0.0
    infer_mean_2 = float(np.mean(infer_series2)) * 1000.0
    infer_std_2 = float(np.std(infer_series2, ddof=1)) * 1000.0 if len(infer_series2) > 1 else 0.0

    x = np.arange(len(methods))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, load_times, width, label="Load (ms)")
    ax.bar(x, warmup_times, width, label="Warmup (ms)")
    ax.bar(x + width, [infer_mean_1, infer_mean_2], width, yerr=[infer_std_1, infer_std_2], capsize=6, label="Inference avg ± std (ms)")

    # 数值标注
    for i, v in enumerate(load_times):
        ax.text(x[i] - width, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    for i, v in enumerate(warmup_times):
        ax.text(x[i], v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    for i, v in enumerate([infer_mean_1, infer_mean_2]):
        ax.text(x[i] + width, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Time (ms)")
    ax.set_title("Load, Warmup, and Post-warmup Inference Time Comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out_path = "./src/figs/time_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"时间对比图已保存至: {out_path}")


plot_time_comparison(load_time1, load_time2, warmup_time1, warmup_time2, times1, times2)

print("所有关节/状态对比图已生成。")
