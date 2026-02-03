import threading
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

class JointActionVisualizer:
    """
    实时关节动作可视化类 - 基于位置索引，不依赖关节名称
    显示14个关节的原始指令、滤波后指令和实际反馈位置
    """
    
    def __init__(self, history_length=1000, update_interval=100):
        """
        初始化可视化器
        
        Args:
            history_length: 历史数据保存长度
            update_interval: 图表更新间隔(ms)
        """
        # 初始化ROS节点（如果尚未初始化）
        try:
            rospy.init_node('joint_action_visualizer', anonymous=True)
        except:
            pass
            
        # 数据缓冲区 - 为每个关节创建独立的时间和数据序列
        self.history_length = history_length
        self.start_time = time.time()
        
        # 原始指令数据和对应时间
        self.raw_actions = [deque(maxlen=history_length) for _ in range(14)]
        self.raw_actions_time = [deque(maxlen=history_length) for _ in range(14)]
        
        # 滤波指令数据和对应时间
        self.filtered_actions = [deque(maxlen=history_length) for _ in range(14)]
        self.filtered_actions_time = [deque(maxlen=history_length) for _ in range(14)]
        
        # 实际位置数据和对应时间
        self.actual_positions = [deque(maxlen=history_length*4) for _ in range(14)]
        self.actual_positions_time = [deque(maxlen=history_length*4) for _ in range(14)]
        
        # 线程安全锁
        self.lock = threading.Lock()
        
        # 最新关节位置数据
        self.latest_left_joint_positions = np.zeros(7)
        self.latest_right_joint_positions = np.zeros(7)
        
        # 关节名称（仅用于显示标签）
        self.joint_names = [
            'Left Shoulder Pan', 'Left Shoulder Lift', 'Left Elbow', 
            'Left Wrist 1', 'Left Wrist 2', 'Left Wrist 3', 'Left Gripper',
            'Right Shoulder Pan', 'Right Shoulder Lift', 'Right Elbow', 
            'Right Wrist 1', 'Right Wrist 2', 'Right Wrist 3', 'Right Gripper'
        ]
        
        # 设置ROS订阅器
        self.setup_subscribers()
        
        # 绘图设置
        self.setup_plots(update_interval)
        
        print("Joint Action Visualizer initialized with ROS topic subscription")
        print("Subscribed to topics:")
        print("  - Raw commands: /joint_commands/raw")
        print("  - Filtered commands: /joint_commands/filtered")
        print("  - Left arm position: /puppet/joint_left")
        print("  - Right arm position: /puppet/joint_right")
        print("Data storage based on position index, not joint names")
    
    def setup_subscribers(self):
        """设置ROS话题订阅器"""
        # 订阅原始关节指令
        rospy.Subscriber("/joint_commands/raw", Float64MultiArray, self.raw_command_callback)
        
        # 订阅滤波后关节指令
        rospy.Subscriber("/joint_commands/filtered", Float64MultiArray, self.filtered_command_callback)
        
        # 订阅实际关节位置
        rospy.Subscriber("/puppet/joint_left", JointState, self.puppet_arm_left_callback)
        
        
        rospy.Subscriber("/puppet/joint_right", JointState, self.puppet_arm_right_callback)
    
    def raw_command_callback(self, msg):
        """原始指令回调函数"""
        current_time = time.time() - self.start_time  # 相对时间
        
        with self.lock:
            if len(msg.data) == 14:  # 确保数据长度正确
                for i in range(14):
                    self.raw_actions[i].append(msg.data[i])
                    self.raw_actions_time[i].append(current_time)
    
    def filtered_command_callback(self, msg):
        """滤波指令回调函数 - 修复版本"""
        current_time = time.time() - self.start_time  # 相对时间
        
        with self.lock:
            # 添加更详细的数据验证
            if hasattr(msg, 'data') and len(msg.data) == 14:
                for i in range(14):
                    self.filtered_actions[i].append(float(msg.data[i]))
                    self.filtered_actions_time[i].append(current_time)
            else:
                # 添加调试输出
                rospy.logwarn(f"Filtered command format error: {len(msg.data) if hasattr(msg, 'data') else 'no data'}")
        
    def puppet_arm_left_callback(self, msg):
        """左臂关节位置回调函数 - 基于位置索引"""
        current_time = time.time() - self.start_time  # 相对时间
        
        with self.lock:
            # 假设左臂有7个关节，按顺序存储到索引0-6
            num_joints = min(7, len(msg.position))
            for i in range(num_joints):
                self.actual_positions[i].append(msg.position[i])
                self.actual_positions_time[i].append(current_time)
            
            # 更新最新位置数据
            if len(msg.position) >= 7:
                self.latest_left_joint_positions = msg.position[:7]
    
    def puppet_arm_right_callback(self, msg):
        """右臂关节位置回调函数 - 基于位置索引"""
        current_time = time.time() - self.start_time  # 相对时间
        
        with self.lock:
            # 假设右臂有7个关节，按顺序存储到索引7-13
            num_joints = min(7, len(msg.position))
            for i in range(num_joints):
                idx = i + 7  # 右臂关节从索引7开始
                self.actual_positions[idx].append(msg.position[i])
                self.actual_positions_time[idx].append(current_time)
            
            # 更新最新位置数据
            if len(msg.position) >= 7:
                self.latest_right_joint_positions = msg.position[:7]
    
    def setup_plots(self, update_interval):
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['axes.labelsize'] = 14
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['legend.fontsize'] = 12
        plt.rcParams['xtick.labelsize'] = 12
        plt.rcParams['ytick.labelsize'] = 12

        self.fig, self.axes = plt.subplots(4, 4, figsize=(15, 10))
        self.fig.suptitle('Real-time Joint Action Monitoring', fontsize=16, fontweight='bold', fontname='Times New Roman')
        self.lines = []

        self.axes = self.axes.flatten()
        for i in range(14):
            ax = self.axes[i]
            ax.set_title(self.joint_names[i], fontdict={'fontsize': 13, 'fontweight': 'bold', 'fontname': 'Times New Roman'})
            ax.set_xlim(0, 10)
            ax.set_ylim(-3.14, 3.14)
            ax.grid(True, which='major', linestyle='--', alpha=0.5)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Position (rad/mm)')

            # 曲线样式优化
            line_raw, = ax.plot([], [], 'r-', label='Raw Cmd', linewidth=2.0, alpha=0.8)
            line_filtered, = ax.plot([], [], 'g--', label='Filtered Cmd', linewidth=2.0)
            line_actual, = ax.plot([], [], 'b-', label='Actual Pos', linewidth=2.5)
            self.lines.append((line_raw, line_filtered, line_actual))

            if i == 0:
                ax.legend(loc='upper right', frameon=False, fontsize=12)

        for i in range(14, 16):
            self.axes[i].set_visible(False)

        self.animation = FuncAnimation(
            self.fig, self.update_plot, 
            interval=update_interval, 
            blit=False,
            save_count=self.history_length,
            cache_frame_data=False
        )
        self.fig.canvas.mpl_connect('close_event', self.on_close)
    
    def update_plot(self, frame):
        """更新图表 - 修复版本"""
        with self.lock:
            updated_lines = []
            current_time = time.time() - self.start_time  # 获取当前相对时间
            
            for i in range(14):
                line_raw, line_filtered, line_actual = self.lines[i]
                
                # 更新各条线的数据
                if self.raw_actions_time[i]:
                    line_raw.set_data(self.raw_actions_time[i], self.raw_actions[i])
                
                if self.filtered_actions_time[i]:
                    line_filtered.set_data(self.filtered_actions_time[i], self.filtered_actions[i])
                
                if self.actual_positions_time[i]:
                    line_actual.set_data(self.actual_positions_time[i], self.actual_positions[i])
                
                # 动态调整Y轴范围
                all_data = []
                if self.raw_actions[i]:
                    all_data.extend(self.raw_actions[i])
                if self.filtered_actions[i]:
                    all_data.extend(self.filtered_actions[i])
                if self.actual_positions[i]:
                    all_data.extend(self.actual_positions[i])
                
                if all_data:
                    min_val, max_val = min(all_data), max(all_data)
                    margin = (max_val - min_val) * 0.1
                    self.axes[i].set_ylim(min_val - margin, max_val + margin)
                
                # 修复X轴滚动逻辑 - 使用相对时间而不是绝对时间
                all_times = []
                if self.raw_actions_time[i]:
                    all_times.extend(self.raw_actions_time[i])
                if self.filtered_actions_time[i]:
                    all_times.extend(self.filtered_actions_time[i])
                if self.actual_positions_time[i]:
                    all_times.extend(self.actual_positions_time[i])
                
                if all_times:
                    # 确保使用当前时间作为参考点
                    max_display_time = max(all_times)
                    min_display_time = max(0, max_display_time - 5)  # 显示最近10秒
                    self.axes[i].set_xlim(min_display_time, max_display_time)
                
                updated_lines.extend([line_raw, line_filtered, line_actual])
            self.fig.canvas.draw_idle()  # 强制刷新
            return updated_lines
    
    def on_close(self, event):
        """处理窗口关闭事件"""
        try:
            self.animation.event_source.stop()
        except:
            pass
    
    def show(self):
        """显示图表（阻塞主线程）"""
        plt.tight_layout()
        plt.show()
    
    def run_in_background(self):
        """在后台线程中运行可视化"""
        def run():
            plt.tight_layout()
            plt.show()
        
        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()
        return self

# 使用示例
if __name__ == "__main__":
    # 创建可视化器
    visualizer = JointActionVisualizer()
    
    # 显示图表（阻塞式）
    visualizer.show()