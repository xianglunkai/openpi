import math
from typing import Tuple
import numpy as np
import time

class LowPassFilter:
    def __init__(self, cutoff_freq: float = 1.0, dt: float = 0.02):
        self.cutoff_freq = cutoff_freq
        self.dt = dt
       
        self.last_value = 0.0
        self.initialized = False
        self.last_time = 0.0
        
    def calc_lowpass_alpha_dt(self, dt: float, cutoff_freq: float) -> float:
        if dt < 0.0 or cutoff_freq < 0.0:
            raise ValueError("dt and cutoff_freq must be non-negative")
        if cutoff_freq == 0.0:
            return 1.0
        if dt == 0.0:
            return 0.0
        rc = 1.0 / (2 * math.pi * cutoff_freq)
        return dt / (dt + rc)
    
    def reset(self, value: float = 0.0):
        self.last_value = value
        self.initialized = True
        self.last_time = time.perf_counter()
        
    def apply(self, value: float) -> float:
        if not self.initialized:
            self.reset(value)
            return value
            
        now = time.perf_counter()
        dt = now - self.last_time
        if dt <= 0.0 or dt > 1.0:
            self.reset(value)
            return value
            
        alpha = self.calc_lowpass_alpha_dt(dt, self.cutoff_freq)    
        self.last_value = alpha * value + (1.0 - alpha) * self.last_value
        
        self.last_time = now
        return self.last_value


class MultiJointLowPassFilter:
    """
    多关节滤波器封装类，支持同时处理多个关节的滤波
    
    Attributes:
        num_joints (int): 关节数量
        filters (list): 每个关节的TDFilter实例列表
    """
    
    def __init__(self, num_joints: int = 7, **filter_params):

        self.num_joints = num_joints
        
        # 为每个关节创建独立的滤波器
        self.filters = []
        for i in range(num_joints):
            # 为每个关节提取特定参数（如果提供了每关节参数）
            joint_params = {}
            for key, value in filter_params.items():
                if isinstance(value, (list, np.ndarray)) and len(value) == num_joints:
                    joint_params[key] = value[i]  # 使用该关节的特定参数
                else:
                    joint_params[key] = value     # 使用通用参数
            
            self.filters.append(LowPassFilter(**joint_params))
    
    def update_all_joints(self, joint_values: np.ndarray) -> np.ndarray:

        if len(joint_values) != self.num_joints:
            raise ValueError(f"Expected {self.num_joints} joint values, got {len(joint_values)}")
        
        filtered_values = np.zeros(self.num_joints)
        
        for i in range(self.num_joints):
            v1 = self.filters[i].apply(joint_values[i])
            filtered_values[i] = v1
        
        return filtered_values
    
    def reset_all_filters(self, initial_values: np.ndarray = None):
        """
        重置所有关节的滤波器状态
        
        Args:
            initial_values: 各关节的初始值，形状为 (num_joints,)
        """
        if initial_values is None:
            initial_values = np.zeros(self.num_joints)
        
        for i, filter_obj in enumerate(self.filters):
            filter_obj.reset(initial_values[i])
    
    
    def set_joint_parameters(self, joint_idx: int, **params):
        """
        设置特定关节的滤波器参数
        
        Args:
            joint_idx: 关节索引（0-based）
            params: 要更新的参数
        """
        if 0 <= joint_idx < self.num_joints:
            self.filters[joint_idx].set_parameters(**params)
        else:
            raise ValueError(f"Joint index {joint_idx} out of range [0, {self.num_joints-1}]")

class TDFilter:
    """
    跟踪微分器（TD）非线性滤波类
    
    基于自抗扰控制（ADRC）理论中的最速控制综合函数（fhan）和非线性函数（fal），
    用于从带噪声的信号中提取平滑的跟踪信号及其微分信号。
    
    Attributes:
        v1 (float): 跟踪状态（滤波后的信号）
        v2 (float): 微分状态（信号的变化率）
        vp (float): 预测状态
        last_filter_ms (int): 上次滤波时间（毫秒）
        _params (dict): 滤波器参数字典
    """
    
    def __init__(self, filt_r0: float = 1.0, filt_n1: float = 1.0, filt_n2: float = 1.0, dt: float = 0.02):
        """
        初始化TD滤波器
        
        Args:
            filt_r0: 速度因子（越大跟踪越快，但可能带来超调）
            filt_n1: 时间步长缩放因子1
            filt_n2: 时间步长缩放因子2
        """
        # 滤波器状态初始化
        self.v1 = 0.0      # 跟踪状态
        self.v2 = 0.0      # 微分状态
        self.vp = 0.0      # 预测状态
        self.last_time = 0  # 上次滤波时间
        self.dt = dt
        
        # 滤波器参数
        self._params = {
            'filt_r0': filt_r0,
            'filt_n1': filt_n1,
            'filt_n2': filt_n2
        }
    
    @staticmethod
    def sign(val: float) -> float:
        """
        符号函数
        
        Args:
            val: 输入值
            
        Returns:
            float: 1.0 (val > 0), -1.0 (val < 0), 0.0 (val == 0)
        """
        if val > 0.0:
            return 1.0
        elif val < 0.0:
            return -1.0
        else:
            return 0.0
    
    @staticmethod
    def fhan(v1: float, v2: float, r0: float, h0: float) -> float:
        """
        最速控制综合函数（Fast Hyperbolic Absolute Value Nonlinear Function）
        
        Args:
            v1: 跟踪状态
            v2: 微分状态
            r0: 速度因子
            h0: 时间步长
            
        Returns:
            float: 控制量
        """
        d = h0 * h0 * r0
        a0 = h0 * v2
        y = v1 + a0
        a1 = math.sqrt(d * (d + 8.0 * abs(y)))
        a2 = a0 + TDFilter.sign(y) * (a1 - d) * 0.5
        sy = (TDFilter.sign(y + d) - TDFilter.sign(y - d)) * 0.5
        a = (a0 + y - a2) * sy + a2
        sa = (TDFilter.sign(a + d) - TDFilter.sign(a - d)) * 0.5
        
        return -r0 * (a / d - TDFilter.sign(a)) * sa - r0 * TDFilter.sign(a)
    
    @staticmethod
    def fal(e: float, alpha: float, delta: float) -> float:
        """
        非线性函数（Nonlinear Function）
        
        Args:
            e: 误差
            alpha: 非线性指数（0 < alpha < 1）
            delta: 线性区间宽度
            
        Returns:
            float: 非线性输出
        """
        if abs(delta) < 1e-6:  # 相当于 is_zero(delta)
            return e
            
        if abs(e) < delta:
            return e / (pow(delta, 1.0 - alpha))
        else:
            return pow(abs(e), alpha) * TDFilter.sign(e)
    
    def update_filter(self, current_value: float) -> float:
        """
        更新滤波器状态
        
        Args:
            current_value: 当前测量值
            current_time_ms: 当前时间（毫秒）
            
        Returns:
            Tuple[float, float]: (滤波后的值, 微分值)
        """
      
        # 检查是否不需要滤波
        if abs(self._params['filt_r0']) < 1e-6:
            self.vp = current_value
            return self.vp
        
        current_time = time.perf_counter()
        h0 = current_time - self.last_time
        
   
        # 超时或首次运行重置滤波器
        if self.last_time == 0 :
            print("首次运行重置滤波器")
            self.reset_filter(current_value)
            return current_value

        # 计算时间步长
        if h0 <= 1e-6 or h0 > 1.0:
            print("时间步长异常，使用默认时间步长")
            h0 = self.dt  # 使用默认时间步长
            
        # 计算缩放后的时间步长
        h1 = self._params['filt_n1'] * h0
        h2 = self._params['filt_n2'] * h0
        
        # 更新滤波器状态
        fh = self.fhan(self.v1 - current_value, self.v2, self._params['filt_r0'], h1)
        self.v1 += h0 * self.v2
        self.v2 += h0 * fh
        self.vp = self.v1 + h2 * self.v2
        
        self.last_time = current_time
        
        return self.vp
    
    def reset_filter(self, initial_value: float = 0.0):
        """
        重置滤波器状态
        
        Args:
            initial_value: 初始值
        """
        self.v1 = initial_value
        self.v2 = 0.0
        self.vp = 0.0
        self.last_time = time.perf_counter()

    
    def set_parameters(self, filt_r0: float = None, filt_n1: float = None, filt_n2: float = None):
        """
        设置滤波器参数
        
        Args:
            filt_r0: 速度因子
            filt_n1: 时间步长缩放因子1
            filt_n2: 时间步长缩放因子2
        """
        if filt_r0 is not None:
            self._params['filt_r0'] = filt_r0
        if filt_n1 is not None:
            self._params['filt_n1'] = filt_n1
        if filt_n2 is not None:
            self._params['filt_n2'] = filt_n2




class MultiJointTDFilter:
    """
    多关节滤波器封装类，支持同时处理多个关节的滤波
    
    Attributes:
        num_joints (int): 关节数量
        filters (list): 每个关节的TDFilter实例列表
    """
    
    def __init__(self, num_joints: int = 6, **filter_params):
        """
        初始化多关节滤波器
        
        Args:
            num_joints: 关节数量
            filter_params: 滤波器参数，可以是标量或列表
        """
        self.num_joints = num_joints
        
        # 为每个关节创建独立的滤波器
        self.filters = []
        for i in range(num_joints):
            # 为每个关节提取特定参数（如果提供了每关节参数）
            joint_params = {}
            for key, value in filter_params.items():
                if isinstance(value, (list, np.ndarray)) and len(value) == num_joints:
                    joint_params[key] = value[i]  # 使用该关节的特定参数
                else:
                    joint_params[key] = value     # 使用通用参数
            
            self.filters.append(TDFilter(**joint_params))
    
    def update_all_joints(self, joint_values: np.ndarray) -> np.ndarray:
        """
        同时更新所有关节的滤波器状态
        
        Args:
            joint_values: 当前各关节的测量值，形状为 (num_joints,)
            current_time_ms: 当前时间（毫秒）
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: 
                - 滤波后的关节值，形状 (num_joints,)
                - 关节微分值，形状 (num_joints,)
        """
        if len(joint_values) != self.num_joints:
            raise ValueError(f"Expected {self.num_joints} joint values, got {len(joint_values)}")
        
        filtered_values = np.zeros(self.num_joints)
        
        for i in range(self.num_joints):
            v1= self.filters[i].update_filter(joint_values[i])
            filtered_values[i] = v1
        
        return filtered_values
    
    def reset_all_filters(self, initial_values: np.ndarray = None):
        """
        重置所有关节的滤波器状态
        
        Args:
            initial_values: 各关节的初始值，形状为 (num_joints,)
        """
        if initial_values is None:
            initial_values = np.zeros(self.num_joints)
        
        for i, filter_obj in enumerate(self.filters):
            filter_obj.reset_filter(initial_values[i])

    
    def set_joint_parameters(self, joint_idx: int, **params):
        """
        设置特定关节的滤波器参数
        
        Args:
            joint_idx: 关节索引（0-based）
            params: 要更新的参数
        """
        if 0 <= joint_idx < self.num_joints:
            self.filters[joint_idx].set_parameters(**params)
        else:
            raise ValueError(f"Joint index {joint_idx} out of range [0, {self.num_joints-1}]")


# 使用示例
if __name__ == "__main__":
    # 创建6关节滤波器，可以为每个关节设置不同的参数
    multi_filter = MultiJointTDFilter(
        num_joints=6,
        filt_r0=[1.0, 1.2, 0.8, 1.0, 1.1, 0.9],  # 每个关节不同的速度因子
        filt_n1=1.0,  # 所有关节相同的参数
        filt_n2=1.0   # 所有关节相同的参数
    )
    
    # 模拟数据：100个时间步，6个关节
    num_time_steps = 100
    joint_data = np.random.rand(num_time_steps, 6)

    
    # 存储结果
    all_filtered = np.zeros((num_time_steps, 6))
    all_derivatives = np.zeros((num_time_steps, 6))
    
    # 滤波处理
    for t in range(num_time_steps):
        filtered = multi_filter.update_all_joints(joint_data[t])
        all_filtered[t] = filtered
      
    
    print("滤波完成！")
    print(f"最终关节值: {all_filtered[-1]}")
 