import collections
import time
from typing import Optional, List
import dm_env
import numpy as np

from examples.mobile_aloha_AgileX import constants
from examples.mobile_aloha_AgileX import robot_utils

import rospy
from std_msgs.msg import Float64MultiArray

# This is the reset position that is used by the standard Aloha runtime.
DEFAULT_RESET_POSITION = [0, -0.96, 1.16, 0, -0.3, 0]

from examples.mobile_aloha_AgileX.td_filter import MultiJointTDFilter, MultiJointLowPassFilter

class RealEnv:
    """
    Environment for real robot bi-manual manipulation
    Action space:      [left_arm_qpos (6),             # absolute joint position
                        left_gripper_positions (1),    # normalized gripper position (0: close, 1: open)
                        right_arm_qpos (6),            # absolute joint position
                        right_gripper_positions (1),]  # normalized gripper position (0: close, 1: open)
                        [vx, wz]                      # optional mobile base velocities

    Observation space: {"qpos": Concat[ left_arm_qpos (6),          # absolute joint position
                                        left_gripper_position (1),  # normalized gripper position (0: close, 1: open)
                                        right_arm_qpos (6),         # absolute joint position
                                        right_gripper_qpos (1)]     # normalized gripper position (0: close, 1: open)
                                        [vx, wz]                   # optional mobile base velocities
                        "qvel": Concat[ left_arm_qvel (6),         # absolute joint velocity (rad)
                                        left_gripper_velocity (1),  # normalized gripper velocity (pos: opening, neg: closing)
                                        right_arm_qvel (6),         # absolute joint velocity (rad)
                                        right_gripper_qvel (1)]     # normalized gripper velocity (pos: opening, neg: closing)
                        "images": {"cam_high": (480x640x3),        # h, w, c, dtype='uint8'
                                "cam_low": (480x640x3),         # h, w, c, dtype='uint8'
                                "cam_left_wrist": (480x640x3),  # h, w, c, dtype='uint8'
                                "cam_right_wrist": (480x640x3)} # h, w, c, dtype='uint8'
    """
    def __init__(self, init_node, *, reset_position: Optional[List[float]] = None, setup_robots: bool = True, control_freq_hz):
        self._reset_position = reset_position[:6] if reset_position else DEFAULT_RESET_POSITION
        self._reset_position_left0= [-0.00133514404296875, 0.00209808349609375, 0.01583099365234375, -0.032616615295410156, -0.00286102294921875, 0.00095367431640625, 0.09]
        self._reset_position_right0 = [-0.00133514404296875, 0.00438690185546875, 0.034523963928222656, -0.053597450256347656, -0.00476837158203125, -0.00209808349609375, 0.09]

       
        self.args =robot_utils.get_arguments()
        self.use_robot_base =  self.args.use_robot_base
        self.ros_operator = robot_utils.RosOperator(self.args)

        if setup_robots:
            self.setup_robots()
        
        if self.args.use_actions_filter:
            
            if self.args.filter_alg_type == "low_pass":
            
                self.left_action_joint_filter = MultiJointLowPassFilter(
                    num_joints=7,
                    cutoff_freq=3,
                    dt = 1.0/ control_freq_hz
                )
                
                self.right_action_joint_filter = MultiJointLowPassFilter(
                    num_joints=7,
                    cutoff_freq=3,
                    dt = 1.0/ control_freq_hz
                )
                
                if self.use_robot_base:
                    self.velocity_filter = MultiJointLowPassFilter(
                        num_joints=2,
                        cutoff_freq=1,
                        dt = 1.0/ control_freq_hz
                    )
            else:
                self.left_action_joint_filter = MultiJointTDFilter(
                    num_joints=7,
                    filt_r0=np.array([1,1,1,1,1,1,1]) * 100,  # 每个关节不同的速度因子
                    filt_n1=np.array([1,1,1,1,1,1,1]) * 5,  # 所有关节相同的参数
                    filt_n2=0.0,   # 所有关节相同的参数
                    dt = 1.0/ control_freq_hz
                )
            
                self.right_action_joint_filter = MultiJointTDFilter(
                    num_joints=7,
                    filt_r0=np.array([1,1,1,1,1,1,1]) * 100,  # 每个关节不同的速度因子
                    filt_n1=np.array([1,1,1,1,1,1,1]) * 5,  # 所有关节相同的参数
                    filt_n2=0.0,   # 所有关节相同的参数
                    dt = 1.0/ control_freq_hz
                )
                
                if self.use_robot_base:
                    self.velocity_filter = MultiJointTDFilter(
                        num_joints=2,
                        filt_r0=np.array([1,1]) * 100,
                        filt_n1=np.array([1,1]) * 5,
                        filt_n2=0.0,
                        dt = 1.0/ control_freq_hz
                    )
            
        self.raw_action_publisher = rospy.Publisher("/joint_commands/raw", Float64MultiArray, queue_size=100)
        self.filter_action_publisher = rospy.Publisher("/joint_commands/filtered", Float64MultiArray, queue_size=100)

    def setup_robots(self):
        return 0
    

    def build_image_dict(self,img_front: np.ndarray,
                        img_left:  np.ndarray,
                        img_right: np.ndarray) -> dict[str, np.ndarray | None]:
        """将三路 RGB 帧封装成 ImageRecorder.get_images 同格式 dict。"""
        return {
            "cam_high":           img_front,
            "cam_high_depth":     None,          # 若无深度帧可置 None
            "cam_left_wrist":     img_left,
            "cam_left_wrist_depth":  None,
            "cam_right_wrist":    img_right,
            "cam_right_wrist_depth": None,
            }
  
    def get_observation(self):
        """
        从 ROS 获取一帧同步观测并封装成 OrderedDict,
        直接替代旧版 get_observation / get_qpos / get_qvel / get_effort。
        """
        # 阻塞直到拿到一帧完整数据
        (img_front, img_left, img_right,
         puppet_arm_left, puppet_arm_right, robot_base) = robot_utils.get_ros_observation(
            self.args, self.ros_operator
        )

        # --- 关节状态 ----------------------------------------------------------
           # 关节状态
        if self.use_robot_base:
            qpos = np.concatenate(
                (np.asarray(puppet_arm_left.position),
                 np.asarray(puppet_arm_right.position),
                 np.asarray([robot_base.twist.twist.linear.x]),
                 np.asarray([robot_base.twist.twist.angular.z])),
                axis=0
            )  # shape = (16,)
        else:
            qpos = np.concatenate(
                (np.asarray(puppet_arm_left.position),
                 np.asarray(puppet_arm_right.position)),
                axis=0
            )  # shape = (14,)
            
        # shape = (14,)
        qvel   = np.concatenate(
            (np.asarray(puppet_arm_left.velocity),
             np.asarray(puppet_arm_right.velocity)),
            axis=0
        )
        effort = np.concatenate(
            (np.asarray(puppet_arm_left.effort),
             np.asarray(puppet_arm_right.effort)),
            axis=0
        )


        # --- 图像 --------------------------------------------------------------
        images = self.build_image_dict(img_front, img_left, img_right)
        # print("2")
        # --- 打包成 OrderedDict -------------------------------------------------
        obs = collections.OrderedDict()
        obs["qpos"]   = qpos
        # obs["qvel"]   = qvel
        # obs["effort"] = effort
        obs["images"] = images
        # print("3")
    
        return obs
    
    def get_reward(self):
        return 0


    def reset(self, *, fake: bool = False):
        """
        复位环境：
            1. 真实模式下重新上电并复位 gripper
            2. 将双臂平滑移动到 _reset_position_left0 / _reset_position_right0
            3. 返回 dm_env 的 FIRST TimeStep
        """
        if not fake:


            #③ 平滑移动到自定义复位姿态（包含张开的 gripper 值 3.55…）
            self.ros_operator.puppet_arm_publish_continuous(
                self._reset_position_left0,
                self._reset_position_right0
            )
            
            # 如果有移动底座，发布零速度
            if self.use_robot_base:
                self.ros_operator.robot_base_publish([0, 0])

        # 给学习框架返回 FIRST 时间步
        # return dm_env.TimeStep(
        #     step_type  = dm_env.StepType.FIRST,
        #     reward     = self.get_reward(),
        #     discount   = None,
        #     observation= self.get_observation()
        # )

    def step(self, action):
        """
        使用 print 方式输出调试信息，不依赖 logging。
        """
        # 解析动作
        if self.use_robot_base:
            left_action = action[:7]      # [arm6, grip_norm]
            right_action = action[7:14]   # [arm6, grip_norm]
            vel_action = action[14:16]    # [vx, wz]
        else:
            state_len = len(action) // 2
            left_action = action[:state_len]   # [arm6, grip_norm]
            right_action = action[state_len:]  # [arm6, grip_norm]
            vel_action = None

        # print("[STEP] raw  action :", [round(x, 3) for x in action])

        # 2) 反归一化夹爪值 -----------------------------------------------------------------
        left_arm_target  = np.array(left_action,  dtype=float)
        right_arm_target = np.array(right_action, dtype=float)
        if self.use_robot_base and vel_action is not None:
            vel_action_target = np.array(vel_action, dtype=float)
        
        # ! useful for pour water
        left_arm_target[6] =  left_arm_target[6].copy() -0.005
        right_arm_target[6] = right_arm_target[6].copy() - 0.005
      
        #! useful for adjust bottle task
        # left_arm_target[6] = tanh_smooth_map(left_arm_target[6].copy())   # Left arm gripper
        # right_arm_target[6] = tanh_smooth_map(right_arm_target[6].copy())  # Right arm gripper

        # 发布原始动作
        raw_msg = Float64MultiArray()
        raw_msg.data =  np.concatenate((left_arm_target.copy(), right_arm_target.copy()))
        self.raw_action_publisher.publish(raw_msg)
        
        # 如果使用动作滤波器
        if self.args.use_actions_filter:
            left_arm_target = self.left_action_joint_filter.update_all_joints(left_arm_target.copy())
            right_arm_target = self.right_action_joint_filter.update_all_joints(right_arm_target.copy())
            filtered_action = np.concatenate([
                left_arm_target,
                right_arm_target
            ])
            
            # 滤波移动底座速度
            if self.use_robot_base and vel_action_target is not None:
                vel_action_target = self.velocity_filter.update_all_joints(vel_action_target.copy())
           

            # 发布滤波后动作
            filtered_msg = Float64MultiArray()
            filtered_msg.data = filtered_action
            self.filter_action_publisher.publish(filtered_msg)
        
        
        # 3) 连续发布 ----------------------------------------------------------------------
        try:
            self.ros_operator.puppet_arm_publish(
                left_arm_target.tolist(),
                right_arm_target.tolist()
            )
            
            if self.use_robot_base and vel_action_target is not None:
                self.ros_operator.robot_base_publish(vel_action_target.tolist())
                
        except Exception as e:
            print("[STEP] ERROR start publish:", e)
            raise


        # 5) 返回新的 dm_env.TimeStep
        # return dm_env.TimeStep(
        #         step_type  = dm_env.StepType.MID,
        #         reward     = self.get_reward(),
        #         discount   = None,
        #         observation= self.get_observation()
        # )

def make_real_env(init_node, *, reset_position: Optional[List[float]] = None, setup_robots: bool = True, control_freq_hz) -> RealEnv:
    return RealEnv(init_node, reset_position=reset_position, setup_robots=setup_robots, control_freq_hz=control_freq_hz)


def tanh_smooth_map(x, threshold=0.05, width=0.01, low_output=0.0, high_output=0.1):
    """
    使用双曲正切(tanh)函数实现平滑映射
    
    参数:
    x: 输入值或数组
    threshold: 中心阈值（过渡中心点）
    width: 过渡区域宽度（值越小过渡越陡峭）
    low_output: 低输出值
    high_output: 高输出值
    """
    # 将输入归一化到以threshold为中心的过渡区间
    normalized = (x - threshold) / width
    
    # 使用tanh函数计算平滑过渡（范围[-1, 1]）
    tanh_result = np.tanh(normalized)
    
    # 将tanh结果从[-1, 1]映射到[low_output, high_output]
    return low_output + (tanh_result + 1) * (high_output - low_output) / 2
