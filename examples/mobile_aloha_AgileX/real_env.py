import collections
import time
from typing import Optional, List
import dm_env
import numpy as np

from examples.mobile_aloha_AgileX import constants
from examples.mobile_aloha_AgileX import robot_utils

# This is the reset position that is used by the standard Aloha runtime.
DEFAULT_RESET_POSITION = [0, -0.96, 1.16, 0, -0.3, 0]


class RealEnv:
    """
    Environment for real robot bi-manual manipulation
    Action space:      [left_arm_qpos (6),             # absolute joint position
                        left_gripper_positions (1),    # normalized gripper position (0: close, 1: open)
                        right_arm_qpos (6),            # absolute joint position
                        right_gripper_positions (1),]  # normalized gripper position (0: close, 1: open)

    Observation space: {"qpos": Concat[ left_arm_qpos (6),          # absolute joint position
                                        left_gripper_position (1),  # normalized gripper position (0: close, 1: open)
                                        right_arm_qpos (6),         # absolute joint position
                                        right_gripper_qpos (1)]     # normalized gripper position (0: close, 1: open)
                        "qvel": Concat[ left_arm_qvel (6),         # absolute joint velocity (rad)
                                        left_gripper_velocity (1),  # normalized gripper velocity (pos: opening, neg: closing)
                                        right_arm_qvel (6),         # absolute joint velocity (rad)
                                        right_gripper_qvel (1)]     # normalized gripper velocity (pos: opening, neg: closing)
                        "images": {"cam_high": (480x640x3),        # h, w, c, dtype='uint8'
                                   "cam_low": (480x640x3),         # h, w, c, dtype='uint8'
                                   "cam_left_wrist": (480x640x3),  # h, w, c, dtype='uint8'
                                   "cam_right_wrist": (480x640x3)} # h, w, c, dtype='uint8'
    """

    def __init__(self, init_node, *, reset_position: Optional[List[float]] = None, setup_robots: bool = True):
        self._reset_position = reset_position[:6] if reset_position else DEFAULT_RESET_POSITION
        self._reset_position_left0= [-0.00133514404296875, 0.00209808349609375, 0.01583099365234375, -0.032616615295410156, -0.00286102294921875, 0.00095367431640625, 0]
        self._reset_position_right0 = [-0.00133514404296875, 0.00438690185546875, 0.034523963928222656, -0.053597450256347656, -0.00476837158203125, -0.00209808349609375, 0]

        self.args =robot_utils.get_arguments()
        self.ros_operator = robot_utils.RosOperator(self.args)

        if setup_robots:
            self.setup_robots()

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
         puppet_arm_left, puppet_arm_right) = robot_utils.get_ros_observation(
            self.args, self.ros_operator
        )

        # --- 关节状态 ----------------------------------------------------------
        qpos   = np.concatenate(
            (np.asarray(puppet_arm_left.position),
             np.asarray(puppet_arm_right.position)),
            axis=0
        )                             # shape = (14,)
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

        # 如果仍然希望把两个夹爪做归一化，可在这里额外处理：
        # qpos[6]  和 qpos[13]  分别是左右夹爪
        # qvel[6]  和 qvel[13] 同理
        # qpos[6]  = constants.PUPPET_GRIPPER_POSITION_NORMALIZE_FN(qpos[6])
        # qpos[13] = constants.PUPPET_GRIPPER_POSITION_NORMALIZE_FN(qpos[13])
        # qvel[6]  = constants.PUPPET_GRIPPER_POSITION_NORMALIZE_FN(qvel[6])
        # qvel[13] = constants.PUPPET_GRIPPER_POSITION_NORMALIZE_FN(qvel[13])
 

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

        # 给学习框架返回 FIRST 时间步
        return dm_env.TimeStep(
            step_type  = dm_env.StepType.FIRST,
            reward     = self.get_reward(),
            discount   = None,
            observation= self.get_observation()
        )

    def step(self, action):
        """
        使用 print 方式输出调试信息，不依赖 logging。
        """
        # 1) 拆左右 7 维动作 ----------------------------------------------------------------
        state_len   = len(action) // 2
        left_action = action[:state_len]         # [arm6, grip_norm]
        right_action = action[state_len:]        # [arm6, grip_norm]

        print("[STEP] raw  action :", [round(x, 3) for x in action])

        # 2) 反归一化夹爪值 -----------------------------------------------------------------
        left_arm_target  = np.array(left_action,  dtype=float)
        right_arm_target = np.array(right_action, dtype=float)

        # left_arm_target[-1]  = constants.PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN(left_arm_target[-1])
        # right_arm_target[-1] = constants.PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN(right_arm_target[-1])

        print("[STEP] left target :", [round(x, 3) for x in left_arm_target])
        print("[STEP] right target:", [round(x, 3) for x in right_arm_target])

        # 3) 连续发布 ----------------------------------------------------------------------
        try:
            self.ros_operator.puppet_arm_publish_continuous(
                left_arm_target.tolist(),
                right_arm_target.tolist()
            )
            print("[STEP] publish thread started OK")
        except Exception as e:
            print("[STEP] ERROR start publish:", e)
            raise

        # 4) 控制周期同步 -------------------------------------------------------------------
        time.sleep(constants.DT)

            # 5) 返回新的 dm_env.TimeStep
        return dm_env.TimeStep(
                step_type  = dm_env.StepType.MID,
                reward     = self.get_reward(),
                discount   = None,
                observation= self.get_observation()
        )


def get_action(master_bot_left, master_bot_right):
    action = np.zeros(14)  # 6 joint + 1 gripper, for two arms
    # Arm actions
    action[:6] = master_bot_left.dxl.joint_states.position[:6]
    action[7 : 7 + 6] = master_bot_right.dxl.joint_states.position[:6]
    # Gripper actions
    action[6] = constants.MASTER_GRIPPER_JOINT_NORMALIZE_FN(master_bot_left.dxl.joint_states.position[6])
    action[7 + 6] = constants.MASTER_GRIPPER_JOINT_NORMALIZE_FN(master_bot_right.dxl.joint_states.position[6])

    return action


def make_real_env(init_node, *, reset_position: Optional[List[float]] = None, setup_robots: bool = True) -> RealEnv:
    return RealEnv(init_node, reset_position=reset_position, setup_robots=setup_robots)
