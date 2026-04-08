# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform

from .isaaclab_tutorial_env_cfg import IsaaclabTutorialEnvCfg

from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import isaaclab.utils.math as math_utils

def define_markers() -> VisualizationMarkers:
    """Define markers with various different shapes."""
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/myMarkers",
        markers={
                "forward": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                    scale=(0.25, 0.25, 0.5),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 1.0)),
                ),
                "command": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                    scale=(0.25, 0.25, 0.5),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                ),
        },
    )
    return VisualizationMarkers(cfg=marker_cfg)


class IsaaclabTutorialEnv(DirectRLEnv):
    cfg: IsaaclabTutorialEnvCfg

    def __init__(self, cfg: IsaaclabTutorialEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.dof_idx, _ = self.robot.find_joints(self.cfg.dof_names)

    def _setup_scene(self):
        # 添加一个机器人到场景中，机器人是一个Articulation对象，包含了机器人的URDF信息和物理属性。机器人会根据cfg.robot_cfg的配置在场景中生成。
        self.robot = Articulation(self.cfg.robot_cfg)
        # 添加一个地面到场景中，地面是一个平面，包含了地面的物理属性。地面会根据cfg.ground_plane_cfg的配置在场景中生成。
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # 复制环境，生成多个并行的环境，每个环境都有一个机器人。复制环境会根据cfg.scene_cfg的配置进行，比如说复制的数量和间距。
        self.scene.clone_environments(copy_from_source=False)
        # 将机器人添加到场景中，机器人会根据复制的环境数量生成多个实例，每个实例都会被这个机器人对象控制。
        # 机器人对象会在每个环境中找到对应的实例，并且把它们的状态和动作同步到这个对象上。
        self.scene.articulations["robot"] = self.robot
        # 设置光照，创建一个半球光，模拟自然光照。光照会根据cfg.light_cfg的配置进行，比如说强度和颜色。
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        self.visualization_markers = define_markers()

        # 这里我们预先计算了一些变量，这些变量在后续的步骤中会用到。比如说up_dir表示上方向的单位向量，yaws表示每个环境中命令的偏航角，commands表示每个环境中的目标方向命令。
        self.up_dir = torch.tensor([0.0, 0.0, 1.0]).cuda()
        self.yaws = torch.zeros((self.cfg.scene.num_envs, 1)).cuda()
        self.commands = torch.randn((self.cfg.scene.num_envs, 3)).cuda()
        # 将命令的z分量设置为0，并且将命令向量归一化，确保它们只表示水平平面上的方向。
        self.commands[:,-1] = 0.0
        self.commands = self.commands/torch.linalg.norm(self.commands, dim=1, keepdim=True)

        # offsets to account for atan range and keep things on [-pi, pi]
        ratio = self.commands[:,1]/(self.commands[:,0]+1E-8)
        gzero = torch.where(self.commands > 0, True, False)
        lzero = torch.where(self.commands < 0, True, False)
        plus = lzero[:,0]*gzero[:,1]
        minus = lzero[:,0]*lzero[:,1]
        offsets = torch.pi*plus - torch.pi*minus
        self.yaws = torch.atan(ratio).reshape(-1,1) + offsets.reshape(-1,1)

        self.marker_locations = torch.zeros((self.cfg.scene.num_envs, 3)).cuda()
        self.marker_offset = torch.zeros((self.cfg.scene.num_envs, 3)).cuda()
        self.marker_offset[:,-1] = 0.5
        self.forward_marker_orientations = torch.zeros((self.cfg.scene.num_envs, 4)).cuda()
        self.command_marker_orientations = torch.zeros((self.cfg.scene.num_envs, 4)).cuda()

    def _visualize_markers(self):
        # 获取机器人当前的位置和姿态
        self.marker_locations = self.robot.data.root_pos_w
        self.forward_marker_orientations = self.robot.data.root_quat_w
        # 将偏航角转换为四元数
        self.command_marker_orientations = math_utils.quat_from_angle_axis(self.yaws, self.up_dir).squeeze()

        # 设置标记高度偏置（放在小车上方）
        loc = self.marker_locations + self.marker_offset
        loc = torch.vstack((loc, loc)) # 合并位置张量
        rots = torch.vstack((self.forward_marker_orientations, self.command_marker_orientations))

        # 渲染标记
        all_envs = torch.arange(self.cfg.scene.num_envs)
        indices = torch.hstack((torch.zeros_like(all_envs), torch.ones_like(all_envs)))
        self.visualization_markers.visualize(loc, rots, marker_indices=indices)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # 将动作保存到对象的属性中，之后在_apply_action方法中使用这个动作来控制机器人。动作是一个张量，包含了每个环境的动作值。
        self.actions = actions.clone()
        self._visualize_markers()

    def _apply_action(self) -> None:
        # 将动作应用到机器人上，控制机器人的关节。
        # 这里使用set_joint_velocity_target方法，将动作作为关节的速度目标值，机器人会根据这个目标值来调整关节的速度。
        # joint_ids参数指定了要控制的关节的索引，这些索引是在初始化时根据cfg.dof_names找到的。
        self.robot.set_joint_velocity_target(self.actions, joint_ids=self.dof_idx)

    def _get_observations(self) -> dict:
        # 获取机器人的状态，作为模型看到的状态空间。这里我们获取了机器人的根部线速度，作为观察值的一部分。
        self.velocity = self.robot.data.root_com_vel_w # 六维度: 线速度 + 角速度
        # 计算机器人在世界坐标系下前进的方向
        self.forwards = math_utils.quat_apply(self.robot.data.root_link_quat_w, self.robot.data.FORWARD_VEC_B)

        # 这里我们设计了一个简单的观察值
        # 只包含了前进方向与指令方向的点积（表示对齐程度），以及它们的叉积（表示偏离方向），
        # 还有机器人在自身坐标系下X轴的速度（表示前进速度）。
        # 这样模型就可以根据这个观察值来判断自己是否在朝着指令的方向前进，以及前进的速度如何。
        # 这个相比于直接把速度和指令作为观察值，更加抽象和简洁，模型需要自己学会从这些信息中推断出正确的行为。
        dot = torch.sum(self.forwards * self.commands, dim=-1, keepdim=True)
        cross = torch.cross(self.forwards, self.commands, dim=-1)[:,-1].reshape(-1,1)
        forward_speed = self.robot.data.root_com_lin_vel_b[:,0].reshape(-1,1)
        obs = torch.hstack((dot, cross, forward_speed))

        observations = {"policy": obs}

        return observations

    def _get_rewards(self) -> torch.Tensor:
        # 计算奖励。这里我们设计了一个奖励函数，鼓励机器人在前进的同时，朝着指令的方向移动。
        # 机器人在自身坐标系下 X 轴的速度
        forward_reward = self.robot.data.root_com_lin_vel_b[:,0].reshape(-1,1)
        # 前进向量与指令向量的点积（对齐度）
        alignment_reward = torch.sum(self.forwards * self.commands, dim=-1, keepdim=True)
        # 逻辑“或”操作，当机器人要么在前进，要么在对齐时，都能得到奖励。这样模型就可以学会两者兼顾，既要前进又要对齐。
        # 这样机器人有的时候可能会牺牲一些前进速度来更好地对齐指令方向，或者牺牲一些对齐度来获得更高的前进速度。总之，这个奖励函数鼓励机器人在这两者之间找到一个平衡点。
        # total_reward = forward_reward + alignment_reward

        # 逻辑"与"操作，当机器人既在前进又在对齐时，才能得到奖励。这样模型就必须同时满足这两个条件，才能获得奖励。
        total_reward = forward_reward * alignment_reward
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # 计算done信号，表示一个episode是否结束。这里我们简单地使用一个时间限制，当episode的长度超过最大值时，就结束这个episode。
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return False, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        # 重置环境。这里我们将机器人的根部状态重置到默认状态，并且根据环境的原点位置进行偏移，确保每个环境中的机器人都在正确的位置。
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        # pick new commands for reset envs
        self.commands[env_ids] = torch.randn((len(env_ids), 3)).cuda()
        self.commands[env_ids,-1] = 0.0
        self.commands[env_ids] = self.commands[env_ids]/torch.linalg.norm(self.commands[env_ids], dim=1, keepdim=True)

        # recalculate the orientations for the command markers with the new commands
        ratio = self.commands[env_ids][:,1]/(self.commands[env_ids][:,0]+1E-8)
        gzero = torch.where(self.commands[env_ids] > 0, True, False)
        lzero = torch.where(self.commands[env_ids]< 0, True, False)
        plus = lzero[:,0]*gzero[:,1]
        minus = lzero[:,0]*lzero[:,1]
        offsets = torch.pi*plus - torch.pi*minus
        self.yaws[env_ids] = torch.atan(ratio).reshape(-1,1) + offsets.reshape(-1,1)

        # set the root state for the reset envs
        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.robot.write_root_state_to_sim(default_root_state, env_ids)
        self._visualize_markers()
