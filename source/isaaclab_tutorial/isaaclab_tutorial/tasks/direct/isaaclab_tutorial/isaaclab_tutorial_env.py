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


class IsaaclabTutorialEnv(DirectRLEnv):
    cfg: IsaaclabTutorialEnvCfg

    def __init__(self, cfg: IsaaclabTutorialEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.dof_idx, _ = self.robot.find_joints(self.cfg.dof_name)

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

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # 将动作保存到对象的属性中，之后在_apply_action方法中使用这个动作来控制机器人。动作是一个张量，包含了每个环境的动作值。
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        # 将动作应用到机器人上，控制机器人的关节。
        # 这里使用set_joint_velocity_target方法，将动作作为关节的速度目标值，机器人会根据这个目标值来调整关节的速度。
        # joint_ids参数指定了要控制的关节的索引，这些索引是在初始化时根据cfg.dof_names找到的。
        self.robot.set_joint_velocity_target(self.actions, joint_ids=self.dof_idx)

    def _get_observations(self) -> dict:
        # 获取机器人的状态，作为模型看到的状态空间。这里我们获取了机器人的根部线速度，作为观察值的一部分。
        self.velocity = self.robot.data.root_com_line_vel_b
        observations = {"policy": self.velocity}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        # 计算奖励。这里我们简单地使用机器人的速度的范数作为奖励，鼓励机器人移动得更快。
        # 你可以根据任务的需求设计更复杂的奖励函数，比如说根据机器人的位置、姿态或者与环境的交互来计算奖励。
        total_reward = torch.linalg.norm(self.velocity, dim=-1, keepdim=True)
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

        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.robot.write_root_state_to_sim(default_root_state, env_ids)
