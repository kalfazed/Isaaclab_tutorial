# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_tutorial.robots.jetbot import JETBOT_CFG

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass


@configclass
class IsaaclabTutorialEnvCfg(DirectRLEnvCfg):
    # env
    # 控制频率，控制器每隔多少个sim step执行一次。比如说decimation=2，sim step是120Hz，那么控制频率就是60Hz
    decimation = 2 
    # 每个episode的长度，单位是秒。比如说episode_length_s=5.0，sim step是120Hz，那么每个episode就是600个sim step。之后reset环境，重新开始一个新的episode。
    episode_length_s = 5.0 

    # - spaces definition
    # 动作空间的维度，比如说动作是一个二维向量，分别是jetson对应的左轮和右轮的速度，那么action_space就是2。
    action_space = 2
    # 模型看到的状态空间的维度，比如说模型看到的是jetson的位置、速度和加速度，那么observation_space就是3。
    #(以后如果接入感知模块，可能会看到更多的状态，比如说摄像头看到的图像，那么observation_space就会更大。)
    observation_space = 3 
    state_space = 0 # 0表示没有额外的状态空间，模型看到的状态空间就是observation_space定义的空间。 

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # robot(s)
    robot_cfg: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # custom parameters/scales
    # - controllable joint
    cart_dof_name = "slider_to_cart"
    pole_dof_name = "cart_to_pole"
    # - action scale
    action_scale = 100.0  # [N]
    # - reward scales
    rew_scale_alive = 1.0
    rew_scale_terminated = -2.0
    rew_scale_pole_pos = -1.0
    rew_scale_cart_vel = -0.01
    rew_scale_pole_vel = -0.005
    # - reset states/conditions
    initial_pole_angle_range = [-0.25, 0.25]  # pole angle sample range on reset [rad]
    max_cart_pos = 3.0  # reset if cart exceeds this position [m]
