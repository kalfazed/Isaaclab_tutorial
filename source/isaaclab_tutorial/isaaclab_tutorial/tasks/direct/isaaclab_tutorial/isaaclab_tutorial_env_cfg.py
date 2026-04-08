# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_tutorial.robots.jetbot import JETBOT_CONFIG

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

    # spaces definition
    # 动作空间的维度，比如说动作是一个二维向量，分别是jetson对应的左轮和右轮的速度，那么action_space就是2。
    action_space = 2
    # 模型看到的状态空间的维度
    # 世界坐标系下的速度向量（vx, vy, vz），以及角速度向量（wx, wy, wz），总共6维。加上三维的指令向量（command_vx, command_vy, command_wz），总共9维。
    #(以后如果接入感知模块，可能会看到更多的状态，比如说摄像头看到的图像，那么observation_space就会更大。)
    observation_space = 9
    state_space = 0 # 0表示没有额外的状态空间，模型看到的状态空间就是observation_space定义的空间。 

    # simulation
    # sim step的频率，单位是Hz。比如说sim_step_hz=120，那么sim step就是120Hz，也就是每秒钟模拟120次物理过程。
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # robot(s)
    # 机器人配置。是一个预设好的USD文件，包含URDF信息。prim_path是机器人在场景中的路径，可以用正则表达式匹配多个机器人。(代表Isaac会在并行的环境中找到所有匹配这个路径的机器人，并且把这个配置应用到它们身上。)
    # 比如说prim_path="/World/envs/env_.*/Robot"，那么就会匹配所有路径以/World/envs/env_开头，后面跟任意字符，最后以/Robot结尾的机器人。
    # 这样就可以在场景中放置多个机器人，每个机器人都会被这个配置控制。
    robot_cfg: ArticulationCfg = JETBOT_CONFIG.replace(prim_path="/World/envs/env_.*/Robot")
    # 机器人可控制的关节名称列表。比如说jetbot有两个轮子，分别是left_wheel_joint和right_wheel_joint，那么dof_names就是["left_wheel_joint", "right_wheel_joint"]。
    # 这样模型就知道它可以控制这两个关节，输出的动作就是这两个关节的目标值。
    dof_names = ["left_wheel_joint", "right_wheel_joint"]

    # scene
    # 场景配置。num_envs是并行环境的数量，比如说num_envs=100，那么就会在场景中创建100个并行的环境. 每个环境都有一个机器人。
    # env_spacing是每个环境之间的间距，单位是米。比如说env_spacing=4.0，那么每个环境之间就有4米的距离，避免它们互相干扰。
    # replicate_physics表示是否复制物理引擎状态到每个环境，这样可以保证每个环境的物理状态是一致的。
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=100, env_spacing=4.0, replicate_physics=True)
