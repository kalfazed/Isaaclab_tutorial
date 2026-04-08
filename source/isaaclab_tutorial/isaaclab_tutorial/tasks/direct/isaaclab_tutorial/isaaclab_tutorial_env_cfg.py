# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_tutorial.robots.jetbot import JETBOT_CONFIG
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.sensors.ray_caster import RayCasterCfg
from isaaclab.sensors.ray_caster.patterns import LidarPatternCfg


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
    # 3维度(dot, cross, forward_speed) + 5维度(雷达数据)，所以总共是8维度。
    observation_space = 8
    state_space = 0 # 0表示没有额外的状态空间，模型看到的状态空间就是observation_space定义的空间。 

    # simulation
    # sim step的频率，单位是Hz。比如说sim_step_hz=120，那么sim step就是120Hz，也就是每秒钟模拟120次物理过程。
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # robot(s)
    # 机器人配置。是一个预设好的USD文件，包含URDF信息。prim_path是机器人在场景中的路径，可以用正则表达式匹配多个机器人。(代表Isaac会在并行的环境中找到所有匹配这个路径的机器人，并且把这个配置应用到它们身上。)
    # 比如说prim_path="/World/envs/env_.*/Robot"，那么就会匹配所有路径以/World/envs/env_开头，后面跟任意字符，最后以/Robot结尾的机器人。
    # 这样就可以在场景中放置多个机器人，每个机器人都会被这个配置控制。
    robot_cfg: ArticulationCfg = JETBOT_CONFIG.replace(prim_path="/World/envs/env_.*/Robot")

    # sensor
    raycaster_cfg = RayCasterCfg(
        # Jetbot USD 根下通常没有 base_link；与 robot_cfg 一致，挂在 articulation 根上即可。
        prim_path="/World/envs/env_.*/Robot",
        # RayCaster 当前仅支持单个 mesh_prim_path；射线只与该网格求交。障碍物需多网格时要等框架支持或自行合并网格。
        mesh_prim_paths=["/World/ground"],
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.1)), # 稍微高一点
        # 使用 LidarPatternCfg 替代之前的 Horizontal 模式
        pattern_cfg=LidarPatternCfg(
            channels=1,                     # 只有一排射线（水平面）
            horizontal_fov_range=(-45.0, 45.0), # 扫描范围：左45度到右45度
            horizontal_res=22.5,            # 分辨率：90度 / (5根线-1) = 22.5度/根
            vertical_fov_range=(0.0, 0.0),  # 垂直方向不扩散
        ),
        max_distance=4.0, # 最远看4米
    )

    # obstacles
    # UsdFileCfg 对 rigid_props 调用的是 modify_rigid_body_properties：根 prim 上若没有现成的
    # RigidBodyAPI 则不会添加，Nucleus 的 cube.usd 常导致 RigidObject 初始化失败。CuboidCfg 使用
    # define_rigid_body_properties，会在根 Xform 上正确应用 RigidBodyAPI。
    obstacle_cfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Obstacle",
        spawn=sim_utils.CuboidCfg(
            size=(1.5, 1.5, 1.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
    )



    # 机器人可控制的关节名称列表。比如说jetbot有两个轮子，分别是left_wheel_joint和right_wheel_joint，那么dof_names就是["left_wheel_joint", "right_wheel_joint"]。
    # 这样模型就知道它可以控制这两个关节，输出的动作就是这两个关节的目标值。
    dof_names = ["left_wheel_joint", "right_wheel_joint"]

    # scene
    # 场景配置。num_envs是并行环境的数量，比如说num_envs=100，那么就会在场景中创建100个并行的环境. 每个环境都有一个机器人。
    # env_spacing是每个环境之间的间距，单位是米。比如说env_spacing=4.0，那么每个环境之间就有4米的距离，避免它们互相干扰。
    # replicate_physics表示是否复制物理引擎状态到每个环境，这样可以保证每个环境的物理状态是一致的。
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=100, env_spacing=4.0, replicate_physics=True)
