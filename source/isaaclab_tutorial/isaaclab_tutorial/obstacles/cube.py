import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg

# 使用 CuboidCfg 而非 UsdFileCfg(cube.usd)：后者对 rigid_props 只做 modify，根 prim 无 RigidBodyAPI 时无效。
CUBE_CONFIG = RigidObjectCfg(
    prim_path="/World/envs/env_.*/Obstacle",
    spawn=sim_utils.CuboidCfg(
        size=(2.0, 2.0, 2.0),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        collision_props=sim_utils.CollisionPropertiesCfg(),
    ),
)
