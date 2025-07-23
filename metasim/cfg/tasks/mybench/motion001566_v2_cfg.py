from metasim.cfg.tasks import BaseTaskCfg
from metasim.utils import configclass
from metasim.cfg.objects import RigidObjCfg

@configclass
class Motion001566V2Cfg(BaseTaskCfg):
    episode_length = 300
    objects = [
        RigidObjCfg(  # 改为 RigidObjCfg
            name="chair",
            mesh_path="models/objects/chair/chair.obj",  # 或 usd_path/urdf_path，如果可用
            default_position=(0.9598, -0.8586, 0.8487),
            default_orientation=(-0.857633, 0, 0, 0.514261),
            collision_enabled=True,  # 明确启用碰撞
            fix_base_link=True  # 如果需要固定基座
        )
    ]
    traj_filepath = "motion_data/mimic_filtered_v2/g1_priv_mimic-71-randclip-001566_v2.pkl"
    cameras = []
