from metasim.cfg.tasks import BaseTaskCfg
from metasim.utils import configclass
from metasim.cfg.objects import ArticulationObjCfg

@configclass
class Motion000472V2Cfg(BaseTaskCfg):
    episode_length = 300
    objects = [
    ArticulationObjCfg(name="chair", mesh_path="models/objects/chair/chair.obj", default_position=(-0.7073, -0.4246, 0.8487), default_orientation=(0.639517, 0, 0, 0.768777))
]
    traj_filepath = "motion_data/mimic_filtered_v2/g1_priv_mimic-71-randclip-000472_v2.pkl"
    cameras = []
