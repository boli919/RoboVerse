from metasim.cfg.tasks import BaseTaskCfg
from metasim.utils import configclass
from metasim.cfg.objects import ArticulationObjCfg

@configclass
class Motion010629V2Cfg(BaseTaskCfg):
    episode_length = 300
    objects = [
    # ArticulationObjCfg(name="chair", mesh_path="models/objects/chair/chair.obj", default_position=(-1.4693, 0.6071, 0.8487), default_orientation=(-0.494953, 0, 0, 0.86892))
]
    traj_filepath = "motion_data/mimic_filtered_v2/g1_priv_mimic-71-randclip-010629_v2.pkl"
    cameras = []
