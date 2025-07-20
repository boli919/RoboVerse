from metasim.cfg.tasks import BaseTaskCfg
from metasim.utils import configclass
from metasim.cfg.objects import ArticulationObjCfg

@configclass
class Motion011645V2Cfg(BaseTaskCfg):
    episode_length = 300
    objects = [
    ArticulationObjCfg(name="chair", mesh_path="chair/chair.obj", default_position=(-2.7314, 5.3259, 5.2776), default_orientation=(0.745668, 0, 0, 0.666318)),
    ArticulationObjCfg(name="table", mesh_path="table/table.obj", default_position=(10.6868, 7.2867, 23.2894), default_orientation=(0.99946, 0, 0, 0.032874))
]
    traj_filepath = "motion_data/mimic_filtered_v2/g1_priv_mimic-71-randclip-011645_v2.pkl"
    cameras = []
