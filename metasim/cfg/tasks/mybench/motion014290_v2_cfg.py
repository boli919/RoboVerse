from metasim.cfg.tasks import BaseTaskCfg
from metasim.utils import configclass
from metasim.cfg.objects import ArticulationObjCfg

@configclass
class Motion014290V2Cfg(BaseTaskCfg):
    episode_length = 300
    objects = [
    ArticulationObjCfg(name="table", mesh_path="table/table.obj", default_position=(2.7168, 3.805, 23.6391), default_orientation=(-0.785053, 0, 0, 0.619429))
]
    traj_filepath = "motion_data/mimic_filtered_v2/g1_priv_mimic-71-randclip-014290_v2.pkl"
    cameras = []
