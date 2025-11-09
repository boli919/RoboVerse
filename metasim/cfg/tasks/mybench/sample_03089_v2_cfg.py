from metasim.cfg.tasks import BaseTaskCfg
from metasim.utils import configclass
from metasim.cfg.objects import ArticulationObjCfg

@configclass
class Sample03089V2Cfg(BaseTaskCfg):
    episode_length = 300
    objects = [
    ArticulationObjCfg(name="table1", mesh_path="models/objects/table1/table1.obj", default_position=(6.7222, 1.1153, 0.0323), default_orientation=(0.0, 0.0, 0.0, 1.0))
]
    traj_filepath = "motion_data/enhanced_new/sample_03089_v2.pkl"
    cameras = []
