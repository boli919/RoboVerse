from metasim.cfg.tasks import BaseTaskCfg
from metasim.utils import configclass
from metasim.cfg.objects import ArticulationObjCfg

@configclass
class WalkStand4433V2Cfg(BaseTaskCfg):
    episode_length = 300
    objects = [
    ArticulationObjCfg(name="table1", mesh_path="models/objects/table1/table1.obj", default_position=(-4.143899917602539, 0.04699999839067459, 0.0323), default_orientation=(0.0, 0.0, 0.0, 1.0))
]
    traj_filepath = "motion_data/walk_stand_img/walk_stand4433_v2.pkl"
    cameras = []
