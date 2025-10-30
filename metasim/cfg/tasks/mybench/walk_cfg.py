# metasim/cfg/tasks/mybench/walk_cfg.py
from metasim.cfg.tasks import BaseTaskCfg        # 或 RLBenchTaskCfg
from metasim.utils import configclass
from metasim.cfg.objects import ArticulationObjCfg

@configclass
class WalkCfg(BaseTaskCfg):
    episode_length = 300
    # 轨迹 (v2.pkl 或 v2 文件夹)
    objects = []
    traj_filepath = "RoboVerse/metasim/data/walk/v2/g1_v2.pkl.gz"
    # 必要的空列表

    cameras = []
