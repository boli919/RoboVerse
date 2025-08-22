from metasim.cfg.tasks import BaseTaskCfg
from metasim.utils import configclass
from metasim.cfg.objects import ArticulationObjCfg

@configclass
class Motion2Cfg(BaseTaskCfg):
    episode_length = 300
    objects = [
    ArticulationObjCfg(name="container", mesh_path="models/objects/container/container.obj", default_position=(-6.739605100254435, -1.0641898168845245, 0.03228600000000114), default_orientation=(1.0, 0.0, 0.0, 0.0)),
    ArticulationObjCfg(name="container2", mesh_path="models/objects/container2/container2.obj", default_position=(-1.2592938700268843, 0.7674228331488233, 0.03228600000000114), default_orientation=(1.0, 0.0, 0.0, 0.0)),
    ArticulationObjCfg(name="shelf", mesh_path="models/objects/shelf/shelf.obj", default_position=(1.2191933477839196, 1.4518147085988988, 0.03228600000000114), default_orientation=(1.0, 0.0, 0.0, 0.0)),
    ArticulationObjCfg(name="table1", mesh_path="models/objects/table1/table1.obj", default_position=(-3.6522998809814453, 0.6442000269889832, 0.0323), default_orientation=(0.770181, 0.0, 0.0, -0.637825)),
    ArticulationObjCfg(name="box4", mesh_path="models/objects/box4/box4.obj", default_position=(-2.3989107709090605, -2.5847751179297607, 0.03228600000000114), default_orientation=(1.0, 0.0, 0.0, 0.0)),
    ArticulationObjCfg(name="examinator", mesh_path="models/objects/examinator/examinator.obj", default_position=(1.703687095617847, 0.5317614966068459, 0.03228600000000114), default_orientation=(-0.7071067811865475, 0.0, 0.0, 0.7071067811865476)),
    ArticulationObjCfg(name="telegraph_pole", mesh_path="models/objects/telegraph_pole/telegraph_pole.obj", default_position=(-4.587247489837461, -0.35503313313183715, 0.03228600000000114), default_orientation=(1.0, 0.0, 0.0, 0.0))
]
    traj_filepath = "motion_data/smpl_v2/walk_stand_v2.pkl"
    cameras = []
