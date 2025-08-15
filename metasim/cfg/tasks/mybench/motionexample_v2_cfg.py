from metasim.cfg.tasks import BaseTaskCfg
from metasim.utils import configclass
from metasim.cfg.objects import ArticulationObjCfg

@configclass
class MotionExampleV2Cfg(BaseTaskCfg):
    episode_length = 300
#     objects = [
#     ArticulationObjCfg(name="container", mesh_path="models/objects/container/container.obj", default_position=(-1.5022145679265848, 0.9872111599718998, 0.03228600000000114), default_orientation=(1.0, 0.0, 0.0, 0.0)),
#     ArticulationObjCfg(name="container2", mesh_path="models/objects/container2/container2.obj", default_position=(1.8519167141926351, -1.3059679330454714, 0.03228600000000114), default_orientation=(1.0, 0.0, 0.0, 0.0)),
#     ArticulationObjCfg(name="shelf", mesh_path="models/objects/shelf/shelf.obj", default_position=(-0.1175731747518663, -1.643890600369255, 0.03228600000000114), default_orientation=(1.0, 0.0, 0.0, 0.0)),
#     ArticulationObjCfg(name="table1", mesh_path="models/objects/table1/table1.obj", default_position=(-2.2269, 0.8723, 0.0323), default_orientation=(0.999733, 0.0, 0.0, 0.0231)),
#     ArticulationObjCfg(name="box4", mesh_path="models/objects/box4/box4.obj", default_position=(-0.950518364279969, -0.4103877606264861, 0.03228600000000114), default_orientation=(1.0, 0.0, 0.0, 0.0)),
#     ArticulationObjCfg(name="examinator", mesh_path="models/objects/examinator/examinator.obj", default_position=(0.8920890116621369, -1.6096126904189345, 0.03228600000000114), default_orientation=(-0.7071067811865475, 0.0, 0.0, 0.7071067811865476)),
#     ArticulationObjCfg(name="telegraph_pole", mesh_path="models/objects/telegraph_pole/telegraph_pole.obj", default_position=(0.6820516819498172, 1.1681862835924153, 0.03228600000000114), default_orientation=(1.0, 0.0, 0.0, 0.0)),
#     ArticulationObjCfg(name="woodfence", mesh_path="models/objects/woodfence/woodfence.obj", default_position=(1.6557522769431972, -0.13671914157894116, 0.03228600000000114), default_orientation=(1.0, 0.0, 0.0, 0.0))
# ]
    traj_filepath = "motion_data/g1_pkl/example_v2.pkl"
    cameras = []
