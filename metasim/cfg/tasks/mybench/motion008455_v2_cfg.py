from metasim.cfg.tasks import BaseTaskCfg
from metasim.utils import configclass
from metasim.cfg.objects import ArticulationObjCfg

@configclass
class Motion008455V2Cfg(BaseTaskCfg):
    episode_length = 300
    objects = [
    ArticulationObjCfg(name="chair", mesh_path="models/objects/chair/chair.obj", default_position=(-7.692, -0.0081, 5.9125), default_orientation=(-0.66027, 0, 0, 0.751028)),
    ArticulationObjCfg(name="table", mesh_path="models/objects/table/table.obj", default_position=(3.7239, -4.2987, 23.6072), default_orientation=(0.21768, 0, 0, 0.97602))
]
    traj_filepath = "motion_data/mimic_filtered_v2/g1_priv_mimic-71-randclip-008455_v2.pkl"
    cameras = []
