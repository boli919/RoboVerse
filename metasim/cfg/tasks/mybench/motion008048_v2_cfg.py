from metasim.cfg.tasks import BaseTaskCfg
from metasim.utils import configclass
from metasim.cfg.objects import ArticulationObjCfg

@configclass
class Motion008048V2Cfg(BaseTaskCfg):
    episode_length = 300
    objects = [
    ArticulationObjCfg(name="chair", mesh_path="models/objects/chair/chair.obj", default_position=(7.6597, 2.3764, 5.6241), default_orientation=(-0.643435, 0, 0, 0.765501)),
    ArticulationObjCfg(name="table", mesh_path="models/objects/table/table.obj", default_position=(1.3598, -6.3895, 23.3722), default_orientation=(-0.510836, 0, 0, 0.859678))
]
    traj_filepath = "motion_data/mimic_filtered_v2/g1_priv_mimic-71-randclip-008048_v2.pkl"
    cameras = []
