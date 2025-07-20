from __future__ import annotations

from metasim.utils.configclass import configclass

from .base_scene_cfg import SceneCfg


@configclass
class KitchenCfg(SceneCfg):
    """Config class for kitchen scene"""

    name: str = "kitchen"
    usd_path: str = "roboverse_data/scenes/walk/Kitchen/Kitchen.obj"
    positions: list[tuple[float, float, float]] = [
        (0,-3,0),
    ]  # XXX: only positions are randomized for now
    default_position: tuple[float, float, float] = (0,-3,0)
    quat: tuple[float, float, float, float] = (0,0,0,1)
    scale: tuple[float, float, float] = (1, 1, 1)
