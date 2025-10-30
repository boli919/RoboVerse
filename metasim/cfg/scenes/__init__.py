# ruff: noqa: F401

from __future__ import annotations

from .base_scene_cfg import SceneCfg
from .manycore_scene_827313_cfg import ManycoreScene827313Cfg
from .tapwater_scene_131_cfg import TapwaterScene131Cfg
from .tapwater_scene_138_cfg import TapwaterScene138Cfg
from .tapwater_scene_144_cfg import TapwaterScene144Cfg
from .tapwater_scene_152_cfg import TapwaterScene152Cfg
from .tapwater_scene_155_cfg import TapwaterScene155Cfg
from .kitchen_cfg import KitchenCfg
from .warehouse_cfg import WarehouseCfg


SCENE_CFGS = {
    "kitchen": KitchenCfg,
}
