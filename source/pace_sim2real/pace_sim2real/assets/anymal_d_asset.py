"""ANYmal D asset setup for the PACE system-identification task."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from pace_sim2real.utils import PaceDCMotorCfg

ASSET_DIR = Path(__file__).with_name("anymal_d")
ANYMAL_D_URDF = ASSET_DIR / "anymal.urdf"

JOINT_ORDER = (
    "LF_HAA",
    "LF_HFE",
    "LF_KFE",
    "RF_HAA",
    "RF_HFE",
    "RF_KFE",
    "LH_HAA",
    "LH_HFE",
    "LH_KFE",
    "RH_HAA",
    "RH_HFE",
    "RH_KFE",
)


def get_spec() -> mujoco.MjSpec:
    """Build a fixed-base simplified ANYmal-D MuJoCo specification.

    ANYbotics distributes visual meshes as Collada files, while the excitation
    and fitting workflow needs only the inertias and collision primitives.  We
    remove visual tags in memory, retaining all joint geometry and inertial
    properties from the official simplified URDF without requiring ROS or a
    mesh conversion tool at runtime.
    """
    if not ANYMAL_D_URDF.exists():
        raise FileNotFoundError(f"ANYmal D asset is missing: {ANYMAL_D_URDF}")
    root = ET.fromstring(ANYMAL_D_URDF.read_text(encoding="utf-8"))
    for link in root.findall("link"):
        for visual in tuple(link.findall("visual")):
            link.remove(visual)
    spec = mujoco.MjSpec.from_string(ET.tostring(root, encoding="unicode"))
    spec.modelname = "anymal_d_pace"
    return spec


ANYDRIVE_PACE_ACTUATOR_CFG = PaceDCMotorCfg(
    joint_names_expr=JOINT_ORDER,
    saturation_effort=140.0,
    effort_limit=89.0,
    velocity_limit=8.5,
    stiffness=85.0,
    damping=0.6,
    encoder_bias=0.0,
    armature=0.0,
    friction=0.0,
    dynamic_friction=0.0,
    viscous_friction=0.0,
    max_delay=10,
)

ANYMAL_D_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(ANYDRIVE_PACE_ACTUATOR_CFG,),
    soft_joint_pos_limit_factor=1.0,
)

ANYMAL_D_INIT_STATE = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 1.0),
    joint_pos={".*": 0.0},
    joint_vel={".*": 0.0},
)


def get_anymal_d_robot_cfg() -> EntityCfg:
    """Return a fresh PACE ANYmal-D entity configuration."""
    return EntityCfg(
        spec_fn=get_spec,
        articulation=ANYMAL_D_ARTICULATION,
        init_state=ANYMAL_D_INIT_STATE,
    )
