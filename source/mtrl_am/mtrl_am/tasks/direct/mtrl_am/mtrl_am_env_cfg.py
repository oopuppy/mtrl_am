# mtrl_am_env_cfg.py
# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg, IdealPDActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils import math as math_utils

PI = getattr(math_utils, "PI", math.pi)


@configclass
class MtrlAmEnvCfg(DirectRLEnvCfg):
    # -----------------
    # env
    # -----------------
    decimation = 1
    episode_length_s = 15.0
    num_envs = 1

    action_space = 6
    observation_space = 42
    state_space = 0

    # -----------------
    # task goal (EE full pose goal) - reward용
    # -----------------
    ee_target_pos = (0.3, 0.0, 1.1)
    ee_target_euler_xyz = (0.0, 0.0, 0.0)

    # -----------------
    # Random EE target
    # -----------------
    randomize_ee_target: bool = True
    z_offset: float = 1.0

    ee_target_yaw_min: float = -PI
    ee_target_yaw_max: float = +PI

    randomize_goal_roll_pitch: bool = True
    ee_target_roll_min: float = -0.35
    ee_target_roll_max: float = +0.35
    ee_target_pitch_min: float = -0.35
    ee_target_pitch_max: float = +0.35

    # -----------------
    # DH params (BASE -> EE) : 4 links
    # -----------------
    dh_params = (
        (0.00071, 0.0, 0.0, 0.0),
        (-0.0014, PI / 2.0, 0.08375, -PI / 2.0),
        (-0.0515, -PI / 2.0, -0.0260, -PI / 2.0),
        (0.378, -PI / 2.0, 0.0, +PI / 2.0),
    )

    # -----------------
    # policy step increments (scaled)
    # -----------------
    dpos_scale = 0.01
    droll_scale = 0.01
    dpitch_scale = 0.01
    dyaw_scale = 0.01

    # -----------------
    # NEW: action->desired increment smoothing (LPF time constants)
    #  - 0.0이면 필터 OFF (그대로)
    #  - 보통 0.05~0.15s 정도 추천 (60Hz에서 꽤 안정적)
    # -----------------
    cmd_dpos_tau: float = 0.5   # [s]
    cmd_drpy_tau: float = 0.5  # [s]

    # -----------------
    # spec / controller gains
    # -----------------
    mass: float = 2.52
    g: float = 9.81

    kp_pos: float | tuple[float, float, float] = (25.0, 25.0, 50.0)
    kv_pos: float | tuple[float, float, float] = (10.0, 10.0, 25.0)
    kp_att: float | tuple[float, float, float] = (10.0, 10.0, 10.0)
    kd_att: float | tuple[float, float, float] = (0.5, 0.5, 0.5)

    k_tau: float = 1e-2

    # -----------------
    # ref smoothing / limiting (v_ref, a_ref, yaw_rate_ref)
    # -----------------
    ref_vel_tau: float = 0.08
    ref_acc_tau: float = 0.12
    ref_yaw_rate_tau: float = 0.08

    ref_vel_max: float = 2.0
    ref_acc_max: float = 12.0
    ref_yaw_rate_max: float = 3.0

    # -----------------
    # reward weights
    # -----------------
    w_ee_pos: float = -0.01
    w_ee_rp: float = -0.01
    w_ee_yaw: float = -0.01
    crash_penalty: float = -10.0

    # termination safety
    terminate_base_far = 10.0
    terminate_base_z_min: float = 0.2

    # -----------------
    # simulation
    # -----------------
    sim: SimulationCfg = SimulationCfg(dt=1 / 240, render_interval=1)

    # -----------------
    # visualization markers
    # -----------------
    enable_debug_vis: bool = True
    debug_vis_scale: float = 0.05
    debug_vis_root_prim: str = "/Visuals/MtrlAmTargets"

    vis_ee_goal: bool = True
    vis_ee_des: bool = True
    vis_base_ref: bool = True

    vis_ee_act: bool = False
    vis_base_act: bool = False
    vis_ee_fk: bool = False

    # -----------------
    # robot
    # -----------------
    quad_path = "/home/yubinkim/workspace/mtrl_am/source/mtrl_am/mtrl_am/tasks/direct/mtrl_am/models/quad_mod.usd"

    init_pos = (0.0, 0.0, 1.0)
    init_euler_xyz = (0.0, 0.0, 0.0)

    joint_target_update_hz: float = 60.0

    # stiffness_dict = {
    #     "Joint_1": 800.0,
    #     "Joint_2": 500.0,
    # }
    # damping_dict = {
    #     "Joint_1": 0.1,
    #     "Joint_2": 5.0,
    # }
    stiffness_dict = {
        "Joint_1": 5000.0,
        "Joint_2": 5000.0,
    }
    damping_dict = {
        "Joint_1": 1.0,
        "Joint_2": 1.0,
    }
    armature_dict = {
        "Joint_1": 0.1,
        "Joint_2": 0.1,
    }
    effort_limit_dict = {
        "Joint_1": 6.0,
        "Joint_2": 3.0,
    }

    quad_cfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/quad",
        spawn=sim_utils.UsdFileCfg(usd_path=quad_path),
        actuators={
            "joint_acts": ImplicitActuatorCfg(
                joint_names_expr=["Joint_1", "Joint_2"],
                stiffness=stiffness_dict,
                damping=damping_dict,
                armature=armature_dict,
                # effort_limit_sim=effort_limit_dict,
                # velocity_limit_sim=3.14,
            ),
        },
        init_state=ArticulationCfg.InitialStateCfg(
            pos=init_pos,
            rot=(1.0, 0.0, 0.0, 0.0),  # wxyz
            joint_pos={"Joint_1": 0.0, "Joint_2": 0.0},
        ),
    )

    # -----------------
    # scene
    # -----------------
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=num_envs, env_spacing=4.0, replicate_physics=True)

    def __post_init__(self):
        try:
            super().__post_init__()
        except AttributeError:
            pass

        self.scene.num_envs = int(self.num_envs)
        self.sim.render_interval = int(self.decimation)

        assert len(self.dh_params) == 4 and all(len(r) == 4 for r in self.dh_params), \
            f"dh_params must be a 4x4 tuple/list, got: {self.dh_params}"
