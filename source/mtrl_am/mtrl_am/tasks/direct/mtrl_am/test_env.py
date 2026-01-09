# mtrl_am_env.py
# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
import math

import torch
import torch.nn.functional as F

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import math as math_utils
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from .mtrl_am_env_cfg import MtrlAmEnvCfg


PI = getattr(math_utils, "PI", math.pi)


def wrap_pi(x: torch.Tensor) -> torch.Tensor:
    return math_utils.wrap_to_pi(x)


def _gain3_to_tensor(g, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(g, (int, float)):
        gg = (float(g), float(g), float(g))
        return torch.tensor(gg, device=device, dtype=dtype).view(1, 3)

    if isinstance(g, torch.Tensor):
        if g.numel() == 1:
            v = float(g.item())
            return torch.tensor((v, v, v), device=device, dtype=dtype).view(1, 3)
        if g.numel() == 3:
            return g.to(device=device, dtype=dtype).view(1, 3)
        raise ValueError(f"Gain tensor must have 1 or 3 elements, got shape={tuple(g.shape)}")

    try:
        if len(g) == 3:
            return torch.tensor((float(g[0]), float(g[1]), float(g[2])), device=device, dtype=dtype).view(1, 3)
    except Exception as e:
        raise ValueError(f"Invalid gain type: {type(g)}") from e

    raise ValueError(f"Gain must be float/int, torch.Tensor(1 or 3), or length-3 sequence. Got: {g}")


def vee(S: torch.Tensor) -> torch.Tensor:
    return torch.stack([S[..., 2, 1], S[..., 0, 2], S[..., 1, 0]], dim=-1)


def hat(v: torch.Tensor) -> torch.Tensor:
    O = torch.zeros((*v.shape[:-1], 3, 3), device=v.device, dtype=v.dtype)
    O[..., 0, 1] = -v[..., 2]
    O[..., 0, 2] = v[..., 1]
    O[..., 1, 0] = v[..., 2]
    O[..., 1, 2] = -v[..., 0]
    O[..., 2, 0] = -v[..., 1]
    O[..., 2, 1] = v[..., 0]
    return O


def so3_error_vee(R: torch.Tensor, R_d: torch.Tensor) -> torch.Tensor:
    """
    SO(3) orientation error (Lee-style):
        e_R = vee( 0.5 * (R_d^T R - R^T R_d) )
    """
    R_T = R.transpose(1, 2)
    R_d_T = R_d.transpose(1, 2)
    e_R_mat = 0.5 * (torch.bmm(R_d_T, R) - torch.bmm(R_T, R_d))
    return vee(e_R_mat)


class MtrlAmEnv(DirectRLEnv):
    cfg: MtrlAmEnvCfg

    def __init__(self, cfg: MtrlAmEnvCfg, render_mode: str | None = None, **kwargs):
        # markers
        self._vis_ee_goal: VisualizationMarkers | None = None
        self._vis_ee_des: VisualizationMarkers | None = None
        self._vis_base_ref: VisualizationMarkers | None = None
        self._vis_ee_act: VisualizationMarkers | None = None
        self._vis_base_act: VisualizationMarkers | None = None
        self._vis_ee_fk: VisualizationMarkers | None = None
        self._marker_indices_frame: torch.Tensor | None = None

        super().__init__(cfg, render_mode, **kwargs)

        # policy/controller == simulation Hz (decimation=1 전제)
        self._dt = float(cfg.sim.dt)

        # DH params
        dh_list = getattr(cfg, "dh_params", None)
        assert dh_list is not None, "cfg.dh_params 필요 (shape=(4,4))"
        self._dh_params = torch.tensor(dh_list, device=self.device, dtype=torch.float32)
        assert self._dh_params.shape == (4, 4), f"cfg.dh_params must be (4,4), got {self._dh_params.shape}"

        # body ids
        base_ids, _ = self.robot.find_bodies("Base")
        ee_ids, _ = self.robot.find_bodies("Link_2_Part_4")
        assert len(base_ids) == 1, f"Expected 1 Base body, got {base_ids}"
        assert len(ee_ids) == 1, f"Expected 1 EE body, got {ee_ids}"
        self._base_body_ids = base_ids
        self._base_body_id = int(base_ids[0])
        self._ee_body_id = int(ee_ids[0])

        # joint ids
        j12_ids, _ = self.robot.find_joints(["Joint_1", "Joint_2"])
        assert len(j12_ids) == 2, f"Expected 2 joints (Joint_1, Joint_2), got {j12_ids}"
        self._joint_ids = torch.as_tensor(j12_ids, device=self.device, dtype=torch.long)

        # inertia (base)
        J = self.robot.data.default_inertia[:, self._base_body_id, :]
        if J.dim() == 3 and J.shape[-2:] == (3, 3):
            self.J_body = J.to(device=self.device, dtype=torch.float32).clone()
        else:
            self.J_body = J.reshape(-1, 3, 3).to(device=self.device, dtype=torch.float32).clone()

        # allocation matrices
        self.D, self.G, self.B_inv = build_allocation_mat(cfg, device=self.device)

        # init pose
        self._init_pos = torch.tensor(cfg.init_pos, device=self.device, dtype=torch.float32).view(1, 3)
        r0, p0, y0 = cfg.init_euler_xyz
        rr0 = torch.full((self.num_envs,), float(r0), device=self.device, dtype=torch.float32)
        pp0 = torch.full((self.num_envs,), float(p0), device=self.device, dtype=torch.float32)
        yy0 = torch.full((self.num_envs,), float(y0), device=self.device, dtype=torch.float32)
        self._init_quat = math_utils.quat_unique(math_utils.quat_from_euler_xyz(rr0, pp0, yy0))  # wxyz

        # goal buffers
        self._ee_goal_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self._ee_goal_quat = torch.zeros((self.num_envs, 4), device=self.device, dtype=torch.float32)
        self._ee_goal_rpy = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)

        self._ee_goal_pos_fixed = (
            torch.tensor(cfg.ee_target_pos, device=self.device, dtype=torch.float32).view(1, 3).repeat(self.num_envs, 1)
        )
        r_fix, p_fix, y_fix = cfg.ee_target_euler_xyz
        self._ee_goal_rpy_fixed = (
            torch.tensor((float(r_fix), float(p_fix), float(y_fix)), device=self.device, dtype=torch.float32)
            .view(1, 3)
            .repeat(self.num_envs, 1)
        )
        rr = self._ee_goal_rpy_fixed[:, 0]
        pp = self._ee_goal_rpy_fixed[:, 1]
        yy = self._ee_goal_rpy_fixed[:, 2]
        self._ee_goal_quat_fixed = math_utils.quat_unique(math_utils.quat_from_euler_xyz(rr, pp, yy))

        # action buffers
        self.actions = torch.zeros((self.num_envs, cfg.action_space), device=self.device, dtype=torch.float32)
        self._prev_actions = torch.zeros_like(self.actions)
        self._motor_thrusts = torch.zeros((self.num_envs, 4), device=self.device, dtype=torch.float32)

        # desired EE state (RAW: policy 적분 상태)  --- LPF 제거: RAW를 그대로 사용
        self._ee_des_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self._ee_des_rpy = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)  # roll,pitch,yaw (raw)
        self._q_des = torch.zeros((self.num_envs, 2), device=self.device, dtype=torch.float32)  # joint1, joint2 (raw)

        # finite diff prev states (for v_ref/a_ref/yaw_rate_ref)
        self._base_pos_ref_prev = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self._base_vel_ref_prev = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self._yaw_ref_prev = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)

        # step buffers (for obs/vis)
        self._ee_des_pos_W = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self._ee_des_quat_W = torch.zeros((self.num_envs, 4), device=self.device, dtype=torch.float32)

        self._base_pos_ref = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self._base_quat_ref = torch.zeros((self.num_envs, 4), device=self.device, dtype=torch.float32)
        self._base_vel_ref = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self._base_acc_ref = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self._yaw_ref = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)
        self._yaw_rate_ref = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)

        # marker indices
        self._marker_indices_frame = torch.zeros((self.num_envs,), device=self.device, dtype=torch.long)

        # ---- initialize desired (raw) to current state ----
        ee_pos_init, _, _, _ = self._get_body_state(self._ee_body_id)
        ee_pos_init = torch.nan_to_num(ee_pos_init, nan=0.0, posinf=0.0, neginf=0.0)
        self._ee_des_pos.copy_(ee_pos_init)

        q_act_12 = self.robot.data.joint_pos[:, self._joint_ids].to(dtype=torch.float32)
        q_act_12 = torch.nan_to_num(q_act_12, nan=0.0, posinf=0.0, neginf=0.0)
        self._q_des.copy_(q_act_12)

        _, base_quat, _, _ = self._get_body_state(self._base_body_id)
        _, _, yaw0 = math_utils.euler_xyz_from_quat(base_quat)
        yaw0 = torch.nan_to_num(yaw0, nan=0.0, posinf=0.0, neginf=0.0)
        self._ee_des_rpy[:, 0] = self._q_des[:, 0]
        self._ee_des_rpy[:, 1] = self._q_des[:, 1]
        self._ee_des_rpy[:, 2] = yaw0

        # initial goal sampling
        self._resample_ee_goal(env_ids=torch.arange(self.num_envs, device=self.device, dtype=torch.long))

        # ---- init prev refs to avoid first finite-diff spike ----
        ee_des_pos_W0, ee_des_quat_W0 = self._compute_desired_ee_pose_world(env_ids=None)  # RAW 기반
        base_pos_ref0, base_quat_ref0, yaw_ref0 = self._compute_base_ref_pose_only(
            ee_pos_des_W=ee_des_pos_W0,
            ee_quat_des_W=ee_des_quat_W0,
            q_joint_12=self.robot.data.joint_pos[:, self._joint_ids].to(dtype=ee_des_pos_W0.dtype),
        )
        self._base_pos_ref_prev.copy_(base_pos_ref0)
        self._base_vel_ref_prev.zero_()
        self._yaw_ref_prev.copy_(yaw_ref0)

        # fill step buffers once
        self._compute_and_store_refs()

    # -----------------------------
    # scene setup
    # -----------------------------
    def _setup_scene(self):
        self.robot = Articulation(self.cfg.quad_cfg)
        self.scene.articulations["robot"] = self.robot

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)

        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # markers
        if bool(getattr(self.cfg, "enable_debug_vis", False)):
            scale = float(getattr(self.cfg, "debug_vis_scale", 0.05))
            root = str(getattr(self.cfg, "debug_vis_root_prim", "/Visuals/MtrlAmTargets"))
            frame_usd = f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd"

            def _make_cfg(name: str) -> VisualizationMarkersCfg:
                return VisualizationMarkersCfg(
                    prim_path=f"{root}/{name}",
                    markers={"frame": sim_utils.UsdFileCfg(usd_path=frame_usd, scale=(scale, scale, scale))},
                )

            if bool(getattr(self.cfg, "vis_ee_goal", False)):
                self._vis_ee_goal = VisualizationMarkers(_make_cfg("ee_goal"))
            if bool(getattr(self.cfg, "vis_ee_des", False)):
                self._vis_ee_des = VisualizationMarkers(_make_cfg("ee_des"))
            if bool(getattr(self.cfg, "vis_base_ref", False)):
                self._vis_base_ref = VisualizationMarkers(_make_cfg("base_ref"))
            if bool(getattr(self.cfg, "vis_ee_act", False)):
                self._vis_ee_act = VisualizationMarkers(_make_cfg("ee_act"))
            if bool(getattr(self.cfg, "vis_base_act", False)):
                self._vis_base_act = VisualizationMarkers(_make_cfg("base_act"))
            if bool(getattr(self.cfg, "vis_ee_fk", False)):
                self._vis_ee_fk = VisualizationMarkers(_make_cfg("ee_fk"))

    # -----------------------------
    # RL loop
    # -----------------------------
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """
        decimation=1 전제:
          - 여기서는 "액션 저장 + 클램프 + prev_action 갱신"만 수행
          - desired/ref 계산 및 컨트롤 입력 생성은 _apply_action()에서 수행
        """
        self._prev_actions.copy_(self.actions)

        if actions is None or actions.numel() == 0:
            self.actions.zero_()
            return

        # safe: dim mismatch truncate/pad
        if actions.shape[-1] != int(self.cfg.action_space):
            a = actions
            if a.shape[-1] > int(self.cfg.action_space):
                a = a[:, : int(self.cfg.action_space)]
            else:
                pad = torch.zeros(
                    (a.shape[0], int(self.cfg.action_space) - a.shape[-1]),
                    device=a.device,
                    dtype=a.dtype,
                )
                a = torch.cat([a, pad], dim=-1)
            actions = a

        self.actions = torch.clamp(actions, -1.0, 1.0)

    def _apply_action(self) -> None:
        """
        action 처리(=desired 업데이트) + ref 계산 + controller + actuator 적용
        -> 전부 simulation step마다 수행
        (LPF 제거: RAW desired를 그대로 사용)
        """
        # 1) action -> raw desired integrate
        self._update_desired_from_action()

        # 2) raw desired -> refs (and store for obs/vis)
        self._compute_and_store_refs()

        # 3) read actual states
        base_pos, base_quat, base_linvel, base_angvel = self._get_body_state(self._base_body_id)

        # debug states
        ee_act_pos_W, ee_act_quat_W, _, _ = self._get_body_state(self._ee_body_id)
        q_act_12 = self.robot.data.joint_pos[:, self._joint_ids]

        # 4) visualize
        if self._vis_ee_goal is not None:
            self._vis_ee_goal.visualize(self._ee_goal_pos, self._ee_goal_quat, marker_indices=self._marker_indices_frame)
        if self._vis_ee_des is not None:
            self._vis_ee_des.visualize(self._ee_des_pos_W, self._ee_des_quat_W, marker_indices=self._marker_indices_frame)
        if self._vis_base_ref is not None:
            self._vis_base_ref.visualize(self._base_pos_ref, self._base_quat_ref, marker_indices=self._marker_indices_frame)

        if self._vis_ee_act is not None:
            self._vis_ee_act.visualize(ee_act_pos_W, ee_act_quat_W, marker_indices=self._marker_indices_frame)
        if self._vis_base_act is not None:
            self._vis_base_act.visualize(base_pos, base_quat, marker_indices=self._marker_indices_frame)

        if self._vis_ee_fk is not None:
            T_W_B_act = tf_from_pos_quat_wxyz(base_pos, base_quat)
            T_B_E_act = fk_dh_T_base_to_ee_from_joints(
                q_joint_12=q_act_12.to(dtype=base_pos.dtype),
                dh_params=self._dh_params.to(dtype=base_pos.dtype),
            )
            T_W_E_fk = torch.bmm(T_W_B_act, T_B_E_act)
            ee_fk_pos = T_W_E_fk[:, 0:3, 3]
            ee_fk_quat = math_utils.quat_unique(quat_from_rotmat_wxyz(T_W_E_fk[:, 0:3, 0:3]))
            self._vis_ee_fk.visualize(ee_fk_pos, ee_fk_quat, marker_indices=self._marker_indices_frame)

        # 5) controller
        u = self._geometric_controller(
            pos_I=base_pos,
            quat_I=base_quat,
            linvel_I=base_linvel,
            omega_I=base_angvel,
            J_I=self.J_body,
            p_ref_I=self._base_pos_ref,
            v_ref_I=self._base_vel_ref,
            a_ref_I=self._base_acc_ref,
            yaw_ref_I_t=self._yaw_ref,
            yaw_rate_ref_I_t=self._yaw_rate_ref,
            cfg=self.cfg,
        )

        # 6) motor allocation -> external wrenches
        T = self._motor_allocation(u_des_I=u, B_inv=self.B_inv, cfg=self.cfg)
        self._motor_thrusts.copy_(T)

        forces_b, torques_b = self._thrusts_to_body_wrenches(thrusts=T, D=self.D, G=self.G)
        self.robot.set_external_force_and_torque(
            forces=forces_b,
            torques=torques_b,
            body_ids=self._base_body_ids,
            env_ids=None,
            is_global=False,
        )

        # 7) joints target: RAW desired q_des로 명령
        joint_target = self.robot.data.joint_pos.clone().to(device=self.device, dtype=torch.float32)
        joint_target[:, self._joint_ids] = self._q_des.to(dtype=torch.float32)
        self.robot.set_joint_position_target(target=joint_target)

    def _get_observations(self) -> dict:
        """
        - 남기는 항목(ee pos err, ee rot err, ee linvel, ee angvel, base linvel, base angvel)은 모두 base frame 기준
        - 추가: prev_action, base 자세(base quat wxyz in world), base frame 기준의 ee goal pose(pos+quat)
        """
        ee_pos_W, ee_quat_W, ee_linvel_W, ee_angvel_W = self._get_body_state(self._ee_body_id)
        base_pos_W, base_quat_W, base_linvel_W, base_angvel_W = self._get_body_state(self._base_body_id)

        # R_WB: base frame -> world
        R_WB = math_utils.matrix_from_quat(base_quat_W)
        R_BW = R_WB.transpose(1, 2)  # world -> base

        def _w_to_b(v_w: torch.Tensor) -> torch.Tensor:
            return torch.bmm(R_BW, v_w.unsqueeze(-1)).squeeze(-1)

        # 1) EE pos err (W에서 차이) -> base frame
        ee_pos_err_W = ee_pos_W - self._ee_goal_pos
        ee_pos_err_B = _w_to_b(ee_pos_err_W)

        # 2) EE rot err (goal 대비) -> base frame에서 계산
        R_ee_W = math_utils.matrix_from_quat(ee_quat_W)
        R_goal_W = math_utils.matrix_from_quat(self._ee_goal_quat)

        R_ee_B = torch.bmm(R_BW, R_ee_W)       # R_BE
        R_goal_B = torch.bmm(R_BW, R_goal_W)   # R_BGoal
        ee_rot_err_goal_B = so3_error_vee(R=R_ee_B, R_d=R_goal_B)

        # 3) velocities -> base frame (표현만 base frame으로 회전)
        ee_linvel_B = _w_to_b(ee_linvel_W)
        ee_angvel_B = _w_to_b(ee_angvel_W)
        base_linvel_B = _w_to_b(base_linvel_W)
        base_angvel_B = _w_to_b(base_angvel_W)

        # 4) base attitude (world 기준, wxyz)
        base_quat_obs = math_utils.quat_unique(base_quat_W)

        # 5) ee goal pose in base frame
        goal_pos_rel_W = self._ee_goal_pos - base_pos_W
        goal_pos_rel_B = _w_to_b(goal_pos_rel_W)
        goal_quat_B = math_utils.quat_unique(quat_from_rotmat_wxyz(R_goal_B))

        prev_act = self._prev_actions

        joint_angle = self.robot.data.joint_pos[:, self._joint_ids].to(dtype=ee_pos_err_B.dtype)

        obs = torch.cat(
            [
                ee_pos_err_B,          # 3
                ee_rot_err_goal_B,     # 3
                ee_linvel_B,           # 3
                ee_angvel_B,           # 3
                base_linvel_B,         # 3
                base_angvel_B,         # 3
                base_quat_obs,         # 4 (wxyz, world)
                goal_pos_rel_B,        # 3 (base frame)
                goal_quat_B,           # 4 (wxyz, base frame)
                joint_angle,           # 2
                prev_act,              # action_space (기본 6)
            ],
            dim=-1,
        )

        obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        ee_pos_W, ee_quat_W, _, _ = self._get_body_state(self._ee_body_id)

        # desired EE pose (world)
        ee_des_pos_W = self._ee_des_pos_W
        ee_des_quat_W = self._ee_des_quat_W

        # desired mixing weight (cfg에서 조절)
        alpha = float(getattr(self.cfg, "reward_des_weight", 0.1))
        alpha = float(max(0.0, min(1.0, alpha)))  # clamp [0,1]

        # --------------------
        # position errors
        # --------------------
        pos_err2_act = torch.sum((ee_pos_W - self._ee_goal_pos) ** 2, dim=-1)
        pos_err2_des = torch.sum((ee_des_pos_W - self._ee_goal_pos) ** 2, dim=-1)

        # --------------------
        # quaternion-based attitude errors (swing/twist about world Z)
        #   - yaw:   twist component around world z-axis
        #   - r/p:   swing component (yaw removed)
        # error metric: err = 1 - |<q1,q2>|^2  in [0,1]
        # --------------------
        def _quat_normalize(q: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
            return q / (torch.linalg.norm(q, dim=-1, keepdim=True) + eps)

        def _quat_conj(q: torch.Tensor) -> torch.Tensor:  # wxyz
            return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)

        def _quat_mul(q: torch.Tensor, r: torch.Tensor) -> torch.Tensor:  # wxyz
            qw, qx, qy, qz = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
            rw, rx, ry, rz = r[..., 0], r[..., 1], r[..., 2], r[..., 3]
            w = qw * rw - qx * rx - qy * ry - qz * rz
            x = qw * rx + qx * rw + qy * rz - qz * ry
            y = qw * ry - qx * rz + qy * rw + qz * rx
            z = qw * rz + qx * ry - qy * rx + qz * rw
            return torch.stack([w, x, y, z], dim=-1)

        def _twist_z(q: torch.Tensor) -> torch.Tensor:
            q = _quat_normalize(q)
            w = q[..., 0:1]
            z = q[..., 3:4]
            twist = torch.cat([w, torch.zeros_like(z), torch.zeros_like(z), z], dim=-1)
            return _quat_normalize(twist)

        def _swing_z(q: torch.Tensor) -> torch.Tensor:
            t = _twist_z(q)
            return _quat_mul(_quat_normalize(q), _quat_conj(t))  # swing = q * t^{-1}

        def _quat_err01(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
            q1 = _quat_normalize(q1)
            q2 = _quat_normalize(q2)
            dot = torch.sum(q1 * q2, dim=-1)
            dot_abs = torch.abs(dot).clamp(0.0, 1.0)
            return 1.0 - dot_abs * dot_abs  # [0,1]

        q_goal = self._ee_goal_quat  # wxyz, world

        # actual
        yaw_err2_act = _quat_err01(_twist_z(ee_quat_W), _twist_z(q_goal))
        rp_err2_act  = _quat_err01(_swing_z(ee_quat_W), _swing_z(q_goal))

        # desired
        yaw_err2_des = _quat_err01(_twist_z(ee_des_quat_W), _twist_z(q_goal))
        rp_err2_des  = _quat_err01(_swing_z(ee_des_quat_W), _swing_z(q_goal))

        # action smoothing: ||a_t - a_{t-1}||^2 (그대로)
        da = self.actions - self._prev_actions
        act_diff2 = torch.sum(da * da, dim=-1)

        # ---- numeric safety ----
        pos_err2_act = torch.nan_to_num(pos_err2_act, nan=1e6, posinf=1e6, neginf=1e6)
        pos_err2_des = torch.nan_to_num(pos_err2_des, nan=1e6, posinf=1e6, neginf=1e6)
        rp_err2_act  = torch.nan_to_num(rp_err2_act,  nan=1.0, posinf=1.0, neginf=1.0)
        rp_err2_des  = torch.nan_to_num(rp_err2_des,  nan=1.0, posinf=1.0, neginf=1.0)
        yaw_err2_act = torch.nan_to_num(yaw_err2_act, nan=1.0, posinf=1.0, neginf=1.0)
        yaw_err2_des = torch.nan_to_num(yaw_err2_des, nan=1.0, posinf=1.0, neginf=1.0)
        act_diff2    = torch.nan_to_num(act_diff2,    nan=1e6, posinf=1e6, neginf=1e6)

        # ---- exp(-x) reward terms ----
        s_pos = float(getattr(self.cfg, "exp_pos_scale", 1.0))
        s_rp  = float(getattr(self.cfg, "exp_rp_scale", 1.0))
        s_yaw = float(getattr(self.cfg, "exp_yaw_scale", 1.0))
        s_act = float(getattr(self.cfg, "exp_action_smooth_scale", 0.1))

        # desired mixing: (1-alpha)*actual + alpha*desired
        r_pos = (1.0 - alpha) * torch.exp(-s_pos * pos_err2_act) + alpha * torch.exp(-s_pos * pos_err2_des)
        r_rp  = (1.0 - alpha) * torch.exp(-s_rp  * rp_err2_act)  + alpha * torch.exp(-s_rp  * rp_err2_des)
        r_yaw = (1.0 - alpha) * torch.exp(-s_yaw * yaw_err2_act) + alpha * torch.exp(-s_yaw * yaw_err2_des)

        # smoothing 그대로
        r_smooth = torch.exp(-s_act * act_diff2)

        # ---- weights ----
        w_pos    = float(getattr(self.cfg, "w_ee_pos", 1.0))
        w_rp     = float(getattr(self.cfg, "w_ee_rp", 1.0))
        w_yaw    = float(getattr(self.cfg, "w_ee_yaw", 1.0))
        w_smooth = float(getattr(self.cfg, "w_action_smooth", 1.0))

        w = torch.tensor([w_pos, w_rp, w_yaw, w_smooth], device=ee_pos_W.device, dtype=ee_pos_W.dtype)
        r = w[0] * r_pos + w[1] * r_rp + w[2] * r_yaw + w[3] * r_smooth

        terminated, _ = self._get_dones()
        r = r * (~terminated).to(dtype=r.dtype)

        return torch.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)


    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        base_pos, _, _, _ = self._get_body_state(self._base_body_id)
        dist = torch.linalg.norm(base_pos - self.scene.env_origins, dim=-1)

        terminated = torch.any(torch.isnan(base_pos), dim=-1)

        terminate_base_far = float(getattr(self.cfg, "terminate_base_far", 10.0))
        terminated = terminated | (dist > terminate_base_far)

        z_min = float(getattr(self.cfg, "terminate_base_z_min", 0.2))
        terminated = terminated | (base_pos[:, 2] < z_min)

        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = list(range(self.num_envs))
        super()._reset_idx(env_ids)

        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        n = env_ids.numel()
        if n == 0:
            return

        self._resample_ee_goal(env_ids)

        # root reset
        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, 0:3] = self.scene.env_origins[env_ids] + self._init_pos.repeat(n, 1)
        root_state[:, 3:7] = self._init_quat[env_ids]
        root_state[:, 7:13] = 0.0

        self.robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)

        # joints reset
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        joint_pos.zero_()
        joint_vel.zero_()
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        self.robot.reset(env_ids)

        # buffers
        self.actions[env_ids] = 0.0
        self._prev_actions[env_ids] = 0.0
        self._motor_thrusts[env_ids] = 0.0

        # desired(raw): set to current
        ee_pos, _, _, _ = self._get_body_state(self._ee_body_id)
        self._ee_des_pos[env_ids] = ee_pos[env_ids].clone()

        q_act = self.robot.data.joint_pos[:, self._joint_ids]
        self._q_des[env_ids] = q_act[env_ids].clone()

        _, base_quat, _, _ = self._get_body_state(self._base_body_id)
        _, _, yaw_base = math_utils.euler_xyz_from_quat(base_quat)
        self._ee_des_rpy[env_ids, 0] = self._q_des[env_ids, 0]
        self._ee_des_rpy[env_ids, 1] = self._q_des[env_ids, 1]
        self._ee_des_rpy[env_ids, 2] = yaw_base[env_ids]

        # reset prev refs to avoid finite-diff spike
        ee_des_quat_sub = math_utils.quat_unique(
            math_utils.quat_from_euler_xyz(
                self._ee_des_rpy[env_ids, 0],
                self._ee_des_rpy[env_ids, 1],
                self._ee_des_rpy[env_ids, 2],
            )
        )
        base_pos_ref_sub, base_quat_ref_sub, yaw_ref_sub = self._compute_base_ref_pose_only(
            ee_pos_des_W=self._ee_des_pos[env_ids],
            ee_quat_des_W=ee_des_quat_sub,
            q_joint_12=self.robot.data.joint_pos[:, self._joint_ids][env_ids].to(dtype=self._ee_des_pos.dtype),
        )
        self._base_pos_ref_prev[env_ids] = base_pos_ref_sub
        self._base_vel_ref_prev[env_ids] = 0.0
        self._yaw_ref_prev[env_ids] = yaw_ref_sub

        # refresh step buffers once
        self._compute_and_store_refs(env_ids=env_ids)

    # -----------------------------
    # helpers
    # -----------------------------
    def _get_body_state(self, body_id: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pos_w = self.robot.data.body_link_pos_w[:, body_id, :]
        quat_w = self.robot.data.body_link_quat_w[:, body_id, :]  # wxyz
        linvel_w = self.robot.data.body_link_lin_vel_w[:, body_id, :]
        angvel_w = self.robot.data.body_link_ang_vel_w[:, body_id, :]
        return pos_w, quat_w, linvel_w, angvel_w

    def _update_desired_from_action(self) -> None:
        """
        action -> (scaled) increment -> integrate into RAW desired
        (policy/controller == sim Hz이므로 step마다 수행)

        ✅ 변경점:
          - actions[:,0:3]을 "base frame 기준 Δpos"로 해석
          - 현재 base attitude(R_WB)로 회전시켜 world Δpos로 만든 뒤 self._ee_des_pos에 적분
        """
        # (B-frame) delta position from action
        dpos_B = float(self.cfg.dpos_scale) * self.actions[:, 0:3]

        droll = float(getattr(self.cfg, "droll_scale", 0.1)) * self.actions[:, 3]
        dpitch = float(getattr(self.cfg, "dpitch_scale", 0.1)) * self.actions[:, 4]
        dyaw = float(getattr(self.cfg, "dyaw_scale", 0.1)) * self.actions[:, 5]

        # --- base frame -> world frame rotation for dpos ---
        _, base_quat_W, _, _ = self._get_body_state(self._base_body_id)
        R_WB = math_utils.matrix_from_quat(base_quat_W)  # base -> world, (N,3,3)

        # dpos_W = R_WB * dpos_B
        dpos_W = torch.bmm(R_WB, dpos_B.unsqueeze(-1)).squeeze(-1)

        # integrate RAW desired EE position (in world)
        self._ee_des_pos = self._ee_des_pos + dpos_W

        # roll/pitch -> RAW joint targets
        self._q_des[:, 0] = self._q_des[:, 0] + droll
        self._q_des[:, 1] = self._q_des[:, 1] + dpitch

        # clamp roll/pitch absolute limits
        self._q_des[:, 0] = torch.clamp(
            self._q_des[:, 0],
            float(getattr(self.cfg, "ee_target_roll_min", -PI * 7 / 18)),
            float(getattr(self.cfg, "ee_target_roll_max", +PI * 2 / 18)),
        )
        self._q_des[:, 1] = torch.clamp(
            self._q_des[:, 1],
            float(getattr(self.cfg, "ee_target_pitch_min", -PI * 10 / 18)),
            float(getattr(self.cfg, "ee_target_pitch_max", +PI * 2 / 18)),
        )

        # RAW yaw desired
        self._ee_des_rpy[:, 2] = wrap_pi(self._ee_des_rpy[:, 2] + dyaw)

        # keep RAW desired rpy consistent with RAW joints
        self._ee_des_rpy[:, 0] = self._q_des[:, 0]
        self._ee_des_rpy[:, 1] = self._q_des[:, 1]

        # safety
        self._ee_des_pos = torch.nan_to_num(self._ee_des_pos, nan=0.0, posinf=0.0, neginf=0.0)
        self._q_des = torch.nan_to_num(self._q_des, nan=0.0, posinf=0.0, neginf=0.0)
        self._ee_des_rpy = torch.nan_to_num(self._ee_des_rpy, nan=0.0, posinf=0.0, neginf=0.0)

    def _quat_from_raw_rpy(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        if env_ids is None:
            r = self._ee_des_rpy[:, 0]
            p = self._ee_des_rpy[:, 1]
            y = self._ee_des_rpy[:, 2]
        else:
            r = self._ee_des_rpy[env_ids, 0]
            p = self._ee_des_rpy[env_ids, 1]
            y = self._ee_des_rpy[env_ids, 2]
        return math_utils.quat_unique(math_utils.quat_from_euler_xyz(r, p, y))

    def _compute_desired_ee_pose_world(self, env_ids: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        LPF 제거: "des pose"는 RAW desired로 계산
        """
        if env_ids is None:
            ee_pos = self._ee_des_pos
            ee_quat = self._quat_from_raw_rpy(env_ids=None)
        else:
            ee_pos = self._ee_des_pos[env_ids]
            ee_quat = self._quat_from_raw_rpy(env_ids=env_ids)
        return ee_pos, ee_quat

    def _compute_and_store_refs(self, env_ids: torch.Tensor | None = None) -> None:
        """
        현재 RAW desired를 기준으로 base ref(및 v/a/yaw_rate)를 계산해서
        obs/vis에서 쓰는 step buffer에 저장
        """
        if env_ids is None:
            ee_des_pos_W, ee_des_quat_W = self._compute_desired_ee_pose_world(env_ids=None)
            base_pos_ref, base_quat_ref, base_vel_ref, base_acc_ref, yaw_ref, yaw_rate_ref = (
                self._compute_base_ref_from_desired(ee_des_pos_W, ee_des_quat_W)
            )

            self._ee_des_pos_W.copy_(ee_des_pos_W)
            self._ee_des_quat_W.copy_(ee_des_quat_W)

            self._base_pos_ref.copy_(base_pos_ref)
            self._base_quat_ref.copy_(base_quat_ref)
            self._base_vel_ref.copy_(base_vel_ref)
            self._base_acc_ref.copy_(base_acc_ref)
            self._yaw_ref.copy_(yaw_ref)
            self._yaw_rate_ref.copy_(yaw_rate_ref)
            return

        # subset update (reset 시)
        ee_des_pos_W, ee_des_quat_W = self._compute_desired_ee_pose_world(env_ids=env_ids)
        base_pos_ref, base_quat_ref, base_vel_ref, base_acc_ref, yaw_ref, yaw_rate_ref = (
            self._compute_base_ref_from_desired(ee_des_pos_W, ee_des_quat_W, env_ids=env_ids)
        )

        self._ee_des_pos_W[env_ids] = ee_des_pos_W
        self._ee_des_quat_W[env_ids] = ee_des_quat_W

        self._base_pos_ref[env_ids] = base_pos_ref
        self._base_quat_ref[env_ids] = base_quat_ref
        self._base_vel_ref[env_ids] = base_vel_ref
        self._base_acc_ref[env_ids] = base_acc_ref
        self._yaw_ref[env_ids] = yaw_ref
        self._yaw_rate_ref[env_ids] = yaw_rate_ref

    def _compute_base_ref_pose_only(
        self,
        ee_pos_des_W: torch.Tensor,
        ee_quat_des_W: torch.Tensor,
        q_joint_12: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dtype = ee_pos_des_W.dtype
        T_W_E_des = tf_from_pos_quat_wxyz(ee_pos_des_W, ee_quat_des_W)

        T_B_E = fk_dh_T_base_to_ee_from_joints(
            q_joint_12=q_joint_12.to(device=ee_pos_des_W.device, dtype=dtype),
            dh_params=self._dh_params.to(dtype=dtype),
        )
        T_E_B = tf_inv(T_B_E)

        T_W_B_ref = torch.bmm(T_W_E_des, T_E_B)
        base_pos_ref = T_W_B_ref[:, 0:3, 3]
        base_quat_ref = math_utils.quat_unique(quat_from_rotmat_wxyz(T_W_B_ref[:, 0:3, 0:3]))

        _, _, yaw_ref = math_utils.euler_xyz_from_quat(base_quat_ref)

        base_pos_ref = torch.nan_to_num(base_pos_ref, nan=0.0, posinf=0.0, neginf=0.0)
        base_quat_ref = torch.nan_to_num(base_quat_ref, nan=0.0, posinf=0.0, neginf=0.0)
        yaw_ref = torch.nan_to_num(yaw_ref, nan=0.0, posinf=0.0, neginf=0.0)
        return base_pos_ref, base_quat_ref, yaw_ref

    def _compute_base_ref_from_desired(
        self,
        ee_pos_des_W: torch.Tensor,
        ee_quat_des_W: torch.Tensor,
        env_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        - policy/controller == sim Hz: dt = cfg.sim.dt
        - base ref는 현재 관절 상태(q_act) 기준으로 계산
        - v_ref/a_ref/yaw_rate_ref는 finite-diff raw
        """
        dtype = ee_pos_des_W.dtype
        device = ee_pos_des_W.device

        T_W_E_des = tf_from_pos_quat_wxyz(ee_pos_des_W, ee_quat_des_W)

        # q_act 사용
        if env_ids is None:
            q_for_ref = self.robot.data.joint_pos[:, self._joint_ids].to(device=device, dtype=dtype)
            base_pos_prev = self._base_pos_ref_prev
            base_vel_prev = self._base_vel_ref_prev
            yaw_prev = self._yaw_ref_prev
        else:
            q_for_ref = self.robot.data.joint_pos[:, self._joint_ids][env_ids].to(device=device, dtype=dtype)
            base_pos_prev = self._base_pos_ref_prev[env_ids]
            base_vel_prev = self._base_vel_ref_prev[env_ids]
            yaw_prev = self._yaw_ref_prev[env_ids]

        q_for_ref = torch.nan_to_num(q_for_ref, nan=0.0, posinf=0.0, neginf=0.0)

        T_B_E = fk_dh_T_base_to_ee_from_joints(
            q_joint_12=q_for_ref,
            dh_params=self._dh_params.to(dtype=dtype),
        )
        T_E_B = tf_inv(T_B_E)

        T_W_B_ref = torch.bmm(T_W_E_des, T_E_B)
        base_pos_ref = T_W_B_ref[:, 0:3, 3]
        base_quat_ref = math_utils.quat_unique(quat_from_rotmat_wxyz(T_W_B_ref[:, 0:3, 0:3]))

        _, _, yaw_ref = math_utils.euler_xyz_from_quat(base_quat_ref)
        yaw_ref = wrap_pi(yaw_ref)

        # finite diff (raw)
        dt = max(self._dt, 1e-6)
        v_raw = (base_pos_ref - base_pos_prev) / dt
        a_raw = (v_raw - base_vel_prev) / dt

        dyaw = wrap_pi(yaw_ref - yaw_prev)
        yaw_rate_raw = dyaw / dt

        # write back prev
        if env_ids is None:
            self._base_pos_ref_prev.copy_(base_pos_ref)
            self._base_vel_ref_prev.copy_(v_raw)
            self._yaw_ref_prev.copy_(yaw_ref)
        else:
            self._base_pos_ref_prev[env_ids] = base_pos_ref
            self._base_vel_ref_prev[env_ids] = v_raw
            self._yaw_ref_prev[env_ids] = yaw_ref

        # safety
        base_pos_ref = torch.nan_to_num(base_pos_ref, nan=0.0, posinf=0.0, neginf=0.0)
        base_quat_ref = torch.nan_to_num(base_quat_ref, nan=0.0, posinf=0.0, neginf=0.0)
        v_raw = torch.nan_to_num(v_raw, nan=0.0, posinf=0.0, neginf=0.0)
        a_raw = torch.nan_to_num(a_raw, nan=0.0, posinf=0.0, neginf=0.0)
        yaw_ref = torch.nan_to_num(yaw_ref, nan=0.0, posinf=0.0, neginf=0.0)
        yaw_rate_raw = torch.nan_to_num(yaw_rate_raw, nan=0.0, posinf=0.0, neginf=0.0)

        return base_pos_ref, base_quat_ref, v_raw, a_raw, yaw_ref, yaw_rate_raw

    def _resample_ee_goal(self, env_ids: torch.Tensor) -> None:
        n = int(env_ids.numel())
        if n == 0:
            return

        # 각 env의 원점 (world 좌표)
        env_origins = self.scene.env_origins[env_ids].to(device=self.device, dtype=torch.float32)  # (n,3)

        # ----------------------------
        # Fixed goal (no randomization)
        # ----------------------------
        if not bool(getattr(self.cfg, "randomize_ee_target", True)):
            # fixed pos/rpy/quaternion을 env origin 기준으로 이동
            self._ee_goal_pos[env_ids] = env_origins + self._ee_goal_pos_fixed[env_ids]
            self._ee_goal_rpy[env_ids] = self._ee_goal_rpy_fixed[env_ids]
            self._ee_goal_quat[env_ids] = self._ee_goal_quat_fixed[env_ids]
            return

        # ----------------------------
        # Random goal (randomization)
        # ----------------------------
        # position: [0,1) 샘플 + z_offset, 그리고 env origin 더하기
        u = torch.rand((n, 3), device=self.device, dtype=torch.float32)
        u[:, 2] += float(self.cfg.z_offset)
        self._ee_goal_pos[env_ids] = env_origins + u

        # yaw: uniform [min, max]
        yaw = (
            float(self.cfg.ee_target_yaw_min)
            + (float(self.cfg.ee_target_yaw_max) - float(self.cfg.ee_target_yaw_min))
            * torch.rand((n,), device=self.device, dtype=torch.float32)
        )

        # roll/pitch: 옵션에 따라 random 또는 0
        if bool(getattr(self.cfg, "randomize_goal_roll_pitch", False)):
            roll = (
                float(self.cfg.ee_target_roll_min)
                + (float(self.cfg.ee_target_roll_max) - float(self.cfg.ee_target_roll_min))
                * torch.rand((n,), device=self.device, dtype=torch.float32)
            )
            pitch = (
                float(self.cfg.ee_target_pitch_min)
                + (float(self.cfg.ee_target_pitch_max) - float(self.cfg.ee_target_pitch_min))
                * torch.rand((n,), device=self.device, dtype=torch.float32)
            )
        else:
            roll = torch.zeros((n,), device=self.device, dtype=torch.float32)
            pitch = torch.zeros((n,), device=self.device, dtype=torch.float32)

        self._ee_goal_rpy[env_ids, 0] = roll
        self._ee_goal_rpy[env_ids, 1] = pitch
        self._ee_goal_rpy[env_ids, 2] = wrap_pi(yaw)

        q = math_utils.quat_unique(
            math_utils.quat_from_euler_xyz(
                roll,
                pitch,
                self._ee_goal_rpy[env_ids, 2],
            )
        )
        self._ee_goal_quat[env_ids] = q

    # -----------------------------
    # geometric controller + allocation
    # -----------------------------
    def _geometric_controller(
        self,
        pos_I: torch.Tensor,
        quat_I: torch.Tensor,
        linvel_I: torch.Tensor,
        omega_I: torch.Tensor,
        J_I: torch.Tensor,
        p_ref_I: torch.Tensor,
        v_ref_I: torch.Tensor,
        a_ref_I: torch.Tensor,
        yaw_ref_I_t: torch.Tensor,
        yaw_rate_ref_I_t: torch.Tensor,
        cfg: MtrlAmEnvCfg,
    ) -> torch.Tensor:
        device = pos_I.device
        dtype = pos_I.dtype

        Kp_pos = _gain3_to_tensor(cfg.kp_pos, device=device, dtype=dtype)
        Kv_pos = _gain3_to_tensor(cfg.kv_pos, device=device, dtype=dtype)
        Kp_att = _gain3_to_tensor(cfg.kp_att, device=device, dtype=dtype)
        Kd_att = _gain3_to_tensor(cfg.kd_att, device=device, dtype=dtype)

        e3_L = torch.tensor([[0.0, 0.0, 1.0]], device=device, dtype=dtype)
        P = torch.diag(torch.tensor([1.0, -1.0, -1.0], device=device, dtype=dtype))
        P_batch = P.view(1, 3, 3)

        pos_L = pos_I @ P
        linvel_L = linvel_I @ P

        R_I = math_utils.matrix_from_quat(quat_I)  # wxyz
        R_L = torch.matmul(P_batch, torch.matmul(R_I, P_batch))
        R_L_T = R_L.transpose(1, 2)

        omega_w_L = omega_I @ P
        omega_b_L = torch.matmul(R_L_T, omega_w_L.unsqueeze(-1)).squeeze(-1)

        J_L = torch.matmul(P_batch, torch.matmul(J_I, P_batch))

        p_ref_L = p_ref_I @ P
        v_ref_L = v_ref_I @ P
        a_ref_L = a_ref_I @ P
        yaw_ref_L_t = -yaw_ref_I_t
        yaw_rate_ref_L_t = -yaw_rate_ref_I_t

        e_pos_L = pos_L - p_ref_L
        e_vel_L = linvel_L - v_ref_L

        A_L = (-(Kp_pos * e_pos_L) - (Kv_pos * e_vel_L) - cfg.mass * cfg.g * e3_L + cfg.mass * a_ref_L)

        A_norm = torch.linalg.norm(A_L, dim=-1, keepdim=True) + 1e-6
        b3d = -A_L / A_norm

        cos_yaw = torch.cos(yaw_ref_L_t)
        sin_yaw = torch.sin(yaw_ref_L_t)
        b1d = torch.stack([cos_yaw, sin_yaw, torch.zeros_like(cos_yaw)], dim=-1)

        b2d = F.normalize(torch.cross(b3d, b1d, dim=-1), dim=-1)
        b1d = torch.cross(b2d, b3d, dim=-1)

        R_d = torch.stack([b1d, b2d, b3d], dim=-1)
        R_d_T = R_d.transpose(1, 2)

        e_R_mat = 0.5 * (torch.matmul(R_d_T, R_L) - torch.matmul(R_L_T, R_d))
        e_R = vee(e_R_mat)

        omega_wd_L = torch.zeros_like(pos_L)
        omega_wd_L[..., 2] = yaw_rate_ref_L_t

        omega_bd_L = torch.matmul(R_d_T, omega_wd_L.unsqueeze(-1)).squeeze(-1)
        R_L_T_Rd = torch.matmul(R_L_T, R_d)
        RtRdOmd_L = torch.matmul(R_L_T_Rd, omega_bd_L.unsqueeze(-1)).squeeze(-1)

        e_omega_L = omega_b_L - RtRdOmd_L

        Re3_L = R_L[..., :, 2]
        f_des_L = -torch.sum(A_L * Re3_L, dim=-1)

        JOm_L = torch.matmul(J_L, omega_b_L.unsqueeze(-1)).squeeze(-1)
        cross_term_L = torch.cross(omega_b_L, JOm_L, dim=-1)

        omega_dot_bd_L = torch.zeros_like(omega_b_L)
        term_inner_L = torch.matmul(hat(omega_b_L), RtRdOmd_L.unsqueeze(-1)).squeeze(-1) - torch.matmul(
            R_L_T_Rd, omega_dot_bd_L.unsqueeze(-1)
        ).squeeze(-1)
        J_term_L = torch.matmul(J_L, term_inner_L.unsqueeze(-1)).squeeze(-1)

        M_des_L = (-(Kp_att * e_R) - (Kd_att * e_omega_L) + cross_term_L - J_term_L)

        M_des_I = M_des_L @ P
        u_des_I = torch.stack([f_des_L, M_des_I[..., 0], M_des_I[..., 1], M_des_I[..., 2]], dim=-1)
        return torch.nan_to_num(u_des_I, nan=0.0, posinf=0.0, neginf=0.0)

    def _motor_allocation(self, u_des_I: torch.Tensor, B_inv: torch.Tensor, cfg: MtrlAmEnvCfg) -> torch.Tensor:
        T_des = torch.matmul(u_des_I, B_inv.T)
        T_MAX = 1.5 * cfg.g
        return torch.clamp(torch.nan_to_num(T_des, nan=0.0, posinf=T_MAX, neginf=0.0), min=0.0, max=T_MAX)

    def _thrusts_to_body_wrenches(self, thrusts: torch.Tensor, D: torch.Tensor, G: torch.Tensor):
        F_I = torch.matmul(thrusts, D.T)
        M_I = torch.matmul(thrusts, G.T)
        return F_I.unsqueeze(1), M_I.unsqueeze(1)


# -----------------------------
# transform helpers
# -----------------------------
def tf_from_pos_quat_wxyz(pos: torch.Tensor, quat_wxyz: torch.Tensor) -> torch.Tensor:
    R = math_utils.matrix_from_quat(quat_wxyz)
    N = pos.shape[0]
    T = torch.zeros((N, 4, 4), device=pos.device, dtype=pos.dtype)
    T[:, 0:3, 0:3] = R
    T[:, 0:3, 3] = pos
    T[:, 3, 3] = 1.0
    return T


def tf_inv(T: torch.Tensor) -> torch.Tensor:
    R = T[:, 0:3, 0:3]
    p = T[:, 0:3, 3]
    R_T = R.transpose(1, 2)
    p_inv = -torch.bmm(R_T, p.unsqueeze(-1)).squeeze(-1)

    Ti = torch.zeros_like(T)
    Ti[:, 0:3, 0:3] = R_T
    Ti[:, 0:3, 3] = p_inv
    Ti[:, 3, 3] = 1.0
    return Ti


def quat_from_rotmat_wxyz(R: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    N = R.shape[0]
    q = torch.zeros((N, 4), device=R.device, dtype=R.dtype)

    tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]

    mask0 = tr > 0.0
    if mask0.any():
        idx0 = torch.nonzero(mask0, as_tuple=True)[0]
        S = torch.sqrt(tr[idx0] + 1.0) * 2.0
        q[idx0, 0] = 0.25 * S
        q[idx0, 1] = (R[idx0, 2, 1] - R[idx0, 1, 2]) / (S + eps)
        q[idx0, 2] = (R[idx0, 0, 2] - R[idx0, 2, 0]) / (S + eps)
        q[idx0, 3] = (R[idx0, 1, 0] - R[idx0, 0, 1]) / (S + eps)

    mask1 = ~mask0
    if mask1.any():
        idx1 = torch.nonzero(mask1, as_tuple=True)[0]
        Rm = R[idx1]
        diag = torch.stack([Rm[:, 0, 0], Rm[:, 1, 1], Rm[:, 2, 2]], dim=-1)
        which = torch.argmax(diag, dim=-1)

        m0 = which == 0
        if m0.any():
            id0 = idx1[torch.nonzero(m0, as_tuple=True)[0]]
            S = torch.sqrt(1.0 + R[id0, 0, 0] - R[id0, 1, 1] - R[id0, 2, 2]) * 2.0
            q[id0, 0] = (R[id0, 2, 1] - R[id0, 1, 2]) / (S + eps)
            q[id0, 1] = 0.25 * S
            q[id0, 2] = (R[id0, 0, 1] + R[id0, 1, 0]) / (S + eps)
            q[id0, 3] = (R[id0, 0, 2] + R[id0, 2, 0]) / (S + eps)

        m1 = which == 1
        if m1.any():
            id1b = idx1[torch.nonzero(m1, as_tuple=True)[0]]
            S = torch.sqrt(1.0 + R[id1b, 1, 1] - R[id1b, 0, 0] - R[id1b, 2, 2]) * 2.0
            q[id1b, 0] = (R[id1b, 0, 2] - R[id1b, 2, 0]) / (S + eps)
            q[id1b, 1] = (R[id1b, 0, 1] + R[id1b, 1, 0]) / (S + eps)
            q[id1b, 2] = 0.25 * S
            q[id1b, 3] = (R[id1b, 1, 2] + R[id1b, 2, 1]) / (S + eps)

        m2 = which == 2
        if m2.any():
            id2 = idx1[torch.nonzero(m2, as_tuple=True)[0]]
            S = torch.sqrt(1.0 + R[id2, 2, 2] - R[id2, 0, 0] - R[id2, 1, 1]) * 2.0
            q[id2, 0] = (R[id2, 1, 0] - R[id2, 0, 1]) / (S + eps)
            q[id2, 1] = (R[id2, 0, 2] + R[id2, 2, 0]) / (S + eps)
            q[id2, 2] = (R[id2, 1, 2] + R[id2, 2, 1]) / (S + eps)
            q[id2, 3] = 0.25 * S

    return math_utils.quat_unique(q)


# -----------------------------
# FK (DH): base -> ee
# -----------------------------
def dh_A_standard_batch(theta: torch.Tensor, d: torch.Tensor, a: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    ct = torch.cos(theta)
    st = torch.sin(theta)
    ca = torch.cos(alpha)
    sa = torch.sin(alpha)

    N = theta.shape[0]
    T = torch.zeros((N, 4, 4), device=theta.device, dtype=theta.dtype)

    T[:, 0, 0] = ct
    T[:, 0, 1] = -st * ca
    T[:, 0, 2] = st * sa
    T[:, 0, 3] = a * ct

    T[:, 1, 0] = st
    T[:, 1, 1] = ct * ca
    T[:, 1, 2] = -ct * sa
    T[:, 1, 3] = a * st

    T[:, 2, 0] = 0.0
    T[:, 2, 1] = sa
    T[:, 2, 2] = ca
    T[:, 2, 3] = d

    T[:, 3, 3] = 1.0
    return T


def fk_dh_T_base_to_ee_from_joints(q_joint_12: torch.Tensor, dh_params: torch.Tensor) -> torch.Tensor:
    assert q_joint_12.shape[1] == 2, f"q_joint_12 must be (N,2), got {q_joint_12.shape}"
    assert dh_params.shape == (4, 4), f"dh_params must be (4,4), got {dh_params.shape}"

    N = q_joint_12.shape[0]
    device = q_joint_12.device
    dtype = q_joint_12.dtype

    a = dh_params[:, 0].to(device=device, dtype=dtype)
    alpha = dh_params[:, 1].to(device=device, dtype=dtype)
    d = dh_params[:, 2].to(device=device, dtype=dtype)
    th0 = dh_params[:, 3].to(device=device, dtype=dtype)

    theta = th0.view(1, 4).expand(N, 4).clone()

    j1 = q_joint_12[:, 0]
    j2 = q_joint_12[:, 1]

    # NOTE: 사용자 기존 매핑 유지
    theta[:, 2] = theta[:, 2] - j1
    theta[:, 3] = theta[:, 3] - j2

    T = torch.eye(4, device=device, dtype=dtype).view(1, 4, 4).repeat(N, 1, 1)
    for i in range(4):
        Ai = dh_A_standard_batch(theta[:, i], d[i].expand(N), a[i].expand(N), alpha[i].expand(N))
        T = torch.bmm(T, Ai)
    return T


# -----------------------------
# allocation helpers
# -----------------------------
def build_allocation_mat(cfg: MtrlAmEnvCfg, device: torch.device | str):
    r_I, d_I = get_rotor_geometry(device=device)
    spin_dirs = torch.tensor([+1.0, -1.0, +1.0, -1.0], dtype=torch.float32, device=device)

    D = d_I.T  # (3,4)

    rxd = torch.cross(r_I, d_I, dim=-1)  # (4,3)
    RcrossD = rxd.T

    k_tau = float(cfg.k_tau)
    TauReact = (-(spin_dirs * k_tau).unsqueeze(-1) * d_I).T
    G = RcrossD + TauReact

    B = torch.empty((4, 4), device=device, dtype=torch.float32)
    B[0, :] = D[2, :]
    B[1, :] = G[0, :]
    B[2, :] = G[1, :]
    B[3, :] = G[2, :]

    B_inv = torch.linalg.inv(B)
    return D, G, B_inv


def get_rotor_geometry(device: torch.device | str):
    r_I = torch.tensor(
        [
            [0.163847, 0.163847, 0.064716],
            [0.163847, -0.163847, 0.064716],
            [-0.163847, -0.163847, 0.064716],
            [-0.163847, 0.163847, 0.064716],
        ],
        dtype=torch.float32,
        device=device,
    )

    d_I = torch.tensor(
        [
            [-0.073913, -0.073913, 0.994522],
            [-0.073913, 0.073913, 0.994522],
            [0.073913, 0.073913, 0.994522],
            [0.073913, -0.073913, 0.994521],
        ],
        dtype=torch.float32,
        device=device,
    )
    return r_I, d_I
