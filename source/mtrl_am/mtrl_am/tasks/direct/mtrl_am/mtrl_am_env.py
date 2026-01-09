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

        # env-step dt (RL step)
        self._dt_env = float(cfg.sim.dt) * int(cfg.decimation)

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

        # desired EE state (integrated target)
        self._ee_des_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self._ee_des_rpy = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)  # roll,pitch,yaw
        self._q_des = torch.zeros((self.num_envs, 2), device=self.device, dtype=torch.float32)  # joint1, joint2

        # finite diff prev states (base ref)
        self._base_pos_ref_prev = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self._base_vel_ref_prev = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)

        self._yaw_ref_prev = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)

        # RL-step computed buffers
        self._ee_des_pos_W_buf = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self._ee_des_quat_W_buf = torch.zeros((self.num_envs, 4), device=self.device, dtype=torch.float32)

        self._base_pos_ref_buf = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self._base_quat_ref_buf = torch.zeros((self.num_envs, 4), device=self.device, dtype=torch.float32)
        self._base_vel_ref_buf = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self._base_acc_ref_buf = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self._yaw_ref_buf = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)
        self._yaw_rate_ref_buf = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)

        # marker indices
        self._marker_indices_frame = torch.zeros((self.num_envs,), device=self.device, dtype=torch.long)

        # ---- initialize desired to current state (pos + joints + base yaw) ----
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

        # ---- IMPORTANT: init base ref prev to avoid first finite-diff spike ----
        ee_des_pos_W0, ee_des_quat_W0 = self._compute_desired_ee_pose_world()
        # NOTE: 여기서는 q_des == q_act로 초기화되어 있으므로 그대로 사용해도 동일
        base_pos_ref0, base_quat_ref0, yaw_ref0 = self._compute_base_ref_pose_only(
            ee_pos_des_W=ee_des_pos_W0,
            ee_quat_des_W=ee_des_quat_W0,
            q_joint_12=self._q_des,
        )
        self._base_pos_ref_prev.copy_(base_pos_ref0)
        self._base_vel_ref_prev.zero_()
        self._yaw_ref_prev.copy_(yaw_ref0)

        # initialize refs/buffers
        self._refresh_refs_once_per_rl_step()

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
        # store prev_action
        self._prev_actions.copy_(self.actions)

        if actions is None or actions.numel() == 0:
            self.actions.zero_()
        else:
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

        self._update_desired_from_action()
        self._refresh_refs_once_per_rl_step()

    def _apply_action(self) -> None:
        base_pos, base_quat, base_linvel, base_angvel = self._get_body_state(self._base_body_id)

        ee_des_pos_W = self._ee_des_pos_W_buf
        ee_des_quat_W = self._ee_des_quat_W_buf

        base_pos_ref = self._base_pos_ref_buf
        base_quat_ref = self._base_quat_ref_buf
        base_vel_ref = self._base_vel_ref_buf
        base_acc_ref = self._base_acc_ref_buf
        yaw_ref = self._yaw_ref_buf
        yaw_rate_ref = self._yaw_rate_ref_buf

        # debug states
        ee_act_pos_W, ee_act_quat_W, _, _ = self._get_body_state(self._ee_body_id)
        q_act_12 = self.robot.data.joint_pos[:, self._joint_ids]

        # visualize
        if self._vis_ee_goal is not None:
            self._vis_ee_goal.visualize(self._ee_goal_pos, self._ee_goal_quat, marker_indices=self._marker_indices_frame)
        if self._vis_ee_des is not None:
            self._vis_ee_des.visualize(ee_des_pos_W, ee_des_quat_W, marker_indices=self._marker_indices_frame)
        if self._vis_base_ref is not None:
            self._vis_base_ref.visualize(base_pos_ref, base_quat_ref, marker_indices=self._marker_indices_frame)

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

        base_pos_ref = torch.zeros_like(base_pos_ref)
        base_vel_ref = torch.zeros_like(base_vel_ref)
        base_acc_ref = torch.zeros_like(base_acc_ref)
        yaw_ref = torch.zeros_like(yaw_ref)
        yaw_rate_ref = torch.zeros_like(yaw_rate_ref)
        base_pos_ref[:, 2] = 1.0  # keep height
        u = self._geometric_controller(
            pos_I=base_pos,
            quat_I=base_quat,
            linvel_I=base_linvel,
            omega_I=base_angvel,
            J_I=self.J_body,
            p_ref_I=base_pos_ref,
            v_ref_I=base_vel_ref,
            a_ref_I=base_acc_ref,
            yaw_ref_I_t=yaw_ref,
            yaw_rate_ref_I_t=yaw_rate_ref,
            cfg=self.cfg,
        )

        T = self._motor_allocation(u_des_I=u, B_inv=self.B_inv, cfg=self.cfg)
        self._motor_thrusts.copy_(T)
        print("thrusts:", T[0].cpu().numpy())

        forces_b, torques_b = self._thrusts_to_body_wrenches(thrusts=T, D=self.D, G=self.G)
        self.robot.set_external_force_and_torque(
            forces=forces_b,
            torques=torques_b,
            body_ids=self._base_body_ids,
            env_ids=None,
            is_global=False,
        )

        # joints: roll/pitch target  (FIX)
        joint_target = self.robot.data.joint_pos.clone().to(device=self.device, dtype=torch.float32)
        joint_target[:, self._joint_ids] = self._q_des.to(dtype=torch.float32)
        # joint_target[:, self._joint_ids] = torch.tensor([-math.pi / 4, -math.pi / 4], device=self.device, dtype=torch.float32).view(1, 2)
        
        # if joint_target[:, self._joint_ids[0]] < math.pi / 2:
        #     joint_target[:, self._joint_ids[0]] += 0.005
        joint_target[:, self._joint_ids[0]] = 0.0
        joint_target[:, self._joint_ids[1]] = 0.0

        self.robot.set_joint_position_target(target=joint_target)

        print(self.robot.data.body_incoming_joint_wrench_b[:, self._joint_ids])

    
        print("base_pos_ref:", base_pos_ref[0].cpu().numpy())
        print("vel_ref:", base_vel_ref[0].cpu().numpy())
        print("acc_ref:", base_acc_ref[0].cpu().numpy())
        print("q_des:", self._q_des[0].cpu().numpy())
        print("q_act:", q_act_12[0].cpu().numpy())
        print("joint_target:", joint_target[:, self._joint_ids].cpu().numpy())

    def _get_observations(self) -> dict:
        ee_pos, ee_quat, ee_linvel, ee_angvel = self._get_body_state(self._ee_body_id)
        base_pos, base_quat, base_linvel, base_angvel = self._get_body_state(self._base_body_id)
        q_act = self.robot.data.joint_pos[:, self._joint_ids]
        qd_act = self.robot.data.joint_vel[:, self._joint_ids]

        pos_err = ee_pos - self._ee_goal_pos

        R_ee = math_utils.matrix_from_quat(ee_quat)
        R_goal = math_utils.matrix_from_quat(self._ee_goal_quat)
        ee_rot_err_goal_so3 = so3_error_vee(R=R_ee, R_d=R_goal)

        # base tracking errors
        base_pos_ref = self._base_pos_ref_buf
        base_vel_ref = self._base_vel_ref_buf
        yaw_ref = self._yaw_ref_buf
        yaw_rate_ref = self._yaw_rate_ref_buf

        base_pos_err = base_pos - base_pos_ref
        base_vel_err = base_linvel - base_vel_ref

        _, _, yaw_base = math_utils.euler_xyz_from_quat(base_quat)
        base_yaw_err = wrap_pi(yaw_base - yaw_ref)

        yaw_rate_meas = base_angvel[:, 2]
        base_yaw_rate_err = yaw_rate_meas - yaw_rate_ref

        # EE desired-actual errors
        ee_des_pos = self._ee_des_pos_W_buf
        ee_des_quat = self._ee_des_quat_W_buf
        ee_pos_err_des = ee_pos - ee_des_pos

        R_des = math_utils.matrix_from_quat(ee_des_quat)
        ee_rot_err_des_so3 = so3_error_vee(R=R_ee, R_d=R_des)

        prev_act = self._prev_actions

        obs = torch.cat(
            [
                pos_err,
                ee_rot_err_goal_so3,
                ee_linvel,
                ee_angvel,
                base_linvel,
                base_angvel,
                q_act,
                qd_act,
                base_pos_err,
                base_vel_err,
                base_yaw_err.unsqueeze(-1),
                base_yaw_rate_err.unsqueeze(-1),
                ee_pos_err_des,
                ee_rot_err_des_so3,
                prev_act,
            ],
            dim=-1,
        )

        obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        ee_pos, ee_quat, _, _ = self._get_body_state(self._ee_body_id)

        pos_err2 = torch.sum((ee_pos - self._ee_goal_pos) ** 2, dim=-1)

        R_ee = math_utils.matrix_from_quat(ee_quat)
        R_goal = math_utils.matrix_from_quat(self._ee_goal_quat)
        e_R = so3_error_vee(R=R_ee, R_d=R_goal)

        rp_err2 = e_R[:, 0] ** 2 + e_R[:, 1] ** 2
        yaw_err2 = e_R[:, 2] ** 2

        pos_err2 = torch.nan_to_num(pos_err2, nan=0.0, posinf=1e6, neginf=1e6)
        rp_err2 = torch.nan_to_num(rp_err2, nan=0.0, posinf=1e6, neginf=1e6)
        yaw_err2 = torch.nan_to_num(yaw_err2, nan=0.0, posinf=1e6, neginf=1e6)

        r = (
            float(self.cfg.w_ee_pos) * pos_err2
            + float(self.cfg.w_ee_rp) * rp_err2
            + float(self.cfg.w_ee_yaw) * yaw_err2
        )

        # crash(terminated) penalty
        terminated, _ = self._get_dones()
        crash_penalty = float(getattr(self.cfg, "crash_penalty", -10.0))
        r = r + crash_penalty * terminated.to(dtype=r.dtype)

        return r

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

        # desired: set to current
        ee_pos, _, _, _ = self._get_body_state(self._ee_body_id)
        self._ee_des_pos[env_ids] = ee_pos[env_ids].clone()

        q_act = self.robot.data.joint_pos[:, self._joint_ids]
        self._q_des[env_ids] = q_act[env_ids].clone()

        _, base_quat, _, _ = self._get_body_state(self._base_body_id)
        _, _, yaw_base = math_utils.euler_xyz_from_quat(base_quat)
        self._ee_des_rpy[env_ids, 0] = self._q_des[env_ids, 0]
        self._ee_des_rpy[env_ids, 1] = self._q_des[env_ids, 1]
        self._ee_des_rpy[env_ids, 2] = yaw_base[env_ids]

        # reset ref prev to "current desired-derived ref" (avoid first diff spike)
        ee_des_pos_sub = self._ee_des_pos[env_ids]
        ee_des_quat_sub = self._quat_from_desired_rpy(env_ids)

        base_pos_ref_sub, base_quat_ref_sub, yaw_ref_sub = self._compute_base_ref_pose_only(
            ee_pos_des_W=ee_des_pos_sub,
            ee_quat_des_W=ee_des_quat_sub,
            q_joint_12=self._q_des[env_ids],
        )

        self._base_pos_ref_prev[env_ids] = base_pos_ref_sub
        self._base_vel_ref_prev[env_ids] = 0.0
        self._yaw_ref_prev[env_ids] = yaw_ref_sub

        # buffers
        self._ee_des_pos_W_buf[env_ids] = ee_des_pos_sub
        self._ee_des_quat_W_buf[env_ids] = ee_des_quat_sub
        self._base_pos_ref_buf[env_ids] = base_pos_ref_sub
        self._base_quat_ref_buf[env_ids] = base_quat_ref_sub
        self._base_vel_ref_buf[env_ids] = 0.0
        self._base_acc_ref_buf[env_ids] = 0.0
        self._yaw_ref_buf[env_ids] = yaw_ref_sub
        self._yaw_rate_ref_buf[env_ids] = 0.0

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
        (필터링 전부 제거 버전)
        action -> (scaled) increment -> integrate into desired
        """
        # 1) raw increments from action (scale only)
        dpos = float(self.cfg.dpos_scale) * self.actions[:, 0:3]
        droll = float(getattr(self.cfg, "droll_scale", 0.1)) * self.actions[:, 3]
        dpitch = float(getattr(self.cfg, "dpitch_scale", 0.1)) * self.actions[:, 4]
        dyaw = float(getattr(self.cfg, "dyaw_scale", 0.1)) * self.actions[:, 5]

        # 2) integrate increments into desired
        # self._ee_des_pos = self._ee_des_pos + dpos

        # roll/pitch -> joint targets
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

        # yaw desired (base handles)
        self._ee_des_rpy[:, 2] = wrap_pi(self._ee_des_rpy[:, 2] + dyaw)

        # keep desired rpy consistent
        self._ee_des_rpy[:, 0] = self._q_des[:, 0]
        self._ee_des_rpy[:, 1] = self._q_des[:, 1]

        # safety
        self._ee_des_pos = torch.nan_to_num(self._ee_des_pos, nan=0.0, posinf=0.0, neginf=0.0)
        self._q_des = torch.nan_to_num(self._q_des, nan=0.0, posinf=0.0, neginf=0.0)
        self._ee_des_rpy = torch.nan_to_num(self._ee_des_rpy, nan=0.0, posinf=0.0, neginf=0.0)

    def _quat_from_desired_rpy(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        if env_ids is None:
            r = self._ee_des_rpy[:, 0]
            p = self._ee_des_rpy[:, 1]
            y = self._ee_des_rpy[:, 2]
        else:
            r = self._ee_des_rpy[env_ids, 0]
            p = self._ee_des_rpy[env_ids, 1]
            y = self._ee_des_rpy[env_ids, 2]
        return math_utils.quat_unique(math_utils.quat_from_euler_xyz(r, p, y))

    def _compute_desired_ee_pose_world(self) -> tuple[torch.Tensor, torch.Tensor]:
        ee_quat_des_W = self._quat_from_desired_rpy(env_ids=None)
        return self._ee_des_pos, ee_quat_des_W

    def _refresh_refs_once_per_rl_step(self) -> None:
        ee_des_pos_W, ee_des_quat_W = self._compute_desired_ee_pose_world()

        base_pos_ref, base_quat_ref, base_vel_ref, base_acc_ref, yaw_ref, yaw_rate_ref = (
            self._compute_base_ref_from_desired(ee_des_pos_W, ee_des_quat_W)
        )

        self._ee_des_pos_W_buf.copy_(ee_des_pos_W)
        self._ee_des_quat_W_buf.copy_(ee_des_quat_W)

        self._base_pos_ref_buf.copy_(base_pos_ref)
        self._base_quat_ref_buf.copy_(base_quat_ref)
        self._base_vel_ref_buf.copy_(base_vel_ref)
        self._base_acc_ref_buf.copy_(base_acc_ref)
        self._yaw_ref_buf.copy_(yaw_ref)
        self._yaw_rate_ref_buf.copy_(yaw_rate_ref)

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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        변경점:
          - base ref 계산에 q_des 대신 q_act(현재 관절 상태)를 사용
          - ref 관련 필터링/클립 전부 제거 (finite-diff RAW만 사용)
        """
        dtype = ee_pos_des_W.dtype

        T_W_E_des = tf_from_pos_quat_wxyz(ee_pos_des_W, ee_quat_des_W)

        # --- IMPORTANT: q_act 사용 ---
        q_act_12 = self.robot.data.joint_pos[:, self._joint_ids].to(dtype=dtype)
        q_act_12 = torch.nan_to_num(q_act_12, nan=0.0, posinf=0.0, neginf=0.0)

        T_B_E = fk_dh_T_base_to_ee_from_joints(
            q_joint_12=self._q_des,
            dh_params=self._dh_params.to(dtype=dtype),
        )
        T_E_B = tf_inv(T_B_E)

        T_W_B_ref = torch.bmm(T_W_E_des, T_E_B)
        base_pos_ref = T_W_B_ref[:, 0:3, 3]
        base_quat_ref = math_utils.quat_unique(quat_from_rotmat_wxyz(T_W_B_ref[:, 0:3, 0:3]))

        _, _, yaw_ref = math_utils.euler_xyz_from_quat(base_quat_ref)
        yaw_ref = wrap_pi(yaw_ref)

        # finite diff (RAW, no filtering)
        dt = max(self._dt_env, 1e-6)

        v_raw = (base_pos_ref - self._base_pos_ref_prev) / dt
        v_raw = torch.zeros_like(v_raw)
        a_raw = (v_raw - self._base_vel_ref_prev) / dt
        a_raw = torch.zeros_like(a_raw)

        dyaw = wrap_pi(yaw_ref - self._yaw_ref_prev)
        yaw_rate_raw = dyaw / dt
        yaw_rate_raw = torch.zeros_like(yaw_rate_raw)

        # update prev states
        self._base_pos_ref_prev.copy_(base_pos_ref)
        self._base_vel_ref_prev.copy_(v_raw)
        self._yaw_ref_prev.copy_(yaw_ref)

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

        if not bool(getattr(self.cfg, "randomize_ee_target", True)):
            self._ee_goal_pos[env_ids] = self._ee_goal_pos_fixed[env_ids]
            self._ee_goal_rpy[env_ids] = self._ee_goal_rpy_fixed[env_ids]
            self._ee_goal_quat[env_ids] = self._ee_goal_quat_fixed[env_ids]
            return

        u = torch.rand((n, 3), device=self.device, dtype=torch.float32)
        self._ee_goal_pos[env_ids] = u
        self._ee_goal_pos[env_ids, 2] += float(self.cfg.z_offset)

        yaw = (
            float(self.cfg.ee_target_yaw_min)
            + (float(self.cfg.ee_target_yaw_max) - float(self.cfg.ee_target_yaw_min))
            * torch.rand((n,), device=self.device, dtype=torch.float32)
        )

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

        q = math_utils.quat_unique(math_utils.quat_from_euler_xyz(roll, pitch, self._ee_goal_rpy[env_ids, 2]))
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

        # print("e_pos_L:", e_pos_L[0].cpu().numpy())
        # print("e_vel_L:", e_vel_L[0].cpu().numpy())

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
def build_allocation_mat(cfg: MtrlAmEnvCfg, device: torch.device):
    r_I, d_I = get_rotor_geometry(device)
    spin_dirs = torch.tensor([+1.0, -1.0, +1.0, -1.0], dtype=torch.float32, device=device)

    D = d_I.T
    rxd = torch.cross(r_I, d_I, dim=-1)
    RcrossD = rxd.T

    k_tau = float(cfg.k_tau) if hasattr(cfg, "k_tau") else 1e-2
    TauReact = (-(spin_dirs * k_tau).unsqueeze(-1) * d_I).T

    G = RcrossD + TauReact

    B = torch.empty(4, 4, device=device, dtype=torch.float32)
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
