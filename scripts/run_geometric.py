#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Geometric SE(3) controller (Lee) on Isaac Sim quadrotor with frame conversion + keyboard-ref (pos/yaw) + smoothed v/a/yaw_rate + per-axis gains + joint1/2 keyboard control (Z/X, C/V). (multi-key hold supported)"
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import math
import time
from dataclasses import dataclass
from typing import Sequence, Union

import torch
import torch.nn.functional as F

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.actuators import ImplicitActuatorCfg, IdealPDActuatorCfg
from isaaclab.sim import SimulationContext
from isaaclab.assets.articulation import ArticulationCfg

# Omniverse keyboard input
import omni.appwindow
import carb.input


# -----------------------------
# 설정
# -----------------------------
Gain3 = Union[float, Sequence[float]]  # float이면 (g,g,g)로 취급, 시퀀스면 (gx,gy,gz)

@dataclass
class CONFIG:
    usd_path: str = "/home/yubinkim/workspace/mtrl_am/source/mtrl_am/mtrl_am/tasks/direct/mtrl_am/models/quad_mod.usd"

    # 시뮬/렌더
    dt: float = 1.0 / 300.0
    render_interval: int = 1
    device: str = "cuda:0"
    warmup_steps: int = 120

    # 씬(필수)
    num_envs: int = 1
    env_spacing: float = 2.0

    # 쿼드 파라미터
    mass: float = 2.48
    g: float = 9.81

    # 초기 ref (Isaac world / I-frame 기준)
    init_pos = torch.tensor([0.0, 0.0, 0.5])
    init_att = torch.tensor([0.0, 0.0, 0.0])  # roll, pitch, yaw (deg, I-frame)

    # 축별 게인 (L-frame 기준)
    kp_pos: float | tuple[float, float, float] = (25.0, 25.0, 50.0)
    kv_pos: float | tuple[float, float, float] = (10.0, 10.0, 25.0)
    kp_att: float | tuple[float, float, float] = (10.0, 10.0, 5.0)
    kd_att: float | tuple[float, float, float] = (0.5, 0.5, 1.0)

    # 키보드 ref step
    pos_step: float = 0.01                 # [m] per control tick while held
    yaw_step_deg: float = 1.0              # [deg] per control tick while held
    fast_mul: float = 5.0                  # Shift held
    z_min: float = 0.05

    # joint keyboard step / limits
    joint_step_deg: float = 1.0            # [deg] per control tick while held
    joint1_limit_deg: tuple[float, float] = (-180.0, 180.0)
    joint2_limit_deg: tuple[float, float] = (-180.0, 180.0)

    # rate smoothing (1st-order LPF time constants)
    tau_v: float = 0.15
    tau_a: float = 0.10
    tau_yaw_rate: float = 0.12

    # 필터 출력 제한
    v_ref_max: float = 3.0
    a_ref_max: float = 10.0
    yaw_rate_max: float = math.radians(180.0)

    # reaction torque scale
    k_tau: float = 1e-2



CFG = CONFIG()
CFG.num_envs = int(args_cli.num_envs)


# -----------------------------
# 씬 구성
# -----------------------------
def design_scene() -> dict[str, Articulation]:
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)

    cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    init_rpy_deg = CFG.init_att
    init_rpy_rad = torch.deg2rad(init_rpy_deg)

    roll = init_rpy_rad[0].unsqueeze(0)
    pitch = init_rpy_rad[1].unsqueeze(0)
    yaw = init_rpy_rad[2].unsqueeze(0)

    quat_batch = math_utils.quat_from_euler_xyz(roll, pitch, yaw)  # (1,4)
    init_quat = quat_batch[0]

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
        "Joint_2": 2500.0,
    }
    damping_dict = {
        "Joint_1": 50.0,
        "Joint_2": 25.0,
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
        spawn=sim_utils.UsdFileCfg(usd_path=CFG.usd_path),
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
            pos=CFG.init_pos.tolist(),
            rot=init_quat,
            joint_pos={"Joint_1": 0.0, "Joint_2": 0.0},
        ),
    )
    quad_cfg.prim_path = "/World/quad"
    quad = Articulation(cfg=quad_cfg)

    return {"quad": quad}


# -----------------------------
# 유틸
# -----------------------------
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


def wrap_pi_scalar(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def wrap_pi_diff(curr: float, prev: float) -> float:
    return wrap_pi_scalar(curr - prev)


def lpf_alpha(dt: float, tau: float) -> float:
    tau = max(float(tau), 1e-6)
    dt = max(float(dt), 1e-6)
    return dt / (tau + dt)


def gain3_to_tensor(g: Gain3, device: torch.device, dtype=torch.float32) -> torch.Tensor:
    if isinstance(g, (int, float)):
        gg = (float(g), float(g), float(g))
    else:
        if len(g) != 3:
            raise ValueError(f"Gain must be float or length-3 sequence, got len={len(g)}: {g}")
        gg = (float(g[0]), float(g[1]), float(g[2]))
    return torch.tensor(gg, device=device, dtype=dtype).view(1, 3)


# -----------------------------
# Geometric controller 함수
# -----------------------------
def geometric_controller(
    pos_I: torch.Tensor,
    linvel_I: torch.Tensor,
    quat_I: torch.Tensor,
    omega_I: torch.Tensor,
    J_I: torch.Tensor,
    p_ref_I: torch.Tensor,
    v_ref_I: torch.Tensor,
    a_ref_I: torch.Tensor,
    yaw_ref_I_t: torch.Tensor,
    yaw_rate_ref_I_t: torch.Tensor,
    cfg: CONFIG,
    P: torch.Tensor,
    P_batch: torch.Tensor,
    e3_L: torch.Tensor,
):
    device = pos_I.device
    N_env = pos_I.shape[0]

    Kp_pos = gain3_to_tensor(cfg.kp_pos, device=device, dtype=pos_I.dtype)
    Kv_pos = gain3_to_tensor(cfg.kv_pos, device=device, dtype=pos_I.dtype)
    Kp_att = gain3_to_tensor(cfg.kp_att, device=device, dtype=pos_I.dtype)
    Kd_att = gain3_to_tensor(cfg.kd_att, device=device, dtype=pos_I.dtype)

    pos_L = pos_I @ P
    linvel_L = linvel_I @ P

    R_I = math_utils.matrix_from_quat(quat_I)
    R_L = torch.matmul(P_batch, torch.matmul(R_I, P_batch))
    R_L_T = R_L.transpose(1, 2)

    omega_w_L = omega_I @ P
    omega_b_L = torch.bmm(R_L_T, omega_w_L.unsqueeze(-1)).squeeze(-1)

    J_L = torch.bmm(P_batch, torch.bmm(J_I, P_batch))

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

    cos_y = torch.cos(yaw_ref_L_t).expand(N_env)
    sin_y = torch.sin(yaw_ref_L_t).expand(N_env)
    b1d = torch.stack([cos_y, sin_y, torch.zeros_like(cos_y)], dim=-1)

    b2d = F.normalize(torch.cross(b3d, b1d, dim=-1), dim=-1)
    b1d = torch.cross(b2d, b3d, dim=-1)

    R_d = torch.stack([b1d, b2d, b3d], dim=-1)
    R_d_T = R_d.transpose(1, 2)

    e_R_mat = 0.5 * (torch.bmm(R_d_T, R_L) - torch.bmm(R_L_T, R_d))
    e_R = vee(e_R_mat)

    omega_wd_L = torch.zeros_like(pos_L)
    omega_wd_L[..., 2] = yaw_rate_ref_L_t

    omega_bd_L = torch.bmm(R_d_T, omega_wd_L.unsqueeze(-1)).squeeze(-1)
    R_L_T_Rd = torch.bmm(R_L_T, R_d)
    RtRdOmd_L = torch.bmm(R_L_T_Rd, omega_bd_L.unsqueeze(-1)).squeeze(-1)

    e_omega_L = omega_b_L - RtRdOmd_L

    Re3_L = R_L[..., :, 2]
    f_des_L = -torch.sum(A_L * Re3_L, dim=-1)

    JOm_L = torch.bmm(J_L, omega_b_L.unsqueeze(-1)).squeeze(-1)
    cross_term_L = torch.cross(omega_b_L, JOm_L, dim=-1)

    omega_dot_bd_L = torch.zeros_like(omega_b_L)
    term_inner_L = (
        torch.bmm(hat(omega_b_L), RtRdOmd_L.unsqueeze(-1)).squeeze(-1)
        - torch.bmm(R_L_T_Rd, omega_dot_bd_L.unsqueeze(-1)).squeeze(-1)
    )
    J_term_L = torch.bmm(J_L, term_inner_L.unsqueeze(-1)).squeeze(-1)

    M_des_L = (-(Kp_att * e_R) - (Kd_att * e_omega_L) + cross_term_L - J_term_L)

    M_des_I = M_des_L @ P
    u_des_I = torch.stack([f_des_L, M_des_I[..., 0], M_des_I[..., 1], M_des_I[..., 2]], dim=-1)
    u_des_I = torch.nan_to_num(u_des_I)

    return u_des_I, e_pos_L, e_R, e_omega_L


def get_rotor_geometry(device: torch.device):
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


def build_allocation_mat(cfg: CONFIG, device: torch.device):
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


def motor_allocation(u_des_I: torch.Tensor, B_inv: torch.Tensor, cfg: CONFIG):
    T_des = torch.matmul(u_des_I, B_inv.T)
    T_MAX = 1.5 * cfg.g
    return torch.clamp(T_des, min=0.0, max=T_MAX)


def thrusts_to_body_wrenches(thrusts: torch.Tensor, D: torch.Tensor, G: torch.Tensor):
    F_I = torch.matmul(thrusts, D.T)
    M_I = torch.matmul(thrusts, G.T)
    return F_I.unsqueeze(1), M_I.unsqueeze(1), F_I, M_I


# -----------------------------
# 키보드: multi-key hold 지원
# -----------------------------
class KeyboardRefCommander:
    """
    매핑:
      W/S : +X / -X
      D/A : +Y / -Y
      R/F : +Z / -Z
      Q/E : +yaw / -yaw

    joints:
      Z/X : Joint_1 + / -
      C/V : Joint_2 + / -

    기타:
      T   : reset ref_pos to current quad position (yaw unchanged)
      ESC : exit
      Shift : fast (x5)

    변경점:
      - KEY_PRESS/KEY_RELEASE로 keys_down 집합을 유지
      - 매 control tick마다 keys_down에 있는 모든 키를 합산 적용
    """
    def __init__(self, init_pos_xyz, init_yaw_rad, cfg: CONFIG):
        self.ref_pos = [float(init_pos_xyz[0]), float(init_pos_xyz[1]), float(init_pos_xyz[2])]
        self.ref_yaw = float(init_yaw_rad)

        self.ref_q1 = 0.0
        self.ref_q2 = 0.0

        self.pos_step = float(cfg.pos_step)
        self.yaw_step = math.radians(float(cfg.yaw_step_deg))
        self.fast_mul = float(cfg.fast_mul)
        self.z_min = float(cfg.z_min)

        self.joint_step = math.radians(float(cfg.joint_step_deg))
        self.q1_min = math.radians(float(cfg.joint1_limit_deg[0]))
        self.q1_max = math.radians(float(cfg.joint1_limit_deg[1]))
        self.q2_min = math.radians(float(cfg.joint2_limit_deg[0]))
        self.q2_max = math.radians(float(cfg.joint2_limit_deg[1]))

        self.reset_to_current_requested = False
        self.exit_requested = False

        # 현재 눌린 키 집합
        self.keys_down: set[int] = set()

        # shift modifier flag (optional)
        self._shift_flag = getattr(carb.input, "kKeyboardModifierFlagShift", None)

        # print rate
        self._last_print_t = 0.0
        self._print_period = 0.15

    def _maybe_print(self):
        now = time.time()
        if now - self._last_print_t < self._print_period:
            return
        self._last_print_t = now
        print(
            f"[REF] pos_I = [{self.ref_pos[0]:+.3f}, {self.ref_pos[1]:+.3f}, {self.ref_pos[2]:+.3f}]  "
            f"yaw_I = {math.degrees(self.ref_yaw):+.1f} deg   "
            f"q1={math.degrees(self.ref_q1):+.1f}deg  q2={math.degrees(self.ref_q2):+.1f}deg"
        )

    def on_keyboard_input(self, e):
        from carb.input import KeyboardEventType

        key = int(e.input)

        # press / repeat -> down
        if e.type in (KeyboardEventType.KEY_PRESS, KeyboardEventType.KEY_REPEAT):
            self.keys_down.add(key)

            # one-shot keys는 KEY_PRESS에서만 처리
            if e.type == KeyboardEventType.KEY_PRESS:
                if key == int(carb.input.KeyboardInput.T):
                    self.reset_to_current_requested = True
                    print("[REF] reset_to_current_requested = True")
                elif key == int(carb.input.KeyboardInput.ESCAPE):
                    self.exit_requested = True
                    print("[REF] exit_requested = True")

        # release -> up
        elif e.type == KeyboardEventType.KEY_RELEASE:
            if key in self.keys_down:
                self.keys_down.remove(key)

        return True

    def _shift_held(self) -> bool:
        # modifiers 기반이 가장 좋지만, 여기서는 키셋에도 포함 가능하도록 둘 다 지원
        if int(carb.input.KeyboardInput.LEFT_SHIFT) in self.keys_down:
            return True
        if int(carb.input.KeyboardInput.RIGHT_SHIFT) in self.keys_down:
            return True
        return False

    def apply_held_keys(self) -> bool:
        """
        현재 keys_down 상태를 한 번 적용.
        반환값: 이번 tick에 실제로 ref가 변했는지 여부
        """
        if not self.keys_down:
            return False

        mult = self.fast_mul if self._shift_held() else 1.0
        dp = self.pos_step * mult
        dy = self.yaw_step * mult
        dq = self.joint_step * mult

        moved = False

        # XY
        if int(carb.input.KeyboardInput.W) in self.keys_down:
            self.ref_pos[0] += dp; moved = True
        if int(carb.input.KeyboardInput.S) in self.keys_down:
            self.ref_pos[0] -= dp; moved = True
        if int(carb.input.KeyboardInput.D) in self.keys_down:
            self.ref_pos[1] += dp; moved = True
        if int(carb.input.KeyboardInput.A) in self.keys_down:
            self.ref_pos[1] -= dp; moved = True

        # Z
        if int(carb.input.KeyboardInput.R) in self.keys_down:
            self.ref_pos[2] += dp; moved = True
        if int(carb.input.KeyboardInput.F) in self.keys_down:
            self.ref_pos[2] -= dp; moved = True

        # yaw
        if int(carb.input.KeyboardInput.Q) in self.keys_down:
            self.ref_yaw += dy; moved = True
        if int(carb.input.KeyboardInput.E) in self.keys_down:
            self.ref_yaw -= dy; moved = True

        # joints
        if int(carb.input.KeyboardInput.Z) in self.keys_down:
            self.ref_q1 += dq; moved = True
        if int(carb.input.KeyboardInput.X) in self.keys_down:
            self.ref_q1 -= dq; moved = True
        if int(carb.input.KeyboardInput.C) in self.keys_down:
            self.ref_q2 += dq; moved = True
        if int(carb.input.KeyboardInput.V) in self.keys_down:
            self.ref_q2 -= dq; moved = True

        # clamp / wrap
        if self.ref_pos[2] < self.z_min:
            self.ref_pos[2] = self.z_min

        self.ref_yaw = wrap_pi_scalar(self.ref_yaw)

        self.ref_q1 = float(min(max(self.ref_q1, self.q1_min), self.q1_max))
        self.ref_q2 = float(min(max(self.ref_q2, self.q2_min), self.q2_max))

        if moved:
            self._maybe_print()

        return moved


def print_key_help():
    print(
        "\n========== Keyboard Reference Control ==========\n"
        "여러 키를 동시에 누른 상태(홀드)로도 입력이 합산됩니다. (예: W+D 대각 이동)\n\n"
        "Move ref_pos (world/I-frame):\n"
        "  W/S : +X / -X\n"
        "  D/A : +Y / -Y\n"
        "  R/F : +Z / -Z   (altitude)\n"
        "Yaw ref (I-frame):\n"
        "  Q/E : +yaw / -yaw\n"
        "Joint control (position target):\n"
        "  Z/X : Joint_1 + / -\n"
        "  C/V : Joint_2 + / -\n"
        "Others:\n"
        "  Shift : faster (x5)\n"
        "  T     : reset ref_pos to current quad position (yaw unchanged)\n"
        "  ESC   : exit\n"
        "==============================================\n"
    )


# -----------------------------
# 메인 루프
# -----------------------------
def run_simulator(sim: sim_utils.SimulationContext, scene: dict[str, Articulation]):
    sim_dt = sim.get_physics_dt()
    count = 0
    print("Simulation dt:", sim_dt)

    control_every = 1  # 4로 하면 60Hz 제어(240Hz 물리 기준)
    dt_ctrl = float(sim_dt * control_every)
    dt_safe = max(dt_ctrl, 1e-6)

    a_v = lpf_alpha(dt_safe, CFG.tau_v)
    a_a = lpf_alpha(dt_safe, CFG.tau_a)
    a_y = lpf_alpha(dt_safe, CFG.tau_yaw_rate)

    quad = scene["quad"]
    base_body_ids, _ = quad.find_bodies("Base")
    if len(base_body_ids) != 1:
        raise RuntimeError(f"Expected single Base body, got {base_body_ids}")
    base_body_ids = list(base_body_ids)
    device = quad.data.root_pos_w.device

    joint_ids, _ = quad.find_joints(["Joint_1", "Joint_2"])
    if len(joint_ids) != 2:
        raise RuntimeError(f"Expected Joint_1, Joint_2, got {joint_ids}")
    joint_ids = torch.as_tensor(joint_ids, device=device, dtype=torch.long)

    D, G, B_inv = build_allocation_mat(CFG, device)

    e3_L = torch.tensor([0.0, 0.0, 1.0], device=device).unsqueeze(0)
    P = torch.diag(torch.tensor([1.0, -1.0, -1.0], device=device))
    P_batch = P.view(1, 3, 3)

    init_yaw_I_deg = float(CFG.init_att[2].item())
    init_yaw_I_rad = math.radians(init_yaw_I_deg)

    commander = KeyboardRefCommander(CFG.init_pos.tolist(), init_yaw_I_rad, CFG)
    print_key_help()

    app_window = omni.appwindow.get_default_app_window()
    keyboard = app_window.get_keyboard()
    input_iface = carb.input.acquire_input_interface()
    keyboard_sub_id = input_iface.subscribe_to_keyboard_events(keyboard, commander.on_keyboard_input)

    for _ in range(CFG.warmup_steps):
        sim.render()

    p_ref_prev = torch.tensor(commander.ref_pos, device=device, dtype=torch.float32).view(1, 3).expand(CFG.num_envs, -1).clone()
    yaw_ref_prev = float(commander.ref_yaw)

    v_ref_f = torch.zeros_like(p_ref_prev)
    a_ref_f = torch.zeros_like(p_ref_prev)
    yaw_rate_f = 0.0
    v_ref_f_prev = v_ref_f.clone()

    try:
        while simulation_app.is_running():
            if commander.exit_requested:
                break

            if count % CFG.render_interval == 0:
                sim.render()

            if count % control_every == 0:
                # ✅ 현재 눌린 모든 키를 합산 적용
                commander.apply_held_keys()

                # 실제 상태 (I-frame)
                pos_I = quad.data.body_link_pos_w[:, base_body_ids, :].squeeze(1).to(device)
                linvel_I = quad.data.body_link_lin_vel_w[:, base_body_ids, :].squeeze(1).to(device)
                quat_I = quad.data.body_link_quat_w[:, base_body_ids, :].squeeze(1).to(device)
                omega_I = quad.data.body_link_ang_vel_w[:, base_body_ids, :].squeeze(1).to(device)
                J_I = quad.data.default_inertia[:, base_body_ids, :].reshape(-1, 3, 3).to(device)

                # T 키로 "현재 위치"로 ref_pos 리셋 (yaw 유지)
                if commander.reset_to_current_requested:
                    commander.ref_pos = [
                        float(pos_I[0, 0].item()),
                        float(pos_I[0, 1].item()),
                        float(pos_I[0, 2].item()),
                    ]
                    commander.reset_to_current_requested = False

                    p_ref_prev = torch.tensor(commander.ref_pos, device=device, dtype=torch.float32).view(1, 3).expand(CFG.num_envs, -1).clone()
                    v_ref_f.zero_()
                    a_ref_f.zero_()
                    v_ref_f_prev = v_ref_f.clone()

                    print("[REF] ref_pos reset to current quad position. (rates cleared)")

                # 현재 ref (I-frame)
                p_ref_I = torch.tensor(commander.ref_pos, device=device, dtype=torch.float32).view(1, 3).expand(CFG.num_envs, -1)

                v_ref_raw = (p_ref_I - p_ref_prev) / dt_safe

                yaw_ref_now = float(commander.ref_yaw)
                dyaw = wrap_pi_diff(yaw_ref_now, yaw_ref_prev)
                yaw_rate_raw = dyaw / dt_safe

                v_ref_f_prev = v_ref_f.clone()

                v_ref_f = (1.0 - a_v) * v_ref_f + a_v * v_ref_raw
                v_ref_f = torch.zeros_like(v_ref_f)
                if CFG.v_ref_max is not None and CFG.v_ref_max > 0:
                    v_norm = torch.linalg.norm(v_ref_f, dim=-1, keepdim=True) + 1e-9
                    v_ref_f = v_ref_f * torch.clamp(CFG.v_ref_max / v_norm, max=1.0)

                a_ref_raw = (v_ref_f - v_ref_f_prev) / dt_safe
                a_ref_f = torch.zeros_like(a_ref_f)
                a_ref_f = (1.0 - a_a) * a_ref_f + a_a * a_ref_raw
                if CFG.a_ref_max is not None and CFG.a_ref_max > 0:
                    a_norm = torch.linalg.norm(a_ref_f, dim=-1, keepdim=True) + 1e-9
                    a_ref_f = a_ref_f * torch.clamp(CFG.a_ref_max / a_norm, max=1.0)

                yaw_rate_f = (1.0 - a_y) * yaw_rate_f + a_y * yaw_rate_raw
                yaw_rate_f = 0.0
                if CFG.yaw_rate_max is not None and CFG.yaw_rate_max > 0:
                    yaw_rate_f = float(max(min(yaw_rate_f, CFG.yaw_rate_max), -CFG.yaw_rate_max))

                yaw_ref_I_t = torch.tensor(yaw_ref_now, device=device, dtype=torch.float32)
                yaw_rate_ref_I_t = torch.tensor(yaw_rate_f, device=device, dtype=torch.float32)

                u_des_I, e_pos_L, e_R, e_omega_L = geometric_controller(
                    pos_I=pos_I,
                    linvel_I=linvel_I,
                    quat_I=quat_I,
                    omega_I=omega_I,
                    J_I=J_I,
                    p_ref_I=p_ref_I,
                    v_ref_I=v_ref_f,
                    a_ref_I=a_ref_f,
                    yaw_ref_I_t=yaw_ref_I_t,
                    yaw_rate_ref_I_t=yaw_rate_ref_I_t,
                    cfg=CFG,
                    P=P,
                    P_batch=P_batch,
                    e3_L=e3_L,
                )

                T_clipped = motor_allocation(u_des_I=u_des_I, B_inv=B_inv, cfg=CFG)
                forces_b, torques_b, F_I, M_I = thrusts_to_body_wrenches(thrusts=T_clipped, D=D, G=G)

                quad.set_external_force_and_torque(
                    forces=forces_b,
                    torques=torques_b,
                    body_ids=base_body_ids,
                    env_ids=None,
                    is_global=False,
                )

                # joint position target
                joint_target = quad.data.joint_pos.clone()
                print("current joint :", joint_target[:, joint_ids])
                q_ref = torch.tensor([commander.ref_q1, commander.ref_q2], device=device, dtype=joint_target.dtype).view(1, 2)
                q_ref = q_ref.expand(CFG.num_envs, -1)
                joint_target[:, joint_ids] = q_ref

                if count % 5 == 0:
                    quad.set_joint_position_target(joint_target)
                print("ref joint     :", q_ref)

                p_ref_prev = p_ref_I.clone()
                yaw_ref_prev = yaw_ref_now

                print(quad.data.body_incoming_joint_wrench_b[:, joint_ids])
                print("thrusts :", T_clipped)


            quad.write_data_to_sim()
            sim.step()
            count += 1
            quad.update(sim_dt)

    finally:
        try:
            input_iface.unsubscribe_to_keyboard_events(keyboard, keyboard_sub_id)
        except Exception:
            pass


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=CONFIG.dt, render_interval=CONFIG.render_interval, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, 2.5, 1.8], [0.0, 0.0, 0.0])

    scene = design_scene()

    sim.reset()
    print("[INFO]: Setup complete...")
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
