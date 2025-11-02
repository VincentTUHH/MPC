"""
UVMS Plant Simulation (standalone)
----------------------------------
A lightweight, configurable plant model you can call after your MPC to:
  • simulate with *mismatched* model parameters (vs. those used inside MPC)
  • choose attitude representation: quaternion or Euler
  • choose vehicle input mode: PWM (8 thrusters) or wrench/velocity-level (6 DOF)
  • advance the state with a single `step()` (Euler or RK4)

This file is intentionally independent of your MPC module. It rebuilds the
symbolic CasADi graphs against (possibly) different parameter sets so you can
stress-test robustness against model mismatch.

Dependencies expected in your repo (same as MPC side):
  - bluerov.dynamics_symbolic as sym_brv
  - manipulator.kinematics_symbolic as sym_manip_kin
  - manipulator.dynamics_symbolic as sym_manip_dyn
  - common.utils_sym / common.utils_math
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any

import numpy as np
import casadi as ca

from bluerov import dynamics_symbolic as sym_brv
from manipulator import (
    kinematics_symbolic as sym_manip_kin,
    dynamics_symbolic as sym_manip_dyn,
)
from common import utils_sym, utils_math
from common.my_package_path import get_package_path


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

def _apply_overrides(params: Dict[str, Any], overrides: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Return a shallow-copied dict with recursively updated keys from `overrides`.
    Use dotted.keys in overrides to patch nested dicts (e.g., "mass.m33": 1.1*value).
    """
    if overrides is None:
        return dict(params)
    out = dict(params)
    for k, v in overrides.items():
        if "." not in k:
            out[k] = v
            continue
        # dotted path
        path = k.split(".")
        cur = out
        for p in path[:-1]:
            if p not in cur or not isinstance(cur[p], dict):
                cur[p] = {}
            cur = cur[p]
        cur[path[-1]] = v
    return out


# --------------------------------------------------------------------------------------
# Simulation model
# --------------------------------------------------------------------------------------

@dataclass
class UVMSPlantSim:
    # --- toggles / structure ---
    use_quaternion: bool = True
    use_pwm: bool = True
    integrator: str = "euler"  # "euler" or "rk4"
    use_fixed_point: bool = False
    fixed_point_iter: int = 2

    # --- vehicle / actuation ---
    v_bat: float = 16.0

    # --- model parameter sources (paths in your repo) ---
    brv_params_path: Optional[str] = None
    dh_params_path: Optional[str] = None
    manip_dyn_params_paths: Optional[list[str]] = None
    manip_params: Optional[Dict[str, Any]] = None
    alpha_params: Optional[Dict[str, Any]] = None

    # --- optional overrides to induce *mismatch* vs MPC ---
    brv_overrides: Optional[Dict[str, Any]] = None
    manip_overrides: Optional[Dict[str, Any]] = None

    # --- (optional) joint constraints for clamping / sanity checks ---
    joint_limits_path: Optional[str] = None

    # --- built fields ---
    _built: Optional[bool] = None

    # dimensions
    N_DOF: Optional[int] = None
    N_JOINTS: Optional[int] = None
    STATE_DIM: Optional[int] = None
    CTRL_DIM: Optional[int] = None

    # casadi function handles
    TAU_COUPLING: Optional[ca.Function] = None
    DYN_FOSSEN: Optional[ca.Function] = None
    F_SYS: Optional[ca.Function] = None
    STEP: Optional[ca.Function] = None

    # pieces from symbolic models
    BLUEROV: Optional[Any] = None
    MANIP_KIN: Optional[Any] = None
    MANIP_DYN: Optional[Any] = None

    # vehicle matrices
    L: Optional[np.ndarray] = None
    MIXER: Optional[np.ndarray] = None
    M_INV: Optional[np.ndarray] = None
    C_FUN: Optional[Any] = None
    D_FUN: Optional[Any] = None
    G_FUN: Optional[Any] = None
    J_FUN: Optional[Any] = None

    # joint limits
    JOINT_LIMITS: Optional[np.ndarray] = None
    JOINT_VEL_MAX: Optional[np.ndarray] = None

    # internal cache
    _last_u: Optional[np.ndarray] = None

    # ----------------------------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------------------------

    def build(self) -> None:
        if self._built:
            return

        # --- Load base parameters as on the MPC side ---
        brv_params = utils_math.load_model_params(self.brv_params_path)
        if self.brv_overrides:
            brv_params = _apply_overrides(brv_params, self.brv_overrides)

        self.manip_params = utils_math.load_dh_params(self.dh_params_path)
        if self.manip_overrides:
            self.manip_params = _apply_overrides(self.manip_params, self.manip_overrides)

        if self.manip_dyn_params_paths is None:
            raise ValueError("manip_dyn_params_paths must be provided (same files as MPC).")
        self.alpha_params = utils_math.load_dynamic_params(self.manip_dyn_params_paths)
        # allow overrides also into alpha params tree
        if self.manip_overrides:
            self.alpha_params = _apply_overrides(self.alpha_params, self.manip_overrides)

        if self.joint_limits_path:
            jl, _, jv, _ = utils_math.load_joint_limits(self.joint_limits_path)
            self.JOINT_LIMITS = np.array(jl).T  # shape (2, nJ)
            self.JOINT_VEL_MAX = np.array(jv).T.reshape(-1)

        # --- Construct symbolic models with (possibly) different params ---
        self.BLUEROV = sym_brv.BlueROVDynamicsSymbolic(brv_params)
        self.MANIP_KIN = sym_manip_kin.KinematicsSymbolic(self.manip_params)
        self.MANIP_DYN = sym_manip_dyn.DynamicsSymbolic(self.MANIP_KIN, self.alpha_params)

        # --- vehicle functions (Euler vs Quaternion flavors) ---
        self.C_FUN = self.BLUEROV.C
        self.D_FUN = self.BLUEROV.D
        self.J_FUN = self.BLUEROV.J_quat if self.use_quaternion else self.BLUEROV.J
        self.G_FUN = self.BLUEROV.g_quat if self.use_quaternion else self.BLUEROV.g

        self.L = self.BLUEROV.L
        self.MIXER = self.BLUEROV.mixer
        self.M_INV = self.BLUEROV.M_inv

        self.N_JOINTS = self.MANIP_DYN.kinematics_.n_joints
        self.N_DOF = self.BLUEROV.M_inv.size1()

        self.STATE_DIM = self.N_JOINTS + self.N_DOF + (7 if self.use_quaternion else 6)
        self.CTRL_DIM = self.N_JOINTS + (8 if self.use_pwm else 6)

        # --- Build CasADi blocks ---
        self.TAU_COUPLING = self._build_tau_coupling()
        self.DYN_FOSSEN = self._build_dyn_fossen()
        eef_pose, J_eef = self._build_eef_blocks()
        self.F_SYS = self._build_f_sys(self.TAU_COUPLING, self.DYN_FOSSEN, eef_pose, J_eef)
        self.STEP = self._build_step(self.F_SYS)

        self._built = True

    # Single simulation step (user convenience) -------------------------------------
    def step(
        self,
        dt: float,
        x: np.ndarray,
        u: np.ndarray,
        *,
        u_prev: Optional[np.ndarray] = None,
        dnu_guess: Optional[np.ndarray] = None,
        f_eef: Optional[np.ndarray] = None,
        l_eef: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Advance plant by one step. Returns (x_next, dnu_pred).
        - x: state (q, nu, eta). Size = STATE_DIM.
        - u: control (uq, uv). Size = CTRL_DIM.
        - u_prev: previous control (for ddq estimate). If None, reuse last call; if
          still None, zeros.
        - dnu_guess: seed for fixed-point predictor; zeros if None.
        """
        assert self._built, "Call build() first."
        x = np.asarray(x).reshape(-1)
        u = np.asarray(u).reshape(-1)
        if u_prev is None:
            if self._last_u is None:
                u_prev = np.zeros_like(u)
            else:
                u_prev = self._last_u
        else:
            u_prev = np.asarray(u_prev).reshape(-1)

        ddq = (u[: self.N_JOINTS] - u_prev[: self.N_JOINTS]) / float(dt)
        dnu_guess = np.zeros(self.N_DOF) if dnu_guess is None else np.asarray(dnu_guess).reshape(-1)
        f_eef = np.zeros(3) if f_eef is None else np.asarray(f_eef).reshape(-1)
        l_eef = np.zeros(3) if l_eef is None else np.asarray(l_eef).reshape(-1)

        x_next, dnu, *_ = self.STEP(
            float(dt),
            x,
            u,
            ddq,
            dnu_guess,
            f_eef,
            l_eef,
        )
        self._last_u = u.copy()
        return np.array(x_next).flatten(), np.array(dnu).flatten()

    # ----------------------------------------------------------------------------------
    # Builders (mostly mirror of your MPC-side graph, kept compact & contained)
    # ----------------------------------------------------------------------------------

    def _build_tau_coupling(self) -> ca.Function:
        q = ca.MX.sym("q", self.N_JOINTS)
        dq = ca.MX.sym("dq", self.N_JOINTS)
        ddq = ca.MX.sym("ddq", self.N_JOINTS)
        v_ref = ca.MX.sym("v_ref", 3)
        a_ref = ca.MX.sym("a_ref", 3)
        w_ref = ca.MX.sym("w_ref", 3)
        dw_ref = ca.MX.sym("dw_ref", 3)
        quat_ref = ca.MX.sym("quat_ref", 4)
        f_eef = ca.MX.sym("f_eef", 3)
        l_eef = ca.MX.sym("l_eef", 3)

        kin = sym_manip_kin.KinematicsSymbolic(self.manip_params)
        dyn = sym_manip_dyn.DynamicsSymbolic(kin, self.alpha_params)
        dyn.kinematics_.update(q)
        tau = dyn.rnem_symbolic(q, dq, ddq, v_ref, a_ref, w_ref, dw_ref, quat_ref, f_eef, l_eef)
        return ca.Function(
            "rnem_func",
            [q, dq, ddq, v_ref, a_ref, w_ref, dw_ref, quat_ref, f_eef, l_eef],
            [tau],
        ).expand()

    def _build_dyn_fossen(self) -> ca.Function:
        nu = ca.MX.sym("nu", self.N_DOF)
        eta = ca.MX.sym("eta", 7 if self.use_quaternion else 6)
        uv = ca.MX.sym("uv", 8 if self.use_pwm else 6)
        tau_c = ca.MX.sym("tau_c", self.N_DOF)
        # uv is the 8 thruster commands [N] here
        tau_v = (self.MIXER @ uv) if self.use_pwm else uv
        dnu = self.M_INV @ (tau_v + tau_c - self.C_FUN(nu) @ nu - self.D_FUN(nu) @ nu - self.G_FUN(eta))
        return ca.Function("dyn_fossen", [eta, nu, uv, tau_c], [dnu]).expand()

    def _build_eef_blocks(self) -> tuple[ca.Function, ca.Function]:
        eta = ca.MX.sym("eta", 7 if self.use_quaternion else 6)
        q = ca.MX.sym("q", self.N_JOINTS)

        kin = sym_manip_kin.KinematicsSymbolic(self.manip_params)
        kin.update(q)

        R_I_B = (
            utils_sym.rotation_matrix_from_quat(eta[3:])
            if self.use_quaternion
            else utils_sym.rotation_matrix_from_euler(eta[3], eta[4], eta[5])
        )

        r_B_0, R_B_0 = self.MANIP_DYN.tf_vec, self.MANIP_DYN.R_reference
        r_0_eef = kin.get_eef_position()
        att_0_eef = kin.get_eef_attitude()
        J_pos, J_rot = kin.get_full_jacobian()

        p_eef = eta[0:3] + R_I_B @ r_B_0 + R_I_B @ R_B_0 @ r_0_eef
        R_eef = R_I_B @ R_B_0 @ utils_sym.rotation_matrix_from_quat(att_0_eef)
        att_eef = utils_sym.rotation_matrix_to_quaternion(R_eef)

        J_eef = ca.MX.zeros((self.N_DOF, self.N_DOF + self.N_JOINTS))
        J_eef[0:3, 0:3] = R_I_B
        J_eef[0:3, 3:6] = -utils_sym.skew(R_I_B @ r_B_0 + R_I_B @ R_B_0 @ r_0_eef) @ R_I_B
        J_eef[0:3, 6:] = R_I_B @ R_B_0 @ J_pos
        J_eef[3:6, 3:6] = R_I_B
        J_eef[3:6, 6:] = R_I_B @ R_B_0 @ J_rot

        f_pose = ca.Function("eef_pose", [eta, q], [p_eef, att_eef]).expand()
        f_Jeef = ca.Function("J_eef_fun", [eta, q], [J_eef]).expand()
        return f_pose, f_Jeef

    def _build_f_sys(
        self,
        tau_coupling: ca.Function,
        dyn_fossen: ca.Function,
        eef_pose: ca.Function,
        J_eef_fun: ca.Function,
    ) -> ca.Function:
        x = ca.MX.sym("x", self.STATE_DIM)
        u = ca.MX.sym("u", self.CTRL_DIM)
        ddq_in = ca.MX.sym("ddq_in", self.N_JOINTS)
        dnu_g = ca.MX.sym("dnu_guess", self.N_DOF)
        f_eef = ca.MX.sym("f_eef", 3)
        l_eef = ca.MX.sym("l_eef", 3)

        q = x[0 : self.N_JOINTS]
        nu = x[self.N_JOINTS : self.N_JOINTS + self.N_DOF]
        eta = x[self.N_JOINTS + self.N_DOF :]
        uq = u[0 : self.N_JOINTS]
        uv = u[self.N_JOINTS :]

        quat = (
            eta[3:]
            if self.use_quaternion
            else utils_sym.euler_to_quat(eta[3], eta[4], eta[5])
        )

        # Optional fixed-point prediction as in MPC
        dnu_fp = dnu_g
        if self.use_fixed_point:
            for _ in range(self.fixed_point_iter):
                a_ref = dnu_fp[0:3]
                dw_ref = dnu_fp[3:6]
                tau_c = tau_coupling(q, uq, ddq_in, nu[0:3], a_ref, nu[3:6], dw_ref, quat, f_eef, l_eef)
                dnu_fp = dyn_fossen(eta, nu, uv, tau_c)
        dnu_predict = dnu_fp

        dq = uq
        v_ref = nu[0:3]
        w_ref = nu[3:6]
        a_ref = dnu_predict[0:3]
        dw_ref = dnu_predict[3:6]

        tau_c = tau_coupling(q, dq, ddq_in, v_ref, a_ref, w_ref, dw_ref, quat, f_eef, l_eef)
        dnu = dyn_fossen(eta, nu, uv, tau_c)
        deta = self.J_FUN(eta) @ nu

        xdot = ca.vertcat(dq, dnu, deta)

        p_eef, att_eef = eef_pose(eta, q)
        J_eef = J_eef_fun(eta, q)

        return ca.Function(
            "f_sys",
            [x, u, ddq_in, dnu_g, f_eef, l_eef],
            [xdot, dnu, J_eef, p_eef, att_eef],
        ).expand()

    def _build_step(self, f_sys: ca.Function) -> ca.Function:
        dt = ca.MX.sym("dt")
        x = ca.MX.sym("x", self.STATE_DIM)
        u = ca.MX.sym("u", self.CTRL_DIM)
        ddq_in = ca.MX.sym("ddq_in", self.N_JOINTS)
        dnu_g = ca.MX.sym("dnu_guess", self.N_DOF)
        f_eef = ca.MX.sym("f_eef", 3)
        l_eef = ca.MX.sym("l_eef", 3)

        def normalize_quat(eta_vec):
            if self.use_quaternion:
                pos = eta_vec[0:3]
                q = eta_vec[3:]
                return ca.vertcat(pos, q / ca.norm_2(q))
            return eta_vec

        if self.integrator.lower() == "euler":
            k1, dnu1, J1, P1, A1 = f_sys(x, u, ddq_in, dnu_g, f_eef, l_eef)
            x_next = x + dt * k1
            x_next = ca.vertcat(
                x_next[0 : self.N_JOINTS + self.N_DOF],
                normalize_quat(x_next[self.N_JOINTS + self.N_DOF :]),
            )
            dnu = dnu1
            J_eef = J1
            p_eef = P1
            att_eef = A1
        elif self.integrator.lower() == "rk4":
            k1, d1, J1, P1, A1 = f_sys(x, u, ddq_in, dnu_g, f_eef, l_eef)
            k2, d2, _, _, _ = f_sys(x + 0.5 * dt * k1, u, ddq_in, dnu_g, f_eef, l_eef)
            k3, d3, _, _, _ = f_sys(x + 0.5 * dt * k2, u, ddq_in, dnu_g, f_eef, l_eef)
            k4, d4, _, _, _ = f_sys(x + dt * k3, u, ddq_in, dnu_g, f_eef, l_eef)

            x_next = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            x_next = ca.vertcat(
                x_next[0 : self.N_JOINTS + self.N_DOF],
                normalize_quat(x_next[self.N_JOINTS + self.N_DOF :]),
            )

            dnu = (d1 + 2 * d2 + 2 * d3 + d4) / 6.0
            J_eef = J1
            p_eef = P1
            att_eef = A1
        else:
            raise ValueError("integrator must be 'euler' or 'rk4'")

        return ca.Function(
            "step",
            [dt, x, u, ddq_in, dnu_g, f_eef, l_eef],
            [x_next, dnu, J_eef, p_eef, att_eef],
        ).expand()


# --------------------------------------------------------------------------------------
# Example usage (remove or keep for quick local testing)
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    # Minimal sanity check that the graph builds and a single step runs.
    # (Relies on your repo's config files being reachable by the default paths.)
    bluerov_package_path = get_package_path('bluerov')
    bluerov_params_path = bluerov_package_path + "/config/model_params.yaml"

    manipulator_package_path = get_package_path('manipulator')
    dh_params_path = manipulator_package_path + "/config/alpha_kin_params.yaml"
    base_tf_bluerov_path = manipulator_package_path + "/config/alpha_base_tf_params_bluerov.yaml"
    inertial_params_dh_path = manipulator_package_path + "/config/alpha_inertial_params_dh.yaml"

    manipulator_dyn_params_paths = [
        dh_params_path,
        base_tf_bluerov_path,
        inertial_params_dh_path
    ]

    sim = UVMSPlantSim(
        use_quaternion=True,
        use_pwm=True,
        integrator="euler",
        use_fixed_point=False,
        v_bat=16.0,
        brv_params_path=bluerov_params_path,
        dh_params_path=dh_params_path,
        manip_dyn_params_paths=manipulator_dyn_params_paths,
        # Example: make the sim a bit heavier in surge (model mismatch)
        # brv_overrides={"mass.m11": 1.05},
    )
    sim.build()

    N_J = sim.N_JOINTS
    N_V = sim.N_DOF
    x = np.zeros(sim.STATE_DIM)
    # init vehicle pose near (0,0,0), attitude = identity quat
    if sim.use_quaternion:
        x[N_J + N_V + 3 : N_J + N_V + 7] = np.array([1.0, 0.0, 0.0, 0.0])

    u = np.zeros(sim.CTRL_DIM)
    x_next, dnu = sim.step(0.05, x, u)
    print("x_next shape:", x_next.shape, "dnu shape:", dnu.shape)
