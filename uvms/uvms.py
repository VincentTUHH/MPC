import threading
from typing import Optional
import time

import numpy as np
import casadi as ca

from bluerov import dynamics_symbolic as sym_brv
from uvms import model as uvms_model
from uvms import Lumelsky

from manipulator import (
    kinematics as manip_kin,
    kinematics_symbolic as sym_manip_kin,
    dynamics_symbolic as sym_manip_dyn,
)

from common import utils_math, utils_sym, animate
from common.my_package_path import get_package_path
from common import animate
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# -----------------------
# Global, read-only variables after init()
# -----------------------
INITIALIZED: bool = False
_LOCK = threading.RLock()

# Model/context objects
BLUEROV_DYN = None           # bluerov.BlueROVDynamics
MANIP_DYN  = None            # manipulator dynamics
MANIP_KIN  = None            # manipulator kinematics
MANIP_KIN_REAL = None

# Symbolic/linearized functions (CasADi)
TAU_COUPLING: Optional[ca.Function] = None
DYN_FOSSEN: Optional[ca.Function] = None
J_EEF: Optional[ca.Function] = None
EEF_POSE: Optional[ca.Function] = None
F_SYS: Optional[ca.Function] = None
STEP: Optional[ca.Function] = None
ROLLOUT_UNROLLED: Optional[ca.Function] = None
LUMELSKY: Optional[ca.Function] = None
LUMELSKY_CASES: Optional[ca.Function] = None
COLLISION: Optional[ca.Function] = None
CONSTRAINT_COLLISION: Optional[ca.Function] = None
CONSTRAINT_TANK: Optional[ca.Function] = None

# Constants/config for fast access
USE_QUATERNION: bool = False
USE_PWM: bool = True
USE_FIXED_POINT: bool = True
FIXED_POINT_ITER: int = 2
V_BAT: float = 16.0
N_HORIZON: int = 10
INTEGRATOR: str = None
N_DOF: int = None
N_JOINTS: int = None
STATE_DIM: int = None
CTRL_DIM: int = None
JOINT_LIMITS: Optional[np.ndarray] = None
JOINT_EFFORTS: Optional[np.ndarray] = None
JOINT_VELOCITIES: Optional[np.ndarray] = None

# Constants from symbolic modules
L = None
MIXER = None
M_INV = None
C_FUN = None
D_FUN = None
G_FUN = None
J_FUN = None

# Q0 = np.array([0.1, 3*np.pi/4, np.pi/2, 0.0])  # Initial joint angles # arm down
Q0 = np.array([np.pi, np.pi/2, 3*np.pi/4, np.pi/5])  # Initial joint angles # arm foward
POS0 = np.array([1.0, 2.0, -0.7])
ATT0_EULER = np.array([0.0, 0.0, 0.0])
ATT0_QUAT = utils_math.euler_to_quat(ATT0_EULER[0], ATT0_EULER[1], ATT0_EULER[2])
VEL0 = np.array([0.0, 0.0, 0.0])
OMEGA0 = np.array([0.0, 0.0, 0.0])
UVMS_MODEL_INSTANCE: Optional[uvms_model.UVMSModel] = None

MANIP_PARAMS: Optional[dict] = None
ALPHA_PARAMS: Optional[dict] = None
BRV_PARAMS: Optional[dict] = None

def _constraint_tank() -> ca.Function:
    vehicle_bound = ca.MX.sym('vehicle_bound', 7)
    underarm_bound = ca.MX.sym('underarm_bound', 7) # swept-sphere

    beta_smoothmin = ca.DM(10.0)
    safety_margin = ca.DM(0.05)
    beta_margin = ca.DM(0.2) # must be greater than safety_margin
    eta_weight = ca.DM(4.0)

    p1_vehicle = vehicle_bound[0:3]
    p2_vehicle = vehicle_bound[3:6]
    radius_vehicle = vehicle_bound[6]

    p1_underarm = underarm_bound[0:3]
    p2_underarm = underarm_bound[3:6]
    radius_underarm = underarm_bound[6]

    # Define the points and normals for 6 planes as CasADi DM arrays
    plane_points = [
        ca.DM([0.0, 0.5, -1.3]),  # Plane 1
        ca.DM([0.0, 0.5, -1.3]),  # Plane 2
        ca.DM([0.0, 0.5, -1.3]),  # Plane 3
        ca.DM([2.0, 3.5, 0.0]),   # Plane 4
        ca.DM([2.0, 3.5, 0.0]),   # Plane 5
        ca.DM([2.0, 3.5, 0.0]),   # Plane 6
    ]

    plane_normals = [
        ca.DM([ 1.0,  0.0,  0.0]),  # Plane 1
        ca.DM([ 0.0,  1.0,  0.0]),  # Plane 2
        ca.DM([ 0.0,  0.0,  1.0]),  # Plane 3
        ca.DM([-1.0,  0.0,  0.0]),  # Plane 4
        ca.DM([ 0.0, -1.0,  0.0]),  # Plane 5
        ca.DM([ 0.0,  0.0, -1.0]),  # Plane 6
    ]

    # Compute signed distances from each plane for vehicle and underarm
    g_vehicle = []
    g_underarm = []
    rho_vehicle = []
    rho_underarm = []
    for i in range(len(plane_points)):
        point = plane_points[i]
        normal = plane_normals[i]
        # Signed distance: dot(normal, (x - point))
        d1_vehicle = ca.dot(normal, p1_vehicle - point)
        d2_vehicle = ca.dot(normal, p2_vehicle - point)
        d1_underarm = ca.dot(normal, p1_underarm - point)
        d2_underarm = ca.dot(normal, p2_underarm - point)
        dmin_vehicle = utils_sym.smoothmin(d1_vehicle, d2_vehicle, beta_smoothmin) - radius_vehicle
        dmin_underarm = utils_sym.smoothmin(d1_underarm, d2_underarm, beta_smoothmin) - radius_underarm
        g_vehicle.append(dmin_vehicle - safety_margin)
        g_underarm.append(dmin_underarm - safety_margin)
        rho_vehicle.append(Lumelsky.seperation_cost_symbolic(eta=eta_weight, beta=beta_margin, d=dmin_vehicle))
        rho_underarm.append(Lumelsky.seperation_cost_symbolic(eta=eta_weight, beta=beta_margin, d=dmin_underarm))


    g_vehicle = ca.vertcat(*g_vehicle)
    g_underarm = ca.vertcat(*g_underarm)

    cost_vehicle = ca.sum(ca.vertcat(*rho_vehicle))
    cost_underarm = ca.sum(ca.vertcat(*rho_underarm))

    return ca.Function('constraint_tank', [vehicle_bound, underarm_bound], [g_vehicle, g_underarm, cost_vehicle, cost_underarm]).expand()

def _constraint_collision(collision_test: ca.Function) -> ca.Function:
    p_vehicle = ca.MX.sym('p_vehicle', 3)
    att_vehicle = ca.MX.sym('att_vehicle', 4) # quaternion [w,x,y,z]
    p_eef = ca.MX.sym('p_eef', 3)
    att_eef = ca.MX.sym('att_eef', 4) # quaternion [w,x,y,z]

    # Self-Collision vehicle - underarm
    radius_vehicle = ca.DM(0.245) # meters
    radius_underarm = ca.DM(0.025) # meters

    # p_eef_offset = ca.DM([0.03, 0.0, 0.0])
    # p_elbow_offset = ca.DM([-0.18, 0.0, 0.0])
    p_vehicle_offset = ca.DM([-0.003, 0.0, 0.022])
    p_vehicle_pill_offset = ca.DM([0.0, 0.1375, 0.0]) # offset to pill end points left and right (L=0.275m/2)
    safety_margin = ca.DM(0.02) # meters # alpha_L
    beta_margin = ca.DM(0.05) # must be greater than safety_margin
    eta_weight = ca.DM(10.0)
    p_unit = ca.DM([1.0, 0.0, 0.0])

    # p_underarm1 = p_eef + utils_sym.quaternion_rotation(att_eef, p_eef_offset)
    # p_underarm2 = p_eef + utils_sym.quaternion_rotation(att_eef, p_elbow_offset)

    p_unit_rot = utils_sym.quaternion_rotation(att_eef, p_unit)
    p_underarm1 = p_eef + p_unit_rot * 0.03
    p_underarm2 = p_eef - p_unit_rot * 0.18

    pill_offset = utils_sym.quaternion_rotation(att_vehicle, p_vehicle_pill_offset)
    vehicle_offset = utils_sym.quaternion_rotation(att_vehicle, p_vehicle_offset)
    p_vehicle1 = p_vehicle + vehicle_offset - pill_offset
    p_vehicle2 = p_vehicle + vehicle_offset + pill_offset

    obj1 = ca.vertcat(p_vehicle1, p_vehicle2, radius_vehicle)
    obj2 = ca.vertcat(p_underarm1, p_underarm2, radius_underarm)

    g_no_collision, d = collision_test(obj1, obj2, safety_margin) # >= 0 means no collision

    rho = Lumelsky.seperation_cost_symbolic(eta=eta_weight, beta=beta_margin, d=d) # beta [m] must be greater than safety margin to be effective

    # g_no_collision = ca.dot(p_vehicle - p_eef, p_vehicle - p_eef)

    return ca.Function('constraint_collision', [p_vehicle, att_vehicle, p_eef, att_eef], [g_no_collision, obj1, obj2, rho]).expand()
    
def _check_collision(lumelsky_fun: ca.Function) -> ca.Function:
    # obj1 = [p1, p2, radius1] # swept-sphere
    # obj2 = [p3, p4, radius2] # swept-sphere
    obj1 = ca.MX.sym('obj1', 7)
    obj2 = ca.MX.sym('obj2', 7)
    safety_margin = ca.MX.sym('safety_margin', 1) # in meters

    # Segment 1
    p11 = obj1[0:3]      # Start point of segment 1 (3D)
    p12 = obj1[3:6]      # End point of segment 1 (3D)
    radius1 = obj1[6]    # Radius of segment 1

    # Segment 2
    p21 = obj2[0:3]      # Start point of segment 2 (3D)
    p22 = obj2[3:6]      # End point of segment 2 (3D)
    radius2 = obj2[6]    # Radius of segment 2

    beta = 20.0 # softclip sharpness. higher = closer to hard clamp, but sharper gradients
    reg_eps = 1e-6 # scale with segment lengths
    k_par = 10.0 # how aggressively to regularize near parallelism
    tau = 0.005 # blend width into the parallel fallback

    # d2, _, _ = lumelsky_fun(p11, p12, p21, p22, beta, reg_eps, k_par, tau)
    d2, _, _ = lumelsky_fun(p11, p12, p21, p22)

    r_sum = radius1 + radius2

    g_no_collision = d2 - (safety_margin + r_sum)**2 # >= 0 means no collision

    #TODO: find computational cheaper version to sqrt
    # Numerically safe sqrt for differentiation
    eps = 1e-8
    d = ca.sqrt(d2 + eps) - r_sum

    return ca.Function('collision_check', [obj1, obj2, safety_margin], [g_no_collision, d]).expand()

def _build_tau_coupling_func() -> ca.Function:
    q = ca.MX.sym('q', N_JOINTS)
    dq = ca.MX.sym('dq', N_JOINTS)
    ddq = ca.MX.sym('ddq', N_JOINTS)
    v_ref = ca.MX.sym('v_ref', 3)
    a_ref = ca.MX.sym('a_ref', 3)
    w_ref = ca.MX.sym('w_ref', 3)
    dw_ref = ca.MX.sym('dw_ref', 3)
    quaternion_ref = ca.MX.sym('quat_ref', 4)
    f_eef = ca.MX.sym('f_eef', 3)
    l_eef = ca.MX.sym('l_eef', 3)

    kin = sym_manip_kin.KinematicsSymbolic(MANIP_PARAMS)
    dyn = sym_manip_dyn.DynamicsSymbolic(kin, ALPHA_PARAMS)

    dyn.kinematics_.update(q)

    tau = dyn.rnem_symbolic(q, dq, ddq, v_ref, a_ref, w_ref, dw_ref, quaternion_ref, f_eef, l_eef)
    return ca.Function(
        'rnem_func',
        [q, dq, ddq, v_ref, a_ref, w_ref, dw_ref, quaternion_ref, f_eef, l_eef],
        [tau]
    ).expand()

def _build_dynamics_fossen_func() -> ca.Function:
    nu   = ca.MX.sym('nu', N_DOF)
    eta  = ca.MX.sym('eta', 7 if USE_QUATERNION else 6)
    uv   = ca.MX.sym('uv', 8 if USE_PWM else 6)
    tau_c = ca.MX.sym('tau_c', N_DOF)

    tau_v = (L * V_BAT * (MIXER @ uv)) if USE_PWM else uv
    dnu  = M_INV @ (tau_v + tau_c - C_FUN(nu) @ nu - D_FUN(nu) @ nu - G_FUN(eta))
    return ca.Function('dyn_fossen', [eta,nu,uv,tau_c], [dnu]).expand()

def _build_eef_blocks() -> tuple[ca.Function, ca.Function]:
    eta = ca.MX.sym('eta', 7 if USE_QUATERNION else 6)
    q   = ca.MX.sym('q', N_JOINTS)

    kin = sym_manip_kin.KinematicsSymbolic(MANIP_PARAMS)
    kin.update(q)

    R_I_B = utils_sym.rotation_matrix_from_quat(eta[3:]) if USE_QUATERNION \
            else utils_sym.rotation_matrix_from_euler(eta[3], eta[4], eta[5])

    r_B_0, R_B_0 = MANIP_DYN.tf_vec, MANIP_DYN.R_reference
    r_0_eef      = kin.get_eef_position()
    att_0_eef    = kin.get_eef_attitude()
    J_pos, J_rot = kin.get_full_jacobian()

    p_eef   = eta[0:3] + R_I_B @ r_B_0 + R_I_B @ R_B_0 @ r_0_eef
    R_eef   = R_I_B @ R_B_0 @ utils_sym.rotation_matrix_from_quat(att_0_eef)
    att_eef = utils_sym.rotation_matrix_to_quaternion(R_eef)

    J_eef = ca.MX.zeros((N_DOF, N_DOF + N_JOINTS))
    J_eef[0:3, 0:3]      = R_I_B
    J_eef[0:3, 3:6]      = -utils_sym.skew(R_I_B @ r_B_0 + R_I_B @ R_B_0 @ r_0_eef) @ R_I_B
    J_eef[0:3, 6:]       = R_I_B @ R_B_0 @ J_pos
    J_eef[3:6, 3:6]      = R_I_B
    J_eef[3:6, 6:]       = R_I_B @ R_B_0 @ J_rot

    f_pose = ca.Function('eef_pose',  [eta, q], [p_eef, att_eef]).expand()
    f_Jeef = ca.Function('J_eef_fun', [eta, q], [J_eef]).expand()
    return f_pose, f_Jeef

def _build_f_sys(
    tau_coupling: ca.Function,
    dyn_fossen: ca.Function,
    eef_pose: ca.Function,
    J_eef_fun: ca.Function,
) -> ca.Function:
    x      = ca.MX.sym('x', STATE_DIM)
    u      = ca.MX.sym('u', CTRL_DIM)
    ddq_in = ca.MX.sym('ddq_in', N_JOINTS)
    dnu_g  = ca.MX.sym('dnu_guess', N_DOF)
    f_eef  = ca.MX.sym('f_eef', 3)
    l_eef  = ca.MX.sym('l_eef', 3)

    q   = x[0:N_JOINTS]
    nu  = x[N_JOINTS:N_JOINTS+N_DOF]
    eta = x[N_JOINTS+N_DOF:]
    uq  = u[0:N_JOINTS]
    uv  = u[N_JOINTS:]

    quat  = eta[3:] if USE_QUATERNION else utils_sym.euler_to_quat(eta[3], eta[4], eta[5])

    dnu_fp = dnu_g
    if USE_FIXED_POINT:
        for _ in range(FIXED_POINT_ITER):
            a_ref  = dnu_fp[0:3]
            dw_ref = dnu_fp[3:6]
            tau_c = tau_coupling(q, uq, ddq_in, nu[0:3], a_ref, nu[3:6], dw_ref, quat, f_eef, l_eef)
            dnu_fp = dyn_fossen(eta, nu, uv, tau_c)

    dnu_predict = dnu_fp

    dq    = uq
    v_ref = nu[0:3]; w_ref = nu[3:6]
    a_ref = dnu_predict[0:3]; dw_ref = dnu_predict[3:6]

    tau_c = tau_coupling(q, dq, ddq_in, v_ref, a_ref, w_ref, dw_ref, quat, f_eef, l_eef)
    dnu   = dyn_fossen(eta, nu, uv, tau_c)
    deta  = J_FUN(eta) @ nu

    xdot = ca.vertcat(dq, dnu, deta)

    p_eef, att_eef = eef_pose(eta, q)
    J_eef          = J_eef_fun(eta, q)

    return ca.Function('f_sys', [x, u, ddq_in, dnu_g, f_eef, l_eef],
                       [xdot, dnu, J_eef, p_eef, att_eef]).expand()

def _build_step_func(f_sys: ca.Function) -> ca.Function:
    dt     = ca.MX.sym('dt')
    x      = ca.MX.sym('x', STATE_DIM)
    u      = ca.MX.sym('u', CTRL_DIM)
    ddq_in = ca.MX.sym('ddq_in', N_JOINTS)
    dnu_g  = ca.MX.sym('dnu_guess', N_DOF)
    f_eef  = ca.MX.sym('f_eef', 3)
    l_eef  = ca.MX.sym('l_eef', 3)

    def normalize_quat(eta_vec):
        if USE_QUATERNION:
            pos = eta_vec[0:3]; q = eta_vec[3:]
            return ca.vertcat(pos, q / ca.norm_2(q))
        return eta_vec

    if INTEGRATOR == "euler":
        k1, dnu1, J1, P1, A1 = f_sys(x, u, ddq_in, dnu_g, f_eef, l_eef)
        x_next = x + dt * k1
        x_next = ca.vertcat(x_next[0:N_JOINTS + N_DOF], normalize_quat(x_next[N_JOINTS + N_DOF:]))
        dnu = dnu1; J_eef = J1; p_eef = P1; att_eef = A1

    elif INTEGRATOR == "rk4":
        k1, d1, J1, P1, A1 = f_sys(x, u, ddq_in, dnu_g, f_eef, l_eef)
        k2, d2, _,  _,  _  = f_sys(x + 0.5*dt*k1, u, ddq_in, dnu_g, f_eef, l_eef)
        k3, d3, _,  _,  _  = f_sys(x + 0.5*dt*k2, u, ddq_in, dnu_g, f_eef, l_eef)
        k4, d4, _,  _,  _  = f_sys(x + dt*k3,     u, ddq_in, dnu_g, f_eef, l_eef)

        x_next = x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        x_next = ca.vertcat(x_next[0:N_JOINTS + N_DOF], normalize_quat(x_next[N_JOINTS + N_DOF:]))

        dnu   = (d1 + 2*d2 + 2*d3 + d4)/6.0
        J_eef = J1; p_eef = P1; att_eef = A1
    else:
        raise ValueError("INTEGRATOR must be 'euler' or 'rk4'")

    return ca.Function('step', [dt, x, u, ddq_in, dnu_g, f_eef, l_eef],
                       [x_next, dnu, J_eef, p_eef, att_eef]).expand()

def _build_rollout_unrolled(step_fun: ca.Function, N: int) -> ca.Function:
    x0     = ca.MX.sym('x0', STATE_DIM)
    U      = ca.MX.sym('U',  CTRL_DIM, N)
    Uprev0 = ca.MX.sym('Uprev0', CTRL_DIM)
    dt     = ca.MX.sym('dt')
    dnu0   = ca.MX.sym('dnu0', N_DOF)
    f_eef  = ca.MX.sym('f_eef', 3)
    l_eef  = ca.MX.sym('l_eef', 3)

    X_list   = [x0]
    DNU_cols = []; J_cols = []; P_cols = []; A_cols = []

    xk     = x0
    u_prev = Uprev0
    dnu_g  = dnu0

    for k in range(N):
        uk     = U[:, k]
        ddq_k  = (uk[0:N_JOINTS] - u_prev[0:N_JOINTS]) / dt
        xk1, dnu_k, Jk, Pk, Ak = step_fun(dt, xk, uk, ddq_k, dnu_g, f_eef, l_eef)

        X_list.append(xk1)
        DNU_cols.append(dnu_k); J_cols.append(Jk); P_cols.append(Pk); A_cols.append(Ak)
        xk = xk1; u_prev = uk; dnu_g = dnu_k

    X_all   = ca.hcat(X_list)
    DNU_all = ca.hcat(DNU_cols)
    J_all   = ca.hcat(J_cols)
    P_all   = ca.hcat(P_cols)
    A_all   = ca.hcat(A_cols)

    return ca.Function('rollout_unrolled',
        [x0, U, Uprev0, dt, dnu0, f_eef, l_eef],
        [X_all, DNU_all, J_all, P_all, A_all]
    ).expand()

def build_ocp_template(dt: float, solver: str, ipopt_opts: dict):
    opti = ca.Opti()

    # --- decision variables (once) ---
    X = opti.variable(STATE_DIM, N_HORIZON+1)
    U = opti.variable(CTRL_DIM, N_HORIZON)

    # --- parameters (change every step) ---
    x0p         = opti.parameter(STATE_DIM)   # initial state
    Uprev0      = opti.parameter(CTRL_DIM)    # previous input for Δu and ddq
    dnu0        = opti.parameter(N_DOF)       # predictor seed
    veh_pos_ref = opti.parameter(3)           # station-keeping reference

    # If you still need these:
    f_eef_p     = opti.parameter(3)
    l_eef_p     = opti.parameter(3)

    # --- initials and boundary condition ---
    opti.subject_to(X[:, 0] == x0p)

    # --- weights (tune as you like) ---
    Qpos = ca.DM.eye(3) * 100.0
    Qvel = ca.DM.eye(N_DOF) * 1.0
    R    = ca.DM.eye(CTRL_DIM) * 1e-4
    Rdu  = ca.DM.eye(CTRL_DIM) * 1e-3

    # --- rollout over horizon ---
    u_prev = Uprev0
    dnu_g  = dnu0
    cost   = 0

    # (Optional) if you’re only station-keeping, use a lean STEP that returns just x_next (and maybe dnu)
    for k in range(N_HORIZON):
        xk = X[:, k]
        uk = U[:, k]
        ddq_k = (uk[0:N_JOINTS] - u_prev[0:N_JOINTS]) / dt

        # Keep existing STEP call:
        xkp1, dnu_k, J_eef, p_eef, att_eef = STEP(dt, xk, uk, ddq_k, dnu_g, f_eef_p, l_eef_p)
        opti.subject_to(X[:, k+1] == xkp1)

        # costs: hold position, damp velocities, modest effort & smoothness
        veh_pos_k = X[N_JOINTS+N_DOF : N_JOINTS+N_DOF+3, k]
        nu_k      = X[N_JOINTS : N_JOINTS+N_DOF, k]
        pos_err_veh   = veh_pos_ref - veh_pos_k

        q_goal = utils_sym.euler_to_quat(ca.DM(0.8), ca.DM(0.0), ca.DM(0.8))
        att_vehicle = xk[N_JOINTS+N_DOF+3 : N_JOINTS+N_DOF+7]
        q_error_veh = utils_sym.quaternion_error(q_goal, att_vehicle)

        pos_err_eef = veh_pos_ref - p_eef

        q_goal = ca.DM([1.0, 0.0, 0.0, 0.0]) # desired quaternion (no rotation)
        q_current = att_eef
        q_error_eef = utils_sym.quaternion_error(q_goal, q_current)

        # cost += pos_err_veh.T @ Qpos @ pos_err_veh
        # cost += q_error_veh.T  @ Qpos @ q_error_veh
        cost += pos_err_eef.T @ Qpos @ pos_err_eef
        # cost += q_error_eef.T @ Qpos @ q_error_eef
        cost += nu_k.T   @ Qvel @ nu_k
        cost += uk.T     @ R    @ uk
        cost += (uk - u_prev).T @ Rdu @ (uk - u_prev)

        if USE_PWM:
            opti.subject_to(U[N_JOINTS:, k] <=  1.0)
            opti.subject_to(U[N_JOINTS:, k] >= -1.0)

        # Joint position limits constraints
        opti.subject_to(U[0:N_JOINTS, k] <= JOINT_VELOCITIES[:N_JOINTS])
        opti.subject_to(U[0:N_JOINTS, k] >= (-1 * JOINT_VELOCITIES[:N_JOINTS]))

        # Joint velocity limits constraints
        opti.subject_to(X[:N_JOINTS, k] <= JOINT_LIMITS[1, :N_JOINTS])
        opti.subject_to(X[:N_JOINTS, k] >= JOINT_LIMITS[0, :N_JOINTS])

        # Collision avoidance
        p_vehicle = xk[N_JOINTS+N_DOF : N_JOINTS+N_DOF+3]
        att_vehicle = xk[N_JOINTS+N_DOF+3 : N_JOINTS+N_DOF+7]
        # self-collision
        g_collision, vehicle_bound, underarm_bound, rho_collision = CONSTRAINT_COLLISION(p_vehicle, att_vehicle, p_eef, att_eef)
        opti.subject_to(g_collision >= 0)
        cost += rho_collision
        # collision with tank
        g_vehicle, g_underarm, cost_vehicle, cost_underarm = CONSTRAINT_TANK(vehicle_bound, underarm_bound)
        opti.subject_to(g_vehicle >= 0)
        opti.subject_to(g_underarm >= 0)
        cost += cost_vehicle + cost_underarm

        #TODO: manipulability cost that drives arm away from singularities

        u_prev = uk
        dnu_g  = dnu_k

    opti.minimize(cost)

    # --- solver options (expand + limited-memory helps!) ---
    # casadi_opts = {"expand": True} # removing that made it faster, but stll no output in terminal
    # casadi_opts = {}
    opti.solver(solver, ipopt_opts) #, solver_options = solver_options)

    # Return handles you’ll reuse
    handles = {
        "opti": opti, "X": X, "U": U,
        "x0p": x0p, "Uprev0": Uprev0, "dnu0": dnu0,
        "veh_pos_ref": veh_pos_ref,
        "f_eef_p": f_eef_p, "l_eef_p": l_eef_p
    }
    return handles

def solve_cftoc(
        handles: dict,
        *,
        x0_val: np.ndarray,
        u_prev0_val: np.ndarray,
        dnu0_val: Optional[np.ndarray],
        U_guess: np.ndarray,
        X_guess: np.ndarray,
        f_eef_val: np.ndarray,
        l_eef_val: np.ndarray,
        veh_pos_ref_val: np.ndarray,
        ref_eef_positions: np.ndarray,
        ref_eef_attitudes: np.ndarray,
        lam_g_prev: Optional[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, float]:
    assert ref_eef_positions.shape == (3, N_HORIZON)
    assert ref_eef_attitudes.shape == (4, N_HORIZON)

    opti = handles["opti"]
    X    = handles["X"]
    U    = handles["U"]

    # Robust defaults
    if dnu0_val is None:
        dnu0_val = np.zeros(N_DOF)

    # Set parameters
    opti.set_value(handles["x0p"], x0_val)
    opti.set_value(handles["Uprev0"], u_prev0_val)
    opti.set_value(handles["dnu0"], dnu0_val)
    opti.set_value(handles["veh_pos_ref"], veh_pos_ref_val)
    opti.set_value(handles["f_eef_p"], f_eef_val)
    opti.set_value(handles["l_eef_p"], l_eef_val)

    # Warm starts
    opti.set_initial(U, U_guess)
    opti.set_initial(X, X_guess)

    if lam_g_prev is not None:
        opti.set_initial(opti.lam_g, lam_g_prev)

    try:
        sol = opti.solve()
        Xv = sol.value(X); Uv = sol.value(U)
        Jv = float(sol.value(opti.f))
        lamg = sol.value(opti.lam_g)     # <-- duals
        return Xv, Uv, Jv, lamg
    except RuntimeError:
        # Optional: inspect infeasibility here
        print("RuntimeError in solver")
        return np.zeros((STATE_DIM, N_HORIZON+1)), np.zeros((CTRL_DIM, N_HORIZON)), 1e6, None
    
# def task_eef_tracking():
#     pass

# def task_station_keeping():
#     pass

def run_mpc(
    *,
    M: int,
    dt: float,
    x0: np.ndarray,
    ref_eef_positions: np.ndarray,
    ref_eef_attitudes: np.ndarray,
    opts: dict,
    solver: str = "ipopt",
    u_prev0: Optional[np.ndarray] = None,
    dnu0: Optional[np.ndarray] = None,
    f_eef_val: Optional[np.ndarray] = None,
    l_eef_val: Optional[np.ndarray] = None,
    veh_pos_ref_val: Optional[np.ndarray] = None,
):
    handles = build_ocp_template(dt=dt, solver=solver, ipopt_opts=opts)
    assert x0.shape[0] == STATE_DIM
    assert ref_eef_positions.shape == (3, M)
    assert ref_eef_attitudes.shape == (4, M)

    if u_prev0 is None: # variable necessary to compute ddq
        u_prev0 = np.zeros(CTRL_DIM)
    if dnu0 is None:
        dnu0 = np.zeros(N_DOF)
    if f_eef_val is None:
        f_eef_val = np.zeros(3)
    if l_eef_val is None:
        l_eef_val = np.zeros(3)

    X_real = np.zeros((STATE_DIM, M+1))
    U_appl = np.zeros((CTRL_DIM, M))
    J_hist = np.zeros(M)
    X_pred = np.zeros((STATE_DIM, N_HORIZON+1, M))
    U_pred = np.zeros((CTRL_DIM, N_HORIZON, M))

    # for initializing the optimization variables / the warm start
    U_guess = np.tile(u_prev0.reshape(-1, 1), (1, N_HORIZON))
    X_guess = np.tile(x0.reshape(-1, 1), (1, N_HORIZON+1))

    X_real[:, 0] = x0

    # TODO when using EKF2 later, use current estimate for accelerations in dnu0, instead of zero array

    # Ensure reference arrays always have N_HORIZON columns for each MPC step
    # Pad reference arrays so that slicing [:, step:step+N_HORIZON] is always valid
    if ref_eef_positions.shape[1] < M + N_HORIZON:
        pad_count = M + N_HORIZON - ref_eef_positions.shape[1]
        ref_eef_positions = np.hstack([ref_eef_positions, np.tile(ref_eef_positions[:, -1][:, None], (1, pad_count))])
        ref_eef_attitudes = np.hstack([ref_eef_attitudes, np.tile(ref_eef_attitudes[:, -1][:, None], (1, pad_count))])

    start_time = None

    lam_g_last = None

    for step in range(M):
        if step == 1:
            start_time = time.time()
        print(f"Step {step + 1} / {M}")
        X_opt, U_opt, J_opt, lam_g_last = solve_cftoc(
            handles=handles,
            x0_val=X_real[:, step],
            u_prev0_val=u_prev0 if step == 0 else U_appl[:, step-1],
            dnu0_val=dnu0,
            U_guess=U_guess,
            X_guess=X_guess,            
            f_eef_val=f_eef_val,
            l_eef_val=l_eef_val,
            veh_pos_ref_val=veh_pos_ref_val,
            ref_eef_positions=ref_eef_positions[:, step:step+N_HORIZON],
            ref_eef_attitudes=ref_eef_attitudes[:, step:step+N_HORIZON],
            lam_g_prev=lam_g_last,
        )

        X_pred[:, :, step] = X_opt
        U_pred[:, :, step] = U_opt
        J_hist[step] = J_opt

        u_apply = U_opt[:, 0]
        U_appl[:, step] = u_apply
        uq_apply = u_apply[0:N_JOINTS]
        uv_apply = u_apply[N_JOINTS:]

        # apply optimal control to system
        # UVMS_MODEL_INSTANCE.update(dt, uq_apply, uv_apply, USE_PWM, V_BAT) 
        # X_real[:, step+1] = UVMS_MODEL_INSTANCE.get_next_state(USE_QUATERNION)
        #####
        xk = X_real[:, step]
        uk = U_appl[:, step]
        u_prev0_val=u_prev0 if step == 0 else U_appl[:, step-1]
        ddq_k = (uk[0:N_JOINTS] - u_prev0_val[0:N_JOINTS]) / dt
        xkp1, _, _, _, _ = STEP(dt, xk, uk, ddq_k, dnu0, f_eef_val, l_eef_val)
        X_real[:, step+1] = xkp1.full().flatten()
        #################################


        U_guess = np.hstack([U_opt[:, 1:], U_opt[:, -1][:, None]])
        X_guess = np.hstack([X_opt[:, 1:], X_opt[:, -1][:, None]])
        X_guess[:, 0] = X_real[:, step+1]

    end_time = time.time()
    print(f"MPC computation time: {end_time - start_time:.2f} seconds")

    return X_real, U_appl, J_hist, X_pred, U_pred

def check_fixed_point(q, nu, eta, uq, uv, ddq_in, dnu_g, quat, f_eef, l_eef):
    dnu_fp = dnu_g
    dnu_history = []
    converged_iter = None
    threshold = 0.1
    dnu_fp_converged = None

    for k in range(FIXED_POINT_ITER):
        a_ref, dw_ref = dnu_fp[0:3], dnu_fp[3:6]
        tau_c = TAU_COUPLING(q, uq, ddq_in, nu[0:3], a_ref, nu[3:6], dw_ref, quat, f_eef, l_eef)
        dnu_fp_new = DYN_FOSSEN(eta, nu, uv, tau_c)
        dnu_history.append(np.array(dnu_fp_new).flatten())
        if converged_iter is None and np.allclose(np.array(dnu_fp_new), np.array(dnu_fp)):
            converged_iter = k + 1
            dnu_fp_converged = np.array(dnu_fp_new).flatten()
        dnu_fp = dnu_fp_new

    if converged_iter is not None:
        print(f"Converged after {converged_iter} iterations.")
        mean_errors = [np.linalg.norm(dnu - dnu_fp_converged) for dnu in dnu_history]
        for i, err in enumerate(mean_errors):
            if err < threshold:
                print(f"First mean error below threshold ({threshold}) at iteration {i+1}: {err}")
                break
    else:
        print("Did not converge.")
        mean_errors = None

    return dnu_fp, dnu_history, mean_errors

def init_uvms_model(
    bluerov_params_path: str = 'model_params.yaml',
    dh_params_path: str = 'alpha_kin_params.yaml',
    joint_limits_path: Optional[str] = None,
    manipulator_dyn_params_paths: Optional[list[str]] = None,
    integrator: str = 'euler',
    use_quaternion: bool = True,
    use_pwm: bool = True,
    use_fixed_point: bool = True,
    v_bat: float = 16.0,
    fixed_point_iter: int = 2,
    n_horizon: int = 10
) -> None:
    global INITIALIZED, BLUEROV_DYN, MANIP_DYN, MANIP_KIN, TAU_COUPLING, DYN_FOSSEN, J_EEF, EEF_POSE, F_SYS, STEP, ROLLOUT_UNROLLED, LUMELSKY, LUMELSKY_CASES, COLLISION, CONSTRAINT_COLLISION, CONSTRAINT_TANK
    global USE_QUATERNION, USE_PWM, USE_FIXED_POINT, FIXED_POINT_ITER, N_HORIZON, V_BAT, L, MIXER, M_INV, JOINT_LIMITS, JOINT_EFFORTS, JOINT_VELOCITIES, INTEGRATOR, N_DOF, N_JOINTS, STATE_DIM, CTRL_DIM, C_FUN, D_FUN, G_FUN, J_FUN
    global Q0, POS0, ATT0_EULER, VEL0, OMEGA0, UVMS_MODEL_INSTANCE, MANIP_KIN_REAL
    global MANIP_PARAMS, ALPHA_PARAMS, BRV_PARAMS

    with _LOCK:
        if INITIALIZED:
            return

        INTEGRATOR = integrator.lower()
        USE_QUATERNION = use_quaternion
        USE_PWM = use_pwm
        USE_FIXED_POINT = use_fixed_point
        V_BAT = v_bat
        FIXED_POINT_ITER = fixed_point_iter
        N_HORIZON = n_horizon

        BRV_PARAMS = utils_math.load_model_params(bluerov_params_path)
        BLUEROV_DYN = sym_brv.BlueROVDynamicsSymbolic(BRV_PARAMS)

        MANIP_PARAMS = utils_math.load_dh_params(dh_params_path)
        if joint_limits_path:
            JOINT_LIMITS, JOINT_EFFORTS, JOINT_VELOCITIES, _ = utils_math.load_joint_limits(joint_limits_path)
            JOINT_LIMITS = np.array(JOINT_LIMITS).T
            JOINT_VELOCITIES = np.array(JOINT_VELOCITIES).T
        ALPHA_PARAMS = utils_math.load_dynamic_params(manipulator_dyn_params_paths)
        MANIP_KIN = sym_manip_kin.KinematicsSymbolic(MANIP_PARAMS)
        MANIP_DYN = sym_manip_dyn.DynamicsSymbolic(MANIP_KIN, ALPHA_PARAMS)
        MANIP_KIN_REAL = manip_kin.Kinematics(MANIP_PARAMS)

        C_FUN = BLUEROV_DYN.C
        D_FUN = BLUEROV_DYN.D
        G_FUN = BLUEROV_DYN.g_quat if USE_QUATERNION else BLUEROV_DYN.g
        J_FUN = BLUEROV_DYN.J_quat if USE_QUATERNION else BLUEROV_DYN.J

        L = BLUEROV_DYN.L
        MIXER = BLUEROV_DYN.mixer
        M_INV = BLUEROV_DYN.M_inv  

        N_JOINTS = MANIP_DYN.kinematics_.n_joints
        N_DOF = BLUEROV_DYN.M_inv.size1() 

        STATE_DIM = N_DOF + (7 if USE_QUATERNION else 6) + N_JOINTS
        CTRL_DIM = N_JOINTS + (8 if USE_PWM else 6)

        TAU_COUPLING = _build_tau_coupling_func()
        DYN_FOSSEN = _build_dynamics_fossen_func()
        EEF_POSE, J_EEF = _build_eef_blocks()
        F_SYS = _build_f_sys(TAU_COUPLING, DYN_FOSSEN, EEF_POSE, J_EEF)
        STEP = _build_step_func(F_SYS)
        ROLLOUT_UNROLLED = _build_rollout_unrolled(STEP, N_HORIZON)
        LUMELSKY = Lumelsky._Lumelsky()
        LUMELSKY_CASES = Lumelsky._Lumelsky_cases()
        COLLISION = _check_collision(LUMELSKY_CASES)
        CONSTRAINT_COLLISION = _constraint_collision(COLLISION)
        CONSTRAINT_TANK = _constraint_tank()

        INITIALIZED = True

        UVMS_MODEL_INSTANCE = uvms_model.UVMSModel(MANIP_PARAMS, ALPHA_PARAMS, BRV_PARAMS, Q0, POS0, ATT0_EULER, VEL0, OMEGA0)

def is_initialized() -> bool:
    return INITIALIZED

def teardown_for_tests() -> None:
    global INITIALIZED, BLUEROV_DYN, MANIP_DYN, TAU_COUPLING, DYN_FOSSEN, G_FUN, J_FUN
    with _LOCK:
        INITIALIZED = False
        BLUEROV_DYN = None
        MANIP_DYN = None
        TAU_COUPLING = None
        DYN_FOSSEN = None
        G_FUN = None
        J_FUN = None

def main():
    bluerov_package_path = get_package_path('bluerov')
    bluerov_params_path = bluerov_package_path + "/config/model_params.yaml"

    manipulator_package_path = get_package_path('manipulator')
    dh_params_path = manipulator_package_path + "/config/alpha_kin_params.yaml"
    joint_limits_path = manipulator_package_path + "/config/alpha_joint_lim_real.yaml"
    base_tf_bluerov_path = manipulator_package_path + "/config/alpha_base_tf_params_bluerov.yaml"
    inertial_params_dh_path = manipulator_package_path + "/config/alpha_inertial_params_dh.yaml"

    manipulator_dyn_params_paths = [
        dh_params_path,
        base_tf_bluerov_path,
        inertial_params_dh_path
    ]
   
    init_uvms_model(
        bluerov_params_path=bluerov_params_path,
        dh_params_path=dh_params_path,
        joint_limits_path=joint_limits_path,
        manipulator_dyn_params_paths=manipulator_dyn_params_paths,
        use_quaternion=True,
        use_pwm=True,
        use_fixed_point=False,
        v_bat=16.0,
        fixed_point_iter=2,
        n_horizon=10
    )

    T_duration = 10.0 # [s]
    dt = 0.05 # sampling [s]
    M = int(T_duration / dt)  # number of MPC steps / control horizon
    x0 = np.concatenate((Q0, VEL0, OMEGA0, POS0, ATT0_QUAT)) if USE_QUATERNION else np.concatenate((Q0, VEL0, OMEGA0, POS0, ATT0_EULER))
    # veh_pos_ref_val = np.array([0.8, 0.0, -0.1])  # desired vehicle position for station-keeping
    veh_pos_ref_val = np.array([3.0, 2.0, -0.7])  # desired vehicle position for station-keeping
    ref_eef_pos = np.zeros((3, M))
    ref_eef_att = np.zeros((4, M))
    # opts = {'ipopt.print_level': 0, 'print_time': 0}
    # opts = {'ipopt.print_level': 5, 'print_time': 1}
    # opts = {
    #     "print_time": 1,
    #     "ipopt.print_level": 5,
    #     "ipopt.sb": "yes",
    #     "ipopt.linear_solver": "mumps",           # or ma57 if available
    #     "ipopt.hessian_approximation": "limited-memory"  # great for large graphs
    # }
    opts = {
    "expand": True,                 # lets CasADi simplify the graph
    "print_time": False,            # no timing line from CasADi
    "ipopt.print_level": 0,         # 0 = silent
    "ipopt.sb": "yes",              # “small banner”: removes the big header
    "ipopt.file_print_level": 0,    # don’t write a log file
    "ipopt.mu_strategy": "adaptive",
    "ipopt.linear_solver": "mumps",     # keep; switch to 'pardiso' if you have it
    "ipopt.max_iter": 100,              # avoid long tail
    "ipopt.acceptable_tol": 1e-3,       # stop earlier if “good enough”
    "ipopt.acceptable_obj_change_tol": 1e-4,

    "ipopt.tol": 1e-6,               # default 1e-8; relax a bit
    "ipopt.constr_viol_tol": 1e-6,   # same

    "ipopt.nlp_scaling_method": "gradient-based", # about robustness

    "ipopt.fast_step_computation": "yes", # fewer factorizations per iter; slight robustness trade-off

    "ipopt.warm_start_init_point": "yes", # warm starting sped up by factor 2 from 6s to 2s for T = 10s and dt = 0,2s
    "ipopt.warm_start_bound_push": 1e-8,
    "ipopt.warm_start_mult_bound_push": 1e-6,

    "ipopt.warm_start_slack_bound_push": 1e-8,  # if you have slacks
    "ipopt.mu_strategy": "adaptive",
    # Optional: can help when reusing duals
    "ipopt.mu_init": 1e-3,

    "ipopt.tol": 1e-6,              # default 1e-8; relax a bit
    "ipopt.constr_viol_tol": 1e-6,  # default 1e-8
    # "ipopt.hessian_approximation": "limited-memory",  # L-BFGS
    # "ipopt.limited_memory_max_history": 20, 
}
    # opts = {
    #     "jit": True,
    #     'compiler': 'shell',
    #     'jit_options': {'compiler': 'gcc', 'flags': ['-O3']}
    # }
    # opts = {
    #     "print_time": False,
    #     "expand": True,
    #     # "structure_detection": 'none','auto',
    #     # "jit": True,
    #     # "jit_options": {'compiler': 'gcc', 'flags': ['-O3']},
    #     "detect_simple_bounds": True,
    #     "fatrop": {            # FATROP-native options
    #         "print_level": 0,
    #         "max_iter": 300,
    #         "tol": 1e-8,
    #         "acceptable_tol": 1e-4,
    #         "acceptable_iter": 7,
    #         "constr_viol_tol": 1e-6,
    #         "warm_start_init_point": True,
    #         "linsol_iterative_refinement": True,
    #         "ls_scaling": True,
    #         "bound_push": 1e-4,
    #         "bound_frac": 1e-4,
    #     }, # https://github.com/ytwboxing/fatrop-fork/blob/811d36ce272f49bca5d857809b6258422e369b37/fatrop/solver/FatropOptions.hpp
    # }
    solver = "ipopt"
    # solver = "fatrop"
    u_prev0 = np.full((CTRL_DIM), 0.1)  # initial control input guess
    dnu0 = None # replace later with EKF2 prediction of accelerations
    f_eef_val = np.zeros(3)
    l_eef_val = np.zeros(3)

    # Check if Q0 satisfies joint limits
    if JOINT_LIMITS is not None:
        if not np.all((Q0 >= JOINT_LIMITS[0, :N_JOINTS]) & (Q0 <= JOINT_LIMITS[1, :N_JOINTS])):
            raise ValueError(f"Initial joint angles Q0={Q0} do not satisfy joint limits {JOINT_LIMITS[:, :N_JOINTS]}")

    X_real, U_appl, J_hist, X_pred, U_pred = run_mpc(M=M,
            dt=dt,
            x0=x0,
            veh_pos_ref_val=veh_pos_ref_val,
            ref_eef_positions=ref_eef_pos,
            ref_eef_attitudes=ref_eef_att,
            opts=opts,
            solver=solver,
            u_prev0=u_prev0,
            dnu0=dnu0,
            f_eef_val=f_eef_val,
            l_eef_val=l_eef_val
            )
    
    R_B_0 = MANIP_DYN.R_reference
    r_B_0 = MANIP_DYN.tf_vec

    # animate.animate_uvms_from_state(X_real, MANIP_PARAMS, R_B_0, r_B_0, dt)
    animate.animate_uvms_with_bounding_spheres(X_real, MANIP_PARAMS, R_B_0, r_B_0, dt, CONSTRAINT_COLLISION)

    # Extract vehicle position history from X_real
    eta_start = N_JOINTS + N_DOF
    vehicle_pos_indices = slice(eta_start, eta_start + 3)
    vehicle_pos_history = X_real[vehicle_pos_indices, :]  # shape: (3, M+1)

    time = np.arange(vehicle_pos_history.shape[1]) * dt

    plt.figure(figsize=(10, 6))
    plt.plot(time, vehicle_pos_history[0, :], label='x')
    plt.plot(time, vehicle_pos_history[1, :], label='y')
    plt.plot(time, vehicle_pos_history[2, :], label='z')
    plt.xlabel('Time [s]')
    plt.ylabel('Vehicle Position [m]')
    plt.title('Vehicle x, y, z Position Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot joint efforts over time
    plt.figure(figsize=(10, 6))
    for i in range(N_JOINTS):
        plt.plot(time[:-1], U_appl[i, :], label=f'Joint {i+1}')
    plt.xlabel('Time [s]')
    plt.ylabel('Joint Effort')
    plt.title('Joint Effort Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot vehicle thruster commands over time
    plt.figure(figsize=(10, 6))
    thruster_start = N_JOINTS
    thruster_end = U_appl.shape[0]
    for i in range(thruster_start, thruster_end):
        plt.plot(time[:-1], U_appl[i, :], label=f'Thruster {i - thruster_start + 1}')
    plt.xlabel('Time [s]')
    plt.ylabel('Thruster Command')
    plt.title('Vehicle Thruster Commands Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Use the first predicted trajectory (X_pred[..., 0]) instead of X_real
    vehicle_pos_pred = X_pred[vehicle_pos_indices, :, 0]  # shape: (3, N_HORIZON+1)
    time_pred = np.arange(vehicle_pos_pred.shape[1]) * dt

    # print("Predicted vehicle positions (first MPC step):")
    # print(vehicle_pos_pred)

    plt.figure(figsize=(10, 6))
    plt.plot(time_pred, vehicle_pos_pred[0, :], label='x (pred)')
    plt.plot(time_pred, vehicle_pos_pred[1, :], label='y (pred)')
    plt.plot(time_pred, vehicle_pos_pred[2, :], label='z (pred)')
    plt.xlabel('Time [s]')
    plt.ylabel('Vehicle Position [m]')
    plt.title('Predicted Vehicle x, y, z Position (First MPC Step)')
    plt.legend()
    plt.grid(True)
    plt.show()

    vehicle_pos_pred = X_pred[vehicle_pos_indices, :, 5]  # shape: (3, N_HORIZON+1)
    time_pred = np.arange(vehicle_pos_pred.shape[1]) * dt

    # print(vehicle_pos_pred)
    # print("Next")
    # print(X_pred)

    # print("Predicted vehicle positions (first MPC step):")
    # print(vehicle_pos_pred)

    plt.figure(figsize=(10, 6))
    plt.plot(time_pred, vehicle_pos_pred[0, :], label='x (pred)')
    plt.plot(time_pred, vehicle_pos_pred[1, :], label='y (pred)')
    plt.plot(time_pred, vehicle_pos_pred[2, :], label='z (pred)')
    plt.xlabel('Time [s]')
    plt.ylabel('Vehicle Position [m]')
    plt.title('Predicted Vehicle x, y, z Position (First MPC Step)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # # Extract eta_history and joint_history from X_real
    # # X_real shape: (STATE_DIM, M+1)
    # # eta is at indices: N_JOINTS + N_DOF : N_JOINTS + N_DOF + (7 if USE_QUATERNION else 6)
    # eta_start = N_JOINTS + N_DOF
    # eta_end = eta_start + (7 if USE_QUATERNION else 6)
    # eta_history = X_real[eta_start:eta_end, :]

    # # joint_history is first N_JOINTS
    # joint_history = X_real[0:N_JOINTS, :]

    # # Animate the UVMS
    # animate.animate_uvms(eta_history, joint_history, dt)
    
    return

if __name__ == "__main__":
    main()
