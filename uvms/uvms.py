import bluerov.bluerov as bluerov
import uvms.model as uvms_model
import manipulator
import numpy as np
import casadi as ca
import bluerov.dynamics_symbolic as sym_brv
import manipulator.dynamics_symbolic as sym_manip_dyn
import manipulator.kinematics_symbolic as sym_manip_kin
import threading
from typing import Optional

from common.my_package_path import get_package_path
import common.utils_math as utils_math
import common.utils_sym as utils_sym
import common.animate as animate

# TODO: 
# lese und extrahiere Matrix-Vektor dynamics for manipulators from Schjolbergs dissertation
# define manipulator dynamics both in recursive Newton-Euler fashion (Trekel, (Ioi)) and in closed form Newton-Euler having Matrices (Scholberg,Fossen)
# dafine manipulator dynamics using Articulated Body Algorithm (McMillan) -> nein, scheint kein Vorteil zu Newton-Euler zu haben, ist auch iteratives Verfahren

# -----------------------
# Globale, read-only Variablen nach init()
# -----------------------
INITIALIZED: bool = False
_LOCK = threading.RLock()

# “Header”-Variablen (Kontexte/Objekte)
BLUEROV_DYN = None           # bluerov.BlueROVDynamics
MANIP_DYN  = None            # z. B. manip.ManipulatorDynamics
MANIP_KIN  = None            # manipulator.kinematics.Kinematics

# Symbolische/Linearisierte Funktionen (CasADi)
TAU_COUPLING: Optional[ca.Function] = None
DYN_FOSSEN: Optional[ca.Function] = None
J_DYN_FOSSEN: Optional[ca.Function] = None
INTEGRATOR_FUNC: Optional[ca.Function] = None
J_EEF: Optional[ca.Function] = None  # Jacobian of end-effector in inertial frame
EEF_POSE: Optional[ca.Function] = None  # Function to compute end-effector pose from eta

# Konstanten/Konfig für schnellen Zugriff
USE_QUATERNION: bool = False
USE_PWM: bool = True
V_BAT: float = 16.0
INTEGRATOR: str = None
N_DOF: int = None  # Number of degrees of freedom (DOF) in the system
N_JOINTS: int = None  # Number of joints in the manipulator
JOINT_LIMITS: Optional[np.ndarray] = None  # Joint limits (min, max) for each joint
JOINT_EFFORTS: Optional[np.ndarray] = None  # Joint efforts (torques) for each joint
JOINT_VELOCITIES: Optional[np.ndarray] = None  # Joint velocities for each joint

# Optional: Konstanten aus deinem Symbolik-Modul (dürfen global sein, sind immutable)
L = None
MIXER = None
M_INV = None
C_FUN = None
D_FUN = None
G_FUN = None  # wird in init gesetzt je nach USE_QUATERNION
J_FUN = None  # dito

Q0 = np.array([np.pi, np.pi * 0.5, np.pi * 0.75, np.pi * 0.5])
POS0 = np.array([0.0, 0.0, 0.0])
ATT0_EULER = np.array([0.0, 0.0, np.pi/4])  # Convert Euler angles to quaternion
ATT0_QUAT = utils_math.euler_to_quat(ATT0_EULER[0], ATT0_EULER[1], ATT0_EULER[2])
VEL0 = np.array([0.0, 0.0, 0.0])
OMEGA0 = np.array([0.0, 0.0, 0.0])
UVMS_MODEL_INSTANCE: Optional[uvms_model.UVMSModel] = None

def _build_tau_coupling_func() -> ca.Function:
    """Koppelmomente/-kräfte vom Manipulator auf das Fahrzeug (symbolisch)."""
    q     = ca.MX.sym('q', N_JOINTS)
    dq    = ca.MX.sym('dq', N_JOINTS)
    ddq   = ca.MX.sym('ddq', N_JOINTS)
    v     = ca.MX.sym('v', 3)
    a     = ca.MX.sym('a', 3)
    w     = ca.MX.sym('w', 3)
    dw    = ca.MX.sym('dw', 3)
    quat  = ca.MX.sym('quat', 4)
    f_eef = ca.MX.sym('f_eef', 3)
    l_eef = ca.MX.sym('l_eef', 3)

    tau_c = MANIP_DYN.rnem_symbolic(q, dq, ddq, v, a, w, dw, quat, f_eef, l_eef)
    return ca.Function('tau_coupling', [q,dq,ddq,v,a,w,dw,quat,f_eef,l_eef], [tau_c])

def _build_dynamics_fossen_func() -> ca.Function:
    """Fossen-Fahrzeugdynamik inkl. Kopplung als CasADi-Funktion dν = f(...)."""
    nu   = ca.MX.sym('nu', N_DOF)
    eta  = ca.MX.sym('eta', 7 if USE_QUATERNION else 6)
    dnuC = ca.MX.sym('dnuC', N_DOF)  # Kopplungsbeschleunigungen (falls extern geschätzt/genutzt)
    q    = ca.MX.sym('q', N_JOINTS)
    ddq  = ca.MX.sym('ddq', N_JOINTS)
    uq   = ca.MX.sym('uq', N_JOINTS)                 # Gelenk-Inputs (z. B. Sollgeschw.)
    uv   = ca.MX.sym('uv', 8 if USE_PWM else 6)      # Thruster-PWM oder direkte Wrenches
    f_eef= ca.MX.sym('f_eef', 3)                       # externe Kontaktkraft am EE
    l_eef= ca.MX.sym('l_eef', 3)                       # Hebelarm EE
    tau_v = (L * V_BAT * (MIXER @ uv)) if USE_PWM else uv

    # !!!!!!!!!!! TAU_COUPLING funktioniert nur mit quaternion
    tau_c = TAU_COUPLING(q, uq, ddq, nu[0:3], dnuC[0:3], nu[3:6], dnuC[3:6],
                        eta[3:7] if USE_QUATERNION else eta[3:6], f_eef, l_eef)

    dnu  = M_INV @ (tau_v + tau_c - C_FUN(nu) @ nu - D_FUN(nu) @ nu - G_FUN(eta))
    return ca.Function('dyn_fossen', [eta,nu,dnuC,q,ddq,uv,uq,f_eef,l_eef], [dnu])

def _build_Jac_dynamics_fossen() -> ca.Function:
    dnuC = ca.MX.sym('dnuC', N_DOF)
    eta  = ca.MX.sym('eta', 7 if USE_QUATERNION else 6)
    nu   = ca.MX.sym('nu', N_DOF)
    q    = ca.MX.sym('q', N_JOINTS)
    ddq  = ca.MX.sym('ddq', N_JOINTS)
    uv   = ca.MX.sym('uv', 8 if USE_PWM else 6)
    uq   = ca.MX.sym('uq', N_JOINTS)
    f_eef= ca.MX.sym('f_eef', 3)
    l_eef= ca.MX.sym('l_eef', 3)

    T_val = DYN_FOSSEN(eta, nu, dnuC, q, ddq, uv, uq, f_eef, l_eef)
    J = ca.jacobian(T_val, dnuC)              # 6x6 (or n_dof x 6)
    return ca.Function('J_T', [eta,nu,dnuC,q,ddq,uv,uq,f_eef,l_eef], [J])

def _build_eef_jacobian() -> ca.Function:
    if MANIP_KIN is None:
            raise RuntimeError("MANIP_KIN must be initialized before building CasADi functions.")
    eta  = ca.MX.sym('eta', 7 if USE_QUATERNION else 6)

    J_eef = ca.MX.zeros((N_DOF, N_DOF + N_JOINTS))

    if USE_QUATERNION:
        R_I_B = utils_sym.rotation_matrix_from_quat(eta[3:])
    else:
        R_I_B = utils_sym.rotation_matrix_from_euler(eta[3], eta[4], eta[5])

    r_B_0 = MANIP_DYN.tf_vec
    R_B_0 = MANIP_DYN.R_reference
    r_0_eef = MANIP_KIN.get_eef_position()
    att_0_eef = MANIP_KIN.get_eef_attitude()

    J_manipulator_pos, J_manipulator_rot = MANIP_KIN.get_full_jacobian()

    J_eef[0:3, 0:3] = R_I_B  # R_B^I

    vec_tmp = R_I_B @ r_B_0 + R_I_B @ R_B_0 @ r_0_eef
    skew_tmp = utils_sym.skew(vec_tmp)
    J_eef[0:3, 3:6] = -skew_tmp @ R_I_B  # -S(r_B,ee^I) * R_B^I
    J_eef[0:3, 6:6+N_JOINTS] = R_I_B @ R_B_0 @ J_manipulator_pos  # J_eta_ee,t^I

    J_eef[3:6, 3:6] = R_I_B  # R_B^I
    J_eef[3:6, 6:6+N_JOINTS] = R_I_B @ R_B_0 @ J_manipulator_rot  # J_eta_ee,r^I

    return ca.Function('J_eef', [eta], [J_eef])

def _build_eef_pose_function() -> ca.Function:
    eta  = ca.MX.sym('eta', 7 if USE_QUATERNION else 6)

    if USE_QUATERNION:
        R_I_B = utils_sym.rotation_matrix_from_quat(eta[3:])
    else:
        R_I_B = utils_sym.rotation_matrix_from_euler(eta[3], eta[4], eta[5])

    r_B_0 = MANIP_DYN.tf_vec
    R_B_0 = MANIP_DYN.R_reference
    r_0_eef = MANIP_KIN.get_eef_position()
    att_0_eef = MANIP_KIN.get_eef_attitude()

    p_eef = eta[0:3] + R_I_B @ r_B_0 + R_I_B @ R_B_0 @ r_0_eef
    att_eef = utils_sym.rotation_matrix_to_quaternion(R_I_B @ R_B_0 @ utils_sym.rotation_matrix_from_quat(att_0_eef))
    return ca.Function('eef_pose', [eta], [p_eef, att_eef])


def _build_integrator_state_update() -> ca.Function:
    dnuC = ca.MX.sym('dnuC', N_DOF)
    eta  = ca.MX.sym('eta', 7 if USE_QUATERNION else 6)
    nu   = ca.MX.sym('nu', N_DOF)
    q    = ca.MX.sym('q', N_JOINTS)
    ddq  = ca.MX.sym('ddq', N_JOINTS)
    uv   = ca.MX.sym('uv', 8 if USE_PWM else 6)
    uq   = ca.MX.sym('uq', N_JOINTS)
    f_eef= ca.MX.sym('f_eef', 3)
    l_eef= ca.MX.sym('l_eef', 3)

    dt   = ca.MX.sym('dt', 1)  # Zeitinkrement

    def f(nu_f, eta_f, q_f):
        dq = uq
        dnu = DYN_FOSSEN(eta_f, nu_f, dnuC, q_f, ddq, uv, uq, f_eef, l_eef)
        deta = J_FUN(eta_f) @ nu_f
        return dq, dnu, deta
    
    def normalize_quat(eta):
        pos_next = eta[0:3]
        quat_next = eta[3:]
        quat_next = quat_next / ca.norm_2(quat_next)
        eta_next = ca.vertcat(pos_next, quat_next)
        return eta_next

    if INTEGRATOR == 'euler':
        dq, dnu, deta = f(nu, eta, q)
        q_next = q + dt * dq
        nu_next = nu + dt * dnu
        eta_next = eta + dt * deta
        if USE_QUATERNION:
            eta_next = normalize_quat(eta_next)
        x_next = ca.vertcat(q_next, nu_next, eta_next)
        return ca.Function('integrator_euler', [dt, nu, eta, dnuC, q, ddq, uv, uq, f_eef, l_eef], [x_next])
                   
    elif INTEGRATOR == 'rk4':
        # k1
        dq1, dnu1, deta1 = f(nu, eta, q)
        q1 = q + 0.5 * dt * dq1
        nu1 = nu + 0.5 * dt * dnu1
        eta1 = eta + 0.5 * dt * deta1

        # k2
        dq2, dnu2, deta2 = f(nu1, eta1, q1)
        q2 = q + 0.5 * dt * dq2
        nu2 = nu + 0.5 * dt * dnu2
        eta2 = eta + 0.5 * dt * deta2

        # k3
        dq3, dnu3, deta3 = f(nu2, eta2, q2)
        q3 = q + dt * dq3
        nu3 = nu + dt * dnu3
        eta3 = eta + dt * deta3

        # k4
        dq4, dnu4, deta4 = f(nu3, eta3, q3)

        q_next = q + (dt / 6.0) * (dq1 + 2*dq2 + 2*dq3 + dq4)
        nu_next = nu + (dt / 6.0) * (dnu1 + 2*dnu2 + 2*dnu3 + dnu4)
        eta_next = eta + (dt / 6.0) * (deta1 + 2*deta2 + 2*deta3 + deta4)
        if USE_QUATERNION:
            eta_next = normalize_quat(eta_next)
        x_next = ca.vertcat(q_next, nu_next, eta_next)
        return ca.Function('integrator_rk4', [dt, nu, eta, dnuC, q, ddq, uv, uq, f_eef, l_eef], [x_next])

def solve_cftoc(N, dt, opts, solver, ref_eef_positions, ref_eef_attitudes):
    opti = ca.Opti()
    if opts is None:
        raise ValueError("Solver options 'opts' must be provided.")
    if solver is None: # 'ipopt'
        raise ValueError("Solver type 'solver' must be provided.")
    opti.solver(solver, opts)

    dnu_fix = 0 # oder aus dem x0 initial guess die differenz aus den ersten beiden Einträgen

    state_dim = N_DOF + (7 if USE_QUATERNION else 6) + N_JOINTS
    control_dim = N_JOINTS + (8 if USE_PWM else 6)  # 8 thrusters or 6 vehicle wrenches

    # x = [q, nu, eta]
    x = opti.variable(state_dim, N+1)  # state vector
    # u = [u_q, tau_v] oder u = [u_q, u_esc]
    # initialize as if for entry at 0 is k=0 and the previous entry for k=-1 is at the very last entry N+1
    u = opti.variable(control_dim, N+1)  # control input

    f_eef = ca.DM.zeros(3)
    l_eef = ca.DM.zeros(3)

    cost = 0.0
    Q_pos = ca.DM.eye(3) * 1.0  # Position tracking cost
    Q_att = ca.DM.eye(3) * 1.0   # Attitude tracking cost
    R_input = ca.DM.eye(control_dim) * 0.1  # Control input cost

    for k in range(N):
        q_k = x[0:N_JOINTS, k]  # joint angles
        nu_k = x[N_JOINTS:N_JOINTS+N_DOF, k] # body velocities
        eta_k = x[N_JOINTS+N_DOF:, k] # position and attitude
        uq_k = u[0:N_JOINTS, k]
        uv_k = u[N_JOINTS:, k]  # vehicle wrenches or ESC commands

        MANIP_DYN.kinematics_.update(q_k)  # Update kinematics with current joint angles

        # Joint acceleration
        ddq_k = (u[0:N_JOINTS, k] - u[0:N_JOINTS, k-1]) / dt

        # Fixed-Point Iteration: implicit solving dnu_k for tau_coupling evaluation in dynamic equations
        for _ in range(2):
            dnu_fix = DYN_FOSSEN(eta_k,nu_k,dnu_fix,q_k,ddq_k,uv_k,uq_k,f_eef,l_eef)

        # Update next state x
        opti.subject_to(x[:, k+1] == INTEGRATOR_FUNC(dt, nu_k, eta_k, dnu_fix, q_k, ddq_k, uv_k, uq_k, f_eef, l_eef))

        # Control input constraints
        if USE_PWM:
            opti.subject_to(u[N_JOINTS:, k] >= -1.0)
            opti.subject_to(u[N_JOINTS:, k] <= 1.0)
        else:
            opti.subject_to(u[N_JOINTS:, k] >= -1.0)
            opti.subject_to(u[N_JOINTS:, k] <= 1.0)

        eef_pos_k, eef_att_k = EEF_POSE(eta_k)  # global end-effector pose
        J_eef = J_EEF(eta_k)  # relates generalized velocities (nu_k and dq_k = uq_k) to end-effector velocities
        J_eef_pinv = ca.pinv(J_eef)
        # e.g.: zeta_desired = J_eef_pinv @ eef_velocity_desired

        pos_error = ref_eef_positions[:,k] - eef_pos_k
        att_error = utils_sym.quaternion_error(ref_eef_attitudes[:,k], eef_att_k)  # quaternion error

        cost += ca.mtimes(pos_error.T, Q_pos @ pos_error)
        cost += ca.mtimes(att_error.T, Q_att @ att_error)  # Position tracking cost

        # TODO: Add constraints and eef trajectory tracking using the manipulator Jacobian (make also a ca.Function for that)

    opti.minimize(cost) 

    try:
        sol = opti.solve()
        return sol.value(x), sol.value(u), sol.value(cost), sol.value(opti.lam_g)
    except RuntimeError:
        print("Infeasible. Debug info:")
        print("x0:", opti.debug.value(x[:, 0]))
        print("u0:", opti.debug.value(u[:, 0]))
        return np.zeros((x.shape[0], N+1)), np.zeros((u.shape[0], N)), 1e6
    
def test_symbolic_uvms_model():
    global Q0, POS0, ATT0_QUAT, VEL0, OMEGA0, N_DOF, N_JOINTS, USE_QUATERNION, USE_PWM, V_BAT, INTEGRATOR

    uv = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # Example control input
    uq = np.array([0.0, 0.0, 0.0, 0.0])  # Example joint velocities
    dt = 0.05
    t_range = 2

    f_eef = np.array([0.0, 0.0, 0.0])  # External force at end-effector
    l_eef = np.array([0.0, 0.0, 0.0])  # External torque at end-effector

    dnu_last = np.zeros(6)  # Initial guess for dnu
    last_uq = np.zeros(4)  # Initial guess for joint velocities

    x_next = np.zeros(N_JOINTS + N_DOF + (7 if USE_QUATERNION else 6))  # State vector for next step
    x = np.zeros(N_JOINTS + N_DOF + (7 if USE_QUATERNION else 6))  # Initial state vector
    x[:N_JOINTS] = Q0  # Set initial joint angles
    x[N_JOINTS:N_JOINTS+N_DOF] = np.concatenate((VEL0, OMEGA0))  # Set initial velocities
    x[N_JOINTS+N_DOF:] = np.concatenate((POS0, ATT0_QUAT))  # Set initial position and attitude

    joint_positions = np.zeros((N_JOINTS + 2, 3))  # Joint positions including base

    R_B_0 = np.array(MANIP_DYN.R_reference)
    r_B_0 = np.array(MANIP_DYN.tf_vec)

    MANIP_DYN.kinematics_.update(x[:N_JOINTS])  # Update kinematics with current joint angles

    eta_history = []

    joint_history = []  # Store joint positions over time

    for _ in range(t_range):
        ddq = (uq - last_uq) /dt
        last_uq = uq

        x_next = np.array(INTEGRATOR_FUNC(dt, x[N_JOINTS:N_JOINTS+N_DOF], x[N_JOINTS+N_DOF:], dnu_last, x[:N_JOINTS], ddq, uv, uq, f_eef, l_eef))
        eta_next = x_next[N_JOINTS+N_DOF:]  # Extract eta from the state vector
        dnu_last = np.array(DYN_FOSSEN(x[N_JOINTS+N_DOF:], x[N_JOINTS:N_JOINTS+N_DOF], dnu_last, x[:N_JOINTS], ddq, uv, uq, f_eef, l_eef))
        x = x_next

        MANIP_DYN.kinematics_.update(x[:N_JOINTS])  # Update kinematics with current joint angles
         
        R_I_B = utils_math.rotation_matrix_from_quat(eta_next[3:])

        joint_positions[0] = np.array(eta_next[0:3] + R_I_B @ r_B_0).flatten() # joint 0
    
        for i in range(N_JOINTS + 1):
            pos_expr = MANIP_KIN.get_link_position(i)                 # MX
            pos_val  = ca.Function('f', [], [pos_expr])()
            pos_val_np = np.array(pos_val['o0']).flatten()
            joint_positions[i+1] = joint_positions[0] + np.array(R_I_B @ R_B_0 @ pos_val_np).flatten()

        etaaa = np.concatenate((eta_next[0:3].flatten(), utils_math.quat_to_euler(eta_next[3:].flatten())))
        eta_history.append(etaaa.reshape(1, -1))  # Ensure eta is row-wise
        joint_history.append(joint_positions.copy())

    eta_history = np.vstack(eta_history)  # Stack rows for eta_history

    eta_history = np.array(eta_history)
    joint_history = np.array(joint_history)

    animate.animate_uvms(eta_history, joint_history, dt)

    #### ---------------------------------------------------------------

    eta_history_real = []

    joint_history_real = []

    for _ in range(t_range):
        UVMS_MODEL_INSTANCE.update(dt, uq, uv, USE_PWM, V_BAT) 
        eta, joint_positions = UVMS_MODEL_INSTANCE.get_uvms_configuration()

        eta_history_real.append(eta.reshape(1, -1))  # Ensure eta is row-wise
        joint_history_real.append(joint_positions.copy())

    eta_history_real = np.vstack(eta_history_real)  # Stack rows for eta_history

    eta_history_real = np.array(eta_history_real)
    joint_history_real = np.array(joint_history_real)

    animate.animate_uvms(eta_history_real, joint_history_real, dt)

    ### --------------------------------------------------------------------
    import matplotlib.pyplot as plt

    # Plot end-effector (last joint) global position over time
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))

    # End-effector position from symbolic simulation
    ee_pos = joint_history[:, -1, :]  # shape: (timesteps, 3)
    ax[0].plot(ee_pos[:, 0], label='EE x (symbolic)')
    ax[0].plot(ee_pos[:, 1], label='EE y (symbolic)')
    ax[0].plot(ee_pos[:, 2], label='EE z (symbolic)')

    # End-effector position from real model
    ee_pos_real = joint_history_real[:, -1, :]
    ax[0].plot(ee_pos_real[:, 0], '--', label='EE x (real)')
    ax[0].plot(ee_pos_real[:, 1], '--', label='EE y (real)')
    ax[0].plot(ee_pos_real[:, 2], '--', label='EE z (real)')

    ax[0].set_title('End-Effector Global Position Over Time')
    ax[0].set_xlabel('Time step')
    ax[0].set_ylabel('Position [m]')
    ax[0].legend()
    ax[0].grid()

    # Plot vehicle position (eta[0:3]) over time
    veh_pos = eta_history[:, 0:3]
    veh_pos_real = eta_history_real[:, 0:3]

    ax[1].plot(veh_pos[:, 0], label='Vehicle x (symbolic)')
    ax[1].plot(veh_pos[:, 1], label='Vehicle y (symbolic)')
    ax[1].plot(veh_pos[:, 2], label='Vehicle z (symbolic)')

    ax[1].plot(veh_pos_real[:, 0], '--', label='Vehicle x (real)')
    ax[1].plot(veh_pos_real[:, 1], '--', label='Vehicle y (real)')
    ax[1].plot(veh_pos_real[:, 2], '--', label='Vehicle z (real)')

    ax[1].set_title('Vehicle Position Over Time')
    ax[1].set_xlabel('Time step')
    ax[1].set_ylabel('Position [m]')
    ax[1].legend()
    ax[1].grid()

    plt.tight_layout()
    plt.show()

    


        





    # uvms_model_instance.update(dt, uq, uv, use_pwm, V_bat) 
    # eta, joint_positions = uvms_model_instance.get_uvms_configuration()





def init_uvms_model(
    bluerov_params_path: str = 'model_params.yaml',
    dh_params_path: str = 'alpha_kin_params.yaml',
    joint_limits_path: Optional[str] = None,
    manipulator_dyn_params_paths: Optional[list[str]] = None,
    integrator: str = 'euler',
    use_quaternion: bool = True,
    use_pwm: bool = True,
    v_bat: float = 16.0
) -> None:
    """Einmalige Initialisierung. Idempotent: mehrfacher Aufruf macht nichts."""
    global INITIALIZED, BLUEROV_DYN, MANIP_DYN, MANIP_KIN, TAU_COUPLING, DYN_FOSSEN, J_DYN_FOSSEN, INTEGRATOR_FUNC, J_EEF, EEF_POSE
    global USE_QUATERNION, USE_PWM, V_BAT, L, MIXER, M_INV, JOINT_LIMITS, JOINT_EFFORTS, JOINT_VELOCITIES, INTEGRATOR, N_DOF, N_JOINTS, C_FUN, D_FUN, G_FUN, J_FUN

    global Q0, POS0, ATT0_EULER, VEL0, OMEGA0, UVMS_MODEL_INSTANCE

    with _LOCK:
        if INITIALIZED:
            return

        # Konfig setzen
        INTEGRATOR = integrator.lower()
        USE_QUATERNION = use_quaternion
        USE_PWM = use_pwm
        V_BAT = v_bat

        # Modelle laden/bauen
        brv_params = utils_math.load_model_params(bluerov_params_path)
        BLUEROV_DYN = sym_brv.BlueROVDynamicsSymbolic(brv_params)

        manip_params = utils_math.load_dh_params(dh_params_path)
        # Falls du Limits brauchst:
        if joint_limits_path:
            JOINT_LIMITS, JOINT_EFFORTS, JOINT_VELOCITIES, _ = utils_math.load_joint_limits(joint_limits_path)
            # 5 entries each for 4 joints + end effector
            # JOINT_LIMITS: tupels (min, max) for each joint
            # JOINT_EFFORTS: max torques for each joint
            # JOINT_VELOCITIES: max velocities for each joint
        alpha_params = utils_math.load_dynamic_params(manipulator_dyn_params_paths)
        # Dein konkreter Dyn-Builder (Passe an deine API an):
        MANIP_KIN = sym_manip_kin.KinematicsSymbolic(manip_params)
        MANIP_DYN = sym_manip_dyn.DynamicsSymbolic(MANIP_KIN, alpha_params)

        C_FUN = BLUEROV_DYN.C
        D_FUN = BLUEROV_DYN.D
        G_FUN = BLUEROV_DYN.g_quat if USE_QUATERNION else BLUEROV_DYN.g
        J_FUN = BLUEROV_DYN.J_quat if USE_QUATERNION else BLUEROV_DYN.J

        L = BLUEROV_DYN.L
        MIXER = BLUEROV_DYN.mixer
        M_INV = BLUEROV_DYN.M_inv  

        N_JOINTS = MANIP_DYN.kinematics_.n_joints
        N_DOF = BLUEROV_DYN.M_inv.size1() 

        # CasADi-Funktionen bauen
        TAU_COUPLING = _build_tau_coupling_func()
        DYN_FOSSEN = _build_dynamics_fossen_func()
        J_DYN_FOSSEN = _build_Jac_dynamics_fossen()
        INTEGRATOR_FUNC = _build_integrator_state_update()
        J_EEF = _build_eef_jacobian()
        EEF_POSE = _build_eef_pose_function()

        INITIALIZED = True

        UVMS_MODEL_INSTANCE = uvms_model.UVMSModel(manip_params, alpha_params, brv_params, Q0, POS0, ATT0_EULER, VEL0, OMEGA0)

def is_initialized() -> bool:
    return INITIALIZED


def teardown_for_tests() -> None:
    """Nur für Tests: globale Referenzen löschen (vorsichtig verwenden)."""
    global INITIALIZED, BLUEROV_DYN, MANIP_DYN, TAU_COUPLING, DYN_FOSSEN, J_DYN_FOSSEN, INTEGRATOR_FUNC
    global USE_QUATERNION, USE_PWM, V_BAT, INTEGRATOR, N_DOF, N_JOINTS, G_FUN, J_FUN
    with _LOCK:
        INITIALIZED = False
        BLUEROV_DYN = None
        MANIP_DYN = None
        TAU_COUPLING = None
        DYN_FOSSEN = None
        G_FUN = None
        J_FUN = None
        INTEGRATOR_FUNC = None
        # Konstanten L/MIXER/M_INV/C_FUN/D_FUN bleiben als Modulkonstanten bestehen

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
        v_bat=16.0
    )
    test_symbolic_uvms_model()
    return

if __name__ == "__main__":
    main()