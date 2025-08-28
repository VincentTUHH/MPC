import numpy as np
import casadi as ca
import matplotlib.pyplot as plt

from uvms import model as uvms_model
from uvms import uvms
from manipulator import kinematics_symbolic as sym_manip_kin

from common import utils_math
from common import animate

Q0 = np.array([0.8, 8.0, 0.4, 0.2]) #np.array([np.pi, np.pi * 0.5, np.pi * 0.75, np.pi * 0.5])
POS0 = np.zeros(3) # np.array([-0.2, -0.5, -0.8])
ATT0_EULER = np.zeros(3) #np.array([0.0, 0.0, np.pi/4])  # Convert Euler angles to quaternion
ATT0_QUAT = utils_math.euler_to_quat(ATT0_EULER[0], ATT0_EULER[1], ATT0_EULER[2])
VEL0 = np.array([0.0, 0.0, 0.0])
OMEGA0 = np.array([0.0, 0.0, 0.0])

UV = np.array([0.2, 0.2, 0.0, -0.1, 0.0, 0.1, 0.0, 0.0])  # Example control input
UQ = np.zeros(4) # np.array([0.0, 0.2, 0.1, 0.0])  # Example joint velocities
DT = 0.05
T_RANGE = 10

F_EEF = np.array([0.0, 0.0, 0.0])
L_EEF = np.array([0.0, 0.0, 0.0])

JOINT_POS = None


def build_joint_positions_function():
    q = ca.MX.sym('q', uvms.N_JOINTS)
    R_I_B = ca.MX.sym('R_I_B', 3, 3)
    pos_vehicle = ca.MX.sym('pos_v', 3)

    kin = sym_manip_kin.KinematicsSymbolic(uvms.MANIP_PARAMS)
    kin.update(q)

    R_B_0 = np.array(uvms.MANIP_DYN.R_reference)
    r_B_0 = np.array(uvms.MANIP_DYN.tf_vec)

    joint_positions = []

    joint_positions.append(pos_vehicle + ca.mtimes(R_I_B, r_B_0))
    for i in range(uvms.N_JOINTS + 1):
        pos_expr = kin.get_link_position(i)
        joint_positions.append(joint_positions[0] + ca.mtimes(R_I_B @ R_B_0, pos_expr))
    joint_positions_array = ca.hcat(joint_positions).T
    return ca.Function('joint_positions', [q, R_I_B, pos_vehicle], [joint_positions_array])

def test_symbolic_model():
    x_next = np.zeros(uvms.STATE_DIM)  # State vector for next step
    x = np.zeros(uvms.STATE_DIM)  # Initial state vector
    x[:uvms.N_JOINTS] = Q0  # Set initial joint angles
    x[uvms.N_JOINTS:uvms.N_JOINTS+uvms.N_DOF] = np.concatenate((VEL0, OMEGA0))  # Set initial velocities
    x[uvms.N_JOINTS+uvms.N_DOF:] = np.concatenate((POS0, ATT0_QUAT))  # Set initial position and attitude

    u = np.zeros(uvms.CTRL_DIM)  # Control input vector
    u[:uvms.N_JOINTS] = UQ  # Set joint velocities
    u[uvms.N_JOINTS:] = UV  # Set end-effector velocities

    joint_positions = np.zeros((uvms.N_JOINTS + 2, 3))

    R_B_0 = np.array(uvms.MANIP_DYN.R_reference)
    r_B_0 = np.array(uvms.MANIP_DYN.tf_vec)

    eta_history = []

    joint_history = []

    last_uq = np.zeros(uvms.N_JOINTS)

    dnu_g = np.zeros(6)

    for _ in range(T_RANGE):
        ddq = (u[:uvms.N_JOINTS] - last_uq) / DT

        x_next, dnu_next, J_eef, p_eef, aatt_eef = uvms.STEP(DT, x, u, ddq, dnu_g, F_EEF, L_EEF)
        eta_next = x_next[uvms.N_JOINTS + uvms.N_DOF:]

        # tau_c = uvms.TAU_COUPLING(x[:uvms.N_JOINTS], u[:uvms.N_JOINTS], ddq, x[uvms.N_JOINTS:uvms.N_JOINTS+3], dnu_g[:3], x[uvms.N_JOINTS+3:uvms.N_JOINTS+uvms.N_DOF:], dnu_g[3:], x[uvms.N_JOINTS+uvms.N_DOF+3:], F_EEF, L_EEF)
        # print(tau_c)

        R_I_B = utils_math.rotation_matrix_from_quat(eta_next[3:])
        etaaa = np.concatenate((np.array(eta_next[0:3]).reshape(-1), utils_math.quat_to_euler(np.array(eta_next[3:]).reshape(-1))))
        eta_history.append(etaaa)  # Ensure eta is row-wise
        joint_history.append(JOINT_POS(x_next[:uvms.N_JOINTS], R_I_B, eta_next[0:3]))
        # print(joint_history[0])

        x = x_next
        last_uq = u[:uvms.N_JOINTS]
        dnu_g = dnu_next


    eta_history = np.array(eta_history)
    joint_history = np.array(joint_history)

    animate.animate_uvms(eta_history, joint_history, DT)

    return joint_history, eta_history

def test_real_model():
    uvms_model_instance = uvms_model.UVMSModel(uvms.MANIP_PARAMS, uvms.ALPHA_PARAMS, uvms.BRV_PARAMS, Q0, POS0, ATT0_EULER, VEL0, OMEGA0)
    print("UVMS Model initialized successfully.")

    eta_history = []

    joint_history = []

    for _ in range(T_RANGE):
        uvms_model_instance.update(DT, UQ, UV, uvms.USE_PWM, uvms.V_BAT) 
        eta, joint_positions = uvms_model_instance.get_uvms_configuration()

        eta_history.append(eta.reshape(1, -1))  # Ensure eta is row-wise
        joint_history.append(joint_positions.copy())

    eta_history = np.vstack(eta_history)  # Stack rows for eta_history

    eta_history = np.array(eta_history)
    joint_history = np.array(joint_history)

    # print(eta_history) 
    # print(joint_history)
    animate.animate_uvms(eta_history, joint_history, DT)

    return joint_history, eta_history

def plottttt(joint_history, joint_history_real, eta_history, eta_history_real):
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


def main():
    global JOINT_POS
    uvms.main()

    # ddq = UQ / DT
    # uvms.check_fixed_point(Q0, np.concatenate((VEL0, OMEGA0)), np.concatenate((POS0, ATT0_QUAT)), UQ, UV, ddq, np.zeros(6), ATT0_QUAT, F_EEF, L_EEF)
    
    JOINT_POS = build_joint_positions_function()

    joint_history, eta_history = test_symbolic_model()

    joint_history_real, eta_history_real = test_real_model()

    plottttt(joint_history, joint_history_real, eta_history, eta_history_real)

    # Compute mean position error and mean attitude error for vehicle
    veh_pos_error = np.linalg.norm(eta_history[:, 0:3] - eta_history_real[:, 0:3], axis=1)
    mean_veh_pos_error = np.mean(veh_pos_error)

    veh_att_error = np.linalg.norm(eta_history[:, 3:6] - eta_history_real[:, 3:6], axis=1)
    mean_veh_att_error = np.mean(veh_att_error)

    # Compute mean position error for end-effector
    ee_pos_error = np.linalg.norm(joint_history[:, -1, :] - joint_history_real[:, -1, :], axis=1)
    mean_ee_pos_error = np.mean(ee_pos_error)

    print(f"Mean vehicle position error: {mean_veh_pos_error:.6f} m")
    print(f"Mean vehicle attitude error: {mean_veh_att_error:.6f} rad")
    print(f"Mean end-effector position error: {mean_ee_pos_error:.6f} m")


    

if __name__ == "__main__":
    main()