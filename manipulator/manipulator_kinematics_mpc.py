import time
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt

from manipulator.kinematics import Kinematics
from manipulator.kinematics_symbolic import KinematicsSymbolic

from common import utils_math
from common.my_package_path import get_package_path
from common.animate import (
    animate_trajectory,
    plot_eef_positions,
    plot_tracking_errors,
    plot_prediction_error,
    plot_joint_angles,
)


'''
MPC for eef trajectory trackoing of a manipulator.
Uses only kinematics.
'''

# -------------------- Trajectory Generation --------------------

def generate_trajectory_with_limits(DH_table, joint_limits, joint_velocities, T=10, fps=30):
    n_joints = DH_table.shape[0] - 1
    timesteps = int(T * fps)
    t = np.linspace(0, T, timesteps)
    q_traj = np.zeros((n_joints, timesteps))
    for i in range(n_joints):
        q_min, q_max = joint_limits[i]
        v_max = abs(joint_velocities[i])
        amplitude = 0.5 * (q_max - q_min)
        period = 2 * T
        w = 2 * np.pi * (i+1) / period
        max_amplitude_by_vel = v_max / w if w > 0 else amplitude
        amplitude = min(amplitude, abs(max_amplitude_by_vel))
        offset = 0.5 * (q_max + q_min)
        q_traj[i, :] = amplitude * np.sin(w * t + i) + offset
    return q_traj

def fourier_traj(a, b, q0, t):
    nF = len(a)
    omega_f = 2 * np.pi * 0.1
    q = np.zeros_like(t)
    dq = np.zeros_like(t)
    ddq = np.zeros_like(t)
    for j in range(1, nF+1):
        q += (a[j-1] / (omega_f*j)) * np.sin(omega_f*j*t) - (b[j-1] / (omega_f*j)) * np.cos(omega_f*j*t)
        dq += a[j-1]*np.cos(omega_f*j*t) + b[j-1]*np.sin(omega_f*j*t)
        ddq += -a[j-1]*omega_f*j*np.sin(omega_f*j*t) + b[j-1]*omega_f*j*np.cos(omega_f*j*t)
    q += q0
    return q, dq, ddq

def excitation_trajectory_with_fourier(T=10.0, fps=50):
    timesteps = int(T * fps)
    t = np.linspace(0, T, timesteps)
    coeffs = {
        0: {'a': [0.3, -0.1, 0.05, -0.02, 0.015, -0.01, 0.005, -0.003, 0.002, -0.001],
            'b': [-0.2, 0.15, -0.05, 0.02, -0.015, 0.01, -0.005, 0.003, -0.002, 0.001],
            'q0': 2.5},
        1: {'a': [-0.25, 0.12, -0.06, 0.03, -0.02, 0.015, -0.007, 0.004, -0.002, 0.001],
            'b': [0.25, -0.08, 0.04, -0.02, 0.015, -0.01, 0.005, -0.003, 0.002, -0.001],
            'q0': 2.0},
        2: {'a': [0.15, -0.05, 0.02, -0.01, 0.008, -0.005, 0.003, -0.002, 0.0015, -0.001],
            'b': [0.1, -0.05, 0.01, -0.005, 0.004, -0.003, 0.002, -0.0015, 0.001, -0.0005],
            'q0': 1.0},
        3: {'a': [0.6, -0.2, 0.1, -0.05, 0.04, -0.03, 0.02, -0.01, 0.005, -0.002],
            'b': [0.4, -0.1, 0.05, -0.02, 0.015, -0.01, 0.007, -0.004, 0.002, -0.001],
            'q0': 3.0}
    }
    q_traj = np.zeros((4, len(t)))
    dq_traj = np.zeros((4, len(t)))
    ddq_traj = np.zeros((4, len(t)))
    for i in range(4):
        a = coeffs[i]['a']
        b = coeffs[i]['b']
        q0 = coeffs[i]['q0']
        q_traj[i], dq_traj[i], ddq_traj[i] = fourier_traj(a, b, q0, t)
    return q_traj, dq_traj, ddq_traj, t

def fourier_traj_vehicle(a, b, p0, t, omega_f):
    nF = len(a)
    traj = np.zeros_like(t)
    vel = np.zeros_like(t)
    acc = np.zeros_like(t)
    for j in range(1, nF+1):
        traj += (a[j-1] / (omega_f*j)) * np.sin(omega_f*j*t) - (b[j-1] / (omega_f*j)) * np.cos(omega_f*j*t)
        vel += a[j-1]*np.cos(omega_f*j*t) + b[j-1]*np.sin(omega_f*j*t)
        acc += -a[j-1]*omega_f*j*np.sin(omega_f*j*t) + b[j-1]*omega_f*j*np.cos(omega_f*j*t)
    traj += p0
    return traj, vel, acc

def excitation_trajectory_vehicle_body(type, T=10.0, fps=50):
    timesteps = int(T * fps)
    t = np.linspace(0, T, timesteps)
    # Fourier coefficients for position (gentle movement)
    pos_coeffs = {
        'a': [0.2, -0.05, 0.02, -0.01, 0.005],
        'b': [0.1, -0.03, 0.01, -0.005, 0.002],
        'p0': [0.0, 0.0, 0.0]
    }
    # Fourier coefficients for orientation (gentle rotation)
    ori_coeffs = {
        'a': [0.05, -0.01, 0.005, -0.002, 0.001],
        'b': [0.03, -0.008, 0.003, -0.001, 0.0005],
        'o0': [1.0, 0.0, 0.0, 0.0]  # initial quaternion (no rotation)
    }
    omega_f = 2 * np.pi * 0.05  # lower frequency for gentle movement

    # Position
    pos_traj = np.zeros((3, timesteps))
    vel_traj = np.zeros((3, timesteps))
    acc_traj = np.zeros((3, timesteps))
    for i in range(3):
        pos_traj[i], vel_traj[i], acc_traj[i] = fourier_traj_vehicle(
            pos_coeffs['a'], pos_coeffs['b'], pos_coeffs['p0'][i], t, omega_f
        )

    for i in range(t.shape[0]):
        if type in ('zero_ref', 'zero_ref_and_wrench'):
            pos_traj[:, i] = np.zeros(3)
            vel_traj[:, i] = np.zeros(3)
            acc_traj[:, i] = np.zeros(3)
        else:
            pos_traj[:, i] = np.array([0.1, 0.2, 0.3])
            vel_traj[:, i] = np.array([0.05, 0.1, 0.15])
            acc_traj[:, i] = np.array([0.01, 0.02, 0.03])

    # Orientation (quaternion), generate gentle rotation about z axis
    angle_traj, ang_vel_traj, ang_acc_traj = fourier_traj_vehicle(
        ori_coeffs['a'], ori_coeffs['b'], 0.0, t, omega_f
    )

    for i in range(t.shape[0]):
        if type in ('zero_ref', 'zero_ref_and_wrench'):
            angle_traj[i] = 0.0
            ang_vel_traj[i] = 0.0
            ang_acc_traj[i] = 0.0
        else:
            angle_traj[i] = 0.1
            ang_vel_traj[i] = 0.05
            ang_acc_traj[i] = 0.01

    # Convert angle_traj to quaternion (rotation about z axis)
    quat_traj = np.zeros((4, timesteps))
    for i in range(timesteps):
        theta = angle_traj[i]
        quat_traj[:, i] = np.array([
            np.cos(theta/2),
            0.0,
            0.0,
            np.sin(theta/2)
        ])

    # Angular velocity and acceleration in body frame (only z axis)
    ang_vel_vec_traj = np.zeros((3, timesteps))
    ang_acc_vec_traj = np.zeros((3, timesteps))
    ang_vel_vec_traj[2] = ang_vel_traj
    ang_acc_vec_traj[2] = ang_acc_traj

    return pos_traj, quat_traj, vel_traj, ang_vel_vec_traj, acc_traj, ang_acc_vec_traj, t

# -------------------- Kinematics --------------------

def compute_forward_kinematics(DH_table, q_traj):
    n_joints = DH_table.shape[0] - 1
    kin = Kinematics(DH_table)
    eef_positions, eef_attitudes, all_links = [], [], []
    for i in range(q_traj.shape[1]):
        q = q_traj[:, i]
        kin.update(q)
        eef_positions.append(kin.get_eef_position())
        eef_attitudes.append(kin.get_eef_attitude())
        joint_positions = [np.array([0.0, 0.0, 0.0])]
        for idx in range(n_joints + 1):
            joint_positions.append(kin.get_link_position(idx))
        all_links.append(np.array(joint_positions))
    return np.array(eef_positions), np.array(eef_attitudes), np.array(all_links)

def kinematic_model(u, dt, q0):
    return q0 + dt * u

# -------------------- Tracking Error --------------------

def compute_tracking_errors(ref_eef_positions, eef_positions, ref_eef_attitudes, eef_attitudes):
    pos_error = np.linalg.norm(ref_eef_positions[:eef_positions.shape[0], :] - eef_positions, axis=1)
    att_error = np.array([
        utils_math.quaternion_error_Niklas(ref_eef_attitudes[i], eef_attitudes[i])
        for i in range(min(len(ref_eef_attitudes), len(eef_attitudes)))
    ])
    return pos_error, att_error

# -------------------- MPC --------------------

def run_mpc(DH_table, joint_limits, joint_efforts, joint_velocities):
    T_trajectory, fps = 10, 50
    dt = 1 / fps
    M, N = T_trajectory * fps, 10
    n_joints = DH_table.shape[0] - 1
    kin_sym = KinematicsSymbolic(DH_table)
    eef_pose_fun = kin_sym.eef_pose_function()
    q_traj, _, _, _ = excitation_trajectory_with_fourier(T=T_trajectory, fps=fps)
    # q_traj = generate_trajectory_with_limits(DH_table, joint_limits, joint_velocities, T_trajectory, fps)
    ref_eef_positions, ref_eef_attitudes, all_links = compute_forward_kinematics(DH_table, q_traj)
    if ref_eef_positions.shape[0] < M + N:
        pad_count = M + N - ref_eef_positions.shape[0]
        last_pos = ref_eef_positions[-1, :]
        last_att = ref_eef_attitudes[-1, :]
        ref_eef_positions = np.vstack([ref_eef_positions, np.tile(last_pos, (pad_count, 1))])
        ref_eef_attitudes = np.vstack([ref_eef_attitudes, np.tile(last_att, (pad_count, 1))])
    q_predict = np.empty((n_joints, N+1, M))
    u_optimal = np.empty((n_joints, M))
    cost = np.empty((M))
    q_real = np.empty((n_joints, M+1))
    q_real[:, 0] = q_traj[:, 0]
    start_time = time.time()
    for step in range(M):
        print(f"Step {step + 1} / {M}")
        ref_pos_for_horizon = ref_eef_positions[step:step+N, :].T
        ref_att_for_horizon = ref_eef_attitudes[step:step+N, :].T
        q_opt, u_optimal_horizon, Jopt = solve_cftoc(
            eef_pose_fun, N, dt, n_joints, q_real[:, step], ref_pos_for_horizon, ref_att_for_horizon,
            joint_limits, joint_efforts, joint_velocities
        )
        q_predict[:, :, step] = q_opt
        u_optimal[:, step] = u_optimal_horizon[:, 0]
        cost[step] = Jopt
        q_real[:, step+1] = kinematic_model(u_optimal[:, step], dt, q_real[:, step])
    elapsed_time = time.time() - start_time
    print(f"MPC completed in {elapsed_time:.2f} seconds")
    predErr = np.zeros((1, M - N + 1))
    for i in range(predErr.shape[1]):
        Error = q_real[:, i:i+N+1] - q_predict[:, :, i]
        predErr[0, i] = np.sum(np.linalg.norm(Error, axis=0))
    return q_real, q_traj, predErr, cost, ref_eef_positions, ref_eef_attitudes, all_links

def solve_cftoc(eef_pose_fun, N, dt, n_joints, q0, ref_eef_positions, ref_eef_attitudes, joint_limits, joint_efforts, joint_velocities):
    ref_eef_positions = ca.DM(ref_eef_positions)
    ref_eef_attitudes = ca.DM(ref_eef_attitudes)
    q0 = ca.DM(q0)
    joint_pos_lower = ca.DM([jl[0] for jl in joint_limits])
    joint_pos_upper = ca.DM([jl[1] for jl in joint_limits])
    joint_vel_lower = ca.DM([-abs(v) for v in joint_velocities])
    joint_vel_upper = ca.DM([abs(v) for v in joint_velocities])
    opti = ca.Opti()
    opts = {'ipopt.print_level': 0, 'print_time': 0}
    opti.solver('ipopt', opts)
    q = opti.variable(n_joints, N+1)
    u = opti.variable(n_joints, N)
    for i in range(N):
        opti.subject_to(q[:, i+1] == q[:, i] + dt * u[:, i])
    for i in range(N+1):
        opti.subject_to(q[:, i] >= joint_pos_lower[:-1])
        opti.subject_to(q[:, i] <= joint_pos_upper[:-1])
    for i in range(N):
        opti.subject_to(u[:, i] >= joint_vel_lower[:-1])
        opti.subject_to(u[:, i] <= joint_vel_upper[:-1])
    opti.subject_to(q[:, 0] == q0)
    opti.set_initial(q, ca.repmat(q0, 1, N+1))
    opti.set_initial(u, ca.DM.zeros(n_joints, N))
    def quaternion_error(q_goal, q_current):
        w_g, x_g, y_g, z_g = q_goal[0], q_goal[1], q_goal[2], q_goal[3]
        w_c, x_c, y_c, z_c = q_current[0], q_current[1], q_current[2], q_current[3]
        v_g = ca.vertcat(x_g, y_g, z_g)
        v_c = ca.vertcat(x_c, y_c, z_c)
        goal_att_tilde = ca.vertcat(
            ca.horzcat(0, -z_g, y_g),
            ca.horzcat(z_g, 0, -x_g),
            ca.horzcat(-y_g, x_g, 0)
        )
        att_error = w_g * v_c - w_c * v_g - goal_att_tilde @ v_c
        return att_error
    Q = ca.DM.eye(3)
    R = ca.DM.eye(n_joints) * 0.001
    cost = 0
    for i in range(N):
        eef_pose = eef_pose_fun(q[:, i])  # shape: (7, 1)
        pos_k = eef_pose[0:3]
        att_k = eef_pose[3:7]
        pos_error = ref_eef_positions[:, i] - pos_k
        att_error = quaternion_error(ref_eef_attitudes[:, i], att_k)
        cost += ca.mtimes([pos_error.T, Q, pos_error]) + ca.mtimes([att_error.T, Q, att_error]) + ca.mtimes([u[:, i].T, R, u[:, i]])
    opti.minimize(cost)
    sol = opti.solve()
    return sol.value(q), sol.value(u), sol.value(cost)

def mpc_test():
    manipulatro_package_path = get_package_path('manipulator')
    kin_params_path = manipulatro_package_path + "/config/alpha_kin_params.yaml"
    joint_limits_path = manipulatro_package_path + "/config/alpha_joint_lim_real.yaml"

    DH_table = utils_math.load_dh_params(kin_params_path)
    joint_limits, joint_efforts, joint_velocities, _ = utils_math.load_joint_limits(joint_limits_path)
    q_real, q_ref, predErr, cost, ref_eef_positions, ref_eef_attitudes, ref_all_links = run_mpc(
        DH_table, joint_limits, joint_efforts, joint_velocities
    )
    eef_positions, eef_attitudes, all_links = compute_forward_kinematics(DH_table, q_real)
    animate_trajectory(ref_all_links)
    animate_trajectory(all_links)
    plot_joint_angles(q_real)
    plot_joint_angles(q_ref)
    pos_error, att_error = compute_tracking_errors(ref_eef_positions, eef_positions, ref_eef_attitudes, eef_attitudes)
    plot_tracking_errors(pos_error, att_error)
    plot_eef_positions(ref_eef_positions, eef_positions)
    plot_prediction_error(predErr)
    plt.show()

def main():
    mpc_test()

if __name__ == "__main__":
    main()