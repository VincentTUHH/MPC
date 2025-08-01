import numpy as np
import casadi as ca
import yaml
import os
import time
import matplotlib.pyplot as plt
from types import SimpleNamespace
from copy import deepcopy

from kinematics import Kinematics
from dynamics import Dynamics
from dynamics_symbolic import DynamicsSymbolic
from scipy.spatial.transform import Rotation as R
from animate import (
    plot_wrench_vs_time_compare, plot_joint_trajectories, animate_trajectory,
    plot_eef_positions, plot_tracking_errors, plot_prediction_error, plot_joint_angles,
    plot_data_over_time
)

# -------------------- YAML & Config Utilities --------------------

def load_yaml(file_path):
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

def recursive_merge(dict1, dict2):
    result = deepcopy(dict1)
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = recursive_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result

def dict_to_namespace(d):
    if not isinstance(d, dict):
        return d
    return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})

def load_dh_parameters(yaml_filename):
    dh_yaml_path = os.path.join(os.path.dirname(__file__), yaml_filename)
    with open(dh_yaml_path, 'r') as f:
        dh_data = yaml.safe_load(f)
    links = []
    if isinstance(dh_data, dict) and '/**' in dh_data and 'ros__parameters' in dh_data['/**']:
        params = dh_data['/**']['ros__parameters']
        for key in sorted(params.keys()):
            link = params[key]
            if isinstance(link, dict):
                d = link.get('d', 0)
                theta = link.get('theta0', 0)
                a = link.get('a', 0)
                alpha = link.get('alp', 0)
                links.append([d, theta, a, alpha])
        DH_table = np.array(links)
    else:
        if isinstance(dh_data, dict) and 'dh_table' in dh_data:
            DH_table = np.array(dh_data['dh_table'])
        else:
            DH_table = np.array(dh_data)
    return DH_table

def load_joint_limits(yaml_filename):
    yaml_path = os.path.join(os.path.dirname(__file__), yaml_filename)
    with open(yaml_path, 'r') as f:
        joint_data = yaml.safe_load(f)
    joint_limits, joint_efforts, joint_velocities, all_joints = [], [], [], {}
    for joint_name in sorted(joint_data.keys()):
        entry = joint_data[joint_name]
        joint_limits.append((entry.get('lower', None), entry.get('upper', None)))
        joint_efforts.append(entry.get('effort', None))
        joint_velocities.append(entry.get('velocity', None))
        all_joints[joint_name] = entry
    return joint_limits, joint_efforts, joint_velocities, all_joints

def load_dynamics_parameters(file_paths):
    merged_dict = {}
    for file in file_paths:
        data = load_yaml(file)
        params = data.get('/**', {}).get('ros__parameters', {})
        merged_dict = recursive_merge(merged_dict, params)
    return dict_to_namespace(merged_dict)

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

def eef_wrench_contact(type, T=10.0, fps=50):
    timesteps = int(T * fps)
    t = np.linspace(0, T, timesteps)
    # Sinusoidal wrench: amplitude and offset for force and torque
    f_amp = np.array([0.5, 0.3, 0.2])
    f_offset = np.array([0.1, 0.8, 0.6])
    l_amp = np.array([0.1, 0.05, 0.03])
    l_offset = np.array([0.2, 0.3, 0.1])

    f_eef = np.zeros((3, timesteps))
    l_eef = np.zeros((3, timesteps))
    for i in range(3):
        f_eef[i, :] = f_offset[i] + f_amp[i] * np.sin(2 * np.pi * (i+1) * t / T)
        l_eef[i, :] = l_offset[i] + l_amp[i] * np.sin(2 * np.pi * (i+1) * t / T)

    for i in range(t.shape[0]):
        if type in ('zero_wrench', 'zero_ref_and_wrench'):
            f_eef[:, i] = np.zeros(3)
            l_eef[:, i] = np.zeros(3)
        else:
            f_eef[:, i] = f_amp
            l_eef[:, i] = l_amp

    return f_eef, l_eef, t

def export_trajectory_txt(filename, t, q_traj, dq_traj, ddq_traj):
    with open(filename, 'w') as f:
        for i in range(len(t)):
            row = [f"{t[i]:.3f}"]
            row += [f"{q_traj[j, i]:.6f}" for j in range(4)]
            row += [f"{dq_traj[j, i]:.6f}" for j in range(4)]
            row += [f"{ddq_traj[j, i]:.6f}" for j in range(4)]
            f.write(', '.join(row) + '\n')

def export_references_txt(filename, t, quat_traj, vel_traj, acc_traj, ang_vel_vec_traj, ang_acc_vec_traj, f_eef_traj, l_eef_traj):
    with open(filename, 'w') as f:
        for i in range(len(t)):
            row = [f"{t[i]:.3f}"]
            row += [f"{quat_traj[j, i]:.6f}" for j in range(4)]
            row += [f"{vel_traj[j, i]:.6f}" for j in range(3)]
            row += [f"{acc_traj[j, i]:.6f}" for j in range(3)]
            row += [f"{ang_vel_vec_traj[j, i]:.6f}" for j in range(3)]
            row += [f"{ang_acc_vec_traj[j, i]:.6f}" for j in range(3)]
            row += [f"{f_eef_traj[j, i]:.6f}" for j in range(3)]
            row += [f"{l_eef_traj[j, i]:.6f}" for j in range(3)]
            f.write(', '.join(row) + '\n')

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

def quaternion_error_np(q_goal, q_current):
    w_g, x_g, y_g, z_g = q_goal
    w_c, x_c, y_c, z_c = q_current
    v_g = np.array([x_g, y_g, z_g])
    v_c = np.array([x_c, y_c, z_c])
    goal_att_tilde = np.array([
        [0, -z_g, y_g],
        [z_g, 0, -x_g],
        [-y_g, x_g, 0]
    ])
    att_error = w_g * v_c - w_c * v_g - goal_att_tilde @ v_c
    return np.linalg.norm(att_error)

def compute_tracking_errors(ref_eef_positions, eef_positions, ref_eef_attitudes, eef_attitudes):
    pos_error = np.linalg.norm(ref_eef_positions[:eef_positions.shape[0], :] - eef_positions, axis=1)
    att_error = np.array([
        quaternion_error_np(ref_eef_attitudes[i], eef_attitudes[i])
        for i in range(min(len(ref_eef_attitudes), len(eef_attitudes)))
    ])
    return pos_error, att_error

# -------------------- MPC --------------------

def run_mpc(DH_table, joint_limits, joint_efforts, joint_velocities):
    T_trajectory, fps = 10, 50, 
    dt = 1/ fps
    M, N = T_trajectory * fps, 10
    n_joints = DH_table.shape[0] - 1
    kin = Kinematics(DH_table)
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
            kin, N, dt, n_joints, q_real[:, step], ref_pos_for_horizon, ref_att_for_horizon,
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

def solve_cftoc(kin, N, dt, n_joints, q0, ref_eef_positions, ref_eef_attitudes, joint_limits, joint_efforts, joint_velocities):
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
        eef_pose = kin.forward_kinematics_symbolic(q[:, i])
        pos_k = eef_pose[:3, 3]
        R_eef = eef_pose[:3, :3]
        att_k = kin.rotation_matrix_to_quaternion(R_eef)
        pos_error = ref_eef_positions[:, i] - pos_k
        att_error = quaternion_error(ref_eef_attitudes[:, i], att_k)
        cost += ca.sumsqr(pos_error) + ca.sumsqr(att_error)
        cost += ca.mtimes([pos_error.T, Q, pos_error]) + ca.mtimes([att_error.T, Q, att_error]) + ca.mtimes([u[:, i].T, R, u[:, i]])
    opti.minimize(cost)
    sol = opti.solve()
    return sol.value(q), sol.value(u), sol.value(cost)

# -------------------- Dynamics --------------------

def dynamics_recursive_newton_euler(model, q, dq, ddq, f_eef, l_eef, v_ref, a_ref, w_ref, dw_ref, g_ref):
    n_links = model.get_number_of_links()
    model.update(q)
    model.forward(v_ref, a_ref, w_ref, dw_ref, g_ref)
    for i in range(1, n_links-1):
        model.link_forward(i, q[i-1], dq[i-1], ddq[i-1])
    model.link_forward(5, None, None, None)
    model.backward(f_eef, l_eef)
    for i in range(n_links - 2, -1, -1):
        model.link_backward(i)
    f_coupling, l_coupling = model.get_reference_wrench()
    return np.concatenate((f_coupling, l_coupling))

def read_wrench_txt(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            parts = [float(x.strip()) for x in line.split(',')]
            data.append(parts)
    data = np.array(data)
    t = data[:, 0]
    tau = data[:, 1:7]
    return t, tau

# -------------------- Symbolic Dynamics --------------------

def rneM_symbolic_func(dyn_model):
    n_joints = 4  # Example number of joints, adjust as needed
    q = ca.MX.sym('q', n_joints)
    dq = ca.MX.sym('dq', n_joints)
    ddq = ca.MX.sym('ddq', n_joints)
    v_ref = ca.MX.sym('v_ref', 3)
    a_ref = ca.MX.sym('a_ref', 3)
    w_ref = ca.MX.sym('w_ref', 3)
    dw_ref = ca.MX.sym('dw_ref', 3)
    quaternion_ref = ca.MX.sym('quat_ref', 4)
    f_eef = ca.MX.sym('f_eef', 3)
    l_eef = ca.MX.sym('l_eef', 3)

    tau = dyn_model.rnem_symbolic(q, dq, ddq, v_ref, a_ref, w_ref, dw_ref, quaternion_ref, f_eef, l_eef)
    
    rneM_func = ca.Function(
        'rneM_func',
        [q, dq, ddq, v_ref, a_ref, w_ref, dw_ref, quaternion_ref, f_eef, l_eef],
        [tau]
    )
    
    return rneM_func

# -------------------- Test & Main --------------------

def dynamics_test(type):
    DH_table = load_dh_parameters('alpha_kin_params.yaml')
    file_paths = [
        'alpha_kin_params.yaml',
        'alpha_base_tf_params_bluerov.yaml',
        'alpha_inertial_params_dh.yaml'
    ]
    alpha_params = load_dynamics_parameters(file_paths)
    dyn = Dynamics(DH_table, alpha_params)
    T = 10.0
    q_traj, dq_traj, ddq_traj, t = excitation_trajectory_with_fourier(T=T)
    # export_trajectory_txt('data/states/01_08_joint_excitation_trajectory.txt', t, q_traj, dq_traj, ddq_traj)
    t_cpp, tau_cpp = read_wrench_txt(f'data/model_output/01_08_cpp_wrench_on_vehicle_{type}.txt')

    pos_traj, quat_traj, vel_traj, ang_vel_vec_traj, acc_traj, ang_acc_vec_traj, t = excitation_trajectory_vehicle_body(type)
    f_eef_traj, l_eef_traj, t = eef_wrench_contact(type)


    tau = []
    f_eef = np.zeros(3)
    l_eef = np.zeros(3)
    v_ref = np.zeros(3)
    a_ref = np.zeros(3)
    w_ref = np.zeros(3)
    dw_ref = np.zeros(3)
    g_ref = np.array([0.0, 0.0, -9.81])
    # g_ref = np.array([0.195425, -0.0995513, -9.80755])
    for i in range(q_traj.shape[1]):
        q = q_traj[:, i]
        dq = dq_traj[:, i]
        ddq = ddq_traj[:, i]

        quaternion_ref = quat_traj[:, i]  # use quaternion trajectory
        # print(f"Quaternion ref: {quaternion_ref}")
        # Rotate gravity vector g_ref by quaternion_ref
        # Convert quaternion_ref (w, x, y, z) to (x, y, z, w) for scipy
        # quat_xyzw = np.array([quaternion_ref[1], quaternion_ref[2], quaternion_ref[3], quaternion_ref[0]])
        r = R.from_quat(quaternion_ref, scalar_first=True)  # Use scalar_first=True for (w, x, y, z) format
        r_num = r.as_matrix()
        # print(f"Rotation matrix: {r_num}")
        g_ref_iter = r_num.T @ g_ref  # Apply rotation to gravity vector
        # print(f"Gravity vector after rotation: {g_ref_iter}")
        # g_ref_iter = r.apply(g_ref.T).T

        v_ref = vel_traj[:, i]
        a_ref = acc_traj[:, i]
        w_ref = ang_vel_vec_traj[:, i]
        dw_ref = ang_acc_vec_traj[:, i]

        f_eef = f_eef_traj[:, i]
        l_eef = l_eef_traj[:, i]

        tau.append(
            dynamics_recursive_newton_euler(
                dyn, q, dq, ddq, f_eef, l_eef, v_ref, a_ref, w_ref, dw_ref, g_ref_iter
            )
        )
    tau = np.array(tau)
    plot_wrench_vs_time_compare(t, tau, tau_cpp, title=f"Wrench Comparison: Python vs C++ real, {type}")
    # Compute error for each of the 6 variables in tau vs tau_cpp
    print("Comparing real and c++")
    if tau.shape[1] == tau_cpp.shape[1]:
        error = np.abs(tau - tau_cpp[:tau.shape[0], :])
        for i in range(6):
            total_error = np.mean(np.linalg.norm(error[:, i]))
            print(f"Total absolute error for variable {i+1}: {total_error:.6f}")
    else:
        print("tau and tau_cpp have different shapes, cannot compute error.")

    return tau

def dynamic_symbolic_test(type):
    DH_table = load_dh_parameters('alpha_kin_params.yaml')
    file_paths = [
        'alpha_kin_params.yaml',
        'alpha_base_tf_params_bluerov.yaml',
        'alpha_inertial_params_dh.yaml'
    ]
    alpha_params = load_dynamics_parameters(file_paths)
    dyn = DynamicsSymbolic(DH_table, alpha_params)
    T = 10.0
    q_traj, dq_traj, ddq_traj, t = excitation_trajectory_with_fourier(T=T)
    # export_trajectory_txt('data/states/01_08_joint_excitation_trajectory.txt', t, q_traj, dq_traj, ddq_traj)
    t_cpp, tau_cpp = read_wrench_txt(f'data/model_output/01_08_cpp_wrench_on_vehicle_{type}.txt')
    
    rneM_func = rneM_symbolic_func(dyn)

    f_eef = np.zeros(3)
    l_eef = np.zeros(3)
    v_ref = np.zeros(3)
    a_ref = np.zeros(3)
    w_ref = np.zeros(3)
    dw_ref = np.zeros(3)

    pos_traj, quat_traj, vel_traj, ang_vel_vec_traj, acc_traj, ang_acc_vec_traj, t = excitation_trajectory_vehicle_body(type)
    f_eef_traj, l_eef_traj, t = eef_wrench_contact(type)

    # export_references_txt(f'data/states/01_08_references_{type}.txt', t, quat_traj, vel_traj, acc_traj, ang_vel_vec_traj, ang_acc_vec_traj, f_eef_traj, l_eef_traj)
    
    tau = []

    for i in range(q_traj.shape[1]):
        q = q_traj[:, i]
        dq = dq_traj[:, i]
        ddq = ddq_traj[:, i]

        # quaternion_ref = np.array([1.0, 0.0, 0.0, 0.0]) # no rotation of body frame
        # # quaternion_ref = np.array([0.9998, 0.0050, 0.0100, 0.0075])

        quaternion_ref = quat_traj[:, i]  # use quaternion trajectory
        # print(f"Quaternion ref: {quaternion_ref}")

        v_ref = vel_traj[:, i]
        a_ref = acc_traj[:, i]
        w_ref = ang_vel_vec_traj[:, i]
        dw_ref = ang_acc_vec_traj[:, i]

        # print(f"v_ref: {v_ref}, a_ref: {a_ref}, w_ref: {w_ref}, dw_ref: {dw_ref}")

        f_eef = f_eef_traj[:, i]
        l_eef = l_eef_traj[:, i]

        # print(f"f_eef: {f_eef}, l_eef: {l_eef}")

        out = rneM_func(q, dq, ddq, v_ref, a_ref, w_ref, dw_ref, quaternion_ref, f_eef, l_eef)
        tau.append(np.array(out).flatten())  # ensures shape (6,)


    tau = np.array(tau)

    plot_wrench_vs_time_compare(t, tau, tau_cpp, title=f"Wrench Comparison: Python vs C++ symbolic, {type}")
    print("Comparing symbolic and cpp:")
    if tau.shape[1] == tau_cpp.shape[1]:
        error = np.abs(tau - tau_cpp[:tau.shape[0], :])
        for i in range(6):
            total_error = np.mean(np.linalg.norm(error[:, i]))
            print(f"Total absolute error for variable {i+1}: {total_error:.6f}")
    else:
        print("tau and tau_cpp have different shapes, cannot compute error.")

    tau_real = dynamics_test(type)
    plot_wrench_vs_time_compare(t, tau, tau_real, title=f"Wrench Comparison: CasADi vs Numpy, {type}")
    print("Comparing CasADi vs Numpy:")
    if tau.shape[1] == tau_real.shape[1]:
        error = np.abs(tau - tau_real[:tau.shape[0], :])
        for i in range(6):
            total_error = np.mean(np.linalg.norm(error[:, i]))
            print(f"Total absolute error for variable {i+1}: {total_error:.6f}")
    else:
        print("tau and tau_cpp have different shapes, cannot compute error.")

    plt.show()


def mpc_test():
    DH_table = load_dh_parameters('alpha_kin_params.yaml')
    joint_limits, joint_efforts, joint_velocities, _ = load_joint_limits('alpha_joint_lim_real.yaml')
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
    # mpc_test()

    # Ask user for trajectory type
    print("Select trajectory type:")
    print("1: const")
    print("2: zero_wrench")
    print("3: zero_ref")
    print("4: zero_ref_and_wrench")
    choice = input("Enter number (1-4): ").strip()

    if choice == "1":
        traj_type = "const"
    elif choice == "2":
        traj_type = "zero_wrench"
    elif choice == "3":
        traj_type = "zero_ref"
    elif choice == "4":
        traj_type = "zero_ref_and_wrench"
    else:
        print("Invalid choice, defaulting to 'const'")
        traj_type = "const"

    # Pass traj_type to test functions
    dynamics_test(traj_type)
    dynamic_symbolic_test(traj_type)

    # dynamics_test()
    # dynamic_symbolic_test()


if __name__ == "__main__":
    main()
