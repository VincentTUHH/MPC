import numpy as np
import casadi as ca
from kinematics import Kinematics
import yaml
import os
from animate import plot_wrench_vs_time_compare, plot_wrench_vs_time, plot_joint_trajectories, animate_trajectory, plot_eef_positions, plot_tracking_errors, plot_prediction_error, plot_joint_angles
import time
import matplotlib.pyplot as plt
from types import SimpleNamespace
from copy import deepcopy

def load_yaml(file_path):
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

def recursive_merge(dict1, dict2):
    """Recursively merge dict2 into dict1."""
    result = deepcopy(dict1)
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = recursive_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result

def dict_to_namespace(d):
    """Convert dictionary to SimpleNamespace recursively for dot notation access."""
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
        print("DH parameters loaded from ROS2 YAML structure.")
    else:
        if isinstance(dh_data, dict) and 'dh_table' in dh_data:
            DH_table = np.array(dh_data['dh_table'])
        else:
            DH_table = np.array(dh_data)
    return DH_table

def load_joint_limits(yaml_filename):
    """
    Load joint limits and other parameters from a YAML file.
    Returns:
        joint_limits: list of (lower, upper) tuples for each joint, in file order
        joint_efforts: list of effort values for each joint, in file order
        joint_velocities: list of velocity values for each joint, in file order
        all_joints: dict with all joint entries
    """
    yaml_path = os.path.join(os.path.dirname(__file__), yaml_filename)
    with open(yaml_path, 'r') as f:
        joint_data = yaml.safe_load(f)
    joint_limits = []
    joint_efforts = []
    joint_velocities = []
    all_joints = {}
    for joint_name in sorted(joint_data.keys()):
        entry = joint_data[joint_name]
        lower = entry.get('lower', None)
        upper = entry.get('upper', None)
        effort = entry.get('effort', None)
        velocity = entry.get('velocity', None)
        joint_limits.append((lower, upper))
        joint_efforts.append(effort)
        joint_velocities.append(velocity)
        all_joints[joint_name] = entry
    return joint_limits, joint_efforts, joint_velocities, all_joints

def generate_trajectory(DH_table, T=10, fps=30):
    n_joints = DH_table.shape[0] - 1
    timesteps = int(T * fps)
    t = np.linspace(0, T, timesteps)
    q_traj = np.zeros((n_joints, timesteps))
    for i in range(n_joints):
        q_traj[i, :] = 0.8 * np.sin(2 * np.pi * (i+1) * t / T + i)
    return q_traj

def generate_trajectory_with_limits(DH_table, joint_limits, joint_velocities, T=10, fps=30):
    """
    Generate a trajectory that respects joint position and velocity limits.
    joint_limits: list of (min, max) tuples for each joint, e.g. [(-1, 1), ...]
    joint_velocities: list of max velocity values for each joint, e.g. [1.0, ...]
    """
    n_joints = DH_table.shape[0] - 1
    timesteps = int(T * fps)
    t = np.linspace(0, T, timesteps)
    q_traj = np.zeros((n_joints, timesteps))
    for i in range(n_joints):
        q_min, q_max = joint_limits[i]
        v_max = abs(joint_velocities[i])
        amplitude = 0.5 * (q_max - q_min)
        # Limit amplitude by velocity: A * w <= v_max => A <= v_max / w
        period = 2 * T  # period for the sine
        w = 2 * np.pi * (i+1) / period
        max_amplitude_by_vel = v_max / w if w > 0 else amplitude
        amplitude = min(amplitude, abs(max_amplitude_by_vel))
        offset = 0.5 * (q_max + q_min)
        q_traj[i, :] = amplitude * np.sin(w * t + i) + offset
    return q_traj

def compute_forward_kinematics(DH_table, q_traj):
    n_joints = DH_table.shape[0] - 1
    # print(f"Computing forward kinematics for {n_joints} joints.")
    kin = Kinematics(DH_table)
    # rotation_test = kin.get_rotation_iminus1_i(4)
    # print("Rotation matrix from i-1 to i with i = 5=ee computed.")
    # print(rotation_test)

    eef_positions = []
    eef_attitudes = []
    all_links = []
    for i in range(q_traj.shape[1]):
        q = q_traj[:, i]
        kin.update(q)
        pos = kin.get_eef_position()
        eef_positions.append(pos)
        att = kin.get_eef_attitude()
        eef_attitudes.append(att)
        joint_positions = [np.array([0.0, 0.0, 0.0])]
        for idx in range(n_joints + 1):
            joint_positions.append(kin.get_link_position(idx))
        joint_positions = np.array(joint_positions)
        all_links.append(joint_positions)
    return np.array(eef_positions), np.array(eef_attitudes), np.array(all_links)

def kinematic_model(u, dt, q0):
    return q0 + dt * u

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

def run_mpc(DH_table, joint_limits, joint_efforts, joint_velocities):
    T_trajectory = 10
    fps = 50
    dt = 1 / fps
    M = T_trajectory * fps
    N = 10

    n_joints = DH_table.shape[0] - 1
    kin = Kinematics(DH_table)
    # q_traj = generate_trajectory(DH_table, T_trajectory, fps)
    # q_traj = generate_trajectory_with_limits(DH_table, joint_limits, joint_velocities, T_trajectory, fps)
    q_traj, _, _, _ = exciation_trajectory_with_fourier(T=T_trajectory, fps=fps)
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

    start_time = time.time()  # Startzeit MPC messen

    for step in range(M):
        ref_pos_for_horizon = ref_eef_positions[step:step+N, :].T
        ref_att_for_horizon = ref_eef_attitudes[step:step+N, :].T
        q_opt, u_optimal_horizon, Jopt = solve_cftoc(
            kin, N, dt, n_joints, q_real[:, step], ref_pos_for_horizon, ref_att_for_horizon, joint_limits, joint_efforts, joint_velocities
        )
        # q_opt, u_optimal_horizon, Jopt = solve_cftoc_point_to_point(
        #     kin, N, dt, n_joints, q_real[:, step], ref_pos_for_horizon[:, -1], ref_att_for_horizon[:, -1],
        #     joint_limits, joint_efforts, joint_velocities
        # )
        q_predict[:, :, step] = q_opt
        u_optimal[:, step] = u_optimal_horizon[:, 0]
        cost[step] = Jopt
        q_real[:, step+1] = kinematic_model(u_optimal[:, step], dt, q_real[:, step])

    elapsed_time = time.time() - start_time  # Benötigte Zeit berechnen
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

    # Convert joint limits to CasADi DM arrays
    joint_pos_lower = ca.DM([jl[0] for jl in joint_limits])
    joint_pos_upper = ca.DM([jl[1] for jl in joint_limits])

    joint_vel_lower = ca.DM([-abs(v) for v in joint_velocities])
    joint_vel_upper = ca.DM([abs(v) for v in joint_velocities])
    joint_effort_lower = ca.DM([-abs(e) for e in joint_efforts])
    joint_effort_upper = ca.DM([abs(e) for e in joint_efforts])

    opti = ca.Opti()
    opts = {'ipopt.print_level': 0, 'print_time': 0}
    opti.solver('ipopt', opts)

    q = opti.variable(n_joints, N+1)
    u = opti.variable(n_joints, N)

    # Discrete kinematic model
    for i in range(N):
        opti.subject_to(q[:, i+1] == q[:, i] + dt * u[:, i])

    # Joint position constraints
    for i in range(N+1):
        opti.subject_to(q[:, i] >= joint_pos_lower[:-1])
        opti.subject_to(q[:, i] <= joint_pos_upper[:-1])

    # Joint velocity constraints
    for i in range(N):
        opti.subject_to(u[:, i] >= joint_vel_lower[:-1])
        opti.subject_to(u[:, i] <= joint_vel_upper[:-1])

    opti.subject_to(q[:, 0] == q0)
    opti.set_initial(q, ca.repmat(q0, 1, N+1))
    opti.set_initial(u, ca.DM.zeros(n_joints, N))

    def quaternion_error(q_goal, q_current):
        w_g = q_goal[0]
        x_g = q_goal[1]
        y_g = q_goal[2]
        z_g = q_goal[3]
        w_c = q_current[0]
        x_c = q_current[1]
        y_c = q_current[2]
        z_c = q_current[3]
        v_g = ca.vertcat(x_g, y_g, z_g)
        v_c = ca.vertcat(x_c, y_c, z_c)
        goal_att_tilde = ca.vertcat(
            ca.horzcat(0, -z_g, y_g),
            ca.horzcat(z_g, 0, -x_g),
            ca.horzcat(-y_g, x_g, 0)
        )
        att_error = w_g * v_c - w_c * v_g - goal_att_tilde @ v_c
        return att_error

    q_weight = ca.DM(1.0)
    Q = q_weight * ca.DM.eye(3)
    u_weight = ca.DM(0.001)
    R = u_weight * ca.DM.eye(n_joints)

    # TODO: Define different cost function, where eef can do anything but stay within an error margin
    # to the reference trajectory
    # so no time restriction (time as fast as possible)


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


def solve_cftoc_time_optimal(
    kin, N, dt, n_joints, q0, ref_eef_positions, ref_eef_attitudes,
    joint_limits, joint_efforts, joint_velocities, epsilon=0.02, epsilon_att=0.05, time_weight=1.0, u_weight=0.001
):
    """
    Time-optimal CFTOC: Minimize time to track trajectory, 
    with position and attitude constraints at start/end, 
    and position margin epsilon along the path.
    """
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
    tau = opti.variable()  # time scaling variable, tau in (0, 1], dt_actual = dt * tau

    opti.subject_to(tau >= 0.05)
    opti.subject_to(tau <= 1.0)

    dt_actual = dt * tau

    # Discrete kinematic model
    for i in range(N):
        opti.subject_to(q[:, i+1] == q[:, i] + dt_actual * u[:, i])

    # Joint position and velocity constraints
    for i in range(N+1):
        opti.subject_to(q[:, i] >= joint_pos_lower[:-1])
        opti.subject_to(q[:, i] <= joint_pos_upper[:-1])
    for i in range(N):
        opti.subject_to(u[:, i] >= joint_vel_lower[:-1])
        opti.subject_to(u[:, i] <= joint_vel_upper[:-1])

    opti.subject_to(q[:, 0] == q0)
    opti.set_initial(q, ca.repmat(q0, 1, N+1))
    opti.set_initial(u, ca.DM.zeros(n_joints, N))
    opti.set_initial(tau, 1.0)

    def quaternion_error(q_goal, q_current):
        w_g = q_goal[0]
        x_g = q_goal[1]
        y_g = q_goal[2]
        z_g = q_goal[3]
        w_c = q_current[0]
        x_c = q_current[1]
        y_c = q_current[2]
        z_c = q_current[3]
        v_g = ca.vertcat(x_g, y_g, z_g)
        v_c = ca.vertcat(x_c, y_c, z_c)
        goal_att_tilde = ca.vertcat(
            ca.horzcat(0, -z_g, y_g),
            ca.horzcat(z_g, 0, -x_g),
            ca.horzcat(-y_g, x_g, 0)
        )
        att_error = w_g * v_c - w_c * v_g - goal_att_tilde @ v_c
        return att_error

    # Path tracking constraints (position only, margin epsilon)
    for i in range(N+1):
        eef_pose = kin.forward_kinematics_symbolic(q[:, i])
        pos_k = eef_pose[:3, 3]
        opti.subject_to(ca.norm_2(ref_eef_positions[:, min(i, ref_eef_positions.shape[1]-1)] - pos_k) <= epsilon)

    # Initial and final attitude constraints (within epsilon_att)
    for idx in [0, N]:
        eef_pose = kin.forward_kinematics_symbolic(q[:, idx])
        R_eef = eef_pose[:3, :3]
        att_k = kin.rotation_matrix_to_quaternion(R_eef)
        att_error = quaternion_error(ref_eef_attitudes[:, min(idx, ref_eef_attitudes.shape[1]-1)], att_k)
        opti.subject_to(ca.norm_2(att_error) <= epsilon_att)

    # Objective: minimize time (tau), and control effort
    cost = time_weight * tau + u_weight * ca.sumsqr(u)
    opti.minimize(cost)

    sol = opti.solve()
    return sol.value(q), sol.value(u), sol.value(cost)

def solve_cftoc_point_to_point(
    kin, N, dt, n_joints, q0, ref_eef_positions, ref_eef_attitudes,
    joint_limits, joint_efforts, joint_velocities, epsilon=0.02, epsilon_att=0.05, u_weight=0.001
):
    """
    Point-to-point CFTOC: Find an optimal path from initial to final reference point,
    minimizing control effort and ensuring end-effector reaches the goal within a margin.
    """
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

    # Discrete kinematic model
    for i in range(N):
        opti.subject_to(q[:, i+1] == q[:, i] + dt * u[:, i])

    # Joint position and velocity constraints
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
        w_g = q_goal[0]
        x_g = q_goal[1]
        y_g = q_goal[2]
        z_g = q_goal[3]
        w_c = q_current[0]
        x_c = q_current[1]
        y_c = q_current[2]
        z_c = q_current[3]
        v_g = ca.vertcat(x_g, y_g, z_g)
        v_c = ca.vertcat(x_c, y_c, z_c)
        goal_att_tilde = ca.vertcat(
            ca.horzcat(0, -z_g, y_g),
            ca.horzcat(z_g, 0, -x_g),
            ca.horzcat(-y_g, x_g, 0)
        )
        att_error = w_g * v_c - w_c * v_g - goal_att_tilde @ v_c
        return att_error

    # Initial and final position constraints (within epsilon)
    eef_pose_init = kin.forward_kinematics_symbolic(q[:, 0])
    pos_init = eef_pose_init[:3, 3]
    opti.subject_to(ca.norm_2(ref_eef_positions[:, 0] - pos_init) <= epsilon)

    eef_pose_final = kin.forward_kinematics_symbolic(q[:, N])
    pos_final = eef_pose_final[:3, 3]
    opti.subject_to(ca.norm_2(ref_eef_positions[:, -1] - pos_final) <= epsilon)

    # Initial and final attitude constraints (within epsilon_att)
    R_init = eef_pose_init[:3, :3]
    att_init = kin.rotation_matrix_to_quaternion(R_init)
    att_error_init = quaternion_error(ref_eef_attitudes[:, 0], att_init)
    opti.subject_to(ca.norm_2(att_error_init) <= epsilon_att)

    R_final = eef_pose_final[:3, :3]
    att_final = kin.rotation_matrix_to_quaternion(R_final)
    att_error_final = quaternion_error(ref_eef_attitudes[:, -1], att_final)
    opti.subject_to(ca.norm_2(att_error_final) <= epsilon_att)

    # Objective: minimize control effort along the path
    cost = u_weight * ca.sumsqr(u)
    opti.minimize(cost)

    sol = opti.solve()
    return sol.value(q), sol.value(u), sol.value(cost)


# dynamics of the manipulator given the recursive newton-euler equations
# given the joint angles q, the joint velocities dq, and the joint accelerations ddq
# returns the joint torque tau_base at the base of the manipulator
def dynamics_recursive_newton_euler(model, q, dq, ddq, f_eef, l_eef, v_ref, a_ref, w_ref, dw_ref, g_ref):
    # 1. compute kinematics
    # 1.1. linear and angular velocities of each link i
    # 1.2. linear and angular accelerations of each link i
    # To compute the velocities and accelerations of each link using DH parameters,
    # you need to propagate angular and linear velocities/accelerations along the chain.
    # This is typically done using the recursive Newton-Euler algorithm.

    
    # Print current values of q, dq, ddq
    print("Current q[0] values: ", ' '.join(f"{val:.5f}" for val in q))
    print("Current dq[0] values: ", ' '.join(f"{val:.5f}" for val in dq))
    print("Current ddq[0] values:", ' '.join(f"{val:.6f}" for val in ddq))
    # Initialize arrays for velocities and accelerations
    n_links = model.get_number_of_links()
    model.update(q) # -> updates especially R included in T

    model.forward(v_ref, a_ref, w_ref, dw_ref, g_ref)

    for i in range(1, n_links-1):
        model.link_forward(i, q[i-1], dq[i-1], ddq[i-1])
    model.link_forward(5, None, None, None)  # End-effector link

    # # Print current kinematics for each link
    # kin_data = model.get_current_kinematics()
    # n_links = len(kin_data['v'])
    # for i in range(n_links):
    #     print(f"Link ID: {i}")
    #     for key in ['v', 'a', 'w', 'dw', 'g', 'v_c', 'a_c', 'dv_c', 'v_b']:
    #         arr = kin_data.get(key, [None]*n_links)[i]
    #         if arr is None:
    #             arr_str = "None"
    #         else:
    #             arr_str = ' '.join(f"{x:.8g}" for x in arr)
    #         print(f"{key}: {arr_str}")

    # # Print rotation matrices from i-1 to i for each link
    # print(f"Rotation matrix from ref to link 0:")
    # print(model.get_R_reference())
    # for i in range(1,6,1):
    #     R_iminus1_i = model.get_rotation_iminus1_i(i)
    #     print(f"Rotation matrix from link {i-1} to {i}:")
    #     print(R_iminus1_i)

    model.backward(f_eef, l_eef)

    for i in range(n_links - 2, 0 - 1, -1):
        model.link_backward(i)

    # tau_data = model.get_current_wrench()
    # n_links = len(tau_data['f'])
    # for i in reversed(range(n_links)):
    #     print(f"Link ID: {i}")
    #     for key in ['f', 'l']:
    #         arr = tau_data.get(key, [None]*n_links)[i]
    #         if arr is None:
    #             arr_str = "None"
    #         else:
    #             arr_str = ' '.join(f"{x:.8g}" for x in arr)
    #         print(f"{key}: {arr_str}")

    # # Print dynamic and hydrodynamic parameters for each link
    # dyn_data = model.get_current_dynamic_parameters()
    # n_links = len(dyn_data['I'])
    # print("Dynamic and Hydrodynamic Parameters:")
    # for i in range(n_links):
    #     print(f"Link ID: {i}")
    #     print("Inertia Matrix (I):\n", dyn_data['I'][i])
    #     print("Added Mass Matrix (M_a):\n", dyn_data['M_a'][i])
    #     print("Added Mass Matrix 12 (M12):\n", dyn_data['M12'][i])
    #     print("Added Mass Matrix 21 (M21):\n", dyn_data['M21'][i])
    #     print("Added Inertia Matrix (I_a):\n", dyn_data['I_a'][i])
    #     print("Buoyancy Mass (m_buoy):", dyn_data['m_buoy'][i])
    #     print("Center of Gravity Offset (r_c):", dyn_data['r_c'][i].T if dyn_data['r_c'][i] is not None else None)
    #     print("Center of Buoyancy Offset (r_b):", dyn_data['r_b'][i].T if dyn_data['r_b'][i] is not None else None)
    #     print("Mass (m):", dyn_data['m'][i])
    #     print("Linear Damping Parameters:", dyn_data['lin_damp_param'][i].T if dyn_data['lin_damp_param'][i] is not None else None)
    #     print("Nonlinear Damping Parameters:", dyn_data['nonlin_damp_param'][i].T if dyn_data['nonlin_damp_param'][i] is not None else None)
    #     print("Translational Damping Matrix (D_t):\n", dyn_data['D_t'][i])
    #     print("Rotational Damping Matrix (D_r):\n", dyn_data['D_r'][i])
    #     print("COB to COG Offset (r_cb):", dyn_data['r_cb'][i].T if dyn_data['r_cb'][i] is not None else None)
    #     print()

    f_coupling, l_coupling = model.get_reference_wrench()

    return np.concatenate((f_coupling, l_coupling))

def excitation_trajectory(T=10.0, fps=50):
    q_min = np.array([0.0, 5/9 * np.pi, 0.0, 0.0])
    q_max = np.array([2*np.pi, np.pi, np.pi, 2*np.pi])
    dq_min = np.array([-0.7, -0.7, -0.7, -0.7])
    dq_max = -dq_min
    ddq_min = np.array([-2, -2, -2, -2 ])
    ddq_max = -ddq_min

    dq_0 = np.array([0.0, 0.0, 0.0, 0.0])

    T = T
    fps = fps

    timesteps = int(T * fps)
    t = np.linspace(0, T, timesteps)

    # Use sum of sinusoids with different frequencies and amplitudes within joint limits
    n_joints = q_min.shape[0]
    q_traj = np.zeros((n_joints, timesteps))
    dq_traj = np.zeros((n_joints, timesteps))
    ddq_traj = np.zeros((n_joints, timesteps))

    for i in range(n_joints):
        # Frequencies chosen to avoid harmonics and excite all modes
        freqs = np.array([0.5, 1.0, 1.5, 2.0]) * (i+1)
        amp = 0.4 * (q_max[i] - q_min[i]) / len(freqs)
        offset = 0.5 * (q_max[i] + q_min[i])
        q_traj[i, :] = offset
        for f in freqs:
            q_traj[i, :] += amp * np.sin(2 * np.pi * f * t / T + np.random.uniform(0, 2*np.pi))
        # Clip to joint limits
        q_traj[i, :] = np.clip(q_traj[i, :], q_min[i], q_max[i])
        dq_traj[i, :] = np.gradient(q_traj[i, :], t)
        dq_traj[i, :] = np.clip(dq_traj[i, :], dq_min[i], dq_max[i])
        ddq_traj[i, :] = np.gradient(dq_traj[i, :], t)
        ddq_traj[i, :] = np.clip(ddq_traj[i, :], ddq_min[i], ddq_max[i])

    return q_traj, dq_traj, ddq_traj, t



def fourier_traj(a, b, q0, t):
    nF = 10
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

def exciation_trajectory_with_fourier(T=10.0, fps=50):
    timesteps = int(T * fps)
    t = np.linspace(0, T, timesteps)
    # coeffs = {
    #     0: {'a': [0.3, -0.1, 0.05, -0.02] + [0]*6,   'b': [-0.2, 0.15, -0.05, 0.02] + [0]*6, 'q0': 2.5},
    #     1: {'a': [-0.25, 0.12, -0.06, 0.03] + [0]*6, 'b': [0.25, -0.08, 0.04, -0.02] + [0]*6, 'q0': 2.0},
    #     2: {'a': [0.15, -0.05, 0.02, -0.01] + [0]*6, 'b': [0.1, -0.05, 0.01, -0.005] + [0]*6, 'q0': 1.0},
    #     3: {'a': [0.6, -0.2, 0.1, -0.05] + [0]*6,    'b': [0.4, -0.1, 0.05, -0.02] + [0]*6,   'q0': 3.0}
    # }
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

    # Fill matrices
    for i in range(4):
        a = coeffs[i]['a']
        b = coeffs[i]['b']
        q0 = coeffs[i]['q0']
        q_traj[i], dq_traj[i], ddq_traj[i] = fourier_traj(a, b, q0, t)

    return q_traj, dq_traj, ddq_traj, t

def export_trajectory_txt(filename, t, q_traj, dq_traj, ddq_traj):
    with open(filename, 'w') as f:
        for i in range(len(t)):
            row = [f"{t[i]:.3f}"]
            row += [f"{q_traj[j, i]:.6f}" for j in range(4)]
            row += [f"{dq_traj[j, i]:.6f}" for j in range(4)]
            row += [f"{ddq_traj[j, i]:.6f}" for j in range(4)]
            f.write(', '.join(row) + '\n')

def read_wrench_txt(filename):
    """
    Reads a .txt file with rows: time, f0x, f0y, f0z, l0x, l0y, l0z (comma separated).
    Returns:
        t: numpy array of times
        tau: numpy array of shape (N, 6) with forces and torques
    """
    data = []
    with open(filename, 'r') as f:
        for line in f:
            parts = [float(x.strip()) for x in line.split(',')]
            data.append(parts)
    data = np.array(data)
    t = data[:, 0]
    tau = data[:, 1:7]
    return t, tau




def main():
    DH_table = load_dh_parameters('alpha_kin_params.yaml')

    file_paths = ['alpha_kin_params.yaml', 'alpha_base_tf_params_bluerov.yaml', 'alpha_inertial_params_dh.yaml']
    merged_dict = {}
    for file in file_paths:
        data = load_yaml(file)
        params = data.get('/**', {}).get('ros__parameters', {})
        merged_dict = recursive_merge(merged_dict, params)
    alpha_params = dict_to_namespace(merged_dict)   # Convert to dot-access

    kin = Kinematics(DH_table, alpha_params)

    T = 10.0
    q_traj, dq_traj, ddq_traj, t = exciation_trajectory_with_fourier(T = T)

    # plot_joint_trajectories(t, q_traj, dq_traj, ddq_traj)
    # plt.show()

   

    export_trajectory_txt('data/states/22_07_measurement_wet_1.txt', t, q_traj, dq_traj, ddq_traj)

    t_cpp, tau_cpp = read_wrench_txt('data/model_output/22_07_measurement_wet_1.txt')

    tau = []

    f_eef = np.array([0.0, 0.0, 0.0])
    l_eef = np.array([0.0, 0.0, 0.0])
    v_ref = np.array([0.0, 0.0, 0.0])
    a_ref = np.array([0.0, 0.0, 0.0])
    w_ref = np.array([0.0, 0.0, 0.0])
    dw_ref = np.array([0.0, 0.0, 0.0])
    g_ref = np.array([0.0, 0.0, -9.81])

    for i in range(q_traj.shape[1]):
    # for i in range(0, 2, 1):
        q = q_traj[:, i]
        dq = dq_traj[:, i]
        ddq = ddq_traj[:, i]

        tau.append(
            dynamics_recursive_newton_euler(
                kin, q, dq, ddq, f_eef, l_eef, v_ref, a_ref, w_ref, dw_ref, g_ref
            )
        )

    tau = np.array(tau)  # shape: (timesteps, 6)
    
    # plot_wrench_vs_time(t, tau, title='Wrench from Python Recursive Newton-Euler Dynamics')
    # plot_wrench_vs_time(t_cpp, tau_cpp, title='Wrench from C++ Model Output')

    plot_wrench_vs_time_compare(t, tau, tau_cpp, title="Wrench Comparison: Python vs C++")

    plt.show()
    
    # ref_eef_positions, ref_eef_attitudes, all_links = compute_forward_kinematics(DH_table, q_traj)
    # animate_trajectory(all_links)


    return

    DH_table = load_dh_parameters('alpha_kin_params.yaml')
    joint_limits, joint_efforts, joint_velocities, all_joints = load_joint_limits('alpha_joint_lim_real.yaml')
    q_real, q_ref, predErr, cost, ref_eef_positions, ref_eef_attitudes, ref_all_links = run_mpc(DH_table, joint_limits, joint_efforts, joint_velocities)
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

if __name__ == "__main__":
    main()
