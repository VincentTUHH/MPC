import numpy as np
import casadi as ca
from kinematics import Kinematics
import yaml
import os
from animate import animate_trajectory, plot_eef_positions, plot_tracking_errors, plot_prediction_error, plot_joint_angles
import time
import matplotlib.pyplot as plt

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
    kin = Kinematics(DH_table)
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
    q_traj = generate_trajectory_with_limits(DH_table, joint_limits, joint_velocities, T_trajectory, fps)
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

def main():
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
