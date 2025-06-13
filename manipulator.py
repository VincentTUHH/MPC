import numpy as np
import casadi as ca
from kinematics import Kinematics
import yaml
import os
from animate import animate_trajectory
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

def generate_trajectory(DH_table, T=10, fps=30):
    n_joints = DH_table.shape[0] - 1
    timesteps = int(T * fps)
    t = np.linspace(0, T, timesteps)
    q_traj = np.zeros((n_joints, timesteps))
    for i in range(n_joints):
        q_traj[i, :] = 0.8 * np.sin(2 * np.pi * (i+1) * t / T + i)
    return q_traj

def compute_forward_kinematics(DH_table, q_traj):
    n_joints = DH_table.shape[0] - 1
    kin = Kinematics(DH_table)
    eef_positions = []
    eef_attitudes = []
    all_links = []
    # print(q_traj)
    for i in range(q_traj.shape[1]):
        q = q_traj[:, i]
        kin.update(q)
        pos = kin.get_eef_position()
        eef_positions.append(pos)
        att = kin.get_eef_attitude()
        eef_attitudes.append(att)
        joint_positions = []
        # Manually add link 0 at (0,0,0)
        joint_positions.append(np.array([0.0, 0.0, 0.0]))
        for idx in range(n_joints + 1):
            joint_positions.append(kin.get_link_position(idx))
        joint_positions = np.array(joint_positions)
        all_links.append(joint_positions)
    return np.array(eef_positions), np.array(eef_attitudes), np.array(all_links)

def kinematic_model(u, dt, q0):
    return q0 + dt * u

def run_mpc_example(DH_table, N, dt, q0):
    # Example: Using CasADi Opti for a High-Level optimization problem
    # Parameters
    N = 10 # Prediction horizon
    dt = 0.1 # Time step
    n_joints = DH_table.shape[0] - 1 # Number of joints based on DH table

    # Kinematic model of Reach Robotics manipulator from DH parameters
    kin = Kinematics(DH_table)

    # Reference end effector trajectory
    q_traj = generate_trajectory(DH_table)
    ref_eef_positions, ref_eef_attitudes, _ = compute_forward_kinematics(DH_table, q_traj)
    ref_eef_positions = ca.DM(ref_eef_positions)
    ref_eef_attitudes = ca.DM(ref_eef_attitudes)
    # Initial joint state (can be set to the first state of the trajectory)
    q0 = ca.DM(q_traj[0, :])

    # Create an optimization problem using CasADi
    opti = ca.Opti()
    opti.solver('ipopt')  # Set the solver to IPOPT 

    q = opti.variable(n_joints, N+1)  # joint states
    u = opti.variable(n_joints, N)    # joint velocity commands (qdot)

    # State update equation)
    for i in range(N):
        opti.subject_to(q[:, i+1] == q[:, i] + dt * u[:, i])  

    # Initial state constraint
    opti.subject_to(q[:, 0] == q0)  

    opti.set_initial(q, ca.repmat(q0, 1, N+1))
    opti.set_initial(u, ca.DM.zeros(n_joints, N))       


    def quaternion_error(q_goal, q_current):
        # q = [w, x, y, z]
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
        # Skew-symmetric matrix for v_g
        goal_att_tilde = ca.vertcat(
            ca.horzcat(0, -z_g, y_g),
            ca.horzcat(z_g, 0, -x_g),
            ca.horzcat(-y_g, x_g, 0)
        )
        att_error = w_g * v_c - w_c * v_g - goal_att_tilde @ v_c
        return att_error

    # For the prediction horizion N calculate the end effector error
    # with regard to the reference trajectory in each step
    cost = 0
    for i in range(N):
        eef_pose = kin.forward_kinematics_symbolic(q[:,i])
        pos_k = eef_pose[:3, 3]
        R_eef = eef_pose[:3, :3]
        att_k = kin.rotation_matrix_to_quaternion(R_eef)

        pos_error = ref_eef_positions[i, :].T - pos_k
        att_error = quaternion_error(ref_eef_attitudes[i, :], att_k)
        cost += ca.sumsqr(pos_error) + ca.sumsqr(att_error)

    opti.minimize(cost + ca.sumsqr(u))  # Add control effort cost

    sol = opti.solve()
    print("u:", sol.value(u[:,1]))# , "y:", sol.value(y), "z:", sol.value(z))
    # Extract the optimal control sequence

    q_real = kinematic_model(sol.value(u[:,1]), dt, q0)
    kin.update(q_real)
    q0 = q_real
    # .... von vorne
    print("q_real:", q_real)


def solve_cftoc(kin, N, dt, n_joints, q0, ref_eef_positions, ref_eef_attitudes):
    # Example: Using CasADi Opti for a High-Level optimization problem
    
    ref_eef_positions = ca.DM(ref_eef_positions)
    ref_eef_attitudes = ca.DM(ref_eef_attitudes)
    # Initial joint state (can be set to the first state of the trajectory)
    q0 = ca.DM(q0)

    # Create an optimization problem using CasADi
    opti = ca.Opti()
    # opti.solver('ipopt')
    opts = {'ipopt.print_level':0, 'print_time':0} # Suppress output
    opti.solver('ipopt', opts)

    q = opti.variable(n_joints, N+1)  # joint states
    u = opti.variable(n_joints, N)    # joint velocity commands (qdot)

    # State update equation)
    for i in range(N):
        opti.subject_to(q[:, i+1] == q[:, i] + dt * u[:, i])  


    # add joint limits for constraints
    # velocity limits constraint u
    # position limits constraint q (vorsicht!!!, weil die refernece trajectory natürlich diese constraint auch einhalten muss, was es gerade nicht tut)

    # Initial state constraint
    opti.subject_to(q[:, 0] == q0)  

    opti.set_initial(q, ca.repmat(q0, 1, N+1))
    opti.set_initial(u, ca.DM.zeros(n_joints, N))       


    def quaternion_error(q_goal, q_current):
        # q = [w, x, y, z]
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
        # Skew-symmetric matrix for v_g
        goal_att_tilde = ca.vertcat(
            ca.horzcat(0, -z_g, y_g),
            ca.horzcat(z_g, 0, -x_g),
            ca.horzcat(-y_g, x_g, 0)
        )
        att_error = w_g * v_c - w_c * v_g - goal_att_tilde @ v_c
        return att_error

    # For the prediction horizion N calculate the end effector error
    # with regard to the reference trajectory in each step
    q_weight = ca.DM(1.0)  # Weight for joint position cost
    Q = q_weight * ca.DM.eye(3)  # Joint position cost weight
    u_weight = ca.DM(0.01)  # Weight for joint velocity cost
    R = u_weight * ca.DM.eye(n_joints)  # Joint velocity cost weight
    

    cost = 0
    for i in range(N):
        eef_pose = kin.forward_kinematics_symbolic(q[:,i])
        pos_k = eef_pose[:3, 3]
        R_eef = eef_pose[:3, :3]
        att_k = kin.rotation_matrix_to_quaternion(R_eef)

        pos_error = ref_eef_positions[:, i] - pos_k
        att_error = quaternion_error(ref_eef_attitudes[:, i], att_k)
        cost += ca.sumsqr(pos_error) + ca.sumsqr(att_error)

        cost += ca.mtimes([pos_error.T, Q, pos_error]) + ca.mtimes([att_error.T, Q, att_error]) + ca.mtimes([u[:, i].T, R, u[:, i]])

    # TODO:
    # chekc dimesnions with Q and R
    # does it work with the att_erro like this
    # is teh rigght u taken when multiplying with R?


    opti.minimize(cost)

    sol = opti.solve()

    return sol.value(q), sol.value(u), sol.value(cost)

def run_mpc(DH_table):
    T_trajectory = 10           # Duration of the trajectory in seconds
    fps = 50                    # Frames per second for the trajectory
    dt = 1 / fps                # Time step for the trajectory / sampling
    M = T_trajectory * fps      # Simulation horizon
    N = 10                      # Prediction horizon

    n_joints = DH_table.shape[0] - 1    # Number of joints based on DH table

    # Kinematic model of Reach Robotics manipulator from DH parameters
    kin = Kinematics(DH_table)
    # Reference end effector trajectory
    q_traj = generate_trajectory(DH_table, T_trajectory, fps)                               # columnwise for each timestep
    ref_eef_positions, ref_eef_attitudes, all_links = compute_forward_kinematics(DH_table, q_traj)  # rows for each timestep

    # Ensure reference arrays always have N points by padding with the last value if needed
    if ref_eef_positions.shape[0] < M + N:
        pad_count = M + N - ref_eef_positions.shape[0]
        last_pos = ref_eef_positions[-1, :]
        last_att = ref_eef_attitudes[-1, :]
        ref_eef_positions = np.vstack([ref_eef_positions, np.tile(last_pos, (pad_count, 1))])
        ref_eef_attitudes = np.vstack([ref_eef_attitudes, np.tile(last_att, (pad_count, 1))])

    q_predict = np.empty((n_joints, N+1, M))        # open loop prediction of joint trajectory
    u_optimal = np.empty((n_joints, M))             # Preallocated control inputs
    cost = np.empty((M))                                # Cost function value
    q_real = np.empty((n_joints, M+1))              # Preallocated real joint trajectory
    q_real[:, 0] = q_traj[:, 0]                     # Initial joint state = first state of the trajectory

    for step in range(M):
        # print(steps)
        # For each step in the simulation horizon, we solve the optimization problem
        # and apply the first control input to the kinematic model
        # and repeat the optimization from there
        ref_pos_for_horizon = ref_eef_positions[step:step+N, :].T  # Reference positions for the prediction horizon, each column is a timestep with global eef x ,y, z
        ref_att_for_horizon = ref_eef_attitudes[step:step+N, :].T  # Reference attitudes for the prediction horizon, each column is a timestep with eef quaternion w, x, y, z
        
        q_opt, u_optimal_horizon, Jopt = solve_cftoc(kin, N, dt, n_joints, q_real[:, step], ref_pos_for_horizon, ref_att_for_horizon)
        
        q_predict[:, :, step] = q_opt  # Store the predicted joint trajectory for this step
        u_optimal[:,step] = u_optimal_horizon[:,0]
        cost[step] = Jopt  # Store the cost for this step

        q_real[:, step+1] = kinematic_model(u_optimal[:,step], dt, q_real[:, step])  # Update the real joint state using the first control input:

    # prediction error
    predErr = np.zeros((1, M - N + 1)) # prediction error for each timestep until the end of the simulation time is first reached
    for i in range(predErr.shape[1]):
        Error = q_real[:, i:i+N+1] - q_predict[:, :, i]
        predErr[0, i] = np.sum(np.linalg.norm(Error, axis=0))

    return q_real, predErr, cost, ref_eef_positions, ref_eef_attitudes, all_links



def main():
    DH_table = load_dh_parameters('alpha_kin_params.yaml')
    # q_traj = generate_trajectory(DH_table)
    # eef_positions, eef_attitudes, all_links = compute_forward_kinematics(DH_table, q_traj)
    # animate_trajectory(all_links)
    # Print all entries for the first link (index 0) across all timesteps
    # run_mpc_example(DH_table)
    q_real, predErr, cost, ref_eef_positions, ref_eef_attitudes, ref_all_links = run_mpc(DH_table)
    eef_positions, eef_attitudes, all_links = compute_forward_kinematics(DH_table, q_real)
    animate_trajectory(all_links)
    animate_trajectory(ref_all_links)

    # Compute tracking errors for position and attitude
    pos_error = np.linalg.norm(ref_eef_positions[:eef_positions.shape[0], :] - eef_positions, axis=1)

    def quaternion_error_np(q_goal, q_current):
        # q = [w, x, y, z]
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

    att_error = np.array([
        quaternion_error_np(ref_eef_attitudes[i], eef_attitudes[i])
        for i in range(min(len(ref_eef_attitudes), len(eef_attitudes)))
    ])

    # Plot tracking errors
    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axs[0].plot(pos_error, label='Position Tracking Error')
    axs[0].set_ylabel('Position Error [m]')
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(att_error, label='Attitude Tracking Error')
    axs[1].set_ylabel('Attitude Error')
    axs[1].set_xlabel('Timestep')
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()

    # eef_positions: actual end effector positions (timesteps, 3)
    # ref_eef_positions: reference end effector positions (timesteps, 3)

    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    labels = ['x', 'y', 'z']

    for i in range(3):
        axs[i].plot(ref_eef_positions[:, i], label='Reference')
        axs[i].plot(eef_positions[:, i], label='Actual')
        axs[i].set_ylabel(f'EEF {labels[i]} [m]')
        axs[i].legend()
        axs[i].grid(True)

    axs[2].set_xlabel('Timestep')
    plt.tight_layout()
    plt.show()

    # Plot prediction error
    plt.figure(figsize=(10, 4))
    plt.plot(predErr[0, :], label='Prediction Error')
    plt.xlabel('Timestep')
    plt.ylabel('Prediction Error')
    plt.title('Prediction Error over Time')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
