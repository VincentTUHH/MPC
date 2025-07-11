import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from matplotlib.lines import Line2D
from scipy.spatial.transform import Rotation as R

def animate_trajectory(all_links, fps=50):
    timesteps = all_links.shape[0]
    # Compute min/max for each axis, add margin
    margin = 0.05  # 5 cm margin
    x_min, x_max = np.min(all_links[:, :, 0]), np.max(all_links[:, :, 0])
    y_min, y_max = np.min(all_links[:, :, 1]), np.max(all_links[:, :, 1])
    z_min, z_max = np.min(all_links[:, :, 2]), np.max(all_links[:, :, 2])
    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([x_min - margin * x_range, x_max + margin * x_range])
    ax.set_ylim([y_min - margin * y_range, y_max + margin * y_range])
    ax.set_zlim([z_min - margin * z_range, z_max + margin * z_range])
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')

    arm_line, = ax.plot([], [], [], 'o-', lw=2, markersize=4, color='tab:blue')
    eef_point, = ax.plot([], [], [], 'ro', markersize=4, label='End Effector')

    def update(frame):
        links = all_links[frame]
        arm_line.set_data(links[:, 0], links[:, 1])
        arm_line.set_3d_properties(links[:, 2])
        eef_point.set_data([links[-1, 0]], [links[-1, 1]])
        eef_point.set_3d_properties([links[-1, 2]])
        return arm_line, eef_point

    ani = FuncAnimation(fig, update, frames=timesteps, interval=1000/fps, blit=True)
    plt.legend()
    plt.show()

def plot_tracking_errors(pos_error, att_error):
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

def plot_eef_positions(ref_eef_positions, eef_positions):
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



def plot_prediction_error(predErr):
    plt.figure(figsize=(10, 4))
    plt.plot(predErr[0, :], label='Prediction Error')
    plt.xlabel('Timestep')
    plt.ylabel('Prediction Error')
    plt.title('Prediction Error over Time')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

def plot_joint_angles(q_traj):
    n_joints, n_steps = q_traj.shape
    fig, axes = plt.subplots(n_joints, 1, figsize=(10, 2.5 * n_joints), sharex=True)
    t = np.arange(n_steps)
    for i in range(n_joints):
        axes[i].plot(t, q_traj[i, :])
        axes[i].set_ylabel(f'Joint {i+1} angle [rad]')
        axes[i].grid(True)
    axes[-1].set_xlabel('Timestep')
    plt.tight_layout()


def animate_bluerov(eta_all, dt, box_size=(0.4571, 0.575, 0.2539)):
    '''Animate the trajectory of a BlueROV vehicle in 3D space.
    Parameters:
    - eta_all: np.ndarray of shape (n_steps, 6) containing the state
      [x, y, z, phi, theta, psi] for each timestep.
    - dt: float, time step duration in seconds.
    - box_size: tuple of floats (length, width, height) representing the
      dimensions of the vehicle's bounding box.
    '''
    
    def plot_vehicle(ax, eta, box_size):
        pos = eta[:3]
        phi, theta, psi = eta[3:6]
        rot = R.from_euler('zyx', [psi, theta, phi]).as_matrix()
        l, w, h = box_size
        corners = np.array([
            [ l/2,  w/2, -h/2],
            [ l/2, -w/2, -h/2],
            [-l/2, -w/2, -h/2],
            [-l/2,  w/2, -h/2],
            [ l/2,  w/2,  h/2],
            [ l/2, -w/2,  h/2],
            [-l/2, -w/2,  h/2],
            [-l/2,  w/2,  h/2]
        ])
        corners_world = (rot @ corners.T).T + pos
        box_lines = [
            [0,1],[1,2],[2,3],[3,0],
            [4,5],[5,6],[6,7],[7,4],
            [0,4],[1,5],[2,6],[3,7]
        ]
        for i,j in box_lines:
            ax.plot(*zip(corners_world[i], corners_world[j]), color='b')
        ax.scatter(*pos, color='r', s=40, label='CoG')
        axis_len = 0.3
        origin = pos
        x_axis = origin + rot @ np.array([axis_len, 0, 0])
        y_axis = origin + rot @ np.array([0, axis_len, 0])
        z_axis = origin + rot @ np.array([0, 0, axis_len])
        ax.plot(*zip(origin, x_axis), color='r')
        ax.plot(*zip(origin, y_axis), color='g')
        ax.plot(*zip(origin, z_axis), color='b')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-2, 1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    def update(frame):
        ax.cla()
        ax.set_xlim([-2, 2])
        ax.set_ylim([-2, 2])
        ax.set_zlim([-2, 1])
        legend_elements = [
            Line2D([0], [0], color='r', lw=2, label='X axis'),
            Line2D([0], [0], color='g', lw=2, label='Y axis'),
            Line2D([0], [0], color='b', lw=2, label='Z axis'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='r', markersize=8, label='CoG')
        ]
        ax.legend(handles=legend_elements, loc='upper left')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plot_vehicle(ax, eta_all[frame], box_size=box_size)
        ax.set_title(f"Step {frame} (t={frame*dt:.2f}s)")

    ani = FuncAnimation(fig, update, frames=eta_all.shape[0], interval=dt*1000)
    plt.show()
    

def plot_vehicle_pos_vs_reference(ref_positions, real_positions):
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    labels = ['x', 'y', 'z']
    for i in range(3):
        axs[i].plot(ref_positions[:, i], label='Reference')
        axs[i].plot(real_positions[:, i], label='Actual')
        axs[i].set_ylabel(f'EEF {labels[i]} [m]')
        axs[i].legend()
        axs[i].grid(True)
    axs[2].set_xlabel('Timestep')
    plt.tight_layout()

def plot_vehicle_pos_vs_reference_time(ref_positions, real_positions, dt):
    # Plot real and reference positions x, y, z over time in separate subplots
    
    n = real_positions.shape[0]
    time = np.arange(n) * dt
    
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axs[0].plot(time, real_positions[:n, 0], label='x (real)')
    axs[0].plot(time, ref_positions[:n, 0], '--', label='x (ref)')
    axs[0].set_ylabel('x [m]')
    axs[0].legend()
    axs[1].plot(time, real_positions[:n, 1], label='y (real)')
    axs[1].plot(time, ref_positions[:n, 1], '--', label='y (ref)')
    axs[1].set_ylabel('y [m]')
    axs[1].legend()
    axs[2].plot(time, real_positions[:n, 2], label='z (real)')
    axs[2].plot(time, ref_positions[:n, 2], '--', label='z (ref)')
    axs[2].set_ylabel('z [m]')
    axs[2].set_xlabel('Time [s]')
    axs[2].legend()
    fig.suptitle('Real vs Reference Vehicle Positions Over Time')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def plot_delta_u(u_optimal, dt):
    delta_u = np.diff(u_optimal, axis=1)
    plt.figure()
    for i in range(delta_u.shape[0]):
        plt.subplot(delta_u.shape[0], 1, i + 1)
        plt.plot(np.arange(delta_u.shape[1]) * dt, delta_u[i, :])
        plt.ylabel(f'Δu[{i}]')
        if i == 0:
            plt.title('Change of Control Inputs (delta u) over Time')
        if i == delta_u.shape[0] - 1:
            plt.xlabel('Time [s]')
        plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_mpc_cost(cost, dt):
    plt.figure()
    plt.plot(np.arange(cost.shape[0]) * dt, cost)
    plt.xlabel('Time [s]')
    plt.ylabel('MPC Cost')
    plt.title('MPC Cost over Time')
    plt.grid(True)
    plt.show()

def plot_vehicle_euler_angles_vs_reference_time(reference_eta, real_eta, dt):
    # Plot real and reference positions x, y, z over time in separate subplots
    n = real_eta.shape[0]
    time = np.arange(n) * dt

    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axs[0].plot(time, real_eta[:n, 3], label='roll (real)')
    axs[0].plot(time, reference_eta[:n, 3], '--', label='roll (ref)')
    axs[0].set_ylabel('phi [m]')
    axs[0].legend()
    axs[1].plot(time, real_eta[:n, 4], label='pitch (real)')
    axs[1].plot(time, reference_eta[:n, 4], '--', label='pitch (ref)')
    axs[1].set_ylabel('theta [m]')
    axs[1].legend()
    axs[2].plot(time, real_eta[:n, 5], label='yaw (real)')
    axs[2].plot(time, reference_eta[:n, 5], '--', label='yaw (ref)')
    axs[2].set_ylabel('psi [m]')
    axs[2].set_xlabel('Time [s]')
    axs[2].legend()
    fig.suptitle('Real vs Reference Vehicle Attitude Over Time')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_velocities(real_nu, dt):
    plt.figure(figsize=(10, 8))
    vel_labels = ['u (surge)', 'v (sway)', 'w (heave)', 'p (roll rate)', 'q (pitch rate)', 'r (yaw rate)']
    for i in range(6):
        plt.subplot(6, 1, i + 1)
        plt.plot(np.arange(real_nu.shape[0]) * dt, real_nu[:, i])
        plt.ylabel(vel_labels[i])
        if i == 0:
            plt.title('Linear and Angular Velocities over Time')
        if i == 5:
            plt.xlabel('Time [s]')
        plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_control_inputs(u_optimal, dt):
    plt.figure()
    for i in range(u_optimal.shape[0]):
        plt.subplot(u_optimal.shape[0], 1, i + 1)
        plt.plot(np.arange(u_optimal.shape[1]) * dt, u_optimal[i, :])
        plt.ylabel(f'u[{i}]')
        if i == 0:
            plt.title('Control Inputs (u) over Time')
        if i == u_optimal.shape[0] - 1:
            plt.xlabel('Time [s]')
        plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_jacobian_condition_number(jacobian_array, dt):
    plt.figure()
    plt.plot(np.arange(jacobian_array.shape[0]) * dt, jacobian_array)
    plt.xlabel('Time [s]')
    plt.ylabel('Jacobian Condition Number')
    plt.title('Jacobian Condition Number over Time')
    plt.grid(True)
    plt.show()
