import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

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
    plt.show()

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
    plt.show()

def plot_prediction_error(predErr):
    plt.figure(figsize=(10, 4))
    plt.plot(predErr[0, :], label='Prediction Error')
    plt.xlabel('Timestep')
    plt.ylabel('Prediction Error')
    plt.title('Prediction Error over Time')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()