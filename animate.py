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