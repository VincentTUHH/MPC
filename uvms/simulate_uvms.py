import uvms.model as model
import common.animate as animate
import numpy as np

def main():
     # Example usage for animate_trajectory
    n_steps = 1
    n_links = 6
    # Simulate a manipulator moving in a circle in 3D
    t = np.linspace(0, 2 * np.pi, n_steps)
    all_links = np.zeros((n_steps, n_links, 3))
    for i in range(n_links):
        radius = 0.2 + 0.05 * i
        all_links[:, i, 0] = radius * np.cos(t + i * 0.2)
        all_links[:, i, 1] = radius * np.sin(t + i * 0.2)
        all_links[:, i, 2] = 0.1 * i + 0.05 * np.sin(t + i * 0.3)
    animate.animate_trajectory(all_links, fps=30)

    # Example usage for animate_bluerov
    eta_all = np.zeros((n_steps, 6))
    eta_all[:, 0] = np.linspace(0, 1, n_steps)  # x
    eta_all[:, 1] = np.sin(t)                   # y
    eta_all[:, 2] = 0.5 * np.cos(t)             # z
    eta_all[:, 3] = 0.1 * np.sin(t)             # phi (roll)
    eta_all[:, 4] = 0.1 * np.cos(t)             # theta (pitch)
    eta_all[:, 5] = t / 2                       # psi (yaw)
    animate.animate_bluerov(eta_all, dt=0.05)

    # Example usage for animate_uvms
    # Use previous eta_all and all_links

    print(eta_all)
    print(all_links)
    animate.animate_uvms(eta_all, all_links, dt=0.05)
    return

if __name__ == "__main__":
    main()