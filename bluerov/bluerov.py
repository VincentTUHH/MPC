import numpy as np

from bluerov import dynamics
from common import utils_math
from common.animate import animate_bluerov
from common.my_package_path import get_package_path

'''
Simulate BlueROV using the dynamics model.
'''

# Hinweis: Das dynamische Modell für den modellbasierten Controller ist in `uvms/bluerov_ctrl/src/twist_model_based_interface.cpp` definiert. 
# Dieses Modell wird nur verwendet, wenn nicht der einfache PID-Controller eingesetzt wird. 
# Die Modellparameter wie die Massenmatrix usw. sind in `uvms/bluerov_ctrl/config/model_params.yaml` festgelegt. 
# Diese dynamischen Modellparameter werden ebenfalls im EKF zur Schätzung der BlueROV-Odometrie in `uvms/bluerov_estimation/src/kf.cpp` verwendet.

def generate_circle_trajectory(
    radius=1.0,
    center=None,
    num_points=500
):
    """
    Generates a circular trajectory in the XY plane at a fixed Z height,
    with the vehicle's x-axis tangential to the circle.

    Returns:
        traj: np.array, shape (num_points, 6), each row is [x, y, z, phi, theta, psi]
    """
    if center is None:
        center = np.array([0.0, 0.0, -1.0])
    theta = np.linspace(-np.pi, np.pi, num_points, endpoint=False)
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)
    z = np.full_like(x, center[2])
    phi = np.zeros_like(x)
    theta_angle = np.zeros_like(x)
    psi = theta + np.pi / 2  # Heading tangent to the path
    traj = np.stack([x, y, z, phi, theta_angle, psi], axis=1)
    return traj

def generate_circle_trajectory_time(
    radius=1.0,
    center=None,
    dt=0.01,
    T=10.0,
    n=1
):
    """
    Generates a circular trajectory in the XY plane at a fixed Z height,
    parameterized by time, with the vehicle's x-axis tangential to the circle.
    """
    if center is None:
        center = np.array([0.0, 0.0, -1.0])
    num_steps_per_circle = int(np.round(T / dt))
    t = np.tile(np.linspace(0, T, num_steps_per_circle, endpoint=False), n)
    theta = 2 * np.pi * (t / T)
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)
    z = np.full_like(x, center[2])
    phi = np.zeros_like(x)
    theta_angle = np.zeros_like(x)
    psi = theta + np.pi / 2
    traj = np.stack([x, y, z, phi, theta_angle, psi], axis=1)
    return traj

def generate_sine_on_circle_trajectory_time(
    radius=1.0,
    center=None,
    dt=0.01,
    T=10.0,
    sine_amplitude=0.2,
    sine_phase=0.0,
    n=3
):
    """
    Generates a seamless time-dependent trajectory:
    - XY is a circle with period T,
    - Z is a sine with period T,
    - Heading (psi) is tangent to the path,
    - Pitch (theta) follows the slope of the path,
    - Roll (phi) is always zero.
    """
    if center is None:
        center = np.array([0.0, 0.0, -1.0])
    num_steps_per_circle = int(np.round(T / dt))
    t = np.tile(np.linspace(0, T, num_steps_per_circle, endpoint=False), n)
    theta_circ = 2 * np.pi * (t / T)
    x = center[0] + radius * np.cos(theta_circ)
    y = center[1] + radius * np.sin(theta_circ)
    z = center[2] + sine_amplitude * np.sin(theta_circ + sine_phase)
    phi = np.zeros_like(x)
    dtheta_dt = 2 * np.pi / T
    dx = -radius * np.sin(theta_circ) * dtheta_dt
    dy =  radius * np.cos(theta_circ) * dtheta_dt
    dz = sine_amplitude * np.cos(theta_circ + sine_phase) * dtheta_dt
    tangent = np.stack([dx, dy, dz], axis=1)
    tangent_unit = tangent / np.linalg.norm(tangent, axis=1, keepdims=True)
    psi = np.arctan2(tangent_unit[:, 1], tangent_unit[:, 0])
    theta_angle = np.arcsin(tangent_unit[:, 2])
    traj = np.stack([x, y, z, phi, theta_angle, psi], axis=1)
    return traj

def generate_linear_trajectory(
    start,
    end,
    T,
    dt
):
    """
    Generates a straight-line trajectory from start to end with constant speed.
    Orientation is chosen such that the x-axis points in the direction of the trajectory.
    """
    num_steps = int(np.round(T / dt)) + 1
    positions = np.linspace(start, end, num_steps)
    direction = end - start
    direction_norm = np.linalg.norm(direction)
    if direction_norm < 1e-8:
        phi = theta = psi = 0.0
    else:
        dir_unit = direction / direction_norm
        psi = np.arctan2(dir_unit[1], dir_unit[0])
        theta = np.arctan2(-dir_unit[2], np.hypot(dir_unit[0], dir_unit[1]))
        phi = 0.0
    phi_arr = np.full((num_steps,), phi)
    theta_arr = np.full((num_steps,), theta)
    psi_arr = np.full((num_steps,), psi)
    traj = np.hstack([positions, phi_arr[:, None], theta_arr[:, None], psi_arr[:, None]])
    return traj

def generate_slow_sine_curve_trajectory(
    start,
    end,
    T,
    dt,
    amplitude=0.2,
    wavelength=5.0,
    vertical_amplitude=0.2
):
    """
    Generates a slow, smooth 3D sine-curve trajectory from start to end.
    The path oscillates laterally and vertically as it moves from start to end.
    """
    num_steps = int(np.round(T / dt)) + 1
    positions = np.linspace(start, end, num_steps)
    direction = end - start
    direction_norm = np.linalg.norm(direction)
    if direction_norm < 1e-8:
        traj = np.hstack([positions, np.zeros((num_steps, 3))])
        return traj
    dir_unit = direction / direction_norm
    if np.allclose(np.abs(dir_unit), [0, 0, 1]):
        perp = np.array([1, 0, 0])
    else:
        perp = np.cross(dir_unit, [0, 0, 1])
        perp /= np.linalg.norm(perp)
    perp_vert = np.cross(dir_unit, perp)
    perp_vert /= np.linalg.norm(perp_vert)
    s = np.linspace(0, direction_norm, num_steps)
    lateral_offset = amplitude * np.sin(2 * np.pi * s / wavelength)
    vertical_offset = vertical_amplitude * np.sin(2 * np.pi * s / (wavelength * 1.5) + np.pi / 3)
    pos = positions + np.outer(lateral_offset, perp) + np.outer(vertical_offset, perp_vert)
    pos_diff = np.gradient(pos, axis=0)
    tangent = pos_diff / np.linalg.norm(pos_diff, axis=1, keepdims=True)
    psi = np.arctan2(tangent[:, 1], tangent[:, 0])
    theta = np.arcsin(tangent[:, 2])
    phi = np.zeros(num_steps)
    traj = np.hstack([pos, phi[:, None], theta[:, None], psi[:, None]])
    return traj

def simulate_bluerov_dynamics(
    dyn,
    eta_0,
    nu_0,
    tau,
    dt,
    timesteps
):
    """
    Simulate BlueROV dynamics given initial state and force/moment input sequence.
    Returns arrays of poses and velocities.
    """
    eta_all = np.zeros((timesteps + 1, 6))
    nu_all = np.zeros((timesteps + 1, 6))
    eta_all[0, :] = eta_0
    nu_all[0, :] = nu_0
    for t in range(timesteps):
        nu_all[t+1, :], eta_all[t+1, :], _ = dyn.forward_dynamics(
            tau[t, :], nu_all[t, :], eta_all[t, :], dt
        )
    return eta_all, nu_all

def simulate_bluerov_dynamics_esc(
    dyn,
    eta_0,
    nu_0,
    u_esc,
    dt,
    timesteps,
    V_bat=16.0
):
    """
    Simulate BlueROV dynamics using ESC (PWM) input.
    Returns arrays of poses, velocities, and thruster forces.
    """
    eta_all = np.zeros((timesteps + 1, 6))
    nu_all = np.zeros((timesteps + 1, 6))
    tau_all = np.zeros((timesteps, 6))
    eta_all[0, :] = eta_0
    nu_all[0, :] = nu_0
    for t in range(timesteps):
        nu_all[t+1, :], eta_all[t+1, :], _, tau_all[t, :] = dyn.forward_dynamics_esc(
            u_esc, nu_all[t, :], eta_all[t, :], dt, V_bat
        )
    return eta_all, nu_all, tau_all

def main():
    bluerov_package_path = get_package_path('bluerov')
    model_params_path = bluerov_package_path + "/config/model_params.yaml"
    bluerov_params = utils_math.load_model_params(model_params_path)
    dyn = dynamics.BlueROVDynamics(bluerov_params)

    eta_0 = np.array([0, 0, -1.0, 0.0, 0.0, 0])
    nu_0 = np.zeros(6)
    dt = 0.1
    timesteps = 100

    # Example: simulate with time-varying force input
    t_vec = np.linspace(0, timesteps * dt, timesteps)
    tau = np.zeros((timesteps, 6))
    amplitude = 3.0
    tau[:, 0] = amplitude * np.sin(2 * np.pi * 0.05 * t_vec)
    tau[:, 1] = amplitude * np.cos(2 * np.pi * 0.03 * t_vec)
    tau[:, 2] = amplitude * np.sin(2 * np.pi * 0.02 * t_vec + np.pi/4)
    eta_all, nu_all = simulate_bluerov_dynamics(dyn, eta_0, nu_0, tau, dt, timesteps)
    animate_bluerov(eta_all, dt=dt)

    # Example: animate a circle trajectory
    traj = generate_circle_trajectory()
    animate_bluerov(traj, dt=dt)

    # Example: animate a sine-on-circle trajectory
    traj = generate_sine_on_circle_trajectory_time(dt=dt, T=10.0)
    animate_bluerov(traj, dt=dt)

    # Example: simulate with constant ESC input
    u_esc = np.array([0.01, 0.01, 0.01, 0.01, 0.0, 0.0, 0.0, 0.0])
    u_esc = np.clip(u_esc, -1.0, 1.0)
    eta_all, nu_all, tau_all = simulate_bluerov_dynamics_esc(
        dyn, eta_0, nu_0, u_esc, dt, timesteps, V_bat=16.0
    )
    animate_bluerov(eta_all, dt=dt)

if __name__ == "__main__":
    main()
