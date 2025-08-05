import numpy as np
import yaml
from common.animate import animate_bluerov
from scipy.spatial.transform import Rotation as Rot
from common.my_package_path import get_package_path
import common.utils_math as utils_math

# Hinweis: Das dynamische Modell für den modellbasierten Controller ist in `uvms/bluerov_ctrl/src/twist_model_based_interface.cpp` definiert. 
# Dieses Modell wird nur verwendet, wenn nicht der einfache PID-Controller eingesetzt wird. 
# Die Modellparameter wie die Massenmatrix usw. sind in `uvms/bluerov_ctrl/config/model_params.yaml` festgelegt. 
# Diese dynamischen Modellparameter werden ebenfalls im EKF zur Schätzung der BlueROV-Odometrie in `uvms/bluerov_estimation/src/kf.cpp` verwendet.

class BlueROVDynamics:
    def __init__(self, params):
        self.mass = params['mass']
        self.inertia = np.array(params['inertia'])
        self.cog = np.array(params['cog'])
        self.added_mass = np.array(params['added_mass'])
        self.buoyancy = params['buoyancy']
        self.cob = np.array(params['cob'])
        self.damping_linear = np.array(params['damping_linear'])
        self.damping_nonlinear = np.array(params['damping_nonlinear'])
        self.gravity = 9.81
        self.L = 2.5166  # scaling factor for PWM to thrust conversion

        self.M = self.compute_mass_matrix()
        self.M_inv = np.linalg.pinv(self.M)

        self.mixer = self.get_mixer_matrix_wu2018()
        self.mixer_inv = np.linalg.pinv(self.mixer)

    def get_mixer_matrix_niklas(self):
        alpha_f = 0.733
        alpha_r = 0.8378
        l_hf = 0.163
        l_hr = 0.177
        l_vx = 0.12
        l_vy = 0.218

        calpha_f = np.cos(alpha_f)
        salpha_f = np.sin(alpha_f)
        calpha_r = np.cos(alpha_r)
        salpha_r = np.sin(alpha_r)

        mixer = np.array([
            [calpha_f, calpha_f, calpha_r, calpha_r, 0, 0, 0, 0],
            [salpha_f, -salpha_f, -salpha_r, salpha_r, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, -1, -1, 1],
            [0, 0, 0, 0, -l_vy, -l_vy, l_vy, l_vy],
            [0, 0, 0, 0, -l_vx, l_vx, -l_vx, l_vx],
            [l_hf, -l_hf, l_hr, -l_hr, 0, 0, 0, 0]
        ])
        return mixer

    def get_mixer_matrix_wu2018(self):
        alpha_f = np.pi / 4
        alpha_r = np.pi / 4
        n1 = np.array([np.cos(alpha_f), np.sin(alpha_f), 0])
        n2 = np.array([np.cos(alpha_f), -np.sin(alpha_f), 0])
        n3 = np.array([np.cos(alpha_r), -np.sin(alpha_r), 0])
        n4 = np.array([np.cos(alpha_r), np.sin(alpha_r), 0])
        n5 = np.array([0, 0, 1])
        n6 = np.array([0, 0, -1])
        n7 = np.array([0, 0, -1])
        n8 = np.array([0, 0, 1])
        l_xt = 0.156
        l_yt = 0.111
        l_zt = 0.0
        l_xb = 0.12
        l_yb = 0.218
        l_zb = 0.085
        l1 = np.array([l_xt, -l_yt, l_zt])
        l2 = np.array([l_xt, l_yt, l_zt])
        l3 = np.array([-l_xt, -l_yt, l_zt])
        l4 = np.array([-l_xt, l_yt, l_zt])
        l5 = np.array([l_xb, -l_yb, -l_zb])
        l6 = np.array([l_xb, l_yb, -l_zb])
        l7 = np.array([-l_xb, -l_yb, -l_zb])
        l8 = np.array([-l_xb, l_yb, -l_zb])

        n_arr = np.stack([n1, n2, n3, n4, n5, n6, n7, n8], axis=1)
        l_arr = np.stack([l1, l2, l3, l4, l5, l6, l7, l8], axis=1)
        cross_arr = np.cross(l_arr.T, n_arr.T).T
        mixer = np.vstack((n_arr, cross_arr))
        mixer[np.abs(mixer) < 1e-6] = 0.0
        return mixer

    def compute_mass_matrix(self):
        M_rb = np.zeros((6, 6))
        M_rb[0:3, 0:3] = self.mass * np.eye(3)
        M_rb[0:3, 3:6] = -self.mass * utils_math.skew(self.cog)
        M_rb[3:6, 0:3] = self.mass * utils_math.skew(self.cog)
        M_rb[3:6, 3:6] = np.diag(self.inertia) - self.mass * utils_math.skew(self.cog) @ utils_math.skew(self.cog)
        M = M_rb + np.diag(self.added_mass)
        return M

    def D(self, nu):
        D = np.zeros((6, 6))
        D[np.diag_indices(6)] = self.damping_linear + self.damping_nonlinear * np.abs(nu)
        return D

    def C(self, nu):
        C = np.zeros((6, 6))
        v = nu[0:3]
        w = nu[3:6]
        mv = self.M[0:3, 0:3] @ v + self.M[0:3, 3:6] @ w
        mw = self.M[3:6, 0:3] @ v + self.M[3:6, 3:6] @ w
        C[0:3, 3:6] = -utils_math.skew(mv)
        C[3:6, 0:3] = -utils_math.skew(mv)
        C[3:6, 3:6] = -utils_math.skew(mw)
        return C

    def g(self, eta):
        phi, theta, psi = eta[3:]
        R = utils_math.rotation_matrix_from_euler(phi, theta, psi)
        fg = self.mass * R.T @ np.array([0, 0, -self.gravity])
        fb = self.buoyancy * R.T @ np.array([0, 0, self.gravity])
        g_vec = np.zeros(6)
        g_vec[0:3] = -(fg + fb)
        g_vec[3:6] = -(np.cross(self.cog, fg) + np.cross(self.cob, fb))
        return g_vec
    
    def g_quat(self, eta):
        """
        Gravity and buoyancy vector for quaternion-based pose.
        eta: [x, y, z, qx, qy, qz, qw]
        Returns: g_vec (6,)
        """
        q = eta[3:]
        R = utils_math.rotation_matrix_from_quat(q)
        fg = self.mass * R.T @ np.array([0, 0, -self.gravity])
        fb = self.buoyancy * R.T @ np.array([0, 0, self.gravity])
        g_vec = np.zeros(6)
        g_vec[0:3] = -(fg + fb)
        g_vec[3:6] = -(np.cross(self.cog, fg) + np.cross(self.cob, fb))
        return g_vec

    def J(self, eta):
        phi, theta, psi = eta[3], eta[4], eta[5]
        R = utils_math.rotation_matrix_from_euler(phi, theta, psi)
        cphi = np.cos(phi)
        sphi = np.sin(phi)
        ctheta = np.cos(theta)
        stheta = np.sin(theta)
        T = np.array([
            [1, sphi * stheta / ctheta, cphi * stheta / ctheta],
            [0, cphi, -sphi],
            [0, sphi / ctheta, cphi / ctheta]
        ])
        J = np.zeros((6, 6))
        J[0:3, 0:3] = R
        J[3:6, 3:6] = T
        return J

    def J_quat(self, eta):
        """
        Jacobian for quaternion-based pose representation.
        eta: [x, y, z, qx, qy, qz, qw]
        Returns: J (6,6) mapping body velocities to pose derivatives.
        """
        q = eta[3:7]
        R = utils_math.rotation_matrix_from_quat(q)
        # Quaternion kinematics matrix
        qx, qy, qz, qw = q
        G = 0.5 * np.array([
            [-qx, -qy, -qz],
            [ qw, -qz,  qy],
            [ qz,  qw, -qx],
            [-qy,  qx,  qw]
        ])
        J = np.zeros((7, 6))
        J[0:3, 0:3] = R
        J[3:7, 3:6] = G
        return J

    def tau_to_thrusts(self, tau):
        return self.mixer_inv @ tau

    def thrusts_to_pwm(self, thrusts, thrust_max=29.0, pwm_min=1100, pwm_max=1900):
        pwm = 1500 + 400 * np.clip(thrusts / thrust_max, -1.0, 1.0)
        pwm = np.clip(pwm, pwm_min, pwm_max)
        return pwm

def rigid_body_jacobian_euler(eta):
    """
    Returns the Jacobian matrix J(eta) that maps body-frame velocities to inertial-frame pose derivatives
    for a 6-DOF underwater vehicle using Euler angles (phi, theta, psi).

    Args:
        eta: np.array, pose vector [x, y, z, phi, theta, psi] (6,)

    Returns:
        J: np.array, Jacobian matrix (6,6)
    """
    phi, theta, psi = eta[3:6]
    cphi = np.cos(phi)
    sphi = np.sin(phi)
    ctheta = np.cos(theta)
    stheta = np.sin(theta)
    ttheta = np.tan(theta)
    cpsi = np.cos(psi)
    spsi = np.sin(psi)

    R = utils_math.rotation_matrix_from_euler(phi, theta, psi)

    # Transformation from angular velocity in body to Euler angle rates
    T = np.array([
        [1, sphi * ttheta, cphi * ttheta],
        [0, cphi,         -sphi],
        [0, sphi / ctheta, cphi / ctheta]
    ])

    J = np.zeros((6, 6))
    J[0:3, 0:3] = R
    J[3:6, 3:6] = T
    return J

def forward_dynamics(dynamics, tau, nu, eta, dt):
    # solve for nu_dot using the equation of motion:
    nu_dot = np.linalg.inv(dynamics.M) @ (tau - dynamics.C(nu) @ nu - dynamics.D(nu) @ nu - dynamics.g(eta))
    # nu_dot = np.linalg.inv(dynamics.M) @ (tau - dynamics.C(nu) @ nu - dynamics.D(nu) @ nu)

    # integrate using Euler forward method to get new velocity
    nu = nu + dt * nu_dot

    # integrate to get new pose
    eta = eta + dt * rigid_body_jacobian_euler(eta) @ nu

    return nu, eta, nu_dot


def forward_dynamics_esc(dynamics, u_esc, nu, eta, dt, V_bat=16.0):
    # solve for nu_dot using the equation of motion:
    tau = dynamics.mixer @ u_esc * dynamics.L * V_bat  # convert ESC signals to thrusts
    nu_dot = np.linalg.inv(dynamics.M) @ (dynamics.mixer @ u_esc * dynamics.L * V_bat - dynamics.C(nu) @ nu - dynamics.D(nu) @ nu - dynamics.g(eta))

    # integrate using Euler forward method to get new velocity
    nu = nu + dt * nu_dot

    # integrate to get new pose
    eta = eta + dt * rigid_body_jacobian_euler(eta) @ nu
    # eta[3] = normalize_angle(eta[3])  # phi
    # eta[4] = np.clip(eta[4], -np.pi/2 + 1e-6, np.pi/2 - 1e-6)  # theta, falls nötig
    # eta[5] = normalize_angle(eta[5])  # psi

    return nu, eta, nu_dot, tau

def forward_dynamics_esc_with_disturbance(
    dynamics, u_esc, nu, eta, dt, V_bat=16.0, disturbance_prob=0.1, force_max=40.0, torque_max=40.0
):
    """
    Like forward_dynamics_esc, but randomly adds external disturbance tau_dist.
    With probability disturbance_prob, tau_dist is a random 6D vector:
    - first 3: force in [-force_max, force_max] N
    - last 3: torque in [-torque_max, torque_max] Nm
    """
    tau = dynamics.mixer @ u_esc * dynamics.L * V_bat

    # Randomly decide whether to add disturbance
    if np.random.rand() < disturbance_prob:
        tau_dist = np.zeros(6)
        tau_dist[:3] = np.random.uniform(-force_max, force_max, 3)
        tau_dist[3:] = np.random.uniform(-torque_max, torque_max, 3)
    else:
        tau_dist = np.zeros(6)

    tau_total = tau + tau_dist

    nu_dot = np.linalg.inv(dynamics.M) @ (
        tau_total - dynamics.C(nu) @ nu - dynamics.D(nu) @ nu - dynamics.g(eta)
    )
    nu = nu + dt * nu_dot
    eta = eta + dt * rigid_body_jacobian_euler(eta) @ nu

    # eta[3] = normalize_angle(eta[3])  # phi
    # eta[4] = np.clip(eta[4], -np.pi/2 + 1e-6, np.pi/2 - 1e-6)  # theta, falls nötig
    # eta[5] = normalize_angle(eta[5])  # psi

    return nu, eta, nu_dot, tau_total

def normalize_angle(angle):
    """Normiert einen Winkel auf [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def inverse_dynamics(dynamics, nu_dot, nu, eta):
    tau = dynamics.M @ nu_dot + dynamics.C(nu) @ nu + dynamics.D(nu) @ nu + dynamics.g(eta)
    return tau

def generate_circle_trajectory(radius=1.0, center=np.array([0.0, 0.0, -1.0]), num_points=500):
    """
    Generates a circular trajectory in the XY plane at a fixed Z height,
    with the vehicle's x-axis tangential to the circle (i.e., heading tangent to the path).

    Args:
        radius: float, radius of the circle (meters)
        center: np.array, center of the circle [x, y, z]
        num_points: int, number of trajectory points

    Returns:
        traj: np.array, shape (num_points, 6), each row is [x, y, z, phi, theta, psi]
    """
    theta = np.linspace(-np.pi, np.pi, num_points, endpoint=False)
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)
    z = np.full_like(x, center[2])
    phi = np.zeros_like(x)
    theta_angle = np.zeros_like(x)
    # Tangent direction: psi = theta + pi/2 (for CCW circle, x-axis tangent)
    psi = theta + np.pi / 2

    traj = np.stack([x, y, z, phi, theta_angle, psi], axis=1)
    return traj

def generate_circle_trajectory_time(
    radius=1.0,
    center=np.array([0.0, 0.0, -1.0]),
    dt=0.01,
    T=10.0,
    n=1
):
    """
    Generates a circular trajectory in the XY plane at a fixed Z height,
    parameterized by time, with the vehicle's x-axis tangential to the circle.

    Args:
        radius: float, radius of the circle (meters)
        center: np.array, center of the circle [x, y, z]
        dt: float, time step (seconds)
        T: float, period of one full circle (seconds)
        n: int, number of circles to perform

    Returns:
        traj: np.array, shape (num_steps, 6), each row is [x, y, z, phi, theta, psi]
    """
    num_steps_per_circle = int(np.round(T / dt))
    t = np.tile(np.linspace(0, T, num_steps_per_circle, endpoint=False), n)
    theta = 2 * np.pi * (t / T)
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)
    z = np.full_like(x, center[2])
    phi = np.zeros_like(x)
    theta_angle = np.zeros_like(x)
    psi = theta + np.pi / 2  # Heading tangent to the path

    traj = np.stack([x, y, z, phi, theta_angle, psi], axis=1)
    return traj


def generate_sine_on_circle_trajectory_time(
    radius=1.0,
    center=np.array([0.0, 0.0, -1.0]),
    dt=0.01,
    T=10.0,
    sine_amplitude=0.2,
    sine_phase=0.0,
    n=3
):
    """
    Generates a seamless time-dependent trajectory: 
    - XY is a circle with period T,
    - Z is a sine with period T (so one up/down per circle),
    - Heading (psi) is tangent to the path,
    - Pitch (theta) follows the slope of the path,
    - Roll (phi) is always zero (no roll).

    Args:
        radius: float, radius of the circle (meters)
        center: np.array, center of the circle [x, y, z]
        dt: float, time step (seconds)
        T: float, period of one full circle (seconds)
        sine_amplitude: float, amplitude of the sine wave in Z (meters)
        sine_phase: float, phase offset for the sine wave (radians)
        n: int, number of circles to perform

    Returns:
        traj: np.array, shape (num_steps, 6), each row is [x, y, z, phi, theta, psi]
    """
    num_steps_per_circle = int(np.round(T / dt))
    t = np.tile(np.linspace(0, T, num_steps_per_circle, endpoint=False), n)
    theta_circ = 2 * np.pi * (t / T)  # angle around the circle, period T

    # XY circle
    x = center[0] + radius * np.cos(theta_circ)
    y = center[1] + radius * np.sin(theta_circ)
    # Z sine, period T
    z = center[2] + sine_amplitude * np.sin(theta_circ + sine_phase)
    phi = np.zeros_like(x)  # no roll

    # Derivatives for tangent vector
    dtheta_dt = 2 * np.pi / T
    dx = -radius * np.sin(theta_circ) * dtheta_dt
    dy =  radius * np.cos(theta_circ) * dtheta_dt
    dz = sine_amplitude * np.cos(theta_circ + sine_phase) * dtheta_dt

    tangent = np.stack([dx, dy, dz], axis=1)
    tangent_norm = np.linalg.norm(tangent, axis=1, keepdims=True)
    tangent_unit = tangent / tangent_norm

    # Heading: tangent in XY plane
    psi = np.arctan2(tangent_unit[:, 1], tangent_unit[:, 0])
    # Pitch: angle of tangent vector with respect to horizontal
    theta_angle = np.arcsin(tangent_unit[:, 2])
    # Roll: always zero

    traj = np.stack([x, y, z, phi, theta_angle, psi], axis=1)
    return traj


def generate_linear_trajectory(start, end, T, dt):
    """
    Generates a straight-line trajectory from start to end with constant speed,
    covering the distance in time T. The orientation is chosen such that the x-axis
    points in the direction of the trajectory and the y-axis is parallel to the horizontal plane.

    Args:
        start: np.array, shape (3,), start position [x, y, z]
        end: np.array, shape (3,), end position [x, y, z]
        T: float, total time to traverse (seconds)
        dt: float, time step (seconds)

    Returns:
        traj: np.array, shape (num_steps, 6), each row is [x, y, z, phi, theta, psi]
    """
    num_steps = int(np.round(T / dt)) + 1
    positions = np.linspace(start, end, num_steps)

    # Direction vector
    direction = end - start
    direction_norm = np.linalg.norm(direction)
    if direction_norm < 1e-8:
        # No movement, keep orientation zero
        phi = theta = psi = 0.0
    else:
        # Normalize direction
        dir_unit = direction / direction_norm
        # Yaw (psi): angle in XY plane
        psi = np.arctan2(dir_unit[1], dir_unit[0])
        # Pitch (theta): angle up/down from horizontal
        theta = np.arctan2(-dir_unit[2], np.hypot(dir_unit[0], dir_unit[1]))
        # Roll (phi): keep zero to keep y-axis horizontal
        phi = 0.0

    # Repeat orientation for all steps
    # orientation = np.tile([phi, theta, psi], (num_steps, 1))
    # traj = np.hstack([positions, orientation])

    # Create arrays for orientation with the same length as positions
    phi_arr = np.full((num_steps,), phi)
    theta_arr = np.full((num_steps,), theta)
    psi_arr = np.full((num_steps,), psi)

    traj = np.hstack([positions, phi_arr[:, None], theta_arr[:, None], psi_arr[:, None]])

    return traj


def generate_slow_sine_curve_trajectory(start, end, T, dt, amplitude=0.2, wavelength=5.0, vertical_amplitude=0.2):
    """
    Generates a slow, smooth 3D sine-curve trajectory from start to end.
    The path oscillates laterally and vertically as it moves from start to end.

    Args:
        start: np.array, shape (3,), start position [x, y, z]
        end: np.array, shape (3,), end position [x, y, z]
        T: float, total time to traverse (seconds)
        dt: float, time step (seconds)
        amplitude: float, amplitude of lateral sine oscillation (meters)
        wavelength: float, wavelength of the sine curve (meters)
        vertical_amplitude: float, amplitude of vertical sine oscillation (meters)

    Returns:
        traj: np.array, shape (num_steps, 6), each row is [x, y, z, phi, theta, psi]
    """
    num_steps = int(np.round(T / dt)) + 1
    positions = np.linspace(start, end, num_steps)
    direction = end - start
    direction_norm = np.linalg.norm(direction)
    if direction_norm < 1e-8:
        # No movement, keep orientation zero
        phi = theta = psi = 0.0
        traj = np.hstack([positions, np.zeros((num_steps, 3))])
        return traj

    # Unit vector along the path
    dir_unit = direction / direction_norm

    # Find a vector perpendicular to dir_unit for lateral oscillation
    # Use z-axis unless path is vertical, then use x-axis
    if np.allclose(np.abs(dir_unit), [0, 0, 1]):
        perp = np.array([1, 0, 0])
    else:
        perp = np.cross(dir_unit, [0, 0, 1])
        perp /= np.linalg.norm(perp)

    # For vertical oscillation, use a vector perpendicular to both dir_unit and perp
    perp_vert = np.cross(dir_unit, perp)
    perp_vert /= np.linalg.norm(perp_vert)

    # Parameter along the path
    s = np.linspace(0, direction_norm, num_steps)
    # Lateral sine oscillation
    lateral_offset = amplitude * np.sin(2 * np.pi * s / wavelength)
    # Vertical sine oscillation (phase shifted for variety)
    vertical_offset = vertical_amplitude * np.sin(2 * np.pi * s / (wavelength * 1.5) + np.pi / 3)

    # Build full 3D positions
    pos = positions + np.outer(lateral_offset, perp) + np.outer(vertical_offset, perp_vert)

    # Compute tangent vectors for orientation
    pos_diff = np.gradient(pos, axis=0)
    tangent = pos_diff / np.linalg.norm(pos_diff, axis=1, keepdims=True)

    # Heading (psi): angle in XY plane
    psi = np.arctan2(tangent[:, 1], tangent[:, 0])
    # Pitch (theta): angle up/down from horizontal
    theta = np.arcsin(tangent[:, 2])
    # Roll (phi): always zero
    phi = np.zeros(num_steps)

    traj = np.hstack([pos, phi[:, None], theta[:, None], psi[:, None]])
    return traj

def main():
    bluerov_package_path = get_package_path('bluerov')
    model_params_path = bluerov_package_path + "/config/model_params.yaml"
    bluerov_params = utils_math.load_model_params(model_params_path)
    dynamics = BlueROVDynamics(bluerov_params)
    print("Mixer matrix:\n", dynamics.mixer)

    eta_0 = np.array([0, 0, -1.0, 0.0, 0.0, 0])  # Initial pose [x, y, z, phi, theta, psi]
    nu_0 = np.array([0, 0, 0, 0, 0, 0])  # Initial velocity [u, v, w, p, q, r]
    dt = 0.1  # Time step
    timesteps = 100  # Number of timesteps

    tau = np.array([-11.37, 0, 0, 0, 0, 0])  # exemplary wrench on the vehicle
    # tau =np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # no forces or moments applied
    # Create a time-varying force input (tau) with max amplitude 5.0, only forces (no moments)
    t_vec = np.linspace(0, timesteps * dt, timesteps)
    tau = np.zeros((timesteps, 6))
    amplitude = 3.0
    tau[:, 0] = amplitude * np.sin(2 * np.pi * 0.05 * t_vec)  # Example: sinusoidal surge force
    tau[:, 1] = amplitude * np.cos(2 * np.pi * 0.03 * t_vec)  # Example: cosine sway force
    tau[:, 2] = amplitude * np.sin(2 * np.pi * 0.02 * t_vec + np.pi/4)  # Example: phase-shifted heave force
    # tau[:, 3:6] remain zero (no moments)

    eta_all = np.zeros((timesteps + 1, 6))  # Store poses
    nu_all = np.zeros((timesteps + 1, 6))  # Store velocities

    eta_all[0, :] = eta_0
    nu_all[0, :] = nu_0

    for t in range(timesteps):
        nu_all[t+1, :], eta_all[t+1, :], _ = forward_dynamics(dynamics, tau[t,:], nu_all[t, :], eta_all[t, :], dt)

    animate_bluerov(eta_all, dt = dt)

    traj = generate_circle_trajectory()
    animate_bluerov(traj, dt = dt)

    traj = generate_sine_on_circle_trajectory_time(dt = dt, T = 10.0)
    animate_bluerov(traj, dt = dt)

    eta_all = np.zeros((timesteps + 1, 6))  # Store poses
    nu_all = np.zeros((timesteps + 1, 6))  # Store velocities

    eta_all[0, :] = eta_0
    nu_all[0, :] = nu_0
    tau_all = np.zeros((timesteps, 6))  # Store thruster forces

    # battery voltage for ESCs
    V_bat = 16.0  # Battery voltage in Volts

    # working with normalized PWM signals for ESCs: [1100, 1900] μs -> [-1, 1]
    # 8 entries for 8 thrusters
    u_esc = np.array([0.01, 0.01, 0.01, 0.01, 0.0, 0.0, 0.0, 0.0])

    # constraint the input to be in the range [-1, 1]
    u_esc = np.clip(u_esc, -1.0, 1.0)

    for t in range(timesteps):
        nu_all[t+1, :], eta_all[t+1, :], _, tau_all[t,:] = forward_dynamics_esc(dynamics, u_esc, nu_all[t, :], eta_all[t, :], dt, V_bat)

    animate_bluerov(eta_all, dt = dt)

    print('Tau all:', tau_all[0,:])

    # TODO: 
    # see how simulation behaves with pwm signals
    # MPC with tau optimization and pwm optimization
    # constraint: 
    # -1 <= u_esc[i] <= 1 for i in [0, 1, ..., 7]
    # minimize pwm sign wechsel, also damit thrust direction changes -> dort vielleicht irgendwie die deadzone einbauen / modellieren / asl constraint setzen?

    # for MPC and later, denormalize PWM signals to [1100, 1900] μs


    

    

if __name__ == "__main__":
    main()