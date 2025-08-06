import numpy as np
import common.utils_math as utils_math

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

    def forward_dynamics(self, tau, nu, eta, dt):
        nu_dot = np.linalg.inv(self.M) @ (tau - self.C(nu) @ nu - self.D(nu) @ nu - self.g(eta))
        nu = nu + dt * nu_dot # Euler forward integration
        eta = eta + dt * self.J(eta) @ nu
        return nu, eta, nu_dot
    
    def forward_dynamics_esc(self, u_esc, nu, eta, dt, V_bat=16.0):
        tau = self.mixer @ u_esc * self.L * V_bat  # convert ESC signals to thrusts
        nu_dot = np.linalg.inv(self.M) @ (tau - self.C(nu) @ nu - self.D(nu) @ nu - self.g(eta))
        nu = nu + dt * nu_dot
        eta = eta + dt * self.J(eta) @ nu
        return nu, eta, nu_dot, tau
    
    def forward_dynamics_esc_with_disturbance(
        self, u_esc, nu, eta, dt, V_bat=16.0, 
        disturbance_prob=0.1, force_max=40.0, torque_max=40.0
    ):
        tau = self.mixer @ u_esc * self.L * V_bat
        # Randomly decide whether to add disturbance
        if np.random.rand() < disturbance_prob:
            tau_dist = np.zeros(6)
            tau_dist[:3] = np.random.uniform(-force_max, force_max, 3)
            tau_dist[3:] = np.random.uniform(-torque_max, torque_max, 3)
        else:
            tau_dist = np.zeros(6)
        tau_total = tau + tau_dist

        nu_dot = np.linalg.inv(self.M) @ (tau_total - self.C(nu) @ nu - self.D(nu) @ nu - self.g(eta))
        nu = nu + dt * nu_dot
        eta = eta + dt * self.J(eta) @ nu
        return nu, eta, nu_dot, tau_total
    
    def inverse_dynamics(self, nu_dot, nu, eta):
        tau = self.M @ nu_dot + self.C(nu) @ nu + self.D(nu) @ nu + self.g(eta)
        return tau