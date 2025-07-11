import casadi as ca
import numpy as np
import bluerov
from animate import animate_bluerov, plot_jacobian_condition_number, plot_vehicle_euler_angles_vs_reference_time, plot_vehicle_pos_vs_reference_time, plot_delta_u, plot_mpc_cost, plot_velocities, plot_control_inputs
import time
import matplotlib.pyplot as plt

# --- Utility Functions ---
class BlueROVDynamicsSymbolic:
    def __init__(self, params):
        self.mass = ca.DM(params['mass'])
        self.inertia = ca.DM(np.array(params['inertia'])) #
        self.cog = ca.DM(np.array(params['cog'])) # center of gravity
        self.added_mass = ca.DM(np.array(params['added_mass'])) # Added mass is a vector of 6 elements, one for each degree of freedom
        self.buoyancy = params['buoyancy']
        self.cob = ca.DM(np.array(params['cob'])) # center of buoyancy
        self.damping_linear = ca.DM(np.array(params['damping_linear']))
        self.damping_nonlinear = ca.DM(np.array(params['damping_nonlinear']))
        self.gravity = ca.DM(9.81)
        self.L = ca.DM(2.5166) # scaling factor for PWM to thrust conversion

        self.M = self.compute_mass_matrix()
        self.M_inv = ca.pinv(self.M)  # Inverse of the mass matrix

        self.mixer = self.get_mixer_matrix_wu2018()
        self.mixer_inv = ca.pinv(self.mixer)

    def get_mixer_matrix_niklas(self):
        # Thruster geometry parameters (from YAML or vehicle config)
        alpha_f = 0.733   # 42 / 180 * pi
        alpha_r = 0.8378  # 48 / 180 * pi
        l_hf = 0.163
        l_hr = 0.177
        l_vx = 0.12
        l_vy = 0.218

        calpha_f = np.cos(alpha_f)
        salpha_f = np.sin(alpha_f)
        calpha_r = np.cos(alpha_r)
        salpha_r = np.sin(alpha_r)

        # Mixer matrix for thrusters (NumPy first)
        mixer_np = np.array([
            [calpha_f, calpha_f  , calpha_r  , calpha_r , 0     , 0     , 0     , 0   ],
            [salpha_f, -salpha_f , -salpha_r , salpha_r , 0     , 0     , 0     , 0   ],
            [0       , 0         , 0         , 0        , 1     , -1    , -1    , 1   ],
            [0       , 0         , 0         , 0        , -l_vy , -l_vy , l_vy  , l_vy],
            [0       , 0         , 0         , 0        , -l_vx , l_vx  , -l_vx , l_vx],
            [l_hf    , -l_hf     , l_hr      , -l_hr    , 0     , 0     , 0     , 0   ]
        ])
        mixer = ca.DM(mixer_np)
        return mixer
    
    def get_mixer_matrix_wu2018(self):
        # Thruster geometry parameters (from Wu 2018)
        alpha_f = np.pi / 4
        alpha_r = np.pi / 4
        # unit direction of thrust
        n1 = np.array([np.cos(alpha_f), np.sin(alpha_f), 0])
        n2 = np.array([np.cos(alpha_f), -np.sin(alpha_f), 0])
        n3 = np.array([np.cos(alpha_r), -np.sin(alpha_r), 0])
        n4 = np.array([np.cos(alpha_r), np.sin(alpha_r), 0])
        n5 = np.array([0, 0, 1])
        n6 = np.array([0, 0, -1])
        n7 = np.array([0, 0, -1])
        n8 = np.array([0, 0, 1])
        # thruster lever arms (from Wu 2018)
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

        n_arr = np.stack([n1, n2, n3, n4, n5, n6, n7, n8], axis=1)  # shape (3, 8)
        l_arr = np.stack([l1, l2, l3, l4, l5, l6, l7, l8], axis=1)  # shape (3, 8)

        # Compute cross product for each column
        cross_arr = np.cross(l_arr.T, n_arr.T).T  # shape (3, 8)

        mixer_np = np.vstack((n_arr, cross_arr))  # shape (6, 8)
        mixer_np[np.abs(mixer_np) < 1e-6] = 0.0  # Enforce exact zeros for small values

        mixer = ca.DM(mixer_np)
        return mixer

    def compute_mass_matrix(self):
        M_rb = ca.DM.zeros(6, 6)
        M_rb[0:3, 0:3] = self.mass * ca.DM.eye(3)
        M_rb[0:3, 3:6] = -self.mass * skew_symmetric_symbolic(self.cog) # self.cog is the vector from body-fixed frame to the center of gravity expressed in the body-fixed frame
        M_rb[3:6, 0:3] = self.mass * skew_symmetric_symbolic(self.cog)
        # inertia tensor with respect to the center of gravity - Steiner's theorem to account for the offset of the center of gravity, 
        # when reference system for the generalized coordinates 
        M_rb[3:6, 3:6] = ca.diag(self.inertia) - self.mass * skew_symmetric_symbolic(self.cog) @ skew_symmetric_symbolic(self.cog)
        M = M_rb + ca.diag(self.added_mass)
        return M
    
    def D(self, nu):  # Damping matrix (CasADi compatible)
        D = ca.MX.zeros(6, 6)
        D_diag = self.damping_linear + self.damping_nonlinear * ca.fabs(nu)
        for i in range(6):
            D[i, i] = D_diag[i]
        return D

    def C(self, nu):  # Coriolis-centripetal matrix: forces due to the motion of the vehicle (CasADi compatible)
        C = ca.MX.zeros(6, 6) # C cant be a ca.DM matrix (numeric values only) as symbolic parameters (MX or SX) - as nu is a vector being optimized - are inserted in C
        v = nu[0:3]  # Linear velocities [u, v, w]
        w = nu[3:6]  # Angular velocities [p, q, r]
        # Compute the Coriolis forces based on the velocity and angular velocity
        C[0:3, 3:6] = -skew_symmetric_symbolic(self.M[0:3, 0:3] @ v + self.M[0:3, 3:6] @ w)
        C[3:6, 0:3] = -skew_symmetric_symbolic(self.M[0:3, 0:3] @ v + self.M[0:3, 3:6] @ w)
        C[3:6, 3:6] = -skew_symmetric_symbolic(self.M[3:6, 0:3] @ v + self.M[3:6, 3:6] @ w)
        return C

    def rotation_matrix_from_euler(self, phi, theta, psi):
        # ZYX convention
        cphi = ca.cos(phi)
        sphi = ca.sin(phi)
        ctheta = ca.cos(theta)
        stheta = ca.sin(theta)
        cpsi = ca.cos(psi)
        spsi = ca.sin(psi)
        R = ca.vertcat(
            ca.horzcat(cpsi * ctheta, cpsi * stheta * sphi - spsi * cphi, cpsi * stheta * cphi + spsi * sphi),
            ca.horzcat(spsi * ctheta, spsi * stheta * sphi + cpsi * cphi, spsi * stheta * cphi - cpsi * sphi),
            ca.horzcat(-stheta,        ctheta * sphi,                   ctheta * cphi)
        )
        return R

    def rotation_matrix_from_quat(self, quat):
        # quat: [w, x, y, z]
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        R = ca.vertcat(
            ca.horzcat(1 - 2*(y**2 + z**2),     2*(x*y - w*z),         2*(x*z + w*y)),
            ca.horzcat(2*(x*y + w*z),           1 - 2*(x**2 + z**2),   2*(y*z - w*x)),
            ca.horzcat(2*(x*z - w*y),           2*(y*z + w*x),         1 - 2*(x**2 + y**2))
        )
        return R

    def g(self, eta):  # gravitational and buoyancy forces + moments (CasADi compatible)
        # eta is the pose vector [x, y, z, phi, theta, psi]
        phi = eta[3]
        theta = eta[4]
        psi = eta[5]
        R = self.rotation_matrix_from_euler(phi, theta, psi)
        fg = self.mass * R.T @ ca.DM([0, 0, -self.gravity])
        fb = self.buoyancy * R.T @ ca.DM([0, 0, self.gravity])
        g_vec = ca.MX.zeros(6, 1)
        g_vec[0:3] = -(fg + fb)
        # Moments due to the forces acting at the center of gravity and center of buoyancy
        g_vec[3:6] = -(skew_symmetric_symbolic(self.cog) @ fg + skew_symmetric_symbolic(self.cob) @ fb)
        return g_vec

    def g_quat(self, eta):  # gravitational and buoyancy forces + moments (CasADi, quaternion version)
        # quat: [w, x, y, z]
        quat = eta[3:]
        R = self.rotation_matrix_from_quat(quat)
        fg = self.mass * R.T @ ca.DM([0, 0, -self.gravity])
        fb = self.buoyancy * R.T @ ca.DM([0, 0, self.gravity])
        g_vec = ca.MX.zeros(6, 1)
        g_vec[0:3] = -(fg + fb)
        g_vec[3:6] = -(skew_symmetric_symbolic(self.cog) @ fg + skew_symmetric_symbolic(self.cob) @ fb)
        return g_vec

    def J(self, eta):
        # Transformation from body to inertial (Euler)
        phi, theta, psi = eta[3], eta[4], eta[5]
        R = self.rotation_matrix_from_euler(phi, theta, psi)
        cphi = ca.cos(phi)
        sphi = ca.sin(phi)
        ctheta = ca.cos(theta)
        stheta = ca.sin(theta)
        T = ca.vertcat(
            ca.horzcat(1, sphi*stheta/ctheta, cphi*stheta/ctheta),
            ca.horzcat(0, cphi, -sphi),
            ca.horzcat(0, sphi/ctheta, cphi/ctheta)
        )
        J = ca.vertcat(
            ca.horzcat(R, ca.DM.zeros(3,3)),
            ca.horzcat(ca.DM.zeros(3,3), T)
        )
        return J

    def J_quat(self, eta):
        # Transformation from body to inertial (quaternion)
        # quat: [w, x, y, z]
        quat = eta[3:]
        R = self.rotation_matrix_from_quat(quat)
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        # Angular velocity mapping for quaternion
        Omega = ca.vertcat(
            ca.horzcat(-x, -y, -z),
            ca.horzcat(w, -z, y),
            ca.horzcat(z, w, -x),
            ca.horzcat(-y, x, w)
        )
        J = ca.vertcat(
            ca.horzcat(R, ca.DM.zeros(3,3)),
            ca.horzcat(ca.DM.zeros(4,3), 0.5*Omega)
        )
        return J

def euler_to_quat_np(roll, pitch, yaw):
    cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
    cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
    cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return np.array([w, x, y, z])

def euler_to_quat_casadi(roll, pitch, yaw):
    cy, sy = ca.cos(yaw * 0.5), ca.sin(yaw * 0.5)
    cp, sp = ca.cos(pitch * 0.5), ca.sin(pitch * 0.5)
    cr, sr = ca.cos(roll * 0.5), ca.sin(roll * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return ca.vertcat(w, x, y, z)

def skew_symmetric_symbolic(v):
    return ca.vertcat(
        ca.horzcat(0, -v[2], v[1]),
        ca.horzcat(v[2], 0, -v[0]),
        ca.horzcat(-v[1], v[0], 0)
    )

# --- Main MPC Function ---

def run_mpc(symbolic_bluerov, real_bluerov, reference_trajectory, T_trajectory, dt, N, integrator, opts, use_pwm=False, use_quaternion=False, V_bat=16.0):
    M = int(T_trajectory / dt)
    n_dof = 6
    n_thruster = 8

    # in optimization the state variable is:
    # q = [nu, eta] where 
    # nu = [u, v, w, p, q, r] (linear and angular velocities)
    # eta = [x, y, z, phi, theta, psi] (pose in Euler angles)
    # or eta = [x, y, z, qw, qx, qy, qz] (pose in quaternion)

    # in real world / simulation the state variable is:
    # q_real = [eta, nu] where
    # eta = [x, y, z, phi, theta, psi] (pose in Euler angles)
    # nu = [u, v, w, p, q, r] (linear and angular velocities)


    # State: [nu; eta] (twist first, then pose)
    if use_quaternion:
        state_dim = n_dof + 7
    else:
        state_dim = n_dof + 6

    q_predict = np.zeros((state_dim, N+1, M))

    q_real = np.zeros((2 * n_dof, M + 1))
    q_real[:n_dof, 0] = reference_trajectory[:, 0]
    if reference_trajectory.shape[1] >= 3: # moving average for velocity
        q_real[n_dof:, 0] = (reference_trajectory[:, 2] - reference_trajectory[:, 0]) / (2 * dt)
    else:
        q_real[n_dof:, 0] = (reference_trajectory[:, 1] - reference_trajectory[:, 0]) / dt

    # dual variables for warm start
    prev_g = None

    # Convert reference trajectory to quaternion if needed
    if use_quaternion:
        reference_trajectory_quat = np.zeros((7, reference_trajectory.shape[1]))
        reference_trajectory_quat[0:3, :] = reference_trajectory[0:3, :]
        for k in range(reference_trajectory.shape[1]):
            quat = euler_to_quat_np(*reference_trajectory[3:6, k])
            reference_trajectory_quat[3:7, k] = quat
        reference_trajectory = reference_trajectory_quat

    u_optimal = np.zeros((n_thruster if use_pwm else n_dof, M))
    cost = np.zeros(M)

    jacobian_array = np.zeros(M)

    # Initial guess for control
    u0 = np.full((u_optimal.shape[0], N), 0.2)

    start_time = time.time()

    for step in range(M):
        print(f"Step {step + 1} / {M}")
        # Reference for horizon
        ref_traj_horizon = reference_trajectory[:, step:step + N]
        if ref_traj_horizon.shape[1] < N:
            pad = np.tile(ref_traj_horizon[:, -1:], (1, N - ref_traj_horizon.shape[1]))
            ref_traj_horizon = np.concatenate([ref_traj_horizon, pad], axis=1)

        # Prepare current state for optimization
        if use_quaternion:
            pos = q_real[:3, step]
            euler = q_real[3:n_dof, step]
            quat = euler_to_quat_np(*euler)
            eta = np.concatenate([pos, quat])
            nu = q_real[n_dof:, step]
            q0 = np.concatenate([nu, eta])
        else:
            nu = q_real[n_dof:, step]
            eta = q_real[:n_dof, step]
            q0 = np.concatenate([nu, eta])

        # Prepare initial guess for q using previous prediction
        if step == 0:
            # First step: use q0 for all horizon steps
            q_init = np.tile(q0.reshape(-1, 1), (1, N + 1))
        else:
            # Use previous prediction, shifted by one, and append last predicted state
            prev_q_pred = q_predict[:, :, step - 1]
            q_init = np.hstack([prev_q_pred[:, 1:], prev_q_pred[:, -1:]])
            # Replace first column with actual state q0
            q_init[:, 0] = q0

        # Solve optimal control problem
        if step == 0:
            # First step: no warm start
            q_opt, u_optimal_horizon, Jopt, prev_g = solve_cftoc(
            N, n_dof, symbolic_bluerov, q_init, ref_traj_horizon, dt,
            u0, n_thruster, V_bat, use_pwm, use_quaternion, 
            integrator, opts, prev_g
            )
        else:
            # Warm start using previous solver
            q_opt, u_optimal_horizon, Jopt, prev_g = solve_cftoc(
            N, n_dof, symbolic_bluerov, q_init, ref_traj_horizon, dt,
            u0, n_thruster, V_bat, use_pwm, use_quaternion, 
            integrator, opts, prev_g
            )

        u_optimal[:, step] = u_optimal_horizon[:, 0]
        u0 = np.hstack([u_optimal_horizon[:, 1:], u_optimal_horizon[:, -1][:, np.newaxis]])
        cost[step] = Jopt
        q_predict[:, :, step] = q_opt

        # Simulate real system (always in Euler)
        if use_pwm:
            # returns nu, eta, nu_dot, tau
            q_real[n_dof:, step + 1], q_real[:n_dof, step + 1], *_ = bluerov.forward_dynamics_esc(
                real_bluerov, u_esc=u_optimal[:, step], nu=q_real[n_dof:, step], eta=q_real[:n_dof, step], dt=dt, V_bat=V_bat
            )
        else:
            # returns nu, eta, nu_dot
            q_real[n_dof:, step + 1], q_real[:n_dof, step + 1], _ = bluerov.forward_dynamics(
                real_bluerov, tau=u_optimal[:, step], nu=q_real[n_dof:, step], eta=q_real[:n_dof, step], dt=dt
            )

        jacobian = bluerov.rigid_body_jacobian_euler(q_real[0:6, step+1])
        jacobian_array[step] = np.linalg.cond(jacobian)
        
    end_time = time.time()
    print(f"MPC computation time: {end_time - start_time:.2f} seconds")
    plot_jacobian_condition_number(jacobian_array, dt)

    return q_real, cost, u_optimal

# --- Optimal Control Problem ---
def integrate_dynamics(symbolic_bluerov, nu_k, eta_k, u_k, dt, V_bat, use_pwm, use_quaternion, integrator):
    # integrator: string, e.g. 'explicit_euler', 'rk4_coupled_coordinates', etc.
    # Returns nu_next, eta_next
    if use_pwm:
        tau = symbolic_bluerov.L * V_bat * (symbolic_bluerov.mixer @ u_k)
    else:
        tau = u_k

    if use_quaternion:
        g_fun = symbolic_bluerov.g_quat
        J_fun = symbolic_bluerov.J_quat
    else:
        g_fun = symbolic_bluerov.g
        J_fun = symbolic_bluerov.J

    def f(nu, eta):
        nu_dot = symbolic_bluerov.M_inv @ (tau - symbolic_bluerov.C(nu) @ nu - symbolic_bluerov.D(nu) @ nu - g_fun(eta))
        eta_dot = J_fun(eta) @ nu
        return nu_dot, eta_dot

    if integrator == 'explicit_euler':
        nu_dot, eta_dot = f(nu_k, eta_k)
        nu_next = nu_k + dt * nu_dot
        eta_next = eta_k + dt * eta_dot
        if use_quaternion:
            # Normalize quaternion part
            pos_next = eta_next[0:3]
            quat_next = eta_next[3:]
            quat_next = quat_next / ca.norm_2(quat_next)
            eta_next = ca.vertcat(pos_next, quat_next)
        return nu_next, eta_next

    elif integrator in ['rk4_coupled_coordinates', 'rk4_decoupled_coordinates', 'rk4_coupled_function']:
        # Standard RK4
        k1_nu, k1_eta = f(nu_k, eta_k)
        k2_nu, k2_eta = f(nu_k + 0.5 * dt * k1_nu, eta_k + 0.5 * dt * k1_eta)
        k3_nu, k3_eta = f(nu_k + 0.5 * dt * k2_nu, eta_k + 0.5 * dt * k2_eta)
        k4_nu, k4_eta = f(nu_k + dt * k3_nu, eta_k + dt * k3_eta)
        nu_next = nu_k + (dt / 6.0) * (k1_nu + 2 * k2_nu + 2 * k3_nu + k4_nu)
        eta_next = eta_k + (dt / 6.0) * (k1_eta + 2 * k2_eta + 2 * k3_eta + k4_eta)
        if use_quaternion:
            pos_next = eta_next[0:3]
            quat_next = eta_next[3:]
            quat_next = quat_next / ca.norm_2(quat_next)
            eta_next = ca.vertcat(pos_next, quat_next)
        return nu_next, eta_next

    elif integrator == 'CasADi_integrator':
        # Use CasADi's built-in integrator
        state = ca.vertcat(nu_k, eta_k)
        state_dim = state.size1()
        u_dim = u_k.size1()
        x = ca.MX.sym('x', state_dim)
        u_sym = ca.MX.sym('u', u_dim)
        if use_pwm:
            tau_sym = symbolic_bluerov.L * V_bat * (symbolic_bluerov.mixer @ u_sym)
        else:
            tau_sym = u_sym
        nu_sym = x[0:nu_k.size1()]
        eta_sym = x[nu_k.size1():]
        if use_quaternion:
            g_fun_sym = symbolic_bluerov.g_quat
            J_fun_sym = symbolic_bluerov.J_quat
        else:
            g_fun_sym = symbolic_bluerov.g
            J_fun_sym = symbolic_bluerov.J
        nu_dot_sym = symbolic_bluerov.M_inv @ (tau_sym - symbolic_bluerov.C(nu_sym) @ nu_sym - symbolic_bluerov.D(nu_sym) @ nu_sym - g_fun_sym(eta_sym))
        eta_dot_sym = J_fun_sym(eta_sym) @ nu_sym
        ode = ca.vertcat(nu_dot_sym, eta_dot_sym)
        dae = {'x': x, 'p': u_sym, 'ode': ode}
        opts = {'tf': dt}
        integrator_fun = ca.integrator('integrator_fun', 'rk', dae, opts)
        res = integrator_fun(x0=state, p=u_k)
        state_next = res['xf']
        if use_quaternion:
            pos_next = state_next[nu_k.size1():nu_k.size1()+3]
            quat_next = state_next[nu_k.size1()+3:]
            quat_next = quat_next / ca.norm_2(quat_next)
            eta_next = ca.vertcat(pos_next, quat_next)
            nu_next = state_next[0:nu_k.size1()]
        else:
            nu_next = state_next[0:nu_k.size1()]
            eta_next = state_next[nu_k.size1():]
        return nu_next, eta_next

    else:
        raise ValueError(f"Unknown integrator type: {integrator}")
    
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
        att_error = ca.mtimes(w_g, v_c) - ca.mtimes(w_c, v_g) - ca.mtimes(goal_att_tilde, v_c)
        return att_error

def solve_cftoc(N, n_dof, symbolic_bluerov, q_init, reference_trajectory, dt, u0, n_thruster, V_bat, use_pwm, use_quaternion, integrator, opts, prev_g):
    # TODO: ich berechne mir ja auch ein q_predict, übernehme das in die nächste Optimierung und setze es als Startwert für q, also nicht
    # immer q0 asl STratwert für alles nehmen. Wobei q_predict der erste Wert durch q0 ersetzt werden sollte, also dem echten state
    # oder ist es vielleicht besser q0 weiterhin als erstes aber die restlichen elemente von reference trajectory zu nehmen? wobei dort kein twist dabei ist...
    
    
    opti = ca.Opti()
    if opts is None:
        raise ValueError("Solver options 'opts' must be provided.")
    opti.solver('ipopt', opts)
    state_dim = n_dof + (7 if use_quaternion else 6)
    q = opti.variable(state_dim, N + 1)
    u = opti.variable(n_thruster if use_pwm else n_dof, N)
    slack = opti.variable(u.shape[0])

    # Initial state constraint
    opti.subject_to(q[:, 0] == ca.DM(q_init[:, 0]))
    opti.subject_to(slack >= 0)

    # Warm start: if previous solver/solution is available, use its values except for q (always use q_init)
    
    opti.set_initial(u, u0)
    opti.set_initial(slack, 0.0)
    # Always use q_init for q
    opti.set_initial(q, q_init)

    # Dynamics and constraints
    for k in range(N):
        nu_k = q[:n_dof, k]
        eta_k = q[n_dof:, k]
        u_k = u[:, k]
        # Use integrate_dynamics for propagation
        nu_next, eta_next = integrate_dynamics(
            symbolic_bluerov, nu_k, eta_k, u_k, dt, V_bat, use_pwm, use_quaternion, integrator
        )
        opti.subject_to(q[:n_dof, k+1] == nu_next)
        opti.subject_to(q[n_dof:, k+1] == eta_next)

        # Control input constraints
        if use_pwm:
            # Introduce slack variable for PWM constraint violation
            opti.subject_to(u[:, k] <= 1.0 + slack)
            opti.subject_to(u[:, k] >= -1.0 - slack)
        else:
            opti.subject_to(u[:, k] <= 25.0 + slack)
            opti.subject_to(u[:, k] >= -25.0 - slack)

    # --- Cost Function (Compact & Efficient) ---
    # Weights
    Q_pos = 5.0 * ca.DM.eye(3)
    Q_att = 5.0 * ca.DM.eye(3)
    R_input = 0.01 * ca.DM.eye(u.shape[0])
    R_delta_u = 0.1 * ca.DM.eye(u.shape[0])  # Weight for control smoothness

    cost = 0
    for k in range(N):
        # Position error
        pos_error = q[n_dof:n_dof+3, k] - reference_trajectory[0:3, k]
        cost += ca.dot(pos_error, Q_pos @ pos_error)

        # Attitude error (always via quaternion error for consistency)
        if use_quaternion:
            att_error = quaternion_error(reference_trajectory[3:, k], q[n_dof+3:, k])
        else:
            ref_quat = euler_to_quat_casadi(*reference_trajectory[3:, k])
            roll, pitch, yaw = ca.vertsplit(q[n_dof+3:, k])
            curr_quat = euler_to_quat_casadi(roll, pitch, yaw)
            att_error = quaternion_error(ref_quat, curr_quat)
        cost += ca.dot(att_error, Q_att @ att_error)

        # Control effort
        cost += ca.dot(u[:, k], R_input @ u[:, k])

        # Control smoothness (delta u)
        # if k > 0:
        #     delta_u = u[:, k] - u[:, k-1]
        #     cost += ca.dot(delta_u, R_delta_u @ delta_u)

        # Add high penalty for slack in cost
        cost += 1e2 * ca.sumsqr(slack)

    opti.callback(lambda _: print("Current cost:", opti.debug.value(cost)))
    opti.minimize(cost)

    # Warm start dual variables if available
    if prev_g is not None:
        try:
            opti.set_initial(opti.lam_g, prev_g)
        except Exception as e:
            print("Warning: Could not set dual variables for warm start:", e)

    try:
        sol = opti.solve()
        return sol.value(q), sol.value(u), sol.value(cost), sol.value(opti.lam_g)
    except RuntimeError:
        print("Infeasible. Debug info:")
        print("q0:", opti.debug.value(q[:, 0]))
        print("u0:", opti.debug.value(u[:, 0]))
        return np.zeros((q.shape[0], N+1)), np.zeros((u.shape[0], N)), 1e6

# --- Main Function ---

def main():
    bluerov_params = bluerov.load_model_params('model_params.yaml')
    bluerov_dynamics = bluerov.BlueROVDynamics(bluerov_params)
    bluerov_symbolic = BlueROVDynamicsSymbolic(bluerov_params)

    T = 20.0
    fps = 20
    dt = 1 / fps
    n = 1
    N = 20  # MPC horizon

    use_quaternion = True  # Switch between Euler and quaternion model for MPC
    use_pwm = True         # Switch between tau and PWM control
    integrator_dict = {
        0: 'explicit_euler',
        1: 'rk4_coupled_coordinates', # improves slightly over explicit euler, but sloooooow
        2: 'rk4_decoupled_coordinates',
        3: 'CasADi_integrator', # Lösung ergibt überhaupt keinen Sinn
        4: 'rk4_coupled_function'
    }
    integrator = integrator_dict.get(0)

    opts = {'ipopt.print_level': 0, 'print_time': 0}


    # trajectory / real_traj states are always in euler angles
    # [eta, nu] where
    # eta = [x, y, z, phi, theta, psi] (pose in Euler angles)
    # nu = [u, v, w, p, q, r] (linear and angular velocities)
    # reference_eta = bluerov.generate_circle_trajectory_time(T=T, dt=dt, n=n).T

    start = np.array([0.0, 0.0, -1.0])  # [x, y, z, roll, pitch, yaw]
    # end = np.array([-4.24, 4.24, -1.0]) # 5/4 pi
    # end = np.array([-4.24, -4.24, -1.0]) # -5/4 pi
    # end = np.array([4, 5, -1.0])
    # end = np.array([-6, 0, -1.0]) # +-pi
    end = np.array([4, 5, -1.0])
    reference_eta = bluerov.generate_linear_trajectory(start, end, T, dt).T


    real_traj, cost, u_optimal = run_mpc(
        bluerov_symbolic, bluerov_dynamics, reference_eta, T*n, dt, N, 
        integrator, opts,
        use_pwm=use_pwm, use_quaternion=use_quaternion, V_bat=16.0
    )

    real_eta = real_traj[:6, :-1].T
    real_nu = real_traj[6:, :-1].T
    reference_eta = reference_eta.T

    # Call plotting functions
    plot_delta_u(u_optimal, dt)
    plot_mpc_cost(cost, dt)
    plot_vehicle_pos_vs_reference_time(reference_eta, real_eta, dt)
    plot_vehicle_euler_angles_vs_reference_time(reference_eta, real_eta, dt)
    plot_velocities(real_nu, dt)
    plot_control_inputs(u_optimal, dt)
    animate_bluerov(real_eta, dt=dt)

if __name__ == "__main__":
    main()
