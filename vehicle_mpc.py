import casadi as ca
import numpy as np
from scipy.spatial.transform import Rotation as Rot
import bluerov
from animate import animate_bluerov, plot_vehicle_pos_vs_reference
import time

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

        self.M = self.compute_mass_matrix()

        self.M_inv = ca.inv(self.M)  # Inverse of the mass matrix

        # Thruster geometry parameters (from YAML)
        alpha_f = 0.733   # 42 / 180 * pi
        alpha_r = 0.8378  # 48 / 180 * pi
        l_hf = 0.163
        l_hr = 0.177
        l_vx = 0.12
        l_vy = 0.218

        calpha_f = ca.cos(alpha_f)
        salpha_f = ca.sin(alpha_f)
        calpha_r = ca.cos(alpha_r)
        salpha_r = ca.sin(alpha_r)

        # Mixer matrix for thrusters (CasADi compatible)
        self.mixer = ca.DM([
            [calpha_f, calpha_f  , calpha_r  , calpha_r , 0     , 0     , 0     , 0   ],
            [salpha_f, -salpha_f , -salpha_r , salpha_r , 0     , 0     , 0     , 0   ],
            [0       , 0         , 0         , 0        , 1     , -1    , -1    , 1   ],
            [0       , 0         , 0         , 0        , -l_vy , -l_vy , l_vy  , l_vy],
            [0       , 0         , 0         , 0        , -l_vx , l_vx  , -l_vx , l_vx],
            [l_hf    , -l_hf     , l_hr      , -l_hr    , 0     , 0     , 0     , 0   ]
        ])
        self.mixer_inv = ca.pinv(self.mixer)

        self.L = ca.DM(2.5166) # scaling factor for PWM to thrust conversion

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

    def g(self, eta):  # gravitational and buoyancy forces + moments (CasADi compatible)
        # eta is the pose vector [x, y, z, phi, theta, psi]
        phi = eta[3]
        theta = eta[4]
        psi = eta[5]

        # CasADi rotation matrix from euler angles (ZYX)
        cphi = ca.cos(phi)
        sphi = ca.sin(phi)
        ctheta = ca.cos(theta)
        stheta = ca.sin(theta)
        cpsi = ca.cos(psi)
        spsi = ca.sin(psi)

        # Rotation matrix (ZYX convention)
        R = ca.vertcat(
            ca.horzcat(cpsi * ctheta, cpsi * stheta * sphi - spsi * cphi, cpsi * stheta * cphi + spsi * sphi),
            ca.horzcat(spsi * ctheta, spsi * stheta * sphi + cpsi * cphi, spsi * stheta * cphi - cpsi * sphi),
            ca.horzcat(-stheta,        ctheta * sphi,                   ctheta * cphi)
        )

        fg = self.mass * R.T @ ca.DM([0, 0, -self.gravity])
        fb = self.buoyancy * R.T @ ca.DM([0, 0, self.gravity])
        g_vec = ca.MX.zeros(6, 1)
        g_vec[0:3] = -(fg + fb)
        # Moments due to the forces acting at the center of gravity and center of buoyancy
        g_vec[3:6] = -(skew_symmetric_symbolic(self.cog) @ fg + skew_symmetric_symbolic(self.cob) @ fb)
        return g_vec

    def C(self, nu):  # Coriolis-centripetal matrix: forces due to the motion of the vehicle (CasADi compatible)
        C = ca.MX.zeros(6, 6) # C cant be a ca.DM matrix (numeric values only) as symbolic parameters (MX or SX) - as nu is a vector being optimized - are inserted in C
        v = nu[0:3]  # Linear velocities [u, v, w]
        w = nu[3:6]  # Angular velocities [p, q, r]
        # Compute the Coriolis forces based on the velocity and angular velocity
        C[0:3, 3:6] = -skew_symmetric_symbolic(self.M[0:3, 0:3] @ v + self.M[0:3, 3:6] @ w)
        C[3:6, 0:3] = -skew_symmetric_symbolic(self.M[0:3, 0:3] @ v + self.M[0:3, 3:6] @ w)
        C[3:6, 3:6] = -skew_symmetric_symbolic(self.M[3:6, 0:3] @ v + self.M[3:6, 3:6] @ w)
        return C
    

    @staticmethod
    def J(eta):
        phi = eta[3]
        theta = eta[4]
        psi = eta[5]
        cphi = ca.cos(phi)
        sphi = ca.sin(phi)
        ctheta = ca.cos(theta)
        stheta = ca.sin(theta)
        ttheta = ca.tan(theta)
        cpsi = ca.cos(psi)
        spsi = ca.sin(psi)

        # Rotation from body to inertial frame (ZYX convention)
        R = ca.vertcat(
            ca.horzcat(cpsi * ctheta, cpsi * stheta * sphi - spsi * cphi, cpsi * stheta * cphi + spsi * sphi),
            ca.horzcat(spsi * ctheta, spsi * stheta * sphi + cpsi * cphi, spsi * stheta * cphi - cpsi * sphi),
            ca.horzcat(-stheta,        ctheta * sphi,                   ctheta * cphi)
        )

        # Transformation from angular velocity in body to Euler angle rates
        T = ca.vertcat(
            ca.horzcat(1, sphi * ttheta, cphi * ttheta),
            ca.horzcat(0, cphi,         -sphi),
            ca.horzcat(0, sphi / ctheta, cphi / ctheta)
        )

        J = ca.MX.zeros(6, 6)
        J[0:3, 0:3] = R
        J[3:6, 3:6] = T
        return J


def skew_symmetric_symbolic(v):
    return ca.vertcat(
        ca.horzcat(0, -v[2], v[1]),
        ca.horzcat(v[2], 0, -v[0]),
        ca.horzcat(-v[1], v[0], 0)
    )

def run_mpc(symbolic_bluerov, real_bluerov, reference_trajectory, T_trajectory, dt, use_pwm=False, V_bat=16.0):
    M = int(T_trajectory / dt)  # total number of timesteps
    N = 20  # horizon length for the MPC
    n_dof = 6  # number of degrees of freedom for the BlueROV
    n_thruster = 8  # number of thrusters

    q_predict = np.zeros((2 * n_dof, N + 1, M))  # predicted trajectory
    if use_pwm:
        u_optimal = np.zeros((n_thruster, M))
    else:
        u_optimal = np.zeros((n_dof, M))  # optimal control inputs
    cost = np.zeros(M)  # cost for each step
    q_real = np.zeros((2 * n_dof, M + 1))  # real
    jacobian_array = np.zeros(M)
    
    q_real[0:6, 0] = reference_trajectory[:, 0]  # initial state
    # q_real[6:, 0] = np.zeros(n_dof)  # initial twist (zero)
    # Estimate initial twist (velocity) guess from reference trajectory
    if reference_trajectory.shape[1] >= 3:
        initial_twist_guess = (reference_trajectory[:, 2] - reference_trajectory[:, 0]) / (2 * dt)
    else:
        initial_twist_guess = (reference_trajectory[:, 1] - reference_trajectory[:, 0]) / dt
    # initial_twist_guess = (reference_trajectory[:, 1] - reference_trajectory[:, 0]) / dt
    q_real[6:, 0] = initial_twist_guess

    start_time = time.time()  # Startzeit MPC messen

    if use_pwm:
        u0 = np.full((n_thruster, N), 0.2)  # initial guess for PWM commands, all entries set to 0.5 (normalized [-1, 1])

    for step in range(M):
        # Ensure reference_trajectory for the horizon has N columns (pad with last column if needed)
        ref_traj_horizon = reference_trajectory[:, step:step + N]
        if ref_traj_horizon.shape[1] < N:
            last_col = ref_traj_horizon[:, -1].reshape(-1, 1)
            pad = np.repeat(last_col, N - ref_traj_horizon.shape[1], axis=1)
            ref_traj_horizon = np.concatenate([ref_traj_horizon, pad], axis=1)

        if use_pwm:
            q_opt, u_optimal_horizon, Jopt = solve_cftoc_pwm(N, n_dof, symbolic_bluerov, q_real[6:, step], q_real[0:6, step], ref_traj_horizon, dt, u0, n_thruster, V_bat)
        else:
            q_opt, u_optimal_horizon, Jopt = solve_cftoc_tau(N, n_dof, symbolic_bluerov, q_real[6:, step], q_real[0:6, step], ref_traj_horizon, dt)

        q_predict[:, :, step] = q_opt
        u_optimal[:, step] = u_optimal_horizon[:, 0] # this mpc solves for tau as control input
        u0 = np.hstack([u_optimal_horizon[:, 1:], u_optimal_horizon[:, -1][:, np.newaxis]])  # shift all values left and repeat the last one
        cost[step] = Jopt
        if use_pwm:
            # in real die u_ESC commands noch umrechnen von PWM [-1, 1] auf tau [1100, 1900]
            q_real[6:, step + 1], q_real[0:6, step + 1], _, _ = bluerov.forward_dynamics_esc(real_bluerov, u_esc = u_optimal[:, step], nu = q_real[6:, step], eta = q_real[0:6, step], dt = dt, V_bat = V_bat)
        else:
            q_real[6:, step + 1], q_real[0:6, step + 1], _ = bluerov.forward_dynamics(real_bluerov, tau = u_optimal[:, step], nu = q_real[6:, step], eta = q_real[0:6, step], dt = dt)

        jacobian = bluerov.rigid_body_jacobian_euler(q_real[0:6, step+1])
        jacobian_array[step] = np.linalg.cond(jacobian)

    elapsed_time = time.time() - start_time  # Benötigte Zeit berechnen
    print(f"MPC completed in {elapsed_time:.2f} seconds")

    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(np.arange(jacobian_array.shape[0]) * dt, jacobian_array)
    plt.xlabel('Time [s]')
    plt.ylabel('Jacobian Condition Number')
    plt.title('Jacobian Condition Number over Time')
    plt.grid(True)
    plt.show()

    return q_real, cost, u_optimal



def solve_cftoc_tau(N, n_dof, symbolic_bluerov, nu0, eta0, reference_trajectory, dt):
    opti = ca.Opti()
    opts = {'ipopt.print_level': 0, 'print_time': 0}
    # opts = {}
    opti.solver('ipopt', opts)

    q = opti.variable(2 * n_dof, N + 1)  # vehicle pose (with euler angles) eta and twist nu
    u = opti.variable(n_dof, N)  # joint velocities

    opti.set_initial(q, ca.repmat(ca.vertcat(eta0, nu0), 1, N + 1))
    opti.set_initial(u, ca.DM.ones(n_dof, N))

    opti.subject_to(q[0:6, 0] == ca.DM(eta0))  # initial pose
    opti.subject_to(q[6:, 0] == ca.DM(nu0))  # initial twist

    for k in range(N):
        for i in range(n_dof):
            opti.subject_to(u[i, k] <= 25.0)
            opti.subject_to(u[i, k] >= -25.0)
        # vehcile dynamics
        opti.subject_to(q[6:, k+1] == q[6:, k] + dt * ca.mtimes(symbolic_bluerov.M_inv, (u[:, k] - ca.mtimes(symbolic_bluerov.C(q[6:, k]), q[6:, k]) - ca.mtimes(symbolic_bluerov.D(q[6:, k]), q[6:, k]) - symbolic_bluerov.g(q[0:6, k]))))
        # vehicle kinematics
        opti.subject_to(q[0:6, k+1] == q[0:6, k] + dt * ca.mtimes(symbolic_bluerov.J(q[0:6,k]), q[6:, k+1]))

    cost = 0

    for k in range(N):
        pos_k = q[0:3, k]
        att_k = q[3:6, k]

        pos_error = ca.DM(reference_trajectory[0:3, k]) - pos_k
        att_error = ca.DM(reference_trajectory[3:6, k]) - att_k # with singularities might not be correct

        cost += ca.sumsqr(pos_error) + ca.sumsqr(att_error)

    opti.callback(lambda: print("Current cost:", opti.debug.value(cost)))

    opti.minimize(cost)

    try:
        sol = opti.solve()
    except RuntimeError:
        print("Infeasible. Checking debug values...")
        print("q0:", opti.debug.value(q[:, 0]))
        print("u0:", opti.debug.value(u[:, 0]))

    return sol.value(q), sol.value(u), sol.value(cost)


def euler_to_quat_casadi(roll, pitch, yaw):
    # Returns quaternion [w, x, y, z] from ZYX Euler angles (roll, pitch, yaw)
    cy = ca.cos(yaw * 0.5)
    sy = ca.sin(yaw * 0.5)
    cp = ca.cos(pitch * 0.5)
    sp = ca.sin(pitch * 0.5)
    cr = ca.cos(roll * 0.5)
    sr = ca.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return [w, x, y, z]

def solve_cftoc_pwm(N, n_dof, symbolic_bluerov, nu0, eta0, reference_trajectory, dt, u0, n_thruster=8, V_bat=16.0):
    # Integrator selection dictionary
    integrator_dict = {
        0: 'explicit_euler',
        1: 'rk4_coupled_coordinates',
        2: 'rk4_decoupled_coordinates',
        3: 'CasADi_integrator'
    }
    integrator = integrator_dict.get(3)  # Change the key to select the integrator

    opti = ca.Opti()
    opts = {'ipopt.print_level': 5, 'print_time': 1, 'ipopt.tol': 1e-4}
    opti.solver('ipopt', opts)

    q = opti.variable(2 * n_dof, N + 1)  # vehicle pose (with euler angles) eta and twist nu
    u = opti.variable(n_thruster, N)  # thruster PWM commands (normalized [-1, 1])

    u_slack = opti.variable()
    opti.subject_to(u_slack >= 0)

    q_init = np.tile(np.concatenate([np.array(eta0).flatten(), np.array(nu0).flatten()]), (N + 1, 1)).T
    opti.set_initial(q, q_init)
    opti.set_initial(u, u0)

    # --- CasADi integrator definition ---
    x_int = ca.MX.sym('x_int', 2 * n_dof)  # eta (6) + nu (6)
    u_int = ca.MX.sym('u_int', n_thruster)

    eta = x_int[0:6]
    nu = x_int[6:12]

    tau = symbolic_bluerov.L * V_bat * (symbolic_bluerov.mixer @ u_int)
    nu_dot = symbolic_bluerov.M_inv @ (tau - symbolic_bluerov.C(nu) @ nu - symbolic_bluerov.D(nu) @ nu - symbolic_bluerov.g(eta))
    eta_dot = symbolic_bluerov.J(eta) @ nu

    xdot = ca.vertcat(eta_dot, nu_dot)
    ode = {'x': x_int, 'p': u_int, 'ode': xdot}

    # The integrator must be created OUTSIDE the loop, but called INSIDE the loop!
    casadi_integrator = ca.integrator('integrator', 'cvodes', ode, {'tf': dt})

    for k in range(N):
        for thruster in range(n_thruster):
            opti.subject_to(u[thruster, k] <= 1.0 + u_slack)
            opti.subject_to(u[thruster, k] >= -1.0 - u_slack)

        if integrator == 'explicit_euler':
            tau = symbolic_bluerov.L * ca.DM(V_bat) * symbolic_bluerov.mixer @ u[:, k]
            nu_dot = symbolic_bluerov.M_inv @ (tau - symbolic_bluerov.C(q[6:, k]) @ q[6:, k] - symbolic_bluerov.D(q[6:, k]) @ q[6:, k] - symbolic_bluerov.g(q[0:6, k]))
            opti.subject_to(q[6:, k+1] == q[6:, k] + dt * nu_dot)
            opti.subject_to(q[0:6, k+1] == q[0:6, k] + dt * symbolic_bluerov.J(q[0:6, k]) @ q[6:, k])
        elif integrator == 'rk4_coupled_coordinates':
            def f(x, u):
                eta = x[0:6]
                nu = x[6:12]
                J_eta = symbolic_bluerov.J(eta)
                tau = symbolic_bluerov.L * V_bat * (symbolic_bluerov.mixer @ u)
                nu_dot = symbolic_bluerov.M_inv @ (tau - symbolic_bluerov.C(nu) @ nu - symbolic_bluerov.D(nu) @ nu - symbolic_bluerov.g(eta))
                eta_dot = J_eta @ nu
                return ca.vertcat(eta_dot, nu_dot)
            x_k = q[:, k]
            k1 = f(x_k, u[:, k])
            k2 = f(x_k + 0.5 * dt * k1, u[:, k])
            k3 = f(x_k + 0.5 * dt * k2, u[:, k])
            k4 = f(x_k + dt * k3, u[:, k])
            x_next = x_k + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            opti.subject_to(q[:, k+1] == x_next)
        elif integrator == 'rk4_decoupled_coordinates':
            def f(nu, eta):
                tau = symbolic_bluerov.L * V_bat * (symbolic_bluerov.mixer @ u[:, k])
                return symbolic_bluerov.M_inv @ (tau - symbolic_bluerov.C(nu) @ nu - symbolic_bluerov.D(nu) @ nu - symbolic_bluerov.g(eta))
            nu_k = q[6:12, k]
            eta_k = q[0:6, k]
            k1 = f(nu_k, eta_k)
            k2 = f(nu_k + 0.5 * dt * k1, eta_k)
            k3 = f(nu_k + 0.5 * dt * k2, eta_k)
            k4 = f(nu_k + dt * k3, eta_k)
            nu_next = nu_k + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            opti.subject_to(q[6:, k+1] == nu_next)
            def f_eta(eta):
                return symbolic_bluerov.J(eta) @ nu_k
            k1 = f_eta(eta_k)
            k2 = f_eta(eta_k + 0.5 * dt * k1)
            k3 = f_eta(eta_k + 0.5 * dt * k2)
            k4 = f_eta(eta_k + dt * k3)
            eta_next = eta_k + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            opti.subject_to(q[0:6, k+1] == eta_next)
        elif integrator == 'CasADi_integrator':
            # Correct usage: call the integrator function with x0 and p
            xk = q[:, k]
            uk = u[:, k]
            x_next = casadi_integrator(x0=xk, p=uk)['xf']
            opti.subject_to(q[:, k+1] == x_next)

    opti.subject_to(q[:, 0] == ca.vertcat(eta0, nu0))

    cost = 0
    q_weight_att = 5.0
    Q_att = q_weight_att * ca.diag(ca.DM([1.0, 1.0, 2.0]))
    q_weight_pos = 5.0
    Q_pos = q_weight_pos * ca.diag(ca.DM([1.0, 1.0, 1.0]))
    u_weight = ca.DM(0.0001)
    R_input = u_weight * ca.DM.eye(n_thruster)
    u_weight_change = ca.DM(0.0001)
    R_change = u_weight_change * ca.DM.eye(n_thruster)

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

    for k in range(N):
        pos_k = q[0:3, k]
        att_k = q[3:6, k]
        ref_quat  = euler_to_quat_casadi(reference_trajectory[3, k], reference_trajectory[4, k], reference_trajectory[5, k])
        curr_quat = euler_to_quat_casadi(att_k[0], att_k[1], att_k[2])
        att_error = quaternion_error(ref_quat, curr_quat)
        pos_error = ca.DM(reference_trajectory[0:3, k]) - pos_k
        cost += ca.mtimes([pos_error.T, Q_pos, pos_error])
        cost += ca.mtimes([att_error.T, Q_att, att_error])

    for k in range(N):
        cost += ca.mtimes([u[:, k].T, R_input, u[:, k]])

    delta_u = u[:, 1:] - u[:, :-1]
    for k in range(N - 1):
        cost += ca.mtimes([delta_u[:, k].T, R_change, delta_u[:, k]])

    cost += 1e2 * ca.sumsqr(u_slack)

    opti.callback(lambda _: print("Current cost:", opti.debug.value(cost)))
    opti.minimize(cost)

    try:
        sol = opti.solve()
        print("Value of pos_error:", opti.debug.value(pos_error))
    except RuntimeError:
        print("Infeasible. Checking debug values...")
        print("q0:", opti.debug.value(q[:, 0]))
        print("u0:", opti.debug.value(u[:, 0]))
        print("Initial cost estimate:", opti.debug.value(cost))

    print("Solver stats:", opti.stats())
    for k in range(N):
        print(f"u[:, {k}] =", sol.value(u[:, k]))

    return sol.value(q), sol.value(u), sol.value(cost)




def main():
    bluerov_params = bluerov.load_model_params('model_params.yaml')
    bluerov_dynamics = bluerov.BlueROVDynamics(bluerov_params)
    bluerov_symbolic = BlueROVDynamicsSymbolic(bluerov_params)

    T = 20.0  # seconds for one period of the sine wave trajectory
    fps = 20  # frames per second
    dt = 1 / fps  # seconds
    n = 1  # number of periods for the sine wave trajectory

    # reference_eta = bluerov.generate_sine_on_circle_trajectory_time(T=T, dt = dt, n=n)

    reference_eta = bluerov.generate_circle_trajectory_time(T=T, dt=dt, n=n)

    # start = np.array([0.0, 0.0, -1.0])  # [x, y, z, roll, pitch, yaw]
    # end = np.array([1.0, 2.0, -1.0])
    # reference_eta = bluerov.generate_linear_trajectory(start, end, T, dt)
    # n=1
    # print("Reference trajectory shape:", reference_eta)
    # return None

    ##################################
    # Ich habe geändert, dass dt = 1 /fps und nicht T/fps ist
    # Dadurch hat er Lösung gefunden innerhalb max iteration, vorher immer bei max_iter=3000 abgebrochen
    # Er war davor auch infeasible, was daran lag, dass reshape trajectory nicht dasselbe gemacht hat wie transpose
    ####################################


    reference_eta = reference_eta.T

    real_traj, cost, u_optimal = run_mpc(bluerov_symbolic, bluerov_dynamics, reference_eta, T*n, dt, use_pwm=True, V_bat=16.0)

    real_traj = real_traj[:, :-1]

    real_eta = real_traj[0:6, :].T
    real_nu = real_traj[6:, :].T

    import matplotlib.pyplot as plt

    # Plot the change of u_optimal (delta u) over time
    plt.figure()
    delta_u = np.diff(u_optimal, axis=1)
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



    plt.figure()
    plt.plot(np.arange(cost.shape[0]) * dt, cost)
    plt.xlabel('Time [s]')
    plt.ylabel('MPC Cost')
    plt.title('MPC Cost over Time')
    plt.grid(True)
    plt.show()

    plot_vehicle_pos_vs_reference(reference_eta.T, real_eta)
    plt.show()

    plt.figure(figsize=(10, 8))
    angle_labels = ['Roll (phi)', 'Pitch (theta)', 'Yaw (psi)']
    for i in range(3):
        plt.subplot(3, 1, i + 1)
        plt.plot(np.arange(reference_eta.shape[1]) * dt, reference_eta[i + 3, :], label='Reference')
        plt.plot(np.arange(real_eta.shape[0]) * dt, real_eta[:, i + 3], label='Real')
        plt.ylabel(angle_labels[i])
        plt.legend()
        plt.grid(True)
        if i == 0:
            plt.title('Euler Angles: Reference vs Real')
    plt.xlabel('Time [s]')
    plt.tight_layout()
    plt.show()

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

    animate_bluerov(real_eta, dt = dt)

    


if __name__ == "__main__":
    main()