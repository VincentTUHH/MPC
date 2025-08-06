import casadi as ca
import numpy as np
import bluerov.dynamics as dynamics
import bluerov.dynamics_symbolic as dynamics_symbolic
import bluerov.bluerov as bluerov
from common.animate import animate_bluerov, plot_vehicle_xy_vs_reference, plot_box_test, plot_pose_error_boxplots, plot_jacobian_condition_number, plot_vehicle_euler_angles_vs_reference_time, plot_vehicle_pos_vs_reference_time, plot_delta_u, plot_mpc_cost, plot_velocities, plot_control_inputs
import time
from scipy.linalg import solve_continuous_are
from common.my_package_path import get_package_path
import common.utils_sym as utils_sym
import common.utils_math as utils_math

def compute_lqr_gain(A, B, Q=None, R=None):
    # Q, R can be tuned; default to identity
    n = A.shape[0]  
    m = B.shape[1]
    if Q is None:
        Q = 5.0 * np.eye(n)
    if R is None:
        R = 0.01 * np.eye(m)
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.inv(R) @ B.T @ P
    return K

def linearize_system(f, x0, u0, eps=1e-6):
    n = x0.size
    m = u0.size
    A = np.zeros((n, n))
    B = np.zeros((n, m))
    fx0u0 = f(x0, u0)
    # Compute A matrix
    for i in range(n):
        dx = np.zeros_like(x0)
        dx[i] = eps
        A[:, i] = (f(x0 + dx, u0) - fx0u0) / eps
    # Compute B matrix
    for j in range(m):
        du = np.zeros_like(u0)
        du[j] = eps
        B[:, j] = (f(x0, u0 + du) - fx0u0) / eps
    return A, B

# --- Main MPC Function ---

def run_mpc(symbolic_bluerov, real_bluerov, reference_trajectory, T_trajectory, dt, N, integrator, opts, use_pwm=False, use_quaternion=False, use_tube_mpc=False, V_bat=16.0):
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
            quat = utils_math.euler_to_quat(*reference_trajectory[3:6, k])
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
            quat = utils_math.euler_to_quat(*euler)
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
        q_opt, u_optimal_horizon, Jopt, prev_g = solve_cftoc(
            N, n_dof, symbolic_bluerov, q_init, ref_traj_horizon, dt,
            u0, n_thruster, V_bat, use_pwm, use_quaternion, 
            integrator, opts, prev_g
        )

        if use_tube_mpc:
            u_star = u_optimal_horizon[:, 0]

            # --- Tube MPC: Feedback Correction ---
            # Linearize system at current operating point
            x0 = q_real[:, step]
            u_lin = np.zeros(n_thruster if use_pwm else n_dof)
            def f(x, u):
                nu = x[:n_dof]
                eta = x[n_dof:]
                nu_next, eta_next = integrate_dynamics(
                    symbolic_bluerov, nu, eta, u, dt, V_bat, use_pwm, False, integrator
                )
                nu_next = np.array(nu_next, dtype=float).flatten()
                eta_next = np.array(eta_next, dtype=float).flatten()
                x_next = np.concatenate([nu_next, eta_next])
                return x_next
            A, B = linearize_system(f, x0, u_lin)
            K = compute_lqr_gain(A, B)

            error = reference_trajectory[:, step] - q_real[:, step]
            u_optimal[:, step] = u_star + K @ error[:K.shape[1]]
        else:
            # No feedback correction, use optimal control directly
            u_optimal[:, step] = u_optimal_horizon[:, 0]
   
        u0 = np.hstack([u_optimal_horizon[:, 1:], u_optimal_horizon[:, -1][:, np.newaxis]])
        cost[step] = Jopt
        q_predict[:, :, step] = q_opt

        # Simulate real system (always in Euler)
        if use_pwm:
            # returns nu, eta, nu_dot, tau
            q_real[n_dof:, step + 1], q_real[:n_dof, step + 1], *_ = real_bluerov.forward_dynamics_esc(
                u_esc=u_optimal[:, step], nu=q_real[n_dof:, step], eta=q_real[:n_dof, step], dt=dt, V_bat=V_bat
            )
            # q_real[n_dof:, step + 1], q_real[:n_dof, step + 1], *_ = bluerov.forward_dynamics_esc_with_disturbance(
            #     real_bluerov, u_esc=u_optimal[:, step], nu=q_real[n_dof:, step], eta=q_real[:n_dof, step], dt=dt, V_bat=V_bat
            # )
        else:
            # returns nu, eta, nu_dot
            q_real[n_dof:, step + 1], q_real[:n_dof, step + 1], _ = real_bluerov.forward_dynamics(
                tau=u_optimal[:, step], nu=q_real[n_dof:, step], eta=q_real[:n_dof, step], dt=dt
            )

        jacobian = real_bluerov.J(q_real[0:6, step+1])
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
    elif integrator == 'semi_implicit_euler':
        # Semi-implicit Euler
        nu_dot, eta_dot = f(nu_k, eta_k)
        nu_next = nu_k + dt * nu_dot
        eta_dot_new = J_fun(eta_k) @ nu_next
        eta_next = eta_k + dt * eta_dot_new
        if use_quaternion:
            # Normalize quaternion part
            pos_next = eta_next[0:3]
            quat_next = eta_next[3:]
            quat_next = quat_next / ca.norm_2(quat_next)
            eta_next = ca.vertcat(pos_next, quat_next)
        return nu_next, eta_next


    else:
        raise ValueError(f"Unknown integrator type: {integrator}")

def solve_cftoc(N, n_dof, symbolic_bluerov, q_init, reference_trajectory, dt, u0, n_thruster, V_bat, use_pwm, use_quaternion, integrator, opts, prev_g):
    opti = ca.Opti()
    if opts is None:
        raise ValueError("Solver options 'opts' must be provided.")
    opti.solver('ipopt', opts)
    state_dim = n_dof + (7 if use_quaternion else 6)
    q = opti.variable(state_dim, N + 1)
    u = opti.variable(n_thruster if use_pwm else n_dof, N)
    slack = opti.variable(u.shape[0])

    # integral error
    # e_int = opti.variable(6, N+1)
    # opti.subject_to(e_int[:, 0] == 0)

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
            # Terminal constraint (soft, with new slack variable)
    
    # Terminal constraint: position and attitude error (via quaternion)
    # slack_terminal_pos = opti.variable(3)
    # slack_terminal_att = opti.variable(3)
    # ref_terminal = ca.DM(reference_trajectory[:, -1])
    # opti.subject_to(slack_terminal_pos >= 0)
    # opti.subject_to(q[n_dof:n_dof+3, N] - ref_terminal[0:3] <= slack_terminal_pos)
    # opti.subject_to(q[n_dof:n_dof+3, N] - ref_terminal[0:3] >= -slack_terminal_pos)
    # opti.subject_to(slack_terminal_att >= 0)
    # if use_quaternion:
    #     att_error_terminal = utils_sym.quaternion_error(ref_terminal[3:], q[n_dof+3:, N])
    # else:
    #     ref_quat_terminal = utils_sym.euler_to_quat(*ref_terminal[3:6])
    #     roll_t, pitch_t, yaw_t = ca.vertsplit(q[n_dof+3:, N])
    #     curr_quat_terminal = utils_sym.euler_to_quat(roll_t, pitch_t, yaw_t)
    #     att_error_terminal = utils_sym.quaternion_error(ref_quat_terminal, curr_quat_terminal)
    # opti.subject_to(att_error_terminal <= slack_terminal_att)
    # opti.subject_to(att_error_terminal >= -slack_terminal_att)

    

    # --- Cost Function (Compact & Efficient) ---
    # Weights
    Q_pos = 5.0 * ca.DM.eye(3)
    Q_att = 5.0 * ca.DM.eye(3)
    # Q_int = 0.5 * ca.DM.eye(6)  # Adjust as needed
    R_input = 0.02 * ca.DM.eye(u.shape[0])
    # R_input = ca.DM.zeros(u.shape[0], u.shape[0]) # only punishing state, doesnt really improve, so maybe it is the system that cant respond fast enough or the horizion is too short
    # R_delta_u = 0.001 * ca.DM.eye(u.shape[0])  # Weight for control smoothness

    cost = 0
    for k in range(N):
        # Position error
        pos_error = q[n_dof:n_dof+3, k] - reference_trajectory[0:3, k]
        cost += ca.dot(pos_error, Q_pos @ pos_error)

        # Attitude error (always via quaternion error for consistency)
        if use_quaternion:
            att_error = utils_sym.quaternion_error(reference_trajectory[3:, k], q[n_dof+3:, k])
        else:
            ref_quat = utils_sym.euler_to_quat(*reference_trajectory[3:, k])
            roll, pitch, yaw = ca.vertsplit(q[n_dof+3:, k])
            curr_quat = utils_sym.euler_to_quat(roll, pitch, yaw)
            att_error = utils_sym.quaternion_error(ref_quat, curr_quat)
        cost += ca.dot(att_error, Q_att @ att_error)

        # Control effort
        cost += ca.dot(u[:, k], R_input @ u[:, k])
    
        # Control smoothness (delta u)
        # if k > 0:
        #     delta_u = u[:, k] - u[:, k-1]
        #     cost += ca.dot(delta_u, R_delta_u @ delta_u)

        # accumulated error / integrated cost on pose error
        # without: small biases tolerated and system has no pressureto correct drift
        # e_int[:, k+1] = e_int[:, k] + dt * ca.vertcat(pos_error, att_error)
        # cost += ca.dot(e_int[:, k+1], Q_int @ e_int[:, k+1]) # k=0 is just 0

        # Terminal cost (soft, includes slack)
        # bringt gefühlt gar nichts, keinen Unterschied, egal wie hoch die Gewichtung ist zwischen *0.1 und *10
        # if k == N - 1:
        #     pos_error_terminal = q[n_dof:n_dof+3, k+1] - reference_trajectory[0:3, k]
        #     cost += ca.dot(pos_error_terminal, 10*Q_pos @ pos_error_terminal)
        #     if use_quaternion:
        #         att_error_terminal = utils_sym.quaternion_error(reference_trajectory[3:, k], q[n_dof+3:, k+1])
        #     else:
        #         ref_quat_terminal = utils_sym.euler_to_quat(*reference_trajectory[3:, k])
        #         roll_t, pitch_t, yaw_t = ca.vertsplit(q[n_dof+3:, k+1])
        #         curr_quat_terminal = utils_sym.euler_to_quat(roll_t, pitch_t, yaw_t)
        #         att_error_terminal = utils_sym.quaternion_error(ref_quat_terminal, curr_quat_terminal)
        #     cost += ca.dot(att_error_terminal, 10*Q_att @ att_error_terminal)

        

        # Add high penalty for slack in cost
        cost += 1e2 * ca.sumsqr(slack)
        #terminal slack
        # cost += 1e4 * (ca.sumsqr(slack_terminal_pos) + ca.sumsqr(slack_terminal_att))

    # opti.callback(lambda _: print("Current cost:", opti.debug.value(cost)))
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
    bluerov_package_path = get_package_path('bluerov')
    model_params_path = bluerov_package_path + "/config/model_params.yaml"
    model_params_disturbed_path = bluerov_package_path + "/config/model_params_disturbed.yaml"    
    bluerov_params_symbolic = utils_math.load_model_params(model_params_path)
    bluerov_params_dynamic = utils_math.load_model_params(model_params_disturbed_path)
    bluerov_dynamics = dynamics.BlueROVDynamics(bluerov_params_dynamic)
    bluerov_symbolic = dynamics_symbolic.BlueROVDynamicsSymbolic(bluerov_params_symbolic)

    T = 20.0
    fps = 20
    dt = 1 / fps
    n = 1
    N = 20  # MPC horizon

    use_quaternion = True  # Switch between Euler and quaternion model for MPC
    use_pwm = True         # Switch between tau and PWM control
    use_tube_mpc = False  # Use Tube MPC for feedback correction
    integrator_dict = {
        0: 'explicit_euler',
        1: 'rk4_coupled_coordinates', # improves slightly over explicit euler, but sloooooow
        2: 'rk4_decoupled_coordinates',
        3: 'CasADi_integrator', # Lösung ergibt überhaupt keinen Sinn
        4: 'rk4_coupled_function',
        5: 'semi_implicit_euler'
    }
    integrator = integrator_dict.get(0)

    opts = {'ipopt.print_level': 0, 'print_time': 0}


    # trajectory / real_traj states are always in euler angles
    # [eta, nu] where
    # eta = [x, y, z, phi, theta, psi] (pose in Euler angles)
    # nu = [u, v, w, p, q, r] (linear and angular velocities)
    # reference_eta = bluerov.generate_circle_trajectory_time(T=T, dt=dt, n=n).T

    reference_eta = bluerov.generate_sine_on_circle_trajectory_time(T=T, dt=dt, n=n).T

    # start = np.array([0.0, 0.0, -1.0])  # [x, y, z, roll, pitch, yaw]
    # end = np.array([-4.24, 4.24, -1.0]) # 5/4 pi
    # # end = np.array([-4.24, -4.24, -1.0]) # -5/4 pi
    # # end = np.array([4, 5, -1.0])
    # # end = np.array([-6, 0, -1.0]) # +-pi
    # end = np.array([4, 5, -1.0])
    # reference_eta = bluerov.generate_linear_trajectory(start, end, T, dt).T

    # reference_eta: shape (6, T), [x, y, z, roll, pitch, yaw]
    reference_eta[3, :] = np.unwrap(reference_eta[3, :])  # roll
    reference_eta[4, :] = np.unwrap(reference_eta[4, :])  # pitch
    reference_eta[5, :] = np.unwrap(reference_eta[5, :])  # yaw


    real_traj, cost, u_optimal = run_mpc(
        bluerov_symbolic, bluerov_dynamics, reference_eta, T*n, dt, N, 
        integrator, opts,
        use_pwm=use_pwm, use_quaternion=use_quaternion, 
        use_tube_mpc=use_tube_mpc, V_bat=16.0
    )

    real_eta = real_traj[:6, :-1].T
    real_nu = real_traj[6:, :-1].T
    reference_eta = reference_eta.T

    # Ensure all trajectories have the same length (number of rows)
    min_len = min(real_eta.shape[0], real_nu.shape[0], reference_eta.shape[0])
    if real_eta.shape[0] != min_len:
        print(f"Trimming real_eta from {real_eta.shape[0]} to {min_len} rows.")
        real_eta = real_eta[:min_len]
    if real_nu.shape[0] != min_len:
        print(f"Trimming real_nu from {real_nu.shape[0]} to {min_len} rows.")
        real_nu = real_nu[:min_len]
    if reference_eta.shape[0] != min_len:
        print(f"Trimming reference_eta from {reference_eta.shape[0]} to {min_len} rows.")
        reference_eta = reference_eta[:min_len]

    # Call plotting functions
    plot_delta_u(u_optimal, dt)
    plot_mpc_cost(cost, dt)
    plot_vehicle_pos_vs_reference_time(reference_eta, real_eta, dt)
    plot_vehicle_euler_angles_vs_reference_time(reference_eta, real_eta, dt)
    plot_velocities(real_nu, dt)
    plot_control_inputs(u_optimal, dt)
    plot_pose_error_boxplots(reference_eta, real_eta)
    plot_vehicle_xy_vs_reference(reference_eta, real_eta)
    animate_bluerov(real_eta, dt=dt)

if __name__ == "__main__":
    main()
