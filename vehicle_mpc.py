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
    N = 3  # horizon length for the MPC
    n_dof = 6  # number of degrees of freedom for the BlueROV
    n_thruster = 8  # number of thrusters

    q_predict = np.zeros((2 * n_dof, N + 1, M))  # predicted trajectory
    if use_pwm:
        u_optimal = np.zeros((n_thruster, M))
    else:
        u_optimal = np.zeros((n_dof, M))  # optimal control inputs
    cost = np.zeros(M)  # cost for each step
    q_real = np.zeros((2 * n_dof, M + 1))  # real
    
    q_real[0:6, 0] = reference_trajectory[:, 0]  # initial state
    # q_real[6:, 0] = np.zeros(n_dof)  # initial twist (zero)
    # Estimate initial twist (velocity) guess from reference trajectory
    initial_twist_guess = (reference_trajectory[:, 1] - reference_trajectory[:, 0]) / dt
    q_real[6:, 0] = initial_twist_guess

    start_time = time.time()  # Startzeit MPC messen

    for step in range(M):
        # Ensure reference_trajectory for the horizon has N columns (pad with last column if needed)
        ref_traj_horizon = reference_trajectory[:, step:step + N]
        if ref_traj_horizon.shape[1] < N:
            last_col = ref_traj_horizon[:, -1].reshape(-1, 1)
            pad = np.repeat(last_col, N - ref_traj_horizon.shape[1], axis=1)
            ref_traj_horizon = np.concatenate([ref_traj_horizon, pad], axis=1)

        if use_pwm:
            q_opt, u_optimal_horizon, Jopt = solve_cftoc_pwm(N, n_dof, symbolic_bluerov, q_real[6:, step], q_real[0:6, step], ref_traj_horizon, dt, n_thruster, V_bat)
        else:
            q_opt, u_optimal_horizon, Jopt = solve_cftoc_tau(N, n_dof, symbolic_bluerov, q_real[6:, step], q_real[0:6, step], ref_traj_horizon, dt)

        q_predict[:, :, step] = q_opt
        u_optimal[:, step] = u_optimal_horizon[:, 0] # this mpc solves for tau as control input
        cost[step] = Jopt
        if use_pwm:
            # in real die u_ESC commands noch umrechnen von PWM [-1, 1] auf tau [1100, 1900]
            q_real[6:, step + 1], q_real[0:6, step + 1], _, _ = bluerov.forward_dynamics_esc(real_bluerov, u_esc = u_optimal[:, step], nu = q_real[6:, step], eta = q_real[0:6, step], dt = dt, V_bat = V_bat)
        else:
            q_real[6:, step + 1], q_real[0:6, step + 1], _ = bluerov.forward_dynamics(real_bluerov, tau = u_optimal[:, step], nu = q_real[6:, step], eta = q_real[0:6, step], dt = dt)

    elapsed_time = time.time() - start_time  # Benötigte Zeit berechnen
    print(f"MPC completed in {elapsed_time:.2f} seconds")

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

    opti.minimize(cost)
    sol = opti.solve()
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

def solve_cftoc_pwm(N, n_dof, symbolic_bluerov, nu0, eta0, reference_trajectory, dt, n_thruster=8, V_bat=16.0):
    opti = ca.Opti()
    opts = {'ipopt.print_level': 0, 'print_time': 0}
    # opts = {}
    opti.solver('ipopt', opts)

    q = opti.variable(2 * n_dof, N + 1)  # vehicle pose (with euler angles) eta and twist nu
    u = opti.variable(n_thruster, N)  # thruster PWM commands (normalized [-1, 1])

    opti.set_initial(q, ca.repmat(ca.vertcat(eta0, nu0), 1, N + 1))
    opti.set_initial(u, 0.5 * ca.DM.ones(n_thruster, N))

    opti.subject_to(q[0:6, 0] == ca.DM(eta0))  # initial pose
    opti.subject_to(q[6:, 0] == ca.DM(nu0))  # initial twist

    for k in range(N):
        for thruster in range(n_thruster):
            opti.subject_to(u[thruster, k] <= 1.0)
            opti.subject_to(u[thruster, k] >= -1.0)

        opti.subject_to(q[3, k] >= -np.pi * 3 / 4)  # phi should be in [-pi/2, pi/2] (roll)
        opti.subject_to(q[3, k] <= np.pi * 3 / 4)
        opti.subject_to(q[4, k] >= -np.pi * 3 /4)  # theta should be in [-pi/2, pi/2] (pitch)
        opti.subject_to(q[4, k] <= np.pi * 3 / 4)

        # vehcile dynamics
        opti.subject_to(q[6:, k+1] == q[6:, k] + dt * ca.mtimes(symbolic_bluerov.M_inv, (symbolic_bluerov.L * ca.DM(V_bat) * ca.mtimes(symbolic_bluerov.mixer, u[:, k]) - ca.mtimes(symbolic_bluerov.C(q[6:, k]), q[6:, k]) - ca.mtimes(symbolic_bluerov.D(q[6:, k]), q[6:, k]) - symbolic_bluerov.g(q[0:6, k]))))
        # vehicle kinematics
        opti.subject_to(q[0:6, k+1] == q[0:6, k] + dt * ca.mtimes(symbolic_bluerov.J(q[0:6,k]), q[6:, k+1]))



    cost = 0

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

        ##########################
        # Indem der attitude error mit quaternions berehcnte wird, habe ich instabilität etwas gelöst
        ##########################

        # Convert Euler angles to quaternion for reference and current state
        ref_quat = euler_to_quat_casadi(reference_trajectory[3, k], reference_trajectory[4, k], reference_trajectory[5, k])
        curr_quat = euler_to_quat_casadi(att_k[0], att_k[1], att_k[2])

        # Quaternion error (shortest arc)
        att_error = quaternion_error(ref_quat, curr_quat)

        pos_error = ca.DM(reference_trajectory[0:3, k]) - pos_k
        # att_error = ca.DM(reference_trajectory[3:6, k]) - att_k # with singularities might not be correct

        cost += ca.sumsqr(pos_error) + ca.sumsqr(att_error)

    opti.minimize(cost)
    sol = opti.solve()
    return sol.value(q), sol.value(u), sol.value(cost)




def main():
    bluerov_params = bluerov.load_model_params('model_params.yaml')
    bluerov_dynamics = bluerov.BlueROVDynamics(bluerov_params)
    bluerov_symbolic = BlueROVDynamicsSymbolic(bluerov_params)

    T = 10.0  # seconds for one period of the sine wave trajectory
    fps = 50  # frames per second
    dt = 1 / fps  # seconds
    n = 1  # number of periods for the sine wave trajectory

    reference_eta = bluerov.generate_sine_on_circle_trajectory_time(T=T, dt = dt, n=n)

    # reference_eta = bluerov.generate_circle_trajectory_time(T=T, dt=dt, n=n)

    # eta_0 = np.array([0, 0, -1.0, 0.0, 0.0, 0])  # Initial pose [x, y, z, phi, theta, psi]
    # nu_0 = np.array([0, 0, 0, 0, 0, 0])  # Initial velocity [u, v, w, p, q, r]
    # tau = np.array([-5.37, 0, 0, 0, 0, 0])  # exemplary wrench on the vehicle

    # eta_all = np.zeros((timesteps + 1, 6))  # Store poses
    # nu_all = np.zeros((timesteps + 1, 6))  # Store velocities

    # eta_all[0, :] = eta_0
    # nu_all[0, :] = nu_0

    # for t in range(timesteps):
    #     nu_all[t+1, :], eta_all[t+1, :], _ = bluerov.forward_dynamics(bluerov_dynamics, tau[t,:], nu_all[t, :], eta_all[t, :], dt)

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

    plt.figure()
    plt.plot(np.arange(cost.shape[0]) * dt, cost)
    plt.xlabel('Time [s]')
    plt.ylabel('MPC Cost')
    plt.title('MPC Cost over Time')
    plt.grid(True)
    plt.show()

    plot_vehicle_pos_vs_reference(reference_eta.T, real_eta)
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