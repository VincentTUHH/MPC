import numpy as np
import common.utils_math as utils_math
import manipulator.kinematics as manip_kinematics
import manipulator.dynamics as manip_dynamics
import bluerov.dynamics as bluerov_dynamics
import scipy.linalg

class UVMSModel:
    def __init__(self, DH_table, alpha_params, bluerov_params_dynamic):
        kinematics = manip_kinematics.Kinematics(DH_table)
        # Both attributes (kinematics and dynamics) point to the same Kinematics object in memory.
        self.manipulator_kinematics = kinematics
        self.manipulator_dynamics = manip_dynamics.Dynamics(kinematics, alpha_params)

        self.bluerov_dynamics = bluerov_dynamics.BlueROVDynamics(bluerov_params_dynamic)

        self.R_B_0 = self.manipulator_dynamics.R_reference
        self.r_B_0 = self.manipulator_dynamics.tf_vec  # vehicle frame to base link position in world frame
    
        self.r_0_eef = np.zeros(3)
        self.att_0_eef = np.zeros(4)
        self.J_manipulator_pos = np.zeros((3, kinematics.n_joints)) # J_eta_ee,t ^0
        self.J_manipulator_rot = np.zeros((3, kinematics.n_joints)) # J_eta_ee,r ^0

        self.R_I_B = np.eye(3)  # Initial rotation from inertial to body frame
        self.p_eef = np.zeros(3)  # End-effector position in inertial frame
        self.att_eef = np.zeros(4)  # End-effector attitude in inertial frame


    def update(self, q, pos, att):
        """
        Update the model state with new joint angles, position, and attitude.
        :param q: Joint angles (numpy array)
        :param pos: Vehicle position (numpy array, shape (3,)) -> p_I_B expressed in I
        :param att: Vehicle attitude as a scipy.spatial.transform.Rotation or equivalent -> quaternion_I_B expressed in I
        """
        self.manipulator_kinematics.update(q)
        self.r_0_eef = self.manipulator_kinematics.get_eef_position()
        self.att_0_eef = self.manipulator_kinematics.get_eef_attitude() # return quaternion
        self.J_manipulator_pos, self.J_manipulator_rot = self.manipulator_kinematics.get_full_jacobian()

        R_I_B = utils_math.rotation_matrix_from_quat(att)
        self.R_I_B = R_I_B
        self.p_eef = pos + R_I_B @ self.r_B_0 + R_I_B @ self.R_B_0 @ self.r_0_eef
        self.att_eef = utils_math.rotation_matrix_to_quaternion(R_I_B @ self.R_B_0 @ utils_math.rotation_matrix_from_quat(self.att_0_eef))

    def get_eef_jacobian(self):
        n_joints = self.manipulator_kinematics.n_joints
        J = np.zeros((6, 6 + n_joints))
        # Translational part: J_eta_ee,t
        J[0:3, 0:3] = self.R_I_B                                                        # R_B ^I
        vec_tmp = self.R_I_B @ self.r_B_0 + self.R_I_B @ self.R_B_0 @ self.r_0_eef
        skew_tmp = utils_math.skew(vec_tmp)
        J[0:3, 3:6] = -skew_tmp @ self.R_I_B                                            # - S(r_B,ee ^I) * R_B ^I
        J[0:3, 6:6+n_joints] = self.R_I_B @ self.R_B_0 @ self.J_manipulator_pos              # J_eta_ee,t ^I
        # Rotational part: J_eta_ee,r
        # J[3:6, 0:3] = 0                                                               # 0
        J[3:6, 3:6] = self.R_I_B                                                        # R_B ^I
        J[3:6, 6:6+n_joints] = self.R_I_B @ self.R_B_0 @ self.J_manipulator_rot              # J_eta_ee,r ^I
        return J
    
    def get_eef_position(self):
        """
        Get the end-effector position in the inertial frame.
        :return: End-effector position (numpy array, shape (3,))
        """
        return self.p_eef
    
    def get_eef_attitude(self):
        """
        Get the end-effector attitude in the inertial frame.
        :return: End-effector attitude as a quaternion (numpy array, shape (4,))
        """
        return self.att_eef
    
    # End effector trajectory tracking in closed inverse kinematics:
    # generalized_velocities_d = get_eef_jacobian * (v_eef_d + K_eef * (pose_eef_d - pose_eef))
    # where pose_error is in first three component the position error: pos_eef_d - get_eef_position()
    # and the attitude error in vector form for the last three components given: att_eef_d and get_eef_attitude()

    def pseudo_inverse(self, J, tol=1e-6):
        # U, S, Vh = np.linalg.svd(J, full_matrices=False)
        # # Invert singular values with tolerance
        # S_inv = np.array([1/s if s > tol else 0 for s in S]) # mitigate division by zero
        # J_pinv = Vh.T @ np.diag(S_inv) @ U.T
        J_pinv = np.linalg.pinv(J)
        return J_pinv # same result as: np.linalg.pinv(J))

    def uvms_dynamics(self, 
                      command,          # either vehicle wrench tau or ESC commands u_esc 8 thrusters
                      q, dq, ddq,       # joint angles, velocities, and accelerations
                      pos, att,         # eta = [pos, att] in inertial frame
                      lin_vel, lin_acc, # nu = [lin_vel, ang_vel] in body frame
                      ang_vel, ang_acc, # nu_dot = [lin_acc, ang_acc] in body frame
                      f_eef, l_eef,     # end-effector force and torque in end effector frame
                      dt, 
                      use_quat=False, use_pwm=False
                      ):  
        V_bat = 16.0  # Battery voltage

        eta = np.concatenate((pos, att)) 
        nu = np.concatenate((lin_vel, ang_vel))
        nu_dot = np.concatenate((lin_acc, ang_acc))

        xi = np.concatenate((eta, q))
        zeta = np.concatenate((nu, dq))

        tau_coupling = self.manipulator_dynamics.rnem(q, dq, ddq, lin_vel, lin_acc, ang_vel, ang_acc, att, f_eef, l_eef)
        tau_vehicle = self.bluerov_dynamics.mixer @ command * self.L * V_bat if use_pwm else command
        
        M_inv = self.bluerov_dynamics.M_inv
        C = self.bluerov_dynamics.C(nu)
        D = self.bluerov_dynamics.D(nu)
        g = self.bluerov_dynamics.g_quat(eta) if use_quat else self.bluerov_dynamics.g(eta)
        
        J_xi = self.J_xi(eta, use_quat=use_quat)  # Jacobian of UVMS relating xi_dot and nu
        
        nu_dot_next = M_inv @ (tau_vehicle + tau_coupling - C @ nu - D @ nu - g)
        nu_next = nu + dt * nu_dot_next  # Euler forward integration
        xi_next = xi + dt * J_xi @ zeta

        eta_next = xi_next[:7] if use_quat else xi_next[:6]
        q_next = xi_next[7:] if use_quat else xi_next[6:] 

        return q_next, eta_next, nu_next, nu_dot_next, tau_vehicle, tau_coupling
    
    def uvms_kinematics(self, pos, att, use_quat=False):
        eta = np.concatenate((pos, att))

        n_joints = self.manipulator_kinematics.n_joints
        vehicle_dim = 7 if use_quat else 6
        state_dim = vehicle_dim + n_joints # dimension of generalized coordinates vector xi

        J_xi = np.zeros((state_dim, 6 + n_joints))

        J_xi[0:vehicle_dim, 0:vehicle_dim] = self.bluerov_dynamics.J_quat(eta) if use_quat else self.bluerov_dynamics.J(eta)
        J_xi[vehicle_dim:state_dim, vehicle_dim:state_dim] = np.eye(n_joints)

    def J_xi(self, eta, use_quat=False): # Jacobian of uvms relating xi_dot and nu
        n_joints = self.manipulator_kinematics.n_joints
        vehicle_dim = 7 if use_quat else 6
        state_dim = vehicle_dim + n_joints

        J_xi = np.zeros((state_dim, 6 + n_joints))

        J_xi[0:vehicle_dim, 0:vehicle_dim] = self.bluerov_dynamics.J_quat(eta) if use_quat else self.bluerov_dynamics.J(eta)
        J_xi[vehicle_dim:state_dim, vehicle_dim:state_dim] = np.eye(n_joints)


    
    def end_effector_tracking(self, p_eef_d, att_eef_d, v_eef_lin_d, v_eef_rot_d, K_eef=4.0): # K_eef is a diagonal gain matrix of size 6x6 for the position error and attitude error in vector form, 4.0 for all in Trekel23
        J_eef = self.get_eef_jacobian()
        J_eef_pinv = self.pseudo_inverse(J_eef)

        pose_error = np.zeros(6)
        pose_error[:3] = p_eef_d - self.p_eef
        pose_error[3:] = utils_math.quaternion_error(att_eef_d, self.att_eef)

        vel_eef_d = np.concatenate((v_eef_lin_d, v_eef_rot_d))

        zeta_d = J_eef_pinv @ (vel_eef_d + K_eef * pose_error) # desired generalized velocities

        return zeta_d
