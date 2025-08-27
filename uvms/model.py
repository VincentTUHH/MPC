import numpy as np
import common.utils_math as utils_math
import manipulator.kinematics as manip_kinematics
import manipulator.dynamics as manip_dynamics
import bluerov.dynamics as bluerov_dynamics
import scipy.linalg

class UVMSModel:
    def __init__(self, DH_table, alpha_params, bluerov_params_dynamic, q0, pos, att, vel, omega):
        # Das Modell geht davon aus, dass die Vehicle Orientierung in Euler Winkeln angegeben ist.
        kinematics = manip_kinematics.Kinematics(DH_table)
        # Both attributes (kinematics and dynamics) point to the same Kinematics object in memory.
        self.manipulator_kinematics = kinematics
        self.manipulator_dynamics = manip_dynamics.Dynamics(kinematics, alpha_params)

        self.bluerov_dynamics = bluerov_dynamics.BlueROVDynamics(bluerov_params_dynamic)

        self.R_B_0 = self.manipulator_dynamics.R_reference
        self.r_B_0 = self.manipulator_dynamics.tf_vec  # vehicle frame to base link position in world frame
    
        self.r_0_eef = np.zeros(3)
        self.att_0_eef = np.zeros(4)
        self.J_manipulator_pos = np.zeros((3, self.manipulator_kinematics.n_joints)) # J_eta_ee,t ^0
        self.J_manipulator_rot = np.zeros((3, self.manipulator_kinematics.n_joints)) # J_eta_ee,r ^0

        self.R_I_B = np.eye(3)  # Initial rotation from inertial to body frame
        self.p_eef = np.zeros(3)  # End-effector position in inertial frame
        self.att_eef = np.zeros(4)  # End-effector attitude in inertial frame

        self.joint_positions = np.zeros((self.manipulator_kinematics.n_joints + 2, 3))  # Joint positions including base

        self.eta = np.concatenate((pos, att))  # Vehicle pose: position + attitude
        self.nu = np.concatenate((vel, omega))  # Body velocities in body frame

        self.q = q0  # Joint angles
        self.manipulator_kinematics.update(self.q)
        self.update_joint_positions()

        self.last_uq = np.zeros(self.manipulator_kinematics.n_joints)  # Last joint velocities
        self.last_dnu = np.zeros(6)  # Last body velocities

        self.f_eef = np.zeros(3)  # End-effector force in end-effector frame
        self.l_eef = np.zeros(3)  # End-effector torque in end-effector frame
        # print(f"real - eta    : {self.eta}")
        # print(f"real - nu    : {self.nu.flatten()}")



    def update(self, dt, uq, uv, use_pwm, V_bat):
        self.manipulator_dynamics.kinematics_.update(self.q)
        ddq = (uq - self.last_uq) / dt
        self.last_uq = uq

        att = utils_math.euler_to_quat(self.eta[3], self.eta[4], self.eta[5])
        tau_coupling = self.manipulator_dynamics.rnem(self.q, uq, ddq, self.nu[:3], self.last_dnu[:3], 
                                                     self.nu[3:], self.last_dnu[3:], att, 
                                                     self.f_eef, self.l_eef)
        # print("tau_coupling:", tau_coupling)
        # print("rnem inputs:")
        # print("  q:", self.q)
        # print("  uq:", uq)
        # print("  ddq:", ddq)
        # print("  nu[:3] (lin_vel):", self.nu[:3])
        # print("  last_dnu[:3] (lin_acc):", self.last_dnu[:3])
        # print("  nu[3:] (ang_vel):", self.nu[3:])
        # print("  last_dnu[3:] (ang_acc):", self.last_dnu[3:])
        # print("  att (quat):", att)
        # print("  f_eef:", self.f_eef)
        # print("  l_eef:", self.l_eef)
        
        if use_pwm:
            tau_vehicle = self.bluerov_dynamics.mixer @ uv * self.bluerov_dynamics.L * V_bat  # convert ESC signals to thrusts
        else:
            tau_vehicle = uv
        
        self.last_dnu = np.linalg.inv(self.bluerov_dynamics.M) @ (tau_vehicle + tau_coupling - 
                                          self.bluerov_dynamics.C(self.nu) @ self.nu - 
                                          self.bluerov_dynamics.D(self.nu) @ self.nu - 
                                          self.bluerov_dynamics.g(self.eta))
        
        self.eta = self.eta + dt * self.bluerov_dynamics.J(self.eta) @ self.nu
        self.nu = self.nu + dt * self.last_dnu
        
        self.q = self.q + dt * uq
        
        

        # self.r_0_eef = self.manipulator_kinematics.get_eef_position()
        # self.att_0_eef = self.manipulator_kinematics.get_eef_attitude() # return quaternion
        # self.J_manipulator_pos, self.J_manipulator_rot = self.manipulator_kinematics.get_full_jacobian()

        R_I_B = utils_math.rotation_matrix_from_euler(self.eta[3], self.eta[4], self.eta[5])
        self.R_I_B = R_I_B

        self.manipulator_dynamics.kinematics_.update(self.q)

        self.update_joint_positions()
        # self.p_eef = self.eta[0:3] + R_I_B @ self.r_B_0 + R_I_B @ self.R_B_0 @ self.r_0_eef
        # self.att_eef = utils_math.rotation_matrix_to_quaternion(R_I_B @ self.R_B_0 @ utils_math.rotation_matrix_from_quat(self.att_0_eef))

        # print(f"real - ddq: {ddq}")
        # print(f"real - eta    : {self.eta}")
        # print(f"real - nu     : {self.nu}")
        # print(f"real - q    : {self.q}")
        # print(f"real -  joint 0 position   : {self.joint_positions[0].flatten()}")
        # print(f"real - nu    : {self.nu.flatten()}")
        # print(f"real - dnu    : {self.last_dnu.flatten()}")
        # print(f"real - tau_coupling    : {tau_coupling.flatten()}")

    def get_next_state(self, use_quaternion):
        if use_quaternion:
            return np.concatenate((self.q, self.nu, self.eta[0:3], utils_math.euler_to_quat(self.eta[3], self.eta[4], self.eta[5])))
        else:
            return np.concatenate((self.q, self.nu, self.eta))
            

    def update_joint_positions(self):
        self.joint_positions[0] = (self.eta[0:3] + self.R_I_B @ self.r_B_0).flatten() # joint 0

        for i in range(self.manipulator_kinematics.n_joints + 1):
            self.joint_positions[i+1] = self.joint_positions[0] + self.R_I_B @ self.R_B_0 @ self.manipulator_kinematics.get_link_position(i).flatten()

    def get_uvms_configuration(self):
        return [self.eta, self.joint_positions]

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

