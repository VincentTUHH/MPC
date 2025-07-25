import numpy as np
import casadi as ca

def dh2matrix(d, theta, a, alpha):
    ct, st = np.cos(theta), np.sin(theta)
    ca_, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st*ca_,  st*sa, a*ct],
        [st,  ct*ca_, -ct*sa, a*st],
        [0,      sa,     ca_,    d],
        [0,       0,      0,    1]
    ])

def dh2matrix_sym(d, theta, a, alpha):
    ct, st = ca.cos(theta), ca.sin(theta)
    ca_, sa = ca.cos(alpha), ca.sin(alpha)
    return ca.vertcat(
        ca.horzcat(ct, -st*ca_,  st*sa, a*ct),
        ca.horzcat(st,  ct*ca_, -ct*sa, a*st),
        ca.horzcat(0,      sa,     ca_,    d),
        ca.horzcat(0,       0,      0,    1)
    )

class Kinematics:
    def __init__(self, DH_table):
        self.DH_table = np.array(DH_table)
        self.n_joints = self.DH_table.shape[0] - 1
        self.q0 = self.DH_table[:, 1].copy()
        self.e3 = np.array([0, 0, 1])
        self.TF_i = [np.zeros((4, 4)) for _ in range(self.n_joints + 1)]
        self.TF_iminus1_i = [np.zeros((4, 4)) for _ in range(self.n_joints + 1)]
        self.update(np.zeros(self.n_joints))

    def updateDHTable(self, q):
        self.DH_table[:self.n_joints, 1] = self.q0[:self.n_joints] + q

    def update(self, q):
        q = np.asarray(q)
        if q.shape[0] != self.n_joints:
            raise ValueError(f"Expected q of length {self.n_joints}, got {q.shape[0]}")
        self.updateDHTable(q)
        self.TF_i[0] = dh2matrix(*self.DH_table[0])
        for i in range(1, self.n_joints + 1):
            self.TF_iminus1_i[i] = dh2matrix(*self.DH_table[i])
            self.TF_i[i] = self.TF_i[i-1] @ self.TF_iminus1_i[i]

    def get_eef_position(self):
        return np.array(self.TF_i[-1][:3, 3])

    def get_eef_attitude(self):
        R = self.TF_i[-1][:3, :3]
        m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
        m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
        m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]
        tr = m00 + m11 + m22

        if tr > 0:
            S = np.sqrt(tr + 1.0) * 2  # S=4*qw
            qw = 0.25 * S
            qx = (m21 - m12) / S
            qy = (m02 - m20) / S
            qz = (m10 - m01) / S
        elif (m00 > m11) and (m00 > m22):
            S = np.sqrt(1.0 + m00 - m11 - m22) * 2  # S=4*qx
            qw = (m21 - m12) / S
            qx = 0.25 * S
            qy = (m01 + m10) / S
            qz = (m02 + m20) / S
        elif m11 > m22:
            S = np.sqrt(1.0 + m11 - m00 - m22) * 2  # S=4*qy
            qw = (m02 - m20) / S
            qx = (m01 + m10) / S
            qy = 0.25 * S
            qz = (m12 + m21) / S
        else:
            S = np.sqrt(1.0 + m22 - m00 - m11) * 2  # S=4*qz
            qw = (m10 - m01) / S
            qx = (m02 + m20) / S
            qy = (m12 + m21) / S
            qz = 0.25 * S
        quat = np.array([qw, qx, qy, qz])
        return quat

    def get_link_position(self, idx):
        return self.TF_i[idx][:3, 3]

    def get_position_jacobian(self):
        J = np.zeros((3, self.n_joints))
        p_eef = self.get_eef_position()
        for i in range(self.n_joints):
            if i == 0:
                J[:, i] = np.cross(self.e3, p_eef)
            else:
                z = self.TF_i[i-1][:3, :3] @ self.e3
                p = self.TF_i[i-1][:3, 3]
                J[:, i] = np.cross(z, p_eef - p)
        return J
    
    def get_rotation_jacobian(self):
        J = np.zeros((3, self.n_joints))
        for i in range(self.n_joints):
            if i == 0:
                J[:, i] = self.e3
            else:
                J[:, i] = self.TF_i[i-1][:3, :3] @ self.e3
        return J

    def get_full_jacobian(self):
        return np.vstack((self.get_position_jacobian(), self.get_rotation_jacobian()))

    def forward_kinematics_symbolic(self, q_sym):
        TF = ca.MX.eye(4)
        for i in range(self.n_joints + 1):
            d       = ca.MX(self.DH_table[i, 0])
            theta_0 = ca.MX(self.DH_table[i, 1])
            a       = ca.MX(self.DH_table[i, 2])
            alpha   = ca.MX(self.DH_table[i, 3]) # muss ich DH table nicht auch updaten?
            theta = theta_0 + (q_sym[i] if i < self.n_joints else 0) # die condition vllt falsch
            TF_i = dh2matrix_sym(d, theta, a, alpha)
            TF = ca.mtimes(TF, TF_i)
        return TF

    @staticmethod
    def rotation_matrix_to_quaternion(R):
        m00, m01, m02 = R[0,0], R[0,1], R[0,2]
        m10, m11, m12 = R[1,0], R[1,1], R[1,2]
        m20, m21, m22 = R[2,0], R[2,1], R[2,2]

        tr = m00 + m11 + m22

        # Symbolic branch selection (CasADi does not support regular if-statements for symbolic data)
        def quat_case1():
            S = ca.sqrt(tr + 1.0) * 2
            qw = 0.25 * S
            qx = (m21 - m12) / S
            qy = (m02 - m20) / S
            qz = (m10 - m01) / S
            return ca.vertcat(qw, qx, qy, qz)

        def quat_case2():
            S = ca.sqrt(1.0 + m00 - m11 - m22) * 2
            qw = (m21 - m12) / S
            qx = 0.25 * S
            qy = (m01 + m10) / S
            qz = (m02 + m20) / S
            return ca.vertcat(qw, qx, qy, qz)

        def quat_case3():
            S = ca.sqrt(1.0 + m11 - m00 - m22) * 2
            qw = (m02 - m20) / S
            qx = (m01 + m10) / S
            qy = 0.25 * S
            qz = (m12 + m21) / S
            return ca.vertcat(qw, qx, qy, qz)

        def quat_case4():
            S = ca.sqrt(1.0 + m22 - m00 - m11) * 2
            qw = (m10 - m01) / S
            qx = (m02 + m20) / S
            qy = (m12 + m21) / S
            qz = 0.25 * S
            return ca.vertcat(qw, qx, qy, qz)

        # Use CasADi symbolic `if_else` to select branch
        quat = ca.if_else(tr > 0, quat_case1(),
                ca.if_else(ca.logic_and(m00 > m11, m00 > m22), quat_case2(),
                ca.if_else(m11 > m22, quat_case3(), quat_case4())))
        
        return quat
