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
        self.TF_i = [np.eye(4) for _ in range(self.n_joints + 1)]
        self.TF_iminus1_i = [np.eye(4) for _ in range(self.n_joints + 1)]
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
        return self.TF_i[-1][:3, 3]

    def get_eef_attitude(self):
        R = self.TF_i[-1][:3, :3]
        m = R.flatten()
        tr = m[0] + m[4] + m[8]
        if tr > 0:
            S = np.sqrt(tr + 1.0) * 2
            qw = 0.25 * S
            qx = (m[5] - m[7]) / S
            qy = (m[2] - m[6]) / S
            qz = (m[3] - m[1]) / S
        elif m[0] > m[4] and m[0] > m[8]:
            S = np.sqrt(1.0 + m[0] - m[4] - m[8]) * 2
            qw = (m[5] - m[7]) / S
            qx = 0.25 * S
            qy = (m[1] + m[3]) / S
            qz = (m[2] + m[6]) / S
        elif m[4] > m[8]:
            S = np.sqrt(1.0 + m[4] - m[0] - m[8]) * 2
            qw = (m[2] - m[6]) / S
            qx = (m[1] + m[3]) / S
            qy = 0.25 * S
            qz = (m[5] + m[7]) / S
        else:
            S = np.sqrt(1.0 + m[8] - m[0] - m[4]) * 2
            qw = (m[3] - m[1]) / S
            qx = (m[2] + m[6]) / S
            qy = (m[5] + m[7]) / S
            qz = 0.25 * S
        return np.array([qw, qx, qy, qz])

    def get_link_position(self, idx):
        return self.TF_i[idx][:3, 3]

    def get_position_jacobian(self):
        J = np.zeros((3, self.n_joints))
        p_eef = self.get_eef_position()
        for i in range(self.n_joints):
            z = self.e3 if i == 0 else self.TF_i[i-1][:3, :3] @ self.e3
            p = np.zeros(3) if i == 0 else self.TF_i[i-1][:3, 3]
            J[:, i] = np.cross(z, p_eef - p)
        return J

    def get_rotation_jacobian(self):
        J = np.zeros((3, self.n_joints))
        for i in range(self.n_joints):
            J[:, i] = self.e3 if i == 0 else self.TF_i[i-1][:3, :3] @ self.e3
        return J

    def get_full_jacobian(self):
        return np.vstack((self.get_position_jacobian(), self.get_rotation_jacobian()))

    def forward_kinematics_symbolic(self, q_sym):
        TF = ca.MX.eye(4)
        for i in range(self.n_joints + 1):
            d, theta_0, a, alpha = map(ca.MX, self.DH_table[i])
            theta = theta_0 + (q_sym[i] if i < self.n_joints else 0)
            TF = ca.mtimes(TF, dh2matrix_sym(d, theta, a, alpha))
        return TF

    @staticmethod
    def rotation_matrix_to_quaternion(R):
        m = [R[0,0], R[0,1], R[0,2], R[1,0], R[1,1], R[1,2], R[2,0], R[2,1], R[2,2]]
        tr = m[0] + m[4] + m[8]
        def case1():
            S = ca.sqrt(tr + 1.0) * 2
            return ca.vertcat(0.25*S, (m[5]-m[7])/S, (m[2]-m[6])/S, (m[3]-m[1])/S)
        def case2():
            S = ca.sqrt(1.0 + m[0] - m[4] - m[8]) * 2
            return ca.vertcat((m[5]-m[7])/S, 0.25*S, (m[1]+m[3])/S, (m[2]+m[6])/S)
        def case3():
            S = ca.sqrt(1.0 + m[4] - m[0] - m[8]) * 2
            return ca.vertcat((m[2]-m[6])/S, (m[1]+m[3])/S, 0.25*S, (m[5]+m[7])/S)
        def case4():
            S = ca.sqrt(1.0 + m[8] - m[0] - m[4]) * 2
            return ca.vertcat((m[3]-m[1])/S, (m[2]+m[6])/S, (m[5]+m[7])/S, 0.25*S)
        return ca.if_else(tr > 0, case1(),
               ca.if_else(ca.logic_and(m[0] > m[4], m[0] > m[8]), case2(),
               ca.if_else(m[4] > m[8], case3(), case4())))
