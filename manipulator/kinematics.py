import numpy as np
import common.utils_math as utils_math

class Kinematics:
    def __init__(self, DH_table):
        self.DH_table = np.array(DH_table)
        self.n_joints = self.DH_table.shape[0] - 1
        self.q0 = self.DH_table[:, 1].copy()
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
        self.TF_i[0] = utils_math.dh2matrix(*self.DH_table[0])
        self.TF_iminus1_i[0] = self.TF_i[0]
        for i in range(1, self.n_joints + 1):
            self.TF_iminus1_i[i] = utils_math.dh2matrix(*self.DH_table[i])
            self.TF_i[i] = self.TF_i[i-1] @ self.TF_iminus1_i[i]

    def get_rotation_iminus1_i(self, idx):
        return self.TF_iminus1_i[idx-1][:3, :3]

    def get_eef_position(self):
        return np.array(self.TF_i[-1][:3, 3])

    def get_eef_attitude(self):
        R = self.TF_i[-1][:3, :3]
        return utils_math.rotation_matrix_to_quaternion(R)

    def get_link_position(self, idx):
        return self.TF_i[idx][:3, 3]

    def get_position_jacobian(self):
        J = np.zeros((3, self.n_joints))
        p_eef = self.get_eef_position()
        for i in range(self.n_joints):
            if i == 0:
                J[:, i] = np.cross(utils_math.UNIT_Z, p_eef)
            else:
                z = self.TF_i[i-1][:3, :3] @ utils_math.UNIT_Z
                p = self.TF_i[i-1][:3, 3]
                J[:, i] = np.cross(z, p_eef - p)
        return J
    
    def get_rotation_jacobian(self):
        J = np.zeros((3, self.n_joints))
        for i in range(self.n_joints):
            if i == 0:
                J[:, i] = utils_math.UNIT_Z
            else:
                J[:, i] = self.TF_i[i-1][:3, :3] @ utils_math.UNIT_Z
        return J

    def get_full_jacobian(self):
        return [self.get_position_jacobian(), self.get_rotation_jacobian()]