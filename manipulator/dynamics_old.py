import numpy as np
from common import utils_math

class Dynamics:
    def __init__(self, DH_table, alpha_params=None):
        self.DH_table = np.array(DH_table)
        self.n_joints = self.DH_table.shape[0] - 1
        self.q0 = self.DH_table[:, 1].copy()
        self.e3 = np.array([0, 0, 1])
        self.TF_i = [np.eye(4) for _ in range(self.n_joints + 1)]
        self.TF_iminus1_i = [np.eye(4) for _ in range(self.n_joints + 1)]
        self.update(np.zeros(self.n_joints))

        if alpha_params is not None:
            self.n_links = len(alpha_params.__dict__)
            self.r_i_1_i = np.zeros((self.n_links, 3))
            self.R_reference = utils_math.rotation_matrix_from_euler(
                alpha_params.link_0.r, alpha_params.link_0.p, alpha_params.link_0.y
            )
            tf_vec = np.array([
                alpha_params.link_0.vec.x,
                alpha_params.link_0.vec.y,
                alpha_params.link_0.vec.z
            ])
            self.r_i_1_i[0] = self.R_reference.T @ tf_vec

            for i in range(1, self.n_links):
                self.r_i_1_i[i] = self.get_DH_link_offset(*self.DH_table[i-1])

            self.inertial = [getattr(alpha_params, f'link_{i}').inertial for i in range(self.n_links)]
            self.active = [getattr(alpha_params, f'link_{i}').active for i in range(self.n_links)]
            self.dh_link = [getattr(alpha_params, f'link_{i}').dh_link for i in range(self.n_links)]

            self.I = [self.inertia_to_matrix(getattr(alpha_params, f'link_{i}').I) if self.inertial[i] else None for i in range(self.n_links)]
            self.M_a = [self.hyd_mass_to_transl_matrix(getattr(alpha_params, f'link_{i}').hyd.mass) if self.inertial[i] else None for i in range(self.n_links)]
            self.M12 = [self.hyd_mass_to_M12(getattr(alpha_params, f'link_{i}').hyd.mass) if self.inertial[i] else None for i in range(self.n_links)]
            self.M21 = [self.hyd_mass_to_M21(getattr(alpha_params, f'link_{i}').hyd.mass) if self.inertial[i] else None for i in range(self.n_links)]
            self.I_a = [self.hyd_mass_to_inertia_matrix(getattr(alpha_params, f'link_{i}').hyd.mass) if self.inertial[i] else None for i in range(self.n_links)]
            self.m_buoy = [getattr(alpha_params, f'link_{i}').hyd.buoy.hyd_mass if self.inertial[i] else None for i in range(self.n_links)]
            self.r_b = [self.get_center_of_buoyancy(getattr(alpha_params, f'link_{i}').hyd.buoy.cob) if self.inertial[i] else None for i in range(self.n_links)]
            self.r_c = [self.get_center_of_gravity(getattr(alpha_params, f'link_{i}').cog) if self.inertial[i] else None for i in range(self.n_links)]
            self.r_cb = [self.r_b[i] - self.r_c[i] if self.inertial[i] else None for i in range(self.n_links)]
            self.m = [getattr(alpha_params, f'link_{i}').mass if self.inertial[i] else None for i in range(self.n_links)]
            self.lin_damp_param = [self.get_lin_damp_params(getattr(alpha_params, f'link_{i}').hyd.lin_damp) if self.inertial[i] else None for i in range(self.n_links)]
            self.nonlin_damp_param = [self.get_nonlin_damp_params(getattr(alpha_params, f'link_{i}').hyd.nonlin_damp) if self.inertial[i] else None for i in range(self.n_links)]
            self.D_t = [np.zeros((3, 3)) if self.inertial[i] else None for i in range(self.n_links)]
            self.D_r = [np.zeros((3, 3)) if self.inertial[i] else None for i in range(self.n_links)]

            self.f = [None] * self.n_links
            self.l = [None] * self.n_links
            self.v = [None] * self.n_links
            self.a = [None] * self.n_links
            self.dv_c = [None] * self.n_links
            self.v_c = [None] * self.n_links
            self.w = [None] * self.n_links
            self.dw = [None] * self.n_links
            self.a_c = [None] * self.n_links
            self.g = [None] * self.n_links
            self.v_b = [None] * self.n_links

    @staticmethod
    def inertia_to_matrix(params):
        return np.array([
            [params.xx, params.xy, params.xz],
            [params.xy, params.yy, params.yz],
            [params.xz, params.yz, params.zz]
        ])

    @staticmethod
    def hyd_mass_to_transl_matrix(params):
        return np.diag([-params.X_dotu, -params.Y_dotv, -params.Z_dotw])

    @staticmethod
    def hyd_mass_to_M12(params):
        return np.array([
            [-params.X_dotp, -params.X_dotq, -params.X_dotr],
            [-params.Y_dotp, -params.Y_dotq, -params.Y_dotr],
            [-params.Z_dotp, -params.Z_dotq, -params.Z_dotr]
        ])

    @staticmethod
    def hyd_mass_to_M21(params):
        return np.array([
            [-params.K_dotu, -params.K_dotv, -params.K_dotw],
            [-params.M_dotu, -params.M_dotv, -params.M_dotw],
            [-params.N_dotu, -params.N_dotv, -params.N_dotw]
        ])

    @staticmethod
    def hyd_mass_to_inertia_matrix(params):
        return np.array([
            [-params.K_dotp, -params.K_dotq, -params.K_dotr],
            [-params.M_dotp, -params.M_dotq, -params.M_dotr],
            [-params.N_dotp, -params.N_dotq, -params.N_dotr]
        ])

    @staticmethod
    def get_center_of_buoyancy(params):
        return np.array([params.x, params.y, params.z])

    @staticmethod
    def get_center_of_gravity(params):
        return np.array([params.x, params.y, params.z])

    @staticmethod
    def get_lin_damp_params(params):
        return np.array([
            params.X_u, params.Y_v, params.Z_w,
            params.K_p, params.M_q, params.N_r
        ])

    @staticmethod
    def get_nonlin_damp_params(params):
        return np.array([
            params.X_absuu, params.Y_absvv, params.Z_absww,
            params.K_abspp, params.M_absqq, params.N_absrr
        ])

    @staticmethod
    def get_DH_link_offset(d, theta, a, alpha):
        R_alpha = np.array([
            [1, 0, 0],
            [0, np.cos(alpha), -np.sin(alpha)],
            [0, np.sin(alpha),  np.cos(alpha)]
        ])
        return R_alpha.T @ np.array([a, 0, d])
    
    def get_current_kinematics(self):
        return {
            'v': self.v, 'a': self.a, 'w': self.w, 'dw': self.dw,
            'g': self.g, 'a_c': self.a_c, 'v_c': self.v_c,
            'dv_c': self.dv_c, 'v_b': self.v_b
        }

    def get_current_dynamic_parameters(self):
        return {
            'I': self.I, 'M_a': self.M_a, 'M12': self.M12, 'M21': self.M21,
            'I_a': self.I_a, 'm_buoy': self.m_buoy, 'r_c': self.r_c,
            'r_b': self.r_b, 'm': self.m, 'lin_damp_param': self.lin_damp_param,
            'nonlin_damp_param': self.nonlin_damp_param, 'D_t': self.D_t,
            'D_r': self.D_r, 'r_cb': self.r_cb
        }

    def get_current_wrench(self):
        return {'f': self.f, 'l': self.l}

    def get_R_reference(self):
        return self.R_reference
    
    def get_number_of_links(self):
        return self.n_links

    def get_reference_wrench(self):
        R_T = self.R_reference.T
        return -R_T @ self.f[0], -R_T @ self.l[0]
    
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
        if idx < 1 or idx > self.n_joints+1:
            raise IndexError(f"Index {idx} out of bounds for n_joints {self.n_joints}")
        return self.TF_iminus1_i[idx-1][:3, :3]

    def forward(self, v_ref, a_ref, w_ref, dw_ref, g_ref):
        R_T = self.R_reference.T
        w0 = R_T @ w_ref
        dw0 = R_T @ dw_ref
        g0 = R_T @ g_ref
        v0 = R_T @ v_ref + utils_math.skew(w0) @ self.r_i_1_i[0]
        a0 = R_T @ a_ref + utils_math.skew(dw0) @ self.r_i_1_i[0] + utils_math.skew(w0) @ (utils_math.skew(w0) @ self.r_i_1_i[0])
        self.w[0], self.dw[0], self.g[0], self.v[0], self.a[0] = w0, dw0, g0, v0, a0
        self.v_b[0] = v0 + utils_math.skew(w0) @ self.r_b[0]
        self.v_c[0] = v0 + utils_math.skew(w0) @ self.r_c[0]
        self.a_c[0] = a0 + utils_math.skew(dw0) @ self.r_c[0] + utils_math.skew(w0) @ (utils_math.skew(w0) @ self.r_c[0])
        self.dv_c[0] = self.a_c[0] - utils_math.skew(w0) @ self.v_c[0]

    def link_forward(self, idx, q, dq, ddq):
        R_T = self.get_rotation_iminus1_i(idx).T
        prev_w, prev_dw, prev_g, prev_v, prev_a = self.w[idx-1], self.dw[idx-1], self.g[idx-1], self.v[idx-1], self.a[idx-1]
        r_i = self.r_i_1_i[idx]
        if not self.active[idx]:
            w = R_T @ prev_w
            dw = R_T @ prev_dw
            g = R_T @ prev_g
        else:
            w = R_T @ prev_w + dq * R_T[:,2]
            dw = R_T @ prev_dw + ddq * R_T[:,2] + dq * utils_math.skew(R_T @ prev_w) @ R_T[:,2]
            g = R_T @ prev_g
        v = R_T @ prev_v + utils_math.skew(w) @ r_i
        a = R_T @ prev_a + utils_math.skew(dw) @ r_i + utils_math.skew(w) @ (utils_math.skew(w) @ r_i)
        self.w[idx], self.dw[idx], self.g[idx], self.v[idx], self.a[idx] = w, dw, g, v, a

        if self.inertial[idx]:
            self.v_b[idx] = v + utils_math.skew(w) @ self.r_b[idx]
            self.v_c[idx] = v + utils_math.skew(w) @ self.r_c[idx]
            self.a_c[idx] = a + utils_math.skew(dw) @ self.r_c[idx] + utils_math.skew(w) @ (utils_math.skew(w) @ self.r_c[idx])
            self.dv_c[idx] = self.a_c[idx] - utils_math.skew(w) @ self.v_c[idx]

    def backward(self, f_eef, l_eef):
        self.f[-1] = f_eef
        self.l[-1] = -utils_math.skew(f_eef) @ self.r_i_1_i[-1] + l_eef

    def link_backward(self, idx):
        f_next = self.get_rotation_iminus1_i(idx+1) @ self.f[idx+1]
        l_next = self.get_rotation_iminus1_i(idx+1) @ self.l[idx+1]
        if self.inertial[idx]:
            v_b_abs = np.abs(self.v_b[idx])
            w_abs = np.abs(self.w[idx])
            D_t = -np.diag(self.nonlin_damp_param[idx][:3] * v_b_abs + self.lin_damp_param[idx][:3])
            D_r = -np.diag(self.nonlin_damp_param[idx][3:] * w_abs + self.lin_damp_param[idx][3:])
            self.D_t[idx], self.D_r[idx] = D_t, D_r

            f_a = (
                self.M_a[idx] @ self.dv_c[idx]
                + self.M12[idx] @ self.dw[idx]
                + utils_math.skew(self.w[idx]) @ (self.M_a[idx] @ self.v_c[idx] + self.M12[idx] @ self.w[idx])
                + D_t @ self.v_b[idx]
                + self.m_buoy[idx] * self.g[idx]
            )
            l_a = (
                self.I_a[idx] @ self.dw[idx]
                + self.M21[idx] @ self.dv_c[idx]
                + utils_math.skew(self.v_c[idx]) @ (self.M_a[idx] @ self.v_c[idx] + self.M12[idx] @ self.w[idx])
                + utils_math.skew(self.w[idx]) @ (self.M21[idx] @ self.v_c[idx] + self.I_a[idx] @ self.w[idx])
                + utils_math.skew(self.r_cb[idx]) @ (D_t @ self.v_b[idx] + self.m_buoy[idx] * self.g[idx])
                + D_r @ self.w[idx]
            )
            self.f[idx] = f_next + self.m[idx] * self.a_c[idx] - self.m[idx] * self.g[idx] + f_a
            self.l[idx] = (
                -utils_math.skew(self.f[idx]) @ (self.r_i_1_i[idx] + self.r_c[idx])
                + l_next
                + utils_math.skew(f_next) @ self.r_c[idx]
                + self.I[idx] @ self.dw[idx]
                + utils_math.skew(self.w[idx]) @ (self.I[idx] @ self.w[idx])
                + l_a
            )
        else:
            self.f[idx] = f_next
            self.l[idx] = -utils_math.skew(self.f[idx]) @ self.r_i_1_i[idx] + l_next
