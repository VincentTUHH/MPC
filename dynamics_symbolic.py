import casadi as ca
import numpy as np

def dh2matrix_sym(d, theta, a, alpha):
    ct, st = ca.cos(theta), ca.sin(theta)
    ca_, sa = ca.cos(alpha), ca.sin(alpha)
    return ca.vertcat(
        ca.horzcat(ct, -st*ca_,  st*sa, a*ct),
        ca.horzcat(st,  ct*ca_, -ct*sa, a*st),
        ca.horzcat(0,      sa,     ca_,    d),
        ca.horzcat(0,       0,      0,    1)
    )

# all variables used together with the optimization variable must be casadi variabeles
# variables taht are constant should be DM as they are faster to evaluate
# all variables whose content depends on the optimization varaible at some point must be MX

# if logic works as long as the variables are no symbolic casadi variables
# casadi builds a satic expression graph, that is hard to differnetiate and optimize as if-conditions
# change the structure of the graph at run-time that leads to problems / is not supported
# if-conditions using standard Python change the expression at code-generation time (before Casadi knows it)

class DynamicsSymbolic:
    def __init__(self, this_DH_table, alpha_params):
        DH_table = np.array(this_DH_table)
        self.DH_table = ca.MX(DH_table)
        self.n_joints = DH_table.shape[0] - 1
        self.q0 = ca.DM(DH_table[:, 1])
        self.e3 = ca.vertcat(0, 0, 1) # 3x1 vector, returns DM, as all elements are python numerics
        self.TF_iminus1_i = [ca.MX.eye(4) for _ in range(self.n_joints + 1)]
        self.update(ca.MX.zeros(self.n_joints, 1))

        if alpha_params is not None:
            
            self.n_links = len(alpha_params.__dict__)
            self.r_i_1_i = [ca.DM.zeros(3, 1) for _ in range(self.n_links)]
            self.R_reference = self.rpy_to_matrix(
                alpha_params.link_0.r, alpha_params.link_0.p, alpha_params.link_0.y
            ) # return DM, as all elements are python numerics
            tf_vec = ca.vertcat(
                alpha_params.link_0.vec.x,
                alpha_params.link_0.vec.y,
                alpha_params.link_0.vec.z
            ) # 3x1 vector, returns DM, as all elements are python numerics
            self.r_i_1_i[0] = self.R_reference.T @ tf_vec

            self.GRAVITY = ca.DM([0, 0, -9.81])  # 3x1 vector, returns DM, as all elements are python numerics

            for i in range(1, self.n_links):
                self.r_i_1_i[i] = self.get_DH_link_offset(*DH_table[i-1]) # must return DM, as self.r_i_1_i[i] is a list of casadi variables

            self.inertial = [getattr(alpha_params, f'link_{i}').inertial for i in range(self.n_links)]
            self.active = [getattr(alpha_params, f'link_{i}').active for i in range(self.n_links)]
            self.dh_link = [getattr(alpha_params, f'link_{i}').dh_link for i in range(self.n_links)]

            self.I = [self.inertia_to_matrix(getattr(alpha_params, f'link_{i}').I) if self.inertial[i] else ca.DM.zeros(3,3) for i in range(self.n_links)]
            self.M_a = [self.hyd_mass_to_transl_matrix(getattr(alpha_params, f'link_{i}').hyd.mass) if self.inertial[i] else ca.DM.zeros(3,3) for i in range(self.n_links)]
            self.M12 = [self.hyd_mass_to_M12(getattr(alpha_params, f'link_{i}').hyd.mass) if self.inertial[i] else ca.DM.zeros(3,3) for i in range(self.n_links)]
            self.M21 = [self.hyd_mass_to_M21(getattr(alpha_params, f'link_{i}').hyd.mass) if self.inertial[i] else ca.DM.zeros(3,3) for i in range(self.n_links)]
            self.I_a = [self.hyd_mass_to_inertia_matrix(getattr(alpha_params, f'link_{i}').hyd.mass) if self.inertial[i] else ca.DM.zeros(3,3) for i in range(self.n_links)]
            self.m_buoy = [getattr(alpha_params, f'link_{i}').hyd.buoy.hyd_mass if self.inertial[i] else ca.DM.zeros(1,1) for i in range(self.n_links)]
            self.r_b = [self.get_center_of_buoyancy(getattr(alpha_params, f'link_{i}').hyd.buoy.cob) if self.inertial[i] else ca.DM.zeros(3,1) for i in range(self.n_links)]
            self.r_c = [self.get_center_of_gravity(getattr(alpha_params, f'link_{i}').cog) if self.inertial[i] else ca.DM.zeros(3,1) for i in range(self.n_links)]
            self.r_cb = [self.r_b[i] - self.r_c[i] if self.inertial[i] else ca.DM.zeros(3,1) for i in range(self.n_links)]
            self.m = [getattr(alpha_params, f'link_{i}').mass if self.inertial[i] else ca.DM.zeros(1,1) for i in range(self.n_links)]
            self.lin_damp_param = [self.get_lin_damp_params(getattr(alpha_params, f'link_{i}').hyd.lin_damp) if self.inertial[i] else ca.DM.zeros(6,1) for i in range(self.n_links)]
            self.nonlin_damp_param = [self.get_nonlin_damp_params(getattr(alpha_params, f'link_{i}').hyd.nonlin_damp) if self.inertial[i] else ca.DM.zeros(6,1) for i in range(self.n_links)]

            self.f = [ca.MX.zeros(3, 1) for _ in range(self.n_links)]
            self.l = [ca.MX.zeros(3, 1) for _ in range(self.n_links)]
            self.v = [ca.MX.zeros(3, 1) for _ in range(self.n_links)]
            self.a = [ca.MX.zeros(3, 1) for _ in range(self.n_links)]
            self.dv_c = [ca.MX.zeros(3, 1) for _ in range(self.n_links)]
            self.v_c = [ca.MX.zeros(3, 1) for _ in range(self.n_links)]
            self.w = [ca.MX.zeros(3, 1) for _ in range(self.n_links)]
            self.dw = [ca.MX.zeros(3, 1) for _ in range(self.n_links)]
            self.a_c = [ca.MX.zeros(3, 1) for _ in range(self.n_links)]
            self.g = [ca.MX.zeros(3, 1) for _ in range(self.n_links)]
            self.v_b = [ca.MX.zeros(6, 1) for _ in range(self.n_links)]

    @staticmethod
    def inertia_to_matrix(params):
        I = np.array([
            [params.xx, params.xy, params.xz],
            [params.xy, params.yy, params.yz],
            [params.xz, params.yz, params.zz]
        ])
        return ca.DM(I)

    @staticmethod
    def hyd_mass_to_transl_matrix(params):
        Ma = np.diag([-params.X_dotu, -params.Y_dotv, -params.Z_dotw])
        return ca.DM(Ma)

    @staticmethod
    def hyd_mass_to_M12(params):
        M12 = np.array([
            [-params.X_dotp, -params.X_dotq, -params.X_dotr],
            [-params.Y_dotp, -params.Y_dotq, -params.Y_dotr],
            [-params.Z_dotp, -params.Z_dotq, -params.Z_dotr]
        ])
        return ca.DM(M12)

    @staticmethod
    def hyd_mass_to_M21(params):
        return ca.vertcat(
            ca.horzcat(-params.K_dotu, -params.K_dotv, -params.K_dotw),
            ca.horzcat(-params.M_dotu, -params.M_dotv, -params.M_dotw),
            ca.horzcat(-params.N_dotu, -params.N_dotv, -params.N_dotw)
        )

    @staticmethod
    def hyd_mass_to_inertia_matrix(params):
        M21 = np.array([
            [-params.K_dotu, -params.K_dotv, -params.K_dotw],
            [-params.M_dotu, -params.M_dotv, -params.M_dotw],
            [-params.N_dotu, -params.N_dotv, -params.N_dotw]
        ])
        return ca.DM(M21)

    @staticmethod
    def get_center_of_buoyancy(params):
        rb = np.array([params.x, params.y, params.z])
        return ca.DM(rb)

    @staticmethod
    def get_center_of_gravity(params):
        rc = np.array([params.x, params.y, params.z])
        return ca.DM(rc)

    @staticmethod
    def get_lin_damp_params(params):
        lin_damp = np.array([
            params.X_u, params.Y_v, params.Z_w,
            params.K_p, params.M_q, params.N_r
        ])
        return ca.DM(lin_damp)

    @staticmethod
    def get_nonlin_damp_params(params):
        non_lin = np.array([
            params.X_absuu, params.Y_absvv, params.Z_absww,
            params.K_abspp, params.M_absqq, params.N_absrr
        ])
        return ca.DM(non_lin)

    @staticmethod
    def get_DH_link_offset(d, theta, a, alpha):
        R_alpha = np.array([
            [1, 0, 0],
            [0, np.cos(alpha), -np.sin(alpha)],
            [0, np.sin(alpha),  np.cos(alpha)]
        ])
        out = R_alpha.T @ np.array([a, 0, d])
        return ca.DM(out)

    @staticmethod
    def rpy_to_matrix(r, p, y):
        cr, sr = np.cos(r), np.sin(r)
        cp, sp = np.cos(p), np.sin(p)
        cy, sy = np.cos(y), np.sin(y)
        R = np.array([
            [cy * cp, cy * sp * sr - cr * sy, sy * sr + cy * cr * sp],
            [cp * sy, cy * cr + sy * sp * sr, cr * sy * sp - cy * sr],
            [-sp,     cp * sr,                cp * cr]
        ])
        return ca.DM(R)

    @staticmethod
    def skew(v):
        return ca.vertcat(
            ca.horzcat(0,     -v[2],  v[1]),
            ca.horzcat(v[2],   0,    -v[0]),
            ca.horzcat(-v[1],  v[0],  0)
        )

    def updateDHTable(self, q):
        for i in range(self.n_joints):
            self.DH_table[i, 1] = self.q0[i] + q[i]

    def update(self, q):
        self.updateDHTable(q)
        self.TF_iminus1_i[0] = dh2matrix_sym(*[self.DH_table[0, j] for j in range(4)])
        for i in range(1, self.n_joints + 1):
            self.TF_iminus1_i[i] = dh2matrix_sym(*[self.DH_table[i, j] for j in range(4)])

    def get_rotation_iminus1_i(self, idx):
        return self.TF_iminus1_i[idx-1][0:3, 0:3]
    
    @staticmethod
    def rotation_from_quaternion(quat):
        # quat must be [w, x, y, z]
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        # Rotation matrix from quaternion
        R = ca.vertcat(
            ca.horzcat(1 - 2*(y**2 + z**2),     2*(x*y - z*w),         2*(x*z + y*w)),
            ca.horzcat(2*(x*y + z*w),           1 - 2*(x**2 + z**2),   2*(y*z - x*w)),
            ca.horzcat(2*(x*z - y*w),           2*(y*z + x*w),         1 - 2*(x**2 + y**2))
        )
        return R
    
    def rnem_symbolic(self, q, dq, ddq, v_ref, a_ref, w_ref, dw_ref, quaternion_ref, f_eef, l_eef):
        g_ref = self.rotation_from_quaternion(quaternion_ref).T @ self.GRAVITY
        self.update(q)
        self.forward_link0(v_ref, a_ref, w_ref, dw_ref, g_ref) # base linnk 0
        self.forward(q, dq, ddq) # link 1 to 4 with their joints 
        self.forward_eef() # link 5 = link eef, no active joint just coordinate transformation
        self.backward_eef(f_eef, l_eef) # link 5 = link eef
        self.backward() # link 4 to 0
        R_T = self.R_reference.T
        return ca.vertcat(-R_T @ self.f[0], -R_T @ self.l[0])


    def forward_link0(self, v_ref, a_ref, w_ref, dw_ref, g_ref):
        R_T = self.R_reference.T
        w0 = R_T @ w_ref
        dw0 = R_T @ dw_ref
        g0 = R_T @ g_ref

        S_w0 = self.skew(w0)
        S_dw0 = self.skew(dw0)

        r_i = self.r_i_1_i[0]
        r_b = self.r_b[0]
        r_c = self.r_c[0]

        S_w0_r_i = S_w0 @ r_i
        S_w0_r_c = S_w0 @ r_c

        v0 = R_T @ v_ref + S_w0_r_i
        a0 = R_T @ a_ref + S_dw0 @ r_i + S_w0 @ S_w0_r_i

        self.w[0] = w0
        self.dw[0] = dw0
        self.g[0] = g0
        self.v[0] = v0
        self.a[0] = a0
        self.v_b[0] = v0 + S_w0 @ r_b
        v_c = v0 + S_w0_r_c
        self.v_c[0] = v_c
        a_c = a0 + S_dw0 @ r_c + S_w0 @ S_w0_r_c
        self.a_c[0] = a_c
        self.dv_c[0] = a_c - S_w0 @ v_c

    def forward(self, q, dq, ddq):
        w_prev, dw_prev, g_prev, v_prev, a_prev = self.w[0], self.dw[0], self.g[0], self.v[0], self.a[0]
        for idx in range(1, self.n_joints+1):
            R_T = self.get_rotation_iminus1_i(idx).T
            r_i = self.r_i_1_i[idx]
            z_R = R_T[:, 2]  # z-axis in the current frame
            w = R_T @ w_prev + dq[idx-1] * z_R
            dw = R_T @ dw_prev + ddq[idx-1] * z_R + dq[idx-1] * self.skew(R_T @ w_prev) @ z_R
            S_w = self.skew(w)
            S_w_r_i = S_w @ r_i
            v = R_T @ v_prev + S_w_r_i
            a = R_T @ a_prev + self.skew(dw) @ r_i + S_w @ (S_w_r_i)
            g = R_T @ g_prev
            self.w[idx] = w
            self.dw[idx] = dw
            self.g[idx] = g
            self.v[idx] = v
            self.a[idx] = a
            w_prev = w
            dw_prev = dw
            g_prev = g
            v_prev = v
            a_prev = a

            S_dw = self.skew(dw)

            r_i = self.r_i_1_i[idx]
            r_b = self.r_b[idx]
            r_c = self.r_c[idx]

            S_w_r_c = S_w @ r_c

            self.v_b[idx] = v + S_w @ r_b
            v_c = v + S_w_r_c
            self.v_c[idx] = v_c
            a_c = a + S_dw @ r_c + S_w @ S_w_r_c
            self.a_c[idx] = a_c
            self.dv_c[idx] = a_c - S_w @ v_c

    def forward_eef(self):
        w_prev, dw_prev, g_prev, v_prev, a_prev = self.w[-2], self.dw[-2], self.g[-2], self.v[-2], self.a[-2]
        R_T = self.get_rotation_iminus1_i(self.n_links-1).T
        r_i = self.r_i_1_i[-1]
        w = R_T @ w_prev
        dw = R_T @ dw_prev
        S_w = self.skew(w)
        S_w_r_i = S_w @ r_i
        v = R_T @ v_prev + S_w_r_i
        a = R_T @ a_prev + self.skew(dw) @ r_i + S_w @ (S_w_r_i)
        g = R_T @ g_prev
        self.w[-1] = w
        self.dw[-1] = dw
        self.g[-1] = g
        self.v[-1] = v
        self.a[-1] = a

    def backward_eef(self, f_eef, l_eef):
        self.f[-1] = f_eef
        self.l[-1] = -self.skew(f_eef) @ self.r_i_1_i[-1] + l_eef

    def backward(self):
        for idx in range(self.n_links-2, -1, -1):
            R = self.get_rotation_iminus1_i(idx+1)
            f_next = R @ self.f[idx+1]
            l_next = R @ self.l[idx+1]

            v_b_abs = ca.fabs(self.v_b[idx])
            w_abs = ca.fabs(self.w[idx])
            D_t = -ca.diag(self.nonlin_damp_param[idx][0:3] * v_b_abs + self.lin_damp_param[idx][0:3])
            D_r = -ca.diag(self.nonlin_damp_param[idx][3:6] * w_abs + self.lin_damp_param[idx][3:6])

            # Precompute repeated expressions
            v_c      = self.v_c[idx]
            dv_c     = self.dv_c[idx]
            w        = self.w[idx]
            dw       = self.dw[idx]
            m        = self.m[idx]
            g        = self.g[idx]

            r_c      = self.r_c[idx]

            M_a      = self.M_a[idx]
            M12      = self.M12[idx]
            M21      = self.M21[idx]
            I_a      = self.I_a[idx]
            I        = self.I[idx]

            Mv_c     = M_a @ v_c
            M12w     = M12 @ w
            Mv_buoy  = D_t @ self.v_b[idx] + self.m_buoy[idx] * g

            S_w      = self.skew(w)

            # Compute forces
            f_a = (
                M_a @ dv_c +
                M12 @ dw +
                S_w @ (Mv_c + M12w) +
                Mv_buoy
            )

            # Compute moments
            l_a = (
                I_a @ dw +
                M21 @ dv_c +
                self.skew(v_c) @ (Mv_c + M12w) +
                S_w @ (M21 @ v_c + I_a @ w) +
                self.skew(self.r_cb[idx]) @ Mv_buoy +
                D_r @ w
            )

            f = f_next + m * self.a_c[idx] - m * g + f_a
            self.f[idx] = f
            self.l[idx] = (
                -self.skew(f) @ (self.r_i_1_i[idx] + r_c)
                + l_next
                + self.skew(f_next) @ r_c
                + I @ dw
                + S_w @ (I @ w)
                + l_a
            )

    def compute_damping(self, idx):
        """
        Compute damping matrices for link idx using CasADi.
        """
        v_b = self.v_b[idx]
        v_b_abs = ca.fabs(v_b)
        D_t = -ca.diag(self.nonlin_damp_param[idx][0:3] * v_b_abs[0:3] + self.lin_damp_param[idx][0:3])
        D_r = -ca.diag(self.nonlin_damp_param[idx][3:6] * v_b_abs[3:6] + self.lin_damp_param[idx][3:6])
        self.D_t[idx] = D_t
        self.D_r[idx] = D_r

    def rnem_function_symbolic(self):
        q = ca.MX.sym('q', self.n_joints)
        dq = ca.MX.sym('dq', self.n_joints)
        ddq = ca.MX.sym('ddq', self.n_joints)
        v_ref = ca.MX.sym('v_ref', 3)
        a_ref = ca.MX.sym('a_ref', 3)
        w_ref = ca.MX.sym('w_ref', 3)
        dw_ref = ca.MX.sym('dw_ref', 3)
        quaternion_ref = ca.MX.sym('quat_ref', 4)
        f_eef = ca.MX.sym('f_eef', 3)
        l_eef = ca.MX.sym('l_eef', 3)

        tau = self.rnem_symbolic(q, dq, ddq, v_ref, a_ref, w_ref, dw_ref, quaternion_ref, f_eef, l_eef)
        
        rnem_func = ca.Function(
            'rnem_func',
            [q, dq, ddq, v_ref, a_ref, w_ref, dw_ref, quaternion_ref, f_eef, l_eef],
            [tau]
        )
        
        return rnem_func