import numpy as np
import casadi as ca

def dh2matrix(d, theta, a, alpha):
    """DH-Parameter zu Homogenen Transformationsmatrix (4x4)"""
    ct = np.cos(theta)
    st = np.sin(theta)
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [0,      sa,     ca,    d],
        [0,       0,      0,    1]
    ])

def dh2matrix_sym(d, theta, a, alpha):
    ct = ca.cos(theta)
    st = ca.sin(theta)
    ca_ = ca.cos(alpha)
    sa = ca.sin(alpha)
    return ca.vertcat(
        ca.horzcat(ct, -st*ca_,  st*sa, a*ct),
        ca.horzcat(st,  ct*ca_, -ct*sa, a*st),
        ca.horzcat(0,      sa,     ca_,    d),
        ca.horzcat(0,       0,      0,    1)
    )

# rpy_vec = {
#     'r': r,
#     'p': p,
#     'y': y,
#     'vec_x': vec[1],
#     'vec_y': vec[1],
#     'vec_z': vec[2]
# }



class Kinematics:
    # def __init__(self, DH_table):
    #     """
    #     DH_table: numpy array mit shape (n_joints+1, 4)
    #     """
    #     self.DH_table = np.array(DH_table)
    #     self.n_joints = self.DH_table.shape[0] - 1
    #     self.q0 = self.DH_table[:, 1].copy()
    #     self.e3 = np.array([0, 0, 1])
    #     self.TF_i = [np.zeros((4, 4)) for _ in range(self.n_joints + 1)]
    #     self.TF_iminus1_i = [np.zeros((4, 4)) for _ in range(self.n_joints + 1)]  # Transformation von i-1 zu i
    #     self.update(np.zeros(self.n_joints))
    
    def __init__(self, DH_table, alpha_params=None):
        """
        Overloaded constructor: DH_table as before, plus extra_param.
        Does the same as the original constructor, plus stores extra_param.
        """
        self.DH_table = np.array(DH_table)
        self.n_joints = self.DH_table.shape[0] - 1
        self.q0 = self.DH_table[:, 1].copy()
        self.e3 = np.array([0, 0, 1])
        self.TF_i = [np.zeros((4, 4)) for _ in range(self.n_joints + 1)]
        self.TF_iminus1_i = [np.zeros((4, 4)) for _ in range(self.n_joints + 1)]  # Transformation von i-1 zu i
        self.update(np.zeros(self.n_joints))

        if alpha_params is not None:
            self.n_links = len(alpha_params.__dict__)
            self.r_i_1_i = np.zeros((self.n_links, 3))

            self.R_reference = self.rpy_to_matrix(alpha_params.link_0.r, alpha_params.link_0.p, alpha_params.link_0.y)
            tf_vec = np.array([alpha_params.link_0.vec.x, alpha_params.link_0.vec.y, alpha_params.link_0.vec.z])
            self.r_i_1_i[0] = self.R_reference.transpose() @ tf_vec

            for i in range(1, self.n_links):
                # Berechne die Translation von i-1 zu i in i frame
                self.r_i_1_i[i] = self.get_DH_link_offset(*self.DH_table[i-1])

            self.inertial = [getattr(alpha_params, f'link_{i}').inertial for i in range(self.n_links)]
            self.active = [getattr(alpha_params, f'link_{i}').active for i in range(self.n_links)]
            self.dh_link = [getattr(alpha_params, f'link_{i}').dh_link for i in range(self.n_links)]

            self.I = []
            for i in range(self.n_links):
                if self.inertial[i]:
                    self.I.append(self.inertia_to_matrix(getattr(alpha_params, f'link_{i}').I))
                else:
                    self.I.append(None)

            self.M_a = []
            for i in range(self.n_links):
                if self.inertial[i]:
                    self.M_a.append(self.hyd_mass_to_transl_matrix(getattr(alpha_params, f'link_{i}').hyd.mass))
                else:
                    self.M_a.append(None)

            self.M12 = []
            for i in range(self.n_links):
                if self.inertial[i]:
                    self.M12.append(self.hyd_mass_to_M12(getattr(alpha_params, f'link_{i}').hyd.mass))
                else:
                    self.M12.append(None)

            self.M21 = []
            for i in range(self.n_links):
                if self.inertial[i]:
                    self.M21.append(self.hyd_mass_to_M21(getattr(alpha_params, f'link_{i}').hyd.mass))
                else:
                    self.M21.append(None)

            self.I_a = []
            for i in range(self.n_links):
                if self.inertial[i]:
                    self.I_a.append(self.hyd_mass_to_inertia_matrix(getattr(alpha_params, f'link_{i}').hyd.mass))
                else:
                    self.I_a.append(None)

            self.m_buoy = []
            for i in range(self.n_links):
                if self.inertial[i]:
                    self.m_buoy.append(getattr(alpha_params, f'link_{i}').hyd.buoy.hyd_mass)
                else:
                    self.m_buoy.append(None)        

            self.r_b = []
            for i in range(self.n_links):
                if self.inertial[i]:
                    self.r_b.append(self.get_center_of_buoyancy(getattr(alpha_params, f'link_{i}').hyd.buoy.cob))
                else:
                    self.r_b.append(None)

            self.r_c = []
            for i in range(self.n_links):
                if self.inertial[i]:
                    self.r_c.append(self.get_center_of_gravity(getattr(alpha_params, f'link_{i}').cog))
                else:
                    self.r_c.append(None)

            self.r_cb = []
            for i in range(self.n_links):
                if self.inertial[i]:
                    self.r_cb.append(self.r_b[i] - self.r_c[i])
                else:
                    self.r_cb.append(None)

            self.m = []
            for i in range(self.n_links):
                if self.inertial[i]:
                    self.m.append(getattr(alpha_params, f'link_{i}').mass)
                else:
                    self.m.append(None)

            self.lin_damp_param = []
            for i in range(self.n_links):   
                if self.inertial[i]:
                    self.lin_damp_param.append(self.get_lin_damp_params(getattr(alpha_params, f'link_{i}').hyd.lin_damp))
                else:
                    self.lin_damp_param.append(None)

            self.nonlin_damp_param = []
            for i in range(self.n_links):
                if self.inertial[i]:
                    self.nonlin_damp_param.append(self.get_nonlin_damp_params(getattr(alpha_params, f'link_{i}').hyd.nonlin_damp))
                else:
                    self.nonlin_damp_param.append(None)

            self.D_t = [np.zeros((3, 3)) if self.inertial[_] else None for _ in range(self.n_links)]  # Translational damping matrix
            self.D_r = [np.zeros((3, 3)) if self.inertial[_] else None for _ in range(self.n_links)]  # Rotational damping matrix
            

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

            # print(self.l)

    def get_current_kinematics(self):
        """
        Returns the current values of v, a, w, dw, g for all links as lists.
        Each entry is a numpy array or None.
        """
        return {
            'v': self.v,
            'a': self.a,
            'w': self.w,
            'dw': self.dw,
            'g': self.g,
            'a_c': self.a_c,
            'v_c': self.v_c,
            'dv_c': self.dv_c,
            'v_b': self.v_b
        }
    
    def get_current_dynamic_parameters(self):
        """
        Returns all dynamic and hydrodynamic parameters for all links as lists.
        Each entry is a numpy array, matrix, or None.
        """
        return {
            'I': self.I,
            'M_a': self.M_a,
            'M12': self.M12,
            'M21': self.M21,
            'I_a': self.I_a,
            'm_buoy': self.m_buoy,
            'r_c': self.r_c,
            'r_b': self.r_b,
            'm': self.m,
            'lin_damp_param': self.lin_damp_param,
            'nonlin_damp_param': self.nonlin_damp_param,
            'D_t': self.D_t,
            'D_r': self.D_r,
            'r_cb': self.r_cb
        }
    
    def get_current_wrench(self):
        """
        Returns the current force (f) and moment (l) values for all links as lists.
        Each entry is a numpy array or None.
        """
        return {'f': self.f, 'l': self.l}
    
    def get_R_reference(self):
        """
        Returns the reference rotation matrix R_reference.
        """
        return self.R_reference

    @staticmethod
    def inertia_to_matrix(params):
        """
        Converts an inertia parameter object to a 3x3 inertia matrix.
        Assumes params has attributes: xx, yy, zz, xy, xz, yz.
        """
        return np.array([
            [params.xx, params.xy, params.xz],
            [params.xy, params.yy, params.yz],
            [params.xz, params.yz, params.zz]
        ])
    

    @staticmethod
    def hyd_mass_to_transl_matrix(params):
        """
        Converts HydMass parameters to a 3x3 translational added mass matrix.
        Assumes params has attributes: X_dotu, Y_dotv, Z_dotw.
        """
        matrix = np.zeros((3, 3))
        matrix[0, 0] = -params.X_dotu
        matrix[1, 1] = -params.Y_dotv
        matrix[2, 2] = -params.Z_dotw
        return matrix
    
    @staticmethod
    def hyd_mass_to_M12(params):
        """
        Converts HydMass parameters to a 3x3 M12 added mass matrix.
        Assumes params has attributes: X_dotp, X_dotq, X_dotr, Y_dotp, Y_dotq, Y_dotr, Z_dotp, Z_dotq, Z_dotr.
        """
        matrix = np.zeros((3, 3))
        matrix[0, 0] = -params.X_dotp
        matrix[0, 1] = -params.X_dotq
        matrix[0, 2] = -params.X_dotr
        matrix[1, 0] = -params.Y_dotp
        matrix[1, 1] = -params.Y_dotq
        matrix[1, 2] = -params.Y_dotr
        matrix[2, 0] = -params.Z_dotp
        matrix[2, 1] = -params.Z_dotq
        matrix[2, 2] = -params.Z_dotr
        return matrix

    @staticmethod
    def hyd_mass_to_M21(params):
        """
        Converts HydMass parameters to a 3x3 M21 added mass matrix.
        Assumes params has attributes: K_dotu, K_dotv, K_dotw, M_dotu, M_dotv, M_dotw, N_dotu, N_dotv, N_dotw.
        """
        matrix = np.zeros((3, 3))
        matrix[0, 0] = -params.K_dotu
        matrix[0, 1] = -params.K_dotv
        matrix[0, 2] = -params.K_dotw
        matrix[1, 0] = -params.M_dotu
        matrix[1, 1] = -params.M_dotv
        matrix[1, 2] = -params.M_dotw
        matrix[2, 0] = -params.N_dotu
        matrix[2, 1] = -params.N_dotv
        matrix[2, 2] = -params.N_dotw
        return matrix
    
    @staticmethod
    def hyd_mass_to_inertia_matrix(params):
        """
        Converts HydMass parameters to a 3x3 inertia-like added mass matrix.
        Assumes params has attributes: K_dotp, K_dotq, K_dotr, M_dotp, M_dotq, M_dotr, N_dotp, N_dotq, N_dotr.
        """
        matrix = np.zeros((3, 3))
        matrix[0, 0] = -params.K_dotp
        matrix[0, 1] = -params.K_dotq
        matrix[0, 2] = -params.K_dotr
        matrix[1, 0] = -params.M_dotp
        matrix[1, 1] = -params.M_dotq
        matrix[1, 2] = -params.M_dotr
        matrix[2, 0] = -params.N_dotp
        matrix[2, 1] = -params.N_dotq
        matrix[2, 2] = -params.N_dotr
        return matrix
    
    @staticmethod
    def get_center_of_buoyancy(params):
        """
        Computes the center of buoyancy for a given alpha_params object.
        Returns a 3D numpy array.
        """
        return np.array([
            params.x,
            params.y,
            params.z
        ])
    
    @staticmethod
    def get_center_of_gravity(params):
        """
        Computes the center of buoyancy for a given alpha_params object.
        Returns a 3D numpy array.
        """
        return np.array([
            params.x,
            params.y,
            params.z
        ])
    
    @staticmethod
    def get_lin_damp_params(params):
        """
        Computes the center of buoyancy for a given alpha_params object.
        Returns a 3D numpy array.
        """
        return np.array([
            params.X_u,
            params.Y_v,
            params.Z_w,
            params.K_p,
            params.M_q,
            params.N_r
        ])
    
    @staticmethod
    def get_nonlin_damp_params(params):
        """
        Computes the center of buoyancy for a given alpha_params object.
        Returns a 3D numpy array.
        """        
        return np.array([
            params.X_absuu,
            params.Y_absvv,
            params.Z_absww,
            params.K_abspp,
            params.M_absqq,
            params.N_absrr
        ])
        

    @staticmethod
    def get_DH_link_offset(d, theta, a, alpha):
        """
        Computes the link offset for given DH parameters.
        Returns a 3D numpy array.
        """
        R_alpha = np.array([
            [1, 0, 0],
            [0, np.cos(alpha), -np.sin(alpha)],
            [0, np.sin(alpha),  np.cos(alpha)]
        ])
        offset = np.array([a, 0, d])
        return R_alpha.T @ offset

    def updateDHTable(self, q):
        """Aktualisiere DH-Tabelle mit aktuellen Gelenkwinkeln"""
        # updates theta values in the DH table
        # q contains the joint angles that change theta between the links
        self.DH_table[:self.n_joints, 1] = self.q0[:self.n_joints] + q

    def update(self, q):
        """Berechne alle Transformationen für gegebenen Gelenkwinkelvektor q"""
        q = np.asarray(q)
        if q.shape[0] != self.n_joints:
            raise ValueError(f"Expected q of length {self.n_joints}, got {q.shape[0]}")
        self.updateDHTable(q)
        # Transformationen berechnen
        self.TF_i[0] = dh2matrix(*self.DH_table[0])
        self.TF_iminus1_i[0] = self.TF_i[0]
        for i in range(1, self.n_joints + 1):
            self.TF_iminus1_i[i] = dh2matrix(*self.DH_table[i])
            self.TF_i[i] = self.TF_i[i-1] @ self.TF_iminus1_i[i]

    def get_rotation_iminus1_i(self, idx):
        """Rückgabe der Rotationsmatrix von i-1 zu i"""
        # idx is the index of the frame (1 to eef = 5 = n_joints + 1)
        if idx < 1 or idx > self.n_joints+1:
            raise IndexError(f"Index {idx} out of bounds for n_joints {self.n_joints}")
        return self.TF_iminus1_i[idx-1][:3, :3]
    
    def get_translation_iminus1_i(self, idx):
        """Rückgabe der Translation von i-1 zu i (3D Vektor) expressed in i-1 frame"""
        # idx is the index of the frame (1 to eef = 5 = n_joints + 1)
        if idx < 1 or idx > self.n_joints+1:
            raise IndexError(f"Index {idx} out of bounds for n_joints {self.n_joints}")
        return self.TF_iminus1_i[idx-1][:3, 3]
    
    @staticmethod
    def rpy_to_matrix(r, p, y):
        """Convert roll, pitch, yaw (r, p, y) to rotation matrix (3x3)"""
        cr = np.cos(r)
        sr = np.sin(r)
        cp = np.cos(p)
        sp = np.sin(p)
        cy = np.cos(y)
        sy = np.sin(y)
        R = np.array([
            [cy * cp, cy * sp * sr - cr * sy, sy * sr + cy * cr * sp],
            [cp * sy, cy * cr + sy * sp * sr, cr * sy * sp - cy * sr],
            [-sp,     cp * sr,                cp * cr]
        ])
        return R
    
    @staticmethod
    def skew(v):
        """
        Returns the skew-symmetric matrix of a 3D vector v.
        Ensures v is a 1D array of length 3.
        """
        return np.array([
            [0,     -v[2],  v[1]],
            [v[2],   0,    -v[0]],
            [-v[1],  v[0],  0]
        ])
    
    def forward(self, v_ref, a_ref, w_ref, dw_ref, g_ref):
        """
        Forward recursion to compute velocities and accelerations for each link.
        Given basic transformation.
        For basis link 0, with perevious link i-1 is reference frame, here bluerov body-fixed frame.
        v_ref: linear velocity of the bluerov expressed in its body-fixed frame (3D)
        a_ref: linear acceleration of the bluerov expressed in its body-fixed frame (3D)
        w_ref: angular velocity of the bluerov expressed in its body-fixed frame (3D)
        dw_ref: angular acceleration of the bluerov expressed in its body-fixed frame (3D)
        g_ref: gravity vector expressed in bluerov's body-fixed frame (3D)

        As link 0 is rigidly fixed to the bkluerov body, the rotation matrix R remains constant
        """
        R_transpose = self.R_reference.transpose()  # Rotation matrix from reference frame to bluerov body-fixed frame
        self.w[0] = R_transpose @ w_ref
        self.dw[0] = R_transpose @ dw_ref
        self.g[0] = R_transpose @ g_ref
        self.v[0] = R_transpose @ v_ref + self.skew(self.w[0]) @ self.r_i_1_i[0]
        self.a[0] = R_transpose @ a_ref + self.skew(self.dw[0]) @ self.r_i_1_i[0] + self.skew(self.w[0]) @ (self.skew(self.w[0]) @ self.r_i_1_i[0])

        self.v_b[0] = self.v[0] + self.skew(self.w[0]) @ self.r_b[0]
        self.a_c[0] = self.a[0] + self.skew(self.dw[0]) @ self.r_c[0] + self.skew(self.w[0]) @ (self.skew(self.w[0]) @ self.r_c[0])

        self.v_c[0] = self.v[0] + self.skew(self.w[0]) @ self.r_c[0]
        self.dv_c[0] = self.a_c[0] - self.skew(self.w[0]) @ self.v_c[0]
    
    def link_forward(self, idx, q, dq, ddq):
        """
        Python equivalent of Link::forward for a single link.
        q: joint position
        dq: joint velocity
        ddq: joint acceleration
        prev_w: angular velocity of previous link (3D)
        prev_dw: angular acceleration of previous link (3D)
        prev_g: gravity vector of previous link (3D)
        prev_v: linear velocity of previous link (3D)
        prev_a: linear acceleration of previous link (3D)
        r_i_1_i: offset vector from previous to current link (3D), optional
        Returns: v, a, w, dw, g for current link
        """
        R_transpose = self.get_rotation_iminus1_i(idx).transpose()  # Rotation matrix from i-1 to i (0 is the base link)

        if not self.active[idx]: # for eef frame just feed forward the previous values and consider velocities due to differnt frame
            self.w[idx] = R_transpose @ self.w[idx-1]
            self.dw[idx] = R_transpose @ self.dw[idx-1]
            self.g[idx] = R_transpose @ self.g[idx-1]
            self.v[idx] = R_transpose @ self.v[idx-1] + self.skew(self.w[idx]) @ self.r_i_1_i[idx]
            self.a[idx] = R_transpose @ self.a[idx-1] + self.skew(self.dw[idx]) @ self.r_i_1_i[idx] +  self.skew(self.w[idx]) @ (self.skew(self.w[idx]) @ self.r_i_1_i[idx])
        else:
            self.w[idx] = R_transpose @ self.w[idx-1] + dq * R_transpose[:,2]
            self.dw[idx] = R_transpose @ self.dw[idx-1] + ddq * R_transpose[:,2] + dq * self.skew(R_transpose @ self.w[idx-1]) @ R_transpose[:,2]
            self.g[idx] = R_transpose @ self.g[idx-1]
            self.v[idx] = R_transpose @ self.v[idx-1] + self.skew(self.w[idx]) @ self.r_i_1_i[idx]
            self.a[idx] = R_transpose @ self.a[idx-1] + self.skew(self.dw[idx]) @ self.r_i_1_i[idx] + self.skew(self.w[idx]) @ (self.skew(self.w[idx]) @ self.r_i_1_i[idx])

        if self.inertial[idx]:
            self.v_b[idx] = self.v[idx] + self.skew(self.w[idx]) @ self.r_b[idx]
            self.a_c[idx] = self.a[idx] + self.skew(self.dw[idx]) @ self.r_c[idx] + self.skew(self.w[idx]) @ (self.skew(self.w[idx]) @ self.r_c[idx])
            self.v_c[idx] = self.v[idx] + self.skew(self.w[idx]) @ self.r_c[idx]
            self.dv_c[idx] = self.a_c[idx] - self.skew(self.w[idx]) @ self.v_c[idx]
        else:
            self.v_b[idx] = None
            self.a_c[idx] = None
            self.v_c[idx] = None
            self.dv_c[idx] = None
    
    def link_backward(self, idx):

        #check again for v_b, w ... if I want them as class variables or given from outside (I think rather class and only the reference kinematics from outside)
        nonlin_damp_param = self.nonlin_damp_param[idx]
        lin_damp_param = self.lin_damp_param[idx]

        if self.D_t[idx] is not None and self.D_r[idx] is not None:
            self.D_t[idx][0, 0] = -nonlin_damp_param[0] * np.abs(self.v_b[idx][0]) - lin_damp_param[0]
            self.D_t[idx][1, 1] = -nonlin_damp_param[1] * np.abs(self.v_b[idx][1]) - lin_damp_param[1]
            self.D_t[idx][2, 2] = -nonlin_damp_param[2] * np.abs(self.v_b[idx][2]) - lin_damp_param[2]
            self.D_r[idx][0, 0] = -nonlin_damp_param[3] * np.abs(self.w[idx][0]) - lin_damp_param[3]
            self.D_r[idx][1, 1] = -nonlin_damp_param[4] * np.abs(self.w[idx][1]) - lin_damp_param[4]
            self.D_r[idx][2, 2] = -nonlin_damp_param[5] * np.abs(self.w[idx][2]) - lin_damp_param[5]

        if self.inertial[idx]:
            f_a = (
                self.M_a[idx] @ self.dv_c[idx]
                + self.M12[idx] @ self.dw[idx]
                + self.skew(self.w[idx]) @ (self.M_a[idx] @ self.v_c[idx])
                + self.skew(self.w[idx]) @ (self.M12[idx] @ self.w[idx])
                + self.D_t[idx] @ self.v_b[idx]
            ) + self.m_buoy[idx] * self.g[idx]

            l_a = (
                self.I_a[idx] @ self.dw[idx]
                + self.M21[idx] @ self.dv_c[idx]
                + self.skew(self.v_c[idx]) @ (self.M_a[idx] @ self.v_c[idx] + self.M12[idx] @ self.w[idx])
                + self.skew(self.w[idx]) @ (self.M21[idx] @ self.v_c[idx] + self.I_a[idx] @ self.w[idx])
                + self.skew(self.r_cb[idx]) @ (self.D_t[idx] @ self.v_b[idx])
                + self.D_r[idx] @ self.w[idx]
            ) + self.skew(self.r_cb[idx]) @ (self.m_buoy[idx] * self.g[idx])
        # f_a = self.m_buoy[idx] * self.g[idx]
        # l_a = self.skew(self.r_cb[idx]) @ (self.m_buoy[idx] * self.g[idx])

        f_i_1 = self.get_rotation_iminus1_i(idx+1) @ self.f[idx+1]
        l_i_1 = self.get_rotation_iminus1_i(idx+1) @ self.l[idx+1]

        print(self.m[idx] * self.g[idx])
        print(self.m[idx] * self.a_c[idx])
        print(f_i_1)
        print(self.f[idx+1])
        print(self.get_rotation_iminus1_i(idx+1))

        if self.inertial[idx]:
            # Compute force for link idx
            self.f[idx] = (
                f_i_1
                + self.m[idx] * self.a_c[idx]
                - self.m[idx] * self.g[idx]
                + f_a
            )

            # Compute moment for link idx
            self.l[idx] = (
                - self.skew(self.f[idx]) @ (self.r_i_1_i[idx] + self.r_c[idx])
                + l_i_1
                + self.skew(f_i_1) @ self.r_c[idx]
                + self.I[idx] @ self.dw[idx]
                + self.skew(self.w[idx]) @ (self.I[idx] @ self.w[idx])
                + l_a
            )
        else:
            # If the link is not inertial, set forces and moments to zero
            self.f[idx] = f_i_1
            self.l[idx] = - self.skew(self.f[idx]) @ self.r_i_1_i[idx] + l_i_1

        print(self.f[idx])

    def backward(self, f_eef, l_eef):
        """
        Backward recursion for the end-effector.
        f_eef: force at the end-effector (3D) given in end-effector frame
        l_eef: moment at the end-effector (3D) given in end-effector frame
        """
        self.f[-1] = f_eef
        self.l[-1] = - self.skew(f_eef) @ self.r_i_1_i[-1] + l_eef

    def get_number_of_links(self):
        """
        Returns the number of links in the kinematic chain.
        Includes the base link (link 0) and the end-effector link.
        """
        return self.n_links

    def get_reference_wrench(self):
        """
        Returns the wrench at the base link (link 0) acting on the bluerov body.
        Expressed in the bluerov body-fixed frame.
        """
        R_transpose = self.R_reference.transpose()  # Rotation matrix from reference frame to bluerov body-fixed frame
        f_ref = - R_transpose @ self.f[0]
        l_ref = - R_transpose @ self.l[0]
        return f_ref, l_ref


        
        


    def get_eef_position(self):
        """Endeffektor-Position (3D)"""
        return np.array(self.TF_i[-1][:3, 3])
    
    def get_eef_attitude(self):
        """Endeffektor-Ausrichtung (4D quaternion)"""
        # Rotationmatrix extrahieren und in Quaternion umwandeln
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
        """Position eines beliebigen Links (3D)"""
        return self.TF_i[idx][:3, 3]

    def get_position_jacobian(self):
        """Positions-Jacobian für den Endeffektor"""
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
        """Rotations-Jacobian für den Endeffektor"""
        J = np.zeros((3, self.n_joints))
        for i in range(self.n_joints):
            if i == 0:
                J[:, i] = self.e3
            else:
                J[:, i] = self.TF_i[i-1][:3, :3] @ self.e3
        return J

    def get_full_jacobian(self):
        """6xN Jacobian (Position + Rotation)"""
        J_pos = self.get_position_jacobian()
        J_rot = self.get_rotation_jacobian()
        return np.vstack((J_pos, J_rot))
    
    def forward_kinematics_symbolic(self, q_sym):
        n_joints = self.DH_table.shape[0] - 1
        TF = ca.MX.eye(4)
        for i in range(n_joints + 1):
            d       = ca.MX(self.DH_table[i, 0])
            theta_0 = ca.MX(self.DH_table[i, 1])
            a       = ca.MX(self.DH_table[i, 2])
            alpha   = ca.MX(self.DH_table[i, 3]) # muss ich DH table nicht auch updaten?
            theta = theta_0 + (q_sym[i] if i < n_joints else 0) # die condition vllt falsch
            TF_i = dh2matrix_sym(d, theta, a, alpha)
            TF = ca.mtimes(TF, TF_i)
        return TF
    
    @staticmethod
    def rotation_matrix_to_quaternion(R):
        """Convert a 3x3 CasADi rotation matrix to a 4D quaternion [qw, qx, qy, qz]"""
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