import numpy as np
import casadi as ca
from common import utils_sym
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def Lumelsky_old(p11, p12, p21, p22):
    d1 = p12 - p11
    d2 = p22 - p21
    d12 = p21 - p11

    D1 = np.dot(d1, d1)  # squared 2-norm of d1
    D2 = np.dot(d2, d2)  # squared 2-norm of d2
    R = np.dot(d1, d2)
    S1 = np.dot(d1, d12)
    S2 = np.dot(d2, d12)

    denominator = D1 * D2 - R * R

    step = 1
    found_result = False
    switch = False

    while not found_result:

        if step == 1:
            if D1 == 0:
                u = 0
                # Swap d1 <-> d2, D1 <-> D2, S1 <-> S2
                switch = True
                d1, d2 = d2, d1
                d12 = -d12
                D1, D2 = D2, D1
                S1, S2 = -S2, -S1
                step = 4
            elif D2 == 0:
                u = 0
                step = 4
            elif D1 == 0 and D2 == 0:
                u = 0
                t = 0
                step = 5
            elif D1 != 0 and D2 != 0 and denominator == 0:
                t = 0
                step = 3
            else:
                step = 2

        elif step == 2:
            t = (S1*D2 - S2*R) / denominator
            if t < 0:
                t = 0
            elif t > 1:
                t = 1
            step = 3
        
        elif step == 3:
            u = (t*R - S2) / D2
            if u < 0:
                u = 0
                step = 4
            elif u > 1:
                u = 1
                step = 4
            else:
                step = 5

        elif step == 4:
            t = (u*R + S1) / D1
            if t < 0:
                t = 0
            elif t > 1:
                t = 1
            step = 5
        
        elif step == 5:
            temp_vec = t*d1 - u*d2 - d12
            MinD_squared = np.dot(temp_vec, temp_vec)
            found_result = True

    if switch:
        t, u = u, t  # swap back
    
    return MinD_squared,t , u

def Lumelsky(p11, p12, p21, p22):
    d1 = p12 - p11
    d2 = p22 - p21
    d12 = p21 - p11

    D1 = np.dot(d1, d1)
    D2 = np.dot(d2, d2)
    R = np.dot(d1, d2)
    S1 = np.dot(d1, d12)
    S2 = np.dot(d2, d12)
    denominator = D1 * D2 - R * R

    # step 1: handle special cases
    if D1 == 0 and D2 == 0:
        t = 0
        u = 0
    elif D1 == 0:
        u = 0
        # Swap d1 <-> d2, D1 <-> D2, S1 <-> S2
        d1, d2 = d2, d1
        d12 = -d12
        D1, D2 = D2, D1
        S1, S2 = -S2, -S1
        # step 4
        t = (u * R + S1) / D1
        t = np.clip(t, 0, 1)

        t, u = u, t  # swap back
        d1, d2 = d2, d1
        d12 = -d12
    elif D2 == 0:
        u = 0
        # step 4
        t = (u * R + S1) / D1
        t = np.clip(t, 0, 1)
    elif denominator == 0:
        t = 0
        # step 3
        u = (t * R - S2) / D2
        if u < 0 or u > 1:
            u = np.clip(u, 0, 1)
            # step 4
            t = (u * R + S1) / D1
            t = np.clip(t, 0, 1)
    else:
        # step 2
        t = (S1 * D2 - S2 * R) / denominator
        t = np.clip(t, 0, 1)
        # step 3
        u = (t * R - S2) / D2
        if u < 0 or u > 1:
            u = np.clip(u, 0, 1)
            # step 4
            t = (u * R + S1) / D1
            t = np.clip(t, 0, 1)

    temp_vec = t * d1 - u * d2 - d12
    MinD_squared = np.dot(temp_vec, temp_vec)
    return MinD_squared, t, u

# ---------- distance: smooth + parallel-safe ----------
def _Lumelsky() -> ca.Function:
    """
    Smooth, CasADi-friendly squared distance between segments [p11,p12] and [p21,p22].
    - Works for 2D/3D/... (matching dims), SX or MX.
    - Degenerate (point) and parallel cases handled without branching.
    - Returns: (d2, t, u, cp1, cp2)
    """
    p11 = ca.MX.sym('p11', 3)
    p12 = ca.MX.sym('p12', 3)
    p21 = ca.MX.sym('p21', 3)
    p22 = ca.MX.sym('p22', 3)
    beta = ca.MX.sym('beta')
    reg_eps = ca.MX.sym('reg_eps')
    k_par = ca.MX.sym('k_par')
    tau = ca.MX.sym('tau')

    d1  = p12 - p11            # segment 1 direction
    d2  = p22 - p21            # segment 2 direction
    d12 = p21 - p11

    D1 = ca.dot(d1, d1)        # ||d1||^2
    D2 = ca.dot(d2, d2)        # ||d2||^2
    R  = ca.dot(d1, d2)        # d1·d2
    S1 = ca.dot(d1, d12)       # d1·(p21 - p11)
    S2 = ca.dot(d2, d12)       # d2·(p21 - p11)

    # Gram determinant (zero when parallel)
    D = D1*D2 - R*R
    denom_norm = D1*D2 + reg_eps # normalize with something of similar scale, as D scales with segment lengths. reg_eps to avoid div0 if either segment is a point
    gamma = D / denom_norm          # ~1 when orthogonal, ~0 when parallel

    # Adaptive Tikhonov (bigger as we get more parallel/degenerate)
    lam_base = reg_eps * (D1 + D2 + 1.0)
    lam = lam_base * (1.0 + k_par*(1.0 - gamma))  # ramps up near parallel

    # --- Solve the 2x2 system (regularized), then soft-refine edges (Lumelsky steps 3–4) ---
    # [ D1   -R ] [ t ] = [ S1 ]
    # [  R  -(D2)] [ u ]   [ S2 ]   (signs chosen to keep A well-conditioned with lam)
    a = D1 + lam
    b = -R
    c =  R
    d = -(D2 + lam)

    det = a*d - b*c
    # keep sign, avoid zero determinant
    det_safe = det + ca.sign(det)*1e-12 + 1e-12

    A_inv_00 =  d / det_safe
    A_inv_01 = -b / det_safe
    A_inv_10 = -c / det_safe
    A_inv_11 =  a / det_safe

    t0 = A_inv_00*S1 + A_inv_01*S2
    u0 = A_inv_10*S1 + A_inv_11*S2

    # Box to [0,1]^2 with your softclip + soft edge re-solves
    t1 = utils_sym.softclip(t0, 0.0, 1.0, beta)
    u1 = utils_sym.softclip((t1*R - S2) / (D2 + lam), 0.0, 1.0, beta)
    t2 = utils_sym.softclip((u1*R + S1) / (D1 + lam), 0.0, 1.0, beta)
    u2 = utils_sym.softclip((t2*R - S2) / (D2 + lam), 0.0, 1.0, beta)

    # --- Parallel fallback (branch-free) ---
    # For parallel lines, a stable choice is: project point->segment first, then refine the other.
    # Using Ericson-like recipe: s = clamp(S1 / D1), then compute u from s.
    t_par = utils_sym.softclip(S1 / (D1 + lam), 0.0, 1.0, beta)
    u_par = utils_sym.softclip((t_par*R - S2) / (D2 + lam), 0.0, 1.0, beta)
    # one refinement back:
    t_par = utils_sym.softclip((u_par*R + S1) / (D1 + lam), 0.0, 1.0, beta)

    # Smoothly blend toward the parallel fallback as gamma→0.
    # Weight w_par in [0,1], ~1 when parallel, ~0 when non-parallel.
    # Using a simple smooth rational: w_par = 1 - gamma / (gamma + τ)
    w_par = 1.0 - gamma / (gamma + tau)

    t = (1.0 - w_par)*t2 + w_par*t_par
    u = (1.0 - w_par)*u2 + w_par*u_par

    diff = t * d1 - u * d2 - d12
    MinD_squared = ca.dot(diff, diff)

    return ca.Function('lumelsky', [p11, p12, p21, p22, beta, reg_eps, k_par, tau], [MinD_squared, t, u]).expand()


def _Lumeslky_cases() -> ca.Function:
    p11 = ca.MX.sym('p11', 3)
    p12 = ca.MX.sym('p12', 3)
    p21 = ca.MX.sym('p21', 3)
    p22 = ca.MX.sym('p22', 3)

    d1  = p12 - p11            # segment 1 direction
    d2  = p22 - p21            # segment 2 direction
    d12 = p21 - p11

    D1 = ca.dot(d1, d1)        # ||d1||^2
    D2 = ca.dot(d2, d2)        # ||d2||^2
    R  = ca.dot(d1, d2)        # d1·d2
    S1 = ca.dot(d1, d12)       # d1·(p21 - p11)
    S2 = ca.dot(d2, d12)       # d2·(p21 - p11)
    denominator = D1 * D2 - R * R

    switch = ca.if_else(D1 == 0, True, False)

    # CasADi branch-free switching using if_else
    d1_sw = ca.if_else(switch, d2, d1)
    d2_sw = ca.if_else(switch, d1, d2)
    d12_sw = ca.if_else(switch, -d12, d12)
    D1_sw = ca.if_else(switch, D2, D1)
    D2_sw = ca.if_else(switch, D1, D2)
    S1_sw = ca.if_else(switch, -S2, S1)
    S2_sw = ca.if_else(switch, -S1, S2)


    t = ca.if_else(ca.logic_and(D1 == 0, D2 == 0), 0,
        ca.if_else(denominator == 0, 0, (S1_sw * D2_sw - S2_sw * R) / denominator)
    )

    t = ca.if_else(t < 0, 0,
        ca.if_else(t > 1, 1, t)
    )

    u = ca.if_else(ca.logic_and(D1 == 0, D2 == 0), 0,
        ca.if_else(D1 == 0, 0,
            ca.if_else(D2 == 0, 0,
                ca.if_else(denominator == 0, (t * R - S2_sw) / D2_sw, (t * R - S2_sw) / D2_sw)
            )
        )
    )

    u = ca.if_else(u < 0, 0,
        ca.if_else(u > 1, 1, u)
    )

    t = (u * R + S1_sw) / D1_sw

    t = ca.if_else(t < 0, 0,
        ca.if_else(t > 1, 1, t)
    )

    # u = ca.if_else(u < 0, 0,
    #     ca.if_else(u > 1, 1, u)
    # )

    # t = ca.if_else(t < 0, 0,
    #     ca.if_else(t > 1, 1, t)
    # )

    diff = t * d1_sw - u * d2_sw - d12_sw
    MinD_squared = ca.dot(diff, diff)

    t_sw = ca.if_else(switch, u, t)
    u_sw = ca.if_else(switch, t, u)

    return ca.Function('lumelsky_cases', [p11, p12, p21, p22], [MinD_squared, t_sw, u_sw]).expand()


def main():



    return ca.Function('lumelsky_cases', [p11, p12, p21, p22], [D1, D2, R, S1, S2]).expand()





def main():
    use_sym_if_else = False
    np.random.seed(42)
    errors = []
    segments = []

    lumelsky_sym = _Lumelsky()

    lumelsky_cases = _Lumeslky_cases()

    for i in range(50):
        # Random segments in 3D
        p11 = np.random.uniform(-5, 5, 3)
        p12 = np.random.uniform(-5, 5, 3)
        p21 = np.random.uniform(-5, 5, 3)
        p22 = np.random.uniform(-5, 5, 3)

        # Add degenerate cases: segment 1 is a point, segment 2 is a point, or both
        if i <= 10:
            # Segment 1 is a point
            p12 = p11.copy()
        elif i <= 20:
            # Segment 2 is a point
            p22 = p21.copy()
        elif i <= 30:
            # Both segments are points
            p12 = p11.copy()
            p22 = p21.copy()
        elif i <= 40:
            # Both segments are parallel
            direction = np.random.uniform(-1, 1, 3)
            direction /= np.linalg.norm(direction)
            p12 = p11 + direction * np.random.uniform(0.1, 5)
            p22 = p21 + direction * np.random.uniform(0.1, 5)

        segments.append((p11, p12, p21, p22))

        # Numeric
        MinD_num, t_num, u_num = Lumelsky(p11, p12, p21, p22)

        # Symbolic
        if use_sym_if_else:
            MinD_sym, t_sym, u_sym = lumelsky_cases(p11, p12, p21, p22)
            MinD_sym = float(MinD_sym)
        else:
            beta = 10.0
            reg_eps = 1e-6
            k_par = 10.0
            tau = 0.005
            MinD_sym, t_sym, u_sym = lumelsky_sym(p11, p12, p21, p22, beta, reg_eps, k_par, tau)
            MinD_sym = float(MinD_sym)

        # print(f"Case {i+1}: Min Distance (numeric) = {np.sqrt(MinD_num):.6f}, Min Distance (symbolic) = {np.sqrt(MinD_sym):.6f}")
        print(f"Case {i+1}: Min Distance (non-squared, numeric) = {MinD_num:.6f}, Min Distance (non-squared, symbolic) = {MinD_sym:.6f}")

        # Error (signed)
        error = np.sqrt(MinD_sym) - np.sqrt(MinD_num)
        errors.append(error)

    # Plot errors
    plt.figure(figsize=(8,4))
    plt.plot(errors, marker='o')
    plt.axhline(0, color='gray', linestyle='--')
    plt.title("Signed error: symbolic vs numeric Lumelsky (50 random cases)")
    plt.xlabel("Case")
    plt.ylabel("Error (symbolic - numeric)")
    plt.tight_layout()
    plt.show()

    # Individual plots for each case with minimum distance line (numeric and symbolic)
    for idx, (p11, p12, p21, p22) in enumerate(segments):
        # Numeric
        MinD_num, t_num, u_num = Lumelsky(p11, p12, p21, p22)
        cp1_num = p11 + t_num * (p12 - p11)
        cp2_num = p21 + u_num * (p22 - p21)

        # Symbolic
        if use_sym_if_else:
            MinD_sym, t_sym, u_sym = lumelsky_cases(p11, p12, p21, p22)
            t_sym = float(t_sym)
            u_sym = float(u_sym)
        else:
            beta = 10.0
            reg_eps = 1e-6
            k_par = 10.0
            MinD_sym, t_sym, u_sym = lumelsky_sym(p11, p12, p21, p22, beta, reg_eps, k_par, tau)
            t_sym = float(t_sym)
            u_sym = float(u_sym)
        cp1_sym = p11 + t_sym * (p12 - p11)
        cp2_sym = p21 + u_sym * (p22 - p21)

        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot([p11[0], p12[0]], [p11[1], p12[1]], [p11[2], p12[2]], color='b', label='Segment 1')
        ax.plot([p21[0], p22[0]], [p21[1], p22[1]], [p21[2], p22[2]], color='r', label='Segment 2')
        # Numeric minimum distance line
        ax.plot([cp1_num[0], cp2_num[0]], [cp1_num[1], cp2_num[1]], [cp1_num[2], cp2_num[2]], color='g', linewidth=2, label='Min Distance (numeric)')
        ax.scatter(*cp1_num, color='g', s=60)
        ax.scatter(*cp2_num, color='g', s=60)
        # Symbolic minimum distance line
        ax.plot([cp1_sym[0], cp2_sym[0]], [cp1_sym[1], cp2_sym[1]], [cp1_sym[2], cp2_sym[2]], color='m', linewidth=2, linestyle='--', label='Min Distance (symbolic)')
        ax.scatter(*cp1_sym, color='m', s=60)
        ax.scatter(*cp2_sym, color='m', s=60)
        ax.set_title(f"Case {idx+1}: Min Distance = {np.sqrt(MinD_num):.4f}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend()
        plt.tight_layout()
        plt.show()

    

    return

if __name__ == "__main__":
    main()
