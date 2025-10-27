from bluerov.thruster_b2_inversepoly import ThrusterInversePoly
import time
from common.my_package_path import get_package_path
import numpy as np
from bluerov import dynamics_symbolic as sym_brv
from manipulator import kinematics as manip_kin
from manipulator import kinematics_symbolic as sym_manip_kin
from manipulator import dynamics_symbolic as sym_manip_dyn
from common import utils_math

import cvxpy as cp
import numpy as np
import io
import contextlib
import re

import itertools
import numpy as np
import cvxpy as cp

import matplotlib.pyplot as plt


def _init_models():
    bluerov_pkg = get_package_path('bluerov')
    manip_pkg = get_package_path('manipulator')

    brv_params_path = bluerov_pkg + "/config/model_params.yaml"
    dh_params_path = manip_pkg + "/config/alpha_kin_params.yaml"
    base_tf_path = manip_pkg + "/config/alpha_base_tf_params_bluerov.yaml"
    inertial_path = manip_pkg + "/config/alpha_inertial_params_dh.yaml"

    brv_params = utils_math.load_model_params(brv_params_path)
    manip_params = utils_math.load_dh_params(dh_params_path)
    alpha_params = utils_math.load_dynamic_params([dh_params_path, base_tf_path, inertial_path])

    bluerov_dyn = sym_brv.BlueROVDynamicsSymbolic(brv_params)
    manip_kin_sym = sym_manip_kin.KinematicsSymbolic(manip_params)
    manip_dyn_sym = sym_manip_dyn.DynamicsSymbolic(manip_kin_sym, alpha_params)
    manip_kin_real = manip_kin.Kinematics(manip_params)

    return bluerov_dyn, manip_kin_sym, manip_dyn_sym, manip_kin_real

def saturation_pwm(pwm_value):
    if pwm_value < 1100:
        return 1100
    elif pwm_value > 1900:
        return 1900
    else:
        return pwm_value
    
def nullspace_adaption(
            f_alt, fdz_min, fdz_max,
            N = np.array([
                [ 0.5, 0.0],
                [ 0.5, 0.0],
                [-0.5, 0.0],
                [-0.5, 0.0],
                [ 0.0, 0.5],
                [ 0.0, 0.5],
                [ 0.0, 0.5],
                [ 0.0, 0.5],
            ], dtype=float),
            fmin=None, fmax=None,
    solver="OSQP", verbose=False
):
    """
    Global solve via side enumeration (<=256 patterns):
      minimize 0.5 * ||N w||^2
      s.t. f = f_alt + N w
           for each i: f_i <= fdz_min_i  or  f_i >= fdz_max_i
           optional saturation: fmin <= f <= fmax

    Args:
        f_alt   : (8,) current thruster vector (e.g., PWM)
        fdz_min : (8,) lower deadzone edge
        fdz_max : (8,) upper deadzone edge
        N       : (8,2) nullspace matrix
        fmin    : (8,) or None
        fmax    : (8,) or None
        solver  : "OSQP" (QP) or "GUROBI" etc.
        verbose : print progress per pattern

    Returns:
        best dict or None. Keys:
          'status','obj','w','f','pattern' (tuple of bools, True=UP, False=DOWN)
    """
    # TODO: I must optimize the thrust values directly in Newton and apply conversion to PWM afterwards
    # From the exact model and given V_bat we know the thrust values that correspond to the deadzone edges
    # and the saturation limits. 
    # So ±f_dz for the deadzone edges and
    # f(u_min, V_bat), f(u_max, V_bat) for the saturation limits.
    f_alt   = np.asarray(f_alt,   dtype=float).reshape(-1)
    fdz_min = np.asarray(fdz_min, dtype=float).reshape(-1)
    fdz_max = np.asarray(fdz_max, dtype=float).reshape(-1)
    N       = np.asarray(N,       dtype=float)
    n = f_alt.size
    assert N.shape[0] == n and N.shape[1] == 2, "Expect N shape (8,2) for 8 thrusters."

    if fmin is not None:
        fmin = np.asarray(fmin, dtype=float).reshape(-1)
        assert fmin.size == n
    if fmax is not None:
        fmax = np.asarray(fmax, dtype=float).reshape(-1)
        assert fmax.size == n

    # Variables (reuse across patterns)
    w = cp.Variable(2)
    f = f_alt + N @ w
    # Optimize for closest result to MPC command
    # objective = cp.Minimize(0.5 * cp.sum_squares(N @ w)) 

    # Alternate objective: minimize control effort (minimize magnitude of f)
    # Can change MPC command significantly, even different thruster signs,
    # as PMC, might have found local optimum only
    objective = cp.Minimize(0.5 * cp.sum_squares(f))

    best = None
    # Enumerate all UP/DOWN patterns (False=DOWN => f<=fdz_min, True=UP => f>=fdz_max)
    for pattern in itertools.product([False, True], repeat=n):
        cons = []
        # Deadzone side constraints
        for i, up in enumerate(pattern):
            if up:
                cons.append(f[i] >= fdz_max[i])
            else:
                cons.append(f[i] <= fdz_min[i])

        # Optional saturation
        if fmin is not None:
            cons.append(f >= fmin)
        if fmax is not None:
            cons.append(f <= fmax)

        prob = cp.Problem(objective, cons)

        try:
            val = prob.solve(solver=getattr(cp, solver), warm_start=True, verbose=verbose)
        except Exception:
            continue

        if prob.status in ("optimal", "optimal_inaccurate"):
            cand = dict(status=prob.status, obj=prob.value,
                        w=w.value.copy(), f=f.value.copy(), pattern=pattern)
            if best is None or cand["obj"] < best["obj"]:
                best = cand

    return best

def _interval_intersection(a, b):
    """Intersection of two intervals [a0,a1] and [b0,b1]; returns None if empty."""
    lo = max(a[0], b[0])
    hi = min(a[1], b[1])
    return (lo, hi) if lo <= hi else None

def _union_of_open_intervals(intervals):
    """Union of open intervals (l, r) on R. Returns merged list of disjoint (l, r)."""
    if not intervals:
        return []
    segs = sorted(intervals, key=lambda x: (x[0], x[1]))
    merged = []
    cur_l, cur_r = segs[0]
    for l, r in segs[1:]:
        if l <= cur_r:  # overlap or touch
            cur_r = max(cur_r, r)
        else:
            merged.append((cur_l, cur_r))
            cur_l, cur_r = l, r
    merged.append((cur_l, cur_r))
    return merged

def _complement_of_open_union(merged_open, box=(-np.inf, np.inf)):
    """
    Complement of a union of open intervals within a box [L,U].
    Returns a list of closed intervals [L_i, U_i] representing feasible set.
    """
    L, U = box
    if L > U:
        return []
    if not merged_open:
        return [(L, U)]
    out = []
    cur = L
    for (l, r) in merged_open:
        if r <= L or l >= U:
            continue
        l_clip = max(l, L)
        r_clip = min(r, U)
        if cur < l_clip:
            out.append((cur, l_clip))
        cur = max(cur, r_clip)
    if cur <= U:
        out.append((cur, U))
    # Deduplicate tiny negatives
    cleaned = []
    for a,b in out:
        if b < a: 
            continue
        cleaned.append((a,b))
    return cleaned

def _project_scalar_onto_union(x, intervals):
    """Project scalar x onto a union of closed intervals; return projected x and squared distance."""
    if not intervals:
        return None, np.inf
    # inside any?
    for (L,U) in intervals:
        if L <= x <= U:
            return x, 0.0
    # else nearest endpoint
    best_x, best_d2 = None, np.inf
    for (L,U) in intervals:
        cand = U if abs(U - x) < abs(L - x) else L
        d2 = (cand - x)**2
        if d2 < best_d2:
            best_x, best_d2 = cand, d2
    return best_x, best_d2

def nullspace_adaption_fast(
    f_alt, fdz_min, fdz_max, fmin=None, fmax=None, objective="Nw"
):
    """
    Global, no-solver solve exploiting N's structure:
      Thrusters 1..4: f_i = f_alt_i + c1_i * w1, with c1 = [ +0.5, +0.5, -0.5, -0.5 ]
      Thrusters 5..8: f_i = f_alt_i + 0.5 * w2
    Constraints (per i):
      Saturation:   fmin_i <= f_i <= fmax_i    (optional)
      Deadzone:     f_i <= fdz_min_i   OR   f_i >= fdz_max_i
    objective:
      "Nw" -> minimize 0.5*||N w||^2  (=> w_free = (0,0))
      "f"  -> minimize 0.5*||f_alt + N w||^2 (quadratic LS in w1,w2)

    Returns dict with: status, w (2,), f (8,), w1_intervals, w2_intervals
    """
    f_alt   = np.asarray(f_alt,   float).reshape(-1)
    fdz_min = np.asarray(fdz_min, float).reshape(-1)
    fdz_max = np.asarray(fdz_max, float).reshape(-1)
    n = f_alt.size
    assert n == 8, "This routine expects 8 thrusters."
    if fmin is None: fmin = -np.inf*np.ones(n)
    if fmax is None: fmax =  np.inf*np.ones(n)
    fmin = np.asarray(fmin, float).reshape(-1)
    fmax = np.asarray(fmax, float).reshape(-1)

    # Coefficients
    c1 = np.array([+0.5, +0.5, -0.5, -0.5])
    c2 = 0.5

    # ---------- Build feasible union for w1 ----------
    # Saturation for w1: per i, w1 must lie in [ (fmin_i - fi)/c1_i , (fmax_i - fi)/c1_i ]
    # (swap endpoints if c1_i < 0). Intersection over i=1..4 is one box [w1_sat_lo, w1_sat_hi].
    w1_sat_lo = -np.inf
    w1_sat_hi =  np.inf
    for i in range(4):
        fi, ci = f_alt[i], c1[i]
        lo_i = (fmin[i] - fi)/ci
        hi_i = (fmax[i] - fi)/ci
        if lo_i > hi_i: lo_i, hi_i = hi_i, lo_i  # swap if ci < 0
        inter = _interval_intersection((w1_sat_lo, w1_sat_hi), (lo_i, hi_i))
        if inter is None:
            return dict(status="infeasible", w=None, f=None, w1_intervals=[], w2_intervals=[])
        w1_sat_lo, w1_sat_hi = inter

    # Deadzone forbidden open-intervals in w1: ( (fdz_min - fi)/ci , (fdz_max - fi)/ci )
    forb_w1 = []
    for i in range(4):
        fi, ci = f_alt[i], c1[i]
        a = (fdz_min[i] - fi)/ci
        b = (fdz_max[i] - fi)/ci
        if a > b: a, b = b, a
        forb_w1.append((a, b))
    forb_w1 = _union_of_open_intervals(forb_w1)

    # Feasible w1 = saturation box \ union(forbidden)
    w1_intervals = _complement_of_open_union(forb_w1, box=(w1_sat_lo, w1_sat_hi))
    if not w1_intervals:
        return dict(status="infeasible", w=None, f=None, w1_intervals=[], w2_intervals=[])

    # ---------- Build feasible union for w2 ----------
    # Saturation for w2: for each i=4..7, w2 ∈ [ (fmin_i - fi)/0.5 , (fmax_i - fi)/0.5 ]
    lo_list = (fmin[4:8] - f_alt[4:8]) / c2
    hi_list = (fmax[4:8] - f_alt[4:8]) / c2
    w2_sat_lo = np.max(np.minimum(lo_list, hi_list))
    w2_sat_hi = np.min(np.maximum(lo_list, hi_list))
    if w2_sat_lo > w2_sat_hi:
        return dict(status="infeasible", w=None, f=None, w1_intervals=w1_intervals, w2_intervals=[])

    # Deadzone forbidden open intervals for w2:
    a = (fdz_min[4:8] - f_alt[4:8]) / c2
    b = (fdz_max[4:8] - f_alt[4:8]) / c2
    a, b = np.minimum(a, b), np.maximum(a, b)
    forb_w2 = _union_of_open_intervals(list(zip(a, b)))

    # Feasible w2 = [w2_sat_lo, w2_sat_hi] \ union(forbidden)
    w2_intervals = _complement_of_open_union(forb_w2, box=(w2_sat_lo, w2_sat_hi))
    if not w2_intervals:
        return dict(status="infeasible", w=None, f=None, w1_intervals=w1_intervals, w2_intervals=[])

    # ---------- Unconstrained minimizer ----------
    if objective == "Nw":
        w1_free = 0.0
        w2_free = 0.0
    elif objective == "f":
        # Minimize 0.5*sum (fi + c*wi)^2 per group; since sum c^2 = 1 for each group:
        # w1_free = -sum_{i=1..4} c1_i * f_alt_i
        # w2_free = -sum_{i=5..8} 0.5 * f_alt_i
        w1_free = -np.sum(c1 * f_alt[:4])
        w2_free = -0.5 * np.sum(f_alt[4:8])
    else:
        raise ValueError("objective must be 'Nw' or 'f'.")

    # ---------- Project onto unions (global under quadratic objective) ----------
    w1, _ = _project_scalar_onto_union(w1_free, w1_intervals)
    w2, _ = _project_scalar_onto_union(w2_free, w2_intervals)

    # Construct f*
    f = np.empty(8, float)
    f[:4] = f_alt[:4] + c1 * w1
    f[4:] = f_alt[4:] + c2 * w2
    return dict(status="optimal", w=np.array([w1, w2]), f=f,
                w1_intervals=w1_intervals, w2_intervals=w2_intervals)

# ---------- Plotting ----------
def plot_lane(f_alt, f_values, fmin, fdz_min, fdz_max, fmax, title, savepath=None):
    # Simplified: assume both f_alt and f_values are provided (array-like)
    f_alt = np.asarray(f_alt, dtype=float).reshape(-1)
    f_values = np.asarray(f_values, dtype=float).reshape(-1)

    f_dz_diff = fdz_max - fdz_min

    # Basic bounds for x axis
    x_min = float(np.min(fmin) - np.max(f_dz_diff))
    x_max = float(np.max(fmax) + np.max(f_dz_diff))

    plt.figure(figsize=(10, 2))
    ax = plt.gca()
    ax.hlines(0, x_min, x_max, linewidth=2, color="black")
    ax.set_ylim(-1, 1)
    ax.set_yticks([])

    # Gray spans: left saturation, deadzone, right saturation
    ax.axvspan(x_min, float(fmin[0]), alpha=0.25, color="gray")                     # left sat
    ax.axvspan(float(fdz_min[0]), float(fdz_max[0]), alpha=0.25, color="gray")      # deadzone
    ax.axvspan(float(fmax[0]), x_max, alpha=0.25, color="gray")                     # right sat

    # Vertical command markers for adapted values (primary)
    n = f_values.size
    for idx, val in enumerate(f_values, start=1):
        ax.vlines(float(val), -0.4, 0.4, linewidth=3, color="#1f77b4", zorder=2)  # blue
        ax.text(float(val), 0.62, str(idx), ha="center", va="bottom", color="#1f77b4", fontsize=8, zorder=5)

    # Original alt commands in different color (orange dashed)
    for idx, val in enumerate(f_alt, start=1):
        ax.vlines(float(val), -0.3, 0.3, linewidth=2, color="#ff7f0e", zorder=3)
        ax.text(float(val), -0.62, str(idx), ha="center", va="top", color="#ff7f0e", fontsize=8, zorder=5)

    # Legend
    # If comparing optimizer vs fast nullspace, override legend labels to "anlytical" and "optimized"
    if "Optimizer vs Fast Nullspace Adaption Result" in (title or ""):
        handles = [
            plt.Line2D([0], [0], color="#1f77b4", lw=3, label="analytical"),
            plt.Line2D([0], [0], color="#ff7f0e", lw=2, label="optimized"),
        ]
    else:
        handles = [
            plt.Line2D([0], [0], color="#1f77b4", lw=3, label="adapted f*"),
            plt.Line2D([0], [0], color="#ff7f0e", lw=2, label="f_alt (original)"),
        ]
    ax.legend(handles=handles, bbox_to_anchor=(1.05, 1.0), loc="upper left", frameon=False)

    ax.set_xlim(x_min, x_max)
    ax.set_xticks([float(fmin[0]), float(fdz_min[0]), float(fdz_max[0]), float(fmax[0])])
    ax.set_xticklabels(
        [f"fmin={int(fmin[0])}", f"dz-={int(fdz_min[0])}", f"dz+={int(fdz_max[0])}", f"fmax={int(fmax[0])}"],
        rotation=45, ha="right"
    )
    ax.set_title(title, pad=10)
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=160, bbox_inches="tight")
    plt.show()
    

def main():
    BLUEROV_DYN, MANIP_KIN, MANIP_DYN, MANIP_KIN_REAL = _init_models()
    print(BLUEROV_DYN.mixer_nullspace)
    B = BLUEROV_DYN.mixer

    bluerov_package_path = get_package_path('bluerov')
    thruster_model_path = bluerov_package_path + "/thruster_models/thruster_inversepoly_deg2.npz"
    model = ThrusterInversePoly.load(thruster_model_path)

    V_batt = 15.0  # Voltage
    f_min, f_max, f_dz_minus, f_dz_plus = model.get_force_limits(V_batt)

    # generate 10 random f_alt vectors
    rng = np.random.default_rng(42)
    f_alts = rng.uniform(f_min, f_max, size=(10, 8))
    u_alts = np.array([model.command_simple(f, V_batt) for f in f_alts.flatten()]).reshape(f_alts.shape)

    # expand scalar limits to 8-element vectors for each thruster
    n_thruster = 8
    f_min    = np.full(n_thruster, f_min, dtype=float)
    f_max    = np.full(n_thruster, f_max, dtype=float)
    f_dz_minus = np.full(n_thruster, f_dz_minus, dtype=float)
    f_dz_plus = np.full(n_thruster, f_dz_plus, dtype=float)

    results = []
    results_u = []
    for i, f_alt_vec in enumerate(f_alts, start=1):
        print(f"\nSample {i}: f_alt = {np.round(f_alt_vec, 1)}")
        t0 = time.time()
        # best_opti = nullspace_adaption(
        #     f_alt_vec, f_dz_minus, f_dz_plus,
        #     N=BLUEROV_DYN.mixer_nullspace,
        #     fmin=f_min, fmax=f_max, solver="OSQP", verbose=False
        # )
        print(f"old: tau_v = {B @ f_alt_vec}")
        best = nullspace_adaption_fast(
            f_alt_vec, f_dz_minus, f_dz_plus,
            fmin=f_min, fmax=f_max, objective="f"
        )
        t_diff = time.time() - t0
        print(f"new: tau_v = {B @ f_alt_vec}")
        print(f"Solved in {t_diff*1000:.6f} ms.")
        if best['status'] == 'infeasible':
            print("  No feasible pattern (deadzone + saturation).")
            results.append(None)
        else:
            print(f"  w: {np.round(best.get('w'), 6)}")
            print(f"  f*: {np.round(best.get('f'), 1)}")
            results.append(best)
            f_new = best.get('f')
            results_u.append(np.array([(model.command_simple(f, V_batt)) for f in f_new]))
            plot_lane(
                f_alt_vec, best['f'], f_min, f_dz_minus, f_dz_plus, f_max,
                title=f"Sample {i}: Nullspace Adaption Result"
            )
            # plot_lane(
            #     best_opti['f'], best['f'], f_min, f_dz_minus, f_dz_plus, f_max,
            #     title=f"Sample {i}: Optimizer vs Fast Nullspace Adaption Result"
            # )
    u_min, u_max, u_dz_minus, u_dz_plus = model.get_pwm_limits(V_batt)
    u_min    = np.full(n_thruster, u_min, dtype=float)
    u_max    = np.full(n_thruster, u_max, dtype=float)
    u_dz_minus = np.full(n_thruster, u_dz_minus, dtype=float)
    u_dz_plus = np.full(n_thruster, u_dz_plus, dtype=float)
    for i, res in enumerate(zip(u_alts, results_u)):
        plot_lane(
                res[0], res[1], fmin=np.array(u_min), fdz_min=u_dz_minus, fdz_max=u_dz_plus, fmax=u_max,
                title=f"Sample {i}: Nullspace Adaption Result PWM"
            )
    return
    bluerov_package_path = get_package_path('bluerov')
    thruster_model_path = bluerov_package_path + "/thruster_models/thruster_inversepoly_deg2.npz"
    model = ThrusterInversePoly.load(thruster_model_path)

    V_batt = 15.0  # Voltage
    desired_force = np.linspace(-80.0, 80.0, 50)  # Desired forces from -2N to 5N

    PWM = [model.command_simple(f, V_batt) for f in desired_force]
    a = [(f, u) for f, u in zip(desired_force.tolist(), PWM)]
    for force, pwm in a:
        print(f"{force:.3f} -> {pwm:.0f}")

    return

if __name__ == "__main__":
    main()