from bluerov.thruster_b2_inversepoly import ThrusterInversePoly
from common.my_package_path import get_package_path
from common import utils_math
from bluerov import dynamics_symbolic as sym_brv
from manipulator import kinematics as manip_kin
from manipulator import kinematics_symbolic as sym_manip_kin
from manipulator import dynamics_symbolic as sym_manip_dyn

import time
import itertools
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt


def _init_models():
    bluerov_pkg = get_package_path("bluerov")
    manip_pkg = get_package_path("manipulator")

    brv_params = utils_math.load_model_params(bluerov_pkg + "/config/model_params.yaml")
    manip_params = utils_math.load_dh_params(manip_pkg + "/config/alpha_kin_params.yaml")
    alpha_params = utils_math.load_dynamic_params([
        manip_pkg + "/config/alpha_kin_params.yaml",
        manip_pkg + "/config/alpha_base_tf_params_bluerov.yaml",
        manip_pkg + "/config/alpha_inertial_params_dh.yaml",
    ])

    bluerov_dyn = sym_brv.BlueROVDynamicsSymbolic(brv_params)
    manip_kin_sym = sym_manip_kin.KinematicsSymbolic(manip_params)
    manip_dyn_sym = sym_manip_dyn.DynamicsSymbolic(manip_kin_sym, alpha_params)
    manip_kin_real = manip_kin.Kinematics(manip_params)

    return bluerov_dyn, manip_kin_sym, manip_dyn_sym, manip_kin_real


def saturation_pwm(pwm_value):
    if pwm_value < 1100:
        return 1100
    if pwm_value > 1900:
        return 1900
    return pwm_value


def nullspace_adaption(
    f_alt,
    fdz_min,
    fdz_max,
    N=np.array(
        [
            [0.5, 0.0],
            [0.5, 0.0],
            [-0.5, 0.0],
            [-0.5, 0.0],
            [0.0, 0.5],
            [0.0, 0.5],
            [0.0, 0.5],
            [0.0, 0.5],
        ],
        dtype=float,
    ),
    fmin=None,
    fmax=None,
    solver="OSQP",
    verbose=False,
):
    """
    Global solve by enumerating UP/DOWN per thruster (2^8 patterns).
    Minimize 0.5 * ||f||^2 subject to f = f_alt + N w and each thruster obeys
    deadzone side (<=fdz_min or >=fdz_max) and optional saturation fmin<=f<=fmax.
    Returns best dict {status,obj,w,f,pattern} or None.
    """
    f_alt = np.asarray(f_alt, dtype=float).reshape(-1)
    fdz_min = np.asarray(fdz_min, dtype=float).reshape(-1)
    fdz_max = np.asarray(fdz_max, dtype=float).reshape(-1)
    N = np.asarray(N, dtype=float)
    n = f_alt.size
    assert N.shape[0] == n and N.shape[1] == 2, "Expect N shape (8,2)."

    if fmin is not None:
        fmin = np.asarray(fmin, dtype=float).reshape(-1)
        assert fmin.size == n
    if fmax is not None:
        fmax = np.asarray(fmax, dtype=float).reshape(-1)
        assert fmax.size == n

    w = cp.Variable(2)
    f = f_alt + N @ w
    objective = cp.Minimize(0.5 * cp.sum_squares(f))

    best = None
    for pattern in itertools.product([False, True], repeat=n):
        cons = []
        for i, up in enumerate(pattern):
            if up:
                cons.append(f[i] >= fdz_max[i])
            else:
                cons.append(f[i] <= fdz_min[i])

        if fmin is not None:
            cons.append(f >= fmin)
        if fmax is not None:
            cons.append(f <= fmax)

        prob = cp.Problem(objective, cons)
        try:
            _ = prob.solve(solver=getattr(cp, solver), warm_start=True, verbose=verbose)
        except Exception:
            continue

        if prob.status in ("optimal", "optimal_inaccurate"):
            cand = dict(status=prob.status, obj=prob.value, w=w.value.copy(), f=f.value.copy(), pattern=pattern)
            if best is None or cand["obj"] < best["obj"]:
                best = cand

    return best

def plot_lane(f_alt, f_values, fmin, fdz_min, fdz_max, fmax, title, savepath=None):
    f_alt = np.asarray(f_alt, dtype=float).reshape(-1)
    f_values = np.asarray(f_values, dtype=float).reshape(-1)

    f_dz_diff = fdz_max - fdz_min
    x_min = float(np.min(fmin) - np.max(f_dz_diff))
    x_max = float(np.max(fmax) + np.max(f_dz_diff))

    plt.figure(figsize=(10, 2))
    ax = plt.gca()
    ax.hlines(0, x_min, x_max, linewidth=2, color="black")
    ax.set_ylim(-1, 1)
    ax.set_yticks([])

    ax.axvspan(x_min, float(fmin[0]), alpha=0.25, color="gray")
    ax.axvspan(float(fdz_min[0]), float(fdz_max[0]), alpha=0.25, color="gray")
    ax.axvspan(float(fmax[0]), x_max, alpha=0.25, color="gray")

    for idx, val in enumerate(f_values, start=1):
        ax.vlines(float(val), -0.4, 0.4, linewidth=3, color="#1f77b4", zorder=2)
        ax.text(float(val), 0.62, str(idx), ha="center", va="bottom", color="#1f77b4", fontsize=8, zorder=5)

    for idx, val in enumerate(f_alt, start=1):
        ax.vlines(float(val), -0.3, 0.3, linewidth=2, color="#ff7f0e", zorder=3)
        ax.text(float(val), -0.62, str(idx), ha="center", va="top", color="#ff7f0e", fontsize=8, zorder=5)

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


def main():
    BLUEROV_DYN, MANIP_KIN, MANIP_DYN, MANIP_KIN_REAL = _init_models()
    # print("Mixer nullspace:", BLUEROV_DYN.mixer_nullspace)
    B = BLUEROV_DYN.mixer

    model = ThrusterInversePoly.load(get_package_path("bluerov") + "/thruster_models/thruster_inversepoly_deg2.npz")
    V_batt = 15.0

    rng = np.random.default_rng(42)
    u_MPC = rng.uniform(-1.0, 1.0, size=(3, 8))
    for _, u in enumerate(u_MPC):
        u_new = model.thruster_adaption(u, V_batt)
    # u_MPC = -1.3 * np.ones((1, 8))  # Test infeasible case
    f_alts = np.array([model.map_mpc_pwm_to_force(u_row, V_batt) for u_row in u_MPC])

    f_min, f_max, f_dz_minus, f_dz_plus = model.get_force_limits(V_batt)

    u_alts = np.array([model.command_simple(f, V_batt) for f in f_alts.flatten()]).reshape(f_alts.shape)

    n_thruster = 8
    f_min = np.full(n_thruster, f_min, dtype=float)
    f_max = np.full(n_thruster, f_max, dtype=float)
    f_dz_minus = np.full(n_thruster, f_dz_minus, dtype=float)
    f_dz_plus = np.full(n_thruster, f_dz_plus, dtype=float)

    results = []
    results_u = []
    count_infeasible = 0
    is_infeasible = False
    for i, f_alt_vec in enumerate(f_alts, start=1):
        # print(f"\nSample {i}: f_alt = {np.round(f_alt_vec, 1)}")
        t0 = time.time()
        # if i == 13:
        #     print(f"Case 13")
        best = model.nullspace_adaption_fast(
            f_alt_vec, f_dz_minus, f_dz_plus, fmin=f_min, fmax=f_max, objective="f"
        )

        t_diff = time.time() - t0
        # print(f"Solved in {t_diff*1000:.3f} ms.")
        # print(f"old tau_v = {B @ f_alt_vec}")
        # print(f"new tau_v = {B @ f_alt_vec}")

        if best["status"] == "infeasible":
            count_infeasible += 1
            is_infeasible = True
            # print("No feasible pattern (deadzone + saturation). Solving without saturation")
            best = model.nullspace_adaption_fast(f_alt_vec, f_dz_minus, f_dz_plus, fmin=None, fmax=None, objective="f")
            if best["status"] == "infeasible":
                print("No feasible pattern even without saturation.")
                continue
            # print("Found feasible pattern without saturation.")

        # print(f"w: {np.round(best.get('w'), 6)}")
        # print(f"f*: {np.round(best.get('f'), 1)}")
        results.append(best)
        f_new = best.get("f")
        # if i == 13:
        #     print(f_new)
        u_new = np.array([model.command_simple(f, V_batt) for f in f_new])
        # if i == 13:
        #     print(u_new)
        u_new = model.clip_pwm_saturation(u_new)
        results_u.append(u_new)
        if is_infeasible:
            plot_lane(f_alt_vec, best["f"], f_min, f_dz_minus, f_dz_plus, f_max, title=f"Sample {i}: Nullspace Adaption Result (Infeasible)")
            u_new = model.normalize_pwm(u_new)
            f_new = np.array([model.map_mpc_pwm_to_force(u_row, V_batt) for u_row in u_new])

            tau_old = B @ f_alt_vec
            tau_new = B @ f_new
            # mean squared error between new and old tau
            tau_error = float(np.sqrt(np.mean((tau_new - tau_old) ** 2)))
            print(f"tau_error: {tau_error:.4f} N")
        is_infeasible = False
        # if i == 13:
        #     print(f"Case 13 ende")

    # plt.show()

    print(f"\nTotal infeasible samples: {count_infeasible} out of {len(f_alts)}")

    u_min, u_max, u_dz_minus, u_dz_plus = model.get_pwm_limits(V_batt)
    u_min = np.full(n_thruster, u_min, dtype=float)
    u_max = np.full(n_thruster, u_max, dtype=float)
    u_dz_minus = np.full(n_thruster, u_dz_minus, dtype=float)
    u_dz_plus = np.full(n_thruster, u_dz_plus, dtype=float)

    for i, (u_alt_row, u_res_row) in enumerate(zip(u_alts, results_u), start=1):
        plot_lane(u_alt_row, u_res_row, fmin=np.array(u_min), fdz_min=u_dz_minus, fdz_max=u_dz_plus, fmax=u_max, title=f"Sample {i}: Nullspace Adaption Result PWM")
    plt.show()

    for i, (u_alt, u) in enumerate(zip(u_alts, u_MPC), start=1):
        u_new = model.mpc_thruster_command_adaption(u, V_batt=16.0)
        plot_lane(u_alt, u_new, fmin=np.array(u_min), fdz_min=u_dz_minus, fdz_max=u_dz_plus, fmax=u_max, title=f"Sample {i}: Nullspace Adaption Result PWM Direct")
    plt.show()
if __name__ == "__main__":
    main()
