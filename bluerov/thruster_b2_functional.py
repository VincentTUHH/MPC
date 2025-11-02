
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Thruster PWM mapping with functional per-voltage polynomial fits (fast runtime).
"""

import argparse
import json
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

from common import utils_math
from common.my_package_path import get_package_path


def _deadband_from_voltage_series(f, u, f_eps):
    """Return u0, u_minus, u_plus from samples at one voltage."""
    f = np.asarray(f).reshape(-1)
    u = np.asarray(u).reshape(-1)

    # Use exact-zero force samples if available
    mask0 = np.abs(f) < f_eps
    if np.any(mask0):
        u0 = float(np.median(u[mask0]))

        # sort by PWM so we can find the contiguous zero interval and take
        # the last non-zero PWM before it and the first non-zero PWM after it
        order = np.argsort(u)
        u_s = u[order]
        mask0_s = mask0[order]

        idxs = np.where(mask0_s)[0]
        if len(idxs) == 0:
            # fallback to percentiles if nothing after sorting (shouldn't happen)
            u_minus = float(np.percentile(u[mask0], 25))
            u_plus = float(np.percentile(u[mask0], 75))
        else:
            # split contiguous runs of zero samples
            runs = np.split(idxs, np.where(np.diff(idxs) != 1)[0] + 1)
            # choose the run whose PWM values are closest to u0
            best_run = min(runs, key=lambda r: np.min(np.abs(u_s[r] - u0)))
            start = best_run[0]
            end = best_run[-1]

            # last non-zero before the zero run
            if start > 0:
                u_minus = float(u_s[start - 1])
            else:
                # no non-zero before run: fallback to first PWM in run
                u_minus = float(u_s[start])

            # first non-zero after the zero run
            if end < len(u_s) - 1:
                u_plus = float(u_s[end + 1])
            else:
                # no non-zero after run: fallback to last PWM in run
                u_plus = float(u_s[end])

        return u0, u_minus, u_plus

    # Fallback: use small-|f| samples
    idx = np.argsort(np.abs(f))[:max(10, len(f)//20)]
    u0 = float(np.median(u[idx]))
    u_minus = float(np.percentile(u[idx], 25))
    u_plus = float(np.percentile(u[idx], 75))

    return u0, u_minus, u_plus


def _fit_poly_constrained(f, u, deg, f_anchor=None, u_anchor=None, monotone=False, lam=1e-6):
    f = np.asarray(f).reshape(-1)
    u = np.asarray(u).reshape(-1)
    n = len(f)
    if n == 0:
        raise ValueError("No data points to fit.")
    Phi = np.column_stack([f**k for k in range(deg+1)])
    a = cp.Variable(deg+1)
    obj = cp.sum_squares(Phi @ a - u) + lam*cp.sum_squares(a)
    cons = []
    if f_anchor is not None and u_anchor is not None:
        cons += [cp.sum([a[k]*(f_anchor**k) for k in range(deg+1)]) == u_anchor]
    if monotone:
        fmin, fmax = float(np.min(f)), float(np.max(f))
        if fmax < fmin + 1e-9:
            fmax = fmin + 1e-3
        K = min(25, max(8, n//20))
        grid = np.linspace(fmin, fmax, K)
        for ff in grid:
            cons += [cp.sum([k*a[k]*(ff**(k-1)) for k in range(1, deg+1)]) >= 0]
    prob = cp.Problem(cp.Minimize(obj), cons)
    prob.solve(solver=cp.OSQP, verbose=False)
    if a.value is None:
        prob.solve(verbose=False)
    if a.value is None:
        raise RuntimeError("Polynomial fit failed. Try different degree or disable monotone.")
    return np.array(a.value).reshape(-1)

def _fit_poly(f, u, deg, lam=1e-6, f_anchor=None, u_anchor=None):
    f = np.asarray(f).reshape(-1)
    u = np.asarray(u).reshape(-1)
    n = len(f)
    if n == 0:
        raise ValueError("No data points to fit.")
    Phi = np.column_stack([f**k for k in range(deg+1)])
    a = cp.Variable(deg+1)
    obj = cp.sum_squares(Phi @ a - u) + lam*cp.sum_squares(a)
    cons = []
    if f_anchor is not None and u_anchor is not None:
        fa = float(f_anchor)
        ua = float(u_anchor)
        cons += [cp.sum([a[k]*(fa**k) for k in range(deg+1)]) == ua]
    prob = cp.Problem(cp.Minimize(obj), cons)
    prob.solve(solver=cp.OSQP, verbose=False)
    if a.value is None:
        prob.solve(verbose=False)
    if a.value is None:
        raise RuntimeError("Polynomial fit failed. Try different degree or adjust constraints/regularization.")
    coeffs_val = np.array(a.value).reshape(-1)
    # compute residual (RMSE)
    residuals = Phi @ coeffs_val - u
    rmse = float(np.sqrt(np.mean(residuals**2)))
    return coeffs_val, rmse


class ThrusterB2Functional:
    def __init__(self, voltages, f_eps, f_dz, deg, coeffs_fwd, coeffs_rev,
                 u0_map, uminus_map, uplus_map,
                 tau_volt=0.02, hysteresis=1.2, u_min=None, u_max=None, f_max_all=None, f_min_all=None, rmse_fwd=None, rmse_rev=None):
        self.voltages = np.array(sorted([float(v) for v in voltages]))
        self.f_eps = float(f_eps)
        self.f_dz = float(f_dz)
        self.deg = int(deg)
        self.coeffs_fwd = {float(v): np.array(c) for v,c in coeffs_fwd.items()}
        self.coeffs_rev = {float(v): np.array(c) for v,c in coeffs_rev.items()}
        self.u0_map = {float(k): float(v) for k, v in u0_map.items()}
        self.uminus_map = {float(k): float(v) for k, v in uminus_map.items()}
        self.uplus_map = {float(k): float(v) for k, v in uplus_map.items()}
        self.tau_volt = float(tau_volt)
        self.hysteresis = float(hysteresis)
        self.u_min = u_min
        self.u_max = u_max
        self.f_max_all = f_max_all
        self.f_min_all = f_min_all
        self.rmse_fwd = np.sum(list(rmse_fwd.values()))/len(rmse_fwd) if rmse_fwd is not None else None
        self.rmse_rev = np.sum(list(rmse_rev.values()))/len(rmse_rev) if rmse_rev is not None else None
        self._Vf = None
        self._in_deadband = True
        self._last_u = None

    def _blend_coeffs(self, V, side="fwd"):
        arrV = self.voltages
        V = float(V)
        if V <= arrV[0]:
            return self.coeffs_fwd[arrV[0]] if side=="fwd" else self.coeffs_rev[arrV[0]]
        if V >= arrV[-1]:
            return self.coeffs_fwd[arrV[-1]] if side=="fwd" else self.coeffs_rev[arrV[-1]]
        j = np.searchsorted(arrV, V)
        V0, V1 = arrV[j-1], arrV[j]
        t = (V - V0)/(V1 - V0 + 1e-12)
        c0 = self.coeffs_fwd[V0] if side=="fwd" else self.coeffs_rev[V0]
        c1 = self.coeffs_fwd[V1] if side=="fwd" else self.coeffs_rev[V1]
        return (1-t)*c0 + t*c1

    def _poly_eval(self, coeffs, x):
        x = np.asarray(x)
        y = np.zeros_like(x, dtype=float) + coeffs[-1]
        for k in range(len(coeffs)-2, -1, -1):
            y = y*x + coeffs[k]
        return y

    def _interp_across_V_scalar(self, V, value_map):
        V = float(V)
        arrV = self.voltages
        if V <= arrV[0]: return value_map[arrV[0]]
        if V >= arrV[-1]: return value_map[arrV[-1]]
        j = np.searchsorted(arrV, V)
        V0, V1 = arrV[j-1], arrV[j]
        t = (V - V0)/(V1 - V0 + 1e-12)
        return (1-t)*value_map[V0] + t*value_map[V1]

    def command(self, f_des, V_meas, dt, rate_limit=None):
        if self._Vf is None:
            self._Vf = float(V_meas)
        alpha = float(np.exp(-dt / max(self.tau_volt, 1e-6)))
        self._Vf = alpha*self._Vf + (1-alpha)*float(V_meas)
        V = float(self._Vf)

        f = float(f_des)
        absf = abs(f)

        if self._in_deadband:
            if absf > self.hysteresis*self.f_eps:
                self._in_deadband = False
        else:
            if absf < self.f_eps/self.hysteresis:
                self._in_deadband = True

        if self._in_deadband:
            u = self._interp_across_V_scalar(V, self.u0_map)
        else:
            if f > 0:
                coeffs = self._blend_coeffs(V, side="fwd")
                u = float(self._poly_eval(coeffs, f))
                u_edge = self._interp_across_V_scalar(V, self.uplus_map)
                u = max(u, u_edge)
            else:
                coeffs = self._blend_coeffs(V, side="rev")
                x = -f
                u = float(self._poly_eval(coeffs, x))
                u_edge = self._interp_across_V_scalar(V, self.uminus_map)
                u = min(u, u_edge)

        if self.u_min is not None: u = max(u, self.u_min)
        if self.u_max is not None: u = min(u, self.u_max)

        if rate_limit is not None and rate_limit > 0 and self._last_u is not None:
            step = rate_limit * dt
            u = float(np.clip(u, self._last_u - step, self._last_u + step))

        self._last_u = float(u)
        return float(u)

    def save(self, path):
        arrV = self.voltages
        def _pack(coeff_map):
            return json.dumps({str(v): coeff_map[v].tolist() for v in arrV})
        np.savez(path,
                 voltages=arrV, f_eps=self.f_eps, deg=self.deg,
                 coeffs_fwd=_pack(self.coeffs_fwd),
                 coeffs_rev=_pack(self.coeffs_rev),
                 u0_vals=np.array([self.u0_map[v] for v in arrV]),
                 uminus_vals=np.array([self.uminus_map[v] for v in arrV]),
                 uplus_vals=np.array([self.uplus_map[v] for v in arrV]),
                 tau_volt=self.tau_volt, hysteresis=self.hysteresis,
                 u_min=self.u_min if self.u_min is not None else np.array([np.nan]),
                 u_max=self.u_max if self.u_max is not None else np.array([np.nan]))

    @classmethod
    def load(cls, path):
        d = np.load(path, allow_pickle=True)
        arrV = d['voltages'].astype(float)
        deg = int(d['deg'])
        u0_map = {float(v): float(u) for v,u in zip(arrV, d['u0_vals'].astype(float))}
        uminus_map = {float(v): float(u) for v,u in zip(arrV, d['uminus_vals'].astype(float))}
        uplus_map  = {float(v): float(u) for v,u in zip(arrV, d['uplus_vals'].astype(float))}
        coeffs_fwd = {float(k): np.array(v) for k,v in json.loads(d['coeffs_fwd'].item()).items()}
        coeffs_rev = {float(k): np.array(v) for k,v in json.loads(d['coeffs_rev'].item()).items()}
        u_min = None if np.isnan(d["u_min"]).any() else float(d["u_min"])
        u_max = None if np.isnan(d["u_max"]).any() else float(d["u_max"])
        return cls(voltages=arrV, f_eps=float(d['f_eps']), deg=deg,
                   coeffs_fwd=coeffs_fwd, coeffs_rev=coeffs_rev,
                   u0_map=u0_map, uminus_map=uminus_map, uplus_map=uplus_map,
                   tau_volt=float(d['tau_volt']), hysteresis=float(d['hysteresis']),
                   u_min=u_min, u_max=u_max)

def print_polynomial(V, coeffs_val, deg, type, rmse):
    # --- Print fitted polynomial ---
    terms = []
    for k, ck in enumerate(coeffs_val):
        if abs(ck) < 1e-12:
            continue
        ck_s = f"{ck:.6g}"
        if k == 0:
            terms.append(ck_s)
        elif k == 1:
            terms.append(f"{ck_s} * x")
        else:
            terms.append(f"{ck_s} * x**{k}")
    poly_str = " + ".join(terms) if terms else "0"
    print(f"V={V}, type={type}: Fitted polynomial (deg={deg}): RMSE={rmse:.6g}, u(x) = {poly_str}")
    # --- End print ---

def train_from_folder(thruster_params_path, f_eps=0.1, f_dz=0.3, deg=2, lam=1e-6, monotone=True,
                      tau_volt=0.02, hysteresis=1.2, u_min=None, u_max=None, min_pts=8):
    data = utils_math.load_all_thruster_data(thruster_params_path)
    voltages = sorted([float(v) for v in data.keys()])

    # compute global force extrema over all voltages
    all_forces = []
    for V in voltages:
        dV = data[V]
        all_forces.extend(dV.get('force', []))
    all_forces = np.asarray(all_forces).astype(float) if len(all_forces) > 0 else np.array([])

    if all_forces.size:
        f_max_all = float(np.max(all_forces))
        f_min_all = float(np.min(all_forces))
    else:
        f_max_all = f_min_all = 0.0

    u0_map, uminus_map, uplus_map = {}, {}, {}
    coeffs_fwd, coeffs_rev = {}, {}
    rmse_fwd, rmse_rev = {}, {}

    for V in voltages:
        pwm = np.asarray(data[V]['pwm']).astype(float)
        f   = np.asarray(data[V]['force']).astype(float)

        u0, u_minus, u_plus = _deadband_from_voltage_series(f, pwm, f_eps)
        u0_map[V] = u0; uminus_map[V] = u_minus; uplus_map[V] = u_plus

        mask_fwd = f > f_eps
        mask_rev = f < -f_eps

        if np.sum(mask_fwd) >= min_pts:
            # coeffs_fwd[V] = _fit_poly_constrained(f[mask_fwd], pwm[mask_fwd], deg=deg,
            #                                       f_anchor=f_eps, u_anchor=u_plus,
            #                                       monotone=monotone, lam=lam)
            coeffs_fwd[V], rmse_fwd[V] = _fit_poly(f[mask_fwd], pwm[mask_fwd], deg=deg, lam=lam, f_anchor=f_dz, u_anchor=u_plus)
            print_polynomial(V, coeffs_fwd[V], deg, "forward", rmse_fwd[V])
        else:
            slope = (np.max(pwm) - u_plus)/max((np.max(f)-f_eps), 1e-6)
            coeffs_fwd[V] = np.array([u_plus - slope*f_eps, slope] + [0.0]*(deg-1))

        if np.sum(mask_rev) >= min_pts:
            x = f[mask_rev]
            # coeffs_rev[V] = _fit_poly_constrained(x, pwm[mask_rev], deg=deg,
            #                                       f_anchor=f_eps, u_anchor=u_minus,
            #                                       monotone=monotone, lam=lam)
            coeffs_rev[V], rmse_rev[V] = _fit_poly(x, pwm[mask_rev], deg=deg, lam=lam, f_anchor=-f_dz, u_anchor=u_minus)
            print_polynomial(V, coeffs_rev[V], deg, "reverse", rmse_rev[V])

        else:
            slope = (np.min(pwm) - u_minus)/max((np.max(-f)-f_eps), 1e-6)
            coeffs_rev[V] = np.array([u_minus - slope*f_eps, slope] + [0.0]*(deg-1))
        print("")

    return ThrusterB2Functional(voltages=voltages, f_eps=f_eps, f_dz=f_dz, deg=deg,
                                coeffs_fwd=coeffs_fwd, coeffs_rev=coeffs_rev,
                                u0_map=u0_map, uminus_map=uminus_map, uplus_map=uplus_map,
                                tau_volt=tau_volt, hysteresis=hysteresis,
                                u_min=u_min, u_max=u_max, f_max_all=f_max_all, f_min_all=f_min_all,
                                rmse_fwd=rmse_fwd, rmse_rev=rmse_rev)


def plot_fits_from_folder(thruster_params_path, model, save_path=None, show=True, max_points_per_voltage=None, voltages_to_plot=None):
    # normalize voltages_to_plot to a list of floats or None
    if voltages_to_plot is not None:
        if isinstance(voltages_to_plot, str):
            # allow comma- or space-separated string
            toks = [t for t in voltages_to_plot.replace(',', ' ').split() if t]
            voltages_to_plot = [float(t) for t in toks]
        else:
            voltages_to_plot = [float(v) for v in voltages_to_plot]

    def _should_plot_voltage(V):
        if voltages_to_plot is None:
            return True
        Vf = float(V)
        for vv in voltages_to_plot:
            if abs(Vf - float(vv)) < 1e-8:
                return True
        return False

    data = utils_math.load_all_thruster_data(thruster_params_path)
    plt.figure()
    for V in voltages_to_plot:
        # if not _should_plot_voltage(V):
        #     continue

        if V not in data:
            key_candidates = [k for k in data.keys() if float(k) == float(V)]
            if key_candidates:
                dV = data[key_candidates[0]]

                f_raw = np.asarray(dV['force']).astype(float)
                u_raw = np.asarray(dV['pwm']).astype(float)
                if max_points_per_voltage is not None and len(f_raw) > max_points_per_voltage:
                    idx = np.linspace(0, len(f_raw)-1, max_points_per_voltage).astype(int)
                    f_raw = f_raw[idx]; u_raw = u_raw[idx]

                # swap: x = PWM, y = Thrust
                plt.scatter(u_raw, f_raw, s=3, alpha=0.35, label=f"raw V={V:g}")
        else:
            dV = data[V]

            f_raw = np.asarray(dV['force']).astype(float)
            u_raw = np.asarray(dV['pwm']).astype(float)
            if max_points_per_voltage is not None and len(f_raw) > max_points_per_voltage:
                idx = np.linspace(0, len(f_raw)-1, max_points_per_voltage).astype(int)
                f_raw = f_raw[idx]; u_raw = u_raw[idx]

            # swap: x = PWM, y = Thrust
            plt.scatter(u_raw, f_raw, s=3, alpha=0.35, label=f"raw V={V:g}")

        def clip_data(u, f):
            _lo = -np.inf
            _hi = np.inf
            if model.u_min is not None:
                _lo = model.u_min
            if model.u_max is not None:
                _hi = model.u_max

            mask = np.isfinite(u)
            if np.isfinite(_lo):
                mask &= (u >= _lo)
            if np.isfinite(_hi):
                mask &= (u <= _hi)
            return u[mask], f[mask]

        # forward side (positive thrust)
        f_min = model.f_dz #ax(model.f_dz, float(np.min(f_raw[f_raw>=model.f_eps])) if np.any(f_raw>=model.f_eps) else model.f_eps)
        f_max = model.f_max_all #float(np.max(f_raw)) if len(f_raw) else f_min+1.0
        f_grid = np.linspace(f_min, f_max, 200)
        cf = model._blend_coeffs(V, side="fwd")
        u_fit_fwd = model._poly_eval(cf, f_grid)
        u_edge = model._interp_across_V_scalar(float(V), model.uplus_map)
        u_fit_fwd = np.maximum(u_fit_fwd, u_edge)
        u_fit_fwd, f_grid = clip_data(u_fit_fwd, f_grid)
        # plot as x = u_fit, y = f_grid
        plt.plot(u_fit_fwd, f_grid, linewidth=1.7, label=f"fit+ V={V:g}")

        # reverse side (negative thrust)
        f_min_r = model.f_min_all #float(np.min(f_raw)) if len(f_raw) else -model.f_eps-1.0
        f_max_r = -model.f_dz
        fr = np.linspace(f_min_r, f_max_r, 200)
        x = fr
        cr = model._blend_coeffs(V, side="rev")
        u_fit_rev = model._poly_eval(cr, x)
        u_edge_r = model._interp_across_V_scalar(float(V), model.uminus_map)
        u_fit_rev = np.minimum(u_fit_rev, u_edge_r)
        u_fit_rev, fr = clip_data(u_fit_rev, fr)
        plt.plot(u_fit_rev, fr, linewidth=1.7, label=f"fit- V={V:g}")

        # shade deadband region (PWM between u_minus and u_plus) and small-thrust band
        u0 = model._interp_across_V_scalar(float(V), model.u0_map)
        u_minus = model._interp_across_V_scalar(float(V), model.uminus_map)
        u_plus  = model._interp_across_V_scalar(float(V), model.uplus_map)
        ax = plt.gca()
        # ax.axvspan(u_minus, u_plus, color='gray', alpha=0.12, zorder=0)  # vertical PWM band
        ax.fill_betweenx([-model.f_eps, model.f_eps],
                         [u_minus, u_minus],
                         [u_plus,  u_plus],
                         color='grey', alpha=0.4, zorder=0, label=f"deadband V={V:g}")  # small-thrust deadband rectangle
        ax.plot([u0], [0.0], marker='o', color='k', markersize=4, label=f"deadband center V={V:g}")  # center deadband marker

        # # draw vertical lines for deadband/pwm thresholds at PWM = u_minus, u0, u_plus
        # u0 = model._interp_across_V_scalar(float(V), model.u0_map)
        # u_minus = model._interp_across_V_scalar(float(V), model.uminus_map)
        # u_plus  = model._interp_across_V_scalar(float(V), model.uplus_map)
        # plt.vlines([u_minus, u0, u_plus], ymin=-model.f_eps, ymax=model.f_eps, linestyles='dotted')

    plt.xlabel("PWM [µs]")
    plt.ylabel("Thrust f [N]")
    plt.title("Thrust vs PWM: polynomial fits per voltage (forward/backward) with raw data")
    plt.grid(True)
    plt.legend(fontsize='small', ncol=2)
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
    if show:
        plt.tight_layout()
        plt.show()

# extend CLI to allow --plot and optional --plot-voltages
def _cli_plot(args, model, thruster_params_path):
    if not getattr(args, "plot", False):
        return

    # parse optional voltage list from several possible arg names
    volt_arg = None
    for attr in ("plot_voltages", "plot-voltages", "plot_vs", "plot_v"):
        if hasattr(args, attr) and getattr(args, attr) is not None:
            volt_arg = getattr(args, attr)
            break

    # helper parse: allow list, comma/space-separated string
    def _parse_voltage_list(s):
        if s is None:
            return None
        if isinstance(s, (list, tuple)):
            return [float(x) for x in s]
        if isinstance(s, str):
            toks = [t for t in s.replace(',', ' ').split() if t]
            return [float(t) for t in toks]
        # single numeric
        try:
            return [float(s)]
        except Exception:
            return None
    voltages = _parse_voltage_list(volt_arg)
    save = args.plot if isinstance(args.plot, str) else None
    plot_fits_from_folder(thruster_params_path, model, save_path=save, show=(save is None), voltages_to_plot=voltages)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=None, help="Folder for CSVs (utils_math loader).")
    ap.add_argument("--f_eps", type=float, default=0.1, help="Deadband force threshold [N]")
    ap.add_argument("--f_dz", type=float, default=0.3, help="Deadzone force width for fitting [N]")
    ap.add_argument("--deg", type=int, default=2, help="Polynomial degree per side")
    ap.add_argument("--lam", type=float, default=1e-4, help="Ridge regularization weight")
    ap.add_argument("--no_monotone", action="store_true", help="Disable monotone derivative constraint")
    ap.add_argument("--tau", type=float, default=0.02, help="Voltage LPF time constant [s]")
    ap.add_argument("--hyst", type=float, default=1.2, help="Deadband hysteresis factor")
    ap.add_argument("--u_min", type=float, default=1100, help="Lower PWM bound (optional)")
    ap.add_argument("--u_max", type=float, default=1900, help="Upper PWM bound (optional)")
    ap.add_argument("--save", type=str, default="thruster_functional_model.npz", help="Output model file")
    ap.add_argument("--plot", nargs='?', const=True, default=False, help="Plot fits; optional path to save.")
    ap.add_argument("--plot-voltages", nargs='?', default=None, help="Optional list of voltages to plot (comma- or space-separated, or multiple args).")
    args = ap.parse_args()

    if args.data_root is None:
        bluerov_package_path = get_package_path('bluerov')
        thruster_params_path = bluerov_package_path + "/thruster_data/"
    else:
        thruster_params_path = args.data_root

    model = train_from_folder(thruster_params_path, f_eps=args.f_eps, f_dz=args.f_dz, deg=args.deg, lam=args.lam,
                              monotone=(not args.no_monotone), tau_volt=args.tau, hysteresis=args.hyst,
                              u_min=args.u_min, u_max=args.u_max)
    
    print("RMS Error total [mus]: forward =", model.rmse_fwd, ", reverse =", model.rmse_rev, "\n")

    model.save(args.save)
    print(f"Saved model to {args.save}")
    print("Example:", "u( f=0.25N, V=15.0V, dt=0.02 ) =", model.command(0.25, 15.0, 0.02))

    _cli_plot(args, model, thruster_params_path)

if __name__ == "__main__":
    main()
