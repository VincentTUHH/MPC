
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Thruster PWM mapping (Option B2, piecewise-monotone per voltage + voltage blending)

- Reuses your loader: utils_math.load_all_thruster_data(thruster_params_path)
- Fits per-voltage piecewise monotone (isotonic) u(f) for forward and reverse sides
- Estimates u0(V), u-(V), u+(V) (deadband center and edges) directly per measured voltage
- Blends across voltages at runtime
- Includes exact-discrete low-pass filter on V (alpha = exp(-dt/tau))

Requirements:
  numpy, pandas, scikit-learn

CLI example:
  python thruster_b2_piecewise.py --data_root <path_to_thruster_data_dir> \
      --f_eps 0.03 --tau 0.02 --hyst 1.2 --save model_piecewise.npz

Programmatic:
  from thruster_b2_piecewise import ThrusterB2Piecewise, train_from_folder
  model = train_from_folder("<path>")
  u = model.command(f_des=0.4, V_meas=15.2, dt=0.02)
"""

import argparse
import json
import numpy as np
from sklearn.isotonic import IsotonicRegression

# Import your utilities (must be in PYTHONPATH)
from common import utils_math
from common.my_package_path import get_package_path

# --------------------------- Plotting ---------------------------
def plot_fits_from_folder(thruster_params_path, model, save_path=None, show=True,
                          max_points_per_voltage=None, voltages_to_plot=None):
    """
    Plot raw data (scatter + connected) and fitted piecewise curves for selected voltages.
    X-axis: PWM (µs), Y-axis: thrust (N).

    voltages_to_plot: None (all) or iterable of voltages (floats or strings) to include.
    """
    import matplotlib.pyplot as plt

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
    plt.figure()  # one chart, many curves

    # iterate in voltage order
    for V in model.voltages:
        if not _should_plot_voltage(V):
            continue

        if V not in data:
            # try string/float key mismatch
            key_candidates = [k for k in data.keys() if float(k) == float(V)]
            if key_candidates:
                dV = data[key_candidates[0]]
            else:
                continue
        else:
            dV = data[V]

        f_raw = np.asarray(dV['force']).astype(float)
        u_raw = np.asarray(dV['pwm']).astype(float)
        if max_points_per_voltage is not None and len(f_raw) > max_points_per_voltage:
            idx = np.linspace(0, len(f_raw)-1, max_points_per_voltage).astype(int)
            f_raw = f_raw[idx]; u_raw = u_raw[idx]

        # raw scatter (now PWM on x, thrust on y)
        plt.scatter(u_raw, f_raw, s=6, alpha=0.4, label=f"raw V={V:g}")

        # connect raw points (sorted by PWM to avoid crazy zig-zag)
        if len(u_raw) > 1:
            order = np.argsort(u_raw)
            plt.plot(u_raw[order], f_raw[order], linewidth=0.8, alpha=0.6)

        # fitted forward curve (f >= f_eps): plot u (x) vs f (y)
        fwd_knots = model.f_grid_fwd.get(float(V), None)
        if fwd_knots is not None and len(fwd_knots) >= 2:
            f_min = max(model.f_eps,
                        float(np.min(f_raw[f_raw >= model.f_eps])) if np.any(f_raw >= model.f_eps) else model.f_eps)
            f_max = float(np.max(f_raw)) if len(f_raw) else fwd_knots[-1]
            f_grid = np.linspace(f_min, f_max, 200)
            u_fit = model._eval_piecewise(f_grid, model.f_grid_fwd[float(V)], model.u_piece_fwd[float(V)])
            # ensure outside deadband
            u_edge = model._interp_across_V_scalar(float(V), model.uplus_map)
            u_fit = np.maximum(u_fit, u_edge)
            # swap axes: PWM on x, thrust on y
            plt.plot(u_fit, f_grid, linewidth=1.5, label=f"fit+ V={V:g}")

        # fitted reverse curve (f <= -f_eps): plot u (x) vs f (y)
        rev_knots = model.f_grid_rev.get(float(V), None)
        if rev_knots is not None and len(rev_knots) >= 2:
            f_min = float(np.min(f_raw)) if len(f_raw) else -rev_knots[-1]
            print(f_min)
            f_max = -model.f_eps
            f_grid = np.linspace(f_min, f_max, 200)
            x = f_grid  # x >= f_eps
            u_fit = model._eval_piecewise(x, model.f_grid_rev[float(V)], model.u_piece_rev[float(V)])
            # ensure outside deadband
            u_edge = model._interp_across_V_scalar(float(V), model.uminus_map)
            u_fit = np.minimum(u_fit, u_edge)
            plt.plot(u_fit, f_grid, linewidth=1.5, label=f"fit- V={V:g}")

        # deadband markers at this V -> vertical lines at PWM values between -f_eps..+f_eps
        u0 = model._interp_across_V_scalar(float(V), model.u0_map)
        u_minus = model._interp_across_V_scalar(float(V), model.uminus_map)
        u_plus = model._interp_across_V_scalar(float(V), model.uplus_map)
        plt.vlines([u_minus, u0, u_plus], ymin=-model.f_eps, ymax=model.f_eps, linestyles='dotted')

    # decor
    plt.xlabel("PWM [µs]")
    plt.ylabel("Thrust f [N]")
    plt.title("Thruster piecewise fits (per voltage) with raw data")
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
    plot_fits_from_folder(thruster_params_path, model, save_path=save, show=(save is None),
                          voltages_to_plot=voltages)


def _deadband_from_voltage_series(f, u, f_eps):
    """Return u0, u_minus, u_plus from samples at one voltage."""
    f = np.asarray(f).reshape(-1)
    u = np.asarray(u).reshape(-1)

    # Use exact-zero force samples if available
    mask0 = np.abs(f) == 0.0
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


def _fit_isotonic_anchor(x, y, x_anchor, y_anchor, w_anchor=1e6):
    """
    Fit isotonic regression y ~ x with an anchor point (x_anchor,y_anchor) of large weight
    to pin the curve exactly at the deadband edge.
    Returns (iso_model, x_sorted, y_pred_sorted).
    """
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)

    # # Append anchor
    # x_aug = np.concatenate([x, [x_anchor]])
    # y_aug = np.concatenate([y, [y_anchor]])
    w = np.ones_like(x)
    # w[-1] = w_anchor

    # Sort by x
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]
    w_sorted = w[order]

    iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
    y_fit = iso.fit_transform(x_sorted, y_sorted, sample_weight=w_sorted)
    return iso, x_sorted, y_fit


class ThrusterB2Piecewise:
    """
    Holds per-voltage piecewise monotone fits (forward/reverse) and deadband edges.
    Blends across two nearest voltages at runtime.
    """
    def __init__(self, voltages, f_eps, u0_map, uminus_map, uplus_map,
                 f_grid_fwd, u_piece_fwd, f_grid_rev, u_piece_rev,
                 tau_volt=0.02, hysteresis=1.2, u_min=None, u_max=None):
        """
        voltages: sorted array (M,) of measured voltages with fits
        u*_map: dict V -> scalar deadband values
        f_grid_fwd[V], u_piece_fwd[V]: the breakpoints and values from isotonic reg (forward side)
        f_grid_rev[V], u_piece_rev[V]: the breakpoints and values for reverse side (domain is x=-f >= f_eps)
        """
        self.voltages = np.array(sorted(list(voltages)))
        self.f_eps = float(f_eps)
        self.u0_map = {float(k): float(v) for k, v in u0_map.items()}
        self.uminus_map = {float(k): float(v) for k, v in uminus_map.items()}
        self.uplus_map = {float(k): float(v) for k, v in uplus_map.items()}
        self.f_grid_fwd = {float(k): np.array(v) for k, v in f_grid_fwd.items()}
        self.u_piece_fwd = {float(k): np.array(v) for k, v in u_piece_fwd.items()}
        self.f_grid_rev = {float(k): np.array(v) for k, v in f_grid_rev.items()}
        self.u_piece_rev = {float(k): np.array(v) for k, v in u_piece_rev.items()}

        self.tau_volt = float(tau_volt)
        self.hysteresis = float(hysteresis)
        self.u_min = u_min
        self.u_max = u_max

        # runtime state
        self._Vf = None
        self._in_deadband = True
        self._last_u = None

    # ----- helpers -----
    def _interp_across_V_scalar(self, V, value_map):
        """Linear interpolation across known voltages for a scalar (deadband edges)."""
        V = float(V)
        arrV = self.voltages
        if V <= arrV[0]: return value_map[arrV[0]]
        if V >= arrV[-1]: return value_map[arrV[-1]]
        j = np.searchsorted(arrV, V)
        V0, V1 = arrV[j-1], arrV[j]
        t = (V - V0)/(V1 - V0 + 1e-12)
        return (1-t)*value_map[V0] + t*value_map[V1]

    def _eval_piecewise(self, x, x_knots, y_knots):
        """Evaluate a monotone piecewise-linear function given knots (vectorized)."""
        x = np.asarray(x)
        y = np.interp(x, x_knots, y_knots)  # linear segments, clipped at ends
        return y

    def _blend_two_curves(self, x, V, grids_dict, vals_dict):
        """Blend piecewise curves at the two nearest voltages for scalar x."""
        arrV = self.voltages
        if V <= arrV[0]:
            return self._eval_piecewise(x, grids_dict[arrV[0]], vals_dict[arrV[0]])
        if V >= arrV[-1]:
            return self._eval_piecewise(x, grids_dict[arrV[-1]], vals_dict[arrV[-1]])
        j = np.searchsorted(arrV, V)
        V0, V1 = arrV[j-1], arrV[j]
        t = (V - V0)/(V1 - V0 + 1e-12)
        y0 = self._eval_piecewise(x, grids_dict[V0], vals_dict[V0])
        y1 = self._eval_piecewise(x, grids_dict[V1], vals_dict[V1])
        return (1-t)*y0 + t*y1

    # ----- runtime command -----
    def command(self, f_des, V_meas, dt, rate_limit=None):
        # exact-discrete LPF on voltage
        if self._Vf is None:
            self._Vf = float(V_meas)
        alpha = float(np.exp(-dt / max(self.tau_volt, 1e-6)))
        self._Vf = alpha*self._Vf + (1-alpha)*float(V_meas)
        V = float(self._Vf)

        f = float(f_des)
        absf = abs(f)

        # hysteresis-based deadband enter/exit
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
                # forward side
                u_raw = float(self._blend_two_curves(f, V, self.f_grid_fwd, self.u_piece_fwd))
                u_edge = self._interp_across_V_scalar(V, self.uplus_map)
                u = max(u_raw, u_edge)
            else:
                # reverse side: domain uses x=-f
                x = -f
                u_raw = float(self._blend_two_curves(x, V, self.f_grid_rev, self.u_piece_rev))
                u_edge = self._interp_across_V_scalar(V, self.uminus_map)
                u = min(u_raw, u_edge)

        # clip if bounds set
        if self.u_min is not None: u = max(u, self.u_min)
        if self.u_max is not None: u = min(u, self.u_max)

        # optional rate limit
        if rate_limit is not None and rate_limit > 0 and self._last_u is not None:
            step = rate_limit * dt
            u = float(np.clip(u, self._last_u - step, self._last_u + step))

        self._last_u = float(u)
        return float(u)

    # ----- save/load -----
    def save(self, path):
        np.savez(path,
                 voltages=self.voltages, f_eps=self.f_eps,
                 u0_vals=np.array([self.u0_map[v] for v in self.voltages]),
                 uminus_vals=np.array([self.uminus_map[v] for v in self.voltages]),
                 uplus_vals=np.array([self.uplus_map[v] for v in self.voltages]),
                 # store piecewise knots per voltage as ragged arrays via JSON
                 f_grid_fwd=json.dumps({str(v): self.f_grid_fwd[v].tolist() for v in self.voltages}),
                 u_piece_fwd=json.dumps({str(v): self.u_piece_fwd[v].tolist() for v in self.voltages}),
                 f_grid_rev=json.dumps({str(v): self.f_grid_rev[v].tolist() for v in self.voltages}),
                 u_piece_rev=json.dumps({str(v): self.u_piece_rev[v].tolist() for v in self.voltages}),
                 tau_volt=self.tau_volt, hysteresis=self.hysteresis,
                 u_min=self.u_min if self.u_min is not None else np.array([np.nan]),
                 u_max=self.u_max if self.u_max is not None else np.array([np.nan]))

    @classmethod
    def load(cls, path):
        d = np.load(path, allow_pickle=True)
        voltages = d['voltages'].astype(float)
        u0_vals = d['u0_vals'].astype(float)
        uminus_vals = d['uminus_vals'].astype(float)
        uplus_vals = d['uplus_vals'].astype(float)
        u0_map      = {float(v): float(u) for v,u in zip(voltages, u0_vals)}
        uminus_map  = {float(v): float(u) for v,u in zip(voltages, uminus_vals)}
        uplus_map   = {float(v): float(u) for v,u in zip(voltages, uplus_vals)}
        f_grid_fwd  = {float(k): np.array(v) for k,v in json.loads(d['f_grid_fwd'].item()).items()}
        u_piece_fwd = {float(k): np.array(v) for k,v in json.loads(d['u_piece_fwd'].item()).items()}
        f_grid_rev  = {float(k): np.array(v) for k,v in json.loads(d['f_grid_rev'].item()).items()}
        u_piece_rev = {float(k): np.array(v) for k,v in json.loads(d['u_piece_rev'].item()).items()}

        u_min = None if np.isnan(d["u_min"]).any() else float(d["u_min"])
        u_max = None if np.isnan(d["u_max"]).any() else float(d["u_max"])

        return cls(voltages=voltages, f_eps=float(d['f_eps']),
                   u0_map=u0_map, uminus_map=uminus_map, uplus_map=uplus_map,
                   f_grid_fwd=f_grid_fwd, u_piece_fwd=u_piece_fwd,
                   f_grid_rev=f_grid_rev, u_piece_rev=u_piece_rev,
                   tau_volt=float(d['tau_volt']), hysteresis=float(d['hysteresis']),
                   u_min=u_min, u_max=u_max)


def train_from_folder(thruster_params_path, f_eps=0.03, tau_volt=0.02, hysteresis=1.2,
                      u_min=None, u_max=None, min_pts=8):
    """
    Train model using your folder loader. Expects a dict: data[V]['pwm'], data[V]['force'].
    Returns a ThrusterB2Piecewise.
    """
    data = utils_math.load_all_thruster_data(thruster_params_path)  # user-provided function
    voltages = sorted(list(data.keys()))
    # Ensure numeric
    voltages = [float(v) for v in voltages]
    voltages = sorted(voltages)

    u0_map, uminus_map, uplus_map = {}, {}, {}
    f_grid_fwd, u_piece_fwd = {}, {}
    f_grid_rev, u_piece_rev = {}, {}

    for V in voltages:
        pwm = np.asarray(data[V]['pwm']).astype(float)
        f   = np.asarray(data[V]['force']).astype(float)
        if V == 10.0:
            print(pwm)
            print(f)

        # Deadband at this voltage
        u0, u_minus, u_plus = _deadband_from_voltage_series(f, pwm, f_eps)
        u0_map[V] = u0; uminus_map[V] = u_minus; uplus_map[V] = u_plus

        # Forward side fit (f > f_eps)
        mask_fwd = f > f_eps
        if np.sum(mask_fwd) >= min_pts:
            # print(f"Fitting forward side at {V}V with {np.sum(mask_fwd)} points.")
            xf = f[mask_fwd]
            yf = pwm[mask_fwd]
            # print(xf)
            # print(yf)
            # Anchor at (f_eps, u_plus)
            iso_fwd, _, _ = _fit_isotonic_anchor(xf, yf, x_anchor=f_eps, y_anchor=u_plus)
            f_grid_fwd[V] = iso_fwd.X_thresholds_.copy()
            u_piece_fwd[V] = iso_fwd.y_thresholds_.copy()
            # print(f"  Forward fit knots: f={f_grid_fwd[V]}, u={u_piece_fwd[V]}")
        else:
            # Fallback: simple 2-point line from (f_eps,u_plus) to (max f, max pwm)
            xf = np.array([f_eps, max(np.max(f), f_eps+1e-3)])
            yf = np.array([u_plus, np.max(pwm)])
            f_grid_fwd[V] = xf
            u_piece_fwd[V] = yf

        # Reverse side fit (f < -f_eps) on x = -f >= f_eps
        mask_rev = f < -f_eps
        if np.sum(mask_rev) >= min_pts:
            # print(f"Fitting reverse side at {V}V with {np.sum(mask_rev)} points.")
            xr = f[mask_rev]
            yr = pwm[mask_rev]
            if V == 10.0:
                print(xr)
                print(yr)
            iso_rev, _, _ = _fit_isotonic_anchor(xr, yr, x_anchor=-f_eps, y_anchor=u_minus)
            f_grid_rev[V] = iso_rev.X_thresholds_.copy()
            u_piece_rev[V] = iso_rev.y_thresholds_.copy()
            if V == 10.0:
                print(f"Reverse fit knots at 10V: f={f_grid_rev[V]}, u={u_piece_rev[V]}")
        else:
            xr = np.array([f_eps, max(np.max(-f), f_eps+1e-3)])
            yr = np.array([u_minus, np.min(pwm)])
            f_grid_rev[V] = xr
            u_piece_rev[V] = yr

    return ThrusterB2Piecewise(voltages=voltages, f_eps=f_eps,
                               u0_map=u0_map, uminus_map=uminus_map, uplus_map=uplus_map,
                               f_grid_fwd=f_grid_fwd, u_piece_fwd=u_piece_fwd,
                               f_grid_rev=f_grid_rev, u_piece_rev=u_piece_rev,
                               tau_volt=tau_volt, hysteresis=hysteresis,
                               u_min=u_min, u_max=u_max)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", help="Folder with your thruster CSVs (where utils_math will look). "
                                        "If omitted, we use your bluerov package path + /thruster_data/.",
                    default=None)
    ap.add_argument("--f_eps", type=float, default=0.03, help="Deadband force threshold (N)")
    ap.add_argument("--tau", type=float, default=0.02, help="Voltage LPF time constant [s]")
    ap.add_argument("--hyst", type=float, default=1.2, help="Deadband hysteresis factor")
    ap.add_argument("--u_min", type=float, default=None, help="Lower PWM bound (optional)")
    ap.add_argument("--u_max", type=float, default=None, help="Upper PWM bound (optional)")
    ap.add_argument("--save", type=str, default="thruster_piecewise_model.npz", help="Output model file")
    ap.add_argument("--plot", nargs='?', const=True, default=False,
                    help="Plot raw data and fitted piecewise curves; optional path to save the figure.")
    ap.add_argument("--plot-voltages", nargs='?', default=None,
                    help="Optional list of voltages to plot (comma- or space-separated, or multiple args).")
    args = ap.parse_args()

    if args.data_root is None:
        bluerov_package_path = get_package_path('bluerov')
        thruster_params_path = bluerov_package_path + "/thruster_data/"
    else:
        thruster_params_path = args.data_root

    model = train_from_folder(thruster_params_path, f_eps=args.f_eps,
                              tau_volt=args.tau, hysteresis=args.hyst,
                              u_min=args.u_min, u_max=args.u_max)
    model.save(args.save)
    print(f"Saved model to {args.save}")
    # Print model fit data for 10 V (placed where the placeholder was)
    V_query = 10.0
    arrV = model.voltages
    idx = int(np.argmin(np.abs(arrV - V_query)))
    V_used = float(arrV[idx])
    print(f"Using nearest trained voltage: {V_used} V (requested {V_query} V)")

    # Reverse side (stored domain is x = -f (>= f_eps)). Print PWM -> thrust (negative f).
    if V_used in model.f_grid_rev:
        x_rev = model.f_grid_rev[V_used]
        u_rev = model.u_piece_rev[V_used]
        # thrust is -x_rev for reverse side; sort by PWM
        pairs_rev = sorted(zip(u_rev, x_rev), key=lambda t: t[0])
        print("Reverse knots (pwm [µs] -> f [N]):")
        for uu, ff in pairs_rev:
            print(f"  {uu:.6g} -> {ff:.6g}")
    else:
        print("No reverse fit for this voltage.")

    # Forward side (print PWM -> thrust)
    if V_used in model.f_grid_fwd:
        f_fwd = model.f_grid_fwd[V_used]
        u_fwd = model.u_piece_fwd[V_used]
        # sort by PWM so we print pwm -> thrust in ascending pwm order
        pairs = sorted(zip(u_fwd, f_fwd), key=lambda t: t[0])
        print("Forward knots (pwm [µs] -> f [N]):")
        for uu, ff in pairs:
            print(f"  {uu:.6g} -> {ff:.6g}")
    else:
        print("No forward fit for this voltage.")

    # Deadband edges: print as PWM -> approximate thrust (deadband)
    u0 = model.u0_map.get(V_used, None)
    u_minus = model.uminus_map.get(V_used, None)
    u_plus = model.uplus_map.get(V_used, None)
    print("Deadband edges (pwm -> approx f):")
    if u_minus is not None:
        print(f"  {u_minus:.6g} -> {-model.f_eps:.6g}  (lower edge)")
    if u0 is not None:
        print(f"  {u0:.6g} -> 0  (center)")
    if u_plus is not None:
        print(f"  {u_plus:.6g} -> {model.f_eps:.6g}  (upper edge)")
    # quick runtime example
    print("u( f=0.2N, V=15.0V, dt=0.02 ) =", model.command(0.2, 15.0, 0.02))
    _cli_plot(args, model, thruster_params_path)
    print("Done.")

if __name__ == "__main__":
    main()
