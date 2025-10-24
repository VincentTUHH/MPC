
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Thruster mapping with *forward* polynomial fits f(u)=thrust per voltage & side,
but the model STORES/USES the **inverse** u = f^{-1}(f_des).

- deg ∈ {2,3,4}
- For deg=2: analytic quadratic inverse (fast, exact)
- For deg>=3: robust real-root solve within valid domain (uses numpy.roots)
- Deadband edges enforced (no PWM in deadzone when |f|>f_eps)
- Voltage blending: blend forward coefficients across the two nearest voltages,
  then invert that blended polynomial for the requested thrust.
"""

import argparse
import json
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

from common import utils_math
from common.my_package_path import get_package_path


def _deadband_from_voltage_series(f, u, f_eps):
    f = np.asarray(f).reshape(-1)
    u = np.asarray(u).reshape(-1)

    mask0 = np.abs(f) < f_eps
    if np.any(mask0):
        u0 = float(np.median(u[mask0]))
        order = np.argsort(u)
        u_s = u[order]
        mask0_s = mask0[order]
        idxs = np.where(mask0_s)[0]
        runs = np.split(idxs, np.where(np.diff(idxs) != 1)[0] + 1) if len(idxs) else []
        if runs:
            best = min(runs, key=lambda r: np.min(np.abs(u_s[r] - u0)))
            s, e = best[0], best[-1]
            u_minus = float(u_s[s] if s==0 else u_s[s-1])
            u_plus  = float(u_s[e] if e==len(u_s)-1 else u_s[e+1])
        else:
            u_minus = float(np.percentile(u[mask0], 25))
            u_plus  = float(np.percentile(u[mask0], 75))
        return u0, u_minus, u_plus

    idx = np.argsort(np.abs(f))[:max(10, len(f)//20)]
    u0 = float(np.median(u[idx]))
    u_minus = float(np.percentile(u[idx], 25))
    u_plus  = float(np.percentile(u[idx], 75))
    return u0, u_minus, u_plus


def _fit_forward_poly(u, f, deg, u_anchor=None, f_anchor=None, lam=1e-6, monotone=True, u_bounds=None):
    u = np.asarray(u).reshape(-1).astype(float)
    f = np.asarray(f).reshape(-1).astype(float)
    if len(u) == 0:
        raise ValueError("No data points for fit.")

    Phi = np.column_stack([u**k for k in range(deg+1)])
    a = cp.Variable(deg+1)
    obj = cp.sum_squares(Phi @ a - f) + lam*cp.sum_squares(a)
    cons = []
    if (u_anchor is not None) and (f_anchor is not None):
        ua = float(u_anchor); fa = float(f_anchor)
        cons.append(cp.sum([a[k]*(ua**k) for k in range(deg+1)]) == fa)
    if monotone:
        if u_bounds is None:
            umin, umax = float(np.min(u)), float(np.max(u))
        else:
            umin, umax = map(float, u_bounds)
        umax = max(umax, umin + 1e-6)
        K = 15
        grid = np.linspace(umin, umax, K)
        for ug in grid:
            cons.append(cp.sum([k*a[k]*(ug**(k-1)) for k in range(1, deg+1)]) >= 0)
    prob = cp.Problem(cp.Minimize(obj), cons)
    prob.solve(solver=cp.OSQP, verbose=False)
    if a.value is None:
        prob.solve(verbose=False)
    if a.value is None:
        raise RuntimeError("Forward polynomial fit failed.")
    coeffs = np.array(a.value).reshape(-1)
    rmse = float(np.sqrt(np.mean((Phi @ coeffs - f)**2)))
    return coeffs, rmse


def _poly_eval_forward(a, u):
    u = np.asarray(u, dtype=float)
    y = np.zeros_like(u) + a[-1]
    for k in range(len(a)-2, -1, -1):
        y = y*u + a[k]
    return y


def _invert_forward_coeffs(a, f_star, side, u_edge, u_min=None, u_max=None):
    a = np.asarray(a).reshape(-1)
    deg = len(a)-1
    f_star = float(f_star)

    p = a.copy()
    p[0] = p[0] - f_star
    coeff_desc = p[::-1]

    if deg == 1:
        b1, b0 = a[1], a[0]
        if abs(b1) < 1e-12:
            u = u_edge
        else:
            u = (f_star - b0)/b1
    elif deg == 2:
        A, B, C = a[2], a[1], a[0]-f_star
        if abs(A) < 1e-12:
            if abs(B) < 1e-12:
                u = u_edge
            else:
                u = -C/B
        else:
            D = B*B - 4*A*C
            if D < 0:
                u = u_edge
            else:
                sqrtD = np.sqrt(D)
                u1 = (-B + sqrtD)/(2*A)
                u2 = (-B - sqrtD)/(2*A)
                cand = []
                if side == "fwd":
                    if u1 >= u_edge: cand.append(u1)
                    if u2 >= u_edge: cand.append(u2)
                    u = max(cand) if cand else u_edge
                else:
                    if u1 <= u_edge: cand.append(u1)
                    if u2 <= u_edge: cand.append(u2)
                    u = min(cand) if cand else u_edge
    else:
        roots = np.roots(coeff_desc)
        roots = roots[np.isreal(roots)].real
        if side == "fwd":
            roots = roots[roots >= u_edge - 1e-9]
            u = float(np.min(roots)) if roots.size else u_edge
        else:
            roots = roots[roots <= u_edge + 1e-9]
            u = float(np.max(roots)) if roots.size else u_edge

    if u_min is not None: u = max(u, u_min)
    if u_max is not None: u = min(u, u_max)
    return float(u)


class ThrusterInversePoly:
    def __init__(self, voltages, f_eps, deg, a_fwd, a_rev,
                 u0_map, uminus_map, uplus_map,
                 tau_volt=0.02, hysteresis=1.2, u_min=None, u_max=None,
                 rmse_fwd=None, rmse_rev=None):
        self.voltages = np.array(sorted([float(v) for v in voltages]))
        self.f_eps = float(f_eps)
        self.deg = int(deg)
        self.a_fwd = {float(v): np.array(c) for v,c in a_fwd.items()}
        self.a_rev = {float(v): np.array(c) for v,c in a_rev.items()}
        self.u0_map = {float(k): float(v) for k, v in u0_map.items()}
        self.uminus_map = {float(k): float(v) for k, v in uminus_map.items()}
        self.uplus_map = {float(k): float(v) for k, v in uplus_map.items()}
        self.tau_volt = float(tau_volt)
        self.hysteresis = float(hysteresis)
        self.u_min = u_min
        self.u_max = u_max
        self.rmse_fwd = np.mean(list(rmse_fwd.values())) if rmse_fwd else None
        self.rmse_rev = np.mean(list(rmse_rev.values())) if rmse_rev else None
        self._Vf = None
        self._in_deadband = True
        self._last_u = None

    def _interp_scalar(self, V, value_map):
        V = float(V); arrV = self.voltages
        if V <= arrV[0]: return value_map[arrV[0]]
        if V >= arrV[-1]: return value_map[arrV[-1]]
        j = np.searchsorted(arrV, V)
        V0, V1 = arrV[j-1], arrV[j]
        t = (V - V0)/(V1 - V0 + 1e-12)
        return (1-t)*value_map[V0] + t*value_map[V1]

    def _blend_forward_coeffs(self, V, side="fwd"):
        arrV = self.voltages
        V = float(V)
        if V <= arrV[0]:
            return self.a_fwd[arrV[0]] if side=="fwd" else self.a_rev[arrV[0]]
        if V >= arrV[-1]:
            return self.a_fwd[arrV[-1]] if side=="fwd" else self.a_rev[arrV[-1]]
        j = np.searchsorted(arrV, V)
        V0, V1 = arrV[j-1], arrV[j]
        t = (V - V0)/(V1 - V0 + 1e-12)
        c0 = self.a_fwd[V0] if side=="fwd" else self.a_rev[V0]
        c1 = self.a_fwd[V1] if side=="fwd" else self.a_rev[V1]
        return (1-t)*c0 + t*c1

    def command(self, f_des, V_meas, dt, rate_limit=None):
        if self._Vf is None:
            self._Vf = float(V_meas)
        alpha = float(np.exp(-dt / max(self.tau_volt, 1e-6)))
        self._Vf = alpha*self._Vf + (1-alpha)*float(V_meas)
        V = float(self._Vf)

        f = float(f_des); af = abs(f)
        if self._in_deadband:
            if af > self.hysteresis*self.f_eps:
                self._in_deadband = False
        else:
            if af < self.f_eps/self.hysteresis:
                self._in_deadband = True

        if self._in_deadband:
            u = self._interp_scalar(V, self.u0_map)
        else:
            if f > 0:
                a = self._blend_forward_coeffs(V, side="fwd")
                u_edge = self._interp_scalar(V, self.uplus_map)
                u = _invert_forward_coeffs(a, f_star=f, side="fwd",
                                           u_edge=u_edge, u_min=self.u_min, u_max=self.u_max)
                u = max(u, u_edge)
            else:
                a = self._blend_forward_coeffs(V, side="rev")
                u_edge = self._interp_scalar(V, self.uminus_map)
                u = _invert_forward_coeffs(a, f_star=f, side="rev",
                                           u_edge=u_edge, u_min=self.u_min, u_max=self.u_max)
                u = min(u, u_edge)

        if rate_limit is not None and rate_limit > 0 and self._last_u is not None:
            step = rate_limit * dt
            u = float(np.clip(u, self._last_u - step, self._last_u + step))

        self._last_u = float(u)
        return float(u)

    def save(self, path):
        arrV = self.voltages
        def _pack(m): return json.dumps({str(v): m[v].tolist() for v in arrV})
        np.savez(path,
                 voltages=arrV, f_eps=self.f_eps, deg=self.deg,
                 a_fwd=_pack(self.a_fwd), a_rev=_pack(self.a_rev),
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
        def _unpack(key):
            return {float(k): np.array(v) for k,v in json.loads(d[key].item()).items()}
        a_fwd = _unpack('a_fwd'); a_rev = _unpack('a_rev')
        u0_map = {float(v): float(u) for v,u in zip(arrV, d['u0_vals'].astype(float))}
        uminus_map = {float(v): float(u) for v,u in zip(arrV, d['uminus_vals'].astype(float))}
        uplus_map  = {float(v): float(u) for v,u in zip(arrV, d['uplus_vals'].astype(float))}
        u_min = None if np.isnan(d["u_min"]).any() else float(d["u_min"])
        u_max = None if np.isnan(d["u_max"]).any() else float(d["u_max"])
        return cls(voltages=arrV, f_eps=float(d['f_eps']), deg=deg,
                   a_fwd=a_fwd, a_rev=a_rev,
                   u0_map=u0_map, uminus_map=uminus_map, uplus_map=uplus_map,
                   tau_volt=float(d['tau_volt']), hysteresis=float(d['hysteresis']),
                   u_min=u_min, u_max=u_max)


def train_from_folder(thruster_params_path, f_eps=0.1, deg=2, lam=1e-6, monotone=True,
                      tau_volt=0.02, hysteresis=1.2, u_min=1100, u_max=1900, min_pts=8):
    data = utils_math.load_all_thruster_data(thruster_params_path)
    voltages = sorted([float(v) for v in data.keys()])

    u0_map, uminus_map, uplus_map = {}, {}, {}
    a_fwd, a_rev = {}, {}
    rmse_fwd, rmse_rev = {}, {}

    for V in voltages:
        pwm = np.asarray(data[V]['pwm']).astype(float)
        f   = np.asarray(data[V]['force']).astype(float)

        u0, u_minus, u_plus = _deadband_from_voltage_series(f, pwm, f_eps)
        u0_map[V] = u0; uminus_map[V] = u_minus; uplus_map[V] = u_plus

        mask_fwd = f > f_eps
        if np.sum(mask_fwd) >= min_pts:
            a_fwd[V], rmse_fwd[V] = _fit_forward_poly(
                u=pwm[mask_fwd], f=f[mask_fwd], deg=deg,
                u_anchor=u_plus, f_anchor=f_eps, lam=lam, monotone=monotone,
                u_bounds=(u_plus, float(np.max(pwm)))
            )
        else:
            slope = (np.max(f)-f_eps) / max((np.max(pwm)-u_plus), 1e-6)
            a_fwd[V] = np.array([f_eps - slope*u_plus, slope] + [0.0]*(deg-1))

        mask_rev = f < -f_eps
        if np.sum(mask_rev) >= min_pts:
            a_rev[V], rmse_rev[V] = _fit_forward_poly(
                u=pwm[mask_rev], f=f[mask_rev], deg=deg,
                u_anchor=u_minus, f_anchor=-f_eps, lam=lam, monotone=monotone,
                u_bounds=(float(np.min(pwm)), u_minus)
            )
        else:
            slope = (-f_eps - np.min(f)) / max((u_minus - np.min(pwm)), 1e-6)
            a_rev[V] = np.array([-f_eps - slope*u_minus, slope] + [0.0]*(deg-1))

    return ThrusterInversePoly(voltages=voltages, f_eps=f_eps, deg=deg,
                               a_fwd=a_fwd, a_rev=a_rev,
                               u0_map=u0_map, uminus_map=uminus_map, uplus_map=uplus_map,
                               tau_volt=tau_volt, hysteresis=hysteresis,
                               u_min=u_min, u_max=u_max,
                               rmse_fwd=rmse_fwd, rmse_rev=rmse_rev)


def plot_forward_fits(thruster_params_path, model, save_path=None, show=True, volts=None, max_points_per_voltage=None):
    data = utils_math.load_all_thruster_data(thruster_params_path)
    plt.figure()
    vs = model.voltages if volts is None else [float(v) for v in volts]

    for V in vs:
        if V not in data:
            keys = [k for k in data.keys() if float(k)==float(V)]
            if not keys: continue
            dV = data[keys[0]]
        else:
            dV = data[V]
        u_raw = np.asarray(dV['pwm']).astype(float)
        f_raw = np.asarray(dV['force']).astype(float)
        if max_points_per_voltage and len(u_raw)>max_points_per_voltage:
            idx = np.linspace(0, len(u_raw)-1, max_points_per_voltage).astype(int)
            u_raw, f_raw = u_raw[idx], f_raw[idx]
        plt.scatter(u_raw, f_raw, s=6, alpha=0.35, label=f"raw V={V:g}")

        u_line = np.linspace(model.uplus_map[V], np.max(u_raw), 200)
        a = model._blend_forward_coeffs(V, side="fwd")
        f_fit = _poly_eval_forward(a, u_line)
        plt.plot(u_line, f_fit, linewidth=1.6, label=f"fit fwd V={V:g}")

        u_line_r = np.linspace(np.min(u_raw), model.uminus_map[V], 200)
        ar = model._blend_forward_coeffs(V, side="rev")
        f_fit_r = _poly_eval_forward(ar, u_line_r)
        plt.plot(u_line_r, f_fit_r, linewidth=1.6, label=f"fit rev V={V:g}")

        u_minus = model.uminus_map[V]; u_plus = model.uplus_map[V]
        ax = plt.gca()
        ax.fill_betweenx([-model.f_eps, model.f_eps], [u_minus,u_minus], [u_plus,u_plus], color='grey', alpha=0.25)

    plt.xlabel("PWM [µs]")
    plt.ylabel("Thrust f [N]")
    plt.title("Forward fits f(u) per voltage (with raw data)")
    plt.grid(True)
    plt.legend(fontsize='small', ncol=2)
    if save_path:
        plt.tight_layout(); plt.savefig(save_path, dpi=150)
    if show:
        plt.tight_layout(); plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=None, help="Folder for CSVs (utils_math loader).")
    ap.add_argument("--f_eps", type=float, default=0.1, help="Deadband threshold [N]")
    ap.add_argument("--deg", type=int, default=2, choices=[1,2,3,4], help="Polynomial degree for f(u)")
    ap.add_argument("--lam", type=float, default=1e-6, help="Ridge regularization")
    ap.add_argument("--no_monotone", action="store_true", help="Disable monotone df/du >= 0 constraint")
    ap.add_argument("--tau", type=float, default=0.02, help="Voltage LPF time constant [s]")
    ap.add_argument("--hyst", type=float, default=1.2, help="Deadband hysteresis factor")
    ap.add_argument("--u_min", type=float, default=1100, help="Lower PWM bound")
    ap.add_argument("--u_max", type=float, default=1900, help="Upper PWM bound")
    ap.add_argument("--save", type=str, default="thruster_inversepoly_model.npz", help="Output model file")
    ap.add_argument("--plot", nargs='?', const=True, default=False, help="Plot forward fits; optional filename.")
    ap.add_argument("--plot_voltages", nargs='*', default=None, help="Limit plot to specific voltages")
    args = ap.parse_args()

    if args.data_root is None:
        bluerov_package_path = get_package_path('bluerov')
        thruster_params_path = bluerov_package_path + "/thruster_data/"
    else:
        thruster_params_path = args.data_root

    model = train_from_folder(thruster_params_path, f_eps=args.f_eps, deg=args.deg, lam=args.lam,
                              monotone=(not args.no_monotone), tau_volt=args.tau, hysteresis=args.hyst,
                              u_min=args.u_min, u_max=args.u_max)

    print("Example:", "u( f=0.5N, V=15.0V, dt=0.02 ) =", model.command(0.5, 15.0, 0.02))
    model.save(args.save)
    print(f"Saved model to {args.save}")

    if args.plot:
        save = args.plot if isinstance(args.plot, str) else None
        plot_forward_fits(thruster_params_path, model, save_path=save, show=(save is None),
                          volts=args.plot_voltages)


if __name__ == "__main__":
    main()
