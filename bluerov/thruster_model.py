import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

from common import utils_math
from common.my_package_path import get_package_path
import casadi as ca

# --- Data Normalization Utilities ---

def normalize_pwm(pwm, pwm_min=1100, pwm_max=1900):
    return 2 * (pwm - pwm_min) / (pwm_max - pwm_min) - 1

def denormalize_pwm(pwm_norm, pwm_min=1100, pwm_max=1900):
    return ((pwm_norm + 1) * (pwm_max - pwm_min) / 2) + pwm_min

# --- Polynomial Fitting Utilities ---

def fit_polynomials(data, order=3, x_key='pwm_norm', y_key='force', zero_intercept=False):
    """Fit polynomials for each voltage in data."""
    poly_coeffs = {}
    for v, d in data.items():
        X = d[x_key][:, np.newaxis]
        y = d[y_key]
        if zero_intercept and order == 1:
            m, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            coeffs = np.array([m[0], 0.0])
        else:
            coeffs = np.polyfit(d[x_key], d[y_key], order)
        poly_coeffs[v] = coeffs
        print(coeffs)
    return poly_coeffs

def fit_polynomials_inverse(data, order=3):
    """Fit inverse polynomials: PWM as a function of Force."""
    return fit_polynomials(data, order, x_key='force', y_key='pwm_norm')

def fit_piecewise_polynomials(data, order=2):
    # Print PWM_norm range for each voltage where force is zero
    print("PWM_norm range for each voltage where force is zero:")
    for v in data:
        pwm_norm = data[v]['pwm_norm']
        force = data[v]['force']
        zero_indices = np.where(np.isclose(force, 0, atol=1e-3))[0]
        if zero_indices.size > 0:
            pwm_norm_zero = pwm_norm[zero_indices]
            print(f"{v}V: PWM_norm zero range = [{pwm_norm_zero.min():.4f}, {pwm_norm_zero.max():.4f}]")
        else:
            print(f"{v}V: No PWM_norm values where force is zero found.")
    
    poly_coeffs_neg = {}
    poly_coeffs_pos = {}
    for v in data:
        pwm_norm = data[v]['pwm_norm']
        force = data[v]['force']
        zero_indices = np.where(np.isclose(force, 0, atol=1e-3))[0]
        if zero_indices.size > 0:
            zero_min = pwm_norm[zero_indices].min()
            zero_max = pwm_norm[zero_indices].max()
            # Negative range: from -1.0 up to and including zero_min
            mask_neg = (pwm_norm >= -1.0) & (pwm_norm <= zero_min)
            if np.sum(mask_neg) >= 3:
                coeffs_neg = np.polyfit(pwm_norm[mask_neg], force[mask_neg], order)
                poly_coeffs_neg[v] = coeffs_neg
                print(f"{v}V quadratic fit (neg: -1.0 to {zero_min:.4f}):", coeffs_neg)
            else:
                print(f"{v}V: Not enough points for negative quadratic fit.")
            # Positive range: from zero_max up to and including 1.0
            mask_pos = (pwm_norm >= zero_max) & (pwm_norm <= 1.0)
            if np.sum(mask_pos) >= 3:
                coeffs_pos = np.polyfit(pwm_norm[mask_pos], force[mask_pos], order)
                poly_coeffs_pos[v] = coeffs_pos
                print(f"{v}V quadratic fit (pos: {zero_max:.4f} to 1.0):", coeffs_pos)
            else:
                print(f"{v}V: Not enough points for positive quadratic fit.")
        else:
            print(f"{v}V: No PWM_norm values where force is zero found.")
    return poly_coeffs_neg, poly_coeffs_pos

def fit_piecewise_polynomials_advanced(data):
    # Enforce polynomial to be zero at the edge (c=0), and extremum at that edge
    # For quadratic: f(x) = a*(x-x0)^2, where x0 is the edge (zero_min or zero_max)
    # So: f(x) = a*(x-x0)^2, which is zero at x0 and has extremum there
    # f(x) = a*x^2 - 2*a*x0*x + a*x0^2
    poly_coeffs_neg = {}
    poly_coeffs_pos = {}
    print("Coefficients for piecewise fits with extremum at zero-force edge: f(x) = a*(x-x0)^2 = a*x^2 - 2*a*x0*x + a*x0^2 = [a, b, c]")
    for v in data:
        pwm_norm = data[v]['pwm_norm']
        force = data[v]['force']
        zero_indices = np.where(np.isclose(force, 0, atol=1e-3))[0]
        if zero_indices.size > 0:
            zero_min = pwm_norm[zero_indices].min()
            zero_max = pwm_norm[zero_indices].max()
            # Negative range: from -1.0 up to and including zero_min
            mask_neg = (pwm_norm >= -1.0) & (pwm_norm <= zero_min)
            x_neg = pwm_norm[mask_neg]
            y_neg = force[mask_neg]
            if len(x_neg) >= 2:
                # Fit a in f(x) = a*(x-zero_min)^2
                X = (x_neg - zero_min) ** 2
                a, _, _, _ = np.linalg.lstsq(X[:, np.newaxis], y_neg, rcond=None)
                # Standard polynomial: a*x^2 - 2*a*zero_min*x + a*zero_min^2
                poly_coeffs = [a[0], -2*a[0]*zero_min, a[0]*zero_min**2]
                poly_coeffs_neg[v] = poly_coeffs
                # Convert np.float64 to plain Python float for printing
                poly_coeffs = [float(c) for c in poly_coeffs]
                print(f"{v}V edge-fit (neg: -1.0 to {zero_min:.4f}):", poly_coeffs) 
            else:
                print(f"{v}V: Not enough points for negative edge fit.")
            # Positive range: from zero_max up to and including 1.0
            mask_pos = (pwm_norm >= zero_max) & (pwm_norm <= 1.0)
            x_pos = pwm_norm[mask_pos]
            y_pos = force[mask_pos]
            if len(x_pos) >= 2:
                X = (x_pos - zero_max) ** 2
                a, _, _, _ = np.linalg.lstsq(X[:, np.newaxis], y_pos, rcond=None)
                poly_coeffs = [a[0], -2*a[0]*zero_max, a[0]*zero_max**2]
                poly_coeffs_pos[v] = poly_coeffs
                poly_coeffs = [float(c) for c in poly_coeffs]
                print(f"{v}V edge-fit (pos: {zero_max:.4f} to 1.0):", poly_coeffs)
            else:
                print(f"{v}V: Not enough points for positive edge fit.")
        else:
            print(f"{v}V: No PWM_norm values where force is zero found.")
    
    return poly_coeffs_neg, poly_coeffs_pos

# --- Nonlinear Fitting Utilities ---

def fit_tanh_inverse(data):
    """Fit tanh function: PWM = a * tanh(b * force + c) + d."""
    def tanh_func(force, a, b, c, d):
        return a * np.tanh(b * force + c) + d

    tanh_params = {}
    for v, dct in data.items():
        try:
            params, _ = curve_fit(tanh_func, dct['force'], dct['pwm'], p0=[1.0, 1.0, 0.0, 0.0])
            tanh_params[v] = params
        except RuntimeError:
            tanh_params[v] = None
    return tanh_params

# --- Bivariate Fitting Utilities ---

def fit_bivariate_piecewise(data, zero_neg=-0.075, zero_pos=0.075, order=2,selected_voltages=[12, 14, 16]):
    """
    Fit bivariate quadratic functions for negative and positive PWM_norm ranges:
    f_neg(PWM_norm, V) = a0*V*(PWM_norm-zero_neg)^2
    f_pos(PWM_norm, V) = a1*V*(PWM_norm-zero_pos)^2
    This ensures zero at zero_neg/zero_pos, and min/max at those points.
    Returns: coeffs_neg, coeffs_pos (each: [a, zero])
    """
    # Only use data for selected voltages
    pwm_norm_all = []
    voltage_all = []
    force_all = []
    pwm_norm_all_pos = []
    voltage_all_pos = []
    force_all_pos = []
    for v in selected_voltages:
        pwm_norm = data[v]['pwm_norm']
        force = data[v]['force']
        mask_neg = (pwm_norm >= -1.0) & (pwm_norm <= zero_neg)
        pwm_norm_all.extend(pwm_norm[mask_neg])
        voltage_all.extend([v] * np.sum(mask_neg))
        force_all.extend(force[mask_neg])
        mask_pos = (pwm_norm >= zero_pos) & (pwm_norm <= 1.0)
        pwm_norm_all_pos.extend(pwm_norm[mask_pos])
        voltage_all_pos.extend([v] * np.sum(mask_pos))
        force_all_pos.extend(force[mask_pos])

    # Negative side fit: f_neg(PWM_norm, V) = a*V*(PWM_norm-zero_neg)^2
    X_neg = np.array(voltage_all) * (np.array(pwm_norm_all) - zero_neg) ** order
    y_neg = np.array(force_all)
    a_neg, _, _, _ = np.linalg.lstsq(X_neg[:, np.newaxis], y_neg, rcond=None)

    # Positive side fit: f_pos(PWM_norm, V) = a*V*(PWM_norm-zero_pos)^2
    X_pos = np.array(voltage_all_pos) * (np.array(pwm_norm_all_pos) - zero_pos) ** order
    y_pos = np.array(force_all_pos)
    a_pos, _, _, _ = np.linalg.lstsq(X_pos[:, np.newaxis], y_pos, rcond=None)

    print(f"Bivariate negative fit: f_neg(PWM_norm, V) = {a_neg[0]:.4f} * V * (PWM_norm - {zero_neg})^{order}")
    print(f"Bivariate positive fit: f_pos(PWM_norm, V) = {a_pos[0]:.4f} * V * (PWM_norm - {zero_pos})^{order}")
    return (a_neg[0], zero_neg), (a_pos[0], zero_pos)

def _s_neg(x, x0):  # links
    return np.maximum(0.0, x0 - x)

def _s_pos(x, x0):  # rechts
    return np.maximum(0.0, x - x0)

# Modell 2: T = ± A(V) * (1 - exp(-k s))^3
def model2_neg(x, V, alpha1, k, x0):  # Kraft < 0
    A = alpha1 * V
    s = _s_neg(x, x0)
    return -A * (1.0 - np.exp(-k * s))**3

def model2_pos(x, V, alpha1, k, x0):  # Kraft > 0
    A = alpha1 * V
    s = _s_pos(x, x0)
    return +A * (1.0 - np.exp(-k * s))**3

def fit_bivariate_piecewise_v2(data, zero_neg=-0.075, zero_pos=0.075, selected_voltages=[12,14,16]):
    # Daten sammeln
    x_neg, V_neg, y_neg = [], [], []
    x_pos, V_pos, y_pos = [], [], []
    for v in selected_voltages:
        x = np.asarray(data[v]['pwm_norm'])
        y = np.asarray(data[v]['force'])
        mneg = (x >= -1.0) & (x <= zero_neg)
        mpos = (x >=  zero_pos) & (x <= 1.0)
        x_neg.extend(x[mneg]); V_neg.extend([v]*np.sum(mneg)); y_neg.extend(y[mneg])
        x_pos.extend(x[mpos]); V_pos.extend([v]*np.sum(mpos)); y_pos.extend(y[mpos])

    x_neg = np.asarray(x_neg); V_neg = np.asarray(V_neg); y_neg = np.asarray(y_neg)
    x_pos = np.asarray(x_pos); V_pos = np.asarray(V_pos); y_pos = np.asarray(y_pos)

    # Fit (nichtlinear). Startwerte und Bounds
    p0_neg = ( np.max(-y_neg)/(np.max(V_neg)+1e-9),  20.0, zero_neg)  # (alpha1, k, x0)
    p0_pos = ( np.max( y_pos)/(np.max(V_pos)+1e-9),  20.0, zero_pos)

    # alpha1>0, k>0; x0 frei (oder fixieren, wenn du genau -/+0.075 willst)
    bounds = ([0.0, 1e-6, zero_neg-0.02], [np.inf, np.inf, zero_neg+0.02])
    popt_neg, _ = curve_fit(lambda xv, a1, k, x0: model2_neg(xv[:,0], xv[:,1], a1, k, x0),
                            np.column_stack([x_neg, V_neg]), y_neg, p0=p0_neg, bounds=bounds, maxfev=20000)

    bounds = ([0.0, 1e-6, zero_pos-0.02], [np.inf, np.inf, zero_pos+0.02])
    popt_pos, _ = curve_fit(lambda xv, a1, k, x0: model2_pos(xv[:,0], xv[:,1], a1, k, x0),
                            np.column_stack([x_pos, V_pos]), y_pos, p0=p0_pos, bounds=bounds, maxfev=20000)

    (alpha1_n, k_n, x0_n) = popt_neg
    (alpha1_p, k_p, x0_p) = popt_pos

    print(f"V2 neg:  T = -({alpha1_n:.4g}·V) * (1 - exp(-{k_n:.3g}·max(0,{x0_n:.4f}-x)))^3")
    print(f"V2 pos:  T = +({alpha1_p:.4g}·V) * (1 - exp(-{k_p:.3g}·max(0,x-{x0_p:.4f})))^3")
    return (alpha1_n, k_n, x0_n), (alpha1_p, k_p, x0_p)

def model3_neg(x, V, alpha1, b, x0):
    A = alpha1 * V
    s = _s_neg(x, x0)
    return -A * (s**3) / (1.0 + b * s**3)

def model3_pos(x, V, alpha1, b, x0):
    A = alpha1 * V
    s = _s_pos(x, x0)
    return +A * (s**3) / (1.0 + b * s**3)

def fit_bivariate_piecewise_v3(data, zero_neg=-0.075, zero_pos=0.075, selected_voltages=[12,14,16]):
    x_neg, V_neg, y_neg = [], [], []
    x_pos, V_pos, y_pos = [], [], []
    for v in selected_voltages:
        x = np.asarray(data[v]['pwm_norm'])
        y = np.asarray(data[v]['force'])
        mneg = (x >= -1.0) & (x <= zero_neg)
        mpos = (x >=  zero_pos) & (x <= 1.0)
        x_neg.extend(x[mneg]); V_neg.extend([v]*np.sum(mneg)); y_neg.extend(y[mneg])
        x_pos.extend(x[mpos]); V_pos.extend([v]*np.sum(mpos)); y_pos.extend(y[mpos])

    x_neg = np.asarray(x_neg); V_neg = np.asarray(V_neg); y_neg = np.asarray(y_neg)
    x_pos = np.asarray(x_pos); V_pos = np.asarray(V_pos); y_pos = np.asarray(y_pos)

    p0_neg = ( np.max(-y_neg)/(np.max(V_neg)+1e-9),  100.0, zero_neg)  # (alpha1, b, x0)
    p0_pos = ( np.max( y_pos)/(np.max(V_pos)+1e-9),  100.0, zero_pos)

    bounds = ([0.0, 1e-9, zero_neg-0.02], [np.inf, np.inf, zero_neg+0.02])
    popt_neg, _ = curve_fit(lambda xv, a1, b, x0: model3_neg(xv[:,0], xv[:,1], a1, b, x0),
                            np.column_stack([x_neg, V_neg]), y_neg, p0=p0_neg, bounds=bounds, maxfev=20000)

    bounds = ([0.0, 1e-9, zero_pos-0.02], [np.inf, np.inf, zero_pos+0.02])
    popt_pos, _ = curve_fit(lambda xv, a1, b, x0: model3_pos(xv[:,0], xv[:,1], a1, b, x0),
                            np.column_stack([x_pos, V_pos]), y_pos, p0=p0_pos, bounds=bounds, maxfev=20000)

    (alpha1_n, b_n, x0_n) = popt_neg
    (alpha1_p, b_p, x0_p) = popt_pos

    print(f"V3 neg:  T = -( {alpha1_n:.4g}·V ) * s^3/(1+{b_n:.3g}·s^3),  s=max(0,{x0_n:.4f}-x)")
    print(f"V3 pos:  T = +( {alpha1_p:.4g}·V ) * s^3/(1+{b_p:.3g}·s^3),  s=max(0,x-{x0_p:.4f})")
    return (alpha1_n, b_n, x0_n), (alpha1_p, b_p, x0_p)

def model_exp_neg(x, V, alpha1, k, x0):
    s = _s_neg(x, x0)
    return -alpha1 * V * (np.exp(k*s) - 1 - k*s - 0.5*(k*s)**2)

def model_exp_pos(x, V, alpha1, k, x0):
    s = _s_pos(x, x0)
    return +alpha1 * V * (np.exp(k*s) - 1 - k*s - 0.5*(k*s)**2)

def fit_piecewise_exp(data, zero_neg=-0.075, zero_pos=0.075, selected_voltages=[12,14,16]):
    x_neg, V_neg, y_neg = [], [], []
    x_pos, V_pos, y_pos = [], [], []
    for v in selected_voltages:
        x = np.asarray(data[v]['pwm_norm'])
        y = np.asarray(data[v]['force'])
        mneg = (x >= -1.0) & (x <= zero_neg)
        mpos = (x >=  zero_pos) & (x <= 1.0)
        x_neg.extend(x[mneg]); V_neg.extend([v]*np.sum(mneg)); y_neg.extend(y[mneg])
        x_pos.extend(x[mpos]); V_pos.extend([v]*np.sum(mpos)); y_pos.extend(y[mpos])
    x_neg = np.asarray(x_neg); V_neg = np.asarray(V_neg); y_neg = np.asarray(y_neg)
    x_pos = np.asarray(x_pos); V_pos = np.asarray(V_pos); y_pos = np.asarray(y_pos)

    # Fix x0_n and x0_p to zero_neg and zero_pos
    def fneg(XV, a1, k):
        return model_exp_neg(XV[:,0], XV[:,1], a1, k, zero_neg)
    def fpos(XV, a1, k):
        return model_exp_pos(XV[:,0], XV[:,1], a1, k, zero_pos)

    p0n = (np.max(-y_neg)/(np.max(V_neg)+1e-9), 10.0)
    p0p = (np.max(y_pos)/(np.max(V_pos)+1e-9), 10.0)

    bounds = ([0.0, 1e-9], [np.inf, np.inf])

    popt_neg, _ = curve_fit(fneg, np.column_stack([x_neg, V_neg]), y_neg, p0=p0n, bounds=bounds, maxfev=20000)
    popt_pos, _ = curve_fit(fpos, np.column_stack([x_pos, V_pos]), y_pos, p0=p0p, bounds=bounds, maxfev=20000)

    alpha1_n, k_n = popt_neg
    alpha1_p, k_p = popt_pos

    print(f"Exp neg: T = -({alpha1_n:.4g}·V) * (exp({k_n:.3g}·max(0,{zero_neg:.4f}-x)) - 1 - {k_n:.3g}·s - 0.5*({k_n:.3g}·s)^2), s=max(0,{zero_neg:.4f}-x)")
    print(f"Exp pos: T = +({alpha1_p:.4g}·V) * (exp({k_p:.3g}·max(0,x-{zero_pos:.4f})) - 1 - {k_p:.3g}·s - 0.5*({k_p:.3g}·s)^2), s=max(0,x-{zero_pos:.4f})")
    return (alpha1_n, k_n, zero_neg), (alpha1_p, k_p, zero_pos)

def _softplus_stable(s, beta):
    # s: Abstand zur Kniestelle (>=0), beta>0
    t = beta * s
    return (np.maximum(t, 0.0) + np.log1p(np.exp(-np.abs(t)))) / beta

def _g(s, beta):
    # Kern mit f(x0)=0: softplus(beta*s)/beta - softplus(0)/beta
    return _softplus_stable(s, beta) - (np.log(2.0) / beta)

def _s_neg(x, x0):
    # linke Seite: Abstand nach links vom Knie (>=0)
    return np.maximum(0.0, x0 - x)

def _s_pos(x, x0):
    # rechte Seite: Abstand nach rechts vom Knie (>=0)
    return np.maximum(0.0, x - x0)

def _model_softplus_neg(x, V, alpha1, beta, x0):
    # "down-looking" Softplus: weit links ~ -alpha1*V*s (linear negativ),
    # nahe x0 glatt -> 0; durchgehend konkav (f''<=0)
    s = _s_neg(x, x0)
    return - V * alpha1 * _g(s, beta) * s

def _model_softplus_pos(x, V, alpha1, beta, x0):
    # "up-looking" Softplus: weit rechts ~ +alpha1*V*s (linear positiv),
    # nahe x0 glatt -> 0; durchgehend konvex (f''>=0)
    s = _s_pos(x, x0)
    return + V * alpha1 * _g(s, beta) * s

def fit_bivariate_piecewise_softplus(
    data,
    zero_neg=-0.075,
    zero_pos=0.075,
    selected_voltages=[12, 14, 16],
    fit_x0=False,
    beta_bounds=(1e-3, 1e3),
    print_summary=True
):
    """
    Piecewise-Softplus-Fit (glatt, ohne Sättigung):
      NEG: f = - V * alpha1 * (softplus(beta*(x0-x)) - softplus(0))    [konkav]
      POS: f = + V * alpha1 * (softplus(beta*(x -x0)) - softplus(0))    [konvex]

    Args:
        data: dict[V] -> {'pwm_norm': array, 'force': array}
        zero_neg, zero_pos: Kniepunkte (linker/rechter Zweig)
        selected_voltages: Spannungen, die gefittet werden
        fit_x0: Kniestellen mitfitten (True) oder fix lassen (False)
        beta_bounds: (min,max) für beta (Aggressivität)
        print_summary: formatiertes Modell-Statement ausgeben

    Returns:
        params_neg, params_pos:
           wenn fit_x0=False: (alpha1, beta, x0_fix)
           wenn fit_x0=True : (alpha1, beta, x0_fit)
    """
    # Daten sammeln
    x_neg, V_neg, y_neg = [], [], []
    x_pos, V_pos, y_pos = [], [], []
    for v in selected_voltages:
        x = np.asarray(data[v]['pwm_norm'])
        y = np.asarray(data[v]['force'])
        mneg = (x >= -1.0) & (x <= zero_neg)
        mpos = (x >=  zero_pos) & (x <= 1.0)
        x_neg.extend(x[mneg]); V_neg.extend([v]*np.sum(mneg)); y_neg.extend(y[mneg])
        x_pos.extend(x[mpos]); V_pos.extend([v]*np.sum(mpos)); y_pos.extend(y[mpos])

    x_neg = np.asarray(x_neg); V_neg = np.asarray(V_neg); y_neg = np.asarray(y_neg)
    x_pos = np.asarray(x_pos); V_pos = np.asarray(V_pos); y_pos = np.asarray(y_pos)

    # Startwerte
    eps = 1e-12
    a0_neg = max(1e-6, (np.nanmax(-y_neg) if y_neg.size else 1.0) / (np.nanmax(V_neg)+eps if V_neg.size else 1.0))
    a0_pos = max(1e-6, (np.nanmax( y_pos) if y_pos.size else 1.0) / (np.nanmax(V_pos)+eps if V_pos.size else 1.0))
    beta0   = 20.0

    if not fit_x0:
        # Kniepunkte fixieren
        fneg = lambda XV, a1, beta: _model_softplus_neg(XV[:,0], XV[:,1], a1, beta, zero_neg)
        fpos = lambda XV, a1, beta: _model_softplus_pos(XV[:,0], XV[:,1], a1, beta, zero_pos)

        bounds2 = ([0.0, beta_bounds[0]], [np.inf, beta_bounds[1]])

        popt_neg, _ = curve_fit(fneg, np.column_stack([x_neg, V_neg]), y_neg,
                                p0=(a0_neg, beta0), bounds=bounds2, maxfev=20000)
        popt_pos, _ = curve_fit(fpos, np.column_stack([x_pos, V_pos]), y_pos,
                                p0=(a0_pos, beta0), bounds=bounds2, maxfev=20000)

        alpha1_n, beta_n = popt_neg
        alpha1_p, beta_p = popt_pos
        params_neg = (alpha1_n, beta_n, zero_neg)
        params_pos = (alpha1_p, beta_p, zero_pos)

        if print_summary:
            print(f"Softplus NEG: f = - V * {alpha1_n:.4g} * (softplus({beta_n:.4g}·(x0 - x)) - softplus(0)), x0={zero_neg:.4f}")
            print(f"Softplus POS: f = + V * {alpha1_p:.4g} * (softplus({beta_p:.4g}·(x - x0)) - softplus(0)), x0={zero_pos:.4f}")
        return params_neg, params_pos

    else:
        # Kniepunkte mitfitten (enge Bounds sinnvoll)
        fneg = lambda XV, a1, beta, x0: _model_softplus_neg(XV[:,0], XV[:,1], a1, beta, x0)
        fpos = lambda XV, a1, beta, x0: _model_softplus_pos(XV[:,0], XV[:,1], a1, beta, x0)

        x0n_min, x0n_max = zero_neg - 0.02, zero_neg + 0.02
        x0p_min, x0p_max = zero_pos - 0.02, zero_pos + 0.02

        bndsn = ([0.0, beta_bounds[0], x0n_min], [np.inf, beta_bounds[1], x0n_max])
        bndsp = ([0.0, beta_bounds[0], x0p_min], [np.inf, beta_bounds[1], x0p_max])

        popt_neg, _ = curve_fit(fneg, np.column_stack([x_neg, V_neg]), y_neg,
                                p0=(a0_neg, beta0, zero_neg), bounds=bndsn, maxfev=20000)
        popt_pos, _ = curve_fit(fpos, np.column_stack([x_pos, V_pos]), y_pos,
                                p0=(a0_pos, beta0, zero_pos), bounds=bndsp, maxfev=20000)

        alpha1_n, beta_n, x0_n = popt_neg
        alpha1_p, beta_p, x0_p = popt_pos
        params_neg = (alpha1_n, beta_n, x0_n)
        params_pos = (alpha1_p, beta_p, x0_p)

        if print_summary:
            print(f"Softplus NEG: f = - V * {alpha1_n:.4g} * (softplus({beta_n:.4g}·(x0 - x)) - softplus(0)), x0={x0_n:.4f}")
            print(f"Softplus POS: f = + V * {alpha1_p:.4g} * (softplus({beta_p:.4g}·(x - x0)) - softplus(0)), x0={x0_p:.4f}")
        return params_neg, params_pos

def fit_bivariate_polynomial(data):
    pwm_data = np.concatenate([d['pwm'] for d in data.values()])
    voltage_data = np.concatenate([np.full_like(d['pwm'], v) for v, d in data.items()])
    thrust_data = np.concatenate([d['force'] for d in data.values()])
    poly = PolynomialFeatures(degree=3, include_bias=True)
    X = np.column_stack((pwm_data, voltage_data))
    X_poly_full = poly.fit_transform(X)
    powers = poly.powers_
    valid_idx = [i for i, (pwm_pow, volt_pow) in enumerate(powers) if volt_pow <= 3]
    X_poly = X_poly_full[:, valid_idx]
    model = LinearRegression()
    model.fit(X_poly, thrust_data)
    selected_powers = powers[valid_idx]
    print("Selected polynomial powers (PWM, Voltage):\n", selected_powers)
    print("Coefficients:\n", model.coef_)
    terms = []
    for coef, (pwm_pow, volt_pow) in zip(model.coef_, selected_powers):
        term = ""
        if pwm_pow == 0 and volt_pow == 0:
            term = f"{coef:.4g}"
        else:
            parts = []
            if pwm_pow > 0:
                parts.append(f"PWM^{pwm_pow}" if pwm_pow > 1 else "PWM")
            if volt_pow > 0:
                parts.append(f"V^{volt_pow}" if volt_pow > 1 else "V")
            term = f"{coef:.4g}*" + "*".join(parts)
        terms.append(term)
    poly_str = " + ".join(terms)
    print("F(PWM, V) =", poly_str)
    return model, poly, valid_idx

def piecewise_polynomial_function(data, poly_coeffs_neg, poly_coeffs_pos):
    def make_piecewise_poly_func(zero_min, zero_max, coeffs_neg, coeffs_pos):
        """
        Returns a function f(x) that evaluates:
        - coeffs_neg polynomial for x in [-1.0, zero_min]
        - 0 for x in (zero_min, zero_max)
        - coeffs_pos polynomial for x in [zero_max, 1.0]
        """
        def f(x):
            x = np.asarray(x)
            out = np.zeros_like(x, dtype=float)
            mask_neg = (x >= -1.0) & (x <= zero_min)
            mask_pos = (x >= zero_max) & (x <= 1.0)
            out[mask_neg] = np.polyval(coeffs_neg, x[mask_neg])
            out[mask_pos] = np.polyval(coeffs_pos, x[mask_pos])
            # Values in (zero_min, zero_max) remain zero
            return out if out.shape != () else float(out)
        return f

    # Generate piecewise polynomial functions for each voltage
    piecewise_poly_funcs = {}
    for v in data:
        pwm_norm = data[v]['pwm_norm']
        force = data[v]['force']
        zero_indices = np.where(np.isclose(force, 0, atol=1e-3))[0]
        if zero_indices.size > 0 and v in poly_coeffs_neg and v in poly_coeffs_pos:
            zero_min = pwm_norm[zero_indices].min()
            zero_max = pwm_norm[zero_indices].max()
            piecewise_poly_funcs[v] = make_piecewise_poly_func(
                zero_min, zero_max, poly_coeffs_neg[v], poly_coeffs_pos[v]
            )
        else:
            piecewise_poly_funcs[v] = lambda x: np.zeros_like(x, dtype=float)
    return piecewise_poly_funcs

# --- Interpolation Utility ---

def thrust_interpolated(pwm_val, voltage_val, voltages, poly_coeffs):
    """Interpolate thrust for a given PWM and voltage."""
    v_arr = np.array(voltages)
    voltage_val = np.clip(voltage_val, v_arr.min(), v_arr.max())
    idx = np.searchsorted(v_arr, voltage_val)
    if idx == 0:
        v0, v1 = v_arr[0], v_arr[1]
    elif idx == len(v_arr):
        v0, v1 = v_arr[-2], v_arr[-1]
    else:
        v0, v1 = v_arr[idx-1], v_arr[idx]
    f0 = np.polyval(poly_coeffs[v0], pwm_val)
    f1 = np.polyval(poly_coeffs[v1], pwm_val)
    return f0 if v1 == v0 else f0 + (f1 - f0) * (voltage_val - v0) / (v1 - v0)

# --- Plotting Utilities ---

def plot_polynomials(data, poly_coeffs, x_key='pwm_norm', y_key='force', xlabel='PWM [mus]', ylabel='Force [N]', title=None):
    # plt.figure(figsize=(8, 6))
    colors = ['b', 'g', 'r', 'c', 'm', 'y']
    for (v, color) in zip(data.keys(), colors):
        x_range = np.linspace(np.min(data[v][x_key]), np.max(data[v][x_key]), 200)
        y_fit = np.polyval(poly_coeffs[v], x_range)
        plt.plot(x_range, y_fit, color=color, linestyle='--', label=f'Poly fit {v}V')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

def plot_piecewise_fit(data, poly_coeffs_neg, poly_coeffs_pos):
    colors = ['b', 'g', 'r', 'c', 'm', 'y']
    for idx, v in enumerate(data.keys()):
        pwm_norm = data[v]['pwm_norm']
        force = data[v]['force']
        zero_indices = np.where(np.isclose(force, 0, atol=1e-3))[0]
        if zero_indices.size > 0:
            zero_min = pwm_norm[zero_indices].min()
            zero_max = pwm_norm[zero_indices].max()
            # Negative fit range
            if v in poly_coeffs_neg:
                x_neg = np.linspace(-1.0, zero_min, 200)
                y_neg = np.polyval(poly_coeffs_neg[v], x_neg)
                plt.plot(x_neg, y_neg, color=colors[idx % len(colors)], linestyle='-', label=f'Neg fit {v}V')
            # Positive fit range
            if v in poly_coeffs_pos:
                x_pos = np.linspace(zero_max, 1.0, 200)
                y_pos = np.polyval(poly_coeffs_pos[v], x_pos)
                plt.plot(x_pos, y_pos, color=colors[idx % len(colors)], linestyle='-', label=f'Pos fit {v}V')
    plt.xlabel('PWM_norm')
    plt.ylabel('Thrust [N]')
    plt.title('Quadratic Fits in Respective Ranges')
    plt.legend()
    plt.grid(True)

def plot_piecewise_fit_functions(x_plot, piecewise_poly_funcs, voltage):
    # Example usage of piecewise polynomial functions
    # Assuming piecewise_poly_funcs is obtained from piecewise_polynomial_function
    y_plot = piecewise_poly_funcs[voltage](x_plot)
    plt.plot(x_plot, y_plot, label=f'Piecewise Fit {voltage}V')
    plt.xlabel('PWM_norm')
    plt.ylabel('Thrust [N]')
    plt.title(f'Piecewise Polynomial Fit for {voltage}V')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_raw_data(data, x_key='pwm_norm', y_key='force', xlabel='PWM [mus]', ylabel='Force [N]', title=None, voltages=[12, 14, 16]):
    # plt.figure(figsize=(8, 6))
    for v in voltages:
        plt.plot(data[v][x_key], data[v][y_key], label=f'{v}V')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.grid(True)

def plot_interpolated_curve(data, voltages, poly_coeffs, interp_voltage=15, x_key='pwm', xlabel='PWM [mus]', ylabel='Force [N]'):
    pwm_range = np.linspace(np.min(data[voltages[0]][x_key]), np.max(data[voltages[0]][x_key]), 200)
    force_interp = [thrust_interpolated(pwm, interp_voltage, voltages, poly_coeffs) for pwm in pwm_range]
    plt.plot(pwm_range, force_interp, color='k', linestyle='--', label=f'Interpolated {interp_voltage}V')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout() 

def plot_simple_physical_model(data):
    # plt.figure(figsize=(8, 6))
    for v, color in zip([12, 14, 16], ['b', 'g', 'r']):
        pwm_norm_range = np.linspace(np.min(data[v]['pwm_norm']), np.max(data[v]['pwm_norm']), 200)
        # K: Linearer fit für die Form thrust = K * (normalized [-1,1]) PWM
        # Fit linear function to approximate K by battery Voltage: K(V_bat) = L * V_bat
        # For datapoints 10V to 20V: L = 2.4481
        # For datapoints 14V to 18V (more or less operating range): L = 2.5116
        thrust = 2.5116 * v * pwm_norm_range
        plt.plot(pwm_norm_range, thrust, color=color, label=f'Thrust = 2.5116*{v}*PWM_norm')
    plt.xlabel('PWM_norm')
    plt.ylabel('Thrust [N]')
    plt.title('Thrust = 2.5116 * Voltage * PWM_norm')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

def plot_bivariate_piecewise_fit(x_plot, coeffs_neg, coeffs_pos, order=2,voltages=[12, 14, 16]):
    def bivariate_piecewise_poly_func(pwm_norm, voltage, coeffs_neg, coeffs_pos):
        """
        Evaluate the bivariate piecewise polynomial:
        - Negative: pwm_norm in [-1, zero_neg]
        - Positive: pwm_norm in [zero_pos, 1]
        - Zero elsewhere
        """
        pwm_norm = np.asarray(pwm_norm)
        out = np.zeros_like(pwm_norm, dtype=float)
        a_neg, zero_neg = coeffs_neg
        a_pos, zero_pos = coeffs_pos
        mask_neg = (pwm_norm >= -1.0) & (pwm_norm <= zero_neg)
        mask_pos = (pwm_norm >= zero_pos) & (pwm_norm <= 1.0)
        out[mask_neg] = a_neg * voltage * (pwm_norm[mask_neg] - zero_neg) ** order
        out[mask_pos] = a_pos * voltage * (pwm_norm[mask_pos] - zero_pos) ** order
        return out if out.shape != () else float(out)
    
    plt.figure()
    for v in voltages:
        plt.plot(x_plot, bivariate_piecewise_poly_func(x_plot, v, coeffs_neg, coeffs_pos),
                 linestyle='-.', label=f'Bivariate {v}V')
    plt.xlabel('PWM_norm')
    plt.ylabel('Thrust [N]')
    plt.title('Bivariate Piecewise Polynomial Fit')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

def plot_model2_piecewise(x_plot, neg_params, pos_params, voltages=[12, 14, 16]):
    """
    Plot the piecewise nonlinear model2 for a list of voltages over x_plot.
    neg_params: (alpha1_n, k_n, x0_n)
    pos_params: (alpha1_p, k_p, x0_p)
    """
    alpha1_n, k_n, x0_n = neg_params
    alpha1_p, k_p, x0_p = pos_params

    plt.figure()
    for v in voltages:
        y_neg = model2_neg(x_plot, v, alpha1_n, k_n, x0_n)
        y_pos = model2_pos(x_plot, v, alpha1_p, k_p, x0_p)
        y_total = y_neg + y_pos
        plt.plot(x_plot, y_total, linestyle='-.', label=f'Model2 {v}V')
    plt.xlabel('PWM_norm')
    plt.ylabel('Thrust [N]')
    plt.title('Model2 Piecewise Nonlinear Fit')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

def plot_model3_piecewise(x_plot, neg_params, pos_params, voltages=[12, 14, 16]):
    """
    Plot the piecewise nonlinear model3 for a list of voltages over x_plot.
    neg_params: (alpha1_n, b_n, x0_n)
    pos_params: (alpha1_p, b_p, x0_p)
    """
    alpha1_n, b_n, x0_n = neg_params
    alpha1_p, b_p, x0_p = pos_params

    plt.figure()
    for v in voltages:
        y_neg = model3_neg(x_plot, v, alpha1_n, b_n, x0_n)
        y_pos = model3_pos(x_plot, v, alpha1_p, b_p, x0_p)
        y_total = y_neg + y_pos
        plt.plot(x_plot, y_total, linestyle='-.', label=f'Model3 {v}V')
    plt.xlabel('PWM_norm')
    plt.ylabel('Thrust [N]')
    plt.title('Model3 Piecewise Nonlinear Fit')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

def plot_piecewise_exp_model(x_plot, neg_params, pos_params, voltages=[12, 14, 16]):
        alpha1_n, k_n, x0_n = neg_params
        alpha1_p, k_p, x0_p = pos_params
        print(f"Exp neg:  T = -({alpha1_n:.4g}·V) * (exp({k_n:.3g}·max(0,{x0_n:.4f}-x)) - 1 - {k_n:.3g}·s - 0.5*({k_n:.3g}·s)^2), s=max(0,{x0_n:.4f}-x)")
        plt.figure()
        for v in voltages:
            y_neg = model_exp_neg(x_plot, v, alpha1_n, k_n, x0_n)
            y_pos = model_exp_pos(x_plot, v, alpha1_p, k_p, x0_p)
            y_total = y_neg + y_pos
            plt.plot(x_plot, y_total, linestyle='-.', label=f'Exp Model {v}V')
        plt.xlabel('PWM_norm')
        plt.ylabel('Thrust [N]')
        plt.title('Piecewise Exponential Model Fit')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

def plot_piecewise_softplus(x_plot, neg_params, pos_params, voltages=[12, 14, 16]):
        alpha1_n, beta_n, x0_n = neg_params
        alpha1_p, beta_p, x0_p = pos_params
        plt.figure()
        for v in voltages:
            y_neg = _model_softplus_neg(x_plot, v, alpha1_n, beta_n, x0_n)
            y_pos = _model_softplus_pos(x_plot, v, alpha1_p, beta_p, x0_p)
            y_total = y_neg + y_pos
            plt.plot(x_plot, y_total, linestyle='-.', label=f'Softplus Model {v}V')
        plt.xlabel('PWM_norm')
        plt.ylabel('Thrust [N]')
        plt.title('Piecewise Softplus Model Fit')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

# --- Model Evaluation ---

def compute_rmse(data, poly_coeffs, x_key='pwm', y_key='force'):
    rmse_all = 0
    print("RMSE for each voltage polynomial fit:")
    for v in data:
        y_fit = np.polyval(poly_coeffs[v], data[v][x_key])
        rmse = np.sqrt(np.mean((data[v][y_key] - y_fit) ** 2))
        print(f"{v}V: RMSE = {rmse:.4f} N")
        rmse_all += rmse
    print(f"Overall RMSE across all voltages: {rmse_all:.4f} N")

# --- CasADi piecewise thrust model with deadzone ---

def casadi_piecewise_smooth_thrust()->ca.Function:
    """
    Symbolic CasADi implementation of the piecewise smooth function f2.
    Args:
        pwm_norm: CasADi SX or MX scalar, normalized PWM in [-1, 1]
        voltage: CasADi SX or MX scalar, battery voltage
    Returns:
        thrust: CasADi SX or MX scalar, thrust value
    """
    voltage = ca.MX.sym('voltage')
    pwm_norm = ca.MX.sym('pwm_norm')

    a = 0.095
    delta = 0.02
    kL = 2.766 * voltage
    kR = 3.556 * voltage

    xL0 = -a - delta
    xL1 = -a + delta
    xR0 =  a - delta
    xR1 =  a + delta
    h = 2 * delta

    def H00(t): return 1 - 10*t**3 + 15*t**4 - 6*t**5
    def H10(t): return t - 6*t**3 + 8*t**4 - 3*t**5
    def H01(t): return 10*t**3 - 15*t**4 + 6*t**5
    def H11(t): return -4*t**3 + 7*t**4 - 3*t**5

    L = lambda x: kL * (x + a)
    R = lambda x: kR * (x - a)

    x = pwm_norm

    # left ramp
    left = ca.if_else(x <= xL0, L(x), 0.0)

    # left C² transition to 0
    t_left = (x - xL0) / h
    left_trans = ca.if_else(
        ca.logic_and(x > xL0, x < xL1),
        L(xL0)*H00(t_left) + h*kL*H10(t_left) + 0.0*H01(t_left) + h*0.0*H11(t_left),
        0.0
    )

    # dead zone
    dead = ca.if_else(ca.logic_and(x >= xL1, x <= xR0), 0.0, 0.0)

    # right C² transition from 0
    t_right = (x - xR0) / h
    right_trans = ca.if_else(
        ca.logic_and(x > xR0, x < xR1),
        0.0*H00(t_right) + h*0.0*H10(t_right) + R(xR1)*H01(t_right) + h*kR*H11(t_right),
        0.0
    )

    # right ramp
    right = ca.if_else(x >= xR1, R(x), 0.0)

    thrust = left + left_trans + dead + right_trans + right

    return ca.Function('pwm_to_thrust', [pwm_norm, voltage], [thrust])

def thrust_from_pwm()->ca.Function:
    """
    Symbolic CasADi implementation of the piecewise smooth function f2.
    Args:
        pwm_norm: CasADi SX or MX scalar, normalized PWM in [-1, 1]
        voltage: CasADi SX or MX scalar, battery voltage
    Returns:
        thrust: CasADi SX or MX scalar, thrust value
    """
    pwm_norm = ca.MX.sym('pwm_norm')
    voltage = ca.MX.sym('voltage')

    L = ca.DM(2.5166)

    a = 0.095
    a = 0.02
    delta = 0.02
    kL = 2.766 * voltage
    kR = 3.556 * voltage

    xL0 = -a - delta
    xL1 = -a + delta
    xR0 =  a - delta
    xR1 =  a + delta
    h = 2 * delta

    # Hermite basis functions as CasADi symbolic expressions for smooth transitions
    def H00(t): return 1 - 10*t**3 + 15*t**4 - 6*t**5
    def H10(t): return t - 6*t**3 + 8*t**4 - 3*t**5
    def H01(t): return 10*t**3 - 15*t**4 + 6*t**5
    def H11(t): return -4*t**3 + 7*t**4 - 3*t**5

    t_left = (pwm_norm - xL0) / h
    t_right = (pwm_norm - xR0) / h

    thrust = ca.if_else(pwm_norm <= xL0, 
                        kL * (pwm_norm + a),
                        ca.if_else(pwm_norm <= xL1,
                                kL * (xL0 + a) * H00(t_left) + h * kL * H10(t_left),
                                    ca.if_else(pwm_norm < xR0, 
                                            0.0,
                                            ca.if_else(pwm_norm < xR1,
                                                        kR * (xR1 - a) * H01(t_right) + h * kR * H11(t_right),
                                                        kR * (pwm_norm - a)
                                                        )
                                            )
                                    )
                        )
    
    # thrust = L * voltage * pwm_norm

    return ca.Function('pwm_to_thrust', [pwm_norm, voltage], [thrust])

def thrust_from_pwm_vec(n=8) -> ca.Function:
    pwm_vec = ca.MX.sym('pwm', n, 1)
    V       = ca.MX.sym('V')

    # f1 = pwm_to_thrust_scalar_tanh()
    f1 = thrust_from_pwm()
    fn = f1.map(n)  # elementwise map
    # map expects (1 x n) inputs; reshape and broadcast V
    thrust_1xn = fn(ca.reshape(pwm_vec, 1, n), ca.repmat(V, 1, n))
    thrust_nx1 = ca.reshape(thrust_1xn, n, 1)
    return ca.Function('pwm_to_thrust_vec', [pwm_vec, V], [thrust_nx1]).expand()

def pwm_to_thrust_scalar_smooth() -> ca.Function:
    pwm = ca.MX.sym('pwm')   # scalar in [-1,1]
    V   = ca.MX.sym('V')     # scalar voltage

    # parameters
    a     = 0.095                 # dead-zone half width
    beta  = 40.0                  # edge sharpness (30–100 works well)
    kL    = 2.766 * V             # left slope
    kR    = 3.556 * V             # right slope
    eps   = 1e-9                  # for smooth abs

    # smooth helpers
    softplus = lambda z: (1.0/beta)*ca.log1p(ca.exp(beta*z))   # smooth ReLU
    abs_s    = lambda z: ca.sqrt(z*z + eps)                    # smooth |z|

    # smooth shrink with zero at 0 and ~0 inside [-a,a] but nonzero slope
    shrink = ca.sign(pwm) * ( softplus(abs_s(pwm) - a) - softplus(-a) )

    # split into smooth positive/negative parts to apply kR/kL
    y_pos = 0.5*(shrink + abs_s(shrink))   # ≈ max(shrink,0)
    y_neg = 0.5*(shrink - abs_s(shrink))   # ≈ min(shrink,0)

    thrust = kR * y_pos + kL * y_neg
    return ca.Function('pwm_to_thrust_scalar_smooth', [pwm, V], [thrust]).expand()

def pwm_to_thrust_scalar_tanh():
    pwm = ca.MX.sym('pwm')   # scalar in [-1,1]
    V   = ca.MX.sym('V')     # scalar voltage

    k = 10.0

    tau_r = lambda z: 0.5 + 0.5 * ca.tanh(k*z)  # smooth step function
    tau_l = lambda z: 0.5 - 0.5 * ca.tanh(k*z)  # smooth step function



    a    = 0.095
    # beta = 1.0/a             # ~10.526; tune 0.7/a .. 1.5/a
    # eps  = 1e-9

    kL = 2.766 * V
    kR = 3.556 * V
    m = 100.0

    # abs_s = lambda z: ca.sqrt(z*z + eps)
    # shrink = pwm - a * ca.tanh(beta * pwm)  # smooth “dead-zone”

    # y_pos = 0.5*(shrink + abs_s(shrink))    # smooth max(shrink, 0)
    # y_neg = 0.5*(shrink - abs_s(shrink))    # smooth min(shrink, 0)

    # thrust = kR * y_pos + kL * y_neg

    thrust = ((pwm/m) + a) * tau_r(pwm) + ((pwm/m) - a) * tau_l(pwm)
    return ca.Function('pwm_to_thrust_scalar_tanh', [pwm, V], [thrust]).expand()

# --- Main Routine ---

def main():
    # Load data
    bluerov_package_path = get_package_path('bluerov')
    thruster_params_path = bluerov_package_path + "/thruster_data/"
    data = utils_math.load_all_thruster_data(thruster_params_path)

    # Normalize PWM
    for v in data:
        data[v]['pwm_norm'] = normalize_pwm(data[v]['pwm'])
    # --- Piecewise Linear Fit ---
    def fit_bivariate_piecewise_linear(data, zero_neg=-0.075, zero_pos=0.075, selected_voltages=[12, 14, 16]):
        """
        Fit bivariate piecewise linear functions:
        For each side (neg/pos), fit f(x, V) = a(V) * (x - x0), where a(V) is linear in V.
        - Negative: fit to points (-1, f_neg) and (zero_neg, 0) for each V
        - Positive: fit to points (zero_pos, 0) and (1, f_pos) for each V
        Returns: (a_neg, zero_neg), (a_pos, zero_pos), where a_neg(V) and a_pos(V) are linear in V
        """
        # Collect (V, f_neg) and (V, f_pos)
        V_list = []
        f_neg_list = []
        f_pos_list = []
        for v in selected_voltages:
            pwm_norm = np.asarray(data[v]['pwm_norm'])
            force = np.asarray(data[v]['force'])
            # Negative side: value at x = -1
            idx_neg = np.argmin(np.abs(pwm_norm + 1.0))
            f_neg = force[idx_neg]
            V_list.append(v)
            f_neg_list.append(f_neg)
            # Positive side: value at x = +1
            idx_pos = np.argmin(np.abs(pwm_norm - 1.0))
            f_pos = force[idx_pos]
            f_pos_list.append(f_pos)
        V_arr = np.array(V_list)
        f_neg_arr = np.array(f_neg_list)
        f_pos_arr = np.array(f_pos_list)
        print(V_arr)
        print(f_neg_arr)
        print(f_pos_arr)
        # Fit f_neg = -a_neg * V * (-1 - zero_neg)
        # Fit f_pos =  a_pos * V * (1 - zero_pos)
        # So a_neg = -f_neg / (V * (-1 - zero_neg)), a_pos = f_pos / (V * (1 - zero_pos))
        a_neg_arr = -f_neg_arr / (V_arr * (-1 - zero_neg))
        a_pos_arr =  f_pos_arr / (V_arr * (1 - zero_pos))
        # Fit a_neg(V) = alpha_neg * V (no offset)
        # For the linear fit, use V * (x - zero_neg) for negative side and V * (x - zero_pos) for positive side
        X_neg = V_arr * (-1 - zero_neg)
        X_pos = V_arr * (1 - zero_pos)
        alpha_neg, _, _, _ = np.linalg.lstsq(X_neg[:, np.newaxis], f_neg_arr, rcond=None)
        alpha_pos, _, _, _ = np.linalg.lstsq(X_pos[:, np.newaxis], f_pos_arr, rcond=None)
        # Return (alpha_neg, zero_neg), (alpha_pos, zero_pos)
        print(f"Piecewise linear NEG: f = -({alpha_neg[0]:.4g}·V) * (x - {zero_neg})")
        print(f"Piecewise linear POS: f = +({alpha_pos[0]:.4g}·V) * (x - {zero_pos})")
        return (alpha_neg[0], zero_neg), (alpha_pos[0], zero_pos)
    # TODO hier ist irgendwas noch falsch

    def bivariate_piecewise_linear_function(zero_neg=-0.075, zero_pos=0.075, coeffs_neg=None, coeffs_pos=None, voltages=[12, 14, 16]):
        """
        Returns a dict of functions f(x, V) for each voltage in voltages.
        Each function evaluates:
          - Negative: f = alpha_neg * V * (x - zero_neg) for x in [-1, zero_neg]
          - Positive: f = alpha_pos * V * (x - zero_pos) for x in [zero_pos, 1]
          - Zero elsewhere
        """
        alpha_neg, zero_neg = coeffs_neg
        alpha_pos, zero_pos = coeffs_pos
        funcs = {}
        for v in voltages:
            def f(x, V=v):
                x = np.asarray(x)
                out = np.zeros_like(x, dtype=float)
                mask_neg = (x >= -1.0) & (x <= zero_neg)
                mask_pos = (x >= zero_pos) & (x <= 1.0)
                out[mask_neg] = alpha_neg * V * (x[mask_neg] - zero_neg)
                out[mask_pos] = alpha_pos * V * (x[mask_pos] - zero_pos)
                return out if out.shape != () else float(out)
            funcs[v] = f
        return funcs

    linear_coeffs_neg, linear_coeffs_pos = fit_bivariate_piecewise_linear(data, zero_neg=-0.095, zero_pos=0.095, selected_voltages=[12, 14, 16])
    piecewise_linear_funcs = bivariate_piecewise_linear_function(zero_neg=-0.095, zero_pos=0.095, coeffs_neg=linear_coeffs_neg, coeffs_pos=linear_coeffs_pos, voltages=[12, 14, 16])
    # Plot the piecewise linear fit for each voltage
    x_plot = np.linspace(-1, 1, 400)
    for v in [12, 14, 16]:
        y_plot = piecewise_linear_funcs[v](x_plot)
        plt.plot(x_plot, y_plot, linestyle='-.', label=f'Piecewise Linear {v}V')
    plt.xlabel('PWM_norm')
    plt.ylabel('Thrust [N]')
    plt.title('Piecewise Linear Fit for Different Voltages')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plot_raw_data(data, x_key='pwm_norm', y_key='force',
                  xlabel='PWM_norm', ylabel='Thrust [N]',
                  title='Raw Thruster Data', voltages=[12, 14, 16])

    plt.show()

    def smootherstep(t):
        t = np.asarray(t)
        out = np.zeros_like(t, dtype=float)
        out = np.where(t <= 0.0, 0.0, out)
        out = np.where(t >= 1.0, 1.0, out)
        mask = (t > 0.0) & (t < 1.0)
        out[mask] = 10*t[mask]**3 - 15*t[mask]**4 + 6*t[mask]**5
        return out

    def f(x, V, a=0.12, delta=0.02):
        ml = 2.845 * V
        mr = 3.657 * V
        L = ml * (x + a)
        R = mr * (x - a)
        sL = smootherstep((x - (-a - delta)) / (2*delta))
        sR = smootherstep((x - ( a - delta)) / (2*delta))
        return L * (1 - sL) + R * sR
    
    def f2(x, V, a=0.095, delta=0.02):
        'The slopes kL and kR are determined by linear fits through '
        'the data points at x=-1 and x=1 for each voltage V from aboves piecewise linear fit'
        kL = 2.766 * V
        kR = 3.556 * V
        L  = lambda x: kL * (x + a)
        R  = lambda x: kR * (x - a)

        'C² Hermite basis functions / quintic Hermite interpolation polynomials'
        'Around the kinks at +-a a C² smooth transition over 2*delta is constructed'
        'The basis functions are there to interpolate between the function values'
        'of this 2*delta range, such that at those edges the function values and'
        'the first and second derivatives match.'
        'Thus the overall function is C² smooth everywhere.'

        def H00(t): return 1 - 10*t**3 + 15*t**4 - 6*t**5
        def H10(t): return t - 6*t**3 + 8*t**4 - 3*t**5
        def H01(t): return 10*t**3 - 15*t**4 + 6*t**5
        def H11(t): return -4*t**3 + 7*t**4 - 3*t**5

        x = np.asarray(x)
        xL0, xL1 = -a - delta, -a + delta
        xR0, xR1 =  a - delta,  a + delta
        h = 2 * delta

        out = np.zeros_like(x, dtype=float)

        # left ramp
        mask_left = x <= xL0
        out[mask_left] = L(x[mask_left])

        # left C² transition to 0
        mask_left_trans = (x > xL0) & (x < xL1)
        t_left = (x[mask_left_trans] - xL0) / h
        v0, s0 = L(xL0), kL
        v1, s1 = 0.0, 0.0
        out[mask_left_trans] = v0*H00(t_left) + h*s0*H10(t_left) + v1*H01(t_left) + h*s1*H11(t_left)

        # dead zone
        mask_dead = (x >= xL1) & (x <= xR0)
        out[mask_dead] = 0.0

        # right C² transition from 0
        mask_right_trans = (x > xR0) & (x < xR1)
        t_right = (x[mask_right_trans] - xR0) / h
        v0, s0 = 0.0, 0.0
        v1, s1 = R(xR1), kR
        out[mask_right_trans] = v0*H00(t_right) + h*s0*H10(t_right) + v1*H01(t_right) + h*s1*H11(t_right)

        # right ramp
        mask_right = x >= xR1
        out[mask_right] = R(x[mask_right])

        return out if out.shape != () else float(out)
    
    # Plot the function f(x, V) for V = 12, 14, 16 over x in [-1, 1]
    x_vals = np.linspace(-1, 1, 400)
    for V in [12, 13.5, 14, 15, 16]:
        y_vals = f2(x_vals, V)
        plt.plot(x_vals, y_vals, linestyle='-.', label=f'V={V}V')
    plt.xlabel('PWM_norm')
    plt.ylabel('Thrust [N]')
    plt.title('Custom Piecewise Function f(x, V) for Different Voltages')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plot_raw_data(data, x_key='pwm_norm', y_key='force',
                  xlabel='PWM_norm', ylabel='Thrust [N]',
                  title='Raw Thruster Data', voltages=[12, 14, 16])
    
    plt.show()

    THRUST = casadi_piecewise_smooth_thrust()
    THRUST2 = thrust_from_pwm()
    THRUST_VEC = thrust_from_pwm_vec(n=2)
    THRUST_SCALAR = pwm_to_thrust_scalar_tanh()

    # Evaluate the CasADi symbolic function for different constant voltages over the full PWM range and plot against raw data

    x_plot = np.linspace(-1, 1, 400)
    voltages_to_plot = [12, 13.5, 14, 15, 16]
    voltages_to_plot = [12, 14, 16]

    plt.figure()
    for V in voltages_to_plot:
        # Evaluate CasADi function for all x_plot at voltage V
        y_casadi = np.array([float(THRUST(x, V)) for x in x_plot])
        y_casadi2 = np.array([float(THRUST2(x, V)) for x in x_plot])
        y_casadi_scalar = np.array([float(THRUST_SCALAR(x, V)) for x in x_plot])
        print([THRUST_VEC(x, V) for x in x_plot])
        plt.plot(x_plot, y_casadi, linestyle='-.', label=f'CasADi V={V}V')
        plt.plot(x_plot, y_casadi2, linestyle='--', label=f'CasADi2 V={V}V')
        plt.plot(x_plot, y_casadi_scalar, linestyle=':', label=f'CasADi Scalar V={V}V')

    # Plot raw data for comparison
    # plot_raw_data(data, x_key='pwm_norm', y_key='force',
    #               xlabel='PWM_norm', ylabel='Thrust [N]',
    #               title='Raw Thruster Data', voltages=[12, 14, 16])

    plt.title('CasADi Piecewise Smooth Thrust vs Raw Data')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return

    poly_coeffs_neg, poly_coeffs_pos = fit_piecewise_polynomials(data, order=2)
    # plot_piecewise_fit(data, poly_coeffs_neg, poly_coeffs_pos)
    poly_coeffs_neg, poly_coeffs_pos = fit_piecewise_polynomials_advanced(data) # quadratic function with zero at edge and extremum at edge
    # plot_piecewise_fit(data, poly_coeffs_neg, poly_coeffs_pos)
    
    # --- Piecewise Polynomial Functions for each voltage ---
    piecewise_poly_funcs = piecewise_polynomial_function(data, poly_coeffs_neg, poly_coeffs_pos) # coefficients must come from fit_piecewise_polynomials_advanced
    x_plot = np.linspace(-1, 1, 400) # pwm_norm range for plotting the piecewise functions

    # neg_params, pos_params = fit_bivariate_piecewise_v2(data, zero_neg=-0.075, zero_pos=0.075, selected_voltages=[12,14,16])
    # neg_params, pos_params = fit_bivariate_piecewise_v3(data, zero_neg=-0.075, zero_pos=0.075, selected_voltages=[12,14,16])

    # plot_model3_piecewise(x_plot, neg_params, pos_params, voltages=[12,14,16])
    # plot_raw_data(data, x_key='pwm_norm', y_key='force',
    #               xlabel='PWM_norm', ylabel='Thrust [N]',
    #               title='Raw Thruster Data', voltages=[12, 14, 16])
    # plt.show()
    # return
    # # Plot the piecewise polynomial function for 14V from -1 to 1
    # plot_piecewise_fit_functions(x_plot, piecewise_poly_funcs, voltage=14)
    # plot_piecewise_fit_functions(x_plot, piecewise_poly_funcs, voltage=16)
    
    # --- Combined Piecewise Function for several Voltages ---
    # Fit bivariate piecewise quadratic functions
    # neg_params, pos_params = fit_piecewise_exp(data, zero_neg=-0.075, zero_pos=0.075, selected_voltages=[12,14,16])
    # plot_piecewise_exp_model(x_plot, neg_params, pos_params, voltages=[12, 14, 16])
    # plot_raw_data(data, x_key='pwm_norm', y_key='force',
    #               xlabel='PWM_norm', ylabel='Thrust [N]',
    #               title='Raw Thruster Data', voltages=[12, 14, 16])
    # plt.show()


    coeffs_neg, coeffs_pos = fit_bivariate_piecewise(data, zero_neg=-0.075, zero_pos=0.075, order=3, selected_voltages=[12, 14, 16])
    plot_bivariate_piecewise_fit(x_plot, coeffs_neg, coeffs_pos, order=3, voltages=[12, 14, 16])
    plot_raw_data(data, x_key='pwm_norm', y_key='force',
                  xlabel='PWM_norm', ylabel='Thrust [N]',
                  title='Raw Thruster Data', voltages=[12, 14, 16])

    plt.show()

    neg_params, pos_params = fit_bivariate_piecewise_softplus(
    data,
    zero_neg=-0.075,
    zero_pos=0.075,
    selected_voltages=[12, 14, 16],
    fit_x0=True,
    beta_bounds=(1e-3, 1e3),
    print_summary=True)

    plot_piecewise_softplus(x_plot, neg_params, pos_params, voltages=[12, 14, 16])
    plot_raw_data(data, x_key='pwm_norm', y_key='force',
                  xlabel='PWM_norm', ylabel='Thrust [N]',
                  title='Raw Thruster Data', voltages=[12, 14, 16])
    plt.show()
    return
    voltages = list(data.keys())

    # Fit and plot polynomials
    print("Polynomial coefficients for each voltage without zero intercept:")
    poly_coeffs = fit_polynomials(data, order=2, zero_intercept=False)
    print("K - Polynomial coefficients for each voltage with zero intercept:")
    poly_coeffs = fit_polynomials(data, order=2, zero_intercept=True)
    # plot_polynomials(data, poly_coeffs, x_key='pwm_norm', y_key='force',
    #                  xlabel='PWM_norm', ylabel='Thrust [N]',
    #                  title='Thruster Force vs PWM_norm for Different Voltages')
    compute_rmse(data, poly_coeffs, x_key='pwm_norm', y_key='force')

    plot_raw_data(data, x_key='pwm_norm', y_key='force',
                  xlabel='PWM_norm', ylabel='Thrust [N]',
                  title='Raw Thruster Data')
    
    # plot_simple_physical_model(data) # Plot simple physical model for comparison

    plt.show()

if __name__ == "__main__":
    main()