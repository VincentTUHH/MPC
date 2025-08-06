import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from common.my_package_path import get_package_path
import common.utils_math as utils_math
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

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

def plot_raw_data(data, x_key='pwm_norm', y_key='force', xlabel='PWM [mus]', ylabel='Force [N]', title=None):
    # plt.figure(figsize=(8, 6))
    for v in data:
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

# --- Main Routine ---

def main():
    # Load data
    bluerov_package_path = get_package_path('bluerov')
    thruster_params_path = bluerov_package_path + "/thruster_data/"
    data = utils_math.load_all_thruster_data(thruster_params_path)

    # Normalize PWM
    for v in data:
        data[v]['pwm_norm'] = normalize_pwm(data[v]['pwm'])

    voltages = list(data.keys())

    # Fit and plot polynomials
    print("Polynomial coefficients for each voltage without zero intercept:")
    poly_coeffs = fit_polynomials(data, order=1, zero_intercept=False)
    print("K - Polynomial coefficients for each voltage with zero intercept:")
    poly_coeffs = fit_polynomials(data, order=1, zero_intercept=True)
    plot_polynomials(data, poly_coeffs, x_key='pwm_norm', y_key='force',
                     xlabel='PWM_norm', ylabel='Thrust [N]',
                     title='Thruster Force vs PWM_norm for Different Voltages')
    compute_rmse(data, poly_coeffs, x_key='pwm_norm', y_key='force')

    # plot_raw_data(data, x_key='pwm_norm', y_key='force',
    #               xlabel='PWM_norm', ylabel='Thrust [N]',
    #               title='Raw Thruster Data')
    
    plot_simple_physical_model(data) # Plot simple physical model for comparison

    plt.show()

if __name__ == "__main__":
    main()