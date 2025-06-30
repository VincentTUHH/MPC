import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit

def read_thruster_csv(csv_path):
    df = pd.read_csv(csv_path, sep=';', decimal=',', engine='python')
    df.dropna(axis=0, how='all', inplace=True)
    df.dropna(axis=1, how='all', inplace=True)
    data = {col.strip(): df[col].to_numpy() for col in df.columns}
    return data

def load_all_thruster_data():
    voltages = [10, 12, 14, 16, 18, 20]
    data = {}
    for v in voltages:
        fname = f'multiplied_cleaned_T200_{v}V.csv'
        d = read_thruster_csv(fname)
        data[v] = {
            'pwm': d['PWM (mus)'],
            'force': d['Force (N)']
        }
    return data

def fit_polynomials(data, order=3):
    poly_coeffs = {}
    for v, d in data.items():
        coeffs = np.polyfit(d['pwm_norm'], d['force'], order)
        print(coeffs)
        # Force fit to have zero intercept (y = mx), so fit with intercept=False
        X = d['pwm_norm'][:, np.newaxis]
        y = d['force']
        # Fit using least squares with no intercept
        m, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        coeffs = np.array([m[0], 0.0]) if order == 1 else np.concatenate([np.zeros(order-1), [m[0], 0.0]])
        
        poly_coeffs[v] = coeffs
        print(poly_coeffs[v])
    return poly_coeffs

def fit_polynomials_inverse(data, order=3):
    poly_coeffs = {}
    for v, d in data.items():
        coeffs = np.polyfit(d['force'], d['pwm_norm'], order)
        poly_coeffs[v] = coeffs
    return poly_coeffs

def fit_sinh_inverse(data):

    def sinh_func(force, a, b, c, d):
        return a * np.tanh(b * force + c) + d

    sinh_params = {}
    for v, dct in data.items():
        force = dct['force']
        pwm = dct['pwm']
        # Initial guess: a, b, c, d
        p0 = [1.0, 1.0, 0.0, 0.0]
        try:
            params, _ = curve_fit(sinh_func, force, pwm, p0=p0)
            sinh_params[v] = params
        except RuntimeError:
            sinh_params[v] = None
            print(f"Could not fit sinh function for {v}V")
    return sinh_params

##### the sinh fit is bad!!!!
##### or is there an easy way to inverse the polynomial that I fit for the forward data????

def plot_sinh_inverse(data, sinh_params):
    colors = ['b', 'g', 'r', 'c', 'm', 'y']
    plt.figure(figsize=(8, 6))
    for (v, color) in zip(data.keys(), colors):
        params = sinh_params.get(v)
        if params is not None:
            force_range = np.linspace(np.min(data[v]['force']), np.max(data[v]['force']), 200)
            pwm_fit = params[0] * np.tan(params[1] * force_range + params[2]) + params[3]
            plt.plot(force_range, pwm_fit, color=color, linestyle=':', label=f'Tan fit {v}V')
    plt.xlabel('Force [N]')
    plt.ylabel('PWM [mus]')
    plt.title('PWM vs Thruster Force vs for Different Voltages')
    plt.grid(True)

def thrust_interpolated(pwm_val, voltage_val, voltages, poly_coeffs):
    v_min, v_max = min(voltages), max(voltages)
    voltage_val = np.clip(voltage_val, v_min, v_max)
    v_arr = np.array(voltages)
    idx = np.searchsorted(v_arr, voltage_val)
    if idx == 0:
        v0, v1 = v_arr[0], v_arr[1]
    elif idx == len(v_arr):
        v0, v1 = v_arr[-2], v_arr[-1]
    else:
        v0, v1 = v_arr[idx-1], v_arr[idx]
    f0 = np.polyval(poly_coeffs[v0], pwm_val)
    f1 = np.polyval(poly_coeffs[v1], pwm_val)
    if v1 == v0:
        return f0
    return f0 + (f1 - f0) * (voltage_val - v0) / (v1 - v0)

def plot_polynomials(data, poly_coeffs):
    voltages = list(data.keys())
    colors = ['b', 'g', 'r', 'c', 'm', 'y']
    plt.figure(figsize=(8, 6))
    for v, color in zip(voltages, colors):
        pwm_range = np.linspace(np.min(data[v]['pwm_norm']), np.max(data[v]['pwm_norm']), 200)
        force_fit = np.polyval(poly_coeffs[v], pwm_range)
        plt.plot(pwm_range, force_fit, color=color, linestyle='--', label=f'Poly fit {v}V')
    plt.xlabel('PWM [mus]')
    plt.ylabel('Force [N]')
    plt.title('Thruster Force vs PWM for Different Voltages')
    plt.grid(True)

def plot_polynomials_inverse(data, poly_coeffs):
    voltages = list(data.keys())
    colors = ['b', 'g', 'r', 'c', 'm', 'y']
    plt.figure(figsize=(8, 6))
    for v, color in zip(voltages, colors):
        force_range = np.linspace(np.min(data[v]['force']), np.max(data[v]['force']), 200)
        pwm_fit = np.polyval(poly_coeffs[v], force_range)
        plt.plot(force_range, pwm_fit, color=color, linestyle='--', label=f'Poly fit {v}V')
    plt.xlabel('Force [N]')
    plt.ylabel('PWM [mus]')
    plt.title('PWM vs Thruster Force vs for Different Voltages')
    plt.grid(True)

def plot_interpolated_curve(data, voltages, poly_coeffs, interp_voltage=15):
    pwm_range = np.linspace(np.min(data[16]['pwm']), np.max(data[16]['pwm']), 200)
    force_interp = [thrust_interpolated(pwm, interp_voltage, voltages, poly_coeffs) for pwm in pwm_range]
    plt.plot(pwm_range, force_interp, color='k', linestyle='--', label=f'Interpolated {interp_voltage}V')

def plot_raw_data(data):
    for v in data:
        plt.plot(data[v]['pwm_norm'], data[v]['force'], label=f'{v}V')
    plt.legend()
    plt.tight_layout()

def plot_raw_data_inverse(data):
    for v in data:
        plt.plot(data[v]['force'], data[v]['pwm_norm'], label=f'{v}V')
    plt.legend()
    plt.tight_layout()

def compute_rmse(data, poly_coeffs):
    rmse_all = 0
    print("RMSE for each voltage polynomial fit:")
    for v in data:
        force_fit = np.polyval(poly_coeffs[v], data[v]['pwm'])
        rmse = np.sqrt(np.mean((data[v]['force'] - force_fit) ** 2))
        print(f"{v}V: RMSE = {rmse:.4f} N")
        rmse_all += rmse
    print(f"Overall RMSE across all voltages: {rmse_all:.4f} N")

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

def plot_bivariate_fit(model, poly, valid_idx, data, voltage=20):
    pwm_span = np.linspace(np.min(data[16]['pwm']), np.max(data[16]['pwm']), 200)
    voltage_const = np.full_like(pwm_span, voltage)
    X_eval = np.column_stack((pwm_span, voltage_const))
    X_eval_poly_full = poly.transform(X_eval)
    X_eval_poly = X_eval_poly_full[:, valid_idx]
    thrust_eval = model.predict(X_eval_poly)
    # Uncomment to plot if desired:
    # plt.plot(pwm_span, thrust_eval, '--', label=f'Poly fit @{voltage}V')


# Fit inverse polynomials (PWM as a function of Force)
def fit_and_plot_inverse(data, order=3):
    poly_coeffs_inv = fit_polynomials_inverse(data, order=order)
    plot_polynomials_inverse(data, poly_coeffs_inv)
    return poly_coeffs_inv

def main():
    data = load_all_thruster_data()

    # Normalize PWM values in data to [-1, 1] range
    def normalize_pwm(pwm):
        return 2 * (pwm - 1100) / (1900 - 1100) - 1
    
    # Add denormalization function for PWM
    def denormalize_pwm(pwm_norm):
        return ((pwm_norm + 1) * (1900 - 1100) / 2) + 1100

    for v in data:
        data[v]['pwm_norm'] = normalize_pwm(data[v]['pwm'])

    voltages = list(data.keys())
    poly_coeffs = fit_polynomials(data, order=1)
    print("Polynomial coefficients for each voltage:")
    for v, coeffs in poly_coeffs.items():
        print(f"{v}V: {coeffs}")
    plot_polynomials(data, poly_coeffs)
    # plot_interpolated_curve(data, voltages, poly_coeffs, interp_voltage=15)
    compute_rmse(data, poly_coeffs)
    # fit_bivariate_polynomial(data)
    # plot_raw_data(data)
    plt.xlabel('PWM [mus]')
    plt.ylabel('Force [N]')
    plt.title('Thruster Force vs PWM for Different Voltages')
    plt.grid(True)
    # plt.show()

    # Plot thrust = 2.5116 * battery_voltage * PWM_norm for 12V, 14V, 16V
    # plt.figure(figsize=(8, 6))
    for v, color in zip([12, 14, 16], ['b', 'g', 'r']):
        pwm_norm_range = np.linspace(np.min(data[v]['pwm_norm']), np.max(data[v]['pwm_norm']), 200)
        thrust = 2.5116 * v * pwm_norm_range
        plt.plot(pwm_norm_range, thrust, color=color, label=f'Thrust = 2.5116*{v}*PWM_norm')
    plt.xlabel('PWM_norm')
    plt.ylabel('Thrust [N]')
    plt.title('Thrust = 2.5116 * Voltage * PWM_norm')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()







    # # Plot the given points (voltage vs slope)
    # voltages_pts = [10, 12, 14, 16, 18, 20]
    # slopes = [22.69051278, 28.72702823, 34.99602483, 40.68488252, 44.89460618, 47.78498596]

    # voltages_pts = [14, 16, 18]
    # slopes = [34.99602483, 40.68488252, 44.89460618]

    # plt.figure(figsize=(6, 4))
    # plt.plot(voltages_pts, slopes, 'o-', label='Slope vs Voltage')
    # plt.xlabel('Voltage [V]')
    # plt.ylabel('Slope (N / PWM_norm)')
    # plt.title('Slope of Linear Fit vs Voltage')
    # plt.grid(True)
    # plt.legend()
    # plt.tight_layout()
    # # plt.show()

    # # Fit a linear polynomial (degree 1) to the (voltage, slope) data
    # coeffs = np.polyfit(voltages_pts, slopes, 1)
    # m, b = coeffs
    # print(f"Fitted linear function: slope = {m:.4f} * voltage + {b:.4f}")

    # # Plot the fitted line
    # voltages_fit = np.linspace(min(voltages_pts), max(voltages_pts), 100)
    # slopes_fit = m * voltages_fit + b
    # plt.plot(voltages_fit, slopes_fit, 'r--', label='Linear Fit')
    # plt.legend()
    # # plt.show()

    # # Fit a linear polynomial (degree 1) with zero intercept to the (voltage, slope) data
    # voltages_arr = np.array(voltages_pts)
    # slopes_arr = np.array(slopes)
    # # Least squares fit with no intercept: slopes = m * voltages
    # m, _, _, _ = np.linalg.lstsq(voltages_arr[:, np.newaxis], slopes_arr, rcond=None)
    # print(f"Fitted function: slope = {m[0]:.4f} * voltage")

    # # Plot the fitted line
    # voltages_fit = np.linspace(min(voltages_pts), max(voltages_pts), 100)
    # slopes_fit = m[0] * voltages_fit
    # plt.plot(voltages_fit, slopes_fit, 'g--', label='Fit: slope = m * voltage')
    # plt.legend()
    # plt.show()









    

    
    # # Plot the original inverse data (Force vs PWM)
    # plt.figure(figsize=(8, 6))
    # plot_raw_data_inverse(data)
    # poly_coeffs_inv = fit_and_plot_inverse(data)
    # plt.show()

    # data = load_all_thruster_data()
    # poly_coeffs_inv = fit_polynomials_inverse(data, order=7)
    # plot_polynomials_inverse(data, poly_coeffs_inv)
    # plot_raw_data_inverse(data)
    # plt.legend()
    # plt.tight_layout()
    # plt.xlabel('Force [N]')
    # plt.ylabel('PWM [mus]')
    # plt.title('Inverse: PWM vs Force for Different Voltages')
    # plt.grid(True)
    # plt.show()


    # tan_params = fit_sinh_inverse(data)
    # plot_sinh_inverse(data, tan_params)
    # plot_raw_data_inverse(data)
    # plt.legend()
    # plt.tight_layout()
    # plt.xlabel('Force [N]')
    # plt.ylabel('PWM [mus]')
    # plt.title('Inverse: PWM vs Force for Different Voltages')
    # plt.grid(True)
    # plt.show()



if __name__ == "__main__":
    main()
