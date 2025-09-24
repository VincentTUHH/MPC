import numpy as np
from matplotlib import pyplot as plt
from scipy.differentiate import derivative
from scipy.interpolate import UnivariateSpline
from common.utils_math import softclip
from common import utils_sym
# from scipy.misc import derivative

def softplus(x):
    """numerically stable calcuation for log(1 + exp(x))"""
    return np.log(1 + np.exp(x))

def softminus(x):
    return -softplus(-x)

def softclip_local(x, a=None, b=None, beta=None):
    """
    Clipping with softplus and softminus, with paramterized corner sharpness.
    Set either (or both) endpoint to None to indicate no clipping at that end.
    """
    # when clipping at both ends, make c dimensionless w.r.t. b - a / 2  
    if a is not None and b is not None:
        beta /= (b - a) / 2

    v = x
    if a is not None:
        v = v - softminus(beta*(x - a)) / beta
    if b is not None:
        v = v - softplus(beta*(x - b)) / beta
    return v

def smoothmin(a, b, beta):
    """Smooth minimum between a and b, with parameterized sharpness."""
    return -1/beta * np.log(np.exp(-beta*a) + np.exp(-beta*b))

def main():
    x_vals = np.linspace(0, 10, 100)
    a_vals = np.sin(x_vals)
    b_vals = np.exp(-x_vals)
    smoothmin_vals_5 = smoothmin(a_vals, b_vals, beta=5.0)
    smoothmin_vals_10 = smoothmin(a_vals, b_vals, beta=10.0)
    smoothmin_vals_20 = smoothmin(a_vals, b_vals, beta=20.0)
    true_min_vals = np.minimum(a_vals, b_vals)

    func = utils_sym.smoothmin
    results = []
    for a,b in zip(a_vals,b_vals):
        results.append(func(a,b,10.0))


    plt.figure()
    plt.plot(x_vals, a_vals, label='sin(x)')
    plt.plot(x_vals, b_vals, label='exp(-x)')
    plt.plot(x_vals, smoothmin_vals_5, label='smoothmin(sin(x), exp(-x), beta=5)')
    plt.plot(x_vals, smoothmin_vals_10, label='smoothmin(sin(x), exp(-x), beta=10)')
    plt.plot(x_vals, smoothmin_vals_20, label='smoothmin(sin(x), exp(-x), beta=20)')
    plt.plot(x_vals, true_min_vals, label='true min(sin(x), exp(-x))', linestyle='--')
    plt.plot(x_vals, results, 'k.', label='CasADi smoothmin (beta=10)')
    plt.legend()
    plt.grid()
    plt.title('sin(x), exp(-x), and smoothmin')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


    x = np.linspace(-2, 2, 1000)

    beta1 = 10.0
    beta2 = 20.0

    spl1 = UnivariateSpline(x, softclip_local(x, 0, 1, beta=beta1), k=4, s=0)
    spl2 = UnivariateSpline(x, softclip_local(x, 0, 1, beta=beta2), k=4, s=0)

    spl3 = UnivariateSpline(x, softclip(x, 0, 1, beta=beta1), k=4, s=0)
    spl4 = UnivariateSpline(x, softclip(x, 0, 1, beta=beta2), k=4, s=0)

    # plt.figure()
    # plt.plot(x, spl1(x), label='softclip_local (beta=5)')
    # plt.plot(x, spl2(x), label='softclip_local (beta=20)')
    # plt.plot(x, np.clip(x, 0, 1), label='hardclip')
    # plt.grid()
    # plt.legend()
    # plt.title('Clipping functions')
    # plt.xlabel('x')
    # plt.ylabel('clipped x')
    # plt.show()

    # plt.figure()
    # plt.plot(x, softplus(x), label='softplus')
    # plt.plot(x, softminus(x), label='softminus')
    # plt.grid()
    # plt.legend()
    # plt.title('Softplus and Softminus Functions')
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.show()

    spl1_deriv = spl1.derivative()
    spl2_deriv = spl2.derivative()

    spl3_deriv = spl3.derivative()
    spl4_deriv = spl4.derivative()

    hardclip_derivative = np.zeros_like(x)
    hardclip_derivative[(x >= 0) & (x <= 1)] = 1

    # plt.figure()
    # plt.plot(x, spl1_deriv(x), label='Spline 1st derivative (beta=5)')
    # plt.plot(x, spl2_deriv(x), label='Spline 1st derivative (beta=20)')
    # plt.plot(x, hardclip_derivative, label='hardclip 1st derivative')
    # plt.grid()
    # plt.legend()
    # plt.title('First Derivative: Softclip vs Hardclip')
    # plt.xlabel('x')
    # plt.ylabel('Derivative')
    # plt.show()

    spl1_second_deriv = spl1_deriv.derivative()
    spl2_second_deriv = spl2_deriv.derivative()

    spl3_second_deriv = spl3_deriv.derivative()
    spl4_second_deriv = spl4_deriv.derivative()
    

    hardclip_second_derivative = np.zeros_like(x)

    # plt.figure()
    # plt.plot(x, spl1_second_deriv(x), label='Spline 2nd derivative (beta=5)')
    # plt.plot(x, spl2_second_deriv(x), label='Spline 2nd derivative (beta=20)')
    # plt.plot(x, hardclip_second_derivative, label='hardclip 2nd derivative')
    # plt.grid()
    # plt.legend()
    # plt.title('Second Derivative of Spline Fits to Softclip')
    # plt.xlabel('x')
    # plt.ylabel('Spline 2nd Derivative')
    # plt.show()


    fig, axs = plt.subplots(3, 1, figsize=(6, 8), sharex=True)

    # Function plot
    axs[0].plot(x, spl3(x), label='softclip_local (beta=5)')
    axs[0].plot(x, spl1(x), label='softclip_local (beta=20)')
    axs[0].plot(x, np.clip(x, 0, 1), label='hardclip')
    axs[0].set_title('Clipping functions')
    axs[0].set_ylabel('clipped x')
    axs[0].legend()
    axs[0].grid()

    # First derivative plot
    axs[1].plot(x, spl3_deriv(x), label='Spline 1st derivative (beta=5)')
    axs[1].plot(x, spl1_deriv(x), label='Spline 1st derivative (beta=20)')
    axs[1].plot(x, hardclip_derivative, label='hardclip 1st derivative')
    axs[1].set_title('First Derivative: Softclip vs Hardclip')
    axs[1].set_ylabel('Derivative')
    axs[1].legend()
    axs[1].grid()

    # Second derivative plot
    axs[2].plot(x, spl3_second_deriv(x), label='Spline 2nd derivative (beta=5)')
    axs[2].plot(x, spl1_second_deriv(x), label='Spline 2nd derivative (beta=20)')
    axs[2].plot(x, hardclip_second_derivative, label='hardclip 2nd derivative')
    axs[2].set_title('Second Derivative of Spline Fits to Softclip')
    axs[2].set_xlabel('x')
    axs[2].set_ylabel('Spline 2nd Derivative')
    axs[2].legend()
    axs[2].grid()

    plt.tight_layout()
    plt.show()
    return



if __name__ == "__main__":
    main()