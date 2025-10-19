import numpy as np

import casadi as ca

import matplotlib.pyplot as plt

from common import utils_math
from common import utils_sym

def quaternion_rotation(q, v):
    """
    Rotate vector v by unit quaternion q.
    v_new = q_conj * v * q = R(q) * v
    where v is treated as a pure quaternion [0, v_x, v_y, v_z]

    Args:
        q: array-like, shape (4,) quaternion [w, x, y, z]
        v: array-like, shape (3,) vector

    Returns:
        Rotated vector, shape (3,)
    """
    q = np.asarray(q, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    w = q[0]

    u = q[1:]  # vector part of quaternion

    t = 2 * utils_math.skew(u) @ v

    v_rot = v + w*t + utils_math.skew(u) @ t

    return v_rot

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

    a = 0.095
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

    return ca.Function('pwm_to_thrust', [pwm_norm, voltage], [thrust])

def n_order_system():
    wn = 21.4857
    zeta = 0.6901
    K = 1.0

    V_bat = 16.0

    THRUST = thrust_from_pwm()

    import scipy.signal

    # Define system: second order transfer function
    num = [K * wn**2]
    den = [1, 2*zeta*wn, wn**2]
    system = scipy.signal.TransferFunction(num, den)

    # Time vector with dt=0.05
    dt = 0.05
    t = np.arange(0, 10 + dt, dt)
    # Input: step at t=1s
    u = np.zeros_like(t)
    u[t >= 1] = 1.0

    # Simulate response
    tout, y, _ = scipy.signal.lsim(system, U=u, T=t)

    # first order system
    tau = 0.0682
    delay = 0.1 #the BasicESC has an internal low pass filter that slightly delays the response to PWM signals

    # First order system with delay
    num1 = [K]
    den1 = [tau, 1]
    system1 = scipy.signal.TransferFunction(num1, den1)

    # Simulate first order response (without delay)
    tout1, y1, _ = scipy.signal.lsim(system1, U=u, T=t)

    # Apply delay by shifting the response
    y1_delayed = np.zeros_like(y1)
    delay_steps = int(np.round(delay / (t[1] - t[0])))
    if delay_steps < len(y1):
        y1_delayed[delay_steps:] = y1[:-delay_steps]

    a = np.exp(-dt/tau)

    # Discrete first order system simulation
    f_k = 0.0
    f_discrete = []
    u_discrete = []
    for k in range(len(t)):
        u_k = u[k]
        f_discrete.append(f_k)
        u_discrete.append(u_k)
        f_k = a*f_k + K*(1-a)*u_k
    f_discrete = np.array(f_discrete)

    # Smooth transition of u from -1 to 1
    u_smooth = np.linspace(-1, 1, len(t))
    f_discrete_smooth = []
    f_k = 0.0
    for k in range(len(t)):
        u_k = u_smooth[k]
        thrust_k = THRUST(u_k, V_bat).full().item()
        f_discrete_smooth.append(f_k)
        f_k = a*f_k + (1-a)*thrust_k
    f_discrete_smooth = np.array(f_discrete_smooth)

    plt.figure()
    # Mark times where u_smooth is between -0.095 and 0.095
    indices = np.where((u_smooth >= -0.075) & (u_smooth <= 0.075))[0]
    for idx in indices:
        plt.axvline(t[idx], color='r', linestyle='-', alpha=0.3)

    
    plt.plot(t, u_smooth, label='PWM (u) Smooth')
    plt.plot(t, f_discrete_smooth, label='Thrust (f) Smooth')
    plt.xlabel('Time [s]')
    plt.ylabel('Value')
    plt.title('Discrete System: Smooth PWM Transition')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Fast transition of u from -1 to 1 (step at t=1s)
    u_fast = np.ones_like(t)
    u_fast[t < 1] = -1.0
    f_discrete_fast = []
    f_k = 0.0
    for k in range(len(t)):
        u_k = u_fast[k]
        thrust_k = THRUST(u_k, V_bat).full().item()
        f_discrete_fast.append(f_k)
        f_k = a*f_k + (1-a)*thrust_k
    f_discrete_fast = np.array(f_discrete_fast)

    plt.figure()
    plt.plot(t, u_fast, label='PWM (u) Fast Step')
    plt.plot(t, f_discrete_fast, label='Thrust (f) Fast Step')
    plt.xlabel('Time [s]')
    plt.ylabel('Value')
    plt.title('Discrete System: Fast PWM Step')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Example: u varies sinusoidally around zero
    u_sin = 0.1 * np.sin(2 * np.pi * t / 2)  # amplitude 0.1, period 2s
    f_discrete_sin = []
    f_k = 0.0
    for k in range(len(t)):
        u_k = u_sin[k]
        thrust_k = THRUST(u_k, V_bat).full().item()
        f_discrete_sin.append(f_k)
        f_k = a*f_k + (1-a)*thrust_k
    f_discrete_sin = np.array(f_discrete_sin)

    plt.figure()
    plt.plot(t, u_sin, label='PWM (u) Sinusoidal')
    plt.plot(t, f_discrete_sin, label='Thrust (f) Sinusoidal')
    plt.xlabel('Time [s]')
    plt.ylabel('Value')
    plt.title('Discrete System: Sinusoidal PWM Around Zero')
    plt.legend()
    plt.grid(True)
    plt.show()
    

    plt.figure()
    plt.plot(tout, y, label='System Response')
    plt.plot(t, u, 'k--', label='Input (Step at 1s)')
    plt.plot(tout1, y1_delayed, label='First Order w/ Delay')
    plt.plot(tout1, y1, label='First Order w/o Delay')
    plt.plot(t, f_discrete, label='Discrete First Order')
    plt.xlabel('Time [s]')
    plt.ylabel('Output')
    plt.title('Second Order System Step Response')
    plt.legend()
    plt.grid(True)
    plt.show()

def gaussian_cost():
    u = np.linspace(-1, 1, 1000)
    u_deadzone = 0.075
    m = 0.2 # margin away from deadzone, the area in which the push should start
    sigma = (u_deadzone + m)/2
    cost = np.exp(-u**2 / (2 * sigma**2))

    plt.figure()
    plt.plot(u, cost, label='Gaussian Cost')
    plt.axvline(u_deadzone, color='r', linestyle='--', label='+u_deadzone')
    plt.axvline(-u_deadzone, color='r', linestyle='--', label='-u_deadzone')
    plt.axvline(u_deadzone + m, color='g', linestyle=':', label='+u_deadzone+m')
    plt.axvline(-u_deadzone - m, color='g', linestyle=':', label='-u_deadzone-m')
    plt.xlabel('u')
    plt.ylabel('Cost')
    plt.title('Gaussian Cost with Deadzone and Margin')
    plt.legend()
    plt.grid(True)
    plt.show()
    return cost

def main():
    gaussian_cost()
    return
    n_order_system()
    return
    q = [0.7071, 0.7071, 0, 0]  # 90 degrees around x-axis
    v = [0, 1, 0]  # y-axis
    v_rotated_quat = quaternion_rotation(q, v)
    rotation_matrix = utils_math.rotation_matrix_from_quat(q)
    v_rotated_rot = rotation_matrix @ v
    np.set_printoptions(precision=2, suppress=True)
    print("Original vector:", np.array(v))
    print("Rotated vector (quaternion):", v_rotated_quat)
    print("Rotated vector (rotation matrix):", v_rotated_rot)

    

    errors = []
    np.random.seed(42)
    for _ in range(10):
        # Random unit quaternion
        q = np.random.randn(4)
        q /= np.linalg.norm(q)
        # Random vector
        v = np.random.randn(3)
        # Rotate using quaternion
        v_rot_quat = quaternion_rotation(q, v)
        # Rotate using rotation matrix
        rot_mat = utils_math.rotation_matrix_from_quat(q)
        v_rot_mat = rot_mat @ v
        # Compute error norm
        error = np.linalg.norm(v_rot_quat - v_rot_mat)
        errors.append(error)

    plt.figure()
    plt.plot(errors, marker='o')
    plt.xlabel('Example')
    plt.ylabel('Rotation Error Norm')
    plt.title('Quaternion vs Rotation Matrix Rotation Error')
    plt.grid(True)
    plt.show()
    return 

if __name__ == "__main__":
    main()