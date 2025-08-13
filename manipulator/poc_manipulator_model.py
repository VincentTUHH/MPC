'''
Proof of Concept manipulator model file.
The recursive newton-euler method (RNEM) is checked against 
the c++ implementation of Niklas Trekel (2023)
'''

import numpy as np
import casadi as ca
from manipulator.dynamics import Dynamics
from manipulator.dynamics_symbolic import DynamicsSymbolic
from common.my_package_path import get_package_path
import matplotlib.pyplot as plt
from common.animate import (
    plot_wrench_vs_time_compare
)
import common.utils_math as utils_math
import manipulator.kinematics as manip_kinematics
import manipulator.kinematics_symbolic as manip_kinematics_symbolic

# -------------------- Trajectory Generation --------------------

def fourier_traj(a, b, q0, t):
    nF = len(a)
    omega_f = 2 * np.pi * 0.1
    q = np.zeros_like(t)
    dq = np.zeros_like(t)
    ddq = np.zeros_like(t)
    for j in range(1, nF+1):
        q += (a[j-1] / (omega_f*j)) * np.sin(omega_f*j*t) - (b[j-1] / (omega_f*j)) * np.cos(omega_f*j*t)
        dq += a[j-1]*np.cos(omega_f*j*t) + b[j-1]*np.sin(omega_f*j*t)
        ddq += -a[j-1]*omega_f*j*np.sin(omega_f*j*t) + b[j-1]*omega_f*j*np.cos(omega_f*j*t)
    q += q0
    return q, dq, ddq

def excitation_trajectory_with_fourier(T=10.0, fps=50):
    timesteps = int(T * fps)
    t = np.linspace(0, T, timesteps)
    coeffs = {
        0: {'a': [0.3, -0.1, 0.05, -0.02, 0.015, -0.01, 0.005, -0.003, 0.002, -0.001],
            'b': [-0.2, 0.15, -0.05, 0.02, -0.015, 0.01, -0.005, 0.003, -0.002, 0.001],
            'q0': 2.5},
        1: {'a': [-0.25, 0.12, -0.06, 0.03, -0.02, 0.015, -0.007, 0.004, -0.002, 0.001],
            'b': [0.25, -0.08, 0.04, -0.02, 0.015, -0.01, 0.005, -0.003, 0.002, -0.001],
            'q0': 2.0},
        2: {'a': [0.15, -0.05, 0.02, -0.01, 0.008, -0.005, 0.003, -0.002, 0.0015, -0.001],
            'b': [0.1, -0.05, 0.01, -0.005, 0.004, -0.003, 0.002, -0.0015, 0.001, -0.0005],
            'q0': 1.0},
        3: {'a': [0.6, -0.2, 0.1, -0.05, 0.04, -0.03, 0.02, -0.01, 0.005, -0.002],
            'b': [0.4, -0.1, 0.05, -0.02, 0.015, -0.01, 0.007, -0.004, 0.002, -0.001],
            'q0': 3.0}
    }
    q_traj = np.zeros((4, len(t)))
    dq_traj = np.zeros((4, len(t)))
    ddq_traj = np.zeros((4, len(t)))
    for i in range(4):
        a = coeffs[i]['a']
        b = coeffs[i]['b']
        q0 = coeffs[i]['q0']
        q_traj[i], dq_traj[i], ddq_traj[i] = fourier_traj(a, b, q0, t)
    return q_traj, dq_traj, ddq_traj, t

def fourier_traj_vehicle(a, b, p0, t, omega_f):
    nF = len(a)
    traj = np.zeros_like(t)
    vel = np.zeros_like(t)
    acc = np.zeros_like(t)
    for j in range(1, nF+1):
        traj += (a[j-1] / (omega_f*j)) * np.sin(omega_f*j*t) - (b[j-1] / (omega_f*j)) * np.cos(omega_f*j*t)
        vel += a[j-1]*np.cos(omega_f*j*t) + b[j-1]*np.sin(omega_f*j*t)
        acc += -a[j-1]*omega_f*j*np.sin(omega_f*j*t) + b[j-1]*omega_f*j*np.cos(omega_f*j*t)
    traj += p0
    return traj, vel, acc

def excitation_trajectory_vehicle_body(type, T=10.0, fps=50):
    timesteps = int(T * fps)
    t = np.linspace(0, T, timesteps)
    # Fourier coefficients for position (gentle movement)
    pos_coeffs = {
        'a': [0.2, -0.05, 0.02, -0.01, 0.005],
        'b': [0.1, -0.03, 0.01, -0.005, 0.002],
        'p0': [0.0, 0.0, 0.0]
    }
    # Fourier coefficients for orientation (gentle rotation)
    ori_coeffs = {
        'a': [0.05, -0.01, 0.005, -0.002, 0.001],
        'b': [0.03, -0.008, 0.003, -0.001, 0.0005],
        'o0': [1.0, 0.0, 0.0, 0.0]  # initial quaternion (no rotation)
    }
    omega_f = 2 * np.pi * 0.05  # lower frequency for gentle movement

    # Position
    pos_traj = np.zeros((3, timesteps))
    vel_traj = np.zeros((3, timesteps))
    acc_traj = np.zeros((3, timesteps))
    for i in range(3):
        pos_traj[i], vel_traj[i], acc_traj[i] = fourier_traj_vehicle(
            pos_coeffs['a'], pos_coeffs['b'], pos_coeffs['p0'][i], t, omega_f
        )

    for i in range(t.shape[0]):
        if type in ('zero_ref', 'zero_ref_and_wrench'):
            pos_traj[:, i] = np.zeros(3)
            vel_traj[:, i] = np.zeros(3)
            acc_traj[:, i] = np.zeros(3)
        else:
            pos_traj[:, i] = np.array([0.1, 0.2, 0.3])
            vel_traj[:, i] = np.array([0.05, 0.1, 0.15])
            acc_traj[:, i] = np.array([0.01, 0.02, 0.03])

    # Orientation (quaternion), generate gentle rotation about z axis
    angle_traj, ang_vel_traj, ang_acc_traj = fourier_traj_vehicle(
        ori_coeffs['a'], ori_coeffs['b'], 0.0, t, omega_f
    )

    for i in range(t.shape[0]):
        if type in ('zero_ref', 'zero_ref_and_wrench'):
            angle_traj[i] = 0.0
            ang_vel_traj[i] = 0.0
            ang_acc_traj[i] = 0.0
        else:
            angle_traj[i] = 0.1
            ang_vel_traj[i] = 0.05
            ang_acc_traj[i] = 0.01

    # Convert angle_traj to quaternion (rotation about z axis)
    quat_traj = np.zeros((4, timesteps))
    for i in range(timesteps):
        theta = angle_traj[i]
        quat_traj[:, i] = np.array([
            np.cos(theta/2),
            0.0,
            0.0,
            np.sin(theta/2)
        ])

    # Angular velocity and acceleration in body frame (only z axis)
    ang_vel_vec_traj = np.zeros((3, timesteps))
    ang_acc_vec_traj = np.zeros((3, timesteps))
    ang_vel_vec_traj[2] = ang_vel_traj
    ang_acc_vec_traj[2] = ang_acc_traj

    return pos_traj, quat_traj, vel_traj, ang_vel_vec_traj, acc_traj, ang_acc_vec_traj, t

def eef_wrench_contact(type, T=10.0, fps=50):
    timesteps = int(T * fps)
    t = np.linspace(0, T, timesteps)
    # Sinusoidal wrench: amplitude and offset for force and torque
    f_amp = np.array([0.5, 0.3, 0.2])
    f_offset = np.array([0.1, 0.8, 0.6])
    l_amp = np.array([0.1, 0.05, 0.03])
    l_offset = np.array([0.2, 0.3, 0.1])

    f_eef = np.zeros((3, timesteps))
    l_eef = np.zeros((3, timesteps))
    for i in range(3):
        f_eef[i, :] = f_offset[i] + f_amp[i] * np.sin(2 * np.pi * (i+1) * t / T)
        l_eef[i, :] = l_offset[i] + l_amp[i] * np.sin(2 * np.pi * (i+1) * t / T)

    for i in range(t.shape[0]):
        if type in ('zero_wrench', 'zero_ref_and_wrench'):
            f_eef[:, i] = np.zeros(3)
            l_eef[:, i] = np.zeros(3)
        else:
            f_eef[:, i] = f_amp
            l_eef[:, i] = l_amp

    return f_eef, l_eef, t

def export_trajectory_txt(filename, t, q_traj, dq_traj, ddq_traj):
    with open(filename, 'w') as f:
        for i in range(len(t)):
            row = [f"{t[i]:.3f}"]
            row += [f"{q_traj[j, i]:.6f}" for j in range(4)]
            row += [f"{dq_traj[j, i]:.6f}" for j in range(4)]
            row += [f"{ddq_traj[j, i]:.6f}" for j in range(4)]
            f.write(', '.join(row) + '\n')

def export_references_txt(filename, t, quat_traj, vel_traj, acc_traj, ang_vel_vec_traj, ang_acc_vec_traj, f_eef_traj, l_eef_traj):
    with open(filename, 'w') as f:
        for i in range(len(t)):
            row = [f"{t[i]:.3f}"]
            row += [f"{quat_traj[j, i]:.6f}" for j in range(4)]
            row += [f"{vel_traj[j, i]:.6f}" for j in range(3)]
            row += [f"{acc_traj[j, i]:.6f}" for j in range(3)]
            row += [f"{ang_vel_vec_traj[j, i]:.6f}" for j in range(3)]
            row += [f"{ang_acc_vec_traj[j, i]:.6f}" for j in range(3)]
            row += [f"{f_eef_traj[j, i]:.6f}" for j in range(3)]
            row += [f"{l_eef_traj[j, i]:.6f}" for j in range(3)]
            f.write(', '.join(row) + '\n')

# -------------------- Test & Main --------------------

def dynamic_test(type):
    manipulator_package_path = get_package_path('manipulator')
    kin_params_path = manipulator_package_path + "/config/alpha_kin_params.yaml"
    base_tf_bluerov_path = manipulator_package_path + "/config/alpha_base_tf_params_bluerov.yaml"
    inertial_params_dh_path = manipulator_package_path + "/config/alpha_inertial_params_dh.yaml"

    DH_table = utils_math.load_dh_params(kin_params_path)
    file_paths = [
        kin_params_path,
        base_tf_bluerov_path,
        inertial_params_dh_path
    ]
    alpha_params = utils_math.load_dynamic_params(file_paths)
    dyn = Dynamics(manip_kinematics.Kinematics(DH_table), alpha_params)
    T = 10.0
    q_traj, dq_traj, ddq_traj, t = excitation_trajectory_with_fourier(T=T)
    t_cpp, tau_cpp = utils_math.read_wrench_txt(f'{manipulator_package_path}/data/model_output/01_08_cpp_wrench_on_vehicle_{type}.txt')

    pos_traj, quat_traj, vel_traj, ang_vel_vec_traj, acc_traj, ang_acc_vec_traj, t = excitation_trajectory_vehicle_body(type)
    f_eef_traj, l_eef_traj, t = eef_wrench_contact(type)

    f_eef = np.zeros(3)
    l_eef = np.zeros(3)
    v_ref = np.zeros(3)
    a_ref = np.zeros(3)
    w_ref = np.zeros(3)
    dw_ref = np.zeros(3)

    tau = []

    for i in range(q_traj.shape[1]):
        q = q_traj[:, i]
        dq = dq_traj[:, i]
        ddq = ddq_traj[:, i]
        quaternion_ref = quat_traj[:, i]
        v_ref = vel_traj[:, i]
        a_ref = acc_traj[:, i]
        w_ref = ang_vel_vec_traj[:, i]
        dw_ref = ang_acc_vec_traj[:, i]
        f_eef = f_eef_traj[:, i]
        l_eef = l_eef_traj[:, i]
        dyn.kinematics_.update(q)
        out = dyn.rnem(q, dq, ddq, v_ref, a_ref, w_ref, dw_ref, quaternion_ref, f_eef, l_eef)
        tau.append(np.array(out).flatten())
    tau = np.array(tau)

    plot_wrench_vs_time_compare(t, tau, tau_cpp, title=f"Wrench Comparison: Python vs C++ real, {type}")
    print("Comparing real and cpp:")
    if tau.shape[1] == tau_cpp.shape[1]:
        error = np.abs(tau - tau_cpp[:tau.shape[0], :])
        for i in range(6):
            total_error = np.mean(np.linalg.norm(error[:, i]))
            print(f"Total absolute error for variable {i+1}: {total_error:.6f}")
    else:
        print("tau and tau_cpp have different shapes, cannot compute error.")

    return tau

def dynamic_symbolic_test(type):
    manipulator_package_path = get_package_path('manipulator')
    kin_params_path = manipulator_package_path + "/config/alpha_kin_params.yaml"
    base_tf_bluerov_path = manipulator_package_path + "/config/alpha_base_tf_params_bluerov.yaml"
    inertial_params_dh_path = manipulator_package_path + "/config/alpha_inertial_params_dh.yaml"

    DH_table = utils_math.load_dh_params(kin_params_path)
    file_paths = [
        kin_params_path,
        base_tf_bluerov_path,
        inertial_params_dh_path
    ]
    alpha_params = utils_math.load_dynamic_params(file_paths)
    dyn = DynamicsSymbolic(manip_kinematics_symbolic.KinematicsSymbolic(DH_table), alpha_params)
    T = 10.0
    q_traj, dq_traj, ddq_traj, t = excitation_trajectory_with_fourier(T=T)
    # export_trajectory_txt(f'{manipulator_package_path}/data/states/01_08_joint_excitation_trajectory.txt', t, q_traj, dq_traj, ddq_traj)
    t_cpp, tau_cpp = utils_math.read_wrench_txt(f'{manipulator_package_path}/data/model_output/01_08_cpp_wrench_on_vehicle_{type}.txt')

    def rnem_function_symbolic():
        n_joints = dyn.kinematics_.n_joints
        q = ca.MX.sym('q', n_joints)
        dq = ca.MX.sym('dq', n_joints)
        ddq = ca.MX.sym('ddq', n_joints)
        v_ref = ca.MX.sym('v_ref', 3)
        a_ref = ca.MX.sym('a_ref', 3)
        w_ref = ca.MX.sym('w_ref', 3)
        dw_ref = ca.MX.sym('dw_ref', 3)
        quaternion_ref = ca.MX.sym('quat_ref', 4)
        f_eef = ca.MX.sym('f_eef', 3)
        l_eef = ca.MX.sym('l_eef', 3)

        dyn.kinematics_.update(q)
        tau = dyn.rnem_symbolic(q, dq, ddq, v_ref, a_ref, w_ref, dw_ref, quaternion_ref, f_eef, l_eef)
        
        rnem_func = ca.Function(
            'rnem_func',
            [q, dq, ddq, v_ref, a_ref, w_ref, dw_ref, quaternion_ref, f_eef, l_eef],
            [tau]
        )
        
        return rnem_func
    
    rneM_func = rnem_function_symbolic()

    f_eef = np.zeros(3)
    l_eef = np.zeros(3)
    v_ref = np.zeros(3)
    a_ref = np.zeros(3)
    w_ref = np.zeros(3)
    dw_ref = np.zeros(3)

    pos_traj, quat_traj, vel_traj, ang_vel_vec_traj, acc_traj, ang_acc_vec_traj, t = excitation_trajectory_vehicle_body(type)
    f_eef_traj, l_eef_traj, t = eef_wrench_contact(type)
    # export_references_txt(f'{manipulator_package_path}/data/states/01_08_references_{type}.txt', t, quat_traj, vel_traj, acc_traj, ang_vel_vec_traj, ang_acc_vec_traj, f_eef_traj, l_eef_traj)


    tau = []

    for i in range(q_traj.shape[1]):
        q = q_traj[:, i]
        dq = dq_traj[:, i]
        ddq = ddq_traj[:, i]
        quaternion_ref = quat_traj[:, i]
        v_ref = vel_traj[:, i]
        a_ref = acc_traj[:, i]
        w_ref = ang_vel_vec_traj[:, i]
        dw_ref = ang_acc_vec_traj[:, i]
        f_eef = f_eef_traj[:, i]
        l_eef = l_eef_traj[:, i]
        out = rneM_func(q, dq, ddq, v_ref, a_ref, w_ref, dw_ref, quaternion_ref, f_eef, l_eef)
        tau.append(np.array(out).flatten())
    tau = np.array(tau)

    plot_wrench_vs_time_compare(t, tau, tau_cpp, title=f"Wrench Comparison: Python vs C++ symbolic, {type}")
    print("Comparing symbolic and cpp:")
    if tau.shape[1] == tau_cpp.shape[1]:
        error = np.abs(tau - tau_cpp[:tau.shape[0], :])
        for i in range(6):
            total_error = np.mean(np.linalg.norm(error[:, i]))
            print(f"Total absolute error for variable {i+1}: {total_error:.6f}")
    else:
        print("tau and tau_cpp have different shapes, cannot compute error.")

    tau_real = dynamic_test(type)
    plot_wrench_vs_time_compare(t, tau, tau_real, title=f"Wrench Comparison: CasADi vs Numpy, {type}")
    print("Comparing CasADi vs Numpy:")
    if tau.shape[1] == tau_real.shape[1]:
        error = np.abs(tau - tau_real[:tau.shape[0], :])
        for i in range(6):
            total_error = np.mean(np.linalg.norm(error[:, i]))
            print(f"Total absolute error for variable {i+1}: {total_error:.6f}")
    else:
        print("tau and tau_cpp have different shapes, cannot compute error.")

    plt.show()

def main():
    # Ask user for trajectory type
    print("Select trajectory type:")
    print("1: const")
    print("2: zero_wrench")
    print("3: zero_ref")
    print("4: zero_ref_and_wrench")
    choice = input("Enter number (1-4): ").strip()

    if choice == "1":
        traj_type = "const"
    elif choice == "2":
        traj_type = "zero_wrench"
    elif choice == "3":
        traj_type = "zero_ref"
    elif choice == "4":
        traj_type = "zero_ref_and_wrench"
    else:
        print("Invalid choice, defaulting to 'const'")
        traj_type = "const"

    # Pass traj_type to test functions
    dynamic_test(traj_type)
    dynamic_symbolic_test(traj_type)

if __name__ == "__main__":
    main()