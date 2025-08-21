import uvms.model as uvms_model
from common.my_package_path import get_package_path
import common.utils_math as utils_math
import numpy as np
import common.animate as animate


def main():
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

    bluerov_package_path = get_package_path('bluerov')
    bluerov_params_path = bluerov_package_path + "/config/model_params.yaml"
    bluerov_params = utils_math.load_model_params(bluerov_params_path)

    q0 = np.array([np.pi, np.pi * 0.5, np.pi * 0.75, np.pi * 0.5])
    pos = np.array([0.0, 0.0, 0.0])
    att = np.array([0.0, 0.0, np.pi/4])  # Euler angles
    vel = np.array([0.0, 0.0, 0.0])
    omega = np.array([0.0, 0.0, 0.0])

    uvms_model_instance = uvms_model.UVMSModel(DH_table, alpha_params, bluerov_params, q0, pos, att, vel, omega)
    print("UVMS Model initialized successfully.")

    use_pwm = True

    uv = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # Example control input
    # uv = np.array([0.0, 0.0, 0.0, 0.0, 0.2, -0.2, -0.2, 0.2])  # Example control input
    uq = np.array([0.0, 0.0, 0.0, 0.0])  # Example joint velocities
    dt = 0.05
    V_bat = 16.0  # Example battery voltage

    eta_history = []

    joint_history = []

    for _ in range(100):
        uvms_model_instance.update(dt, uq, uv, use_pwm, V_bat) 
        eta, joint_positions = uvms_model_instance.get_uvms_configuration()

        eta_history.append(eta.reshape(1, -1))  # Ensure eta is row-wise
        joint_history.append(joint_positions.copy())

    eta_history = np.vstack(eta_history)  # Stack rows for eta_history

    eta_history = np.array(eta_history)
    joint_history = np.array(joint_history)

    # print(eta_history) 
    # print(joint_history)
    animate.animate_uvms(eta_history, joint_history, dt)
    

if __name__ == "__main__":
    main()