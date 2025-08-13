import uvms.model as uvms_model
from common.my_package_path import get_package_path
import common.utils_math as utils_math
import numpy as np


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

    uvms_model_instance = uvms_model.UVMSModel(DH_table, alpha_params)
    print("UVMS Model initialized successfully.")

    # print("r_B_0:", uvms_model_instance.r_B_0)
    # print("R_B_0:", uvms_model_instance.R_B_0)

    uvms_model_instance.update(np.array([0.0, 0.0, 0.0, 0.0]),
                               np.array([0.0, 0.0, 0.0]),
                               utils_math.euler_to_quat(0.0, 0.0, 0.0))
    # print("End-effector position in inertial frame: \n", uvms_model_instance.p_eef)
    # print("End-effector attitude in inertial frame: \n", utils_math.rotation_matrix_from_quat(uvms_model_instance.att_eef))

    


if __name__ == "__main__":
    main()