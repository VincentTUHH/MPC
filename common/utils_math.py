import numpy as np
from scipy.spatial.transform import Rotation as Rot
import yaml
from copy import deepcopy
from types import SimpleNamespace

# Define commonly used constants
GRAVITY_VECTOR = np.array([0, 0, -9.81])

def rotation_matrix_from_quat(q):
    """
    Convert quaternion [w, x, y, z] to rotation matrix (3x3).
    """
    return Rot.from_quat(q, scalar_first=True).as_matrix()

def rotation_matrix_from_euler(phi, theta, psi):
    """
    Create a rotation matrix from Euler angles (phi, theta, psi) = (roll, pitch, yaw).
    ZYX extrinsic convention is used
    """
    # cphi = np.cos(phi)
    # sphi = np.sin(phi)
    # ctheta = np.cos(theta)
    # stheta = np.sin(theta)
    # cpsi = np.cos(psi)
    # spsi = np.sin(psi)
    # R = np.array([
    #     [cpsi * ctheta, cpsi * stheta * sphi - spsi * cphi, cpsi * stheta * cphi + spsi * sphi],
    #     [spsi * ctheta, spsi * stheta * sphi + cpsi * cphi, spsi * stheta * cphi - cpsi * sphi],
    #     [-stheta,        ctheta * sphi,                     ctheta * cphi]
    # ])
    R = Rot.from_euler('ZYX', [psi, theta, phi]).as_matrix() # case sensitive: upper case is extrinsic -> rotating in world frame
    return R

def euler_to_quat(roll, pitch, yaw):
    cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
    cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
    cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return np.array([w, x, y, z])

def skew(v):
    """
    Create a skew-symmetric matrix from a vector.
    """
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

def load_yaml(file_path):
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

def load_model_params(yaml_path):
    """
    Load model parameters from a ROS2 YAML config file.
    Args:
        yaml_path: str, path to the YAML file
    Returns:
        params: dict, containing model parameters
    """
    data = load_yaml(yaml_path)
    # Find the first key (wildcard node name)
    node_key = list(data.keys())[0]
    model_params = data[node_key]['ros__parameters']['model']
    return model_params

def load_dh_params(yaml_filename):
    dh_data = load_yaml(yaml_filename)
    links = []
    if isinstance(dh_data, dict) and '/**' in dh_data and 'ros__parameters' in dh_data['/**']:
        params = dh_data['/**']['ros__parameters']
        for key in sorted(params.keys()):
            link = params[key]
            if isinstance(link, dict):
                d = link.get('d', 0)
                theta = link.get('theta0', 0)
                a = link.get('a', 0)
                alpha = link.get('alp', 0)
                links.append([d, theta, a, alpha])
        DH_table = np.array(links)
    else:
        if isinstance(dh_data, dict) and 'dh_table' in dh_data:
            DH_table = np.array(dh_data['dh_table'])
        else:
            DH_table = np.array(dh_data)
    return DH_table

def load_joint_limits(yaml_filename):
    joint_data = load_yaml(yaml_filename)
    joint_limits, joint_efforts, joint_velocities, all_joints = [], [], [], {}
    for joint_name in sorted(joint_data.keys()):
        entry = joint_data[joint_name]
        joint_limits.append((entry.get('lower', None), entry.get('upper', None)))
        joint_efforts.append(entry.get('effort', None))
        joint_velocities.append(entry.get('velocity', None))
        all_joints[joint_name] = entry
    return joint_limits, joint_efforts, joint_velocities, all_joints

def load_dynamic_params(file_paths):
    merged_dict = {}
    for file in file_paths:
        data = load_yaml(file)
        params = data.get('/**', {}).get('ros__parameters', {})
        merged_dict = recursive_merge(merged_dict, params)
    return dict_to_namespace(merged_dict)

def recursive_merge(dict1, dict2):
    result = deepcopy(dict1)
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = recursive_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result

def dict_to_namespace(d):
    if not isinstance(d, dict):
        return d
    return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})

def read_wrench_txt(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            parts = [float(x.strip()) for x in line.split(',')]
            data.append(parts)
    data = np.array(data)
    t = data[:, 0]
    tau = data[:, 1:7]
    return t, tau

def dh2matrix(d, theta, a, alpha):
    ct, st = np.cos(theta), np.sin(theta)
    ca_, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st*ca_,  st*sa, a*ct],
        [st,  ct*ca_, -ct*sa, a*st],
        [0,      sa,     ca_,    d],
        [0,       0,      0,    1]
    ])

def quaternion_error_Niklas(q_goal, q_current):
    w_g, x_g, y_g, z_g = q_goal
    w_c, x_c, y_c, z_c = q_current
    v_g = np.array([x_g, y_g, z_g])
    v_c = np.array([x_c, y_c, z_c])
    goal_att_tilde = np.array([
        [0, -z_g, y_g],
        [z_g, 0, -x_g],
        [-y_g, x_g, 0]
    ])
    att_error = w_g * v_c - w_c * v_g - goal_att_tilde @ v_c
    return np.linalg.norm(att_error)