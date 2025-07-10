import bluerov
import manipulator
import numpy as np

# TODO: 
# lese und extrahiere Matrix-Vektor dynamics for manipulators from Schjolbergs dissertation
# define manipulator dynamics both in recursive Newton-Euler fashion (Trekel, (Ioi)) and in closed form Newton-Euler having Matrices (Scholberg,Fossen)
# dafine manipulator dynamics using Articulated Body Algorithm (McMillan) -> nein, scheint kein Vorteil zu Newton-Euler zu haben, ist auch iteratives Verfahren


def wrench_rnea(q, qd, qdd):
    """
    Recursive Newton-Euler Algorithm for computing the wrench (force/torque) at the end-effector.
    
    Parameters:
    q : np.ndarray
        Joint angles (n_joints,)
    qd : np.ndarray
        Joint velocities (n_joints,)
    qdd : np.ndarray
        Joint accelerations (n_joints,)

    Returns:
    wrench : np.ndarray
        Wrench at the base link 0
    """
    # Placeholder for the actual implementation
    # This function should compute the wrench based on the manipulator's dynamics
    return np.zeros(6)


def main():
    bluerov_params = bluerov.load_model_params('model_params.yaml')
    bluerov_dynamics = bluerov.BlueROVDynamics(bluerov_params)

    manipulator_params = manipulator.load_dh_params('alpha_kin_params.yaml')
    joint_limits, joint_efforts, joint_velocities, all_joints = manipulator.load_joint_limits('alpha_joint_lim_real.yaml')


    return

if __name__ == "__main__":
    main()