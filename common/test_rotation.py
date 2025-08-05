import numpy as np
from scipy.spatial.transform import Rotation as Rot

def rotation_matrix_from_euler(phi, theta, psi):
    """
    Compute rotation matrix from ZYX extrinsic Euler angles.
    Angles: phi (roll, X), theta (pitch, Y), psi (yaw, Z)
    """
    cphi = np.cos(phi)
    sphi = np.sin(phi)
    ctheta = np.cos(theta)
    stheta = np.sin(theta)
    cpsi = np.cos(psi)
    spsi = np.sin(psi)
    R = np.array([
        [cpsi * ctheta, cpsi * stheta * sphi - spsi * cphi, cpsi * stheta * cphi + spsi * sphi],
        [spsi * ctheta, spsi * stheta * sphi + cpsi * cphi, spsi * stheta * cphi - cpsi * sphi],
        [-stheta,        ctheta * sphi,                     ctheta * cphi]
    ])
    return R

def test_equivalence():
    np.set_printoptions(precision=4, suppress=True)
    print("Comparing manual ZYX extrinsic vs scipy ZYX intrinsic (reversed input)...\n")

    for i in range(5):
        # Random angles between -pi and pi
        phi, theta, psi = np.random.uniform(-np.pi, np.pi, 3)

        R_manual = rotation_matrix_from_euler(phi, theta, psi)
        R_scipy = Rot.from_euler('ZYX', [psi, theta, phi]).as_matrix()

        diff = R_manual - R_scipy
        max_diff = np.max(np.abs(diff))

        print(f"Test {i+1}:")
        print(f"  roll (phi):   {phi:.4f}")
        print(f"  pitch (theta):{theta:.4f}")
        print(f"  yaw (psi):    {psi:.4f}")
        print(f"  Max abs diff: {max_diff:.2e}")
        print(f"  Diff matrix:\n{diff}\n")

if __name__ == "__main__":
    test_equivalence()