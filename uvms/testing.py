from bluerov.thruster_b2_inversepoly import ThrusterInversePoly
import time
from common.my_package_path import get_package_path
import numpy as np


def main():
    bluerov_package_path = get_package_path('bluerov')
    thruster_model_path = bluerov_package_path + "/thruster_models/thruster_inversepoly_deg2.npz"
    model = ThrusterInversePoly.load(thruster_model_path)

    V_batt = 15.0  # Voltage
    desired_force = np.linspace(-80.0, 80.0, 50)  # Desired forces from -2N to 5N

    PWM = [model.command_simple(f, V_batt) for f in desired_force]
    a = [(f, u) for f, u in zip(desired_force.tolist(), PWM)]
    for force, pwm in a:
        print(f"{force:.3f} -> {pwm:.0f}")


    return

if __name__ == "__main__":
    main()