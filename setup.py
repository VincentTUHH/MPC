from setuptools import setup, find_packages

setup(
    name="mpc_uvms",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pyyaml",
        "scipy",
        "matplotlib",
        "pandas",
        "scikit-learn",  # Correct name for sklearn
        "casadi",
    ],
)