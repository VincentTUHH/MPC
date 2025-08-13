# UVMS MPC Framework

This repository contains a modular framework for underwater robot control using Model Predictive Control (MPC). It includes simulation, control, and utility components for a BlueROV equipped with the Reach Alpha 5 arm.

---

## 🛠️ Setup Instructions

Follow these steps to get started with this project after cloning:

### 1. Clone the repository

```bash
git clone https://github.com/your-username/bluerov-mpc.git
cd bluerov-mpc
```

### 2. Create a virtual environment (recommended)

To avoid global package conflict and keep dependencies isolated:
```bash
python3 -m venv .venv
```

### 3. Activate virtual environemnt

On macOS / Linux
```bash
source .venv/bin/activate
```

### 4. Intsall the packages and dependencies

Once the virtual environment is activated (the path in the terminal begins with `(.venv)`):
```bash
pip install -e .
```
This installs the project in “editable mode”, so changes to the source code apply immediately.

### 5. Run scripts

From the root of the repo execute scripts like:
```bash
python3 -m package_name.file_name
```
Example:
```bash
python3 -m bluerov.vehicle_mpc
```

### 6. Deactivate the virtual environment

```bash
deactivate
```

## 🧠 macOS Users (Apple Silicon): CasADi + IPOPT Setup

If you’re using CasADi with the IPOPT solver on macOS and encounter an error like:
```bash
Library not loaded: @rpath/libgfortran.5.dylib
```
this means IPOPT can’t find the Fortran runtime (libgfortran). You need to expose the library manually.

### 1. Install GCC via Homebrew

```bash
brew install gcc
```

You might encounter problems here, look at troubleshooting below for help.

### 2. Find the libgfortran.5.dylib path

Run:

```bash
find /opt/homebrew -name "libgfortran.5.dylib"
```

Example output:
```bash
/opt/homebrew/Cellar/gcc/15.1.0/lib/gcc/15/libgfortran.5.dylib
```

### 3. Update the virtual environment to find the library

Open `.venv/bin/activate` and add the following line at the bottom:
```bash
export DYLD_LIBRARY_PATH="/opt/homebrew/Cellar/gcc/15.1.0/lib/gcc/15:$DYLD_LIBRARY_PATH"
```
Make sure the path matches the output from step 2!

### 4. Activate the venv

Run as before:
```bash
source .venv/bin/activate
```
But now the necessary dependencies for IPOPT are found now.

## 🧯 Troubleshooting

### 🧨 Homebrew refuses to update or install packages

If you encounter an error like:
```bash
error: Not a valid ref: refs/remotes/origin/main
```
and `brew update`, `brew install`, or `brew tap` all fail, your Homebrew installation may be corrupted — especially after an unshallow git fetch.

#### 1. Remove Homebrew

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/uninstall.sh)"
```
You might also need to locate the `homebrew` or `Homebrew` folder if that fails, then also do:
```bash
sudo rm -rf /usr/local/Homebrew /usr/local/Caskroom /usr/local/bin/brew
```

Check if homebrew is installed. No result is expected when:
```bash
which brew
```

#### 2. Reinstall Homebrew

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install.sh)"
```

On Apple Silicon the folder should be under `/opt/homebrew`.

#### 3. Final steps

After reinstalling the terminal will tell what to do next, like adding Homebrew to the shell.
Example:
```bash
echo >> /Users/my_user.name/.zprofile
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> /Users/my_user.name/.zprofile
eval "$(/opt/homebrew/bin/brew shellenv)"
source ~/.zprofile
```

#### 4. Test brew

Does is exist?
```bash
which brew
```

Then update
```bash
brew update
```

[Jump to CasADi + IPOPT Setup for macOS (Apple Silicon)](#-macos-users-apple-silicon-casadi--ipopt-setup)


## The Code

The codebase is organized in packages. You need to be in the root path to execute with `python3 -m file`

### Manipulator

Proof of Concept to verify the recursive Newton-Euler method against the C++ implementtaion from Trekel23
```bash
python3 -m manipulator.poc_manipulator_model
```

An MPC solely for a manipulator that is earth fixed at (0, 0, 0). Uses only kinematic equations.
```bash
python3 -m manipulator.manipulator_kinematics_mpc
```

### Bluerov

Simulate the BlueROV on different trajectories.
```bash
python3 -m bluerov.bluerov
```

Estimates a (linear) model of the thruster using its characteristic curves at different voltages (see Blue Robotics).
```bash
python3 -m bluerov.thruster_model
```

An MPC solely for the BlueROV to track trajectories on the basis of Fossen's equation of motion.
```bash
python3 -m bluerov.vehicle_mpc
```