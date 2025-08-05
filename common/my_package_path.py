import os

def get_package_path(package_name):
    """
    Returns the absolute path to the given package name within the parent directory of the parent directory of the current file.
    Raises FileNotFoundError if the package is not found.
    """
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    package_path = os.path.join(parent_dir, package_name)
    if not os.path.isdir(package_path):
        raise FileNotFoundError(f"Package '{package_name}' not found in '{parent_dir}'")
    return package_path
