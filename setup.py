"""Shim for Ubuntu 22.04 setuptools (59.x), which cannot do a PEP 660
editable install from pyproject.toml alone. Used on the robot PC:

    pip install --user --no-deps --no-build-isolation -e .

--no-deps keeps the ROS-provided numpy/scipy untouched.
"""

from setuptools import find_packages, setup

setup(
    name="athome",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages("src"),
    python_requires=">=3.9",
)
