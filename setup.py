import os
from typing import List
from setuptools import find_packages, setup

# Basic package info
NAME = "deep_learning_project_template"
VERSION = "0.0.1"


def read_requirements(fname: str) -> List[str]:
    """Read requirements from file and return as list"""
    with open(os.path.join(os.path.dirname(__file__), fname)) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

# Requirements
install_requires = read_requirements("requirements/requirements.txt")
extras_require = {
    "dev": read_requirements("requirements/requirements-dev.txt"),
    "test": read_requirements("requirements/requirements-test.txt"),
}

setup(
    name=NAME,
    version=VERSION,
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=install_requires,
    extras_require=extras_require,
    python_requires=">=3.10",
)
