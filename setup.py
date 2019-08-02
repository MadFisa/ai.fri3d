import os
import shutil
import subprocess
import sys

# for consistent encoding
from codecs import open

from setuptools import find_packages, setup

version_py = open(os.path.join(os.path.dirname(__file__), "version.py")).read().strip().split("=")[-1].replace('"', "")

# Get the long description from the README file
with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), "README.rst"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="ai.fri3d",
    version="{ver}".format(ver=version_py),
    description="FRi3D model of coronal mass ejections",
    long_description=long_description,
    url="https://bitbucket.org/isavnin/ai.fri3d",
    author="Alexey Isavnin",
    author_email="alexey.isavnin@gmail.com",
    license="MIT",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
    keywords="research space solar physics science model coronal mass ejection flare CME",
    packages=find_packages("src", exclude=["test*"]),
    package_dir={"": "src"},
    install_requires=["numpy", "scipy", "matplotlib", "numba", "astropy", "fastdtw", "ai.cs"],
)
