import os
import shutil
import subprocess
import sys
from codecs import open

from setuptools import find_packages, setup, Extension

version_py = open(os.path.join(os.path.dirname(__file__), "version.py")).read().strip().split("=")[-1].replace('"', "")

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
    ext_modules=[Extension("ai.fri3d.lib", [os.path.join("src", "ai", "fri3d", "lib.pyx")])],
    install_requires=["numpy", "scipy", "matplotlib", "astropy", "fastdtw", "ai.cs"],
    setup_requires=["setuptools>=18.0", "cython"],
)
