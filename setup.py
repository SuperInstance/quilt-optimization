"""quilt-optimization — NVIDIA cuOpt as a Quilt substrate."""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    readme = f.read()

setup(
    name="quilt-optimization",
    version="0.1.0",
    description="NVIDIA cuOpt (LP/MILP/QP/routing) as a Quilt substrate",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/SuperInstance/quilt-optimization",
    author="SuperInstance",
    author_email="team@superinstance.dev",
    license="Apache-2.0",
    python_requires=">=3.11",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        # No hard deps — mock backends work with just stdlib
        # scipy is optional for the LP mock backend (HiGHS solver)
        # numpy is optional (used by scipy)
    ],
    extras_require={
        "cuopt": [
            # Real cuOpt requires NVIDIA GPU + these packages
            "nvidia-cuda-runtime-cu12>=12.9",
            "cuopt-server-cu12>=26.10",
            "cuopt-sh-client>=26.10",
        ],
        "lp-mock": [
            # For LP mock backend: scipy.optimize.linprog + milp
            "scipy>=1.10",
            "numpy>=1.24",
        ],
        "dev": [
            "scipy>=1.10",
            "numpy>=1.24",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
