from setuptools import setup, find_packages

setup(
    name="tsdynamics",
    version="0.1.0",
    description="A package for simulating dynamical systems and processing tools for timeseries.",
    author="Daniel Estevez",
    author_email="kemossabee@gmail.com",
    url="https://github.com/El3ssar/TSDynamics",
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        "ddeint==0.3.0",
        "numpy==1.26.4",
        "numba==0.60.0",
        "pandas==2.2.3",
        "scipy==1.14.1",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
)

