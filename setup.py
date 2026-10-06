from setuptools import setup, find_packages

setup(
    name="qutip-bio-heom",
    version="0.1.0",
    author="Jungbin Choi",
    description="A scalable Python wrapper pipeline for non-Markovian quantum dynamics (HEOM) in 3D biological networks.",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "qutip>=5.0.0",
        "matplotlib>=3.4.0",
        "joblib>=1.0.0"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires='>=3.8',
)