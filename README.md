# QuTiP-bio-HEOM

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23004020.svg)](https://doi.org/10.5281/zenodo.23004020)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A scalable, fault-tolerant Python pipeline built on [QuTiP](https://qutip.org/) to simulate non-Markovian quantum dynamics (HEOM) in 3D biological macromolecular networks.

## Overview
Simulating complex biological environments using the Hierarchical Equations of Motion (HEOM) imposes severe computational bottlenecks. This tool bypasses these limitations by automating the translation of 3D spatial coordinates into exact distance matrices, coupling each node to independent Drude-Lorentz thermal baths, and ensuring recoverability against hardware halts.

## Repository Contents
- `figure_1.py` ~ `figure_8.py`: Python source codes used to run the simulations and generate the corresponding manuscript figures.
- `Figure_1.png` ~ `Figure_8.png`: The final plotted images used in the validation study.
- `N7_Ultimate_Sweep_results.npy`: The final numerical dataset generated from the 400-coordinate phase-space sweep.

## Execution Note & Fault Tolerance (Important)
The massive phase-space sweep across the 3D Tryptophan network was executed in a Google Colab Standard instance. The full 400-coordinate sweep takes approximately 11.4 hours to complete. 
To manage cloud hardware limits, the pipeline utilizes an **auto-checkpointing mechanism**. If execution is interrupted, re-running the script will automatically resume from the latest `.npy` checkpoint file.

## Dependencies
- Python 3.x
- QuTiP (Quantum Toolbox in Python)
- NumPy
- Matplotlib

## Citation
If you use this pipeline in your research, please cite the corresponding Zenodo archive and our upcoming methodology paper:

> Choi, Jungbin. (2026). Scalable Python Pipeline for Non-Markovian Quantum Dynamics in 3D Biological Networks (v1.0). Zenodo. https://doi.org/10.5281/zenodo.23004020
