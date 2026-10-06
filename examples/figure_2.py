"""
==============================================================================
Script: figure_2.py
Project: Scalable Python Pipeline for Non-Markovian Quantum Dynamics
Author: Jungbin Choi
License: MIT License
Description: 
    Solver Sanity Check (1D Uniform Chain).
    Executes a high-resolution parameter sweep to map the transfer efficiency 
    in an idealized N=7 uniform chain prior to 3D bottleneck integration.
==============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import itertools
from joblib import Parallel, delayed
from qutip import Qobj, basis
from qutip.solver.heom import HEOMSolver, DrudeLorentzBath
import time

# 1. N=7 System Fixed Parameters (1D Uniform Chain Baseline)
N = 7
E_site = 35700.0
V_coupling = 8.0

# Pre-compute fixed objects to minimize overhead during parallel sweeping
H_matrix = np.zeros((N, N))
for i in range(N):
    H_matrix[i, i] = E_site
    if i < N - 1:
        H_matrix[i, i+1] = V_coupling
        H_matrix[i+1, i] = V_coupling
H_sys = Qobj(H_matrix)

psi0 = basis(N, 0)
rho0 = psi0 * psi0.dag()
T = 215.0
Q = basis(N, 0) * basis(N, 0).dag()

# Optimized time array to accelerate massive parallel execution
tlist = np.linspace(0, 1.0, 50) 

def run_sweep_N7(lam, gamma):
    """
    Single simulation core: Evaluates the energy transfer to the terminal 
    tryptophan (Site 7) using the HEOMSolver.
    """
    try:
        bath = DrudeLorentzBath(Q, lam=lam, gamma=gamma, T=T, Nk=1)
        solver = HEOMSolver(H_sys, bath, max_depth=3)
        result = solver.run(rho0, tlist)

        # Survival probability at the terminal node (Site 7) at final time t=1.0
        P7_final = result.states[-1][N-1, N-1].real
        return [lam, gamma, P7_final]
    except Exception as e:
        # Fault tolerance: Return NaN instead of crashing the parallel sweep
        return [lam, gamma, np.nan]

def execute_pipeline():
    print("Initiating N=7 high-resolution parallel sweep (400 coordinates)...")
    
    # 2. Configure 20x20 phase-space parameter grid
    lam_range = np.linspace(10, 500, 20)
    gamma_range = np.linspace(50, 1000, 20)
    param_grid = list(itertools.product(lam_range, gamma_range))

    # 3. Massively Parallel Execution (Optimized for standard cloud CPU instances)
    start_time = time.time()
    results = Parallel(n_jobs=-1, verbose=5)(delayed(run_sweep_N7)(l, g) for l, g in param_grid)
    end_time = time.time()
    print(f"Sweep completed successfully. Total execution time: {end_time - start_time:.2f} seconds")

    # 4. Data Serialization and Visualization
    results_array = np.array(results)
    data_filename = "N7_final_sweep_results.npy"
    np.save(data_filename, results_array)
    print(f"Computational datasets serialized to '{data_filename}'.")

    # Reshape array for heatmap generation
    Z = results_array[:, 2].reshape(20, 20).T

    plt.figure(figsize=(9, 7))
    im = plt.imshow(Z, extent=[10, 500, 50, 1000], origin='lower', aspect='auto', cmap='magma')
    plt.colorbar(im, label="Transfer Efficiency (P7 at t=1.0)")
    plt.xlabel(r"Reorganization Energy, $\lambda$ (cm$^{-1}$)")
    plt.ylabel(r"Cutoff Frequency, $\gamma$ (cm$^{-1}$)")
    plt.title("N=7 Tryptophan Network: Long-Range Energy Transfer Efficiency")
    
    output_filename = "Figure_2.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Heatmap visualization saved as '{output_filename}'.")
    
    # Display if running in an interactive environment
    plt.show()

if __name__ == "__main__":
    execute_pipeline()
