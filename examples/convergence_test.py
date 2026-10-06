"""
==============================================================================
Script: convergence_test.py
Project: Scalable Python Pipeline for Non-Markovian Quantum Dynamics
Author: Jungbin Choi
License: MIT License
Description: 
    Convergence and Sensitivity Analysis.
    Evaluates the numerical stability and convergence of the HEOM solver 
    across varying hierarchical depths (max_depth) and ensemble sizes 
    (n_traj) for the full 3D N=7 tryptophan network.
==============================================================================
"""

import numpy as np
from qutip import Qobj, basis
from qutip.solver.heom import HEOMSolver, DrudeLorentzBath
import time

def run_convergence_test():
    print("Initiating Convergence & Sensitivity Analysis...")

    # 1. 3D N=7 System Configuration (Consistent with manuscript)
    N = 7
    E_site_base = 35700.0
    V0 = 8.0
    R0 = 14.0
    T = 300.0 

    # Fix stable benchmark point (yielding approx. 7.5e-5 efficiency)
    lam = 50.0   
    gamma = 100.0

    # Empirical 3D coordinates from structural data
    coords = np.array([
        [0.0, 0.0, 0.0],
        [14.0, 0.0, 0.0],
        [25.31, 11.31, 0.0],
        [44.40, 30.40, 0.0],
        [55.01, 41.01, 0.0],
        [69.15, 55.15, 0.0],
        [79.05, 65.05, 0.0]
    ])

    # Construct Distance Matrix and Baseline Hamiltonian
    R_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                R_matrix[i, j] = np.linalg.norm(coords[i] - coords[j])

    # Isotropic orientation factor explicitly defined
    kappa_factor = np.sqrt(2.0 / 3.0)
    
    H_base = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                H_base[i, j] = V0 * kappa_factor * (R0 / R_matrix[i, j])**3

    # 2. HEOM Solver Common Environment Parameters
    psi0 = basis(N, 0)
    rho0 = psi0 * psi0.dag()
    tlist = np.linspace(0, 1.0, 50)

    # Attach independent Drude-Lorentz baths to each site
    Q_ops = [basis(N, i) * basis(N, i).dag() for i in range(N)]
    baths = [DrudeLorentzBath(Q_ops[i], lam=lam, gamma=gamma, T=T, Nk=1) for i in range(N)]

    # QuTiP v5 advanced stability options
    opts = {"nsteps": 20000, "method": "bdf", "atol": 1e-10, "rtol": 1e-8}

    # 3. Core function to evaluate P7 transfer efficiency
    def evaluate_convergence(depth, n_traj):
        P7_sum = 0.0
        for traj in range(n_traj):
            H_MC = H_base.copy()
            # Inject static disorder for ensemble averaging
            for i in range(N):
                H_MC[i, i] = E_site_base + np.random.normal(0, 50.0)

            solver = HEOMSolver(Qobj(H_MC), baths, max_depth=depth, options=opts)
            result = solver.run(rho0, tlist)
            P7_sum += result.states[-1][N-1, N-1].real
            
        return P7_sum / n_traj

    # 4. Test configurations for depth and ensemble size
    depths = [2, 3, 4]
    n_trajs = [1, 3, 5]

    print(f"\nFixed Params: lambda={lam}, gamma={gamma}\n")
    print(f"{'Max Depth':<12} | {'Ensemble (n)':<15} | {'P7 Transfer Efficiency':<25} | {'Time (s)'}")
    print("-" * 75)

    # 5. Execute sequential evaluation across combinations
    for d in depths:
        for n in n_trajs:
            t0 = time.time()
            # Unfixed random seed ensures genuine physical noise sampling
            eff = evaluate_convergence(d, n)
            t1 = time.time()
            
            # Output formatted using scientific notation
            print(f"{d:<12} | {n:<15} | {eff:<25.4e} | {t1-t0:.2f}")

    print("-" * 75)
    print("Convergence test completed successfully.")

if __name__ == "__main__":
    run_convergence_test()