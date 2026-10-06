"""
==============================================================================
Script: figure_7.py
Project: Scalable Python Pipeline for Non-Markovian Quantum Dynamics
Author: Jungbin Choi
License: MIT License
Description: 
    Detailed Node Population Tracking.
    Tracks site-specific exciton population fluctuations over time in an 
    N=7 tryptophan network using the HEOM solver to visualize quantum 
    energy transfer dynamics.
==============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from qutip import Qobj, basis
from qutip.solver.heom import HEOMSolver, DrudeLorentzBath
import time

def run_population_tracking():
    print("Initiating N=7 HEOM population tracking simulation...")

    # 1. N=7 System Parameter Setup
    N = 7
    E_site = 35700.0  # Baseline site energy (cm^-1)
    V_coupling = 8.0  # Nearest-neighbor electronic coupling (cm^-1)

    # Initialize 7x7 Hamiltonian matrix
    H_matrix = np.zeros((N, N))

    # Assign diagonal (site energies) and off-diagonal (couplings) elements
    for i in range(N):
        H_matrix[i, i] = E_site
        if i < N - 1:
            H_matrix[i, i+1] = V_coupling
            H_matrix[i+1, i] = V_coupling

    H_sys = Qobj(H_matrix)
    print("N=7 Hamiltonian successfully constructed.")

    # 2. Initial State: 100% excitation localized on the first tryptophan (Site 1)
    psi0 = basis(N, 0)
    rho0 = psi0 * psi0.dag()

    # 3. Drude-Lorentz Bath Setup (Single environment noise for baseline tracking)
    T = 215.0     # Equivalent to 310K physiological temp in cm^-1
    lam = 50.0    # Reorganization energy (cm^-1)
    gamma = 100.0 # Cutoff frequency (cm^-1)

    # Bath coupling localized on the initial excitation site
    Q = basis(N, 0) * basis(N, 0).dag()
    bath = DrudeLorentzBath(Q, lam=lam, gamma=gamma, T=T, Nk=1)

    # 4. HEOM Solver Execution
    print("Executing HEOM dynamics... (This may take a moment)")
    start_time = time.time()

    # max_depth set to 3 for stable accuracy across the 7-node network
    HEOM_solver = HEOMSolver(H_sys, bath, max_depth=3)
    tlist = np.linspace(0, 1.0, 100)
    result = HEOM_solver.run(rho0, tlist)

    end_time = time.time()
    print(f"Simulation complete. Execution time: {end_time - start_time:.2f} seconds")

    # 5. Visualization: Energy probability tracking for Sites 1, 4, and 7
    P1 = [rho[0, 0].real for rho in result.states]
    P4 = [rho[3, 3].real for rho in result.states]
    P7 = [rho[6, 6].real for rho in result.states]

    plt.figure(figsize=(10, 6))
    plt.plot(tlist, P1, label="Site 1 (Initial)", color='blue', linewidth=2)
    plt.plot(tlist, P4, label="Site 4 (Middle)", color='green', linewidth=2, linestyle='-.')
    plt.plot(tlist, P7, label="Site 7 (End)", color='red', linewidth=2, linestyle='--')

    plt.title("N=7 Tryptophan Network HEOM Dynamics")
    plt.xlabel("Time")
    plt.ylabel("Probability")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save output matching the manuscript figure designation
    output_filename = "Figure_7.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Plot successfully saved as '{output_filename}'.")
    
    # Display if running in an interactive environment
    plt.show()

if __name__ == "__main__":
    run_population_tracking()
