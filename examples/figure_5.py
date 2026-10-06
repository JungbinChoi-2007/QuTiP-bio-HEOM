"""
==============================================================================
Script: figure_5.py
Project: Scalable Python Pipeline for Non-Markovian Quantum Dynamics
Author: Jungbin Choi
License: MIT License
Description: 
    Geometry Module Validation (Automated 3D Distance Couplings).
    Demonstrates the translation of empirical structural data into a 
    Euclidean distance matrix and evaluates exciton trapping across a 
    27.0 Å spatial gap.
==============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from qutip import Qobj, basis
from qutip.solver.heom import HEOMSolver, DrudeLorentzBath
import time

def run_geometry_mapping():
    print("Initiating N=7 HEOM simulation with realistic spatial couplings...")

    # 1. Empirical Distance Configuration (in Angstroms Å)
    # Represents sequential distances in a microtubule tryptophan network,
    # including the critical 27.0 Å alpha-beta monomer boundary gap.
    distances = [14.0, 16.0, 27.0, 15.0, 20.0, 14.0]

    # Construct 7x7 Distance Matrix (R_matrix)
    N = 7
    R_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i == j:
                R_matrix[i, j] = 1e-10  # Prevent division by zero errors
            else:
                # Calculate cumulative linear distance between nodes i and j
                dist = sum(distances[min(i,j):max(i,j)])
                R_matrix[i, j] = dist

    # 2. Hamiltonian Construction via Dipole-Dipole Interaction Formula
    E_site = 35700.0
    V0 = 8.0     # Baseline coupling at R0 (cm^-1)
    R0 = 14.0    # Baseline distance (Å)
    
    # Isotropic orientation factor explicitly defined in Manuscript Eq. (2)
    kappa = np.sqrt(2/3) 

    H_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i == j:
                H_matrix[i, j] = E_site
            else:
                # Inverse-cubed distance dependence: V = V0 * kappa * (R0/R)^3
                H_matrix[i, j] = V0 * kappa * (R0 / R_matrix[i, j])**3

    H_sys = Qobj(H_matrix)
    print("Distance-based H_matrix successfully constructed.")

    # 3. Environment Parameters and HEOM Initialization
    psi0 = basis(N, 0)
    rho0 = psi0 * psi0.dag()
    T = 215.0      # Equivalent to 310K physiological temp in cm^-1
    lam = 50.0
    gamma = 100.0
    
    # Bath coupling localized on the initial excitation site
    Q = basis(N, 0) * basis(N, 0).dag()

    bath = DrudeLorentzBath(Q, lam=lam, gamma=gamma, T=T, Nk=1)
    HEOM_solver = HEOMSolver(H_sys, bath, max_depth=3)
    tlist = np.linspace(0, 1.0, 100)

    # 4. Execution
    start_time = time.time()
    result = HEOM_solver.run(rho0, tlist)
    end_time = time.time()
    print(f"Simulation complete. Execution time: {end_time - start_time:.2f} seconds")

    # 5. Extract Populations and Visualize
    P1 = [rho[0, 0].real for rho in result.states]
    P4 = [rho[3, 3].real for rho in result.states]
    P7 = [rho[6, 6].real for rho in result.states]

    plt.figure(figsize=(10, 6))
    plt.plot(tlist, P1, label="Site 1 (Initial)", color='blue', linewidth=2)
    plt.plot(tlist, P4, label="Site 4 (Middle)", color='green', linewidth=2, linestyle='-.')
    plt.plot(tlist, P7, label="Site 7 (End)", color='red', linewidth=2, linestyle='--')

    plt.title("Dynamics with Realistic 3D Distance Couplings")
    plt.xlabel("Time")
    plt.ylabel("Probability")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save output matching the manuscript figure designation
    output_filename = "Figure_5.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Plot successfully saved as '{output_filename}'.")
    
    plt.show()

if __name__ == "__main__":
    run_geometry_mapping()
