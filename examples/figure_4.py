"""
==============================================================================
Script: figure_4.py
Project: Scalable Python Pipeline for Non-Markovian Quantum Dynamics
Author: Jungbin Choi
License: MIT License
Description: 
    Cross-Workflow Consistency (Non-Markovian Environment).
    Simulates exciton dynamics using the Hierarchical Equations of Motion (HEOM)
    with a Drude-Lorentz bath to demonstrate the suppression of unmitigated 
    decay compared to the Markovian baseline.
==============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from qutip import Qobj, basis
from qutip.solver.heom import HEOMSolver, DrudeLorentzBath

def run_non_markovian_heom():
    # 1. Hamiltonian Setup (Site energies and coupling constant in cm^-1)
    E1, E2 = 35700.0, 35700.0 
    V = 8.0
    H_sys = Qobj([[E1, V], [V, E2]])

    # 2. Initial State and Density Matrix (100% excitation on Site 1)
    psi0 = basis(2, 0)
    rho0 = psi0 * psi0.dag()

    # 3. Bath Parameters
    # T = 215.0 cm^-1 corresponds to the thermal energy (k_B * T) at 
    # physiological temperature 310K, ensuring consistent energy units.
    T = 215.0 
    lam = 50.0     # Reorganization energy in cm^-1
    gamma = 100.0  # Cutoff frequency in cm^-1

    # 4. Bath Coupling Operator (Coupling localized at Site 1)
    Q = Qobj([[1, 0], [0, 0]])
    bath = DrudeLorentzBath(Q, lam=lam, gamma=gamma, T=T, Nk=1)

    # 5. HEOM Solver Initialization (max_depth restricted to 3 for stability)
    HEOM_solver = HEOMSolver(H_sys, bath, max_depth=3)
    
    # 6. Time array and Execution
    tlist = np.linspace(0, 0.5, 200)
    print("Executing Non-Markovian HEOM simulation...")
    result = HEOM_solver.run(rho0, tlist)
    print("Simulation complete. Generating plot...")

    # 7. Extracting Probabilities
    P1_heom = [rho[0, 0].real for rho in result.states]
    P2_heom = [rho[1, 1].real for rho in result.states]

    # 8. Visualization
    plt.figure(figsize=(10, 6))
    plt.plot(tlist, P1_heom, label="Tryptophan 1 (W103)", color='blue', linewidth=2)
    plt.plot(tlist, P2_heom, label="Tryptophan 2 (W407)", color='red', linestyle='--', linewidth=2)

    plt.title("HEOM Dynamics: Non-Markovian Environment (T=310K)")
    plt.xlabel("Time")
    plt.ylabel("Probability")
    plt.legend()
    
    # Save output matching the manuscript figure designation
    output_filename = "Figure_4.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Plot successfully saved as '{output_filename}'.")
    
    plt.show()

if __name__ == "__main__":
    run_non_markovian_heom()
