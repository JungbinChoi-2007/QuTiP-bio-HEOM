"""
==============================================================================
Script: figure_1.py
Project: Scalable Python Pipeline for Non-Markovian Quantum Dynamics
Author: Jungbin Choi
License: MIT License
Description: 
    Solver Sanity Check (N=2 Dimer). 
    Demonstrates baseline quantum energy transfer in a simplified two-site 
    tryptophan network using the Lindblad master equation via QuTiP.
==============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from qutip import Qobj, basis, mesolve

def run_simulation():
    # 1. Hamiltonian Setup (Site energies and coupling constant in cm^-1)
    E1, E2 = 35700.0, 35750.0
    V = 8.0
    H_sys = Qobj([[E1, V], 
                  [V, E2]])

    # 2. Initial State: 100% excitation localized on the first tryptophan (W103)
    psi0 = basis(2, 0) 

    # 3. Lindblad Collapse Operators
    # Simplified representation of the energy dissipation rate
    gamma = 2.0  # Arbitrary decoherence rate parameter
    c_ops = [np.sqrt(gamma) * Qobj([[0, 1], [0, 0]])]

    # 4. Time array for the simulation (200 observation steps)
    tlist = np.linspace(0, 1.0, 200)

    # 5. Observables: Probability of excitation at each site
    P1 = Qobj([[1, 0], [0, 0]])
    P2 = Qobj([[0, 0], [0, 1]])

    # 6. Time evolution using the Lindblad master equation (mesolve)
    print("Running N=2 master equation simulation...")
    result = mesolve(H_sys, psi0, tlist, c_ops, [P1, P2])
    print("Simulation complete. Generating plot...")

    # 7. Visualization of the results
    plt.figure(figsize=(10, 6))
    plt.plot(tlist, result.expect[0], label="Tryptophan 1 (W103)", color='blue', linewidth=2)
    plt.plot(tlist, result.expect[1], label="Tryptophan 2 (W407)", color='red', linewidth=2, linestyle='--')

    plt.title("Quantum Energy Transfer in N=2 Tryptophan Network")
    plt.xlabel("Time (Arbitrary Units)")
    plt.ylabel("Probability")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the figure automatically (Crucial for headless cloud/HPC execution)
    output_filename = "Figure_1.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Plot successfully saved as '{output_filename}'.")
    
    # Display if running in an interactive environment
    plt.show()

if __name__ == "__main__":
    run_simulation()
