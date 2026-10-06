"""
==============================================================================
Script: figure_3.py
Project: Scalable Python Pipeline for Non-Markovian Quantum Dynamics
Author: Jungbin Choi
License: MIT License
Description: 
    Cross-Workflow Consistency (Markovian Baseline).
    Simulates exciton dynamics under Markovian memoryless assumptions 
    to demonstrate rapid oscillatory decay in a 2-site network.
==============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from qutip import Qobj, basis, mesolve

def run_markovian_baseline():
    # 1. Hamiltonian Setup (Site energies and coupling constant in cm^-1)
    E1, E2 = 35700.0, 35700.0
    V = 8.0
    H_sys = Qobj([[E1, V], [V, E2]])

    # 2. Initial State: 100% excitation localized on the first tryptophan
    psi0 = basis(2, 0) 

    # 3. Lindblad Collapse Operators (Markovian dissipation)
    gamma = 2.0
    c_ops = [np.sqrt(gamma) * Qobj([[0, 1], [0, 0]])]

    # 4. Time array and observable states
    tlist = np.linspace(0, 1.0, 200)
    P1 = Qobj([[1, 0], [0, 0]])
    P2 = Qobj([[0, 0], [0, 1]])

    # 5. Time evolution calculation via Lindblad master equation
    print("Executing Markovian baseline simulation...")
    result = mesolve(H_sys, psi0, tlist, c_ops, [P1, P2])
    print("Simulation complete. Generating plot...")

    # 6. Result Serialization and Visualization
    plt.figure(figsize=(10, 6))
    plt.plot(tlist, result.expect[0], label="Tryptophan 1 (W103)", color='blue', linewidth=2)
    plt.plot(tlist, result.expect[1], label="Tryptophan 2 (W407)", color='red', linestyle='--', linewidth=2)

    plt.title("Quantum Energy Transfer: Markovian Baseline")
    plt.xlabel("Time")
    plt.ylabel("Probability")
    plt.legend()
    
    # Save output matching the manuscript figure designation
    output_filename = "Figure_3.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Plot successfully saved as '{output_filename}'.")
    
    plt.show()

if __name__ == "__main__":
    run_markovian_baseline()
