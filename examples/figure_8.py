"""
==============================================================================
Script: figure_8.py
Project: Scalable Python Pipeline for Non-Markovian Quantum Dynamics
Author: Jungbin Choi
License: MIT License
Description: 
    Provisional Biological Demonstration & Pipeline Validation.
    Executes a massively parallel 20x20 parameter sweep evaluating the 
    ensemble-averaged transfer efficiency across the complete 3D empirical 
    geometry. Features dynamic auto-checkpointing for cloud environments.
==============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import itertools
import os
from qutip import Qobj, basis
from qutip.solver.heom import HEOMSolver, DrudeLorentzBath
import time
import gc
from google.colab import drive

def execute_ultimate_pipeline():
    print("Mounting Google Drive for persistent auto-checkpointing...")
    drive.mount('/content/drive')
    
    # Persistent storage paths for fault-tolerant execution
    save_path_npy = '/content/drive/MyDrive/N7_Ultimate_Sweep_results.npy'
    save_path_png = '/content/drive/MyDrive/N7_Ultimate_Sweep_heatmap.png'

    print("Initiating N=7 Ultimate 3D Sweep (20x20 Grid) with Auto-Recovery...")

    # 1. Base System Parameters
    N = 7
    E_site_base = 35700.0
    V0 = 8.0
    R0 = 14.0
    T = 300.0  # Physiological temperature (300K) corresponding to approx 208 cm^-1

    # Empirical 3D Coordinates (in Angstroms)
    coords = np.array([
        [0.0, 0.0, 0.0],
        [14.0, 0.0, 0.0],
        [25.31, 11.31, 0.0],
        [44.40, 30.40, 0.0],
        [55.01, 41.01, 0.0],
        [69.15, 55.15, 0.0],
        [79.05, 65.05, 0.0]
    ])

    # 2. Distance Matrix and Hamiltonian Mapping
    R_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                R_matrix[i, j] = np.linalg.norm(coords[i] - coords[j])

    kappa_factor = np.sqrt(2.0 / 3.0)  # Isotropic orientation factor

    H_base = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                H_base[i, j] = V0 * kappa_factor * (R0 / R_matrix[i, j])**3

    psi0 = basis(N, 0)
    rho0 = psi0 * psi0.dag()
    tlist = np.linspace(0, 1.0, 50)
    Q_ops = [basis(N, i) * basis(N, i).dag() for i in range(N)]

    # 3. Core Simulation Logic with Static Disorder (Monte Carlo)
    def run_ultimate_model(lam, gamma, n_traj=3):
        baths = [DrudeLorentzBath(Q_ops[i], lam=lam, gamma=gamma, T=T, Nk=1) for i in range(N)]
        P7_sum = 0.0

        for traj in range(n_traj):
            H_MC = H_base.copy()
            # Inject Gaussian static disorder to site energies
            for i in range(N):
                H_MC[i, i] = E_site_base + np.random.normal(0, 50.0)

            try:
                solver = HEOMSolver(Qobj(H_MC), baths, max_depth=3)
                result = solver.run(rho0, tlist)
                P7_sum += result.states[-1][N-1, N-1].real

                # Aggressive memory management for constrained hardware
                del solver
                del result
                gc.collect()

            except MemoryError:
                # Truncation limits hit: return NaN for parameter combination
                return [lam, gamma, np.nan]

        return [lam, gamma, P7_sum / n_traj]

    # 4. Phase-Space Grid Generation
    lam_range = np.linspace(10, 500, 20)
    gamma_range = np.linspace(50, 1000, 20)
    param_grid = list(itertools.product(lam_range, gamma_range))

    results = []
    start_idx = 0

    # ==========================================
    # Fault Tolerance: Auto-Recovery / Resume Logic
    # ==========================================
    if os.path.exists(save_path_npy):
        try:
            previous_results = np.load(save_path_npy)
            results = previous_results.tolist()
            start_idx = len(results)
            print(f"\n[INFO] Validated checkpoint found. Resuming execution from node {start_idx}...\n")
        except:
            print("\n[WARNING] Checkpoint corrupted or unreadable. Initiating fresh sweep.\n")

    if start_idx < len(param_grid):
        start_time = time.time()
        print(f"Executing sequential loop for real-time serialization: Node {start_idx+1} to {len(param_grid)}...")

        for idx in range(start_idx, len(param_grid)):
            l, g = param_grid[idx]

            step_start = time.time()
            res = run_ultimate_model(l, g)
            results.append(res)
            step_end = time.time()

            print(f"[{idx+1}/{len(param_grid)}] Lam={l:.1f}, Gamma={g:.1f} completed in {step_end - step_start:.1f}s")

            # Auto-checkpoint every 10 iterations to bypass execution halts
            if (idx + 1) % 10 == 0:
                np.save(save_path_npy, np.array(results))
                print(f"  => [CHECKPOINT] State securely serialized at node {idx+1}.")

        end_time = time.time()
        print(f"\n[COMPLETED] Sweep execution finished. Total segment time: {end_time - start_time:.2f} seconds")

    # ==========================================
    # Final Data Serialization and Visualization
    # ==========================================
    results_array = np.array(results)
    np.save(save_path_npy, results_array)
    print(f"Final dataset synchronized to '{save_path_npy}'.")

    if len(results_array) == len(param_grid):
        Z = results_array[:, 2].reshape(20, 20).T

        plt.figure(figsize=(10, 8))
        im = plt.imshow(Z, extent=[10, 500, 50, 1000], origin='lower', aspect='auto', cmap='magma')
        
        # Scientific notation formatting for extremely low probability values
        cbar = plt.colorbar(im, label="Transfer Efficiency (P7 at t=1.0, 3-Ensemble Avg)")
        cbar.formatter.set_powerlimits((0, 0))

        plt.xlabel(r"Reorganization Energy, $\lambda$ (cm$^{-1}$)")
        plt.ylabel(r"Cutoff Frequency, $\gamma$ (cm$^{-1}$)")
        plt.title("Ultimate 3D Tryptophan Network: Quantum Coherence Preservation")
        
        output_filename = "Figure_8.png"
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        # Also save to drive for persistence
        plt.savefig(save_path_png, dpi=300, bbox_inches='tight')
        print(f"Heatmap rendered and saved as '{output_filename}'.")
        
        plt.show()
    else:
        print("Execution halted prior to completion. Re-run script to resume from latest checkpoint.")

if __name__ == "__main__":
    execute_ultimate_pipeline()
