import numpy as np
import matplotlib.pyplot as plt
from qutip import Qobj, basis
from qutip.solver.heom import HEOMSolver, DrudeLorentzBath

E1, E2 = 35700.0, 35700.0 
V = 8.0
H_sys = Qobj([[E1, V], [V, E2]])

psi0 = basis(2, 0)
rho0 = psi0 * psi0.dag()

T = 215.0 
lam = 50.0  
gamma = 100.0 

Q = Qobj([[1, 0], [0, 0]])
bath = DrudeLorentzBath(Q, lam=lam, gamma=gamma, T=T, Nk=1)

HEOM_solver = HEOMSolver(H_sys, bath, max_depth=3)
tlist = np.linspace(0, 0.5, 200)
result = HEOM_solver.run(rho0, tlist)

P1_heom = [rho[0, 0].real for rho in result.states]
P2_heom = [rho[1, 1].real for rho in result.states]

plt.figure(figsize=(10, 6))
plt.plot(tlist, P1_heom, label="Tryptophan 1 (W103)", color='blue')
plt.plot(tlist, P2_heom, label="Tryptophan 2 (W407)", color='red', linestyle='--')

plt.title("HEOM Dynamics: Non-Markovian Environment (T=310K)")
plt.xlabel("Time")
plt.ylabel("Probability")
plt.legend()
plt.savefig("heom_result.png") 
plt.show()