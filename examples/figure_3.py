import numpy as np
import matplotlib.pyplot as plt
from qutip import Qobj, basis, mesolve

# 1. 해밀토니안 세팅
E1, E2 = 35700.0, 35700.0
V = 8.0
H_sys = Qobj([[E1, V], [V, E2]])

# 2. 초기 상태
psi0 = basis(2, 0) 

# 3. 붕괴 연산자 (Lindblad Collapse Operator)
gamma = 2.0
c_ops = [np.sqrt(gamma) * Qobj([[0, 1], [0, 0]])]

# 4. 시간 및 관측 상태
tlist = np.linspace(0, 1.0, 200)
P1 = Qobj([[1, 0], [0, 0]])
P2 = Qobj([[0, 0], [0, 1]])

# 5. 시간 진화 계산
result = mesolve(H_sys, psi0, tlist, c_ops, [P1, P2])

# 6. 결과 저장 및 시각화
plt.figure(figsize=(10, 6))
plt.plot(tlist, result.expect[0], label="Tryptophan 1 (W103)", color='blue')
plt.plot(tlist, result.expect[1], label="Tryptophan 2 (W407)", color='red', linestyle='--')

plt.title("Quantum Energy Transfer: Markovian Baseline")
plt.xlabel("Time")
plt.ylabel("Probability")
plt.legend()
plt.savefig("markov_result.png") # 이미지를 파일로 자동 저장
plt.show()