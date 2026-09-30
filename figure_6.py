import numpy as np
import matplotlib.pyplot as plt
from qutip import Qobj, basis, mesolve

# 1. 해밀토니안 세팅 (에너지 및 결합 상수)
E1, E2 = 35700.0, 35750.0
V = 8.0
H_sys = Qobj([[E1, V], 
              [V, E2]])

# 2. 초기 상태 설정: 에너지가 첫 번째 트립토판(W103)에 100% 집중된 상태
psi0 = basis(2, 0) 

# 3. 붕괴 연산자 (Lindblad Collapse Operator)
# 논문의 에너지 손실률(k_Dis) 개념을 단순화하여 반영
gamma = 2.0  # 임의의 결맞음 감쇠(Decoherence) 변수
c_ops = [np.sqrt(gamma) * Qobj([[0, 1], [0, 0]])]

# 4. 시간 배열 설정 (0부터 특정 시간까지 200개 구간으로 쪼개어 관측)
tlist = np.linspace(0, 1.0, 200)

# 5. 관측할 상태 (각 위치에 에너지가 존재할 확률)
P1 = Qobj([[1, 0], [0, 0]])
P2 = Qobj([[0, 0], [0, 1]])

# 6. 마스터 방정식(mesolve)을 통한 시간 진화 풀이
result = mesolve(H_sys, psi0, tlist, c_ops, [P1, P2])

# 7. 결과 시각화 (Matplotlib 그래프)
plt.figure(figsize=(10, 6))
plt.plot(tlist, result.expect[0], label="Tryptophan 1 (W103)", color='blue', linewidth=2)
plt.plot(tlist, result.expect[1], label="Tryptophan 2 (W407)", color='red', linewidth=2, linestyle='--')

plt.title("Quantum Energy Transfer in N=2 Tryptophan Network")
plt.xlabel("Time (Arbitrary Units)")
plt.ylabel("Probability")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()