import numpy as np
import matplotlib.pyplot as plt
from qutip import Qobj, basis
from qutip.solver.heom import HEOMSolver, DrudeLorentzBath
import time

print("N=7 트립토판 네트워크 HEOM 테스트 시작...")

# 1. N=7 시스템 파라미터 세팅
N = 7
E_site = 35700.0  # 기본 위치 에너지
V_coupling = 8.0  # 인접 트립토판 간의 평균 결합 에너지

# 7x7 영행렬 생성
H_matrix = np.zeros((N, N))

# 대각선 (위치 에너지) 및 인접 요소 (전자 결합) 할당
for i in range(N):
    H_matrix[i, i] = E_site
    if i < N - 1:
        H_matrix[i, i+1] = V_coupling
        H_matrix[i+1, i] = V_coupling

# 추가적인 장거리 결합 (예: 1번과 3번 등)을 논문 수치에 맞춰 나중에 수동으로 튜닝할 수 있습니다.
H_sys = Qobj(H_matrix)
print("N=7 해밀토니안 구축 완료:\n", H_sys)

# 2. 초기 상태: 에너지가 첫 번째 트립토판(index 0)에 100% 존재
psi0 = basis(N, 0)
rho0 = psi0 * psi0.dag()

# 3. Drude-Lorentz 수조 세팅 (단일 환경 노이즈 테스트)
T = 215.0
lam = 50.0
gamma = 100.0

# 첫 번째 트립토판에 환경이 작용한다고 가정 (N=7 크기에 맞춤)
Q = basis(N, 0) * basis(N, 0).dag()
bath = DrudeLorentzBath(Q, lam=lam, gamma=gamma, T=T, Nk=1)

# 4. HEOM 솔버 실행 (max_depth를 3으로 약간 올려서 정확도 확보)
print("HEOM 연산 진행 중... (시간이 다소 소요될 수 있습니다.)")
start_time = time.time()

HEOM_solver = HEOMSolver(H_sys, bath, max_depth=3)
tlist = np.linspace(0, 1.0, 100)
result = HEOM_solver.run(rho0, tlist)

end_time = time.time()
print(f"연산 완료! 소요 시간: {end_time - start_time:.2f} 초")

# 5. 결과 시각화 (1번, 4번, 7번 트립토판의 에너지 확률 변화 관찰)
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
plt.savefig("N7_heom_test.png")
plt.show()