import numpy as np
import matplotlib.pyplot as plt
from qutip import Qobj, basis
from qutip.solver.heom import HEOMSolver, DrudeLorentzBath
import time

print("실제 3D 거리를 반영한 N=7 HEOM 시뮬레이션 테스트...")

# 1. 트립토판 사이의 실제 거리 설정 (단위: 옹스트롬 Å)
# Kalra et al. (2022) Table S2 데이터를 참고한 7개 노드 간의 대표적 거리 간격
# 예: 14.0Å (같은 모노머 내부), 27.0Å (알파-베타 모노머 사이 경계) 등
distances = [14.0, 16.0, 27.0, 15.0, 20.0, 14.0]

# 7x7 거리 행렬(R_matrix) 생성
N = 7
R_matrix = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        if i == j:
            R_matrix[i, j] = 1e-10  # 자기 자신과의 거리는 0으로 인한 오류 방지
        else:
            # i번째와 j번째 트립토판 사이의 누적 직선 거리 계산
            dist = sum(distances[min(i,j):max(i,j)])
            R_matrix[i, j] = dist

# 2. 쌍극자 상호작용 공식을 이용한 해밀토니안(H_matrix) 변환
E_site = 35700.0
V0 = 8.0     # 14.0 Å 기준 결합 에너지
R0 = 14.0    # 기준 거리

H_matrix = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        if i == j:
            H_matrix[i, j] = E_site
        else:
            # 거리의 세제곱에 반비례하는 결합 에너지 할당 (V = V0 * (R0/R)^3)
            H_matrix[i, j] = V0 * (R0 / R_matrix[i, j])**3

H_sys = Qobj(H_matrix)
print("3D 거리 기반 H_matrix 완성!")

# 3. 환경 파라미터 및 HEOM 셋업 (테스트용 단일 세트)
psi0 = basis(N, 0)
rho0 = psi0 * psi0.dag()
T = 215.0
lam = 50.0
gamma = 100.0
Q = basis(N, 0) * basis(N, 0).dag()

bath = DrudeLorentzBath(Q, lam=lam, gamma=gamma, T=T, Nk=1)
HEOM_solver = HEOMSolver(H_sys, bath, max_depth=3)
tlist = np.linspace(0, 1.0, 100)

# 4. 연산 실행
start_time = time.time()
result = HEOM_solver.run(rho0, tlist)
end_time = time.time()
print(f"연산 완료! 소요 시간: {end_time - start_time:.2f} 초")

# 5. 결과 시각화
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
plt.show()