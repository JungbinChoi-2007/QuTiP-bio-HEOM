import numpy as np
import matplotlib.pyplot as plt
import itertools
from joblib import Parallel, delayed
from qutip import Qobj, basis
from qutip.solver.heom import HEOMSolver, DrudeLorentzBath
import time
import os

print("모듈 로드 완료. 파라미터 세팅 시작...")

# 1. 고정 파라미터 설정 (Tuszynski, 2022 기반)
E1, E2 = 35700.0, 35700.0
V = 8.0
H_sys = Qobj([[E1, V], [V, E2]])
psi0 = basis(2, 0)
rho0 = psi0 * psi0.dag()
T = 215.0
Q = Qobj([[1, 0], [0, 0]])
tlist = np.linspace(0, 0.5, 50) # 스윕 시 시간 해상도를 낮춰 최적화

# 2. 단일 파라미터 시뮬레이션 함수
def run_single_heom(lam, gamma):
    try:
        bath = DrudeLorentzBath(Q, lam=lam, gamma=gamma, T=T, Nk=1)
        solver = HEOMSolver(H_sys, bath, max_depth=2) # 스윕용 얕은 깊이
        result = solver.run(rho0, tlist)

        # 마지막 시간(0.5)에서 첫 번째 트립토판에 에너지가 남아있을 확률 추출
        P1_final = result.states[-1][0, 0].real
        return [lam, gamma, P1_final]
    except Exception as e:
        return [lam, gamma, np.nan]

# 3. 파라미터 스캔 범위 설정 (Coarse-grained 10x10 격자)
lam_range = np.linspace(10, 500, 10)
gamma_range = np.linspace(50, 1000, 10)
param_grid = list(itertools.product(lam_range, gamma_range))

# 4. 병렬 연산 실행 (Colab 자원 풀가동)
print(f"총 {len(param_grid)} 개의 조합 스윕 시작!")
start_time = time.time()

# n_jobs=-1 은 가용 코어 모두 사용
results = Parallel(n_jobs=-1, verbose=5)(delayed(run_single_heom)(l, g) for l, g in param_grid)

end_time = time.time()
print(f"스윕 완료! 소요 시간: {end_time - start_time:.2f} 초")

# 5. 결과 배열 저장 및 기초 히트맵 시각화
results_array = np.array(results)
np.save("heom_sweep_results.npy", results_array)
print("결과 데이터(npy) 저장 완료.")

# 시각화를 위해 데이터 형태 변환 (10x10 격자)
Z = results_array[:, 2].reshape(10, 10).T

plt.figure(figsize=(8, 6))
# origin='lower' 옵션은 y축이 아래에서 위로 증가하도록 설정합니다.
im = plt.imshow(Z, extent=[10, 500, 50, 1000], origin='lower', aspect='auto', cmap='viridis')
plt.colorbar(im, label="Probability (P1 at t=0.5)")
plt.xlabel("Reorganization Energy, $\lambda$ (cm$^{-1}$)")
plt.ylabel("Cutoff Frequency, $\gamma$ (cm$^{-1}$)")
plt.title("HEOM Parameter Sweep: Quantum Coherence Preservation")
plt.savefig("heom_heatmap.png")
plt.show()