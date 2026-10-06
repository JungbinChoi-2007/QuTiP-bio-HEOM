import numpy as np
import matplotlib.pyplot as plt
import itertools
from joblib import Parallel, delayed
from qutip import Qobj, basis
from qutip.solver.heom import HEOMSolver, DrudeLorentzBath
import time

print("N=7 트립토판 네트워크 고해상도 병렬 스윕 시작 (총 400개 조합)...")

# 1. N=7 시스템 고정 파라미터
N = 7
E_site = 35700.0
V_coupling = 8.0

H_matrix = np.zeros((N, N))
for i in range(N):
    H_matrix[i, i] = E_site
    if i < N - 1:
        H_matrix[i, i+1] = V_coupling
        H_matrix[i+1, i] = V_coupling
H_sys = Qobj(H_matrix)

psi0 = basis(N, 0)
rho0 = psi0 * psi0.dag()
T = 215.0
Q = basis(N, 0) * basis(N, 0).dag()

tlist = np.linspace(0, 1.0, 50) # 스윕 속도를 위해 시간 간격 최적화

# 2. 단일 시뮬레이션 함수 (끝단 Site 7의 에너지 전달량 측정)
def run_sweep_N7(lam, gamma):
    try:
        bath = DrudeLorentzBath(Q, lam=lam, gamma=gamma, T=T, Nk=1)
        solver = HEOMSolver(H_sys, bath, max_depth=3)
        result = solver.run(rho0, tlist)

        # t=1.0(최종 시간)에서 7번째 트립토판(Site 7)에 도달한 에너지 확률
        P7_final = result.states[-1][N-1, N-1].real
        return [lam, gamma, P7_final]
    except Exception as e:
        return [lam, gamma, np.nan]

# 3. 20x20 고해상도 격자 설정
lam_range = np.linspace(10, 500, 20)
gamma_range = np.linspace(50, 1000, 20)
param_grid = list(itertools.product(lam_range, gamma_range))

# 4. 병렬 연산 실행 (Colab 자원 풀가동)
start_time = time.time()
results = Parallel(n_jobs=-1, verbose=5)(delayed(run_sweep_N7)(l, g) for l, g in param_grid)
end_time = time.time()
print(f"고해상도 스윕 완료! 소요 시간: {end_time - start_time:.2f} 초")

# 5. 결과 저장 및 최종 히트맵 시각화
results_array = np.array(results)
np.save("N7_final_sweep_results.npy", results_array)

Z = results_array[:, 2].reshape(20, 20).T

plt.figure(figsize=(9, 7))
im = plt.imshow(Z, extent=[10, 500, 50, 1000], origin='lower', aspect='auto', cmap='magma')
plt.colorbar(im, label="Transfer Efficiency (P7 at t=1.0)")
plt.xlabel("Reorganization Energy, $\lambda$ (cm$^{-1}$)")
plt.ylabel("Cutoff Frequency, $\gamma$ (cm$^{-1}$)")
plt.title("N=7 Tryptophan Network: Long-Range Energy Transfer Efficiency")
plt.savefig("N7_final_heatmap.png", dpi=300) # 고화질 논문용 이미지 저장
plt.show()