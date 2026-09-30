import numpy as np
import matplotlib.pyplot as plt
import itertools
import os
from qutip import Qobj, basis
from qutip.solver.heom import HEOMSolver, DrudeLorentzBath
import time
import gc
from google.colab import drive

print("구글 드라이브 마운트 진행 중... (팝업 창에서 권한을 허용해 주세요)")
drive.mount('/content/drive')
save_path_npy = '/content/drive/MyDrive/N7_Ultimate_Sweep_results.npy'
save_path_png = '/content/drive/MyDrive/N7_Ultimate_Sweep_heatmap.png'

print("4대 물리적 한계를 모두 극복한 [최종 논문용 20x20 런] 시작... (중간 저장 및 이어하기 기능 탑재)")

N = 7
E_site_base = 35700.0
V0 = 8.0
R0 = 14.0
T = 300.0 # 논문 기준 생리학적 온도(300K)로 수정

coords = np.array([
    [0.0, 0.0, 0.0],
    [14.0, 0.0, 0.0],
    [25.31, 11.31, 0.0],
    [44.40, 30.40, 0.0],
    [55.01, 41.01, 0.0],
    [69.15, 55.15, 0.0],
    [79.05, 65.05, 0.0]
])

R_matrix = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        if i != j:
            R_matrix[i, j] = np.linalg.norm(coords[i] - coords[j]) # 뺄셈 기호 복구

kappa_factor = np.sqrt(2.0 / 3.0)

H_base = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        if i != j:
            H_base[i, j] = V0 * kappa_factor * (R0 / R_matrix[i, j])**3

psi0 = basis(N, 0)
rho0 = psi0 * psi0.dag()
tlist = np.linspace(0, 1.0, 50)

Q_ops = [basis(N, i) * basis(N, i).dag() for i in range(N)] # 곱셈 기호 복구

def run_ultimate_model(lam, gamma, n_traj=3):
    baths = [DrudeLorentzBath(Q_ops[i], lam=lam, gamma=gamma, T=T, Nk=1) for i in range(N)]
    P7_sum = 0.0

    for traj in range(n_traj):
        H_MC = H_base.copy()
        for i in range(N):
            H_MC[i, i] = E_site_base + np.random.normal(0, 50.0)

        try:
            solver = HEOMSolver(Qobj(H_MC), baths, max_depth=3)
            result = solver.run(rho0, tlist)
            P7_sum += result.states[-1][N-1, N-1].real

            del solver
            del result
            gc.collect()

        except MemoryError:
            return [lam, gamma, np.nan]

    return [lam, gamma, P7_sum / n_traj]

# 20x20 고해상도 그리드
lam_range = np.linspace(10, 500, 20)
gamma_range = np.linspace(50, 1000, 20)
param_grid = list(itertools.product(lam_range, gamma_range))

results = []
start_idx = 0

# ==========================================
# 핵심 업데이트: 이어하기(Resume) 로직
# ==========================================
if os.path.exists(save_path_npy):
    try:
        previous_results = np.load(save_path_npy)
        results = previous_results.tolist()
        start_idx = len(results)
        print(f"\n[알림] 구글 드라이브에서 이전 데이터({start_idx}개)를 발견했습니다. 이어서 시뮬레이션을 재개합니다!\n")
    except:
        print("\n[알림] 기존 저장 파일을 읽을 수 없어 처음부터 시작합니다.\n")

if start_idx < len(param_grid):
    start_time = time.time()
    print(f"총 {len(param_grid)}개 파라미터 조합 중 {start_idx+1}번째부터 순차 연산 시작...")

    # Parallel 대신 for문 사용하여 실시간 세이브 구축
    for idx in range(start_idx, len(param_grid)):
        l, g = param_grid[idx]

        step_start = time.time()
        res = run_ultimate_model(l, g)
        results.append(res)
        step_end = time.time()

        print(f"[{idx+1}/{len(param_grid)}] lam={l:.1f}, gamma={g:.1f} 완료 (소요시간: {step_end - step_start:.1f}초)")

        # 10개 연산마다 구글 드라이브에 안전하게 자동 중간 저장
        if (idx + 1) % 10 == 0:
            np.save(save_path_npy, np.array(results))
            print(f"  => [중간 세이브] 현재까지 {idx+1}개 안전하게 저장되었습니다.")

    end_time = time.time()
    print(f"\n[최종 연산 완료] 추가 소요 시간: {end_time - start_time:.2f} 초")

# ==========================================
# 최종 결과 저장 및 시각화
# ==========================================
results_array = np.array(results)
np.save(save_path_npy, results_array)
print(f"데이터 파일 최종 저장 완료: {save_path_npy}")

if len(results_array) == len(param_grid):
    Z = results_array[:, 2].reshape(20, 20).T

    plt.figure(figsize=(10, 8))
    im = plt.imshow(Z, extent=[10, 500, 50, 1000], origin='lower', aspect='auto', cmap='magma')
    plt.colorbar(im, label="Transfer Efficiency (P7 at t=1.0, 3-Ensemble Avg)")

    plt.xlabel(r"Reorganization Energy, $\lambda$ (cm$^{-1}$)")
    plt.ylabel(r"Cutoff Frequency, $\gamma$ (cm$^{-1}$)")
    plt.title("Ultimate 3D Tryptophan Network: Quantum Coherence Preservation")

    plt.savefig(save_path_png, dpi=300, bbox_inches='tight')
    print(f"히트맵 이미지 저장 완료: {save_path_png}")
    plt.show()
else:
    print("아직 모든 연산이 끝나지 않아 히트맵을 렌더링할 수 없습니다. 다시 실행하여 남은 연산을 마쳐주세요.")