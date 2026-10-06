import numpy as np
from qutip import Qobj, basis
from qutip.solver.heom import HEOMSolver, DrudeLorentzBath
import time

print("수렴성 및 민감도(Convergence & Sensitivity) 테스트 시작...")

# 1. 3D 기반 N=7 시스템 기본 세팅 (논문과 동일)
N = 7
E_site_base = 35700.0
V0 = 8.0
R0 = 14.0
T = 300.0 

# 안정적인 벤치마크 포인트 고정 (약 7.5e-5 효율이 나오는 지점)
lam = 50.0   
gamma = 100.0

# 실제 3D 거리 좌표 (figure_3.py 데이터)
coords = np.array([
    [0.0, 0.0, 0.0],
    [14.0, 0.0, 0.0],
    [25.31, 11.31, 0.0],
    [44.40, 30.40, 0.0],
    [55.01, 41.01, 0.0],
    [69.15, 55.15, 0.0],
    [79.05, 65.05, 0.0]
])

# 거리 행렬 및 기본 해밀토니안 구성
R_matrix = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        if i != j:
            R_matrix[i, j] = np.linalg.norm(coords[i] - coords[j])

kappa_factor = np.sqrt(2.0 / 3.0)
H_base = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        if i != j:
            H_base[i, j] = V0 * kappa_factor * (R0 / R_matrix[i, j])**3

# 2. HEOM 솔버 공통 환경 변수 세팅
psi0 = basis(N, 0)
rho0 = psi0 * psi0.dag()
tlist = np.linspace(0, 1.0, 50)

# 각 사이트(노드)마다 독립적인 환경 노이즈(Bath) 부착
Q_ops = [basis(N, i) * basis(N, i).dag() for i in range(N)]
baths = [DrudeLorentzBath(Q_ops[i], lam=lam, gamma=gamma, T=T, Nk=1) for i in range(N)]

# QuTiP v5 최신 안정화 옵션
opts = {"nsteps": 20000, "method": "bdf", "atol": 1e-10, "rtol": 1e-8}

# 3. 특정 depth와 n에 대해 P7 전송 효율을 계산하는 함수
def evaluate_convergence(depth, n_traj):
    P7_sum = 0.0
    for traj in range(n_traj):
        H_MC = H_base.copy()
        # 정적 무질서(Static disorder) 주입 - 앙상블 평균용
        for i in range(N):
            H_MC[i, i] = E_site_base + np.random.normal(0, 50.0)

        solver = HEOMSolver(Qobj(H_MC), baths, max_depth=depth, options=opts)
        result = solver.run(rho0, tlist)
        P7_sum += result.states[-1][N-1, N-1].real
        
    return P7_sum / n_traj

# 4. 테스트할 조합 목록 (교수님 요청 사항)
depths = [2, 3, 4]
n_trajs = [1, 3, 5]

print(f"Fixed Params: lambda={lam}, gamma={gamma}\n")
print(f"{'Max Depth':<12} | {'Ensemble (n)':<15} | {'P7 Transfer Efficiency':<25} | {'Time (s)'}")
print("-" * 72)

# 5. 조합별 순차 연산 실행
for d in depths:
    for n in n_trajs:
        t0 = time.time()
        # 난수 시드(Seed)를 고정하지 않아 진짜 물리적 노이즈가 반영되게 함
        eff = evaluate_convergence(d, n)
        t1 = time.time()
        # 소수점 출력을 과학적 표기법(Scientific notation)으로 포맷팅
        print(f"{d:<12} | {n:<15} | {eff:<25.4e} | {t1-t0:.2f}")

print("-" * 72)
print("수렴성 테스트 완료.")