import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['OMP_NUM_THREADS'] = '1'

import matplotlib

matplotlib.use('Agg')

import torch

torch.set_num_threads(1)

import torch.nn as nn
import torch.optim as optim
import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import time

# ==========================================
# 0. 全局配置 (强制 GPU)
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_SAMPLES = 80
EPOCHS = 500
D_MAX = 2

# ==========================================
# 1. 真实空间耦合噪声引擎
# ==========================================
gamma = 0.15
K0 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, np.sqrt(1 - gamma)]], dtype=complex)
K1 = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, np.sqrt(gamma)], [0, 0, 0, 0]], dtype=complex)
custom_coupled_kraus = [K0, K1]


def apply_coupled_noise(wires):
    for w in wires:
        qml.DepolarizingChannel(0.01, wires=w)
    if len(wires) >= 2:
        for i in range(len(wires) - 1):
            qml.QubitChannel(custom_coupled_kraus, wires=[wires[i], wires[i + 1]])


# ==========================================
# 2. 核心创新：严格分层截断掩码模型
# ==========================================
class TrueHierarchicalModel(nn.Module):
    def __init__(self, num_qubits, d_max=2):
        super().__init__()
        self.n = num_qubits
        self.num_basis = 4 ** num_qubits
        pauli_names = ["".join(p) for p in product(["I", "X", "Y", "Z"], repeat=num_qubits)]

        # 唯一参与优化的增量张量
        self.W_delta_new = nn.Parameter(torch.zeros(self.num_basis, self.num_basis, dtype=torch.float32))

        # 核心掩码逻辑：完全按照你的分层截断思路设计
        self.register_buffer('strict_mask', torch.zeros(self.num_basis, self.num_basis))
        opt_count = 0

        for i, p1 in enumerate(pauli_names):
            for j, p2 in enumerate(pauli_names):
                w1 = sum(1 for c in p1 if c != "I")
                w2 = sum(1 for c in p2 if c != "I")
                d = sum(1 for a, b in zip(p1, p2) if a != b)

                if num_qubits == 2:
                    # 【基础阶段】：构建2-qubit基础噪声模型
                    if (w1 <= 2 or w2 <= 2) and d <= d_max:
                        self.strict_mask[i, j] = 1.0
                        opt_count += 1
                else:
                    # 【递进阶段】：N-qubit时，仅对 weight=N 且 d<=d_max 的高阶项进行优化
                    if (w1 == num_qubits or w2 == num_qubits) and d <= d_max:
                        self.strict_mask[i, j] = 1.0
                        opt_count += 1

        self.register_buffer('Identity', torch.eye(self.num_basis))

        if num_qubits == 2:
            print(f"   [基础阶段] N=2 | 构建基础模型 (放行 w<=2 且 d<={d_max} 的项)")
        else:
            print(f"   [递进阶段] N={num_qubits} | 掩码锁定: 仅优化 weight={num_qubits} 且 d<={d_max} 的高阶项")
        print(f"   [参数缩减] 矩阵总维度: {self.num_basis}x{self.num_basis} | 实际优化参数量: {opt_count}")

    def forward(self, x):
        # 干净纯粹的物理架构：恒等矩阵 + (增量参数 * 严格掩码)
        W_eff = self.Identity + (self.W_delta_new * self.strict_mask)
        return x @ W_eff.T


# ==========================================
# 3. 实验流程 (绕开 PennyLane 崩溃的 GPU 极速版)
# ==========================================
def run_experiment(num_qubits):
    print(f"\n{'=' * 50}\n  测试 N = {num_qubits} (纯净掩码 + GPU直测)\n{'=' * 50}")

    dev_ideal = qml.device("default.qubit", wires=num_qubits)
    dev_noisy = qml.device("default.mixed", wires=num_qubits)
    paulis = ["".join(p) for p in product(["I", "X", "Y", "Z"], repeat=num_qubits)]

    mat_dict = {
        "I": np.eye(2, dtype=complex),
        "X": np.array([[0, 1], [1, 0]], dtype=complex),
        "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
        "Z": np.array([[1, 0], [0, -1]], dtype=complex)
    }
    obs_matrices = []
    for p_str in paulis:
        m = mat_dict[p_str[0]]
        for char in p_str[1:]:
            m = np.kron(m, mat_dict[char])
        obs_matrices.append(m)

    obs_tensor = torch.tensor(np.array(obs_matrices), dtype=torch.complex64).to(device)

    @qml.qnode(dev_ideal)
    def circuit_ideal(p):
        for i in range(num_qubits): qml.RX(p[0, i], wires=i); qml.RY(p[1, i], wires=i)
        for i in range(num_qubits - 1): qml.CNOT(wires=[i, i + 1])
        return qml.state()

    @qml.qnode(dev_noisy)
    def circuit_noisy(p):
        for i in range(num_qubits): qml.RX(p[0, i], wires=i); qml.RY(p[1, i], wires=i)
        for i in range(num_qubits - 1): qml.CNOT(wires=[i, i + 1])
        apply_coupled_noise(range(num_qubits))
        return qml.density_matrix(wires=range(num_qubits))

    print(f" 正在生成极小样本数据集...")
    X_list, Y_list = [], []
    for _ in range(TRAIN_SAMPLES):
        p = np.random.uniform(0, 2 * np.pi, (2, num_qubits))
        psi = torch.tensor(circuit_ideal(p), dtype=torch.complex64).to(device)
        rho = torch.tensor(circuit_noisy(p), dtype=torch.complex64).to(device)

        # GPU 内并行计算所有 Pauli 期望值
        O_psi = torch.einsum('mij,j->mi', obs_tensor, psi)
        exp_ideal = torch.einsum('i,mi->m', psi.conj(), O_psi).real
        exp_noisy = torch.einsum('ij,mji->m', rho, obs_tensor).real

        X_list.append(exp_ideal.cpu().numpy())
        Y_list.append(exp_noisy.cpu().numpy())

    X_train = torch.tensor(np.array(X_list), dtype=torch.float32).to(device)
    Y_train = torch.tensor(np.array(Y_list), dtype=torch.float32).to(device)

    model = TrueHierarchicalModel(num_qubits).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
    loss_fn = nn.MSELoss()
    history = []

    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        loss = loss_fn(model(X_train), Y_train)
        loss.backward()
        optimizer.step()
        scheduler.step()

        history.append(loss.item())
        if (epoch + 1) % 50 == 0:
            print(f"   Epoch {epoch + 1}/{EPOCHS} | MSE: {loss.item():.6e}")

    return history


if __name__ == "__main__":
    qubit_list = [2, 3, 4, 5]
    results = {}

    start_time = time.time()
    for n in qubit_list:
        results[n] = run_experiment(n)

    print(f"\n 总耗时: {time.time() - start_time:.1f} 秒。")


    def smooth_curve(points, factor=0.85):
        smoothed = []
        for point in points:
            if smoothed:
                previous = smoothed[-1]
                smoothed.append(previous * factor + point * (1 - factor))
            else:
                smoothed.append(point)
        return smoothed


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # --- 图 (a)：基础阶段 (N=2) ---
    raw_n2 = results[2]


    smooth_n2 = smooth_curve(raw_n2, factor=0.85)

    # 画一条浅色的真实数据作为背景
    ax1.plot(raw_n2, color='#1f77b4', alpha=0.2, linewidth=1.5)
    # 画深色的平滑主线
    ax1.plot(smooth_n2, color='#1f77b4', linewidth=3, label=r'N=2 (Base Model: $w \leq 2$)')

    ax1.set_yscale('log')
    ax1.set_xlabel('Training Epochs', fontsize=13)
    ax1.set_ylabel('Mean Squared Error (Log Scale)', fontsize=13)
    ax1.set_title('(a) Foundational Model Convergence', fontsize=14, fontweight='bold')
    ax1.grid(True, ls='--', alpha=0.6)
    ax1.legend(loc='upper right', fontsize=11)

    # --- 图 (b)：递进阶段 (N>2) ---
    colors = ['#ff7f0e', '#2ca02c', '#d62728']
    for idx, n in enumerate([3, 4, 5]):
        raw_data = results[n]
        smooth_data = smooth_curve(raw_data, factor=0.8)  # 高阶本来就平滑，微调即可

        ax2.plot(raw_data, color=colors[idx], alpha=0.2, linewidth=1.5)
        ax2.plot(smooth_data, color=colors[idx], linewidth=2.5,
                 label=rf'N={n} (High-Order Only: $w={n}$)')

    ax2.set_yscale('log')
    ax2.set_xlabel('Training Epochs', fontsize=13)
    ax2.set_title(r'(b) High-Order Residual Extraction ($N>2$)', fontsize=14, fontweight='bold')

    ax2.set_ylim(5e-4, 2e-3)
    ax2.grid(True, ls='--', alpha=0.6)
    ax2.legend(loc='upper right', fontsize=11)

    plt.tight_layout()
    plt.savefig('hierarchical_two_stage_smooth.eps', format='eps', bbox_inches='tight')
    plt.savefig('hierarchical_two_stage_smooth.pdf', format='pdf', dpi=300)
    print("已保存至 'hierarchical_two_stage_smooth.eps/pdf'")