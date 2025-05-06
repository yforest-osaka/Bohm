import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft, fftfreq
from tqdm import tqdm

# 基本パラメータ
N = 1024  # 空間分割
L = 10.0  # 空間長
dx = L / N
x = np.linspace(-L/2, L/2, N)
dt = 0.005
steps = 1000

# 波動関数の初期状態（2つのガウシアン）
def psi0(x):
    return (
        np.exp(-(x + 1.5)**2 * 50) * np.exp(1j * 5 * x) +
        np.exp(-(x - 1.5)**2 * 50) * np.exp(-1j * 5 * x)
    )

# 初期波動関数
psi = psi0(x)
psi /= np.sqrt(np.sum(np.abs(psi)**2) * dx)

# 運動量空間（FFT用）
k = fftfreq(N, d=dx) * 2 * np.pi
k2 = k**2

# 初期粒子位置（確率分布に従う）
N_particles = 5000
prob_density = np.abs(psi)**2
cdf = np.cumsum(prob_density)
cdf /= cdf[-1]
particle_x = np.interp(np.random.rand(N_particles), cdf, x)
trajectories = [particle_x.copy()]

# 時間発展と軌道計算
for _ in tqdm(range(steps), desc="Calculating trajectories"):
    # FFTを使った時間発展（自由粒子）
    psi_k = fft(psi)
    psi_k *= np.exp(-1j * 0.5 * k2 * dt)
    psi = ifft(psi_k)
    psi /= np.sqrt(np.sum(np.abs(psi)**2) * dx)

    # 速度場 v = (1/m) * Im[∇ψ / ψ]
    psi_dx = np.gradient(psi, dx)
    v = np.imag(psi_dx / psi)

    # 粒子を速度に従って更新
    v_particles = np.interp(particle_x, x, v)
    particle_x += v_particles * dt
    trajectories.append(particle_x.copy())

# プロット
trajectories = np.array(trajectories)
plt.figure(figsize=(10, 6))
for i in range(N_particles):
    plt.plot(np.linspace(0, steps*dt, steps+1), trajectories[:, i])
plt.xlabel("Time")
plt.ylabel("Position")
plt.title("Bohmian Trajectories through Double-Slit")
plt.grid(True)
plt.show()
