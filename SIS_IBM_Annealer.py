
# SIS parameter estimation with DA, QUBO, and QAOA
# IBM simulator/QPU and annealing runs
#
# SIS model
# dS/dt = -beta*S*I/N + gamma*I
# dI/dt =  beta*S*I/N - gamma*I
#
# Nudged infected-state equation
# I-equation is nudged with prevalence I:
# dI/dt = beta*S*I/N - gamma*I + mu_I*(I_obs(t) - I)
#
# Relative prevalence mismatch
# prevalence mismatch:
# J(theta) = sum_i |I_obs(t_i) - I_nudged(t_i)|^2 / sum_i |I_obs(t_i)|^2
#
# Estimated parameters
# beta, gamma
#
# Workflow
# 1. Coarse DA grid
# 2. Local refined box around coarse DA minimum
# 3. Continuous quadratic surrogate on local box
# 4. QUBO on refined local grid
# 5. IBM simulator, IBM QPU, and quantum annealer
#
# Reported methods
# 1. IBM simulator
# 2. IBM QPU
# 3. Quantum annealer
#
# Local QAOA is used to tune the IBM circuit angles.
# It is not treated as a separate result.


# Package setup

import sys
import subprocess
import importlib.util

def install_if_missing(package_import_name, pip_name):
    if importlib.util.find_spec(package_import_name) is None:
        subprocess.check_call([
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            pip_name
        ])

install_if_missing("qiskit", "qiskit")
install_if_missing("qiskit_aer", "qiskit-aer")
install_if_missing("qiskit_ibm_runtime", "qiskit-ibm-runtime")
install_if_missing("scipy", "scipy")
install_if_missing("matplotlib", "matplotlib")
install_if_missing("pandas", "pandas")
install_if_missing("dimod", "dimod")
install_if_missing("neal", "dwave-neal")


# Imports

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from getpass import getpass
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from itertools import product, combinations

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler


plt.rcParams.update({
    "font.size": 18,
    "axes.titlesize": 22,
    "axes.labelsize": 20,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 13,
    "figure.titlesize": 22,
    "lines.linewidth": 2.8,
    "axes.grid": True,
    "grid.alpha": 0.30,
})


# Run options

RUN_IBM_SIMULATOR = True
RUN_IBM_QPU = True
RUN_QUANTUM_ANNEALER = True

IBM_QPU_BACKEND_NAME = "ibm_kingston"


# Problem setup

# State order: S, I
N = 8000.0

S0 = 7990.0
I0 = 10.0

y0_true = np.array([S0, I0], dtype=float)

TRUE_BETA = 0.65
TRUE_GAMMA = 0.25

true_theta = np.array(
    [TRUE_BETA, TRUE_GAMMA],
    dtype=float
)

param_names = [
    r"beta",
    r"gamma",
]

d_theta = len(true_theta)

t_start = 0.0
t_end = 40.0
n_time = 300
t_eval = np.linspace(t_start, t_end, n_time)

# Nudging gain
mu_I = 5.0

# Cost weight
w_I = 1.0

# Coarse search bounds
beta_min_global, beta_max_global = 0.35, 0.95
gamma_min_global, gamma_max_global = 0.10, 0.45

theta_min_global = np.array(
    [beta_min_global, gamma_min_global],
    dtype=float
)

theta_max_global = np.array(
    [beta_max_global, gamma_max_global],
    dtype=float
)

# Grid sizes
M_coarse = 8

# Local box around the coarse minimum
# After the coarse DA grid, the refined box will be centered at the coarse minimum.
# This prevents the global quadratic surrogate from creating a false boundary minimum.
LOCAL_HALF_WIDTH = np.array(
    [0.12, 0.08],
    dtype=float
)

M_fine = 32
bits_per_param = 5
n_bits = d_theta * bits_per_param

# Surrogate/QUBO weight
lambda_weight = 2.0
eps = 1e-14

# QAOA settings
p_depth = 4
num_starts = 3
maxiter_qaoa = 300
shots = 4096

# Annealer settings
NUM_READS = 5000
NUM_SWEEPS = 2000
RANDOM_SEED = 123

rng = np.random.default_rng(RANDOM_SEED)


print("========================================================")
print("SIS DA--QUBO--QAOA parameter estimation")
print("States              : S, I")
print("Nudging observable  : prevalence I(t)")
print("Nudging term        : mu_I*(I_obs(t)-I)")
print("Cost observable     : prevalence I(t)")
print("Cost weight         : w_I =", w_I)
print(f"Population N        = {N}")
print(f"Initial condition   = S0={S0}, I0={I0}")
print(f"True beta           = {TRUE_BETA}")
print(f"True gamma          = {TRUE_GAMMA}")
print("========================================================")


# 4. Basic binary utilities

def int_to_bits(value, n_bits):
    return np.array(
        [(value >> k) & 1 for k in range(n_bits - 1, -1, -1)],
        dtype=int
    )


def bits_to_int(bits):
    value = 0

    for bit in bits:
        value = 2 * value + int(bit)

    return value


def bit_array_to_int(bits):
    return bits_to_int(bits)


def bits_from_int(integer, n_bits):
    return int_to_bits(integer, n_bits)


# 5. SIS model

def sis_rhs(t, y, beta, gamma):
    S, I = y

    infection = beta * S * I / N

    dS = -infection + gamma * I
    dI = infection - gamma * I

    return [dS, dI]


def sis_rhs_nudged_I(
    t,
    y,
    beta,
    gamma,
    I_obs_interp,
    mu_I
):
    S, I = y

    I_obs = float(I_obs_interp(t))

    infection = beta * S * I / N

    dS = -infection + gamma * I
    dI = infection - gamma * I + mu_I * (I_obs - I)

    return [dS, dI]


def solve_sis(beta, gamma, y0, t_array):
    sol = solve_ivp(
        lambda t, y: sis_rhs(
            t,
            y,
            beta,
            gamma
        ),
        (t_array[0], t_array[-1]),
        y0,
        t_eval=t_array,
        rtol=1e-8,
        atol=1e-10,
        method="RK45",
    )

    if not sol.success:
        raise RuntimeError("SIS solve failed.")

    return sol.y.T


def solve_sis_nudged(beta, gamma, y0, t_array, I_obs_interp):
    sol = solve_ivp(
        lambda t, y: sis_rhs_nudged_I(
            t,
            y,
            beta,
            gamma,
            I_obs_interp,
            mu_I,
        ),
        (t_array[0], t_array[-1]),
        y0,
        t_eval=t_array,
        rtol=1e-8,
        atol=1e-10,
        method="RK45",
    )

    if not sol.success:
        raise RuntimeError("Nudged SIS solve failed.")

    return sol.y.T


# 6. Synthetic data

true_traj = solve_sis(
    TRUE_BETA,
    TRUE_GAMMA,
    y0_true,
    t_eval,
)

S_true = true_traj[:, 0]
I_true = true_traj[:, 1]

I_obs_interp = interp1d(
    t_eval,
    I_true,
    kind="linear",
    fill_value="extrapolate",
    bounds_error=False,
)

I_scale = np.sum(I_true ** 2) + eps


# 7. DA cost

def nudged_cost(theta):
    beta, gamma = theta

    if beta <= 0 or gamma <= 0:
        return np.inf

    try:
        traj = solve_sis_nudged(
            beta,
            gamma,
            y0_true,
            t_eval,
            I_obs_interp,
        )
    except Exception:
        return np.inf

    I_nudged = traj[:, 1]

    infected_part = np.sum((I_true - I_nudged) ** 2)

    return float(w_I * infected_part)


# 8. Coarse 8 x 8 DA grid

coarse_levels = [
    np.linspace(theta_min_global[p], theta_max_global[p], M_coarse)
    for p in range(d_theta)
]

coarse_indices = list(product(range(M_coarse), repeat=d_theta))
n_coarse = len(coarse_indices)

theta_train_coarse = np.zeros((n_coarse, d_theta))
cost_train_coarse = np.zeros(n_coarse)

print("\n===== COARSE 8 x 8 DA GRID =====")
print("Nudging with prevalence I(t)")
print("Cost with prevalence I(t)")
print("Number of DA evaluations:", n_coarse)

for row, idx_tuple in enumerate(coarse_indices):
    theta = np.array([
        coarse_levels[p][idx_tuple[p]]
        for p in range(d_theta)
    ])

    theta_train_coarse[row, :] = theta
    cost_train_coarse[row] = nudged_cost(theta)

best_coarse_idx = int(np.argmin(cost_train_coarse))
best_coarse_theta = theta_train_coarse[best_coarse_idx]

print("\n===== COARSE GRID RESULT =====")
print(f"Coarse-grid minimum beta  = {best_coarse_theta[0]:.8f}")
print(f"Coarse-grid minimum gamma = {best_coarse_theta[1]:.6f}")
print(f"True beta                 = {TRUE_BETA:.8f}")
print(f"True gamma                = {TRUE_GAMMA:.6f}")


# 9. Local refined box around coarse minimum

theta_min = np.maximum(
    theta_min_global,
    best_coarse_theta - LOCAL_HALF_WIDTH
)

theta_max = np.minimum(
    theta_max_global,
    best_coarse_theta + LOCAL_HALF_WIDTH
)

print("\n===== LOCAL REFINED PARAMETER BOX =====")
print(f"beta  in [{theta_min[0]:.8f}, {theta_max[0]:.8f}]")
print(f"gamma in [{theta_min[1]:.6f}, {theta_max[1]:.6f}]")

# Training grid for the local quadratic surrogate.
# Use the local refined grid itself as the DA training grid.
# This is still small for SIS: 32 x 32 = 1024 DA evaluations.
fine_levels = [
    np.linspace(theta_min[p], theta_max[p], M_fine)
    for p in range(d_theta)
]

local_indices = list(product(range(M_fine), repeat=d_theta))
n_local = len(local_indices)

theta_train = np.zeros((n_local, d_theta))
cost_train = np.zeros(n_local)

print("\n===== LOCAL 32 x 32 DA GRID FOR SURROGATE =====")
print("Number of local DA evaluations:", n_local)

for row, idx_tuple in enumerate(local_indices):
    if (row + 1) % 100 == 0:
        print(f"Evaluating local DA point {row + 1}/{n_local}")

    theta = np.array([
        fine_levels[p][idx_tuple[p]]
        for p in range(d_theta)
    ])

    theta_train[row, :] = theta
    cost_train[row] = nudged_cost(theta)

best_local_idx = int(np.argmin(cost_train))
best_local_theta = theta_train[best_local_idx]

print("\n===== LOCAL DA GRID RESULT =====")
print(f"Local DA minimum beta  = {best_local_theta[0]:.8f}")
print(f"Local DA minimum gamma = {best_local_theta[1]:.6f}")
print(f"True beta              = {TRUE_BETA:.8f}")
print(f"True gamma             = {TRUE_GAMMA:.6f}")


# 10. Continuous quadratic surrogate fit

def quadratic_features(theta):
    theta = np.asarray(theta, dtype=float)

    feats = [1.0]
    feats.extend(theta)

    for i in range(d_theta):
        for j in range(i, d_theta):
            feats.append(theta[i] * theta[j])

    return np.array(feats, dtype=float)


X_train = np.vstack([
    quadratic_features(theta)
    for theta in theta_train
])

cmin = np.min(cost_train)
cmax = np.max(cost_train)

cost_scaled = (cost_train - cmin) / (cmax - cmin + eps)

weights = np.exp(-lambda_weight * cost_scaled)

A = X_train * np.sqrt(weights)[:, None]
b = cost_scaled * np.sqrt(weights)

surrogate_coeffs, *_ = np.linalg.lstsq(
    A,
    b,
    rcond=None
)

train_pred = X_train @ surrogate_coeffs

weighted_rmse_ls = np.sqrt(
    np.sum(weights * (train_pred - cost_scaled) ** 2)
    / np.sum(weights)
)

e_inf_ls = np.max(np.abs(train_pred - cost_scaled))

print("\n===== CONTINUOUS SURROGATE FIT =====")
print(f"Weighted RMSE_LS = {weighted_rmse_ls:.6e}")
print(f"e_inf_LS         = {e_inf_ls:.6e}")


def surrogate_value(theta):
    return float(
        quadratic_features(theta) @ surrogate_coeffs
    )


# 11. Binary encoding

def theta_from_bits(bits):
    bits = np.asarray(bits, dtype=int)

    theta = np.zeros(d_theta)
    indices = np.zeros(d_theta, dtype=int)

    for p in range(d_theta):
        block = bits[p * bits_per_param:(p + 1) * bits_per_param]
        idx = bits_to_int(block)

        if idx >= M_fine:
            raise ValueError("Decoded index outside refined grid.")

        theta[p] = fine_levels[p][idx]
        indices[p] = idx

    return theta, indices


# 12. Refined surrogate grid and QUBO fit

def qubo_features(bits):
    bits = np.asarray(bits, dtype=int)

    feats = [1.0]
    feats.extend(bits.astype(float))

    for i in range(len(bits)):
        for j in range(i + 1, len(bits)):
            feats.append(float(bits[i] * bits[j]))

    return np.array(feats, dtype=float)


fine_indices = list(product(range(M_fine), repeat=d_theta))
L_fine = len(fine_indices)

fine_theta = np.zeros((L_fine, d_theta))
all_bits = []
all_basis_ints = []

for row, idx_tuple in enumerate(fine_indices):
    theta = np.array([
        fine_levels[p][idx_tuple[p]]
        for p in range(d_theta)
    ])

    fine_theta[row, :] = theta

    bits = np.concatenate([
        int_to_bits(idx_tuple[p], bits_per_param)
        for p in range(d_theta)
    ])

    all_bits.append(bits)
    all_basis_ints.append(bit_array_to_int(bits))

all_bits = np.array(all_bits)
all_basis_ints = np.array(all_basis_ints)

fine_surrogate_costs = np.array([
    surrogate_value(theta)
    for theta in fine_theta
])

fine_surrogate_costs = fine_surrogate_costs - np.min(fine_surrogate_costs)
fine_surrogate_costs = fine_surrogate_costs / (
    np.max(fine_surrogate_costs) + eps
)
fine_surrogate_costs = np.clip(fine_surrogate_costs, 0.0, 1.0)

fine_surrogate_best_idx = int(np.argmin(fine_surrogate_costs))
fine_surrogate_theta = fine_theta[fine_surrogate_best_idx]

print("\n===== REFINED 32 x 32 SURROGATE GRID =====")
print("Fine-grid points searched by surrogate/QUBO:", L_fine)
print(f"Fine surrogate minimum beta  = {fine_surrogate_theta[0]:.8f}")
print(f"Fine surrogate minimum gamma = {fine_surrogate_theta[1]:.6f}")
print("Bits/qubits for refined grid =", n_bits)


Phi = np.vstack([
    qubo_features(bits)
    for bits in all_bits
])

qubo_weights = np.exp(-lambda_weight * fine_surrogate_costs)
Wsqrt = np.sqrt(qubo_weights)

A_qubo = Phi * Wsqrt[:, None]
b_qubo = fine_surrogate_costs * Wsqrt

qubo_coeffs_raw, *_ = np.linalg.lstsq(
    A_qubo,
    b_qubo,
    rcond=None
)

qubo_energies_raw_rows = Phi @ qubo_coeffs_raw

qubo_raw_min = np.min(qubo_energies_raw_rows)
qubo_raw_max = np.max(qubo_energies_raw_rows)
qubo_raw_range = qubo_raw_max - qubo_raw_min + eps

qubo_energies_rows = (qubo_energies_raw_rows - qubo_raw_min) / qubo_raw_range
qubo_energies_rows = np.clip(qubo_energies_rows, 0.0, None)

qubo_coeffs = qubo_coeffs_raw / qubo_raw_range
qubo_coeffs[0] = (qubo_coeffs_raw[0] - qubo_raw_min) / qubo_raw_range

mse_qubo = np.mean(
    (qubo_energies_rows - fine_surrogate_costs) ** 2
)

weighted_mse_qubo = np.average(
    (qubo_energies_rows - fine_surrogate_costs) ** 2,
    weights=qubo_weights
)

N_states = 2 ** n_bits
qubo_energies = np.zeros(N_states)

for row, basis_int in enumerate(all_basis_ints):
    qubo_energies[basis_int] = qubo_energies_rows[row]

qubo_best_row = int(np.argmin(qubo_energies_rows))
qubo_best_bits = all_bits[qubo_best_row]
qubo_theta, qubo_indices = theta_from_bits(qubo_best_bits)

print("\n===== QUBO FIT ON REFINED GRID =====")
print(f"Number of QUBO coefficients: {Phi.shape[1]}")
print(f"QUBO MSE against refined surrogate: {mse_qubo:.6e}")
print(f"QUBO weighted MSE against refined surrogate: {weighted_mse_qubo:.6e}")
print("QUBO minimum bitstring =", "".join(map(str, qubo_best_bits)))
print(f"QUBO minimum beta      = {qubo_theta[0]:.8f}")
print(f"QUBO minimum gamma     = {qubo_theta[1]:.6f}")


# 13. QUBO/BQM utilities

def qubo_coeffs_to_bqm_terms(qubo_coeffs, n_bits):
    constant = qubo_coeffs[0]

    linear = {}
    quadratic = {}

    for i in range(n_bits):
        coeff = qubo_coeffs[1 + i]

        if abs(coeff) > 1e-14:
            linear[i] = coeff

    quad_coeffs = qubo_coeffs[1 + n_bits:]
    pair_list = list(combinations(range(n_bits), 2))

    for coeff, (i, j) in zip(quad_coeffs, pair_list):
        if abs(coeff) > 1e-14:
            quadratic[(i, j)] = coeff

    return constant, linear, quadratic


def qubo_energy_from_coeffs(bits, qubo_coeffs):
    bits = np.asarray(bits).astype(float)

    energy = qubo_coeffs[0]

    energy += np.dot(
        qubo_coeffs[1:1 + n_bits],
        bits
    )

    quad_coeffs = qubo_coeffs[1 + n_bits:]
    pair_list = list(combinations(range(n_bits), 2))

    for coeff, (i, j) in zip(quad_coeffs, pair_list):
        energy += coeff * bits[i] * bits[j]

    return energy


qubo_constant, qubo_linear, qubo_quadratic = qubo_coeffs_to_bqm_terms(
    qubo_coeffs,
    n_bits
)

print("\n===== BQM / QUBO MODEL =====")
print("Constant term:", qubo_constant)
print("Number of linear terms:", len(qubo_linear))
print("Number of quadratic terms:", len(qubo_quadratic))


# 14. Quantum annealer simulator

def solve_qubo_with_neal(linear, quadratic, constant):
    import dimod
    import neal

    bqm = dimod.BinaryQuadraticModel(
        linear,
        quadratic,
        constant,
        dimod.BINARY
    )

    sampler = neal.SimulatedAnnealingSampler()

    sampleset = sampler.sample(
        bqm,
        num_reads=NUM_READS,
        num_sweeps=NUM_SWEEPS,
        seed=RANDOM_SEED
    )

    best_sample = sampleset.first.sample

    best_bits = np.array(
        [best_sample[i] for i in range(n_bits)],
        dtype=int
    )

    best_energy = sampleset.first.energy

    return best_bits, best_energy


def solve_qubo_with_builtin_annealer():
    rng_local = np.random.default_rng(RANDOM_SEED)

    best_bits_global = None
    best_energy_global = np.inf

    for read in range(NUM_READS):
        bits = rng_local.integers(0, 2, size=n_bits)

        current_energy = qubo_energy_from_coeffs(bits, qubo_coeffs)

        T0 = 1.0
        Tf = 1e-4

        for sweep in range(NUM_SWEEPS):
            temperature = T0 * (Tf / T0) ** (
                sweep / max(NUM_SWEEPS - 1, 1)
            )

            q = rng_local.integers(0, n_bits)

            new_bits = bits.copy()
            new_bits[q] = 1 - new_bits[q]

            new_energy = qubo_energy_from_coeffs(new_bits, qubo_coeffs)
            delta = new_energy - current_energy

            if delta < 0:
                bits = new_bits
                current_energy = new_energy
            else:
                accept_probability = np.exp(-delta / max(temperature, 1e-12))

                if rng_local.random() < accept_probability:
                    bits = new_bits
                    current_energy = new_energy

        if current_energy < best_energy_global:
            best_energy_global = current_energy
            best_bits_global = bits.copy()

    return best_bits_global, best_energy_global


annealer_theta = np.full(d_theta, np.nan)
annealer_indices = np.full(d_theta, -1)
annealer_bits = np.zeros(n_bits, dtype=int)

if RUN_QUANTUM_ANNEALER:
    print("\n===== RUNNING QUANTUM ANNEALER SIMULATOR =====")

    try:
        annealer_bits, annealer_qubo_energy = solve_qubo_with_neal(
            qubo_linear,
            qubo_quadratic,
            qubo_constant
        )

        print("Used solver: dwave-neal simulated annealer")

    except Exception as e:
        print("dwave-neal not available or failed.")
        print("Using built-in simulated annealer fallback.")
        print("Reason:", str(e))

        annealer_bits, annealer_qubo_energy = solve_qubo_with_builtin_annealer()

    annealer_theta, annealer_indices = theta_from_bits(annealer_bits)

    print("\n===== QUANTUM ANNEALER RESULT =====")
    print("Annealer bitstring =", "".join(map(str, annealer_bits)))
    print(f"Annealer QUBO energy = {annealer_qubo_energy:.6e}")
    print(f"Decoded beta index  = {annealer_indices[0]}")
    print(f"Decoded gamma index = {annealer_indices[1]}")
    print(f"Annealer beta       = {annealer_theta[0]:.8f}")
    print(f"Annealer gamma      = {annealer_theta[1]:.6f}")


# 15. Local QAOA angle optimization only

def apply_mixer(state, angle, n):
    new_state = state.copy()

    c = np.cos(angle)
    s = -1j * np.sin(angle)

    for q in range(n):
        step = 2 ** q
        block = 2 * step

        updated = new_state.copy()

        for start in range(0, 2 ** n, block):
            for offset in range(step):
                i0 = start + offset
                i1 = i0 + step

                a0 = new_state[i0]
                a1 = new_state[i1]

                updated[i0] = c * a0 + s * a1
                updated[i1] = s * a0 + c * a1

        new_state = updated

    return new_state


def qaoa_state(params, energies, n, depth):
    gammas = params[:depth]
    betas = params[depth:]

    state = np.ones(2 ** n, dtype=complex) / np.sqrt(2 ** n)

    for r in range(depth):
        state = np.exp(-1j * gammas[r] * energies) * state
        state = apply_mixer(state, betas[r], n)

    return state


def qaoa_expectation(params, energies, n, depth):
    state = qaoa_state(params, energies, n, depth)
    probs = np.abs(state) ** 2

    return float(np.sum(probs * energies))


print("\n===== OPTIMIZING QAOA ANGLES FOR IBM CIRCUIT =====")
print(f"QAOA depth p = {p_depth}")
print(f"Number of starts = {num_starts}")
print(f"Max COBYLA iterations per start = {maxiter_qaoa}")
print(f"Shots for simulator/QPU = {shots}")

best_res = None
best_val = np.inf

for seed in range(num_starts):
    rng_local = np.random.default_rng(seed)

    init_gammas = rng_local.uniform(0.0, 2.0 * np.pi, size=p_depth)
    init_betas = rng_local.uniform(0.0, np.pi, size=p_depth)
    x0 = np.concatenate([init_gammas, init_betas])

    res = minimize(
        qaoa_expectation,
        x0,
        args=(qubo_energies, n_bits, p_depth),
        method="COBYLA",
        options={
            "maxiter": maxiter_qaoa,
            "rhobeg": 0.5,
        },
    )

    print(f"Start {seed + 1}/{num_starts}: objective = {res.fun:.6e}")

    if res.fun < best_val:
        best_val = float(res.fun)
        best_res = res

if best_res is None:
    raise RuntimeError("QAOA angle optimization failed.")

opt_angles = best_res.x
opt_gammas = opt_angles[:p_depth]
opt_betas = opt_angles[p_depth:]


# 16. Convert QUBO coefficients to Ising coefficients

def qubo_coeffs_to_ising(qubo_coeffs, n_bits):
    a0 = qubo_coeffs[0]
    linear = qubo_coeffs[1:1 + n_bits]

    pair_list = list(combinations(range(n_bits), 2))
    quad_coeffs = qubo_coeffs[1 + n_bits:]

    h = np.zeros(n_bits)
    J_dict = {}

    const = a0

    for i in range(n_bits):
        const += linear[i] / 2.0
        h[i] += -linear[i] / 2.0

    for coeff, (i, j) in zip(quad_coeffs, pair_list):
        const += coeff / 4.0
        h[i] += -coeff / 4.0
        h[j] += -coeff / 4.0
        J_dict[(i, j)] = coeff / 4.0

    return const, h, J_dict


ising_const, h_ising, J_ising = qubo_coeffs_to_ising(
    qubo_coeffs,
    n_bits
)


# 17. Build QAOA circuit for IBM backends

def build_qaoa_circuit_ibm(h, J_dict, gammas, betas):
    n = len(h)
    p = len(gammas)

    qc = QuantumCircuit(n, n)

    for q in range(n):
        qc.h(q)

    for layer in range(p):
        gamma = gammas[layer]
        beta_angle = betas[layer]

        for i in range(n):
            if abs(h[i]) > 1e-12:
                qc.rz(2.0 * gamma * h[i], i)

        for (i, j), Jij in J_dict.items():
            if abs(Jij) > 1e-12:
                qc.cx(i, j)
                qc.rz(2.0 * gamma * Jij, j)
                qc.cx(i, j)

        for i in range(n):
            qc.rx(2.0 * beta_angle, i)

    # Measurement convention matches MSB-first bit encoding.
    for i in range(n):
        qc.measure(i, n - 1 - i)

    return qc


qc_ibm = build_qaoa_circuit_ibm(
    h_ising,
    J_ising,
    opt_gammas,
    opt_betas
)

print("\n===== QAOA CIRCUIT INFO =====")
print("Number of qubits:", qc_ibm.num_qubits)
print("Circuit depth before transpilation:", qc_ibm.depth())
print("Number of operations before transpilation:", qc_ibm.count_ops())


# 18. IBM utilities

def wait_for_job_completion(job, label, max_polls=180, poll_seconds=30):
    terminal_success_keywords = ["DONE", "COMPLETED"]
    terminal_failure_keywords = ["ERROR", "CANCELLED", "CANCELED", "FAILED"]

    for poll in range(max_polls):
        status = job.status()
        status_text = str(status).upper()

        print(f"{label} poll {poll + 1}/{max_polls}: {status}")

        if any(key in status_text for key in terminal_success_keywords):
            return True

        if any(key in status_text for key in terminal_failure_keywords):
            raise RuntimeError(f"{label} job ended with status: {status}")

        time.sleep(poll_seconds)

    return False


def get_sampler_counts(job_result):
    pub_result = job_result[0]
    data = pub_result.data

    if hasattr(data, "c") and hasattr(data.c, "get_counts"):
        return data.c.get_counts()

    if hasattr(data, "meas") and hasattr(data.meas, "get_counts"):
        return data.meas.get_counts()

    if hasattr(data, "keys"):
        for key in data.keys():
            item = getattr(data, key)

            if hasattr(item, "get_counts"):
                return item.get_counts()

    for key in dir(data):
        if key.startswith("_"):
            continue

        try:
            item = getattr(data, key)
        except Exception:
            continue

        if hasattr(item, "get_counts"):
            return item.get_counts()

    raise RuntimeError("Could not extract counts from SamplerV2 result.")


def bitstring_to_bits(bitstring):
    clean = str(bitstring).replace(" ", "")

    return np.array(
        [int(b) for b in clean],
        dtype=int
    )


def decode_counts_lowest_energy(counts, label):
    best_bitstring = None
    best_energy = np.inf
    best_count = 0

    for bitstring, count in counts.items():
        bits = bitstring_to_bits(bitstring)
        basis_int = bit_array_to_int(bits)
        energy = qubo_energies[basis_int]

        if energy < best_energy:
            best_energy = energy
            best_bitstring = bitstring
            best_count = count

    if best_bitstring is None:
        raise RuntimeError(f"No sampled bitstrings were returned for {label}.")

    best_bits = bitstring_to_bits(best_bitstring)
    theta, indices = theta_from_bits(best_bits)

    print(f"\n===== {label} RESULT =====")
    print(f"Best sampled bitstring      : {best_bitstring}")
    print(f"Best sampled count          : {best_count}")
    print(f"Decoded beta index          : {indices[0]}")
    print(f"Decoded gamma index         : {indices[1]}")
    print(f"{label} beta                : {theta[0]:.8f}")
    print(f"{label} gamma               : {theta[1]:.6f}")

    return {
        "label": label,
        "bitstring": best_bitstring,
        "count": best_count,
        "indices": indices,
        "theta": theta,
    }


sim_theta = np.full(d_theta, np.nan)
sim_indices = np.full(d_theta, -1)
qpu_theta = np.full(d_theta, np.nan)
qpu_indices = np.full(d_theta, -1)
qpu_job_id = None


# 19. IBM simulator and QPU

if RUN_IBM_SIMULATOR or RUN_IBM_QPU:
    if "IBM_QUANTUM_TOKEN" not in os.environ or not os.environ["IBM_QUANTUM_TOKEN"].strip():
        os.environ["IBM_QUANTUM_TOKEN"] = getpass("Enter your IBM Quantum API token: ")

    service = QiskitRuntimeService(
        channel="ibm_quantum_platform",
        token=os.environ["IBM_QUANTUM_TOKEN"]
    )

    try:
        qpu_backend = service.backend(IBM_QPU_BACKEND_NAME)
    except Exception as e:
        raise RuntimeError(
            f"Could not access backend {IBM_QPU_BACKEND_NAME}. "
            "Check that your IBM account has access to this backend."
        ) from e

    print("\n===== IBM BACKEND SELECTED =====")
    print("IBM QPU backend:", qpu_backend.name)

    try:
        print("Backend status:", qpu_backend.status())
    except Exception as e:
        print("Could not read backend status:", e)

    if qc_ibm.num_qubits > qpu_backend.num_qubits:
        raise RuntimeError(
            f"Circuit needs {qc_ibm.num_qubits} qubits, but {qpu_backend.name} "
            f"has only {qpu_backend.num_qubits} qubits."
        )

    if RUN_IBM_SIMULATOR:
        ibm_simulator = AerSimulator.from_backend(qpu_backend)

        sim_pass_manager = generate_preset_pass_manager(
            backend=ibm_simulator,
            optimization_level=3
        )

        qc_ibm_simulator = sim_pass_manager.run(qc_ibm)

        print("\n===== IBM SIMULATOR CIRCUIT INFO =====")
        print("Simulator circuit depth:", qc_ibm_simulator.depth())
        print("Simulator circuit operations:", qc_ibm_simulator.count_ops())

        sim_job = ibm_simulator.run(
            qc_ibm_simulator,
            shots=shots
        )

        print("\n===== IBM SIMULATOR JOB SUBMITTED =====")
        print("Initial status:", sim_job.status())

        sim_result = sim_job.result()
        counts_ibm_simulator = sim_result.get_counts()

        print("\n===== IBM SIMULATOR COUNTS SUMMARY =====")
        print("Total simulator shots:", sum(counts_ibm_simulator.values()))
        print("Number of unique simulator bitstrings:", len(counts_ibm_simulator))

        print("\nTop 20 simulator bitstrings:")

        for bitstring, count in sorted(
            counts_ibm_simulator.items(),
            key=lambda x: x[1],
            reverse=True
        )[:20]:
            print(bitstring, count)

        sim_decoded = decode_counts_lowest_energy(
            counts_ibm_simulator,
            "IBM SIMULATOR"
        )

        sim_theta = sim_decoded["theta"]
        sim_indices = sim_decoded["indices"]

    if RUN_IBM_QPU:
        qpu_pass_manager = generate_preset_pass_manager(
            backend=qpu_backend,
            optimization_level=3
        )

        qc_ibm_qpu = qpu_pass_manager.run(qc_ibm)

        print("\n===== IBM QPU CIRCUIT INFO =====")
        print("QPU circuit depth:", qc_ibm_qpu.depth())
        print("QPU circuit operations:", qc_ibm_qpu.count_ops())

        sampler = Sampler(mode=qpu_backend)

        qpu_job = sampler.run(
            [qc_ibm_qpu],
            shots=shots
        )

        qpu_job_id = qpu_job.job_id()

        print("\n===== IBM QPU JOB SUBMITTED =====")
        print("IBM QPU job ID:", qpu_job_id)
        print("Initial status:", qpu_job.status())

        qpu_completed = wait_for_job_completion(
            qpu_job,
            "IBM QPU",
            max_polls=180,
            poll_seconds=30
        )

        if not qpu_completed:
            raise TimeoutError(
                "IBM QPU job did not finish within the polling limit. "
                f"Job ID: {qpu_job_id}"
            )

        qpu_result = qpu_job.result()
        counts_ibm_qpu = get_sampler_counts(qpu_result)

        print("\n===== IBM QPU COUNTS SUMMARY =====")
        print("Total QPU shots:", sum(counts_ibm_qpu.values()))
        print("Number of unique QPU bitstrings:", len(counts_ibm_qpu))

        print("\nTop 20 QPU bitstrings:")

        for bitstring, count in sorted(
            counts_ibm_qpu.items(),
            key=lambda x: x[1],
            reverse=True
        )[:20]:
            print(bitstring, count)

        qpu_decoded = decode_counts_lowest_energy(
            counts_ibm_qpu,
            "IBM QPU"
        )

        qpu_theta = qpu_decoded["theta"]
        qpu_indices = qpu_decoded["indices"]


# 20. Final parameter table

def relative_percent_error(estimate, truth):
    return 100.0 * abs(estimate - truth) / abs(truth)


parameter_table = pd.DataFrame({
    "Parameter": param_names,

    "True value": true_theta,

    "Coarse DA estimate": best_coarse_theta,

    "Local DA estimate": best_local_theta,

    "QUBO estimate": qubo_theta,

    "QUBO relative error (%)": [
        relative_percent_error(qubo_theta[i], true_theta[i])
        for i in range(d_theta)
    ],

    "IBM simulator estimate": sim_theta,

    "IBM simulator relative error (%)": [
        relative_percent_error(sim_theta[i], true_theta[i])
        if not np.isnan(sim_theta[i]) else np.nan
        for i in range(d_theta)
    ],

    "IBM QPU estimate": qpu_theta,

    "IBM QPU relative error (%)": [
        relative_percent_error(qpu_theta[i], true_theta[i])
        if not np.isnan(qpu_theta[i]) else np.nan
        for i in range(d_theta)
    ],

    "Quantum annealer estimate": annealer_theta,

    "Quantum annealer relative error (%)": [
        relative_percent_error(annealer_theta[i], true_theta[i])
        if not np.isnan(annealer_theta[i]) else np.nan
        for i in range(d_theta)
    ],
})

print("\n===== PARAMETER ESTIMATION TABLE =====")
print(parameter_table.to_string(index=False))


# 21. Trajectory plots

def solve_for_plot(theta):
    beta, gamma = theta

    return solve_sis_nudged(
        beta,
        gamma,
        y0_true,
        t_eval,
        I_obs_interp,
    )


def plot_compartment(
    t,
    true_values,
    sim_values,
    qpu_values,
    annealer_values,
    ylabel,
    title,
    filename
):
    plt.figure(figsize=(10.5, 6.3))

    plt.plot(
        t,
        true_values,
        linestyle="-",
        linewidth=3.5,
        label="True"
    )

    if sim_values is not None:
        plt.plot(
            t,
            sim_values,
            linestyle="--",
            linewidth=2.6,
            marker="o",
            markevery=35,
            markersize=5,
            label="IBM simulator"
        )

    if qpu_values is not None:
        plt.plot(
            t,
            qpu_values,
            linestyle="-.",
            linewidth=2.8,
            marker="s",
            markevery=42,
            markersize=5,
            label="IBM QPU"
        )

    if annealer_values is not None:
        plt.plot(
            t,
            annealer_values,
            linestyle=":",
            linewidth=3.1,
            marker="^",
            markevery=48,
            markersize=5.5,
            label="Quantum annealer"
        )

    plt.xlabel("Time")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(frameon=True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()


sim_traj = None
qpu_traj = None
annealer_traj = None

if RUN_IBM_SIMULATOR and not np.any(np.isnan(sim_theta)):
    sim_traj = solve_for_plot(sim_theta)

if RUN_IBM_QPU and not np.any(np.isnan(qpu_theta)):
    qpu_traj = solve_for_plot(qpu_theta)

if RUN_QUANTUM_ANNEALER and not np.any(np.isnan(annealer_theta)):
    annealer_traj = solve_for_plot(annealer_theta)

plot_compartment(
    t_eval,
    true_traj[:, 0],
    None if sim_traj is None else sim_traj[:, 0],
    None if qpu_traj is None else qpu_traj[:, 0],
    None if annealer_traj is None else annealer_traj[:, 0],
    ylabel=r"$S(t)$",
    title="Susceptible population",
    filename="SIS_S_true_ibm_annealer.png"
)

plot_compartment(
    t_eval,
    true_traj[:, 1],
    None if sim_traj is None else sim_traj[:, 1],
    None if qpu_traj is None else qpu_traj[:, 1],
    None if annealer_traj is None else annealer_traj[:, 1],
    ylabel=r"$I(t)$",
    title="Infected population",
    filename="SIS_I_true_ibm_annealer.png"
)


# 22. Result summary

results_summary = {
    "qpu_job_id": qpu_job_id,
    "param_names": param_names,
    "true_theta": true_theta.tolist(),
    "coarse_da_theta": best_coarse_theta.tolist(),
    "local_da_theta": best_local_theta.tolist(),
    "qubo_theta": qubo_theta.tolist(),
    "sim_theta": sim_theta.tolist(),
    "qpu_theta": qpu_theta.tolist(),
    "annealer_theta": annealer_theta.tolist(),
    "qubo_indices": qubo_indices.tolist(),
    "sim_indices": sim_indices.tolist(),
    "qpu_indices": qpu_indices.tolist(),
    "annealer_indices": annealer_indices.tolist(),
    "p_depth": p_depth,
    "num_starts": num_starts,
    "maxiter_qaoa": maxiter_qaoa,
    "shots": shots,
    "cost_observable": "I(t)",
    "nudged_observable": "I(t)",
    "cost_weights": {
        "w_I": w_I,
    },
}

print("\n===== RESULT SUMMARY =====")
for key, value in results_summary.items():
    print(f"{key}: {value}")

print("\nSaved figures:")
print("SIS_S_true_ibm_annealer.png")
print("SIS_I_true_ibm_annealer.png")


!pip install -q qiskit qiskit-ibm-runtime

import os
from getpass import getpass
from qiskit_ibm_runtime import QiskitRuntimeService

if "IBM_QUANTUM_TOKEN" not in os.environ or not os.environ["IBM_QUANTUM_TOKEN"].strip():
    os.environ["IBM_QUANTUM_TOKEN"] = getpass("Enter your IBM Quantum API token: ")

service = QiskitRuntimeService(
    channel="ibm_quantum_platform",
    token=os.environ["IBM_QUANTUM_TOKEN"]
)

IBM_QPU_JOB_ID = "d91rc6fccmks73d5o19g"

job = service.job(IBM_QPU_JOB_ID)

print("Job ID:", job.job_id())
print("Status:", job.status())

result = job.result()
print("Result retrieved.")


def get_sampler_counts(job_result):
    pub_result = job_result[0]
    data = pub_result.data

    if hasattr(data, "c") and hasattr(data.c, "get_counts"):
        return data.c.get_counts()

    if hasattr(data, "meas") and hasattr(data.meas, "get_counts"):
        return data.meas.get_counts()

    if hasattr(data, "keys"):
        for key in data.keys():
            item = getattr(data, key)
            if hasattr(item, "get_counts"):
                return item.get_counts()

    for key in dir(data):
        if key.startswith("_"):
            continue

        try:
            item = getattr(data, key)
        except Exception:
            continue

        if hasattr(item, "get_counts"):
            return item.get_counts()

    raise RuntimeError("Could not extract counts from SamplerV2 result.")


counts_ibm_qpu = get_sampler_counts(result)

print("Total QPU shots:", sum(counts_ibm_qpu.values()))
print("Number of unique QPU bitstrings:", len(counts_ibm_qpu))

print("\nTop 20 QPU bitstrings:")
for bitstring, count in sorted(
    counts_ibm_qpu.items(),
    key=lambda x: x[1],
    reverse=True
)[:20]:
    print(bitstring, count)


# SIS parameter estimation with DA, QUBO, and QAOA
# IBM simulator/QPU and annealing runs
#
# SIS model
# dS/dt = -beta*S*I/N + gamma*I
# dI/dt =  beta*S*I/N - gamma*I
#
# Nudged infected-state equation
# I-equation is nudged with prevalence I:
# dI/dt = beta*S*I/N - gamma*I + mu_I*(I_obs(t) - I)
#
# Relative prevalence mismatch
# prevalence mismatch:
# J(theta) = sum_i |I_obs(t_i) - I_nudged(t_i)|^2 / sum_i |I_obs(t_i)|^2
#
# Estimated parameters
# beta, gamma
#
# Workflow
# 1. Coarse DA grid
# 2. Local refined box around coarse DA minimum
# 3. Continuous quadratic surrogate on local box
# 4. QUBO on refined local grid
# 5. IBM simulator, IBM QPU, and quantum annealer
#
# Reported methods
# 1. QUBO
# 2. IBM simulator
# 3. IBM QPU
# 4. Quantum annealer
#
# Local QAOA is used to tune the IBM circuit angles.
# It is not treated as a separate result.


# Package setup

import sys
import subprocess
import importlib.util

def install_if_missing(package_import_name, pip_name):
    if importlib.util.find_spec(package_import_name) is None:
        subprocess.check_call([
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            pip_name
        ])

install_if_missing("qiskit", "qiskit")
install_if_missing("qiskit_aer", "qiskit-aer")
install_if_missing("qiskit_ibm_runtime", "qiskit-ibm-runtime")
install_if_missing("scipy", "scipy")
install_if_missing("matplotlib", "matplotlib")
install_if_missing("pandas", "pandas")
install_if_missing("dimod", "dimod")
install_if_missing("neal", "dwave-neal")


# Imports

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from getpass import getpass
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from itertools import product, combinations

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler


plt.rcParams.update({
    "font.size": 18,
    "axes.titlesize": 22,
    "axes.labelsize": 20,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 13,
    "figure.titlesize": 22,
    "lines.linewidth": 2.8,
    "axes.grid": True,
    "grid.alpha": 0.30,
})


# Run options

RUN_IBM_SIMULATOR = True
RUN_IBM_QPU = True
RUN_QUANTUM_ANNEALER = True

IBM_QPU_BACKEND_NAME = "ibm_kingston"


# Problem setup

# State order: S, I
N = 8000.0

S0 = 7990.0
I0 = 10.0

y0_true = np.array([S0, I0], dtype=float)

TRUE_BETA = 0.65
TRUE_GAMMA = 0.25

true_theta = np.array(
    [TRUE_BETA, TRUE_GAMMA],
    dtype=float
)

param_names = [
    r"beta",
    r"gamma",
]

d_theta = len(true_theta)

t_start = 0.0
t_end = 40.0
n_time = 300
t_eval = np.linspace(t_start, t_end, n_time)

# Nudging gain
mu_I = 5.0

# Cost weight
w_I = 1.0

# Coarse search bounds
beta_min_global, beta_max_global = 0.35, 0.95
gamma_min_global, gamma_max_global = 0.10, 0.45

theta_min_global = np.array(
    [beta_min_global, gamma_min_global],
    dtype=float
)

theta_max_global = np.array(
    [beta_max_global, gamma_max_global],
    dtype=float
)

# Grid sizes
M_coarse = 8

# Local box around the coarse minimum
# After the coarse DA grid, the refined box will be centered at the coarse minimum.
# This prevents the global quadratic surrogate from creating a false boundary minimum.
LOCAL_HALF_WIDTH = np.array(
    [0.12, 0.08],
    dtype=float
)

M_fine = 32
bits_per_param = 5
n_bits = d_theta * bits_per_param

# Surrogate/QUBO weight
lambda_weight = 2.0
eps = 1e-14

# QAOA settings
p_depth = 4
num_starts = 3
maxiter_qaoa = 300
shots = 4096

# Annealer settings
NUM_READS = 5000
NUM_SWEEPS = 2000
RANDOM_SEED = 123

rng = np.random.default_rng(RANDOM_SEED)


print("========================================================")
print("SIS DA--QUBO--QAOA parameter estimation")
print("States              : S, I")
print("Nudging observable  : prevalence I(t)")
print("Nudging term        : mu_I*(I_obs(t)-I)")
print("Cost observable     : prevalence I(t)")
print("Cost weight         : w_I =", w_I)
print(f"Population N        = {N}")
print(f"Initial condition   = S0={S0}, I0={I0}")
print(f"True beta           = {TRUE_BETA}")
print(f"True gamma          = {TRUE_GAMMA}")
print("========================================================")


# 4. Basic binary utilities

def int_to_bits(value, n_bits):
    return np.array(
        [(value >> k) & 1 for k in range(n_bits - 1, -1, -1)],
        dtype=int
    )


def bits_to_int(bits):
    value = 0

    for bit in bits:
        value = 2 * value + int(bit)

    return value


def bit_array_to_int(bits):
    return bits_to_int(bits)


def bits_from_int(integer, n_bits):
    return int_to_bits(integer, n_bits)


# 5. SIS model

def sis_rhs(t, y, beta, gamma):
    S, I = y

    infection = beta * S * I / N

    dS = -infection + gamma * I
    dI = infection - gamma * I

    return [dS, dI]


def sis_rhs_nudged_I(
    t,
    y,
    beta,
    gamma,
    I_obs_interp,
    mu_I
):
    S, I = y

    I_obs = float(I_obs_interp(t))

    infection = beta * S * I / N

    dS = -infection + gamma * I
    dI = infection - gamma * I + mu_I * (I_obs - I)

    return [dS, dI]


def solve_sis(beta, gamma, y0, t_array):
    sol = solve_ivp(
        lambda t, y: sis_rhs(
            t,
            y,
            beta,
            gamma
        ),
        (t_array[0], t_array[-1]),
        y0,
        t_eval=t_array,
        rtol=1e-8,
        atol=1e-10,
        method="RK45",
    )

    if not sol.success:
        raise RuntimeError("SIS solve failed.")

    return sol.y.T


def solve_sis_nudged(beta, gamma, y0, t_array, I_obs_interp):
    sol = solve_ivp(
        lambda t, y: sis_rhs_nudged_I(
            t,
            y,
            beta,
            gamma,
            I_obs_interp,
            mu_I,
        ),
        (t_array[0], t_array[-1]),
        y0,
        t_eval=t_array,
        rtol=1e-8,
        atol=1e-10,
        method="RK45",
    )

    if not sol.success:
        raise RuntimeError("Nudged SIS solve failed.")

    return sol.y.T


# 6. Synthetic data

true_traj = solve_sis(
    TRUE_BETA,
    TRUE_GAMMA,
    y0_true,
    t_eval,
)

S_true = true_traj[:, 0]
I_true = true_traj[:, 1]

I_obs_interp = interp1d(
    t_eval,
    I_true,
    kind="linear",
    fill_value="extrapolate",
    bounds_error=False,
)

I_scale = np.sum(I_true ** 2) + eps


# 7. DA cost

def nudged_cost(theta):
    beta, gamma = theta

    if beta <= 0 or gamma <= 0:
        return np.inf

    try:
        traj = solve_sis_nudged(
            beta,
            gamma,
            y0_true,
            t_eval,
            I_obs_interp,
        )
    except Exception:
        return np.inf

    I_nudged = traj[:, 1]

    infected_part = np.sum((I_true - I_nudged) ** 2)

    return float(w_I * infected_part)


# 8. Coarse 8 x 8 DA grid

coarse_levels = [
    np.linspace(theta_min_global[p], theta_max_global[p], M_coarse)
    for p in range(d_theta)
]

coarse_indices = list(product(range(M_coarse), repeat=d_theta))
n_coarse = len(coarse_indices)

theta_train_coarse = np.zeros((n_coarse, d_theta))
cost_train_coarse = np.zeros(n_coarse)

print("\n===== COARSE 8 x 8 DA GRID =====")
print("Nudging with prevalence I(t)")
print("Cost with prevalence I(t)")
print("Number of DA evaluations:", n_coarse)

for row, idx_tuple in enumerate(coarse_indices):
    theta = np.array([
        coarse_levels[p][idx_tuple[p]]
        for p in range(d_theta)
    ])

    theta_train_coarse[row, :] = theta
    cost_train_coarse[row] = nudged_cost(theta)

best_coarse_idx = int(np.argmin(cost_train_coarse))
best_coarse_theta = theta_train_coarse[best_coarse_idx]

print("\n===== COARSE GRID RESULT =====")
print(f"Coarse-grid minimum beta  = {best_coarse_theta[0]:.8f}")
print(f"Coarse-grid minimum gamma = {best_coarse_theta[1]:.6f}")
print(f"True beta                 = {TRUE_BETA:.8f}")
print(f"True gamma                = {TRUE_GAMMA:.6f}")


# 9. Local refined box around coarse minimum

theta_min = np.maximum(
    theta_min_global,
    best_coarse_theta - LOCAL_HALF_WIDTH
)

theta_max = np.minimum(
    theta_max_global,
    best_coarse_theta + LOCAL_HALF_WIDTH
)

print("\n===== LOCAL REFINED PARAMETER BOX =====")
print(f"beta  in [{theta_min[0]:.8f}, {theta_max[0]:.8f}]")
print(f"gamma in [{theta_min[1]:.6f}, {theta_max[1]:.6f}]")

# Training grid for the local quadratic surrogate.
# Use the local refined grid itself as the DA training grid.
# This is still small for SIS: 32 x 32 = 1024 DA evaluations.
fine_levels = [
    np.linspace(theta_min[p], theta_max[p], M_fine)
    for p in range(d_theta)
]

local_indices = list(product(range(M_fine), repeat=d_theta))
n_local = len(local_indices)

theta_train = np.zeros((n_local, d_theta))
cost_train = np.zeros(n_local)

print("\n===== LOCAL 32 x 32 DA GRID FOR SURROGATE =====")
print("Number of local DA evaluations:", n_local)

for row, idx_tuple in enumerate(local_indices):
    if (row + 1) % 100 == 0:
        print(f"Evaluating local DA point {row + 1}/{n_local}")

    theta = np.array([
        fine_levels[p][idx_tuple[p]]
        for p in range(d_theta)
    ])

    theta_train[row, :] = theta
    cost_train[row] = nudged_cost(theta)

best_local_idx = int(np.argmin(cost_train))
best_local_theta = theta_train[best_local_idx]

print("\n===== LOCAL DA GRID RESULT =====")
print(f"Local DA minimum beta  = {best_local_theta[0]:.8f}")
print(f"Local DA minimum gamma = {best_local_theta[1]:.6f}")
print(f"True beta              = {TRUE_BETA:.8f}")
print(f"True gamma             = {TRUE_GAMMA:.6f}")


# 10. Continuous quadratic surrogate fit

def quadratic_features(theta):
    theta = np.asarray(theta, dtype=float)

    feats = [1.0]
    feats.extend(theta)

    for i in range(d_theta):
        for j in range(i, d_theta):
            feats.append(theta[i] * theta[j])

    return np.array(feats, dtype=float)


X_train = np.vstack([
    quadratic_features(theta)
    for theta in theta_train
])

cmin = np.min(cost_train)
cmax = np.max(cost_train)

cost_scaled = (cost_train - cmin) / (cmax - cmin + eps)

weights = np.exp(-lambda_weight * cost_scaled)

A = X_train * np.sqrt(weights)[:, None]
b = cost_scaled * np.sqrt(weights)

surrogate_coeffs, *_ = np.linalg.lstsq(
    A,
    b,
    rcond=None
)

train_pred = X_train @ surrogate_coeffs

weighted_rmse_ls = np.sqrt(
    np.sum(weights * (train_pred - cost_scaled) ** 2)
    / np.sum(weights)
)

e_inf_ls = np.max(np.abs(train_pred - cost_scaled))

print("\n===== CONTINUOUS SURROGATE FIT =====")
print(f"Weighted RMSE_LS = {weighted_rmse_ls:.6e}")
print(f"e_inf_LS         = {e_inf_ls:.6e}")


def surrogate_value(theta):
    return float(
        quadratic_features(theta) @ surrogate_coeffs
    )


# 11. Binary encoding

def theta_from_bits(bits):
    bits = np.asarray(bits, dtype=int)

    theta = np.zeros(d_theta)
    indices = np.zeros(d_theta, dtype=int)

    for p in range(d_theta):
        block = bits[p * bits_per_param:(p + 1) * bits_per_param]
        idx = bits_to_int(block)

        if idx >= M_fine:
            raise ValueError("Decoded index outside refined grid.")

        theta[p] = fine_levels[p][idx]
        indices[p] = idx

    return theta, indices


# 12. Refined surrogate grid and QUBO fit

def qubo_features(bits):
    bits = np.asarray(bits, dtype=int)

    feats = [1.0]
    feats.extend(bits.astype(float))

    for i in range(len(bits)):
        for j in range(i + 1, len(bits)):
            feats.append(float(bits[i] * bits[j]))

    return np.array(feats, dtype=float)


fine_indices = list(product(range(M_fine), repeat=d_theta))
L_fine = len(fine_indices)

fine_theta = np.zeros((L_fine, d_theta))
all_bits = []
all_basis_ints = []

for row, idx_tuple in enumerate(fine_indices):
    theta = np.array([
        fine_levels[p][idx_tuple[p]]
        for p in range(d_theta)
    ])

    fine_theta[row, :] = theta

    bits = np.concatenate([
        int_to_bits(idx_tuple[p], bits_per_param)
        for p in range(d_theta)
    ])

    all_bits.append(bits)
    all_basis_ints.append(bit_array_to_int(bits))

all_bits = np.array(all_bits)
all_basis_ints = np.array(all_basis_ints)

fine_surrogate_costs = np.array([
    surrogate_value(theta)
    for theta in fine_theta
])

fine_surrogate_costs = fine_surrogate_costs - np.min(fine_surrogate_costs)
fine_surrogate_costs = fine_surrogate_costs / (
    np.max(fine_surrogate_costs) + eps
)
fine_surrogate_costs = np.clip(fine_surrogate_costs, 0.0, 1.0)

fine_surrogate_best_idx = int(np.argmin(fine_surrogate_costs))
fine_surrogate_theta = fine_theta[fine_surrogate_best_idx]

print("\n===== REFINED 32 x 32 SURROGATE GRID =====")
print("Fine-grid points searched by surrogate/QUBO:", L_fine)
print(f"Fine surrogate minimum beta  = {fine_surrogate_theta[0]:.8f}")
print(f"Fine surrogate minimum gamma = {fine_surrogate_theta[1]:.6f}")
print("Bits/qubits for refined grid =", n_bits)


Phi = np.vstack([
    qubo_features(bits)
    for bits in all_bits
])

qubo_weights = np.exp(-lambda_weight * fine_surrogate_costs)
Wsqrt = np.sqrt(qubo_weights)

A_qubo = Phi * Wsqrt[:, None]
b_qubo = fine_surrogate_costs * Wsqrt

qubo_coeffs_raw, *_ = np.linalg.lstsq(
    A_qubo,
    b_qubo,
    rcond=None
)

qubo_energies_raw_rows = Phi @ qubo_coeffs_raw

qubo_raw_min = np.min(qubo_energies_raw_rows)
qubo_raw_max = np.max(qubo_energies_raw_rows)
qubo_raw_range = qubo_raw_max - qubo_raw_min + eps

qubo_energies_rows = (qubo_energies_raw_rows - qubo_raw_min) / qubo_raw_range
qubo_energies_rows = np.clip(qubo_energies_rows, 0.0, None)

qubo_coeffs = qubo_coeffs_raw / qubo_raw_range
qubo_coeffs[0] = (qubo_coeffs_raw[0] - qubo_raw_min) / qubo_raw_range

mse_qubo = np.mean(
    (qubo_energies_rows - fine_surrogate_costs) ** 2
)

weighted_mse_qubo = np.average(
    (qubo_energies_rows - fine_surrogate_costs) ** 2,
    weights=qubo_weights
)

N_states = 2 ** n_bits
qubo_energies = np.zeros(N_states)

for row, basis_int in enumerate(all_basis_ints):
    qubo_energies[basis_int] = qubo_energies_rows[row]

qubo_best_row = int(np.argmin(qubo_energies_rows))
qubo_best_bits = all_bits[qubo_best_row]
qubo_theta, qubo_indices = theta_from_bits(qubo_best_bits)

print("\n===== QUBO FIT ON REFINED GRID =====")
print(f"Number of QUBO coefficients: {Phi.shape[1]}")
print(f"QUBO MSE against refined surrogate: {mse_qubo:.6e}")
print(f"QUBO weighted MSE against refined surrogate: {weighted_mse_qubo:.6e}")
print("QUBO minimum bitstring =", "".join(map(str, qubo_best_bits)))
print(f"QUBO minimum beta      = {qubo_theta[0]:.8f}")
print(f"QUBO minimum gamma     = {qubo_theta[1]:.6f}")


# 13. QUBO/BQM utilities

def qubo_coeffs_to_bqm_terms(qubo_coeffs, n_bits):
    constant = qubo_coeffs[0]

    linear = {}
    quadratic = {}

    for i in range(n_bits):
        coeff = qubo_coeffs[1 + i]

        if abs(coeff) > 1e-14:
            linear[i] = coeff

    quad_coeffs = qubo_coeffs[1 + n_bits:]
    pair_list = list(combinations(range(n_bits), 2))

    for coeff, (i, j) in zip(quad_coeffs, pair_list):
        if abs(coeff) > 1e-14:
            quadratic[(i, j)] = coeff

    return constant, linear, quadratic


def qubo_energy_from_coeffs(bits, qubo_coeffs):
    bits = np.asarray(bits).astype(float)

    energy = qubo_coeffs[0]

    energy += np.dot(
        qubo_coeffs[1:1 + n_bits],
        bits
    )

    quad_coeffs = qubo_coeffs[1 + n_bits:]
    pair_list = list(combinations(range(n_bits), 2))

    for coeff, (i, j) in zip(quad_coeffs, pair_list):
        energy += coeff * bits[i] * bits[j]

    return energy


qubo_constant, qubo_linear, qubo_quadratic = qubo_coeffs_to_bqm_terms(
    qubo_coeffs,
    n_bits
)

print("\n===== BQM / QUBO MODEL =====")
print("Constant term:", qubo_constant)
print("Number of linear terms:", len(qubo_linear))
print("Number of quadratic terms:", len(qubo_quadratic))


qpu_decoded = decode_counts_lowest_energy(
    counts_ibm_qpu,
    "IBM QPU"
)

qpu_theta = qpu_decoded["theta"]
qpu_indices = qpu_decoded["indices"]

print("IBM QPU theta:", qpu_theta)
print("IBM QPU indices:", qpu_indices)


def solve_for_plot(theta):
    beta, gamma = theta

    return solve_sis_nudged(
        beta,
        gamma,
        y0_true,
        t_eval,
        I_obs_interp,
    )


def plot_compartment(
    t,
    true_values,
    sim_values,
    qpu_values,
    annealer_values,
    ylabel,
    title,
    filename
):
    plt.figure(figsize=(10.5, 6.3))

    plt.plot(
        t,
        true_values,
        linestyle="-",
        linewidth=3.5,
        label="True"
    )

    if sim_values is not None:
        plt.plot(
            t,
            sim_values,
            linestyle="--",
            linewidth=2.6,
            marker="o",
            markevery=35,
            markersize=5,
            label="IBM simulator"
        )

    if qpu_values is not None:
        plt.plot(
            t,
            qpu_values,
            linestyle="-.",
            linewidth=2.8,
            marker="s",
            markevery=42,
            markersize=5,
            label="IBM QPU"
        )

    if annealer_values is not None:
        plt.plot(
            t,
            annealer_values,
            linestyle=":",
            linewidth=3.1,
            marker="^",
            markevery=48,
            markersize=5.5,
            label="Quantum annealer"
        )

    plt.xlabel("Time")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(frameon=True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()


sim_traj = None
qpu_traj = None
annealer_traj = None

if RUN_IBM_SIMULATOR and not np.any(np.isnan(sim_theta)):
    sim_traj = solve_for_plot(sim_theta)

if RUN_IBM_QPU and not np.any(np.isnan(qpu_theta)):
    qpu_traj = solve_for_plot(qpu_theta)

if RUN_QUANTUM_ANNEALER and not np.any(np.isnan(annealer_theta)):
    annealer_traj = solve_for_plot(annealer_theta)

plot_compartment(
    t_eval,
    true_traj[:, 0],
    None if sim_traj is None else sim_traj[:, 0],
    None if qpu_traj is None else qpu_traj[:, 0],
    None if annealer_traj is None else annealer_traj[:, 0],
    ylabel=r"$S(t)$",
    title="Susceptible population",
    filename="SIS_S_true_ibm_annealer.png"
)

plot_compartment(
    t_eval,
    true_traj[:, 1],
    None if sim_traj is None else sim_traj[:, 1],
    None if qpu_traj is None else qpu_traj[:, 1],
    None if annealer_traj is None else annealer_traj[:, 1],
    ylabel=r"$I(t)$",
    title="Infected population",
    filename="SIS_I_true_ibm_annealer.png"
)


# 22. Result summary

results_summary = {
    "qpu_job_id": qpu_job_id,
    "param_names": param_names,
    "true_theta": true_theta.tolist(),
    "coarse_da_theta": best_coarse_theta.tolist(),
    "local_da_theta": best_local_theta.tolist(),
    "qubo_theta": qubo_theta.tolist(),
    "sim_theta": sim_theta.tolist(),
    "qpu_theta": qpu_theta.tolist(),
    "annealer_theta": annealer_theta.tolist(),
    "qubo_indices": qubo_indices.tolist(),
    "sim_indices": sim_indices.tolist(),
    "qpu_indices": qpu_indices.tolist(),
    "annealer_indices": annealer_indices.tolist(),
    "p_depth": p_depth,
    "num_starts": num_starts,
    "maxiter_qaoa": maxiter_qaoa,
    "shots": shots,
    "cost_observable": "I(t)",
    "nudged_observable": "I(t)",
    "cost_weights": {
        "w_I": w_I,
    },
}

print("\n===== RESULT SUMMARY =====")
for key, value in results_summary.items():
    print(f"{key}: {value}")

print("\nSaved figures:")
print("SIS_S_true_ibm_annealer.png")
print("SIS_I_true_ibm_annealer.png")


# SIS trajectory plot using hard-coded parameter estimates
# S(t) and I(t) on the same graph
# Legend grouped by compartment:
# left column = S(t), right column = I(t)

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Font settings

plt.rcParams.update({
    "font.size": 22,
    "axes.titlesize": 30,
    "axes.labelsize": 28,
    "xtick.labelsize": 24,
    "ytick.labelsize": 24,
    "legend.fontsize": 16,
    "figure.titlesize": 30,
    "mathtext.fontset": "dejavusans",
})

# SIS setup

N = 8000.0
S0 = 7990.0
I0 = 10.0
y0_true = np.array([S0, I0], dtype=float)

t_eval = np.linspace(0, 40, 300)

# Reference parameters
theta_true = np.array([0.65, 0.25], dtype=float)

# Estimated parameters used for this plot
theta_ibm_simulator = np.array([0.66576037, 0.25258065], dtype=float)
theta_ibm_qpu       = np.array([0.66576037, 0.25258065], dtype=float)
theta_annealer      = np.array([0.66576037, 0.25258065], dtype=float)

# SIS model

def sis_rhs(t, y, beta, gamma):
    S, I = y
    dSdt = -beta * S * I / N + gamma * I
    dIdt = beta * S * I / N - gamma * I
    return [dSdt, dIdt]


def solve_sis_for_plot(theta):
    beta, gamma = theta

    sol = solve_ivp(
        lambda t, y: sis_rhs(t, y, beta, gamma),
        (t_eval[0], t_eval[-1]),
        y0_true,
        t_eval=t_eval,
        rtol=1e-9,
        atol=1e-11,
    )

    if not sol.success:
        raise RuntimeError(sol.message)

    return sol.y.T

# Solve the trajectories

true_traj     = solve_sis_for_plot(theta_true)
sim_traj      = solve_sis_for_plot(theta_ibm_simulator)
qpu_traj      = solve_sis_for_plot(theta_ibm_qpu)
annealer_traj = solve_sis_for_plot(theta_annealer)

# Plot results

fig, ax = plt.subplots(figsize=(15, 8.5))

# Reference solution
line_true_S, = ax.plot(
    t_eval,
    true_traj[:, 0],
    color="black",
    linestyle="-",
    linewidth=4.5,
    label=r"True $S(t)$"
)

line_true_I, = ax.plot(
    t_eval,
    true_traj[:, 1],
    color="dimgray",
    linestyle="-",
    linewidth=4.5,
    label=r"True $I(t)$"
)

# IBM simulator
line_sim_S, = ax.plot(
    t_eval,
    sim_traj[:, 0],
    color="tab:blue",
    linestyle="-",
    linewidth=3.2,
    marker="o",
    markevery=35,
    markersize=7,
    label=r"IBM simulator $S(t)$"
)

line_sim_I, = ax.plot(
    t_eval,
    sim_traj[:, 1],
    color="tab:purple",
    linestyle="-",
    linewidth=3.2,
    marker="o",
    markevery=35,
    markersize=7,
    label=r"IBM simulator $I(t)$"
)

# IBM Kingston QPU
line_qpu_S, = ax.plot(
    t_eval,
    qpu_traj[:, 0],
    color="tab:red",
    linestyle=":",
    linewidth=3.8,
    marker="s",
    markevery=42,
    markersize=8,
    label=r"IBM Kingston QPU $S(t)$"
)

line_qpu_I, = ax.plot(
    t_eval,
    qpu_traj[:, 1],
    color="tab:orange",
    linestyle=":",
    linewidth=3.8,
    marker="s",
    markevery=42,
    markersize=8,
    label=r"IBM Kingston QPU $I(t)$"
)

# Annealing result
line_ann_S, = ax.plot(
    t_eval,
    annealer_traj[:, 0],
    color="tab:green",
    linestyle="-.",
    linewidth=3.5,
    marker="^",
    markevery=48,
    markersize=8,
    label=r"Quantum annealer $S(t)$"
)

line_ann_I, = ax.plot(
    t_eval,
    annealer_traj[:, 1],
    color="tab:pink",
    linestyle="-.",
    linewidth=3.5,
    marker="^",
    markevery=48,
    markersize=8,
    label=r"Quantum annealer $I(t)$"
)

# Labels and styling

ax.set_xlabel("Time", fontsize=30)
ax.set_ylabel("Population", fontsize=30)
ax.set_title("SIS trajectory comparison", fontsize=32)

ax.tick_params(axis="both", labelsize=25)
ax.grid(True, alpha=0.30)

# Legend:
# left column = all S(t)
# right column = all I(t)

handles = [
    line_true_S,
    line_sim_S,
    line_qpu_S,
    line_ann_S,
    line_true_I,
    line_sim_I,
    line_qpu_I,
    line_ann_I,
]

labels = [
    r"True $S(t)$",
    r"IBM simulator $S(t)$",
    r"IBM Kingston QPU $S(t)$",
    r"Quantum annealer $S(t)$",
    r"True $I(t)$",
    r"IBM simulator $I(t)$",
    r"IBM Kingston QPU $I(t)$",
    r"Quantum annealer $I(t)$",
]

ax.legend(
    handles,
    labels,
    loc="upper right",
    bbox_to_anchor=(1.0, 1.0),
    ncol=2,
    frameon=True,
    fontsize=16,
    handlelength=3.2,
    columnspacing=1.8,
    handletextpad=0.7,
    borderaxespad=0.35,
)

plt.tight_layout()

filename = "SIS_trajectories_true_ibm_annealer.png"
plt.savefig(filename, dpi=300, bbox_inches="tight")
plt.show()

print(f"\nSaved figure: {filename}")


# SIS parameter estimation with DA, QUBO, and QAOA
# IBM simulator, Kingston QPU, and annealing run

import sys
import subprocess
import importlib.util


def install_if_missing(import_name, pip_name):
    if importlib.util.find_spec(import_name) is None:
        subprocess.check_call([
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            pip_name
        ])


install_if_missing("qiskit", "qiskit")
install_if_missing("qiskit_aer", "qiskit-aer")
install_if_missing("qiskit_ibm_runtime", "qiskit-ibm-runtime")
install_if_missing("scipy", "scipy")
install_if_missing("matplotlib", "matplotlib")
install_if_missing("pandas", "pandas")
install_if_missing("dimod", "dimod")
install_if_missing("neal", "dwave-neal")

import os
import time
from getpass import getpass
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from scipy.optimize import minimize

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler


plt.rcParams.update({
    "font.size": 18,
    "axes.titlesize": 22,
    "axes.labelsize": 21,
    "xtick.labelsize": 17,
    "ytick.labelsize": 17,
    "legend.fontsize": 14,
    "figure.titlesize": 22,
    "lines.linewidth": 2.8,
    "axes.grid": True,
    "grid.alpha": 0.30,
})


# Run settings

RUN_IBM_SIMULATOR = True
RUN_IBM_QPU = True
RUN_QUANTUM_ANNEALER = True

IBM_QPU_BACKEND_NAME = "ibm_kingston"

beta_true = 0.65
gamma_true = 0.25

true_parameter = np.array([beta_true, gamma_true], dtype=float)
parameter_names = [r"$\beta$", r"$\gamma$"]

y0_true = np.array([0.99875, 0.00125], dtype=float)
y0_nudge = y0_true.copy()

t_eval = np.linspace(0.0, 40.0, 300)

mu = 0.5

M_coarse = 8
M_fine = 32
bits_per_param = 5
d_theta = 2
n_bits = bits_per_param * d_theta

beta_min, beta_max = 0.55, 0.75
gamma_min, gamma_max = 0.18, 0.32

lambda_weight = 8.0
eps = 1e-14

p_depth = 5
num_starts = 3
maxiter_qaoa = 1000
shots = 4096

NUM_READS = 5000
NUM_SWEEPS = 2000
RANDOM_SEED = 123

rng = np.random.default_rng(RANDOM_SEED)


# SIS model and synthetic data

def sis_rhs(t, y, beta, gamma):
    S, I = y

    dS = -beta * S * I + gamma * I
    dI = beta * S * I - gamma * I

    return [dS, dI]


def solve_sis(beta, gamma, y0, times):
    sol = solve_ivp(
        lambda t, y: sis_rhs(t, y, beta, gamma),
        (times[0], times[-1]),
        y0,
        t_eval=times,
        rtol=1e-10,
        atol=1e-12,
        method="RK45"
    )

    if not sol.success:
        raise RuntimeError("SIS solve failed.")

    return sol.y.T


true_traj = solve_sis(
    beta_true,
    gamma_true,
    y0_true,
    t_eval
)

S_true = true_traj[:, 0]
I_true = true_traj[:, 1]
I_data = I_true.copy()


# Nudged model and DA cost

def nudged_sis_rhs(t, y, beta, gamma, mu, t_data, I_data):
    S, I = y

    I_obs = np.interp(t, t_data, I_data)

    dS = -beta * S * I + gamma * I
    dI = beta * S * I - gamma * I + mu * (I_obs - I)

    return [dS, dI]


def nudged_cost(beta, gamma, mu, times, I_data, y0_nudge):
    if beta <= 0.0 or gamma <= 0.0:
        return np.inf

    sol = solve_ivp(
        lambda t, y: nudged_sis_rhs(
            t,
            y,
            beta,
            gamma,
            mu,
            times,
            I_data
        ),
        (times[0], times[-1]),
        y0_nudge,
        t_eval=times,
        rtol=1e-9,
        atol=1e-11,
        method="RK45"
    )

    if not sol.success:
        return np.inf

    I_nudged = sol.y[1, :]

    return float(np.sum((I_nudged - I_data) ** 2))


# Coarse 8 x 8 DA grid

beta_coarse = np.linspace(beta_min, beta_max, M_coarse)
gamma_coarse = np.linspace(gamma_min, gamma_max, M_coarse)

coarse_points = []
coarse_costs = []

print("\n===== COARSE 8 x 8 DA GRID =====")

for beta in beta_coarse:
    for gamma in gamma_coarse:
        cost = nudged_cost(
            beta,
            gamma,
            mu,
            t_eval,
            I_data,
            y0_nudge
        )

        coarse_points.append((beta, gamma))
        coarse_costs.append(cost)

coarse_points = np.asarray(coarse_points, dtype=float)
coarse_costs = np.asarray(coarse_costs, dtype=float)

coarse_best_idx = int(np.argmin(coarse_costs))
coarse_best_beta, coarse_best_gamma = coarse_points[coarse_best_idx]

print(f"Number of expensive DA cost evaluations: {len(coarse_costs)}")
print(f"Coarse-grid minimum beta  = {coarse_best_beta:.6f}")
print(f"Coarse-grid minimum gamma = {coarse_best_gamma:.6f}")
print(f"Coarse-grid minimum cost  = {coarse_costs[coarse_best_idx]:.6e}")
print(f"True beta                 = {beta_true:.6f}")
print(f"True gamma                = {gamma_true:.6f}")


# Continuous quadratic surrogate

def parameter_features(beta, gamma):
    return np.array([
        1.0,
        beta,
        gamma,
        beta ** 2,
        beta * gamma,
        gamma ** 2
    ], dtype=float)


X_coarse = np.vstack([
    parameter_features(beta, gamma)
    for beta, gamma in coarse_points
])

cost_min = np.min(coarse_costs)
cost_max = np.max(coarse_costs)

coarse_scaled = (
    coarse_costs - cost_min
) / (
    cost_max - cost_min + eps
)

weights = np.exp(-lambda_weight * coarse_scaled)

A = X_coarse * np.sqrt(weights)[:, None]
b = coarse_scaled * np.sqrt(weights)

surrogate_coeffs, *_ = np.linalg.lstsq(
    A,
    b,
    rcond=None
)


def continuous_surrogate(beta, gamma):
    return float(
        parameter_features(beta, gamma) @ surrogate_coeffs
    )


coarse_surrogate_pred = X_coarse @ surrogate_coeffs

rmse_ls = np.sqrt(
    np.sum(
        weights * (coarse_surrogate_pred - coarse_scaled) ** 2
    ) / np.sum(weights)
)

einf_ls = np.max(
    np.abs(coarse_surrogate_pred - coarse_scaled)
)

print("\n===== CONTINUOUS SURROGATE FIT =====")
print(f"Weighted RMSE_LS = {rmse_ls:.6e}")
print(f"e_inf_LS         = {einf_ls:.6e}")


# Refined 32 x 32 surrogate grid

beta_fine = np.linspace(beta_min, beta_max, M_fine)
gamma_fine = np.linspace(gamma_min, gamma_max, M_fine)

fine_points = []
fine_indices = []

for i_beta in range(M_fine):
    for i_gamma in range(M_fine):
        fine_points.append((
            beta_fine[i_beta],
            gamma_fine[i_gamma]
        ))
        fine_indices.append((i_beta, i_gamma))

fine_points = np.asarray(fine_points, dtype=float)

fine_surrogate_costs = np.array([
    continuous_surrogate(beta, gamma)
    for beta, gamma in fine_points
])

fine_surrogate_costs = (
    fine_surrogate_costs - np.min(fine_surrogate_costs)
)

fine_surrogate_costs = fine_surrogate_costs / (
    np.max(fine_surrogate_costs) + eps
)

fine_surrogate_costs = np.clip(
    fine_surrogate_costs,
    0.0,
    1.0
)

fine_surrogate_best_idx = int(
    np.argmin(fine_surrogate_costs)
)

fine_surrogate_best_beta, fine_surrogate_best_gamma = (
    fine_points[fine_surrogate_best_idx]
)

print("\n===== REFINED 32 x 32 SURROGATE GRID =====")
print(f"Fine-grid points searched by surrogate/QUBO: {len(fine_points)}")
print(f"Fine surrogate minimum beta  = {fine_surrogate_best_beta:.6f}")
print(f"Fine surrogate minimum gamma = {fine_surrogate_best_gamma:.6f}")
print(f"Bits/qubits for refined grid = {n_bits}")


# Binary encoding

def int_to_bits(value, n):
    return np.array([
        (value >> k) & 1
        for k in range(n - 1, -1, -1)
    ], dtype=int)


def bits_to_int(bits):
    value = 0

    for bit in bits:
        value = 2 * value + int(bit)

    return value


def encode_indices(i_beta, i_gamma):
    return np.concatenate([
        int_to_bits(i_beta, bits_per_param),
        int_to_bits(i_gamma, bits_per_param)
    ])


def decode_bits(bits):
    bits = np.asarray(bits, dtype=int)

    i_beta = bits_to_int(
        bits[:bits_per_param]
    )

    i_gamma = bits_to_int(
        bits[bits_per_param:]
    )

    if i_beta >= M_fine or i_gamma >= M_fine:
        raise ValueError(
            "Decoded bitstring is outside the refined grid."
        )

    return i_beta, i_gamma


def parameters_from_indices(i_beta, i_gamma):
    return (
        beta_fine[i_beta],
        gamma_fine[i_gamma]
    )


def bit_array_to_int(bits):
    return bits_to_int(bits)


all_bits = []
all_basis_ints = []

for i_beta, i_gamma in fine_indices:
    bits = encode_indices(i_beta, i_gamma)

    all_bits.append(bits)
    all_basis_ints.append(
        bit_array_to_int(bits)
    )

all_bits = np.asarray(all_bits, dtype=int)
all_basis_ints = np.asarray(all_basis_ints, dtype=int)


# QUBO fit

def qubo_features(bits):
    bits = np.asarray(bits, dtype=float)

    features = [1.0]
    features.extend(bits)

    for i, j in combinations(range(len(bits)), 2):
        features.append(bits[i] * bits[j])

    return np.asarray(features, dtype=float)


Phi = np.vstack([
    qubo_features(bits)
    for bits in all_bits
])

qubo_weights = np.exp(
    -lambda_weight * fine_surrogate_costs
)

Wsqrt = np.sqrt(qubo_weights)

A_qubo = Phi * Wsqrt[:, None]
b_qubo = fine_surrogate_costs * Wsqrt

qubo_coeffs_raw, *_ = np.linalg.lstsq(
    A_qubo,
    b_qubo,
    rcond=None
)

qubo_energies_raw = Phi @ qubo_coeffs_raw

qubo_raw_min = np.min(qubo_energies_raw)
qubo_raw_max = np.max(qubo_energies_raw)
qubo_raw_range = qubo_raw_max - qubo_raw_min + eps

qubo_energies_rows = (
    qubo_energies_raw - qubo_raw_min
) / qubo_raw_range

qubo_energies_rows = np.clip(
    qubo_energies_rows,
    0.0,
    None
)

qubo_coeffs = qubo_coeffs_raw / qubo_raw_range
qubo_coeffs[0] = (
    qubo_coeffs_raw[0] - qubo_raw_min
) / qubo_raw_range

mse_qubo = np.mean(
    (qubo_energies_rows - fine_surrogate_costs) ** 2
)

qubo_energies = np.zeros(2 ** n_bits)

for row, basis_int in enumerate(all_basis_ints):
    qubo_energies[basis_int] = qubo_energies_rows[row]

qubo_best_row = int(
    np.argmin(qubo_energies_rows)
)

qubo_best_bits = all_bits[qubo_best_row]

qubo_i_beta, qubo_i_gamma = decode_bits(
    qubo_best_bits
)

qubo_beta, qubo_gamma = parameters_from_indices(
    qubo_i_beta,
    qubo_i_gamma
)

print("\n===== QUBO FIT ON REFINED GRID =====")
print(f"Number of QUBO coefficients: {Phi.shape[1]}")
print(f"QUBO MSE against refined surrogate: {mse_qubo:.6e}")
print(f"QUBO minimum bitstring = {''.join(map(str, qubo_best_bits))}")
print(f"QUBO minimum beta      = {qubo_beta:.6f}")
print(f"QUBO minimum gamma     = {qubo_gamma:.6f}")


# QUBO utilities

def qubo_coeffs_to_bqm_terms(coeffs, number_of_bits):
    constant = float(coeffs[0])
    linear = {}
    quadratic = {}

    for i in range(number_of_bits):
        coefficient = float(coeffs[1 + i])

        if abs(coefficient) > 1e-14:
            linear[i] = coefficient

    pair_list = list(
        combinations(range(number_of_bits), 2)
    )

    quadratic_coeffs = coeffs[
        1 + number_of_bits:
    ]

    for coefficient, pair in zip(
        quadratic_coeffs,
        pair_list
    ):
        coefficient = float(coefficient)

        if abs(coefficient) > 1e-14:
            quadratic[pair] = coefficient

    return constant, linear, quadratic


def qubo_energy_from_coeffs(bits, coeffs):
    bits = np.asarray(bits, dtype=float)

    energy = float(coeffs[0])

    energy += np.dot(
        coeffs[1:1 + n_bits],
        bits
    )

    pair_list = list(
        combinations(range(n_bits), 2)
    )

    quadratic_coeffs = coeffs[
        1 + n_bits:
    ]

    for coefficient, (i, j) in zip(
        quadratic_coeffs,
        pair_list
    ):
        energy += coefficient * bits[i] * bits[j]

    return float(energy)


qubo_constant, qubo_linear, qubo_quadratic = (
    qubo_coeffs_to_bqm_terms(
        qubo_coeffs,
        n_bits
    )
)


# Quantum annealer simulator

def solve_qubo_with_neal(
    linear,
    quadratic,
    constant
):
    import dimod
    import neal

    bqm = dimod.BinaryQuadraticModel(
        linear,
        quadratic,
        constant,
        dimod.BINARY
    )

    sampler = neal.SimulatedAnnealingSampler()

    sampleset = sampler.sample(
        bqm,
        num_reads=NUM_READS,
        num_sweeps=NUM_SWEEPS,
        seed=RANDOM_SEED
    )

    best_sample = sampleset.first.sample

    best_bits = np.array([
        best_sample[i]
        for i in range(n_bits)
    ], dtype=int)

    return (
        best_bits,
        float(sampleset.first.energy)
    )


annealer_result = None

if RUN_QUANTUM_ANNEALER:
    print("\n===== QUANTUM ANNEALER SIMULATOR =====")

    annealer_bits, annealer_energy = (
        solve_qubo_with_neal(
            qubo_linear,
            qubo_quadratic,
            qubo_constant
        )
    )

    annealer_i_beta, annealer_i_gamma = (
        decode_bits(annealer_bits)
    )

    annealer_beta, annealer_gamma = (
        parameters_from_indices(
            annealer_i_beta,
            annealer_i_gamma
        )
    )

    annealer_result = {
        "label": "Quantum annealer",
        "bitstring": "".join(
            map(str, annealer_bits)
        ),
        "energy": annealer_energy,
        "beta": annealer_beta,
        "gamma": annealer_gamma
    }

    print(f"Best bitstring: {annealer_result['bitstring']}")
    print(f"QUBO energy   : {annealer_result['energy']:.6e}")
    print(f"beta          : {annealer_result['beta']:.6f}")
    print(f"gamma         : {annealer_result['gamma']:.6f}")


# Local QAOA angle optimization

def apply_mixer(state, angle, number_of_bits):
    new_state = state.copy()

    c = np.cos(angle)
    s = -1j * np.sin(angle)

    for q in range(number_of_bits):
        step = 2 ** q
        block = 2 * step
        updated = new_state.copy()

        for start in range(
            0,
            2 ** number_of_bits,
            block
        ):
            for offset in range(step):
                i0 = start + offset
                i1 = i0 + step

                a0 = new_state[i0]
                a1 = new_state[i1]

                updated[i0] = c * a0 + s * a1
                updated[i1] = s * a0 + c * a1

        new_state = updated

    return new_state


def qaoa_state(
    parameters,
    energies,
    number_of_bits,
    depth
):
    gammas = parameters[:depth]
    betas = parameters[depth:]

    state = np.ones(
        2 ** number_of_bits,
        dtype=complex
    ) / np.sqrt(2 ** number_of_bits)

    for layer in range(depth):
        state = (
            np.exp(
                -1j * gammas[layer] * energies
            ) * state
        )

        state = apply_mixer(
            state,
            betas[layer],
            number_of_bits
        )

    return state


def qaoa_expectation(
    parameters,
    energies,
    number_of_bits,
    depth
):
    state = qaoa_state(
        parameters,
        energies,
        number_of_bits,
        depth
    )

    probabilities = np.abs(state) ** 2

    return float(
        np.sum(probabilities * energies)
    )


print("\n===== QAOA ANGLE OPTIMIZATION =====")

best_result = None
best_expectation = np.inf

for seed in range(num_starts):
    rng_local = np.random.default_rng(seed)

    initial_gammas = rng_local.uniform(
        0.0,
        2.0 * np.pi,
        size=p_depth
    )

    initial_betas = rng_local.uniform(
        0.0,
        np.pi,
        size=p_depth
    )

    initial_parameters = np.concatenate([
        initial_gammas,
        initial_betas
    ])

    result = minimize(
        qaoa_expectation,
        initial_parameters,
        args=(
            qubo_energies,
            n_bits,
            p_depth
        ),
        method="COBYLA",
        options={
            "maxiter": maxiter_qaoa,
            "rhobeg": 0.5
        }
    )

    print(
        f"Start {seed + 1}/{num_starts}: "
        f"expectation = {result.fun:.6e}"
    )

    if result.fun < best_expectation:
        best_expectation = float(result.fun)
        best_result = result

if best_result is None:
    raise RuntimeError(
        "QAOA angle optimization failed."
    )

optimal_angles = best_result.x
optimal_gammas = optimal_angles[:p_depth]
optimal_betas = optimal_angles[p_depth:]


# Ising conversion and QAOA circuit

def qubo_coeffs_to_ising(coeffs, number_of_bits):
    constant = float(coeffs[0])

    linear = coeffs[
        1:1 + number_of_bits
    ]

    pair_list = list(
        combinations(range(number_of_bits), 2)
    )

    quadratic_coeffs = coeffs[
        1 + number_of_bits:
    ]

    h = np.zeros(number_of_bits)
    J = {}

    for i in range(number_of_bits):
        constant += linear[i] / 2.0
        h[i] += -linear[i] / 2.0

    for coefficient, (i, j) in zip(
        quadratic_coeffs,
        pair_list
    ):
        constant += coefficient / 4.0
        h[i] += -coefficient / 4.0
        h[j] += -coefficient / 4.0
        J[(i, j)] = coefficient / 4.0

    return constant, h, J


ising_constant, h_ising, J_ising = (
    qubo_coeffs_to_ising(
        qubo_coeffs,
        n_bits
    )
)


def build_qaoa_circuit(
    h,
    J,
    gammas,
    betas
):
    number_of_bits = len(h)
    depth = len(gammas)

    circuit = QuantumCircuit(
        number_of_bits,
        number_of_bits
    )

    for q in range(number_of_bits):
        circuit.h(q)

    for layer in range(depth):
        gamma_angle = gammas[layer]
        beta_angle = betas[layer]

        for i in range(number_of_bits):
            if abs(h[i]) > 1e-12:
                circuit.rz(
                    2.0 * gamma_angle * h[i],
                    i
                )

        for (i, j), coefficient in J.items():
            if abs(coefficient) > 1e-12:
                circuit.cx(i, j)
                circuit.rz(
                    2.0
                    * gamma_angle
                    * coefficient,
                    j
                )
                circuit.cx(i, j)

        for i in range(number_of_bits):
            circuit.rx(
                2.0 * beta_angle,
                i
            )

    for i in range(number_of_bits):
        circuit.measure(
            i,
            number_of_bits - 1 - i
        )

    return circuit


qaoa_circuit = build_qaoa_circuit(
    h_ising,
    J_ising,
    optimal_gammas,
    optimal_betas
)


# IBM Quantum connection

ibm_simulator_result = None
ibm_qpu_result = None

if RUN_IBM_SIMULATOR or RUN_IBM_QPU:
    if (
        "IBM_QUANTUM_TOKEN" not in os.environ
        or not os.environ[
            "IBM_QUANTUM_TOKEN"
        ].strip()
    ):
        os.environ[
            "IBM_QUANTUM_TOKEN"
        ] = getpass(
            "Enter your IBM Quantum API token: "
        )

    service = QiskitRuntimeService(
        channel="ibm_quantum_platform",
        token=os.environ["IBM_QUANTUM_TOKEN"]
    )

    qpu_backend = service.backend(
        IBM_QPU_BACKEND_NAME
    )

    print("\n===== IBM BACKEND =====")
    print("Backend:", qpu_backend.name)

    pass_manager = (
        generate_preset_pass_manager(
            backend=qpu_backend,
            optimization_level=3
        )
    )

    optimized_circuit = pass_manager.run(
        qaoa_circuit
    )


def extract_counts(job_result):
    pub_result = job_result[0]
    data = pub_result.data

    if (
        hasattr(data, "c")
        and hasattr(data.c, "get_counts")
    ):
        return data.c.get_counts()

    if (
        hasattr(data, "meas")
        and hasattr(data.meas, "get_counts")
    ):
        return data.meas.get_counts()

    if hasattr(data, "keys"):
        for key in data.keys():
            item = getattr(data, key)

            if hasattr(item, "get_counts"):
                return item.get_counts()

    for key in dir(data):
        if key.startswith("_"):
            continue

        try:
            item = getattr(data, key)
        except Exception:
            continue

        if hasattr(item, "get_counts"):
            return item.get_counts()

    raise RuntimeError(
        "Could not extract counts from SamplerV2 result."
    )


def decode_counts(counts, label):
    best_bitstring = None
    best_energy = np.inf

    for bitstring in counts:
        cleaned = bitstring.replace(" ", "")

        if len(cleaned) != n_bits:
            continue

        bits = np.array(
            [int(bit) for bit in cleaned],
            dtype=int
        )

        energy = qubo_energy_from_coeffs(
            bits,
            qubo_coeffs
        )

        if energy < best_energy:
            best_energy = energy
            best_bitstring = cleaned

    if best_bitstring is None:
        raise RuntimeError(
            f"No valid bitstrings were returned for {label}."
        )

    best_bits = np.array(
        [int(bit) for bit in best_bitstring],
        dtype=int
    )

    i_beta, i_gamma = decode_bits(
        best_bits
    )

    beta, gamma = parameters_from_indices(
        i_beta,
        i_gamma
    )

    result = {
        "label": label,
        "bitstring": best_bitstring,
        "energy": best_energy,
        "beta": beta,
        "gamma": gamma
    }

    print(f"\n===== {label.upper()} RESULT =====")
    print(f"Best bitstring: {best_bitstring}")
    print(f"QUBO energy   : {best_energy:.6e}")
    print(f"beta          : {beta:.6f}")
    print(f"gamma         : {gamma:.6f}")

    return result


# IBM simulator

if RUN_IBM_SIMULATOR:
    ibm_simulator = AerSimulator.from_backend(
        qpu_backend
    )

    simulator_circuit = (
        ibm_simulator.run(
            optimized_circuit,
            shots=shots
        )
    )

    simulator_counts = (
        simulator_circuit.result().get_counts()
    )

    ibm_simulator_result = decode_counts(
        simulator_counts,
        "IBM simulator"
    )


# IBM QPU

if RUN_IBM_QPU:
    sampler = Sampler(
        mode=qpu_backend
    )

    qpu_job = sampler.run(
        [optimized_circuit],
        shots=shots
    )

    print("\nIBM QPU job ID:", qpu_job.job_id())

    qpu_job_result = qpu_job.result()
    qpu_counts = extract_counts(
        qpu_job_result
    )

    ibm_qpu_result = decode_counts(
        qpu_counts,
        "IBM Kingston QPU"
    )


# Parameter table and trajectories

def relative_percent_error(
    estimate,
    truth
):
    return (
        100.0
        * abs(estimate - truth)
        / abs(truth)
    )


results = []

if ibm_simulator_result is not None:
    results.append(ibm_simulator_result)

if ibm_qpu_result is not None:
    results.append(ibm_qpu_result)

if annealer_result is not None:
    results.append(annealer_result)

table_data = {
    "Parameter": parameter_names,
    "True value": [
        beta_true,
        gamma_true
    ]
}

for result in results:
    table_data[
        f"{result['label']} estimate"
    ] = [
        result["beta"],
        result["gamma"]
    ]

    table_data[
        f"{result['label']} relative error (%)"
    ] = [
        relative_percent_error(
            result["beta"],
            beta_true
        ),
        relative_percent_error(
            result["gamma"],
            gamma_true
        )
    ]

parameter_table = pd.DataFrame(
    table_data
)

print("\n===== PARAMETER ESTIMATION TABLE =====")
print(parameter_table.to_string(index=False))


trajectory_results = {}

for result in results:
    trajectory_results[result["label"]] = (
        solve_sis(
            result["beta"],
            result["gamma"],
            y0_true,
            t_eval
        )
    )


# Trajectory plot

plt.figure(figsize=(14, 7))

plt.plot(
    t_eval,
    S_true,
    color="black",
    linewidth=3.2,
    label=r"True $S(t)$"
)

plt.plot(
    t_eval,
    I_true,
    color="dimgray",
    linewidth=3.2,
    label=r"True $I(t)$"
)

plot_settings = {
    "IBM simulator": {
        "S_color": "tab:blue",
        "I_color": "tab:purple",
        "linestyle": "-",
        "marker": "o"
    },
    "IBM Kingston QPU": {
        "S_color": "tab:red",
        "I_color": "tab:orange",
        "linestyle": ":",
        "marker": "s"
    },
    "Quantum annealer": {
        "S_color": "tab:green",
        "I_color": "tab:pink",
        "linestyle": "--",
        "marker": "^"
    }
}

for label, trajectory in trajectory_results.items():
    settings = plot_settings[label]

    plt.plot(
        t_eval,
        trajectory[:, 0],
        color=settings["S_color"],
        linestyle=settings["linestyle"],
        marker=settings["marker"],
        markevery=22,
        markersize=6,
        linewidth=2.8,
        label=rf"{label} $S(t)$"
    )

    plt.plot(
        t_eval,
        trajectory[:, 1],
        color=settings["I_color"],
        linestyle=settings["linestyle"],
        marker=settings["marker"],
        markevery=22,
        markersize=6,
        linewidth=2.8,
        label=rf"{label} $I(t)$"
    )

plt.xlabel("Time")
plt.ylabel("Population fraction")
plt.title("SIS trajectory comparison")

plt.legend(
    ncol=2,
    frameon=True,
    loc="upper right"
)

plt.grid(True, alpha=0.30)
plt.tight_layout()

plt.savefig(
    "SIS.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()



# Final summary

print("\n===== FINAL SUMMARY =====")
print(f"True beta  = {beta_true:.6f}")
print(f"True gamma = {gamma_true:.6f}")

for result in results:
    print(f"\n{result['label']}")
    print(f"beta      = {result['beta']:.6f}")
    print(f"gamma     = {result['gamma']:.6f}")
    print(f"bitstring = {result['bitstring']}")

print("\nFitting diagnostics")
print(f"RMSE_LS   = {rmse_ls:.6e}")
print(f"e_inf_LS  = {einf_ls:.6e}")
print(f"MSE_QUBO  = {mse_qubo:.6e}")

print("\nSaved figures")
print("SIS.png")
print("SIS_cost_landscape.png")


# Trajectory plot

fig, ax = plt.subplots(figsize=(15, 8.5))

line_true_S, = ax.plot(
    t_eval,
    S_true,
    color="black",
    linestyle="-",
    linewidth=4.5,
    label=r"True $S(t)$"
)

line_true_I, = ax.plot(
    t_eval,
    I_true,
    color="dimgray",
    linestyle="-",
    linewidth=4.5,
    label=r"True $I(t)$"
)

line_sim_S, = ax.plot(
    t_eval,
    trajectory_results["IBM simulator"][:, 0],
    color="tab:blue",
    linestyle="-",
    linewidth=3.2,
    marker="o",
    markevery=35,
    markersize=7,
    label=r"IBM simulator $S(t)$"
)

line_sim_I, = ax.plot(
    t_eval,
    trajectory_results["IBM simulator"][:, 1],
    color="tab:purple",
    linestyle="-",
    linewidth=3.2,
    marker="o",
    markevery=35,
    markersize=7,
    label=r"IBM simulator $I(t)$"
)

line_qpu_S, = ax.plot(
    t_eval,
    trajectory_results["IBM Kingston QPU"][:, 0],
    color="tab:red",
    linestyle=":",
    linewidth=3.8,
    marker="s",
    markevery=42,
    markersize=8,
    label=r"IBM Kingston QPU $S(t)$"
)

line_qpu_I, = ax.plot(
    t_eval,
    trajectory_results["IBM Kingston QPU"][:, 1],
    color="tab:orange",
    linestyle=":",
    linewidth=3.8,
    marker="s",
    markevery=42,
    markersize=8,
    label=r"IBM Kingston QPU $I(t)$"
)

line_ann_S, = ax.plot(
    t_eval,
    trajectory_results["Quantum annealer"][:, 0],
    color="tab:green",
    linestyle="-.",
    linewidth=3.5,
    marker="^",
    markevery=48,
    markersize=8,
    label=r"Quantum annealer $S(t)$"
)

line_ann_I, = ax.plot(
    t_eval,
    trajectory_results["Quantum annealer"][:, 1],
    color="tab:pink",
    linestyle="-.",
    linewidth=3.5,
    marker="^",
    markevery=48,
    markersize=8,
    label=r"Quantum annealer $I(t)$"
)

ax.set_xlabel("Time", fontsize=30)
ax.set_ylabel("Population fraction", fontsize=30)
ax.set_title("SIS trajectory comparison", fontsize=32)

ax.tick_params(axis="both", labelsize=25)
ax.grid(True, alpha=0.30)

handles = [
    line_true_S,
    line_sim_S,
    line_qpu_S,
    line_ann_S,
    line_true_I,
    line_sim_I,
    line_qpu_I,
    line_ann_I,
]

labels = [
    r"True $S(t)$",
    r"IBM simulator $S(t)$",
    r"IBM Kingston QPU $S(t)$",
    r"Quantum annealer $S(t)$",
    r"True $I(t)$",
    r"IBM simulator $I(t)$",
    r"IBM Kingston QPU $I(t)$",
    r"Quantum annealer $I(t)$",
]

ax.legend(
    handles,
    labels,
    loc="upper right",
    bbox_to_anchor=(1.0, 1.0),
    ncol=2,
    frameon=True,
    fontsize=16,
    handlelength=3.2,
    columnspacing=1.8,
    handletextpad=0.7,
    borderaxespad=0.35,
)

plt.tight_layout()

filename = "SIS.png"
plt.savefig(
    filename,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(f"\nSaved figure: {filename}")
