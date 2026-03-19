"""
=============================================================
KAELION Paper 1b — 5 modelos, N=10,12
=============================================================
Plataforma: Kaggle (CPU ilimitado)
Fecha: 2026-02-26

PARÁMETROS (verificar antes de ejecutar):
  Conjunto B: J=1.0, h=0.5, b=0.5, OBC
  W = Z_0, V = Z_r para r=1,...,N-1
  D_MAX = 5N (uniforme)
  Omega: AMBOS — full (d>=1) y late (d>=D_MAX//2)
  Floquet order: exp(-ih·ΣX) · exp(-ib·ΣZ) · exp(-iJ·ΣZZ)

MODELOS (5, NO 6 — Floquet eliminado = idéntico a mixing):
  1. KI_chaotic:      J=1.0, h=0.5, b=0.5, OBC
  2. KI_mixing:       J=0.1, h=0.5, b=0.0, OBC
  3. KI_dual_unitary: J=h=b=π/4, OBC
  4. Integrable:      exp(-iπ/4·ΣX), sin entangling
  5. SYK:             all-to-all 2-body Pauli, 10 seeds (42-51)

ESTIMACIÓN DE TIEMPO:
  N=10 (dim=1024): ~53 min total (SYK ~41 min, otros ~3 min c/u)
  N=12 (dim=4096): ~14h total (SYK ~10h, otros ~1h c/u)
  → N=12 SYK puede ser demasiado. Considerar correr sin SYK primero.

VERIFICACIÓN: N=10 KI_chaotic Ω_late(r=1) debe ser < 0.16
  (N=8 dio 0.1587 con D_MAX=5N)
=============================================================
"""
import numpy as np
from scipy.linalg import expm
import json
import time
import os

# ============================================================
# CONFIGURACIÓN
# ============================================================
N_LIST = [10, 12]  # Ajustar si N=12 es demasiado lento
SYK_SEEDS = list(range(42, 52))  # 10 seeds
SYK_ENABLED = True  # Poner False si SYK N=12 es inviable

# ============================================================
# PARÁMETROS CONJUNTO B
# ============================================================
print("=" * 70)
print("KAELION Paper 1b — Kaggle: 5 modelos, N=10,12")
print("=" * 70)
print(f"  KI_chaotic:      J=1.0, h=0.5, b=0.5, OBC")
print(f"  KI_mixing:       J=0.1, h=0.5, b=0.0, OBC")
print(f"  KI_dual_unitary: J=h=b=π/4, OBC")
print(f"  Integrable:      exp(-iπ/4·ΣX)")
print(f"  SYK:             10 seeds, all-to-all Pauli")
print(f"  D_MAX = 5N, Omega: full + late")
print(f"  Floquet order: exp(-ih·ΣX) · exp(-ib·ΣZ) · exp(-iJ·ΣZZ)")
print(f"  N_LIST = {N_LIST}")
print(f"  SYK_ENABLED = {SYK_ENABLED}")
print("=" * 70)

# ============================================================
# MATRICES DE PAULI
# ============================================================
I2 = np.eye(2, dtype=complex)
X_mat = np.array([[0, 1], [1, 0]], dtype=complex)
Y_mat = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z_mat = np.array([[1, 0], [0, -1]], dtype=complex)

def tensor_op(op, site, N):
    ops = [I2] * N
    ops[site] = op
    result = ops[0]
    for i in range(1, N):
        result = np.kron(result, ops[i])
    return result

# ============================================================
# CONSTRUCTORES DE MODELOS
# ============================================================

def build_KI_chaotic(N):
    dim = 2**N
    J, h, b = 1.0, 0.5, 0.5
    H_X = sum(h * tensor_op(X_mat, i, N) for i in range(N))
    H_Z = sum(b * tensor_op(Z_mat, i, N) for i in range(N))
    H_ZZ = sum(J * tensor_op(Z_mat, i, N) @ tensor_op(Z_mat, i+1, N) for i in range(N-1))
    U = expm(-1j * H_X) @ expm(-1j * H_Z) @ expm(-1j * H_ZZ)
    return U, {"J": J, "h": h, "b": b, "boundary": "OBC"}

def build_KI_mixing(N):
    dim = 2**N
    J, h, b = 0.1, 0.5, 0.0
    H_X = sum(h * tensor_op(X_mat, i, N) for i in range(N))
    H_Z = sum(b * tensor_op(Z_mat, i, N) for i in range(N))
    H_ZZ = sum(J * tensor_op(Z_mat, i, N) @ tensor_op(Z_mat, i+1, N) for i in range(N-1))
    U = expm(-1j * H_X) @ expm(-1j * H_Z) @ expm(-1j * H_ZZ)
    return U, {"J": J, "h": h, "b": b, "boundary": "OBC"}

def build_KI_dual_unitary(N):
    dim = 2**N
    J = h = b = np.pi / 4
    H_X = sum(h * tensor_op(X_mat, i, N) for i in range(N))
    H_Z = sum(b * tensor_op(Z_mat, i, N) for i in range(N))
    H_ZZ = sum(J * tensor_op(Z_mat, i, N) @ tensor_op(Z_mat, i+1, N) for i in range(N-1))
    U = expm(-1j * H_X) @ expm(-1j * H_Z) @ expm(-1j * H_ZZ)
    return U, {"J": J, "h": h, "b": b, "boundary": "OBC"}

def build_Integrable(N):
    dim = 2**N
    H_X = sum((np.pi / 4) * tensor_op(X_mat, i, N) for i in range(N))
    U = expm(-1j * H_X)
    return U, {"type": "Clifford", "angle": "pi/4", "entangling": False}

def build_SYK(N, seed=42):
    rng = np.random.RandomState(seed)
    dim = 2**N
    H = np.zeros((dim, dim), dtype=complex)
    paulis = [X_mat, Y_mat, Z_mat]
    for i in range(N):
        for j in range(i + 1, N):
            for a in range(3):
                for bb in range(3):
                    coupling = rng.randn() / np.sqrt(N)
                    H += coupling * tensor_op(paulis[a], i, N) @ tensor_op(paulis[bb], j, N)
    H = (H + H.conj().T) / 2
    U = expm(-1j * H)
    return U, {"type": "SYK", "seed": seed}

# ============================================================
# CÓMPUTO DE OTOC
# ============================================================

def compute_otoc_profile(U, N, D_MAX):
    dim = 2**N
    W = tensor_op(Z_mat, 0, N)
    results = {}

    for r in range(1, N):
        V = tensor_op(Z_mat, r, N)
        C_values = np.zeros(D_MAX + 1)
        Ud = np.eye(dim, dtype=complex)

        for d in range(D_MAX + 1):
            if d > 0:
                Ud = U @ Ud
            Wd = Ud.conj().T @ W @ Ud
            product = Wd.conj().T @ V.conj().T @ Wd @ V
            C_values[d] = np.abs(np.trace(product) / dim)

        C0 = C_values[0]
        d_late = D_MAX // 2

        omega_full = np.mean(C_values[1:]) / C0 if C0 > 1e-15 else 1.0
        omega_late = np.mean(C_values[d_late:]) / C0 if C0 > 1e-15 else 1.0
        C_late = C_values[d_late:]
        omega_var = np.std(C_late) / np.mean(C_late) if np.mean(C_late) > 1e-15 else 0.0

        results[r] = {
            "C0": float(C0),
            "omega_full": float(omega_full),
            "omega_late": float(omega_late),
            "omega_var": float(omega_var),
        }
    return results

# ============================================================
# EJECUCIÓN PRINCIPAL
# ============================================================

non_syk_models = {
    "KI_chaotic": build_KI_chaotic,
    "KI_mixing": build_KI_mixing,
    "KI_dual_unitary": build_KI_dual_unitary,
    "Integrable": build_Integrable,
}

all_results = {
    "metadata": {
        "project": "Kaelion Paper 1b",
        "date": "2026-02-26",
        "platform": "Kaggle",
        "protocol": {
            "D_MAX": "5N (uniforme)",
            "boundary": "OBC",
            "W": "Z_0", "V": "Z_r",
            "omega_full": "mean(|C(d)| for d=1,...,D_MAX) / C(0)",
            "omega_late": "mean(|C(d)| for d=D_MAX//2,...,D_MAX) / C(0)",
            "floquet_order": "exp(-ih·ΣX) · exp(-ib·ΣZ) · exp(-iJ·ΣZZ)",
        },
        "notes": [
            "Floquet prethermal ELIMINADO (= KI_mixing con estos params)",
            "KI_dual_unitary Ω=1.000: Pauli Z ciego a dual-unitary scrambling",
            "KI_mixing Ω→1 con D_MAX grande: quasi-periodicidad domina",
            "Tres mecanismos Ω≈1: sin scrambling (Integ), invisible (dual-u), recurrencia (mixing)",
            "KI_mixing: Ω_full < Ω_late para N>=6 (OTOC baja y sube por recurrencia)",
        ]
    },
    "data": {}
}

total_start = time.time()

# --- Non-SYK models ---
for model_name, builder in non_syk_models.items():
    all_results["data"][model_name] = {}

    for N in N_LIST:
        D_MAX = 5 * N
        dim = 2**N
        mem_gb = (dim**2 * 16) / 1e9  # complex128

        print(f"\n{'='*60}")
        print(f"  {model_name}, N={N} (dim={dim}), D_MAX={D_MAX}, mem~{mem_gb:.2f}GB")
        print(f"{'='*60}")

        t0 = time.time()
        U, params = builder(N)
        t_build = time.time() - t0

        err = np.max(np.abs(U.conj().T @ U - np.eye(dim)))
        print(f"  Build: {t_build:.1f}s, unitarity: {err:.2e}")

        t1 = time.time()
        profile = compute_otoc_profile(U, N, D_MAX)
        t_otoc = time.time() - t1
        total = time.time() - t0

        # Print profile
        print(f"  OTOC: {t_otoc:.1f}s")
        print(f"  {'r':>3} {'Ω_full':>8} {'Ω_late':>8} {'Ω_var':>8}")
        for r in sorted(profile.keys()):
            p = profile[r]
            print(f"  {r:3d} {p['omega_full']:8.4f} {p['omega_late']:8.4f} {p['omega_var']:8.4f}")

        ol_f = profile[1]["omega_full"]
        ol_l = profile[1]["omega_late"]
        r_half = N // 2
        oh_l = profile[r_half]["omega_late"]
        of_l = profile[N-1]["omega_late"]

        print(f"  Resumen: Ω_local={ol_l:.4f}, Ω_half={oh_l:.4f}, Ω_far={of_l:.4f}")
        print(f"  Tiempo: {total:.1f}s")

        # Verificación KI_chaotic
        if model_name == "KI_chaotic" and N == 10:
            if ol_l < 0.16:
                print(f"  ✅ Ω_late(r=1)={ol_l:.4f} < 0.16 (esperado)")
            else:
                print(f"  ⚠️ Ω_late(r=1)={ol_l:.4f} >= 0.16 (revisar)")

        all_results["data"][model_name][str(N)] = {
            "N": N, "dim": dim, "D_MAX": D_MAX,
            "parameters": params,
            "omega_profile": {
                str(r): profile[r] for r in sorted(profile.keys())
            },
            "summary": {
                "omega_local_full": ol_f, "omega_local_late": ol_l,
                "omega_half_late": oh_l, "omega_far_late": of_l
            },
            "time_seconds": total
        }

        # Guardar incremental
        with open("paper1b_5models_N10-12_D5N.json", 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"  [guardado incremental]")

# --- SYK ---
if SYK_ENABLED:
    all_results["data"]["SYK"] = {}

    for N in N_LIST:
        D_MAX = 5 * N
        dim = 2**N

        print(f"\n{'='*60}")
        print(f"  SYK (10 seeds), N={N} (dim={dim}), D_MAX={D_MAX}")
        print(f"{'='*60}")

        t0 = time.time()
        seed_profiles = []

        for si, seed in enumerate(SYK_SEEDS):
            ts = time.time()
            U, params = build_SYK(N, seed=seed)
            profile = compute_otoc_profile(U, N, D_MAX)
            seed_profiles.append(profile)
            elapsed = time.time() - ts
            ol = profile[1]["omega_late"]
            print(f"    seed {seed}: Ω_late(r=1)={ol:.4f}, time={elapsed:.1f}s")

        # Average
        avg_profile = {}
        for r in range(1, N):
            omega_fulls = [sp[r]["omega_full"] for sp in seed_profiles]
            omega_lates = [sp[r]["omega_late"] for sp in seed_profiles]
            omega_vars = [sp[r]["omega_var"] for sp in seed_profiles]
            C0s = [sp[r]["C0"] for sp in seed_profiles]

            avg_profile[r] = {
                "omega_full": float(np.mean(omega_fulls)),
                "omega_late": float(np.mean(omega_lates)),
                "omega_var": float(np.mean(omega_vars)),
                "C0": float(np.mean(C0s)),
                "omega_late_std": float(np.std(omega_lates)),
            }

        total = time.time() - t0
        ol_l = avg_profile[1]["omega_late"]
        ol_std = avg_profile[1]["omega_late_std"]
        r_half = N // 2
        oh_l = avg_profile[r_half]["omega_late"]
        of_l = avg_profile[N-1]["omega_late"]

        print(f"  Promedio: Ω_late(r=1)={ol_l:.4f}±{ol_std:.4f}")
        print(f"  Tiempo total: {total:.1f}s ({total/60:.1f} min)")

        all_results["data"]["SYK"][str(N)] = {
            "N": N, "dim": dim, "D_MAX": D_MAX,
            "parameters": {"type": "SYK", "seeds": SYK_SEEDS, "n_seeds": len(SYK_SEEDS)},
            "omega_profile": {
                str(r): avg_profile[r] for r in sorted(avg_profile.keys())
            },
            "summary": {
                "omega_local_late": ol_l, "omega_local_std": ol_std,
                "omega_half_late": oh_l, "omega_far_late": of_l
            },
            "time_seconds": total
        }

        # Guardar incremental
        with open("paper1b_5models_N10-12_D5N.json", 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"  [guardado incremental]")
else:
    print("\n⚠️ SYK DISABLED. Correr por separado si necesario.")

# ============================================================
# RESUMEN FINAL
# ============================================================
total_elapsed = time.time() - total_start

print(f"\n{'='*70}")
print(f"EJECUCIÓN COMPLETADA — {total_elapsed/60:.1f} min total")
print(f"{'='*70}")

print(f"\n{'Modelo':<20} ", end="")
for N in N_LIST:
    print(f"{'N='+str(N):>10}", end="")
print()
print("-" * (20 + 10 * len(N_LIST)))

for model in ["KI_chaotic", "KI_mixing", "KI_dual_unitary", "Integrable"]:
    print(f"{model:<20} ", end="")
    for N in N_LIST:
        N_str = str(N)
        if N_str in all_results["data"].get(model, {}):
            ol = all_results["data"][model][N_str]["summary"]["omega_local_late"]
            print(f"{ol:10.4f}", end="")
        else:
            print(f"{'—':>10}", end="")
    print()

if SYK_ENABLED and "SYK" in all_results["data"]:
    print(f"{'SYK':<20} ", end="")
    for N in N_LIST:
        N_str = str(N)
        if N_str in all_results["data"]["SYK"]:
            ol = all_results["data"]["SYK"][N_str]["summary"]["omega_local_late"]
            std = all_results["data"]["SYK"][N_str]["summary"]["omega_local_std"]
            print(f" {ol:.4f}±{std:.4f}", end="")
        else:
            print(f"{'—':>10}", end="")
    print()

print(f"\n✅ Guardado: paper1b_5models_N10-12_D5N.json")
print(f"\nSIGUIENTE: consolidar con N=4-8 en paper1b_master_data.json")
