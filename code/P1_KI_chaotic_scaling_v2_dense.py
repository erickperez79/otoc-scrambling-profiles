"""
=============================================================
KAELION PROJECT -- KI_chaotic Omega(N) Scaling v2 (DENSE)
=============================================================
PHASE 1: N=4,6,8,10,12 using full matrix approach.
Verified line-by-line against P1_omega_scaling_v2.py (session 19).

Modelo: Kicked Ising chaotic, Conjunto B
  J=1.0, h=0.5, b=0.5, OBC

OTOC: C(d,r) = Tr[W(d)† V W(d) V] / dim  (trace-based)
D_MAX = max(20, 3*N)

Memory requirements:
  N=4:  0.001 GB  |  N=6: 0.001 GB  |  N=8:  0.01 GB
  N=10: 0.13 GB   |  N=12: 4.29 GB   |  N=14: 77 GB (SKIP)

Plataforma: Kaggle CPU / GitHub Actions (7GB RAM)
Fecha: 2026-02-19, sesion 20
=============================================================
"""

import numpy as np
from scipy.linalg import expm
import json, time, os

# ============================================================
# PARAMETERS
# ============================================================

J, h, b = 1.0, 0.5, 0.5           # Conjunto B
LATE_FRACTION = 0.5                 # Match P1v2
N_LIST = [4, 6, 8, 10, 12]         # Dense limit: N=12 max
MAX_MEMORY_GB = 6.0                 # Safety margin for Kaggle/GHA

def get_D_MAX(N):
    """D_MAX scales with N (matches P1v2)."""
    return max(20, 3 * N)

# Reference values from P1v2 Kaggle run (Conjunto B, trace OTOC)
REFERENCE = {
    4:  {"omega_local": 0.2801},
    6:  {"omega_local": 0.2253},
    8:  {"omega_local": 0.2182},
    10: {"omega_local": 0.2170},
}

OUTPUT_FILE = "ki_chaotic_scaling_v2_dense.json"

# ============================================================
# PAULI MATRICES
# ============================================================

I2 = np.eye(2, dtype=complex)
X_mat = np.array([[0, 1], [1, 0]], dtype=complex)
Z_mat = np.array([[1, 0], [0, -1]], dtype=complex)

# ============================================================
# TENSOR PRODUCT
# ============================================================

def tensor_op(op, site, N):
    """Single-site operator on N-qubit system."""
    ops = [I2] * N
    ops[site] = op
    result = ops[0]
    for i in range(1, N):
        result = np.kron(result, ops[i])
    return result

# ============================================================
# KICKED ISING FLOQUET OPERATOR (matches P1v2 exactly)
# ============================================================

def kicked_ising_U(N, J=1.0, h=0.5, b=0.5):
    """
    U = exp(-i h ΣX) · exp(-i b ΣZ) · exp(-i J ΣZ_iZ_{i+1})
    THREE-STEP structure. OBC: i = 0..N-2.
    IDENTICAL to P1v2 kicked_ising_U.
    """
    dim = 2**N
    H_ZZ = np.zeros((dim, dim), dtype=complex)
    for i in range(N - 1):
        H_ZZ += J * tensor_op(Z_mat, i, N) @ tensor_op(Z_mat, i+1, N)
    H_X = np.zeros((dim, dim), dtype=complex)
    for i in range(N):
        H_X += h * tensor_op(X_mat, i, N)
    H_Z = np.zeros((dim, dim), dtype=complex)
    for i in range(N):
        H_Z += b * tensor_op(Z_mat, i, N)
    return expm(-1j * H_X) @ expm(-1j * H_Z) @ expm(-1j * H_ZZ)

# ============================================================
# OTOC COMPUTATION (TRACE-BASED, matches P1v2 exactly)
# ============================================================

def compute_otoc_profile(U, N):
    """
    Compute C(d, r) for W=Z_0, V=Z_r for ALL r=1..N-1.
    C(d,r) = Tr[W(d)† V W(d) V] / dim

    Memory optimization for N=12: compute V on-the-fly
    instead of precomputing all V_ops.
    """
    D_MAX = get_D_MAX(N)
    dim = 2**N
    W = tensor_op(Z_mat, 0, N)

    # For N<=10, precompute V operators (fast, low memory)
    # For N=12, compute on-the-fly to save ~3 GB
    precompute_V = (N <= 10)

    if precompute_V:
        V_ops = {}
        for r in range(1, N):
            V_ops[r] = tensor_op(Z_mat, r, N)

    C_all = {r: [] for r in range(1, N)}

    Ud = np.eye(dim, dtype=complex)

    for d in range(D_MAX + 1):
        if d > 0:
            Ud = U @ Ud
        Wd = Ud.conj().T @ W @ Ud  # W(d) = U^{-d} W U^d

        for r in range(1, N):
            if precompute_V:
                V = V_ops[r]
            else:
                V = tensor_op(Z_mat, r, N)
            product = Wd.conj().T @ V @ Wd @ V
            C = np.trace(product) / dim
            C_all[r].append(C)

        if d % 10 == 0 and d > 0:
            print(f"    depth {d}/{D_MAX}")

    for r in C_all:
        C_all[r] = np.array(C_all[r])

    return C_all

def compute_omega_from_C(C_values, D_MAX):
    """
    Omega = mean(|C|_late) / |C(0)|
    Omega_var = std(|C|_late) / mean(|C|_late)
    IDENTICAL to P1v2.
    """
    C_abs = np.abs(C_values)
    C0 = C_abs[0]

    if C0 < 1e-15:
        return 1.0, 0.0, float(C0)

    n_depths = len(C_values)
    d_sat_idx = int(n_depths * LATE_FRACTION)
    C_late = C_abs[d_sat_idx:]

    mean_late = np.mean(C_late)
    std_late = np.std(C_late)

    omega = mean_late / C0
    omega_var = std_late / mean_late if mean_late > 1e-15 else 0.0

    return float(omega), float(omega_var), float(C0)

# ============================================================
# MAIN COMPUTATION
# ============================================================

def run_single_N(N):
    """Run full Omega(r) profile for a single N."""
    dim = 2**N
    D_MAX = get_D_MAX(N)

    print(f"\n{'='*60}")
    print(f"  N = {N} (dim = {dim}), D_MAX = {D_MAX}")
    print(f"{'='*60}")

    # Memory check (one matrix = dim^2 * 16 bytes)
    mem_gb = (dim**2 * 16) / 1e9
    print(f"  Single matrix: {mem_gb:.2f} GB")
    # We need ~4 matrices simultaneously: U, Ud, Wd, product
    total_mem = mem_gb * 4
    print(f"  Estimated peak: {total_mem:.2f} GB")
    if total_mem > MAX_MEMORY_GB:
        print(f"  ⚠️  Exceeds {MAX_MEMORY_GB} GB limit. Skipping.")
        return None

    t_start = time.time()

    print("  Building Floquet operator...")
    U = kicked_ising_U(N, J, h, b)
    t_build = time.time() - t_start
    print(f"  Build time: {t_build:.1f}s")

    print("  Computing OTOC profile (trace-based)...")
    C_all = compute_otoc_profile(U, N)
    t_otoc = time.time() - t_start
    print(f"  OTOC time: {t_otoc:.1f}s")

    # Compute Omega for each distance
    omega_profile = {}
    omega_var_profile = {}

    for r in range(1, N):
        omega, omega_var, C0 = compute_omega_from_C(C_all[r], D_MAX)
        omega_profile[r] = omega
        omega_var_profile[r] = omega_var
        print(f"  r={r:2d}: Omega={omega:.6f}, Omega_var={omega_var:.6f}, C0={C0:.4f}")

    t_total = time.time() - t_start

    r_half = N // 2
    result = {
        "N": N,
        "dim": dim,
        "D_MAX": D_MAX,
        "method": "dense_trace",
        "omega_profile": {str(r): float(v) for r, v in omega_profile.items()},
        "omega_var_profile": {str(r): float(v) for r, v in omega_var_profile.items()},
        "omega_local": float(omega_profile.get(1, float('nan'))),
        "omega_half": float(omega_profile.get(r_half, float('nan'))),
        "omega_far": float(omega_profile.get(N-1, float('nan'))),
        "omega_global": float(np.mean(list(omega_profile.values()))),
        "time_sec": round(t_total, 1),
        "params": {"J": J, "h": h, "b": b, "boundary": "OBC"},
    }

    print(f"\n  Summary: Omega_local={result['omega_local']:.4f}, "
          f"Omega_half={result['omega_half']:.4f}, "
          f"Omega_far={result['omega_far']:.4f}")
    print(f"  Total time: {t_total:.1f}s")

    return result

def verify_reference(result, N):
    """Check against P1v2 reference values."""
    if N in REFERENCE:
        ref = REFERENCE[N]["omega_local"]
        got = result["omega_local"]
        diff = abs(ref - got)
        ok = diff < 0.005
        symbol = "✅" if ok else "❌"
        print(f"  Verification N={N}: ref={ref:.4f}, got={got:.4f}, "
              f"diff={diff:.4f} {symbol}")
        if not ok:
            print(f"  ⚠️  MISMATCH > 0.005! Check code.")
        return ok
    return True

# ============================================================
# ANALYSIS
# ============================================================

def run_analysis(all_results):
    """gamma(N) diagnostic and plots."""
    print("\n" + "=" * 60)
    print("ANALISIS: gamma(N) = d ln(Omega) / d ln(N)")
    print("  Constante --> Hip. C (power-law)")
    print("  Varia     --> Hip. A (logistica)")
    print("=" * 60)

    Ns = sorted([int(k) for k in all_results.keys()])

    analysis = {}

    for metric in ["omega_half", "omega_local"]:
        print(f"\n--- {metric} ---")
        vals = []
        for N in Ns:
            v = all_results[str(N)].get(metric, float('nan'))
            if not np.isnan(v) and v > 0:
                vals.append((N, v))

        if len(vals) < 2:
            print("  Insuficientes datos.")
            continue

        gammas = []
        for i in range(1, len(vals)):
            N1, O1 = vals[i-1]
            N2, O2 = vals[i]
            if O1 > 0 and O2 > 0:
                gamma = (np.log(O2) - np.log(O1)) / (np.log(N2) - np.log(N1))
                gammas.append({"N1": N1, "N2": N2, "gamma": gamma,
                               "Omega_mid": (O1+O2)/2})
                print(f"  N={N1}->{N2}: gamma = {gamma:.4f}, "
                      f"Omega_mid = {(O1+O2)/2:.4f}")

        if len(gammas) >= 2:
            g_vals = [g["gamma"] for g in gammas]
            mean_g = np.mean(g_vals)
            cv = np.std(g_vals) / abs(mean_g) if abs(mean_g) > 1e-10 else float('inf')
            print(f"  Mean gamma = {mean_g:.4f}, CV = {cv:.3f}")
            verdict = "CONSTANTE (Hip. C)" if cv < 0.1 else \
                      "VARIA (Hip. A)" if cv > 0.3 else "INTERMEDIO"
            print(f"  --> {verdict}")
            analysis[metric] = {
                "gammas": gammas, "mean": mean_g, "cv": cv, "verdict": verdict
            }

    return analysis

def make_plots(all_results):
    """Generate diagnostic plots."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        Ns = sorted([int(k) for k in all_results.keys()])

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: Omega(N) metrics
        ax = axes[0, 0]
        for metric, label, color in [
            ("omega_local", "Omega_local (r=1)", "blue"),
            ("omega_half", "Omega_half (r=N/2)", "red"),
            ("omega_far", "Omega_far (r=N-1)", "green"),
        ]:
            ns, vs = [], []
            for N in Ns:
                v = all_results[str(N)].get(metric, float('nan'))
                if not np.isnan(v):
                    ns.append(N); vs.append(v)
            if ns:
                ax.plot(ns, vs, 'o-', label=label, color=color, markersize=6)
        ax.set_xlabel("N"); ax.set_ylabel("Omega")
        ax.set_title("KI_chaotic: Omega vs N"); ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Plot 2: Omega(r) profiles
        ax = axes[0, 1]
        cmap = plt.cm.viridis
        for idx, N in enumerate(Ns):
            profile = all_results[str(N)].get("omega_profile", {})
            if profile:
                rs = sorted([int(k) for k in profile.keys()])
                omegas = [profile[str(r)] for r in rs]
                color = cmap(idx / max(len(Ns)-1, 1))
                ax.plot([r/N for r in rs], omegas, 'o-', label=f"N={N}",
                       color=color, markersize=4, linewidth=1)
        ax.set_xlabel("r/N"); ax.set_ylabel("Omega(r)")
        ax.set_title("Omega(r/N) Profiles"); ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

        # Plot 3: gamma(N) for omega_half
        ax = axes[1, 0]
        vals_half = [(N, all_results[str(N)]["omega_half"])
                     for N in Ns if all_results[str(N)].get("omega_half", 0) > 0]
        if len(vals_half) >= 2:
            g_ns, g_vals = [], []
            for i in range(1, len(vals_half)):
                N1, O1 = vals_half[i-1]; N2, O2 = vals_half[i]
                g = (np.log(O2) - np.log(O1)) / (np.log(N2) - np.log(N1))
                g_ns.append((N1+N2)/2); g_vals.append(g)
            ax.plot(g_ns, g_vals, 'rs-', markersize=8, linewidth=2)
            ax.axhline(y=np.mean(g_vals), color='gray', linestyle='--', alpha=0.5,
                       label=f"mean={np.mean(g_vals):.3f}")
            ax.set_xlabel("N (midpoint)"); ax.set_ylabel("gamma(N)")
            ax.set_title("gamma = d ln(Omega_half) / d ln(N)\n"
                         "Constant→Hip.C | Varies→Hip.A")
            ax.legend(); ax.grid(True, alpha=0.3)

        # Plot 4: log-log
        ax = axes[1, 1]
        if len(vals_half) >= 2:
            ns_arr = np.array([v[0] for v in vals_half])
            os_arr = np.array([v[1] for v in vals_half])
            ax.plot(np.log(ns_arr), np.log(os_arr), 'ro-', markersize=8)
            if len(vals_half) >= 3:
                coeffs = np.polyfit(np.log(ns_arr), np.log(os_arr), 1)
                ax.plot(np.log(ns_arr), np.polyval(coeffs, np.log(ns_arr)),
                       'b--', label=f"alpha={-coeffs[0]:.3f}")
                ax.legend()
            ax.set_xlabel("ln(N)"); ax.set_ylabel("ln(Omega_half)")
            ax.set_title("Log-log: Omega_half vs N"); ax.grid(True, alpha=0.3)

        plt.suptitle(f"KI_chaotic Scaling (v2 DENSE)\n"
                     f"J={J}, h={h}, b={b}, OBC, D_MAX=max(20,3N)",
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig("ki_chaotic_scaling_v2_dense.png", dpi=150, bbox_inches='tight')
        print(f"\nSaved: ki_chaotic_scaling_v2_dense.png")
    except ImportError:
        print("\nMatplotlib not available.")

# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("KAELION -- KI_chaotic Omega(N) Scaling v2 DENSE")
    print("OTOC: Tr[W(d)† V W(d) V] / dim (matches P1v2)")
    print(f"Params: J={J}, h={h}, b={b}, OBC")
    print(f"D_MAX = max(20, 3N)")
    print(f"Sizes: {N_LIST}")
    print(f"Memory limit: {MAX_MEMORY_GB} GB")
    print("=" * 60)

    all_results = {}

    # Load existing results if any
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, 'r') as f:
                all_results = json.load(f)
            print(f"\nLoaded {len(all_results)} existing results")
        except:
            all_results = {}

    for N in N_LIST:
        N_key = str(N)

        if N_key in all_results:
            print(f"\n  N={N} already computed. Skipping.")
            continue

        try:
            result = run_single_N(N)
            if result is None:
                continue

            verify_reference(result, N)
            all_results[N_key] = result

            # Save incrementally
            with open(OUTPUT_FILE, 'w') as f:
                json.dump(all_results, f, indent=2)
            print(f"  Saved to {OUTPUT_FILE}")

        except MemoryError:
            print(f"\n  ❌ MemoryError at N={N}. Stopping dense phase.")
            break
        except Exception as e:
            print(f"\n  ❌ Error at N={N}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Analysis
    analysis = run_analysis(all_results)

    # Save analysis
    with open(OUTPUT_FILE, 'w') as f:
        json.dump({"data": all_results, "analysis": analysis}, f, indent=2)

    # Plots
    make_plots(all_results)

    # Summary table
    Ns = sorted([int(k) for k in all_results.keys()])
    print("\n" + "=" * 60)
    print("RESUMEN FINAL")
    print("=" * 60)
    print(f"{'N':>4} {'dim':>8} {'D_MAX':>5} {'O_local':>8} {'O_half':>8} "
          f"{'O_far':>8} {'Time':>8}")
    print("-" * 60)
    for N in Ns:
        r = all_results[str(N)]
        print(f"{N:4d} {r['dim']:8d} {r['D_MAX']:5d} {r['omega_local']:8.4f} "
              f"{r['omega_half']:8.4f} {r['omega_far']:8.4f} "
              f"{r['time_sec']:7.1f}s")

    print("\n" + "=" * 60)
    print("SIGUIENTE PASO:")
    print("  1. Verificar que N=4-10 coincidan con referencia P1v2")
    print("  2. Si OK: ejecutar v3_typicality para N=14-20")
    print("  3. Enviar JSON + PNG a Epistemologico")
    print("=" * 60)

if __name__ == "__main__":
    main()
