"""
=============================================================
KAELION PROJECT -- KI_chaotic Omega(N) Scaling v3 (TYPICALITY)
=============================================================
PHASE 2: N=14,16,18,20 using gate-based statevector + typicality.

KEY INNOVATION: No dim×dim matrices stored.
  - U applied via local gates (diagonal phases + single-qubit rotations)
  - OTOC via quantum typicality: C ≈ <ψ|W(d)†VW(d)V|ψ> averaged
  - Memory: O(dim) per vector, NOT O(dim²) per matrix

Memory per statevector (complex128):
  N=14: 0.26 MB   |  N=16: 1.0 MB   |  N=18: 4.0 MB
  N=20: 16 MB     |  N=22: 64 MB    |  N=24: 256 MB

Estimated runtime (10 samples, Kaggle CPU):
  N=14: ~2 min    |  N=16: ~15 min   |  N=18: ~2 hours
  N=20: ~12 hours |  (N=22: ~2 days, optional)

Modelo: Kicked Ising chaotic, Conjunto B
  J=1.0, h=0.5, b=0.5, OBC

OTOC: C(d,r) = Tr[W(d)† V W(d) V] / dim
  ≈ <ψ|W(d)† V W(d) V|ψ>  (typicality)
  W = Z_0, V = Z_r

INSTRUCCIONES:
1. Run v2_dense first for N=4-12 (MUST verify references)
2. Place v2 output JSON in same directory
3. Run this script -- it loads v2 data and extends to N=14+
4. Results merge into single analysis

Fecha: 2026-02-19, sesion 20
=============================================================
"""

import numpy as np
import json, time, os, sys

# ============================================================
# PARAMETERS
# ============================================================

J, h, b = 1.0, 0.5, 0.5           # Conjunto B
LATE_FRACTION = 0.5                 # Match P1v2
N_TYPICALITY = [14, 16, 18, 20]    # Typicality targets
N_SAMPLES = 10                      # Random states for typicality
SEED = 20260219                     # Reproducible

def get_D_MAX(N):
    """D_MAX scales with N. 5N ensures scrambling saturates at all distances."""
    return max(20, 5 * N)

# v2 dense results file (for merging)
V2_FILE = "ki_chaotic_scaling_v2_dense.json"
OUTPUT_FILE = "ki_chaotic_scaling_v3_combined.json"

# ============================================================
# GATE-BASED STATEVECTOR OPERATIONS
# ============================================================

def precompute_gate_data(N, J, h, b):
    """
    Precompute all gate data needed for U and U† application.
    Returns dict with phase vectors and rotation parameters.
    
    U = exp(-ihΣX) · exp(-ibΣZ) · exp(-iJΣZ_iZ_{i+1})
    
    All stored as 1D arrays of size dim = 2^N.
    Total memory: ~3 vectors × dim × 16 bytes
    N=20: ~48 MB. Negligible.
    """
    dim = 2**N
    bits = np.arange(dim, dtype=np.int64)
    
    # Phase for exp(-i J Σ Z_i Z_{i+1}): diagonal in comp basis
    diag_ZZ = np.zeros(dim, dtype=np.float64)
    for i in range(N - 1):
        bit_i = (bits >> (N - 1 - i)) & 1
        bit_ip1 = (bits >> (N - 1 - (i + 1))) & 1
        zz = 1 - 2 * (bit_i ^ bit_ip1)  # +1 same, -1 different
        diag_ZZ += J * zz
    phase_ZZ = np.exp(-1j * diag_ZZ)
    phase_ZZ_conj = phase_ZZ.conj()  # For U†
    
    # Phase for exp(-i b Σ Z_i): diagonal
    diag_Z = np.zeros(dim, dtype=np.float64)
    for i in range(N):
        bit_i = (bits >> (N - 1 - i)) & 1
        z = 1 - 2 * bit_i
        diag_Z += b * z
    phase_Z = np.exp(-1j * diag_Z)
    phase_Z_conj = phase_Z.conj()
    
    # Pauli Z sign vectors for each site (for W=Z_0 and V=Z_r)
    z_signs = {}
    for site in range(N):
        mask = (bits >> (N - 1 - site)) & 1
        z_signs[site] = (1 - 2 * mask).astype(np.float64)
    
    return {
        "phase_ZZ": phase_ZZ,
        "phase_ZZ_conj": phase_ZZ_conj,
        "phase_Z": phase_Z,
        "phase_Z_conj": phase_Z_conj,
        "z_signs": z_signs,
        "cos_h": np.cos(h),
        "sin_h": np.sin(h),
        "N": N,
        "dim": dim,
    }

def apply_rx_all_qubits(state, cos_h, sin_h, N):
    """
    Apply exp(-i h X) to ALL qubits simultaneously.
    Uses reshape trick for vectorized single-qubit gates.
    
    Rx = [[cos(h), -i sin(h)], [-i sin(h), cos(h)]]
    """
    state = state.reshape([2] * N)
    for qubit in range(N):
        # Move target qubit to last axis
        state = np.moveaxis(state, qubit, -1)
        # Apply rotation: new[...,0] = c*old[...,0] - is*old[...,1]
        s0 = state[..., 0].copy()
        s1 = state[..., 1].copy()
        state[..., 0] = cos_h * s0 + (-1j * sin_h) * s1
        state[..., 1] = (-1j * sin_h) * s0 + cos_h * s1
        # Move back
        state = np.moveaxis(state, -1, qubit)
    return state.reshape(-1)

def apply_rx_all_qubits_dag(state, cos_h, sin_h, N):
    """
    Apply exp(+i h X) = Rx† to ALL qubits.
    Rx† = [[cos(h), +i sin(h)], [+i sin(h), cos(h)]]
    """
    state = state.reshape([2] * N)
    for qubit in range(N):
        state = np.moveaxis(state, qubit, -1)
        s0 = state[..., 0].copy()
        s1 = state[..., 1].copy()
        state[..., 0] = cos_h * s0 + (1j * sin_h) * s1
        state[..., 1] = (1j * sin_h) * s0 + cos_h * s1
        state = np.moveaxis(state, -1, qubit)
    return state.reshape(-1)

def apply_U(state, gd):
    """
    Apply U = exp(-ihΣX) · exp(-ibΣZ) · exp(-iJΣZZ) to statevector.
    Order: ZZ first, then Z, then X (rightmost acts first).
    """
    state = state * gd["phase_ZZ"]       # exp(-iJΣZZ)
    state = state * gd["phase_Z"]        # exp(-ibΣZ)
    state = apply_rx_all_qubits(state, gd["cos_h"], gd["sin_h"], gd["N"])
    return state

def apply_U_dag(state, gd):
    """
    Apply U† = exp(+iJΣZZ) · exp(+ibΣZ) · exp(+ihΣX).
    Reverse order of U.
    """
    state = apply_rx_all_qubits_dag(state, gd["cos_h"], gd["sin_h"], gd["N"])
    state = state * gd["phase_Z_conj"]
    state = state * gd["phase_ZZ_conj"]
    return state

def apply_pauli_z(state, site, gd):
    """Apply Z on site: multiply by ±1 based on bit value."""
    return state * gd["z_signs"][site]

def random_statevector(dim, rng):
    """Generate Haar-random statevector."""
    psi = rng.randn(dim) + 1j * rng.randn(dim)
    psi /= np.linalg.norm(psi)
    return psi

# ============================================================
# TYPICALITY OTOC COMPUTATION
# ============================================================

def compute_otoc_typicality(N, gd, n_samples=10, seed=12345):
    """
    Compute C(d, r) for W=Z_0, V=Z_r using typicality.
    
    C(d,r) = Tr[W(d)† V W(d) V] / dim
           ≈ <ψ| W(d)† V W(d) V |ψ>   (averaged over Haar random ψ)
    
    Since W = Z_0 is Hermitian: W(d)† = W(d).
    
    Algorithm per sample per depth d:
      1. |f_d> = U^d|ψ>  (incremental forward evolution)
      2. |g_d> = W|f_d>   (apply Z_0)
      3. |a_d> = U^{-d}|g_d>  (backward evolution: W(d)|ψ>)
      4. For each r:
         a. |h_d^r> = U^d V_r|ψ>  (incremental forward of V|ψ>)
         b. |k_d^r> = W|h_d^r>    (apply Z_0)  
         c. |b_d^r> = U^{-d}|k_d^r>  (backward: W(d)V_r|ψ>)
         d. C_sample(d,r) = <a_d| V_r |b_d^r>
    
    Memory: ~(2 + 2*(N-1)) vectors = ~2N vectors
    N=20: ~2*20*16MB = 640 MB. Fine.
    """
    dim = gd["dim"]
    D_MAX = get_D_MAX(N)
    rng = np.random.RandomState(seed)
    
    # Accumulate C(d,r) over samples
    C_accum = {r: np.zeros(D_MAX + 1, dtype=complex) for r in range(1, N)}
    
    t0 = time.time()
    
    for s in range(n_samples):
        psi = random_statevector(dim, rng)
        
        # Forward-evolving states (maintained incrementally)
        fwd_psi = psi.copy()  # Will become U^d|ψ>
        fwd_Vpsi = {}         # Will become U^d V_r|ψ>
        for r in range(1, N):
            fwd_Vpsi[r] = apply_pauli_z(psi, r, gd)
        
        for d in range(D_MAX + 1):
            # Forward evolve one step (except d=0)
            if d > 0:
                fwd_psi = apply_U(fwd_psi, gd)
                for r in range(1, N):
                    fwd_Vpsi[r] = apply_U(fwd_Vpsi[r], gd)
            
            # Apply W = Z_0 to forward states
            g_psi = apply_pauli_z(fwd_psi, 0, gd)  # W·U^d|ψ>
            
            # Backward evolve g_psi: a_d = U^{-d}·W·U^d|ψ> = W(d)|ψ>
            a_d = g_psi.copy()
            for _ in range(d):
                a_d = apply_U_dag(a_d, gd)
            
            # For each distance r
            for r in range(1, N):
                # W · U^d · V_r|ψ>
                g_Vpsi = apply_pauli_z(fwd_Vpsi[r], 0, gd)
                
                # Backward: b_d = U^{-d}·W·U^d·V_r|ψ> = W(d)V_r|ψ>
                b_d = g_Vpsi.copy()
                for _ in range(d):
                    b_d = apply_U_dag(b_d, gd)
                
                # C(d,r) = <W(d)ψ| V_r |W(d)V_r ψ> = <a_d|Z_r|b_d>
                Vb_d = apply_pauli_z(b_d, r, gd)
                C_sample = np.vdot(a_d, Vb_d)  # conjugate-linear in first arg
                C_accum[r][d] += C_sample
            
            # Progress for long runs
            if d > 0 and d % max(1, D_MAX // 5) == 0:
                elapsed = time.time() - t0
                rate = (s * (D_MAX + 1) + d + 1) / elapsed
                total_ops = n_samples * (D_MAX + 1)
                eta = (total_ops - (s * (D_MAX + 1) + d + 1)) / rate
                print(f"    Sample {s+1}/{n_samples}, depth {d}/{D_MAX} "
                      f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")
        
        t_sample = time.time() - t0
        print(f"    Sample {s+1}/{n_samples} complete ({t_sample:.1f}s total)")
    
    # Average over samples
    C_all = {}
    for r in range(1, N):
        C_all[r] = C_accum[r] / n_samples
    
    return C_all

# ============================================================
# CONVERGENCE CHECK
# ============================================================

def check_typicality_convergence(N, gd, seed=12345, max_samples=20, tol=0.01):
    """
    Quick convergence test: run increasing samples for r=1, d=D_MAX
    and check when Omega stabilizes to within tol.
    Returns recommended n_samples.
    """
    dim = gd["dim"]
    D_MAX = get_D_MAX(N)
    rng = np.random.RandomState(seed)
    
    d_test = D_MAX  # Test at last depth (most variable)
    r_test = 1      # Test at r=1 (omega_local)
    
    C_samples = []
    omega_running = []
    
    print(f"  Convergence check (N={N}, r={r_test}, d={d_test}):")
    
    for s in range(max_samples):
        psi = random_statevector(dim, rng)
        
        # Forward evolve
        fwd = psi.copy()
        fwd_V = apply_pauli_z(psi, r_test, gd)
        for _ in range(d_test):
            fwd = apply_U(fwd, gd)
            fwd_V = apply_U(fwd_V, gd)
        
        # W(d)|ψ>
        g = apply_pauli_z(fwd, 0, gd)
        a = g.copy()
        for _ in range(d_test):
            a = apply_U_dag(a, gd)
        
        # W(d)V|ψ>
        gv = apply_pauli_z(fwd_V, 0, gd)
        bv = gv.copy()
        for _ in range(d_test):
            bv = apply_U_dag(bv, gd)
        
        Vb = apply_pauli_z(bv, r_test, gd)
        C_val = np.vdot(a, Vb)
        C_samples.append(abs(C_val))
        
        # Running average
        running_mean = np.mean(C_samples)
        omega_running.append(running_mean)
        
        if s >= 2:
            # Check stability: compare last 3 running means
            recent = omega_running[-3:]
            cv = np.std(recent) / np.mean(recent) if np.mean(recent) > 1e-15 else 0
            print(f"    n={s+1}: |C|_mean={running_mean:.6f}, CV_last3={cv:.4f}")
            if cv < tol and s >= 4:
                print(f"  ✅ Converged at n_samples={s+1}")
                return s + 1
    
    print(f"  ⚠️  Not fully converged at {max_samples} samples, using {max_samples}")
    return max_samples

# ============================================================
# OMEGA FROM C VALUES (same as v2)
# ============================================================

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
# MAIN COMPUTATION FOR SINGLE N
# ============================================================

def run_single_N(N, n_samples=N_SAMPLES):
    """Run full Omega(r) profile for a single N using typicality."""
    dim = 2**N
    D_MAX = get_D_MAX(N)
    
    print(f"\n{'='*60}")
    print(f"  N = {N} (dim = {dim:,}), D_MAX = {D_MAX}")
    print(f"  Method: typicality ({n_samples} samples)")
    print(f"{'='*60}")
    
    # Memory estimate
    n_vectors = 2 * N  # ~2N simultaneous vectors
    mem_mb = n_vectors * dim * 16 / 1e6
    print(f"  Estimated memory: {mem_mb:.1f} MB ({n_vectors} vectors)")
    
    # Time estimate
    # Cost ~ n_samples × D_MAX² × N × dim (approximate)
    ops_estimate = n_samples * D_MAX**2 * N * dim / 2
    print(f"  Estimated ops: {ops_estimate:.1e}")
    
    t_start = time.time()
    
    # Precompute gate data
    print("  Precomputing gate data...")
    gd = precompute_gate_data(N, J, h, b)
    t_gates = time.time() - t_start
    print(f"  Gate data ready ({t_gates:.1f}s)")
    
    # Optional: convergence check for first N
    # Uncomment to auto-tune n_samples:
    # n_samples = check_typicality_convergence(N, gd, seed=SEED)
    
    # Compute OTOC profile
    print(f"  Computing OTOC profile (typicality, {n_samples} samples)...")
    C_all = compute_otoc_typicality(N, gd, n_samples=n_samples, seed=SEED)
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
        "method": f"typicality_{n_samples}samples",
        "n_samples": n_samples,
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
    print(f"  Total time: {t_total:.1f}s ({t_total/60:.1f} min)")
    
    return result

# ============================================================
# ANALYSIS (gamma diagnostic)
# ============================================================

def run_analysis(all_results):
    """gamma(N) diagnostic across ALL N values (dense + typicality)."""
    print("\n" + "=" * 60)
    print("ANALISIS COMBINADO: gamma(N) = d ln(Omega) / d ln(N)")
    print("  Constante (CV<0.1) --> Hip. C (power-law)")
    print("  Varia (CV>0.3)     --> Hip. A (logistica)")
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
                print(f"  N={N:3d}: {metric}={v:.6f} "
                      f"[{all_results[str(N)].get('method', 'unknown')}]")
        
        if len(vals) < 2:
            print("  Insuficientes datos.")
            continue
        
        gammas = []
        for i in range(1, len(vals)):
            N1, O1 = vals[i-1]
            N2, O2 = vals[i]
            if O1 > 0 and O2 > 0:
                gamma = (np.log(O2) - np.log(O1)) / (np.log(N2) - np.log(N1))
                gammas.append({"N1": N1, "N2": N2, "gamma": round(gamma, 6),
                               "Omega_mid": round((O1+O2)/2, 6)})
                print(f"  N={N1}->{N2}: gamma = {gamma:.4f}, "
                      f"Omega_mid = {(O1+O2)/2:.4f}")
        
        if len(gammas) >= 2:
            g_vals = [g["gamma"] for g in gammas]
            mean_g = np.mean(g_vals)
            std_g = np.std(g_vals)
            cv = std_g / abs(mean_g) if abs(mean_g) > 1e-10 else float('inf')
            
            # Also check trend (is gamma changing monotonically?)
            diffs = np.diff(g_vals)
            monotonic = all(d > 0 for d in diffs) or all(d < 0 for d in diffs)
            
            print(f"\n  Mean gamma = {mean_g:.4f}")
            print(f"  Std gamma  = {std_g:.4f}")
            print(f"  CV         = {cv:.3f}")
            print(f"  Monotonic  = {monotonic}")
            
            if cv < 0.1:
                verdict = "CONSTANTE --> Hip. C (power-law, un punto fijo)"
            elif cv > 0.3:
                verdict = "VARIA --> Hip. A (logistica, dos puntos fijos)"
            else:
                verdict = "INTERMEDIO --> No concluyente, necesita N mayor"
            
            print(f"  VEREDICTO: {verdict}")
            
            analysis[metric] = {
                "values": [(N, all_results[str(N)].get(metric))
                           for N in Ns if all_results[str(N)].get(metric, 0) > 0],
                "gammas": gammas,
                "mean_gamma": round(mean_g, 6),
                "std_gamma": round(std_g, 6),
                "cv": round(cv, 4),
                "monotonic": monotonic,
                "verdict": verdict,
            }
    
    # Fit comparison: power-law vs logistic
    print("\n--- FIT COMPARISON ---")
    vals_half = [(N, all_results[str(N)]["omega_half"])
                 for N in Ns if all_results[str(N)].get("omega_half", 0) > 0]
    
    if len(vals_half) >= 4:
        ns = np.array([v[0] for v in vals_half], dtype=float)
        omegas = np.array([v[1] for v in vals_half], dtype=float)
        
        # Power-law fit: ln(Omega) = -alpha*ln(N) + ln(A)
        log_n = np.log(ns)
        log_o = np.log(omegas)
        coeffs_pl = np.polyfit(log_n, log_o, 1)
        alpha_pl = -coeffs_pl[0]
        A_pl = np.exp(coeffs_pl[1])
        resid_pl = np.sum((log_o - np.polyval(coeffs_pl, log_n))**2)
        
        print(f"  Power-law: Omega = {A_pl:.4f} * N^(-{alpha_pl:.4f})")
        print(f"    Residual (log-log): {resid_pl:.6f}")
        
        # Logistic prediction: Omega = 1/(1 + A*N^c) → log(1/Omega - 1) = log(A) + c*log(N)
        if all(o < 1 and o > 0 for o in omegas):
            y_log = np.log(1/omegas - 1)
            coeffs_lg = np.polyfit(log_n, y_log, 1)
            c_lg = coeffs_lg[0]
            A_lg = np.exp(coeffs_lg[1])
            omega_pred_lg = 1 / (1 + A_lg * ns**c_lg)
            resid_lg = np.sum((log_o - np.log(omega_pred_lg))**2)
            
            print(f"  Logistic: Omega = 1/(1 + {A_lg:.4f}*N^{c_lg:.4f})")
            print(f"    Residual (log-log): {resid_lg:.6f}")
            
            if resid_lg < resid_pl * 0.5:
                print(f"  --> Logistic BETTER (ratio: {resid_pl/resid_lg:.1f}x)")
            elif resid_pl < resid_lg * 0.5:
                print(f"  --> Power-law BETTER (ratio: {resid_lg/resid_pl:.1f}x)")
            else:
                print(f"  --> Comparable (ratio: {resid_pl/max(resid_lg,1e-15):.1f}x)")
            
            analysis["fit_comparison"] = {
                "power_law": {"alpha": round(alpha_pl, 6), "A": round(A_pl, 6),
                              "residual": round(resid_pl, 8)},
                "logistic": {"c": round(c_lg, 6), "A": round(A_lg, 6),
                             "residual": round(resid_lg, 8)},
            }
    
    return analysis

def make_plots(all_results, analysis):
    """Generate comprehensive diagnostic plots."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        Ns = sorted([int(k) for k in all_results.keys()])
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Plot 1: Omega(N) all metrics
        ax = axes[0, 0]
        for metric, label, color in [
            ("omega_local", "Ω_local (r=1)", "blue"),
            ("omega_half", "Ω_half (r=N/2)", "red"),
            ("omega_far", "Ω_far (r=N-1)", "green"),
        ]:
            ns, vs = [], []
            for N in Ns:
                v = all_results[str(N)].get(metric, float('nan'))
                if not np.isnan(v):
                    ns.append(N); vs.append(v)
            if ns:
                ax.plot(ns, vs, 'o-', label=label, color=color, markersize=6)
        ax.axvline(x=12.5, color='gray', linestyle=':', alpha=0.5, label='dense|typicality')
        ax.set_xlabel("N"); ax.set_ylabel("Ω")
        ax.set_title("KI_chaotic: Omega vs N"); ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Omega(r/N) profiles
        ax = axes[0, 1]
        cmap = plt.cm.viridis
        for idx, N in enumerate(Ns):
            profile = all_results[str(N)].get("omega_profile", {})
            if profile:
                rs = sorted([int(k) for k in profile.keys()])
                omegas = [profile[str(r)] for r in rs]
                color = cmap(idx / max(len(Ns)-1, 1))
                ax.plot([r/N for r in rs], omegas, 'o-', label=f"N={N}",
                       color=color, markersize=3, linewidth=1)
        ax.set_xlabel("r/N"); ax.set_ylabel("Ω(r)")
        ax.set_title("Ω(r/N) Profiles"); ax.legend(fontsize=6, ncol=2)
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Log-log Omega_half vs N
        ax = axes[0, 2]
        vals_half = [(N, all_results[str(N)]["omega_half"])
                     for N in Ns if all_results[str(N)].get("omega_half", 0) > 0]
        if len(vals_half) >= 2:
            ns_arr = np.array([v[0] for v in vals_half])
            os_arr = np.array([v[1] for v in vals_half])
            ax.plot(np.log(ns_arr), np.log(os_arr), 'ro-', markersize=8)
            if len(vals_half) >= 3:
                coeffs = np.polyfit(np.log(ns_arr), np.log(os_arr), 1)
                x_fit = np.linspace(np.log(ns_arr.min()), np.log(ns_arr.max()), 50)
                ax.plot(x_fit, np.polyval(coeffs, x_fit),
                       'b--', label=f"α={-coeffs[0]:.3f}", linewidth=2)
                ax.legend(fontsize=10)
        ax.set_xlabel("ln(N)"); ax.set_ylabel("ln(Ω_half)")
        ax.set_title("Log-log: Ω_half vs N"); ax.grid(True, alpha=0.3)
        
        # Plot 4: gamma(N) for omega_half -- THE KEY DIAGNOSTIC
        ax = axes[1, 0]
        if len(vals_half) >= 2:
            g_ns, g_vals = [], []
            for i in range(1, len(vals_half)):
                N1, O1 = vals_half[i-1]; N2, O2 = vals_half[i]
                g = (np.log(O2) - np.log(O1)) / (np.log(N2) - np.log(N1))
                g_ns.append((N1+N2)/2); g_vals.append(g)
            
            # Color by method
            colors = ['blue' if n < 13 else 'red' for n in g_ns]
            ax.scatter(g_ns, g_vals, c=colors, s=80, zorder=5, edgecolors='black')
            ax.plot(g_ns, g_vals, 'k-', alpha=0.3, linewidth=1)
            
            mean_g = np.mean(g_vals)
            ax.axhline(y=mean_g, color='gray', linestyle='--', alpha=0.5,
                       label=f"mean={mean_g:.3f}")
            
            # Show CV
            cv = np.std(g_vals) / abs(mean_g) if abs(mean_g) > 1e-10 else float('inf')
            ax.set_xlabel("N (midpoint)", fontsize=11)
            ax.set_ylabel("γ(N)", fontsize=11)
            ax.set_title(f"γ = d ln(Ω_half) / d ln(N)\n"
                         f"CV={cv:.3f} | Blue=dense, Red=typicality",
                         fontsize=10)
            ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
        
        # Plot 5: gamma(N) for omega_local
        ax = axes[1, 1]
        vals_local = [(N, all_results[str(N)]["omega_local"])
                      for N in Ns if all_results[str(N)].get("omega_local", 0) > 0]
        if len(vals_local) >= 2:
            g_ns, g_vals = [], []
            for i in range(1, len(vals_local)):
                N1, O1 = vals_local[i-1]; N2, O2 = vals_local[i]
                g = (np.log(O2) - np.log(O1)) / (np.log(N2) - np.log(N1))
                g_ns.append((N1+N2)/2); g_vals.append(g)
            colors = ['blue' if n < 13 else 'red' for n in g_ns]
            ax.scatter(g_ns, g_vals, c=colors, s=80, zorder=5, edgecolors='black')
            ax.plot(g_ns, g_vals, 'k-', alpha=0.3, linewidth=1)
            mean_g = np.mean(g_vals)
            cv = np.std(g_vals) / abs(mean_g) if abs(mean_g) > 1e-10 else float('inf')
            ax.axhline(y=mean_g, color='gray', linestyle='--', alpha=0.5,
                       label=f"mean={mean_g:.3f}")
            ax.set_xlabel("N (midpoint)"); ax.set_ylabel("γ(N)")
            ax.set_title(f"γ = d ln(Ω_local) / d ln(N)\nCV={cv:.3f}")
            ax.legend(); ax.grid(True, alpha=0.3)
        
        # Plot 6: beta_eff vs Omega (the DeepSeek Q#3 diagnostic)
        ax = axes[1, 2]
        for metric, label, marker in [
            ("omega_half", "Ω_half", "s"),
            ("omega_local", "Ω_local", "o"),
        ]:
            vals = [(N, all_results[str(N)].get(metric, float('nan')))
                    for N in Ns if all_results[str(N)].get(metric, 0) > 0]
            if len(vals) >= 2:
                betas, omegas_mid = [], []
                for i in range(1, len(vals)):
                    N1, O1 = vals[i-1]; N2, O2 = vals[i]
                    beta = (O2 - O1) / (np.log(N2) - np.log(N1))
                    betas.append(beta)
                    omegas_mid.append((O1+O2)/2)
                ax.plot(omegas_mid, betas, f'{marker}-', label=label, markersize=6)
        
        # Overlay theoretical curves
        omega_theory = np.linspace(0.05, 0.4, 100)
        # Hip A: beta = -c*Omega*(1-Omega), fit c from data
        if 'fit_comparison' in analysis and 'logistic' in analysis.get('fit_comparison', {}):
            c_fit = abs(analysis['fit_comparison']['logistic']['c'])
            beta_logistic = -c_fit * omega_theory * (1 - omega_theory)
            ax.plot(omega_theory, beta_logistic, 'g--', alpha=0.5, label=f'Hip.A (c={c_fit:.2f})')
        # Hip C: beta = -alpha*Omega
        if 'fit_comparison' in analysis and 'power_law' in analysis.get('fit_comparison', {}):
            alpha_fit = analysis['fit_comparison']['power_law']['alpha']
            beta_power = -alpha_fit * omega_theory
            ax.plot(omega_theory, beta_power, 'm--', alpha=0.5, label=f'Hip.C (α={alpha_fit:.2f})')
        
        ax.set_xlabel("Ω"); ax.set_ylabel("β_eff = ΔΩ/Δln(N)")
        ax.set_title("β_eff vs Ω (DeepSeek Q#3 diagnostic)")
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
        
        plt.suptitle(f"KI_chaotic Scaling v3 COMBINED (dense N≤12 + typicality N≥14)\n"
                     f"J={J}, h={h}, b={b}, OBC | {N_SAMPLES} samples",
                     fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig("ki_chaotic_scaling_v3_combined.png", dpi=150, bbox_inches='tight')
        print(f"\nSaved: ki_chaotic_scaling_v3_combined.png")
    except ImportError:
        print("\nMatplotlib not available.")

# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("KAELION -- KI_chaotic Omega(N) Scaling v3 TYPICALITY")
    print(f"Gate-based statevector, {N_SAMPLES} Haar-random samples")
    print(f"Params: J={J}, h={h}, b={b}, OBC")
    print(f"D_MAX = max(20, 5N)")
    print(f"Sizes (typicality): {N_TYPICALITY}")
    print("=" * 60)
    
    # Load v2 dense results
    all_results = {}
    if os.path.exists(V2_FILE):
        try:
            with open(V2_FILE, 'r') as f:
                v2_data = json.load(f)
            # Handle both formats: {"4": {...}} or {"data": {"4": {...}}}
            if "data" in v2_data:
                v2_data = v2_data["data"]
            all_results.update(v2_data)
            print(f"\nLoaded {len(all_results)} dense results from {V2_FILE}")
            
            # Quick sanity check
            if "4" in all_results:
                ol4 = all_results["4"].get("omega_local", 0)
                if abs(ol4 - 0.2801) > 0.005:
                    print(f"  ⚠️  N=4 omega_local={ol4:.4f} != 0.2801 reference!")
                    print(f"  ABORTING: Dense results may be incorrect.")
                    return
                else:
                    print(f"  ✅ N=4 omega_local={ol4:.4f} matches reference")
        except Exception as e:
            print(f"  Could not load {V2_FILE}: {e}")
    else:
        print(f"\n  ⚠️  No dense results found ({V2_FILE}).")
        print(f"  Run v2_dense first for N=4-12!")
    
    # Load existing v3 results
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, 'r') as f:
                v3_data = json.load(f)
            if "data" in v3_data:
                v3_data = v3_data["data"]
            # Only load typicality results (don't overwrite dense)
            for k, v in v3_data.items():
                if k not in all_results:
                    all_results[k] = v
            print(f"Loaded existing v3 results")
        except:
            pass
    
    # Run typicality for each N
    for N in N_TYPICALITY:
        N_key = str(N)
        
        if N_key in all_results:
            print(f"\n  N={N} already computed. Skipping.")
            continue
        
        try:
            result = run_single_N(N, n_samples=N_SAMPLES)
            if result is None:
                continue
            
            all_results[N_key] = result
            
            # Save incrementally
            with open(OUTPUT_FILE, 'w') as f:
                json.dump({"data": all_results}, f, indent=2)
            print(f"  Saved to {OUTPUT_FILE}")
            
        except MemoryError:
            print(f"\n  ❌ MemoryError at N={N}. Stopping.")
            break
        except KeyboardInterrupt:
            print(f"\n  ⚠️  Interrupted at N={N}. Saving partial results.")
            with open(OUTPUT_FILE, 'w') as f:
                json.dump({"data": all_results}, f, indent=2)
            break
        except Exception as e:
            print(f"\n  ❌ Error at N={N}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Combined analysis
    analysis = run_analysis(all_results)
    
    # Save everything
    with open(OUTPUT_FILE, 'w') as f:
        json.dump({"data": all_results, "analysis": analysis}, f, indent=2)
    
    # Plots
    make_plots(all_results, analysis)
    
    # Summary table
    Ns = sorted([int(k) for k in all_results.keys()])
    print("\n" + "=" * 70)
    print("RESUMEN FINAL COMBINADO")
    print("=" * 70)
    print(f"{'N':>4} {'dim':>8} {'Method':>15} {'O_local':>8} {'O_half':>8} "
          f"{'O_far':>8} {'Time':>10}")
    print("-" * 70)
    for N in Ns:
        r = all_results[str(N)]
        method = r.get('method', 'unknown')[:15]
        t = r.get('time_sec', 0)
        t_str = f"{t:.1f}s" if t < 60 else f"{t/60:.1f}min" if t < 3600 else f"{t/3600:.1f}h"
        print(f"{N:4d} {r['dim']:8d} {method:>15} {r['omega_local']:8.4f} "
              f"{r['omega_half']:8.4f} {r['omega_far']:8.4f} "
              f"{t_str:>10}")
    
    print("\n" + "=" * 60)
    print("K5 VERDICT:")
    for metric in ["omega_half", "omega_local"]:
        if metric in analysis:
            a = analysis[metric]
            print(f"  {metric}: CV={a['cv']:.3f} --> {a['verdict']}")
    print("=" * 60)

if __name__ == "__main__":
    main()
