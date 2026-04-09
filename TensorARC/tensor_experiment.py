#!/usr/bin/env python3
"""
Tensor Distinguishability Experiment
=====================================

Compares two families of order-3 tensors T in R^{N x N x N}:

  Type 1 (Generic CP):    T = sum_{r=1}^{R} u_r (x) w_r (x) z_r         R = A*B
  Type 2 (Shared mode-1): T = sum_{i=1}^A sum_{j=1}^B v_i (x) w_{ij} (x) z_{ij}

In Type 2 the mode-1 vector v_i is shared across all B terms in group i.
All component vectors have i.i.d. N(0,1) entries.

Regime of interest: R = AB << N^2,  A >> N.

Analytical kurtosis prediction
-------------------------------
  Type 1 excess kurtosis of entries:  24 / R
  Type 2 excess kurtosis of entries:  (72 + 6B) / R
  Ratio (Type 2 / Type 1):           3 + B/4

The shared mode-1 vectors create higher-order correlations among entries,
producing measurably heavier tails in Type 2.

Usage
-----
  python tensor_experiment.py                        # default parameters
  python tensor_experiment.py --N 50 --A 500 --B 2   # custom
  python tensor_experiment.py --sweep                 # parameter sweep (fixed R, varying B)
"""

import argparse
import os
import time
import numpy as np
from scipy import stats as sp_stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -- Tensor generation --------------------------------------------------------

def generate_type1(N, R, rng):
    """Generic CP tensor: sum_{r=1}^R u_r (x) w_r (x) z_r."""
    U = rng.standard_normal((R, N))
    W = rng.standard_normal((R, N))
    Z = rng.standard_normal((R, N))
    return np.einsum("ri,rj,rk->ijk", U, W, Z, optimize=True)


def generate_type2(N, A, B, rng):
    """Structured tensor: sum_i sum_j v_i (x) w_{ij} (x) z_{ij}  (shared mode-1)."""
    V = rng.standard_normal((A, N))
    W = rng.standard_normal((A, B, N))
    Z = rng.standard_normal((A, B, N))
    return np.einsum("ai,abj,abk->ijk", V, W, Z, optimize=True)


# -- Analysis helpers ---------------------------------------------------------

def mode_unfold(T, mode):
    """Matricize tensor T along the given mode (0, 1, or 2)."""
    return np.reshape(np.moveaxis(T, mode, 0), (T.shape[mode], -1))


def normalized_svs(M):
    """Singular values normalized so the largest is 1."""
    s = np.linalg.svd(M, compute_uv=False)
    return s / (s[0] + 1e-30)


def effective_rank(svs):
    """Shannon-entropy effective rank from (already-normalized) singular values."""
    p = svs[svs > 1e-15] ** 2
    p /= p.sum()
    return float(np.exp(-np.sum(p * np.log(p + 1e-30))))


def compute_stats(T):
    """Full statistics (uses SVD). For the restricted experiment, see compute_low_degree_stats."""
    N = T.shape[0]
    s = {}

    # -- Entry-level --
    flat = T.ravel()
    s["entry_kurtosis"] = float(sp_stats.kurtosis(flat, fisher=True))
    s["entry_std"] = float(np.std(flat))

    # -- Mode unfolding spectra --
    for m in range(3):
        svs = normalized_svs(mode_unfold(T, m))
        s[f"mode{m}_eff_rank"] = effective_rank(svs)
        s[f"mode{m}_nuclear"] = float(np.sum(svs))
        s[f"mode{m}_sv5"] = float(svs[min(5, len(svs) - 1)])
        s[f"mode{m}_tail"] = float(np.sum(svs[N // 2:]))

    # -- Mode-0 slice analysis: T[n,:,:] for each n --
    slice_eranks, slice_sv2s = [], []
    for n in range(N):
        svs = normalized_svs(T[n, :, :])
        slice_eranks.append(effective_rank(svs))
        if len(svs) > 1:
            slice_sv2s.append(float(svs[1]))
    s["slice_erank_mean"] = float(np.mean(slice_eranks))
    s["slice_erank_std"] = float(np.std(slice_eranks))
    s["slice_sv2_mean"] = float(np.mean(slice_sv2s))

    # -- Slice correlation structure --
    slices = T.reshape(N, -1)
    norms = np.linalg.norm(slices, axis=1, keepdims=True) + 1e-15
    G = (slices / norms) @ (slices / norms).T
    tri = G[np.triu_indices(N, k=1)]
    s["slice_corr_mean"] = float(np.mean(np.abs(tri)))

    # -- Random-probe contraction: T(x, ., .) for random unit x --
    rng_p = np.random.default_rng(999)
    probe_eranks = []
    for _ in range(10):
        x = rng_p.standard_normal(N)
        Mx = np.einsum("ijk,i->jk", T, x / np.linalg.norm(x), optimize=True)
        probe_eranks.append(effective_rank(normalized_svs(Mx)))
    s["probe_erank_mean"] = float(np.mean(probe_eranks))

    s["frobenius"] = float(np.linalg.norm(T))
    return s


def compute_low_degree_stats(T):
    """Degree <= 4 polynomial invariants only (spectral power <= 4).

    Allowed operations on any mode-m Gram matrix G_m = M_m @ M_m^T:
      tr(G_m)   = sum sigma_i^2  = ||T||_F^2      (power 2)
      tr(G_m^2) = sum sigma_i^4                     (power 4)

    Plus cross-mode degree-4 contractions, entry 4th moment,
    and slice-level decompositions.
    """
    N = T.shape[0]
    s = {}

    # -- Degree 2: Frobenius norm squared  (= tr(G_m) for any m) --
    frob_sq = float(np.sum(T ** 2))
    s["frob_sq"] = frob_sq
    inv_frob4 = 1.0 / (frob_sq ** 2 + 1e-30)

    # -- Degree 4: tr(G_m^2) = ||G_m||_F^2 = sum sigma_i^4  per mode --
    for m in range(3):
        M = mode_unfold(T, m)       # N x N^2
        G = M @ M.T                 # N x N
        tr_G2 = float(np.sum(G ** 2))
        s[f"tr_G{m}_sq"] = tr_G2
        s[f"tr_G{m}_sq_norm"] = tr_G2 * inv_frob4   # 1/eff_rank in power-2 sense

    # -- Degree 4: entry 4th moment  sum T_{ijk}^4 --
    entry_4th = float(np.sum(T ** 4))
    s["entry_4th"] = entry_4th
    s["entry_4th_norm"] = entry_4th * inv_frob4

    # -- Degree 4: cross-mode invariant  (all-different pairing) --
    #    I = sum_{a,a'} || T[a,:,:] @ T[a',:,:]^T ||_F^2
    #    Computed via  C[a,d,b,f] = sum_c T[a,b,c]*T[d,f,c]
    #    then  I = sum C[a,d,b,f]*C[a,d,f,b]
    C = np.einsum("abc,dfc->adbf", T, T, optimize=True)
    cross = float(np.sum(C * C.transpose(0, 1, 3, 2)))
    s["cross_mode"] = cross
    s["cross_mode_norm"] = cross * inv_frob4

    # -- Degree 4: slice Frobenius concentration  sum_n ||T[n,:,:]||_F^4  per mode --
    #    (= diagonal part of tr(G_m^2))
    for m in range(3):
        M = mode_unfold(T, m)
        slice_norms_sq = np.sum(M ** 2, axis=1)        # ||row_n||^2 for each n
        frob4_sum = float(np.sum(slice_norms_sq ** 2))
        s[f"slice{m}_frob4"] = frob4_sum
        s[f"slice{m}_frob4_norm"] = frob4_sum * inv_frob4

    # -- Degree 4: sum_n tr((S_n^T S_n)^2)  per mode  (slice spectral power-4) --
    #    = sum of sigma_i^4 within each slice, summed over slices
    for m in range(3):
        slices_m = np.moveaxis(T, m, 0)     # (N, N, N)
        sv4_total = 0.0
        for n in range(N):
            Sn = slices_m[n]                 # N x N
            Gn = Sn.T @ Sn                   # N x N
            sv4_total += np.sum(Gn ** 2)     # tr(Gn^2)
        s[f"slice{m}_sv4"] = float(sv4_total)
        s[f"slice{m}_sv4_norm"] = float(sv4_total) * inv_frob4

    return s


# -- Experiment runner --------------------------------------------------------

def run(N, A, B, num_samples, seed, stats_fn=None):
    """Generate samples of both types and collect statistics."""
    if stats_fn is None:
        stats_fn = compute_stats
    R = A * B
    rng = np.random.default_rng(seed)
    stats1, stats2 = [], []
    for i in range(num_samples):
        stats1.append(stats_fn(generate_type1(N, R, rng)))
        stats2.append(stats_fn(generate_type2(N, A, B, rng)))
        if (i + 1) % max(1, num_samples // 5) == 0:
            print(f"    {i + 1}/{num_samples}")
    return stats1, stats2


def test_all(stats1, stats2):
    """Two-sample KS and Welch t-tests on every scalar statistic."""
    keys = sorted(stats1[0].keys())
    results = {}
    for k in keys:
        v1 = np.array([s[k] for s in stats1])
        v2 = np.array([s[k] for s in stats2])
        ks_stat, ks_p = sp_stats.ks_2samp(v1, v2)
        t_stat, t_p = sp_stats.ttest_ind(v1, v2, equal_var=False)
        results[k] = dict(
            mean1=float(np.mean(v1)), std1=float(np.std(v1)),
            mean2=float(np.mean(v2)), std2=float(np.std(v2)),
            ks=float(ks_stat), ks_p=float(ks_p),
            t=float(t_stat), t_p=float(t_p),
        )
    return results


# -- Reporting ----------------------------------------------------------------

def print_report(results, N, A, B):
    R = A * B
    ranked = sorted(results.items(), key=lambda x: x[1]["ks_p"])
    hdr = (f"N={N}  A={A}  B={B}  R=AB={R}  "
           f"N^2={N**2}  R/N^2={R / N**2:.2f}  A/N={A / N:.0f}")
    print(f"\n{'=' * 84}")
    print(f"  {hdr}")
    print(f"{'=' * 84}")
    print(f"  {'Statistic':<25} {'mean(T1)':>9} {'mean(T2)':>9} "
          f"{'KS':>7} {'p(KS)':>10} {'sig':>4}")
    print(f"  {'-' * 72}")

    n_sig = 0
    for k, r in ranked:
        sig = ("***" if r["ks_p"] < 0.001 else
               "**"  if r["ks_p"] < 0.01  else
               "*"   if r["ks_p"] < 0.05  else "")
        if r["ks_p"] < 0.05:
            n_sig += 1
        print(f"  {k:<25} {r['mean1']:>9.4f} {r['mean2']:>9.4f} "
              f"{r['ks']:>7.3f} {r['ks_p']:>10.2e} {sig:>4}")

    print(f"  {'-' * 72}")
    print(f"  Significant (p < 0.05): {n_sig}/{len(results)}")

    # Analytical kurtosis comparison
    pred1 = 24.0 / R
    pred2 = (72.0 + 6.0 * B) / R
    print(f"\n  Kurtosis (predicted) :  "
          f"Type1 = {pred1:.5f}   Type2 = {pred2:.5f}   ratio = {pred2 / pred1:.2f}")
    ek = results.get("entry_kurtosis")
    if ek:
        obs_ratio = ek["mean2"] / (ek["mean1"] + 1e-30)
        print(f"  Kurtosis (observed)  :  "
              f"Type1 = {ek['mean1']:.5f}   Type2 = {ek['mean2']:.5f}   ratio = {obs_ratio:.2f}")

    if n_sig == 0:
        print("\n  >>> INDISTINGUISHABLE by all statistics tested.")
    else:
        best_k, best_r = ranked[0]
        print(f"\n  >>> DISTINGUISHABLE  "
              f"(best discriminator: {best_k}, KS p = {best_r['ks_p']:.2e})")


def make_plots(stats1, stats2, results, N, A, B, outdir):
    """Save diagnostic plots to outdir."""
    os.makedirs(outdir, exist_ok=True)
    R = A * B
    tag = f"N={N}  A={A}  B={B}  R={R}"
    ranked = sorted(results.items(), key=lambda x: x[1]["ks_p"])

    # -- 1. KS bar chart --
    fig, ax = plt.subplots(figsize=(10, max(4, len(ranked) * 0.35)))
    names = [r[0] for r in ranked]
    vals = [r[1]["ks"] for r in ranked]
    colors = ["#e74c3c" if r[1]["ks_p"] < 0.01 else
              "#f0ad4e" if r[1]["ks_p"] < 0.05 else "#5cb85c"
              for r in ranked]
    ax.barh(range(len(names)), vals, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("KS statistic")
    ax.set_title(f"Distinguishability  ({tag})\n"
                 "red: p<.01 | orange: p<.05 | green: p>=.05")
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "ks_summary.png"), dpi=150)
    plt.close(fig)

    # -- 2. Top-6 histograms --
    top = [r[0] for r in ranked[:6]]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, key in zip(axes.ravel(), top):
        v1 = [s[key] for s in stats1]
        v2 = [s[key] for s in stats2]
        r = results[key]
        ax.hist(v1, bins=20, alpha=0.55, label="Type 1 (generic)", color="#3498db")
        ax.hist(v2, bins=20, alpha=0.55, label="Type 2 (structured)", color="#e74c3c")
        ax.set_title(f"{key}\nKS={r['ks']:.3f}  p={r['ks_p']:.1e}", fontsize=9)
        ax.legend(fontsize=7)
    plt.suptitle(f"Top Distinguishing Statistics  ({tag})")
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "top_histograms.png"), dpi=150)
    plt.close(fig)

    # -- 3. Averaged SV spectra for each mode unfolding --
    rng = np.random.default_rng(42)
    n_show = 20
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for m, ax in enumerate(axes):
        svs1_all, svs2_all = [], []
        for _ in range(n_show):
            svs1_all.append(normalized_svs(mode_unfold(generate_type1(N, R, rng), m)))
            svs2_all.append(normalized_svs(mode_unfold(generate_type2(N, A, B, rng), m)))
        ax.semilogy(np.mean(svs1_all, axis=0), "b-", alpha=0.8, label="Type 1")
        ax.semilogy(np.mean(svs2_all, axis=0), "r-", alpha=0.8, label="Type 2")
        ax.set_title(f"Mode-{m} Unfolding SVs")
        ax.set_xlabel("Index")
        ax.set_ylabel("Normalized SV")
        ax.legend()
    plt.suptitle(f"Average Singular Value Spectra  ({tag})")
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "sv_spectra.png"), dpi=150)
    plt.close(fig)

    # -- 4. Kurtosis comparison across B (only in sweep mode) --
    # This is generated per-config; the sweep summary is printed to console.

    print(f"    Plots saved to {outdir}/")


# -- Tetrahedral contraction --------------------------------------------------

def tetra_contraction(T):
    """Tetrahedral self-contraction of a 3-way tensor.

    Places T at all 4 vertices of a tetrahedron and contracts along edges:
      Z = Σ_{a,b,c,d,e,f} T[a,b,c] T[a,d,e] T[b,d,f] T[c,e,f]

    Edges (vertex pairs → shared index):
      (1,2)→a  (1,3)→b  (1,4)→c  (2,3)→d  (2,4)→e  (3,4)→f

    3 edges pair same-mode indices (a,d,f), 3 pair cross-mode (b,c,e),
    making this a genuinely mode-mixing degree-4 invariant.

    Computed in O(N^5) via two intermediate contractions.
    """
    # M1[b,c,d,e] = Σ_a T[a,b,c] T[a,d,e]          O(N^5)
    M1 = np.einsum("abc,ade->bcde", T, T, optimize=True)
    # M2[b,d,c,e] = Σ_f T[b,d,f] T[c,e,f]          O(N^5)
    M2 = np.einsum("bdf,cef->bdce", T, T, optimize=True)
    # Z = Σ_{b,c,d,e} M1[b,c,d,e] M2[b,d,c,e]      O(N^4)
    return float(np.einsum("bcde,bdce->", M1, M2))


def run_tetra(N, A, B, num_samples, seed):
    """Compute tetrahedral contraction for many samples of both types."""
    R = A * B
    rng = np.random.default_rng(seed)
    z1, z2, f1, f2 = [], [], [], []
    for i in range(num_samples):
        T1 = generate_type1(N, R, rng)
        T2 = generate_type2(N, A, B, rng)
        z1.append(tetra_contraction(T1))
        z2.append(tetra_contraction(T2))
        f1.append(float(np.sum(T1 ** 2) ** 2))   # ||T||_F^4
        f2.append(float(np.sum(T2 ** 2) ** 2))
        if (i + 1) % max(1, num_samples // 5) == 0:
            print(f"    {i + 1}/{num_samples}")
    return np.array(z1), np.array(z2), np.array(f1), np.array(f2)


def tetra_experiment(N, A, B, num_samples, seed, outdir):
    """Full tetrahedral contraction experiment with report and plots."""
    R = A * B
    print(f"\n{'#' * 60}")
    print(f"  TETRAHEDRAL CONTRACTION")
    print(f"  N={N}  A={A}  B={B}  R={R}  "
          f"(R/N^2={R / N**2:.2f}, A/N={A / N:.0f})")
    print(f"{'#' * 60}")

    t0 = time.time()
    z1, z2, f1, f2 = run_tetra(N, A, B, num_samples, seed)
    elapsed = time.time() - t0
    print(f"    {num_samples} pairs in {elapsed:.1f}s")

    # Normalized version: Z / ||T||_F^4
    zn1 = z1 / f1
    zn2 = z2 / f2

    # Tests on raw and normalized
    ks_raw, p_raw = sp_stats.ks_2samp(z1, z2)
    ks_norm, p_norm = sp_stats.ks_2samp(zn1, zn2)
    t_raw, tp_raw = sp_stats.ttest_ind(z1, z2, equal_var=False)
    t_norm, tp_norm = sp_stats.ttest_ind(zn1, zn2, equal_var=False)

    # Chi-squared and KL divergence on normalized histograms
    all_vals = np.concatenate([zn1, zn2])
    n_bins = max(15, int(np.sqrt(num_samples)))
    bin_edges = np.histogram_bin_edges(all_vals, bins=n_bins)
    h1, _ = np.histogram(zn1, bins=bin_edges)
    h2, _ = np.histogram(zn2, bins=bin_edges)

    # Chi-squared test for homogeneity
    # Pool bins with < 5 expected counts from the edges
    h1f, h2f = h1.astype(float), h2.astype(float)
    # Merge low-count bins from the tails
    while len(h1f) > 3 and (h1f[0] + h2f[0] < 5):
        h1f = np.concatenate([[h1f[0] + h1f[1]], h1f[2:]])
        h2f = np.concatenate([[h2f[0] + h2f[1]], h2f[2:]])
    while len(h1f) > 3 and (h1f[-1] + h2f[-1] < 5):
        h1f = np.concatenate([h1f[:-2], [h1f[-2] + h1f[-1]]])
        h2f = np.concatenate([h2f[:-2], [h2f[-2] + h2f[-1]]])
    n1_total, n2_total = h1f.sum(), h2f.sum()
    expected1 = (h1f + h2f) * n1_total / (n1_total + n2_total)
    expected2 = (h1f + h2f) * n2_total / (n1_total + n2_total)
    mask = (expected1 > 0) & (expected2 > 0)
    chi2 = float(np.sum((h1f[mask] - expected1[mask]) ** 2 / expected1[mask])
                 + np.sum((h2f[mask] - expected2[mask]) ** 2 / expected2[mask]))
    chi2_dof = int(mask.sum() - 1)
    chi2_p = float(1 - sp_stats.chi2.cdf(chi2, chi2_dof)) if chi2_dof > 0 else 1.0

    # KL divergence (symmetrized: Jensen-Shannon)
    eps = 1e-10
    p1 = h1f / (h1f.sum() + eps) + eps
    p2 = h2f / (h2f.sum() + eps) + eps
    p1 /= p1.sum()
    p2 /= p2.sum()
    m = 0.5 * (p1 + p2)
    kl_1m = float(np.sum(p1 * np.log(p1 / m)))
    kl_2m = float(np.sum(p2 * np.log(p2 / m)))
    jsd = 0.5 * kl_1m + 0.5 * kl_2m   # Jensen-Shannon divergence

    print(f"\n  {'':30} {'mean(T1)':>14} {'mean(T2)':>14} {'KS':>7} {'p(KS)':>10}")
    print(f"  {'-' * 78}")
    print(f"  {'Z (raw)':30} {np.mean(z1):>14.4e} {np.mean(z2):>14.4e} "
          f"{ks_raw:>7.3f} {p_raw:>10.2e}")
    print(f"  {'Z / ||T||_F^4 (normalized)':30} {np.mean(zn1):>14.6f} {np.mean(zn2):>14.6f} "
          f"{ks_norm:>7.3f} {p_norm:>10.2e}")
    print(f"  {'-' * 78}")
    print(f"  {'std(T1)':30} {np.std(z1):>14.4e} {np.std(z2):>14.4e}")
    print(f"  {'std normalized':30} {np.std(zn1):>14.6f} {np.std(zn2):>14.6f}")
    print(f"  {'-' * 78}")
    print(f"  Chi-squared test:   chi2 = {chi2:.2f},  dof = {chi2_dof},  p = {chi2_p:.4e}")
    print(f"  Jensen-Shannon div: JSD  = {jsd:.6f}  (0 = identical, ln2 = maximally different)")
    print(f"  KL(T1||M) = {kl_1m:.6f},  KL(T2||M) = {kl_2m:.6f}")

    if min(p_norm, chi2_p) < 0.05:
        best_p = min(p_norm, chi2_p)
        print(f"\n  >>> DISTINGUISHABLE via tetrahedral contraction (best p = {best_p:.2e})")
    else:
        print(f"\n  >>> NOT distinguishable (KS p = {p_norm:.2e}, chi2 p = {chi2_p:.2e})")

    # -- Plots --
    os.makedirs(outdir, exist_ok=True)
    tag = f"N={N}  A={A}  B={B}  R={R}"

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.hist(z1, bins=25, alpha=0.55, label="Type 1 (generic)", color="#3498db")
    ax.hist(z2, bins=25, alpha=0.55, label="Type 2 (structured)", color="#e74c3c")
    ax.set_xlabel("Z (raw)")
    ax.set_title(f"Tetrahedral contraction (raw)\nKS={ks_raw:.3f}  p={p_raw:.1e}")
    ax.legend()

    ax = axes[1]
    ax.hist(zn1, bins=25, alpha=0.55, label="Type 1 (generic)", color="#3498db")
    ax.hist(zn2, bins=25, alpha=0.55, label="Type 2 (structured)", color="#e74c3c")
    ax.set_xlabel("Z / ||T||_F^4")
    ax.set_title(f"Tetrahedral contraction (normalized)\nKS={ks_norm:.3f}  p={p_norm:.1e}")
    ax.legend()

    plt.suptitle(f"Tetrahedral Contraction Histograms  ({tag})")
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "tetra_histograms.png"), dpi=150)
    plt.close(fig)
    print(f"    Plot saved to {outdir}/tetra_histograms.png")


# -- CLI ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--N", type=int, default=30, help="Bond dimension (default 30)")
    ap.add_argument("--A", type=int, default=150,
                    help="Number of groups in Type 2 (default 150, should be >> N)")
    ap.add_argument("--B", type=int, default=2,
                    help="Group size in Type 2 (default 2)")
    ap.add_argument("--samples", type=int, default=100,
                    help="Samples per tensor type (default 100)")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed (default 42)")
    ap.add_argument("--outdir", type=str, default=None,
                    help="Output directory for plots (auto-generated if omitted)")
    ap.add_argument("--sweep", action="store_true",
                    help="Sweep B in {1,2,3,4,6} at fixed R ~ N^2/3")
    ap.add_argument("--low-degree", action="store_true",
                    help="Use only degree<=4 invariants (tr G, tr G^2, cross-mode, entry 4th moment)")
    ap.add_argument("--tetra", action="store_true",
                    help="Run tetrahedral self-contraction experiment")
    args = ap.parse_args()

    # -- Tetrahedral contraction mode --
    if args.tetra:
        if args.sweep:
            N = args.N
            R_target = N * N // 3
            cfgs = []
            for B in [1, 2, 3, 4, 6]:
                A = max(N + 1, R_target // B)
                cfgs.append((N, A, B))
        else:
            cfgs = [(args.N, args.A, args.B)]
        for N, A, B in cfgs:
            outdir = args.outdir or os.path.join(
                "TensorARC", f"plots_N{N}_A{A}_B{B}")
            tetra_experiment(N, A, B, args.samples, args.seed, outdir)
        return

    if args.sweep:
        N = args.N
        R_target = N * N // 3
        cfgs = []
        for B in [1, 2, 3, 4, 6]:
            A = max(N + 1, R_target // B)
            cfgs.append((N, A, B))
        print(f"Sweep mode: N={N}, R_target~{R_target}, "
              f"varying B in {{1,2,3,4,6}}")
        print(f"  B=1 is the CONTROL (Type 2 reduces to Type 1).\n")
    else:
        cfgs = [(args.N, args.A, args.B)]

    stats_fn = compute_low_degree_stats if args.low_degree else compute_stats
    if args.low_degree:
        print("*** LOW-DEGREE MODE: only degree<=4 polynomial invariants ***\n")

    sweep_summary = []

    for N, A, B in cfgs:
        R = A * B
        print(f"\n{'#' * 60}")
        print(f"  N={N}  A={A}  B={B}  R={R}  "
              f"(R/N^2={R / N**2:.2f}, A/N={A / N:.0f})")
        print(f"{'#' * 60}")

        t0 = time.time()
        s1, s2 = run(N, A, B, args.samples, args.seed, stats_fn=stats_fn)
        elapsed = time.time() - t0
        print(f"    Generated {args.samples} pairs in {elapsed:.1f}s")

        res = test_all(s1, s2)
        print_report(res, N, A, B)

        outdir = args.outdir or os.path.join("TensorARC", f"plots_N{N}_A{A}_B{B}")
        make_plots(s1, s2, res, N, A, B, outdir)

        # Collect sweep data
        ek = res.get("entry_kurtosis", {})
        n_sig = sum(1 for r in res.values() if r["ks_p"] < 0.05)
        sweep_summary.append(dict(
            N=N, A=A, B=B, R=R,
            n_sig=n_sig,
            kurt1=ek.get("mean1", 0), kurt2=ek.get("mean2", 0),
            kurt_pred1=24.0 / R, kurt_pred2=(72.0 + 6.0 * B) / R,
        ))

    # -- Sweep summary table --
    if len(sweep_summary) > 1:
        print(f"\n\n{'=' * 84}")
        print("  SWEEP SUMMARY")
        print(f"{'=' * 84}")
        print(f"  {'B':>3} {'A':>5} {'R':>6} {'R/N^2':>6} "
              f"{'kurt1_obs':>10} {'kurt1_pred':>11} "
              f"{'kurt2_obs':>10} {'kurt2_pred':>11} "
              f"{'ratio_obs':>10} {'ratio_pred':>11} {'#sig':>5}")
        print(f"  {'-' * 80}")
        for d in sweep_summary:
            r_obs = d["kurt2"] / (d["kurt1"] + 1e-30)
            r_pred = d["kurt_pred2"] / (d["kurt_pred1"] + 1e-30)
            print(f"  {d['B']:>3} {d['A']:>5} {d['R']:>6} "
                  f"{d['R'] / N**2:>6.2f} "
                  f"{d['kurt1']:>10.5f} {d['kurt_pred1']:>11.5f} "
                  f"{d['kurt2']:>10.5f} {d['kurt_pred2']:>11.5f} "
                  f"{r_obs:>10.2f} {r_pred:>11.2f} "
                  f"{d['n_sig']:>5}")
        print(f"\n  B=1 is the control (identical structure). "
              f"Predicted ratio = 3 + B/4.")


if __name__ == "__main__":
    main()
