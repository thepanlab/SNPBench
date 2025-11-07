# spikein_simulation.py
"""
Synthetic SNP spike-ins for benchmarking DNN interpretability

Overview
- Generates additive, dominant, recessive, and epistatic (2-SNP) synthetic 
  variants aligned to an participant ID list (IID) list and a standardized 
  phenotype vector y.
- Writes IID-aligned CSV genotypes, a minimal hardcall VCF (optionally FAM- 
  aligned to a real PLINK dataset), and a manifest CSV describing 
  each synthetic variant.

Assumptions
- y is already residualized on covariates and standardized (z-score or inverse
 normal transform).
- Warning provided if y looks off-scale (i.e., not ~N(0,1)).

Effect models (how each synthetic signal is constructed)
- Additive:
  * Build latent z correlated with y at target rho, discretize by HWE 
    thresholds to genotypes {0,1,2}.
  * Enforce small “linearity” deviation across class means using a shape guard.
  * Ensure sufficient homozygous alt counts via a MAF floor tied to n22_min.
- Dominant:
  * Correlate the carrier indicator 1[g>0] with y at target rho.
  * Within carriers, split 1 vs 2 according to HWE, ~independent of y.
  * Apply a shape guard so means satisfy m1 ~ m2 and both differ from m0.
- Recessive:
  * Correlate the indicator 1[g==2] with y at target rho.
  * Among non-recessive genotypes, split 0 vs. 1 by HWE, ~independent of y.
  * Apply a shape guard so m0 ~ m1 and m2 differs.
  * Constrain expected # of homozygous alts to an interval for stability.
- Epistatic (2-SNP interaction):
  * Construct two SNPs with small main-effect correlations to y.
  * Tune a parameter by bisection so corr(y, (gA-2pA)*(gB-2pB)) matches a 
    target magnitude.
  * Record the achieved interaction correlation and the alpha that was used.

Inputs:
- --config: path to a YAML file that contains all inputs, ranges, controls, and output settings.

Outputs
- {out_prefix}.syn.csv: IID-aligned hardcalls in {0,1,2}.
- {out_prefix}.manifest.csv: design and observed stats per variant.
- {out_prefix}.syn.vcf: minimal VCF on synthetic chromosome (default "26").
- {out_prefix}.run_config.yaml: YAML snapshot of the resolved configuration.
"""

from pathlib import Path
import argparse
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm
import yaml  # PyYAML

# ------------------------------ basic checks ------------------------------

def check_y_is_standardized(y, tol=5e-2):
    """Warn if y deviates from about N(0,1). We do not modify y; this only flags scale issues."""
    m, s = float(y.mean()), float(y.std())
    if not (abs(m) < tol and abs(s - 1.0) < tol):
        print(
            f"[WARN] y may not be standardized: mean={m:.3f}, std={s:.3f}. "
            "This is fine for INT, but rho targets are on that scale.",
            file=sys.stderr,
        )

# ------------------------------ MAF utilities -----------------------------

def sample_maf_in_range(real_mafs, low, high, rng):
    """
    Draw a MAF in [low, high].
    Prefer sampling from the empirical distribution; fall back to a Beta prior if needed.
    """
    if real_mafs is not None:
        pool = real_mafs[(real_mafs >= low) & (real_mafs <= high)]
        if pool.size:
            return float(pool[rng.integers(pool.size)])
    return float(np.clip(rng.beta(0.7, 3.0), low, min(high, 0.49)))

def draw_add_maf(real_mafs, n_samples, rng, n22_min=1000, p_min=None, max_tries=200):
    """
    Draw a MAF for additive SNPs ensuring enough homozygous ALT (n22) are expected under HWE.
    This stabilizes regressions/means across genotype strata.
    """
    if p_min is None:
        p_min = (n22_min / max(1, n_samples)) ** 0.5
    for _ in range(max_tries):
        p = float(rng.choice(real_mafs)) if (real_mafs is not None and real_mafs.size) else float(rng.beta(0.7, 3.0))
        if p >= p_min:
            return p
    return float(np.clip(np.median(real_mafs) if real_mafs is not None else 0.10, p_min, 0.49))

# ------------------------------ discretization ----------------------------

def hwe_thresholds_for_maf(p):
    """Compute latent-N(0,1) thresholds that yield HWE genotype probs for target MAF p."""
    t0 = norm.ppf((1.0 - p) ** 2)
    t1 = norm.ppf((1.0 - p) ** 2 + 2.0 * p * (1.0 - p))
    return t0, t1

def discretize_latent_to_genotype(z, maf):
    """Discretize a latent standard normal into {0,1,2} using HWE thresholds for MAF=maf."""
    t0, t1 = hwe_thresholds_for_maf(maf)
    g = np.zeros_like(z, dtype=np.int8)
    g[z > t0] = 1
    g[z > t1] = 2
    return g

def correlate_with_target(y_std, rho, rng):
    """
    Construct latent z approximately equal to rho * y + sqrt(1 - rho^2) * e, then re-standardize.
    Used to induce target correlation before HWE discretization.
    """
    y = (y_std - y_std.mean()) / (y_std.std() + 1e-12)
    e = rng.standard_normal(y.shape[0])
    e = (e - e.mean()) / (e.std() + 1e-12)
    z = rho * y + np.sqrt(max(1e-8, 1 - rho**2)) * e
    return (z - z.mean()) / (z.std() + 1e-12)

# ------------------------------ shape guards ------------------------------

def add_shape_ok(g, y, thresh=0.30):
    """
    Additive guard: means should move roughly linearly across genotype classes 0, 1, 2.
    Compare differences m1 - m0 and m2 - m1 and bound their relative deviation.
    """
    m0 = y[g == 0].mean()
    m1 = y[g == 1].mean()
    m2 = y[g == 2].mean()
    d01 = m1 - m0
    d12 = m2 - m1
    rel_dev = abs(d12 - d01) / max(abs(d01), abs(d12), 1e-12)
    return bool(rel_dev <= thresh)

def dom_shape_ok(g, y, tol_equal=0.25, tol_step=0.002):
    """
    Dominant guard: m1 and m2 should be similar and both separated from m0 by a small step.
    This avoids accidental additive-like shapes.
    """
    m0 = y[g==0].mean() if (g==0).any() else 0.0
    m1 = y[g==1].mean() if (g==1).any() else 0.0
    m2 = y[g==2].mean() if (g==2).any() else 0.0
    eq = abs(m2 - m1) / max(abs(m1), abs(m2), 1e-12)
    step = (m1 + m2)/2.0 - m0
    return (eq <= tol_equal) and (abs(step) >= tol_step)

def rec_shape_ok(g, y, tol_equal=0.25, tol_step=0.002):
    """
    Recessive guard: m0 and m1 should be similar and m2 separated by a small step.
    This avoids accidental dominant/additive shapes.
    """
    m0 = y[g==0].mean() if (g==0).any() else 0.0
    m1 = y[g==1].mean() if (g==1).any() else 0.0
    m2 = y[g==2].mean() if (g==2).any() else 0.0
    eq = abs(m1 - m0) / max(abs(m0), abs(m1), 1e-12)
    step = m2 - (m0 + m1)/2.0
    return (eq <= tol_equal) and (abs(step) >= tol_step)

# ------------------------------ encodings & stats -------------------------

def epi_corr(a, b):
    """Plain Pearson correlation with defensive centering and scaling."""
    a = a - a.mean()
    b = b - b.mean()
    return float(np.dot(a, b) / (len(a) * (a.std() + 1e-12) * (b.std() + 1e-12)))

def corr(y, x):
    """Pearson correlation utility (used for observed main-effect checks)."""
    y0 = y - y.mean()
    x0 = x - x.mean()
    return float(np.dot(y0, x0) / (len(y0) * (y0.std() + 1e-12) * (x0.std() + 1e-12)))

def obs_stats(y, g):
    """Return genotype counts and observed MAF for book-keeping in the manifest."""
    n0 = int((g == 0).sum())
    n1 = int((g == 1).sum())
    n2 = int((g == 2).sum())
    maf_obs = (n1 + 2 * n2) / (2.0 * len(g))
    return n0, n1, n2, float(maf_obs)

def carrier_frac(n1, n2, n):
    """P(g>0) under the realized sample—useful for dominant sanity checks."""
    return (n1 + n2) / float(max(1, n))

def hom_alt_frac(n2, n):
    """P(g==2) under the realized sample—useful for recessive sanity checks."""
    return n2 / float(max(1, n))

# ------------------------------ builders ---------------------------------

def make_dominant_snp_strict(y_std, maf, rho, rng):
    """
    Build g so that C = 1[g>0] has correlation approximately equal to rho with y, while among carriers
    the 1 vs 2 split follows HWE and is about independent of y.
    """
    n = y_std.shape[0]
    p0 = (1 - maf) ** 2
    p1 = 2 * maf * (1 - maf)
    p2 = maf ** 2
    pc = 1 - p0
    z = correlate_with_target(y_std, rho, rng)
    t = norm.ppf(1 - pc)
    C = (z > t).astype(np.int8)
    prob1 = p1 / max(p1 + p2, 1e-12)
    u = rng.random(n)
    g = np.zeros(n, dtype=np.int8)
    mask = C == 1
    g[mask] = (u[mask] > prob1).astype(np.int8) + 1
    return g

def make_recessive_snp_strict(y_std, maf, rho, rng):
    """
    Build g so that R = 1[g==2] has correlation approximately equal to rho with y, while among R==0
    the 0 vs 1 split follows HWE and is about independent of y.
    """
    n = y_std.shape[0]
    p0 = (1 - maf) ** 2
    p1 = 2 * maf * (1 - maf)
    p2 = maf ** 2
    z = correlate_with_target(y_std, rho, rng)
    t = norm.ppf(1 - p2)
    R = (z > t).astype(np.int8)
    prob1 = p1 / max(p0 + p1, 1e-12)
    u = rng.random(n)
    g = np.zeros(n, dtype=np.int8)
    notR = R == 0
    g[notR] = (u[notR] < prob1).astype(np.int8)
    g[R == 1] = 2
    return g

def resolve_epi_controls(epi_controls):
    """Merge user-provided epistasis controls with defaults."""
    defaults = {"main_abs_cap": 0.02, "rho_tol": 0.01, 
                "max_alpha_iters": 16, "max_resamples_per_alpha": 6}
    if not epi_controls:
        return defaults
    out = dict(defaults)
    for k, v in epi_controls.items():
        if k in out and v is not None:
            out[k] = v
    return out

def make_epistatic_pair_strict(
    y_std,
    maf_a,
    maf_b,
    rho_target,
    rng,
    main_abs_cap=0.02,
    rho_tol=0.01,
    max_alpha_iters=16,
    max_resamples_per_alpha=6,
):
    """
    Construct an epistatic pair (gA, gB) where:
      - corr(y, gA) and corr(y, gB) are kept small (<= main_abs_cap),
      - the interaction H = (gA - 2*pA) * (gB - 2*pB) achieves abs(corr(y, H)) close to abs(rho_target).
    Bisection over alpha scales the y*u term to hit the interaction target.
    """
    y = (y_std - y_std.mean()) / (y_std.std() + 1e-12)
    n = y.shape[0]
    sign = 1.0 if rho_target >= 0 else -1.0
    rho_mag = abs(rho_target)

    lo, hi = 0.0, 0.95
    best = None
    for _ in range(max_alpha_iters):
        alpha = 0.5 * (lo + hi)
        chosen = None
        for _try in range(max_resamples_per_alpha):
            u = rng.standard_normal(n); u = (u - u.mean()) / (u.std() + 1e-12)
            e = rng.standard_normal(n); e = (e - e.mean()) / (e.std() + 1e-12)
            v = sign * alpha * (y * u) + np.sqrt(max(1.0 - alpha**2, 1e-8)) * e
            u = (u - u.mean()) / (u.std() + 1e-12)
            v = (v - v.mean()) / (v.std() + 1e-12)

            gA = discretize_latent_to_genotype(u, maf_a)
            gB = discretize_latent_to_genotype(v, maf_b)

            rA = epi_corr(y, gA.astype(float))
            rB = epi_corr(y, gB.astype(float))
            if abs(rA) > main_abs_cap or abs(rB) > main_abs_cap:
                continue

            H = (gA.astype(float) - 2.0 * maf_a) * (gB.astype(float) - 2.0 * maf_b)
            rH = epi_corr(y, H)
            err = abs(rH - rho_mag)
            cand = (err, gA, gB, rA, rB, rH, alpha)
            if chosen is None or err < chosen[0]:
                chosen = cand
            if err <= rho_tol:
                break

        if chosen is None:
            hi = alpha
            continue

        err, gA, gB, rA, rB, rH, alpha = chosen
        if rH < rho_mag:
            lo = alpha
        else:
            hi = alpha

        if err <= rho_tol:
            best = chosen
            break
        if (best is None) or (err < best[0]):
            best = chosen

    if best is None:
        alpha = 0.05
        u = rng.standard_normal(n)
        u = (u - u.mean()) / (u.std() + 1e-12)
        e = rng.standard_normal(n)
        e = (e - e.mean()) / (e.std() + 1e-12)
        v = sign * alpha * (y * u) + np.sqrt(max(1.0 - alpha**2, 1e-8)) * e
        u = (u - u.mean()) / (u.std() + 1e-12)
        v = (v - v.mean()) / (v.std() + 1e-12)
        gA = discretize_latent_to_genotype(u, maf_a)
        gB = discretize_latent_to_genotype(v, maf_b)
        rA = epi_corr(y, gA.astype(float))
        rB = epi_corr(y, gB.astype(float))
        H = (gA.astype(float) - 2.0 * maf_a) * (gB.astype(float) - 2.0 * maf_b)
        rH = epi_corr(y, H)
        return gA.astype(np.int8), gB.astype(np.int8), rA, rB, rH, alpha

    _, gA, gB, rA, rB, rH, alpha = best
    return gA.astype(np.int8), gB.astype(np.int8), rA, rB, rH, alpha

# ------------------------------ 
#  Main Sim Generator 
# ------------------------------

def generate_synthetic_variants(
    y_std,
    n_add=100,
    n_nonlin=100,
    real_mafs=None,
    rng_seed=42,
    add_rho_range=(0.05, 0.15),
    nonlin_design=None,  # {"dominant":40,"recessive":40,"epistatic":20}
    dom_rho_range=(0.06, 0.12),
    rec_rho_range=(0.06, 0.12),
    epi_rho_range=(0.06, 0.12),
    add_min_n22=1000,
    add_dev_max=0.30,
    add_max_tries=8,
    maf_ranges=None,
    min_max_dom_carrier_frac=(0.05, 0.10),
    min_max_rec_hom_n=(50, 80),
    dom_shape_max_tries=10,
    rec_shape_max_tries=10,
    epi_cfg=None,
):
    """
    Main entry: constructs additive + nonlinear (dominant, recessive, epistatic) variants.
    Emits:
      syn_X: (n, total_syn) int8; manifest: per-variant design + observed stats for QC.
    """
    if maf_ranges is None:
        maf_ranges = {"dominant": (0.02, 0.08), "recessive": (0.01, 0.05), "epistatic": (0.01, 0.10)}

    rng = np.random.default_rng(rng_seed)
    n = y_std.shape[0]
    check_y_is_standardized(y_std)

    if nonlin_design is None:
        nonlin_design = {"dominant": 40, "recessive": 40, "epistatic": 20}
    assert sum([nonlin_design["dominant"], nonlin_design["recessive"], 2 * nonlin_design["epistatic"]]) == n_nonlin, \
        "n_nonlin must equal dominant + recessive + 2*epistatic"

    syn_cols = []
    rows = []

    # Additive
    add_rhos = np.sign(np.random.default_rng(rng_seed + 11).uniform(-1, 1, size=n_add)) * \
               np.random.default_rng(rng_seed + 13).uniform(*add_rho_range, size=n_add)
    for k, rho in enumerate(add_rhos, start=1):
        p = draw_add_maf(real_mafs, n, rng, n22_min=add_min_n22)
        g = None
        for _ in range(add_max_tries):
            z = correlate_with_target(y_std, float(rho), rng)
            g_try = discretize_latent_to_genotype(z, float(p))
            if add_shape_ok(g_try, y_std, thresh=add_dev_max):
                g = g_try; break
            g = g_try
        syn_cols.append(g.astype(np.int8))
        n0, n1, n2, maf_observed = obs_stats(y_std, g)
        rho_observed = corr(y_std, g.astype(float))
        rows.append({
            "variant_id": f"SYN_ADD_{k:03d}",
            "effect_type": "additive",
            "maf_target": float(p),
            "rho_target": float(rho),
            "effect_sign": "pos" if rho >= 0 else "neg",
            "nonlinear_def": "additive(g)",
            "base_variants": "",
            "maf_observed": maf_observed,
            "rho_main_observed": rho_observed,
            "n0": n0, "n1": n1, "n2": n2,
        })

    # Dominant
    low, high = maf_ranges["dominant"]
    dom_min, dom_max = min_max_dom_carrier_frac
    for k in range(nonlin_design["dominant"]):
        for _ in range(100):
            p = sample_maf_in_range(real_mafs, low, high, rng)
            carriers = 2 * p * (1 - p) + p * p
            if dom_min <= carriers <= dom_max:
                break
        else:
            target = np.clip(dom_min, 1e-8, dom_max - 1e-8)
            p = 1.0 - np.sqrt(max(0.0, 1.0 - target))
            p = float(np.clip(p, low, min(high, 0.49)))
        rho_mag = float(np.random.default_rng(rng_seed + 21).uniform(*dom_rho_range))
        rho = rho_mag if rng.random() < 0.5 else -rho_mag
        g = None
        for _ in range(dom_shape_max_tries):
            g_try = make_dominant_snp_strict(y_std, p, rho, rng)
            if dom_shape_ok(g_try, y_std):
                g = g_try; break
            g = g_try
        syn_cols.append(g.astype(np.int8))
        enc = (g > 0).astype(float)
        rho_observed = corr(y_std, enc)
        n0, n1, n2, maf_observed = obs_stats(y_std, g)
        rows.append({
            "variant_id": f"SYN_DOM_{k+1:03d}",
            "effect_type": "dominant",
            "maf_target": p,
            "rho_target": rho,
            "effect_sign": "pos" if rho >= 0 else "neg",
            "nonlinear_def": "dominant(g)",
            "base_variants": "",
            "maf_observed": maf_observed,
            "rho_main_observed": rho_observed,
            "n0": n0, "n1": n1, "n2": n2,
            "carrier_frac_observed": carrier_frac(n1, n2, len(g)),
        })

    # Recessive
    N = y_std.shape[0]
    low, high = maf_ranges["recessive"]
    rec_min, rec_max = min_max_rec_hom_n
    for k in range(nonlin_design["recessive"]):
        for _ in range(100):
            p = sample_maf_in_range(real_mafs, low, high, rng)
            n22 = int(round(N * p * p))
            if rec_min <= n22 <= rec_max:
                break
        else:
            target_n = np.clip(rec_min, 1, rec_max)
            p = np.sqrt(target_n / max(1, N))
            p = float(np.clip(p, low, min(high, 0.49)))
        rho_mag = float(np.random.default_rng(rng_seed + 31).uniform(*rec_rho_range))
        rho = rho_mag if rng.random() < 0.5 else -rho_mag
        g = None
        for _ in range(rec_shape_max_tries):
            g_try = make_recessive_snp_strict(y_std, p, rho, rng)
            if rec_shape_ok(g_try, y_std):
                g = g_try; break
            g = g_try
        syn_cols.append(g.astype(np.int8))
        enc = (g == 2).astype(float)
        rho_observed = corr(y_std, enc)
        n0, n1, n2, maf_observed = obs_stats(y_std, g)
        rows.append({
            "variant_id": f"SYN_REC_{k+1:03d}",
            "effect_type": "recessive",
            "maf_target": p,
            "rho_target": rho,
            "effect_sign": "pos" if rho >= 0 else "neg",
            "nonlinear_def": "recessive(g)",
            "base_variants": "",
            "maf_observed": maf_observed,
            "rho_main_observed": rho_observed,
            "n0": n0, "n1": n1, "n2": n2,
            "hom_alt_frac_observed": hom_alt_frac(n2, len(g)),
        })

    # Epistatic
    epi_pairs = nonlin_design["epistatic"]
    low, high = maf_ranges["epistatic"]
    epi_control = resolve_epi_controls(epi_cfg)
    for k in range(epi_pairs):
        pA = sample_maf_in_range(real_mafs, low, high, rng)
        pB = sample_maf_in_range(real_mafs, low, high, rng)
        rho_mag = float(np.random.default_rng(rng_seed + 41).uniform(*epi_rho_range))
        rho_epi = rho_mag if rng.random() < 0.5 else -rho_mag
        gA, gB, rA, rB, rH, alpha = make_epistatic_pair_strict(
            y_std, pA, pB, rho_epi, rng,
            main_abs_cap=epi_control["main_abs_cap"],
            rho_tol=epi_control["rho_tol"],
            max_alpha_iters=epi_control["max_alpha_iters"],
            max_resamples_per_alpha=epi_control["max_resamples_per_alpha"],
        )
        n0A, n1A, n2A, maf_obs_A = obs_stats(y_std, gA)
        n0B, n1B, n2B, maf_obs_B = obs_stats(y_std, gB)
        pair_id = f"EPI_{k+1:03d}"
        syn_cols.append(gA)
        rows.append({
            "variant_id": f"SYN_EPI_{k+1:03d}_A",
            "effect_type": "epistatic",
            "pair_id": pair_id,
            "maf_target": pA,
            "maf_observed": maf_obs_A,
            "rho_target": 0.0,
            "rho_main_observed": rA,
            "interaction_target_rho": rho_epi,
            "achieved_r_interaction": rH,
            "interaction_r_diff": rH - rho_epi,
            "alpha_used": alpha,
            "effect_sign": "n/a",
            "nonlinear_def": "H=(gA-2pA)*(gB-2pB)",
            "base_variants": f"SYN_EPI_{k+1:03d}_B",
            "n0": n0A, "n1": n1A, "n2": n2A,
        })
        syn_cols.append(gB)
        rows.append({
            "variant_id": f"SYN_EPI_{k+1:03d}_B",
            "effect_type": "epistatic",
            "pair_id": pair_id,
            "maf_target": pB,
            "maf_observed": maf_obs_B,
            "rho_target": 0.0,
            "rho_main_observed": rB,
            "interaction_target_rho": rho_epi,
            "achieved_r_interaction": rH,
            "interaction_r_diff": rH - rho_epi,
            "alpha_used": alpha,
            "effect_sign": "n/a",
            "nonlinear_def": "H=(gA-2pA)*(gB-2pB)",
            "base_variants": f"SYN_EPI_{k+1:03d}_A",
            "n0": n0B, "n1": n1B, "n2": n2B,
        })

    syn_X = np.column_stack(syn_cols).astype(np.int8) if syn_cols else np.empty((n, 0), dtype=np.int8)
    manifest = pd.DataFrame(rows)
    return syn_X, manifest

# ------------------------------ I/O helpers -------------------------------
def read_table_auto(path):
    """
    1) try whitespace-delimited (works for space or tab),
    2) fall back to pandas' delimiter sniffer,
    3) finally try comma.
    """
    path = str(path)
    try:
        return pd.read_csv(path, sep=r"\s+")
    except Exception:
        try:
            return pd.read_csv(path, sep=None, engine="python")
        except Exception:
            return pd.read_csv(path)

def load_fam_iids(plink_prefix):
    fam_path = f"{plink_prefix}.fam"
    fam = pd.read_csv(
        fam_path, sep=r"\s+", header=None,
        names=["FID","IID","PAT","MAT","SEX","PHENO"], dtype={0:str,1:str}
    )
    fam["FID"] = fam["FID"].astype(str)
    fam["IID"] = fam["IID"].astype(str)
    return fam[["FID","IID"]].copy()


def load_y_and_iids_from_plink_tables(phenotypes_path, pheno_col,
                                      fam_prefix, standardize=False):
    """
    Build y and IID list from a PLINK-style phenotype table (FID, IID, ...),
    and align to the .fam order from fam_prefix. Rows with missing phenotype
    are dropped, and the IID order follows the .fam among the remaining IIDs.
    """
    # phenotypes table
    ph = read_table_auto(phenotypes_path)
    ph.columns = [c.strip() for c in ph.columns]
    name_map = {c.lower(): c for c in ph.columns}
    # if "fid" not in name_map or "iid" not in name_map:
    if "fid" not in name_map or "iid" not in name_map:
        raise ValueError("Phenotype table must include FID and IID columns.")
    FID, IID = name_map["fid"], name_map["iid"]
    if pheno_col not in ph.columns:
        raise ValueError(f"Phenotype column '{pheno_col}' not found in {phenotypes_path}.")
    ph = ph[[FID, IID, pheno_col]].copy()
    ph[FID] = ph[FID].astype(str)
    ph[IID] = ph[IID].astype(str)
    ph = ph[ph[pheno_col].notna()].copy()

    # fam order
    fam_ids = load_fam_iids(fam_prefix)  # FID, IID
    merged = fam_ids.merge(ph, on=[FID, IID], how="inner")  # keeps fam order, filters to available y

    if merged.empty:
        raise ValueError("No overlapping IIDs between .fam and phenotype table with non-missing labels.")

    y = merged[pheno_col].astype(float).to_numpy()
    iids = merged["IID"].astype(str).tolist()

    if standardize:
        mu, sd = float(np.nanmean(y)), float(np.nanstd(y))
        sd = sd if sd > 0 else 1.0
        y = (y - mu) / sd
        
    check_y_is_standardized(y)
    return y, iids

def load_vector(path):
    """Load 1D vector from .npy, .txt, or .csv; squeeze singleton dims."""
    ext = Path(path).suffix.lower()
    if ext == ".npy":
        arr = np.load(path)
    else:
        sep = "," if ext == ".csv" else None
        arr = np.loadtxt(path, delimiter=sep)
    return arr.squeeze()

def write_run_config_yaml(out_prefix, cfg):
    """Write YAML run configuration capturing parameters and file paths."""
    path = f"{out_prefix}.run_config.yaml"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"[OK] Wrote run config: {path}")
    return path

def write_syn_csv(iids, syn_X, manifest, out_prefix):
    """Write IID-aligned synthetic genotypes as CSV (plus separate manifest CSV elsewhere)."""
    df = pd.DataFrame(syn_X, index=iids, columns=manifest["variant_id"].tolist())
    df.insert(0, "IID", iids)
    geno_csv = f"{out_prefix}.syn_genotypes.csv"
    Path(geno_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(geno_csv, index=False)
    print(f"[OK] Wrote: {geno_csv}")

def write_vcf(iids, syn_X, manifest, out_vcf, chrom="26", 
              start_pos=1_000_000, step=10):
    """Write a minimal VCF in the given IID order (no PLINK FAM alignment)."""
    Path(out_vcf).parent.mkdir(parents=True, exist_ok=True)
    with open(out_vcf, "w") as f:
        f.write("##fileformat=VCFv4.2\n")
        f.write(f"##contig=<ID={chrom}>\n")
        hdr = "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(iids) + "\n"
        f.write(hdr)
        pos = start_pos
        for j, vid in enumerate(manifest["variant_id"]):
            g = syn_X[:, j]
            calls = np.where(g == 0, "0/0", np.where(g == 1, "0/1", "1/1"))
            f.write(f"{chrom}\t{pos}\t{vid}\tA\tC\t.\tPASS\t.\tGT\t" + "\t".join(calls.tolist()) + "\n")
            pos += step
    print(f"[OK] Wrote VCF: {out_vcf}")

def write_syn_vcf_aligned_to_fam(
    real_bed_prefix,
    syn_X,
    manifest,
    sample_ids,
    out_vcf,
    chrom="26",
    start_pos=1_000_000,
    step=10,
):
    """
    Write a hardcall VCF (GT only) aligned to the .fam order from real_bed_prefix.
    Only IIDs present in both the .fam and sample_ids are written, preserving .fam order.
    """
    fam_path = real_bed_prefix + ".fam"
    assert Path(fam_path).is_file(), f"Real genotypes .fam not found: {fam_path}"
    fam = pd.read_csv(
        fam_path, sep=r"\s+", header=None,
        names=["FID","IID","PAT","MAT","SEX","PHENO"], dtype={0:str,1:str}
    )
    fam["IID"] = fam["IID"].astype(str)
    sample_ids = np.array(sample_ids).astype(str)

    # intersection in fam order
    sample_set = set(sample_ids.tolist())
    fam_sub = fam[fam["IID"].isin(sample_set)].copy()
    if fam_sub.empty:
        raise ValueError("No intersecting IIDs between .fam and provided sample_ids.")
    iid_to_row = {iid: i for i, iid in enumerate(sample_ids)}
    row_idx = np.array([iid_to_row[iid] for iid in fam_sub["IID"]], dtype=np.int64)
    G = syn_X[row_idx, :].astype(np.int8)
    vids = manifest["variant_id"].tolist()
    Path(out_vcf).parent.mkdir(parents=True, exist_ok=True)
    with open(out_vcf, "w") as f:
        f.write("##fileformat=VCFv4.2\n")
        f.write(f"##contig=<ID={chrom}>\n")
        header = "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(fam_sub["IID"]) + "\n"
        f.write(header)
        pos = start_pos
        for j, vid in enumerate(vids):
            f.write(f"{chrom}\t{pos}\t{vid}\tA\tC\t.\tPASS\t.\tGT")
            col = G[:, j]
            calls = np.where(col == 0, "0/0", np.where(col == 1, "0/1", "1/1"))
            f.write("\t" + "\t".join(calls.tolist()) + "\n")
            pos += step
    print(f"[OK] Wrote synthetic VCF aligned to fam: {out_vcf}")


def load_allele_freqs(path):
    """Load allele frequencies from PLINK2 .afreq file"""
    df = pd.read_csv(path, sep="\t")
    afreq = df["A1_FREQ"].astype(float).to_numpy()
    mafs = np.minimum(afreq, 1.0 - afreq)
    mafs = mafs[np.isfinite(mafs)]
    mafs = mafs[(mafs > 0.0) & (mafs < 0.5)]
    return mafs

def validate_config(cfg):
    """
    Validate and normalize spike-in config dict
    - Ensures required keys exist.
    - Backfills defaults if not provided.
    - Normalizes structures to the shapes expected by the generator.
    """
    if "inputs" not in cfg:
        raise ValueError("Missing 'inputs' in config.")
    req = ["phenotypes", "pheno_col", "genotypes"]
    for k in req:
        if k not in cfg["inputs"]:
            raise ValueError(f"Missing inputs.{k} in config.")

    cfg.setdefault("meta", {})
    cfg["meta"].setdefault("rng_seed", cfg["meta"].get("seed", 42))
    cfg["meta"].setdefault("mode", "sim")
    cfg["meta"].setdefault("seed", cfg["meta"].get("rng_seed", 42))

    cfg.setdefault("design", {})
    d = cfg["design"]
    d.setdefault("n_add", 100)
    d.setdefault("n_nonlin", 100)
    d.setdefault("nonlin_design", {"dominant": 40, "recessive": 40, "epistatic": 20})

    cfg.setdefault("ranges", {})
    r = cfg["ranges"]
    r.setdefault("add_rho_range", [0.05, 0.15])
    r.setdefault("dom_rho_range", [0.06, 0.12])
    r.setdefault("rec_rho_range", [0.06, 0.12])
    r.setdefault("epi_rho_range", [0.06, 0.12])
    r.setdefault("maf_ranges", {"dominant": [0.02, 0.08], "recessive": [0.01, 0.05], "epistatic": [0.01, 0.10]})

    cfg.setdefault("controls", {})
    c = cfg["controls"]
    c.setdefault("additive", {"add_min_n22": 1000, "add_dev_max": 0.30, "add_max_tries": 8})
    c.setdefault("dominant", {"min_max_dom_carrier_frac": [0.05, 0.10], "dom_shape_max_tries": 10})
    c.setdefault("recessive", {"min_max_rec_hom_n": [50, 80], "rec_shape_max_tries": 10})
    c.setdefault("epistatic", {"main_abs_cap": 0.02, "rho_tol": 0.01, "max_alpha_iters": 16, "max_resamples_per_alpha": 6})

    cfg.setdefault("outputs", {})
    o = cfg["outputs"]
    if "out_root" not in o and "out_prefix" not in o:
        raise ValueError("outputs.out_root or outputs.out_prefix must be provided.")
    o.setdefault("write_csv", True)
    o.setdefault("write_vcf", True)
    o.setdefault("vcf_chrom", "26")
    o.setdefault("vcf_start", 1_000_000)
    o.setdefault("vcf_step", 10)

    nd = d["nonlin_design"]
    expected = int(nd.get("dominant", 0)) + int(nd.get("recessive", 0)) + 2 * int(nd.get("epistatic", 0))
    if expected != int(d["n_nonlin"]):
        raise ValueError(f"design.n_nonlin ({d['n_nonlin']}) must equal dominant + recessive + 2*epistatic ({expected}).")
    return cfg


def build_out_prefix(outputs, meta, design):
    if "out_root" in outputs and outputs["out_root"]:
        out_root = Path(outputs["out_root"]).resolve()
        ndom = int(design["nonlin_design"].get("dominant", 0))
        nrec = int(design["nonlin_design"].get("recessive", 0))
        nepi = int(design["nonlin_design"].get("epistatic", 0))
        dirtag = f"add{design['n_add']}_dom{ndom}_rec{nrec}"
        if nepi > 0:
            dirtag += f"_epi{int(2*nepi)}"
        dirtag += f"_seed{meta.get('seed', meta.get('rng_seed', 42))}"
        dirname = f"spikein_{meta.get('mode','sim')}_{dirtag}"
        out_dir = out_root / dirname
        out_prefix = str(out_dir / "simset")
        print(f"New directory for spike-in genotypes: {out_dir}")
    else:
        out_prefix = outputs["out_prefix"]
        Path(out_prefix).parent.mkdir(parents=True, exist_ok=True)
    return out_prefix

# ------------------------------ CLI --------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Synthetic SNP spike-in simulator (config-driven).")
    ap.add_argument("--config", required=True, help="Path to spikein_config.yml")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    cfg = validate_config(cfg)

    # Required inputs (always present now)
    phenos_path = cfg["inputs"]["phenotypes"]
    pheno_col = cfg["inputs"]["pheno_col"]
    plink_prefix = cfg["inputs"]["genotypes"]

    # y, IIDs aligned to .fam (subset to samples with non-missing y)
    y, iids = load_y_and_iids_from_plink_tables(
        phenotypes_path=phenos_path,
        pheno_col=pheno_col,
        fam_prefix=plink_prefix,
        standardize=bool(cfg["inputs"].get("standardize", False)),
    )
    # precomputed MAFs (optional): allow .afreq/.frq/.npy/.csv/.txt
    real_mafs = None
    if cfg["inputs"].get("real_mafs_path"):
        real_mafs = load_allele_freqs(cfg["inputs"]["real_mafs_path"])
    
    # build out_prefix
    out_prefix = build_out_prefix(cfg["outputs"], cfg["meta"], cfg["design"])
    
    # generate synthetic variants
    syn_X, manifest = generate_synthetic_variants(
        y_std=y,
        n_add=cfg["design"]["n_add"],
        n_nonlin=cfg["design"]["n_nonlin"],
        real_mafs=real_mafs,
        rng_seed=cfg["meta"]["rng_seed"],
        add_rho_range=tuple(cfg["ranges"]["add_rho_range"]),
        nonlin_design=cfg["design"]["nonlin_design"],
        dom_rho_range=tuple(cfg["ranges"]["dom_rho_range"]),
        rec_rho_range=tuple(cfg["ranges"]["rec_rho_range"]),
        epi_rho_range=tuple(cfg["ranges"]["epi_rho_range"]),
        add_min_n22=cfg["controls"]["additive"]["add_min_n22"],
        add_dev_max=cfg["controls"]["additive"]["add_dev_max"],
        add_max_tries=cfg["controls"]["additive"]["add_max_tries"],
        maf_ranges={
            "dominant": tuple(cfg["ranges"]["maf_ranges"]["dominant"]),
            "recessive": tuple(cfg["ranges"]["maf_ranges"]["recessive"]),
            "epistatic": tuple(cfg["ranges"]["maf_ranges"]["epistatic"]),
        },
        min_max_dom_carrier_frac=tuple(cfg["controls"]["dominant"]["min_max_dom_carrier_frac"]),
        min_max_rec_hom_n=tuple(cfg["controls"]["recessive"]["min_max_rec_hom_n"]),
        dom_shape_max_tries=cfg["controls"]["dominant"]["dom_shape_max_tries"],
        rec_shape_max_tries=cfg["controls"]["recessive"]["rec_shape_max_tries"],
        epi_cfg=cfg["controls"]["epistatic"],
    )

    # Persist run-config snapshot
    write_run_config_yaml(out_prefix, cfg)

    # manifest csv
    manifest_out = f"{out_prefix}.manifest.csv"
    Path(manifest_out).parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_out, index=False)
    print(f"[OK] Wrote manifest CSV: {manifest_out}")

    # iid-aligned csv (these iids are already in .fam order subset)
    if cfg["outputs"].get("write_csv", True):
        write_syn_csv(iids, syn_X, manifest, out_prefix)

    # always write fam-aligned VCF using the provided genotypes prefix
    if cfg["outputs"].get("write_vcf", True):
        vcf_out = f"{out_prefix}.syn.vcf"
        chrom = cfg["outputs"].get("vcf_chrom", "26")
        start_pos = int(cfg["outputs"].get("vcf_start", 1_000_000))
        step = int(cfg["outputs"].get("vcf_step", 10))

        assert (Path(plink_prefix + ".bed").is_file()
                and Path(plink_prefix + ".fam").is_file()
                and Path(plink_prefix + ".bim").is_file()), \
            f"Real genotypes BIM/FAM/BED not found: {plink_prefix}"
        
        write_syn_vcf_aligned_to_fam(
            real_bed_prefix=str(plink_prefix),
            syn_X=syn_X,
            manifest=manifest,
            sample_ids=np.array(iids, dtype=object),
            out_vcf=vcf_out,
            chrom=chrom,
            start_pos=start_pos,
            step=step,
        )


if __name__ == "__main__":
    main()
