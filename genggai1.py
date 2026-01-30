# -*- coding: utf-8 -*-
"""
ARL Benchmarking (Top-journal validation version)
- 20 benchmark functions
- Baselines: PSO / SA (budget-matched) / SSA
- Framework: ARL (full) + ablations via flags (no new algorithms)
- Outputs:
  * per function: convergence (median + IQR), boxplot (final fitness)
  * per function: diagnostics for ARL (step stats, density stats, clip saturation)
  * summary: mean±std, median, IQR; Friedman ranks; Wilcoxon (paired) + Holm; effect sizes
- Robust PNG saving (Agg backend), reproducible seeds, strict output structure

Dependencies: numpy, pandas, matplotlib, openpyxl (for xlsx)
Optional: scipy (for p-values; script works without scipy, but p-values become "NA")
"""

import os
import argparse
import math
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # critical: guarantees PNG generation in headless environments
import matplotlib.pyplot as plt

# ---------------- Optional SciPy (p-values) ----------------
try:
    from scipy.stats import wilcoxon, chi2
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

# ---------------- Global style ----------------
plt.rcParams["font.family"] = "Arial"
plt.rcParams["axes.unicode_minus"] = False

# ---------------- Algorithms order ----------------
# Remove ACO as per your request; keep common baselines
ALGO_ORDER = ["ARL", "ARL_noDensity", "ARL_noPCA", "ARL_noLeader", "PSO", "SA", "SSA"]

# ---------------- 20 benchmark functions ----------------
def Sphere(x): return float(np.sum(np.asarray(x)**2))

def High_Conditioned_Elliptic(x):
    x = np.asarray(x); d = len(x)
    if d <= 1: return float(np.sum(x**2))
    coeffs = (10**6)**(np.arange(d)/(d-1))
    return float(np.sum(coeffs * x**2))

def Discus(x): return float(1e6*np.asarray(x)[0]**2 + np.sum(np.asarray(x)[1:]**2))
def Bent_Cigar(x): return float(np.asarray(x)[0]**2 + 1e6*np.sum(np.asarray(x)[1:]**2))

def Ackley(x):
    x = np.asarray(x); d = len(x)
    return float(-20*np.exp(-0.2*np.sqrt(np.sum(x**2)/d))
                 - np.exp(np.sum(np.cos(2*np.pi*x))/d) + 20 + np.e)

def Rastrigin(x): return float(10*len(x) + np.sum(np.asarray(x)**2 - 10*np.cos(2*np.pi*np.asarray(x))))

def Griewank(x):
    x = np.asarray(x); d = len(x)
    return float(np.sum(x**2)/4000 - np.prod(np.cos(x/np.sqrt(np.arange(1, d+1)))) + 1)

def Michalewicz(x, m=10):
    x = np.clip(np.asarray(x), 0, np.pi)
    i = np.arange(1, len(x)+1)
    return float(-np.sum(np.sin(x) * (np.sin(i*x**2/np.pi))**(2*m)))

def Lunacek_Bi_Rastrigin(x):
    x = np.asarray(x); d = len(x)
    mu0 = 2.5; s = 1.0 - 1.0/(2*np.sqrt(d+20)-8.2)
    mu1 = -np.sqrt((mu0**2 - 1.0)/s)
    first = np.sum((x - mu0)**2); second = d + s*np.sum((x - mu1)**2)
    ras = np.sum(1 - np.cos(2*np.pi*(x - mu0)))
    return float(min(first, second) + 10*ras)

def Schwefel(x):
    x = np.asarray(x)
    return float(418.9829*len(x) - np.sum(x*np.sin(np.sqrt(np.abs(x)))))

def Diagonal_Valley(x): return float(np.sum((np.arange(1, len(x)+1))*np.asarray(x)**2))

def Linear_Slope(x):
    x = np.asarray(x).flatten()
    coeffs = np.arange(1, len(x)+1)
    return float(np.sum(coeffs * x))

def Katsuura(x):
    x = np.asarray(x); d = len(x); prod = 1.0
    for i in range(d):
        s = 0.0
        for j in range(1, 33):
            s += np.abs(2**j*x[i] - np.round(2**j*x[i]))/2**j
        prod *= (1+(i+1)*s)**(10/(d**1.2))
    return float(prod*10.0/(d**2))

def Bent_Cigar_Hessian(x):
    x = np.asarray(x)
    return float(x[0]**2 + 1e6*np.sum([(i+1)*xi**2 for i,xi in enumerate(x[1:])]))

def HGBat(x):
    x = np.asarray(x); d=len(x); s1=np.sum(x**2); s2=np.sum(x)
    return float(np.sqrt(np.abs(s1**2-s2**2))+(0.5*s1+s2)/d+0.5)

def HappyCat(x, alpha=1/8):
    x = np.asarray(x); d=len(x); s1=np.sum(x**2)
    return float(((s1-d)**2)**alpha+(0.5*s1+np.sum(x))/d+0.5)

def Zakharov(x):
    x = np.asarray(x); i=np.arange(1,len(x)+1)
    s1=np.sum(x**2); s2=np.sum(0.5*i*x)
    return float(s1+s2**2+s2**4)

def Sum_Squares(x): return float(np.sum((np.arange(1,len(x)+1))*np.asarray(x)**2))

def Levy(x):
    x=np.asarray(x); w=1+(x-1)/4
    term1=np.sin(np.pi*w[0])**2
    term2=np.sum((w[:-1]-1)**2*(1+10*np.sin(np.pi*w[:-1]+1)**2))
    term3=(w[-1]-1)**2*(1+np.sin(2*np.pi*w[-1])**2)
    return float(term1+term2+term3)

def Schaffer_F6(x):
    x=np.asarray(x)
    if len(x)<2: return 0.0
    xi,xj=x[:-1],x[1:]
    num=np.sin(np.sqrt(xi**2+xj**2))**2-0.5
    den=(1+0.001*(xi**2+xj**2))**2
    return float(np.sum(0.5+num/den))

FUNCTIONS = {
 "Sphere": (Sphere, (-100,100)),
 "High_Conditioned_Elliptic": (High_Conditioned_Elliptic, (-100,100)),
 "Discus": (Discus, (-100,100)),
 "Bent_Cigar": (Bent_Cigar, (-100,100)),
 "Ackley": (Ackley, (-32.768,32.768)),
 "Rastrigin": (Rastrigin, (-5.12,5.12)),
 "Griewank": (Griewank, (-600,600)),
 "Michalewicz": (Michalewicz, (0,np.pi)),
 "Lunacek_Bi_Rastrigin": (Lunacek_Bi_Rastrigin, (-5,5)),
 "Schwefel": (Schwefel, (-500,500)),
 "Diagonal_Valley": (Diagonal_Valley, (-100,100)),
 "Linear_Slope": (Linear_Slope, (-5,5)),
 "Katsuura": (Katsuura, (-5,5)),
 "Bent_Cigar_Hessian": (Bent_Cigar_Hessian, (-100,100)),
 "HGBat": (HGBat, (-5,5)),
 "HappyCat": (HappyCat, (-5,5)),
 "Zakharov": (Zakharov, (-5,10)),
 "Sum_Squares": (Sum_Squares, (-10,10)),
 "Levy": (Levy, (-10,10)),
 "Schaffer_F6": (Schaffer_F6, (-100,100)),
}

# ---------------- Utility ----------------
def _mirror_reflect(x, lb, ub):
    x = np.array(x, dtype=float, copy=True)
    lb, ub = np.array(lb), np.array(ub)
    rng = ub - lb
    for j in range(x.shape[-1]):
        if rng[j] <= 0:
            x[..., j] = np.clip(x[..., j], lb[j], ub[j])
            continue
        y = x[..., j] - lb[j]
        y = np.mod(y, 2 * rng[j])
        y = np.where(y > rng[j], 2 * rng[j] - y, y)
        x[..., j] = lb[j] + y
    return x

def _pca_partition(pop, n_regions, use_pca=True, rng=None):
    """
    Returns:
      regions: (n,) region index
      density: (n_regions,) region density
      proj: (n,) 1D projection used for partitioning (for diagnostics)
    """
    n, d = pop.shape
    if rng is None:
        rng = np.random.default_rng(0)

    X = pop - pop.mean(axis=0, keepdims=True)

    if use_pca and d > 1:
        # 1st principal direction via SVD
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        pc1 = Vt[0]
        proj = X @ pc1
    else:
        # fallback: random projection (keeps "regionalisation" concept without PCA)
        v = rng.normal(size=(d,))
        v = v / (np.linalg.norm(v) + 1e-12)
        proj = X @ v

    edges = np.percentile(proj, np.linspace(0, 100, n_regions + 1))
    regions = np.searchsorted(edges, proj, side="right") - 1
    regions = np.clip(regions, 0, n_regions - 1)
    counts = np.bincount(regions, minlength=n_regions)
    density = counts / float(n)
    return regions, density, proj

def _safe_eval(func, X):
    vals = np.apply_along_axis(func, 1, X)
    return np.nan_to_num(vals, nan=1e10, posinf=1e10, neginf=1e10)

# ---------------- ARL (single function with ablation flags) ----------------
def ARL(
    func, dim, bounds, max_iter=30, n=30, seed=0,
    n_regions=10,
    p_region_update=0.7,
    use_pca=True,
    use_density=True,
    use_leader=True,
    step_clip=(0.05, 0.7)
):
    """
    ARL framework run.
    Returns:
      best_x, best_f, conv (len=max_iter)
      diagnostics dict (step stats, density stats, clip saturation)
    """
    rng = np.random.default_rng(seed)

    lb, ub = bounds
    lb = np.array([lb]*dim) if np.isscalar(lb) else np.array(lb, dtype=float)
    ub = np.array([ub]*dim) if np.isscalar(ub) else np.array(ub, dtype=float)

    pop = rng.uniform(lb, ub, size=(n, dim))
    fit = _safe_eval(func, pop)

    g_idx = int(np.argmin(fit))
    gbest = pop[g_idx].copy()
    gfit = float(fit[g_idx])

    conv = []
    step_mean, step_std = [], []
    dens_std, dens_entropy = [], []
    clip_lo_ratio, clip_hi_ratio = [], []

    lo, hi = step_clip

    for t in range(max_iter):
        regions, density, proj = _pca_partition(pop, n_regions=n_regions, use_pca=use_pca, rng=rng)

        # base schedule: 0.5 -> 0.05
        base_step = 0.5 * (1.0 - 0.9 * t / float(max_iter))

        # density modulation
        if use_density:
            mult = (1.2 - density[regions])
        else:
            mult = np.ones_like(regions, dtype=float)

        step = base_step * mult
        step_clipped = np.clip(step, lo, hi)

        # diagnostics: clip saturation
        clip_lo_ratio.append(float(np.mean(step <= lo)))
        clip_hi_ratio.append(float(np.mean(step >= hi)))
        step_mean.append(float(np.mean(step_clipped)))
        step_std.append(float(np.std(step_clipped)))

        # density diagnostics
        dens_std.append(float(np.std(density)))
        # entropy of region occupancy
        p = density + 1e-12
        dens_entropy.append(float(-np.sum(p * np.log(p))))

        # leaders (region best or global best)
        region_leader_idx = {}
        if use_leader:
            for r in range(n_regions):
                idx = np.where(regions == r)[0]
                if idx.size > 0:
                    r_best = idx[np.argmin(fit[idx])]
                    region_leader_idx[r] = int(r_best)

        new_pop = pop.copy()

        for i in range(n):
            r = int(regions[i])
            if rng.random() < p_region_update:
                if use_leader and (r in region_leader_idx) and (region_leader_idx[r] == i):
                    # leader: pull toward global best
                    coeff = rng.normal(loc=0.9, scale=0.1)
                    new_pop[i] = pop[i] + step_clipped[i] * coeff * (gbest - pop[i])
                else:
                    # follower / exploration: random reference
                    coeff = rng.uniform(-0.5, 1.5)
                    rand_point = rng.uniform(lb, ub, size=dim)
                    new_pop[i] = pop[i] + step_clipped[i] * coeff * (rand_point - pop[i])
            else:
                # isotropic random walk
                dirv = rng.normal(size=dim)
                dirv = dirv / (np.linalg.norm(dirv) + 1e-12)
                new_pop[i] = pop[i] + step_clipped[i] * dirv * (ub - lb) * 0.25

        new_pop = _mirror_reflect(new_pop, lb, ub)
        new_fit = _safe_eval(func, new_pop)

        # update global best
        g_idx2 = int(np.argmin(new_fit))
        if float(new_fit[g_idx2]) < gfit:
            gfit = float(new_fit[g_idx2])
            gbest = new_pop[g_idx2].copy()

        pop, fit = new_pop, new_fit
        conv.append(float(gfit))

    diag = {
        "step_mean": np.array(step_mean),
        "step_std": np.array(step_std),
        "dens_std": np.array(dens_std),
        "dens_entropy": np.array(dens_entropy),
        "clip_lo_ratio": np.array(clip_lo_ratio),
        "clip_hi_ratio": np.array(clip_hi_ratio),
    }
    return gbest, gfit, conv, diag

# ---------------- Baselines ----------------
def PSO(func, dim, bounds, max_iter=30, n=30, seed=0):
    rng = np.random.default_rng(seed)
    lb, ub = bounds
    lb = np.array([lb]*dim) if np.isscalar(lb) else np.array(lb, dtype=float)
    ub = np.array([ub]*dim) if np.isscalar(ub) else np.array(ub, dtype=float)

    pos = rng.uniform(lb, ub, (n, dim))
    vel = np.zeros((n, dim))
    fit = _safe_eval(func, pos)

    pbest = pos.copy()
    pfit = fit.copy()

    g_idx = int(np.argmin(fit))
    gbest = pos[g_idx].copy()
    gfit = float(fit[g_idx])

    conv = []
    for t in range(max_iter):
        w = 0.7; c1 = 1.5; c2 = 1.5
        r1 = rng.random((n, dim))
        r2 = rng.random((n, dim))
        vel = w*vel + c1*r1*(pbest - pos) + c2*r2*(gbest - pos)
        pos = pos + vel
        pos = _mirror_reflect(pos, lb, ub)

        fit = _safe_eval(func, pos)

        improved = fit < pfit
        pbest[improved] = pos[improved]
        pfit[improved] = fit[improved]

        g_idx = int(np.argmin(fit))
        if float(fit[g_idx]) < gfit:
            gfit = float(fit[g_idx])
            gbest = pos[g_idx].copy()
        conv.append(float(gfit))

    return gbest, gfit, conv

def SSA(func, dim, bounds, max_iter=30, n=30, seed=0):
    rng = np.random.default_rng(seed)
    lb, ub = bounds
    lb = np.array([lb]*dim) if np.isscalar(lb) else np.array(lb, dtype=float)
    ub = np.array([ub]*dim) if np.isscalar(ub) else np.array(ub, dtype=float)

    pos = rng.uniform(lb, ub, (n, dim))
    fit = _safe_eval(func, pos)
    g_idx = int(np.argmin(fit))
    gbest = pos[g_idx].copy()
    gfit = float(fit[g_idx])

    conv = []
    for t in range(max_iter):
        c1 = 2 * np.exp(-((4*t/max_iter)**2))
        # simple SSA leader-follower update
        for i in range(n):
            if i == 0:
                pos[i] = gbest + c1 * (rng.random(dim)*(ub - lb) + lb - gbest)
            else:
                pos[i] = (pos[i] + pos[i-1]) / 2.0

        pos = _mirror_reflect(pos, lb, ub)
        fit = _safe_eval(func, pos)

        g_idx = int(np.argmin(fit))
        if float(fit[g_idx]) < gfit:
            gfit = float(fit[g_idx])
            gbest = pos[g_idx].copy()

        conv.append(float(gfit))

    return gbest, gfit, conv

def SA_budget_matched(func, dim, bounds, max_iter=30, n=30, seed=0):
    """
    Budget-matched SA:
    Pop-based methods evaluate ~ n candidates per iteration.
    Here, one SA 'iteration' performs n proposals, matching evaluation budget.
    """
    rng = np.random.default_rng(seed)
    lb, ub = bounds
    lb = np.array([lb]*dim) if np.isscalar(lb) else np.array(lb, dtype=float)
    ub = np.array([ub]*dim) if np.isscalar(ub) else np.array(ub, dtype=float)

    x = rng.uniform(lb, ub, size=(dim,))
    fx = float(np.nan_to_num(func(x), nan=1e10, posinf=1e10, neginf=1e10))
    best = x.copy()
    fbest = fx

    T0 = 1.0
    alpha = 0.95

    conv = []
    for t in range(max_iter):
        T = T0 * (alpha ** t)
        # n proposals per iteration
        for _ in range(n):
            x_new = x + rng.normal(0, 0.1*(ub - lb), size=(dim,))
            x_new = _mirror_reflect(x_new, lb, ub)
            f_new = float(np.nan_to_num(func(x_new), nan=1e10, posinf=1e10, neginf=1e10))
            if (f_new < fx) or (rng.random() < np.exp(-(f_new - fx) / max(T, 1e-12))):
                x, fx = x_new, f_new
            if fx < fbest:
                best, fbest = x.copy(), fx
        conv.append(float(fbest))

    return best, float(fbest), conv

# ---------------- Stats helpers ----------------
def median_iqr_curve(curves_2d):
    """
    curves_2d: (runs, iters)
    returns: median (iters,), q25, q75
    """
    med = np.median(curves_2d, axis=0)
    q25 = np.percentile(curves_2d, 25, axis=0)
    q75 = np.percentile(curves_2d, 75, axis=0)
    return med, q25, q75

def cliff_delta(x, y):
    """
    Cliff's delta for two samples (nonparametric effect size).
    """
    x = np.asarray(x); y = np.asarray(y)
    nx = len(x); ny = len(y)
    # O(nx*ny) is okay here (20 runs); keep exact
    gt = 0
    lt = 0
    for xi in x:
        gt += np.sum(xi > y)
        lt += np.sum(xi < y)
    return (gt - lt) / float(nx * ny + 1e-12)

def paired_cohens_d(x, y):
    """
    Paired Cohen's d (effect size on paired differences).
    """
    x = np.asarray(x); y = np.asarray(y)
    d = x - y
    return float(np.mean(d) / (np.std(d, ddof=1) + 1e-12))

def holm_correction(pvals, alpha=0.05):
    """
    Holm-Bonferroni correction. Returns adjusted p-values and reject decisions.
    """
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m, dtype=float)
    reject = np.zeros(m, dtype=bool)
    for k, idx in enumerate(order):
        adj[idx] = min((m - k) * pvals[idx], 1.0)
    # step-down reject
    for k, idx in enumerate(order):
        threshold = alpha / (m - k)
        if pvals[idx] <= threshold:
            reject[idx] = True
        else:
            break
    return adj, reject

def friedman_ranks(table_funcs_by_alg):
    """
    table_funcs_by_alg: dict alg -> array of length Nfunc (lower is better)
    returns: avg ranks per alg, chi2_F statistic (p-value if scipy available)
    """
    algs = list(table_funcs_by_alg.keys())
    N = len(next(iter(table_funcs_by_alg.values())))
    k = len(algs)

    M = np.vstack([table_funcs_by_alg[a] for a in algs]).T  # (N, k)
    ranks = np.zeros_like(M, dtype=float)
    for i in range(N):
        # rank within a function (lower fitness => better => rank 1)
        order = np.argsort(M[i])
        r = np.empty(k, dtype=float)
        r[order] = np.arange(1, k+1, dtype=float)
        ranks[i] = r

    avg_ranks = ranks.mean(axis=0)
    Rj = avg_ranks

    chi2_F = (12*N)/(k*(k+1)) * np.sum(Rj**2) - 3*N*(k+1)

    p = None
    if SCIPY_OK:
        p = float(1.0 - chi2.cdf(chi2_F, df=k-1))
    return algs, avg_ranks, float(chi2_F), p

# ---------------- Plotting ----------------
def plot_conv_median_iqr(curves_by_alg, title, outpath, highlight="ARL"):
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.figure(figsize=(7.2, 5.0))

    # plot non-highlight first
    keys = list(curves_by_alg.keys())
    keys_sorted = [k for k in keys if k != highlight] + ([highlight] if highlight in keys else [])

    for alg in keys_sorted:
        C = np.asarray(curves_by_alg[alg], dtype=float)
        if C.ndim != 2 or C.shape[0] == 0:
            continue
        med, q25, q75 = median_iqr_curve(C)

        # Force visibility: ARL on top with marker + high zorder
        if alg == highlight:
            plt.semilogy(med, linewidth=2.6, marker="o", markersize=3.2,
                         markevery=max(1, len(med)//12), zorder=10, label=alg)
        else:
            plt.semilogy(med, linewidth=1.6, zorder=3, label=alg)
        plt.fill_between(np.arange(len(med)), q25, q75, alpha=0.12)

    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("Best fitness (log scale)")
    plt.grid(True, ls="--", alpha=0.5)
    plt.legend(frameon=False, fontsize=9)
    plt.tight_layout()
    plt.savefig(outpath, dpi=500)
    plt.close()

def plot_box_final(final_by_alg, title, outpath, order):
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.figure(figsize=(7.2, 5.0))
    data = [final_by_alg[a] for a in order if a in final_by_alg]
    labels = [a for a in order if a in final_by_alg]
    plt.boxplot(data, labels=labels, showmeans=True)
    plt.yscale("log")
    plt.title(title)
    plt.ylabel("Final fitness (log scale)")
    plt.grid(True, ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(outpath, dpi=500)
    plt.close()

def plot_diag_arls(diag_runs, title, outpath):
    """
    diag_runs: list of diag dict, each diag contains arrays len=max_iter
    """
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    # stack
    keys = ["step_mean", "step_std", "dens_std", "dens_entropy", "clip_lo_ratio", "clip_hi_ratio"]
    plt.figure(figsize=(7.2, 5.0))
    for k in keys:
        A = np.vstack([d[k] for d in diag_runs])  # (runs, iters)
        med = np.median(A, axis=0)
        q25 = np.percentile(A, 25, axis=0)
        q75 = np.percentile(A, 75, axis=0)
        # keep everything on same axes (dimensionless indicators); use linear scale
        plt.plot(med, linewidth=1.8, label=k)
        plt.fill_between(np.arange(len(med)), q25, q75, alpha=0.10)

    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("Diagnostic value")
    plt.grid(True, ls="--", alpha=0.5)
    plt.legend(frameon=False, fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(outpath, dpi=500)
    plt.close()

# ---------------- Safe wrapper ----------------
def safe_run_arl(func, dim, bounds, max_iter, n, seed, **kwargs):
    try:
        g, f, conv, diag = ARL(func, dim, bounds, max_iter=max_iter, n=n, seed=seed, **kwargs)
        conv = np.asarray(conv, dtype=float)
        if conv.size < max_iter:
            conv = np.pad(conv, (0, max_iter-conv.size), mode="edge")
        return g, float(f), conv, diag
    except Exception:
        # fail-safe
        return None, 1e10, np.ones((max_iter,), dtype=float)*1e10, {
            "step_mean": np.ones((max_iter,))*np.nan,
            "step_std": np.ones((max_iter,))*np.nan,
            "dens_std": np.ones((max_iter,))*np.nan,
            "dens_entropy": np.ones((max_iter,))*np.nan,
            "clip_lo_ratio": np.ones((max_iter,))*np.nan,
            "clip_hi_ratio": np.ones((max_iter,))*np.nan,
        }

def safe_run_baseline(solver, func, dim, bounds, max_iter, n, seed):
    try:
        g, f, conv = solver(func, dim, bounds, max_iter=max_iter, n=n, seed=seed)
        conv = np.asarray(conv, dtype=float)
        if conv.size < max_iter:
            conv = np.pad(conv, (0, max_iter-conv.size), mode="edge")
        return g, float(f), conv
    except Exception:
        return None, 1e10, np.ones((max_iter,), dtype=float)*1e10

# ---------------- Main benchmarking ----------------
def run_all(max_iter=30, runs=20, dim=30, pop_size=30, outdir="exp_arl_topj"):
    os.makedirs(outdir, exist_ok=True)
    figdir = os.path.join(outdir, "fig"); os.makedirs(figdir, exist_ok=True)
    csvdir = os.path.join(outdir, "csv"); os.makedirs(csvdir, exist_ok=True)
    curdir = os.path.join(outdir, "curves"); os.makedirs(curdir, exist_ok=True)
    diagdir = os.path.join(outdir, "diag"); os.makedirs(diagdir, exist_ok=True)

    # Store:
    # final_results[fname][alg] = list of final fitness (runs)
    # curves[fname][alg] = array (runs, iters)
    final_results = {}
    curves = {}
    diag_ARL = {}  # diagnostics only for ARL/full (runs list)

    for fname, (func, bounds) in FUNCTIONS.items():
        final_results[fname] = {}
        curves[fname] = {}
        diag_ARL[fname] = []

        # ---------- ARL variants (single ARL function with flags) ----------
        arl_variants = {
            "ARL": dict(use_pca=True, use_density=True, use_leader=True, n_regions=10),
            "ARL_noDensity": dict(use_pca=True, use_density=False, use_leader=True, n_regions=10),
            "ARL_noPCA": dict(use_pca=False, use_density=True, use_leader=True, n_regions=10),
            "ARL_noLeader": dict(use_pca=True, use_density=True, use_leader=False, n_regions=10),
        }

        for alg, flags in arl_variants.items():
            vals = []
            C = []
            for r in range(runs):
                _, f, conv, diag = safe_run_arl(func, dim, bounds, max_iter, pop_size, seed=r, **flags)
                vals.append(f)
                C.append(conv)
                if alg == "ARL":
                    diag_ARL[fname].append(diag)

            final_results[fname][alg] = np.array(vals, dtype=float)
            curves[fname][alg] = np.vstack(C)

        # ---------- Baselines ----------
        baselines = {
            "PSO": PSO,
            "SA": SA_budget_matched,
            "SSA": SSA,
        }

        for alg, solver in baselines.items():
            vals = []
            C = []
            for r in range(runs):
                _, f, conv = safe_run_baseline(solver, func, dim, bounds, max_iter, pop_size, seed=r)
                vals.append(f)
                C.append(conv)
            final_results[fname][alg] = np.array(vals, dtype=float)
            curves[fname][alg] = np.vstack(C)

        # ---------- Save per-function raw ----------
        df_final = pd.DataFrame({a: final_results[fname][a] for a in ALGO_ORDER if a in final_results[fname]})
        df_final.to_csv(os.path.join(csvdir, f"{fname}_final.csv"), index=False)

        np.savez_compressed(
            os.path.join(curdir, f"{fname}_curves.npz"),
            **{a: curves[fname][a] for a in ALGO_ORDER if a in curves[fname]}
        )

        # ---------- Figures ----------
        plot_box_final(final_results[fname], title=f"{fname}: Final fitness distribution",
                       outpath=os.path.join(figdir, f"{fname}_box.png"),
                       order=ALGO_ORDER)

        # Ablation convergence: highlight ARL
        ablation_keys = ["ARL", "ARL_noDensity", "ARL_noPCA", "ARL_noLeader"]
        plot_conv_median_iqr({k: curves[fname][k] for k in ablation_keys},
                             title=f"{fname}: ARL ablations (median and IQR)",
                             outpath=os.path.join(figdir, f"{fname}_ablation_conv.png"),
                             highlight="ARL")

        # Baseline convergence: ARL vs PSO/SA/SSA
        base_keys = ["ARL", "PSO", "SA", "SSA"]
        plot_conv_median_iqr({k: curves[fname][k] for k in base_keys},
                             title=f"{fname}: ARL vs baselines (median and IQR)",
                             outpath=os.path.join(figdir, f"{fname}_baseline_conv.png"),
                             highlight="ARL")

        # ARL diagnostics: show whether density modulation is active or clipped
        plot_diag_arls(diag_ARL[fname],
                       title=f"{fname}: ARL mechanism diagnostics (median and IQR)",
                       outpath=os.path.join(diagdir, f"{fname}_ARL_diagnostics.png"))

    # ---------------- Summary tables ----------------
    summary_rows = []
    for fname in FUNCTIONS.keys():
        for alg in ALGO_ORDER:
            x = final_results[fname][alg]
            summary_rows.append({
                "Function": fname,
                "Algorithm": alg,
                "Mean": float(np.mean(x)),
                "Std": float(np.std(x, ddof=1)),
                "Median": float(np.median(x)),
                "IQR": float(np.percentile(x, 75) - np.percentile(x, 25)),
                "Best": float(np.min(x)),
                "Worst": float(np.max(x)),
            })
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(os.path.join(outdir, "summary.csv"), index=False)

    # Wide summary (median per function) for Friedman ranks
    median_table = df_summary.pivot_table(index="Function", columns="Algorithm", values="Median", aggfunc="first")
    median_table.to_csv(os.path.join(outdir, "median_table.csv"))

    # Friedman ranks (across functions, using median)
    fr_data = {alg: median_table[alg].values for alg in median_table.columns}
    algs, avg_ranks, chi2F, pF = friedman_ranks(fr_data)
    df_fried = pd.DataFrame({"Algorithm": algs, "AvgRank": avg_ranks})
    df_fried["Friedman_chi2"] = chi2F
    df_fried["Friedman_p"] = (pF if pF is not None else "NA")
    df_fried.to_csv(os.path.join(outdir, "friedman_ranks.csv"), index=False)

    # Wilcoxon paired tests: ARL vs others (paired by seed/run) on per-function medians? (preferred: per-function final arrays)
    # Here we do paired tests over runs for each function, then aggregate p-values with Holm across all comparisons (functions*comparisons).
    comparisons = [a for a in ALGO_ORDER if a != "ARL"]
    test_rows = []
    all_p = []
    all_meta = []

    for fname in FUNCTIONS.keys():
        x = final_results[fname]["ARL"]
        for alg in comparisons:
            y = final_results[fname][alg]
            # paired
            pval = None
            if SCIPY_OK:
                try:
                    stat, pval = wilcoxon(x, y, alternative="less", zero_method="wilcox")
                    pval = float(pval)
                except Exception:
                    pval = None

            cd = cliff_delta(x, y)
            d_paired = paired_cohens_d(y, x)  # positive => y larger than x (ARL better)
            row = {
                "Function": fname,
                "Compare": f"ARL vs {alg}",
                "Wilcoxon_p": (pval if pval is not None else "NA"),
                "CliffsDelta(ARL,Other)": float(cd),
                "PairedCohen_d(Other-ARL)": float(d_paired),
            }
            test_rows.append(row)
            if pval is not None:
                all_p.append(pval)
                all_meta.append((fname, alg))

    # Holm correction across all available p-values
    if len(all_p) > 0:
        adj, rej = holm_correction(np.array(all_p), alpha=0.05)
        adj_map = {(f, a): (float(adj[i]), bool(rej[i])) for i, (f, a) in enumerate(all_meta)}
        for row in test_rows:
            if row["Wilcoxon_p"] == "NA":
                row["Holm_p"] = "NA"
                row["Holm_reject@0.05"] = "NA"
            else:
                # parse alg name
                alg = row["Compare"].split("vs")[-1].strip()
                f = row["Function"]
                row["Holm_p"], row["Holm_reject@0.05"] = adj_map.get((f, alg), ("NA", "NA"))
    else:
        for row in test_rows:
            row["Holm_p"] = "NA"
            row["Holm_reject@0.05"] = "NA"

    df_tests = pd.DataFrame(test_rows)
    df_tests.to_csv(os.path.join(outdir, "wilcoxon_effects.csv"), index=False)

    # Excel summary
    xlsx_path = os.path.join(outdir, "summary.xlsx")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df_summary.to_excel(writer, sheet_name="summary_long", index=False)
        median_table.to_excel(writer, sheet_name="median_table")
        df_fried.to_excel(writer, sheet_name="friedman_ranks", index=False)
        df_tests.to_excel(writer, sheet_name="wilcoxon_effects", index=False)

    # Sanity check: ensure PNG exists for at least one function
    sample_png = os.path.join(figdir, "Sphere_ablation_conv.png")
    if not os.path.exists(sample_png):
        raise RuntimeError("PNG output not found. Check filesystem permissions or matplotlib backend.")

    print("Done. Outputs saved to:", outdir)
    print(" - Figures:", figdir)
    print(" - Diagnostics:", diagdir)
    print(" - CSV:", csvdir)
    print(" - Curves:", curdir)
    print(" - Summary:", xlsx_path)
    if not SCIPY_OK:
        print("Note: SciPy not found. p-values are reported as 'NA'. Effect sizes and ranks are still computed.")

# ---------------- CLI ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter", type=int, default=30, help="iterations")
    parser.add_argument("--runs", type=int, default=20, help="independent runs")
    parser.add_argument("--dim", type=int, default=30, help="dimension")
    parser.add_argument("--pop", type=int, default=30, help="population size / budget unit")
    parser.add_argument("--outdir", type=str, default="exp_arl_topj", help="output dir")
    args = parser.parse_args()

    run_all(max_iter=args.iter, runs=args.runs, dim=args.dim, pop_size=args.pop, outdir=args.outdir)
