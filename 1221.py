#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARL benchmark suite (publication-style, budget = function evaluations).

What’s improved in this revised script (vs your current version)
1) Strict FE-budget fairness across ALL algorithms
   - PSO / SSA are converted to per-evaluation (no “pop-evals per generation” overshoot).
   - All algorithms stop exactly at max_fes.
2) Unified boundary handling (reflect) for ALL algorithms
   - Removes structural bias from “ARL uses reflect while others use clip”.
3) Shared initial population + cached initial fitness (X0, fit0)
   - Each algorithm *counts* pop evaluations for initialization (fes=pop), but reuses cached values.
   - Ensures identical start points for fairness, and improves runtime without changing shows.
4) Stronger and more stable ARL (still FE-budget compliant)
   - Adds a cheap “elite-diagonal sampling” operator (quasi-CMA flavor) to handle rotated/ill-conditioned cases.
   - Uses p-best guidance (from top 20%) instead of single leader-only in DE-like operator (more robust).
   - Makes region count k adaptive when regions are active (based on normalized diversity).
5) Metrics are computed per-run then aggregated
   - AUC and slice-AUC are computed per replicate and then median-aggregated (more reviewer-friendly).
6) Additional evidence outputs (do not change your existing tables/plots)
   - ARL_Diagnostics/<Group>/<Function>/ : trigger on/off, k over time, operator usage, diversity/dispersion/EVR1.
   - AblationContribution/ : ΔAUC(ablations − ARL) distributions and sorted bars.
   - FrameworkAdvantage/ stays (already in your script), now robust to new metric aggregation.

Outputs remain under outputs_arl_pub/ with Baselines/ Ablations/ TriggerStudy/ plus:
  ARL_Diagnostics/
  AblationContribution/

Dependencies: numpy, matplotlib
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import time
import zlib
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Optional, Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# -----------------------------
# Publication styling
# -----------------------------

NATURE = {
    "ARL": "#D55E00",
    "DE": "#0072B2",
    "PSO": "#009E73",
    "SA": "#E69F00",
    "SSA": "#CC79A7",
    "ARL_noPCA": "#56B4E9",
    "ARL_noDensity": "#3C5488",
    "ARL_noLeader": "#00A087",
    "ARL_noRegion": "#F39B7F",
    "ARL_AlwaysRegion": "#4DBBD5",
    "ARL_NeverRegion": "#7A7A7A",
    "ARL_TwoPhase": "#E64B35",
}

NEUTRAL_GRAY = "#666666"
GRID_GRAY = "#C7C7C7"


def set_pub_style():
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 500,
        "font.family": "DejaVu Sans",
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 12,
        "axes.linewidth": 1.4,
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "xtick.major.width": 1.3,
        "ytick.major.width": 1.3,
        "grid.color": GRID_GRAY,
        "grid.linestyle": "--",
        "grid.alpha": 0.55,
        "axes.grid": True,
    })


def stable_hash32(s: str) -> int:
    return int(zlib.crc32(s.encode("utf-8")) & 0xFFFFFFFF)


def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def make_cmap_nature() -> LinearSegmentedColormap:
    colors = ["#F7F7F7", "#BBD7EA", "#4DBBD5", "#00A087", "#E69F00", "#D55E00"]
    return LinearSegmentedColormap.from_list("nature_seq", colors)


# -----------------------------
# Benchmark suite (30 functions)
# -----------------------------

@dataclass
class TestFunction:
    name: str
    f: Callable[[np.ndarray], float]
    lb: float
    ub: float
    dim: int
    f_opt: float = 0.0
    shift: Optional[np.ndarray] = None
    rot: Optional[np.ndarray] = None


def _orthogonal_matrix(dim: int, rng: np.random.Generator) -> np.ndarray:
    A = rng.normal(size=(dim, dim))
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def _apply_shift_rot(x: np.ndarray, shift: Optional[np.ndarray], rot: Optional[np.ndarray]) -> np.ndarray:
    z = x
    if shift is not None:
        z = z - shift
    if rot is not None:
        z = z @ rot
    return z


def sphere(z: np.ndarray) -> float:
    return float(np.sum(z * z))


def rosenbrock(z: np.ndarray) -> float:
    return float(np.sum(100.0 * (z[1:] - z[:-1] ** 2) ** 2 + (z[:-1] - 1.0) ** 2))


def rastrigin(z: np.ndarray) -> float:
    A = 10.0
    return float(A * z.size + np.sum(z * z - A * np.cos(2 * np.pi * z)))


def ackley(z: np.ndarray) -> float:
    a, b, c = 20.0, 0.2, 2 * np.pi
    n = z.size
    s1 = np.sum(z * z)
    s2 = np.sum(np.cos(c * z))
    return float(-a * np.exp(-b * np.sqrt(s1 / n)) - np.exp(s2 / n) + a + math.e)


def griewank(z: np.ndarray) -> float:
    i = np.arange(1, z.size + 1)
    return float(1.0 + np.sum(z * z) / 4000.0 - np.prod(np.cos(z / np.sqrt(i))))


def schwefel(z: np.ndarray) -> float:
    return float(418.9829 * z.size - np.sum(z * np.sin(np.sqrt(np.abs(z) + 1e-12))))


def levy(z: np.ndarray) -> float:
    w = 1.0 + (z - 1.0) / 4.0
    term1 = np.sin(np.pi * w[0]) ** 2
    term3 = (w[-1] - 1.0) ** 2 * (1 + np.sin(2 * np.pi * w[-1]) ** 2)
    wi = w[:-1]
    term2 = np.sum((wi - 1.0) ** 2 * (1 + 10 * np.sin(np.pi * wi + 1) ** 2))
    return float(term1 + term2 + term3)


def zakharov(z: np.ndarray) -> float:
    i = np.arange(1, z.size + 1)
    s1 = np.sum(z * z)
    s2 = np.sum(0.5 * i * z)
    return float(s1 + s2 ** 2 + s2 ** 4)


def dixon_price(z: np.ndarray) -> float:
    term1 = (z[0] - 1.0) ** 2
    i = np.arange(2, z.size + 1)
    term2 = np.sum(i * (2 * z[1:] ** 2 - z[:-1]) ** 2)
    return float(term1 + term2)


def sum_squares(z: np.ndarray) -> float:
    i = np.arange(1, z.size + 1)
    return float(np.sum(i * z * z))


def bent_cigar(z: np.ndarray) -> float:
    return float(z[0] ** 2 + 1e6 * np.sum(z[1:] ** 2))


def discus(z: np.ndarray) -> float:
    return float(1e6 * z[0] ** 2 + np.sum(z[1:] ** 2))


def elliptic(z: np.ndarray) -> float:
    n = z.size
    i = np.arange(n)
    weights = (1e6) ** (i / (n - 1 + 1e-12))
    return float(np.sum(weights * z * z))


def ridge(z: np.ndarray) -> float:
    return float(z[0] + 100.0 * np.sqrt(np.sum(z[1:] ** 2) + 1e-12))


def happy_cat(z: np.ndarray) -> float:
    n = z.size
    r2 = np.sum(z * z)
    sumz = np.sum(z)
    return float(((r2 - n) ** 2) ** 0.125 + (0.5 * r2 + sumz) / n + 0.5)


def hgbat(z: np.ndarray) -> float:
    n = z.size
    r2 = np.sum(z * z)
    sumz = np.sum(z)
    return float((np.abs(r2 ** 2 - sumz ** 2) ** 0.5) + (0.5 * r2 + sumz) / n + 0.5)


def weierstrass(z: np.ndarray) -> float:
    a, b, kmax = 0.5, 3.0, 20
    k = np.arange(0, kmax + 1)
    ak = a ** k
    bk = b ** k
    term1 = 0.0
    for ak_i, bk_i in zip(ak, bk):
        term1 += np.sum(ak_i * np.cos(2 * np.pi * (bk_i * (z + 0.5))))
    term2 = z.size * np.sum(ak * np.cos(2 * np.pi * (bk * 0.5)))
    return float(term1 - term2)


def katsuura(z: np.ndarray) -> float:
    n = z.size
    prod = 1.0
    for i in range(n):
        s = 0.0
        for j in range(1, 33):
            s += abs(2 ** j * z[i] - np.round(2 ** j * z[i])) / (2 ** j)
        prod *= (1 + (i + 1) * s) ** (10.0 / (n ** 1.2))
    return float(prod - 1.0)


def salomon(z: np.ndarray) -> float:
    r = np.sqrt(np.sum(z * z))
    return float(1 - np.cos(2 * np.pi * r) + 0.1 * r)


def alpine1(z: np.ndarray) -> float:
    return float(np.sum(np.abs(z * np.sin(z) + 0.1 * z)))


def alpine2(z: np.ndarray) -> float:
    return float(np.prod(np.sqrt(np.abs(z) + 1e-12) * np.sin(z)))


def step(z: np.ndarray) -> float:
    return float(np.sum(np.floor(z + 0.5) ** 2))


def bohachevsky(z: np.ndarray) -> float:
    s = 0.0
    for i in range(0, z.size - 1, 2):
        x, y = z[i], z[i + 1]
        s += x ** 2 + 2 * y ** 2 - 0.3 * np.cos(3 * np.pi * x) - 0.4 * np.cos(4 * np.pi * y) + 0.7
    if z.size % 2 == 1:
        s += z[-1] ** 2
    return float(s)


def trid(z: np.ndarray) -> float:
    s1 = np.sum((z - 1) ** 2)
    s2 = np.sum(z[1:] * z[:-1])
    return float(s1 - s2)


def brown(z: np.ndarray) -> float:
    s = 0.0
    for i in range(z.size - 1):
        s += (z[i] ** 2) ** (z[i + 1] ** 2 + 1) + (z[i + 1] ** 2) ** (z[i] ** 2 + 1)
    return float(s)


def quartic(z: np.ndarray) -> float:
    i = np.arange(1, z.size + 1)
    return float(np.sum(i * (z ** 4)))


def powell_sum(z: np.ndarray) -> float:
    i = np.arange(1, z.size + 1)
    return float(np.sum(np.abs(z) ** (i + 1)))


def shifted_sphere(z: np.ndarray) -> float:
    return float(np.sum(z * z))


def make_suite(dim: int, suite_seed: int, n_funcs: int = 30) -> List[TestFunction]:
    rng = np.random.default_rng(suite_seed)
    templates = [
        ("Sphere", sphere, -100, 100, True, True),
        ("Rosenbrock", rosenbrock, -30, 30, True, True),
        ("Rastrigin", rastrigin, -5.12, 5.12, True, True),
        ("Ackley", ackley, -32.768, 32.768, True, True),
        ("Griewank", griewank, -600, 600, True, True),
        ("Schwefel", schwefel, -500, 500, False, True),
        ("Levy", levy, -10, 10, True, True),
        ("Zakharov", zakharov, -5, 10, True, True),
        ("DixonPrice", dixon_price, -10, 10, True, True),
        ("SumSquares", sum_squares, -10, 10, True, True),
        ("BentCigar", bent_cigar, -100, 100, True, True),
        ("Discus", discus, -100, 100, True, True),
        ("Elliptic", elliptic, -100, 100, True, True),
        ("Ridge", ridge, -100, 100, True, True),
        ("HappyCat", happy_cat, -100, 100, True, True),
        ("HGBat", hgbat, -100, 100, True, True),
        ("Weierstrass", weierstrass, -0.5, 0.5, True, True),
        ("Katsuura", katsuura, -100, 100, True, True),
        ("Salomon", salomon, -100, 100, True, True),
        ("Alpine1", alpine1, -10, 10, True, True),
        ("Alpine2", alpine2, 0.0, 10.0, True, True),
        ("Step", step, -100, 100, True, True),
        ("Bohachevsky", bohachevsky, -100, 100, True, True),
        ("Trid", trid, -100, 100, True, True),
        ("Brown", brown, -1, 4, True, True),
        ("Quartic", quartic, -2, 2, True, True),
        ("PowellSum", powell_sum, -1, 1, True, True),
        ("Sphere_wide", shifted_sphere, -1000, 1000, True, True),
        ("Rastrigin_wide", rastrigin, -10.24, 10.24, True, True),
        ("Ackley_wide", ackley, -40, 40, True, True),
    ]
    templates = templates[:min(len(templates), n_funcs)]

    suite: List[TestFunction] = []
    for (name, base_f, lb, ub, use_rot, use_shift) in templates:
        shift = rng.uniform(lb * 0.25, ub * 0.25, size=dim) if use_shift else None
        rot = _orthogonal_matrix(dim, rng) if use_rot else None

        def wrap(base_fun=base_f, shift_=shift, rot_=rot):
            def f(x: np.ndarray) -> float:
                z = _apply_shift_rot(x, shift_, rot_)
                return float(base_fun(z))
            return f

        suite.append(TestFunction(
            name=name, f=wrap(), lb=float(lb), ub=float(ub), dim=dim, f_opt=0.0, shift=shift, rot=rot
        ))
    return suite


# -----------------------------
# Utilities / metrics
# -----------------------------

def make_checkpoints(max_fes: int, n_points: int) -> np.ndarray:
    n_points = max(60, int(n_points))
    cps = np.unique(np.round(np.linspace(0, max_fes, n_points)).astype(int))
    if cps[0] != 0:
        cps = np.concatenate([[0], cps])
    if cps[-1] != max_fes:
        cps = np.concatenate([cps, [max_fes]])
    return cps


def safe_gap(best: float, f_opt: float, eps: float = 1e-12) -> float:
    g = best - f_opt
    if not np.isfinite(g):
        return 1e12
    return float(max(g, eps))


def auc_trapz(y: np.ndarray, x: np.ndarray) -> float:
    return float(np.trapz(y, x))


def rank_lower_is_better(values: Dict[str, float]) -> Dict[str, float]:
    keys = list(values.keys())
    v = np.array([values[k] for k in keys], dtype=float)
    order = np.argsort(v)
    ranks = np.empty_like(v)
    ranks[order] = np.arange(1, len(keys) + 1, dtype=float)
    for i in range(len(keys)):
        same = np.where(np.isclose(v, v[i], rtol=0, atol=1e-15))[0]
        if same.size > 1:
            ranks[same] = np.mean(ranks[same])
    return {k: float(ranks[idx]) for idx, k in enumerate(keys)}


def clip_box(X: np.ndarray, lb: float, ub: float) -> np.ndarray:
    return np.clip(X, lb, ub)


def reflect_box(X: np.ndarray, lb: float, ub: float) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    span = float(ub - lb)
    if (not np.isfinite(span)) or span <= 0.0:
        return np.clip(X, lb, ub)
    Y = (X - lb) % (2.0 * span)
    Y = np.where(Y > span, 2.0 * span - Y, Y)
    return lb + Y


def levy_flight_step(dim: int, rng: np.random.Generator, beta: float = 1.5) -> np.ndarray:
    beta = float(beta)
    num = math.gamma(1.0 + beta) * math.sin(math.pi * beta / 2.0)
    den = math.gamma((1.0 + beta) / 2.0) * beta * (2.0 ** ((beta - 1.0) / 2.0))
    sigma_u = (num / (den + 1e-300)) ** (1.0 / beta)
    u = rng.normal(0.0, sigma_u, size=dim)
    v = rng.normal(0.0, 1.0, size=dim)
    return u / (np.abs(v) ** (1.0 / beta) + 1e-12)


def init_population(dim: int, pop: int, lb: float, ub: float, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(lb, ub, size=(pop, dim))


def pca_embed(X: np.ndarray, out_dim: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    Xc = X - X.mean(axis=0, keepdims=True)
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = (S ** 2)
    evr = var / (var.sum() + 1e-12)
    Z = Xc @ Vt[:out_dim].T
    return Z, evr


def kmeans_simple(Z: np.ndarray, k: int, rng: np.random.Generator, max_iter: int = 25) -> np.ndarray:
    n, d = Z.shape
    if k <= 1 or n <= 1:
        return np.zeros(n, dtype=int)
    k = min(k, n)

    centers = np.empty((k, d), dtype=float)
    centers[0] = Z[int(rng.integers(0, n))]

    dist2 = np.full(n, np.inf, dtype=float)
    for i in range(1, k):
        dist2 = np.minimum(dist2, np.sum((Z - centers[i - 1]) ** 2, axis=1))
        s = dist2.sum()
        if not np.isfinite(s) or s <= 0:
            centers[i] = Z[int(rng.integers(0, n))]
            continue
        probs = dist2 / s
        probs = np.clip(probs, 0.0, 1.0)
        ps = probs.sum()
        if ps <= 0 or not np.isfinite(ps):
            probs[:] = 1.0 / n
        else:
            probs /= ps
        centers[i] = Z[int(rng.choice(n, p=probs))]

    labels = np.zeros(n, dtype=int)
    for _ in range(max_iter):
        dmat = np.sum((Z[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        new_labels = np.argmin(dmat, axis=1)
        if np.all(new_labels == labels):
            break
        labels = new_labels
        for j in range(k):
            mask = labels == j
            if np.any(mask):
                centers[j] = Z[mask].mean(axis=0)
            else:
                centers[j] = Z[int(rng.integers(0, n))]
    return labels


def arl_should_use_regions(pop: np.ndarray, fit: np.ndarray, evr: np.ndarray, gen: int, max_gens: int) -> bool:
    coord_std = pop.std(axis=0)
    diversity = float(np.mean(coord_std))
    f_med = float(np.median(fit))
    mad = float(np.median(np.abs(fit - f_med)) + 1e-12)
    rel_disp = mad / (abs(f_med) + 1e-12)
    evr1 = float(evr[0]) if evr.size > 0 else 0.0

    t = gen / max(1, max_gens)
    if t < 0.35:
        return True
    if t > 0.75 and (diversity < 0.05) and (evr1 > 0.75) and (rel_disp < 0.15):
        return False
    return (diversity > 0.08) or (rel_disp > 0.20)


def arl_regions_gate(
    trigger_mode: str | None,
    pop: np.ndarray,
    fit: np.ndarray,
    evr: np.ndarray,
    gen: int,
    max_gens: int,
) -> bool:
    m = (trigger_mode or "adaptive").strip().lower()
    t = float(gen) / float(max(1, max_gens))

    if m in {"always", "always_on", "on"}:
        return True
    if m in {"never", "off", "none", "disable"}:
        return False
    if m in {"early", "early_only"}:
        return t < 0.5
    if m in {"late", "late_only"}:
        return t >= 0.5

    if m in {"two_phase", "explore_then_exploit", "early_on_late_off"}:
        if t < 0.30:
            return True
        if t > 0.80:
            return False
        return arl_should_use_regions(pop, fit, evr, gen, max_gens)

    if m in {"adaptive_v1", "adaptive_stateless"}:
        return arl_should_use_regions(pop, fit, evr, gen, max_gens)

    return arl_should_use_regions(pop, fit, evr, gen, max_gens)


@dataclass
class AdaptiveRegionsTrigger:
    lb: np.ndarray
    ub: np.ndarray
    max_gens: int
    patience: int
    warmup_frac: float = 0.25
    cooldown_frac: float = 0.15
    ema_alpha: float = 0.25
    theta_on: float = 0.55
    theta_off: float = 0.40
    min_hold: int = 6
    force_off_evr1: float = 0.90
    force_off_div: float = 0.015

    active: bool = True
    ema_score: float = 0.0
    last_switch_gen: int = 0

    def _clip01(self, x: float) -> float:
        return 0.0 if x <= 0.0 else (1.0 if x >= 1.0 else float(x))

    def feature_score(self, pop: np.ndarray, fit: np.ndarray, evr: np.ndarray) -> Tuple[float, float, float, float]:
        eps = 1e-12
        span = np.maximum(self.ub - self.lb, eps)
        div_norm = float(np.mean(np.std(pop, axis=0)) / float(np.mean(span)))
        med = float(np.median(fit))
        mad = float(np.median(np.abs(fit - med))) + eps
        rel_disp = float(mad / (abs(med) + eps))
        evr1 = float(evr[0]) if (isinstance(evr, np.ndarray) and evr.size > 0) else 0.0

        f_div = self._clip01(div_norm / 0.08)
        f_disp = self._clip01(rel_disp / 0.20)
        f_flat = self._clip01((1.0 - evr1) / 0.30)

        score_raw = 0.55 * f_div + 0.35 * f_disp + 0.10 * f_flat
        return float(score_raw), div_norm, rel_disp, evr1

    def update(self, pop: np.ndarray, fit: np.ndarray, evr: np.ndarray, gen: int, stagn: int) -> bool:
        t = float(gen) / float(max(1, self.max_gens))

        if t < self.warmup_frac:
            self.active = True
            return True
        if t > (1.0 - self.cooldown_frac):
            self.active = False
            return False

        score_raw, div_norm, _, evr1 = self.feature_score(pop, fit, evr)

        if gen <= 0:
            self.ema_score = score_raw
        else:
            self.ema_score = float(self.ema_alpha * score_raw + (1.0 - self.ema_alpha) * self.ema_score)

        if (evr1 >= self.force_off_evr1) and (div_norm <= self.force_off_div):
            if self.active and (gen - self.last_switch_gen) >= self.min_hold:
                self.active = False
                self.last_switch_gen = gen
            return self.active

        if (stagn >= max(3, self.patience // 2)) and (div_norm > (self.force_off_div * 2.5)):
            if (not self.active) and (gen - self.last_switch_gen) >= self.min_hold and self.ema_score >= self.theta_off:
                self.active = True
                self.last_switch_gen = gen
            return self.active

        if (gen - self.last_switch_gen) < self.min_hold:
            return self.active

        if self.active:
            if self.ema_score < self.theta_off:
                self.active = False
                self.last_switch_gen = gen
        else:
            if self.ema_score > self.theta_on:
                self.active = True
                self.last_switch_gen = gen

        return self.active


# -----------------------------
# Cached init unpacker
# -----------------------------

def unpack_init(X0: Optional[Any]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Allow X0 to be either:
      - None
      - np.ndarray (pop, dim)
      - (np.ndarray X0, np.ndarray fit0)
    """
    if X0 is None:
        return None, None
    if isinstance(X0, tuple) and len(X0) == 2:
        X, f = X0
        return np.array(X, copy=True), np.array(f, copy=True)
    return np.array(X0, copy=True), None


# -----------------------------
# ARL optimizer (FE-budget)
# -----------------------------

def run_ARL(
    fun: Callable[[np.ndarray], float],
    lb: float,
    ub: float,
    dim: int,
    max_fes: int,
    pop: int,
    rng: np.random.Generator,
    X0: Optional[Any] = None,
    ablation: Optional[str] = None,
    trigger_mode: str = "adaptive",
    n_points: int = 220,
    record_diag: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Optional[Dict[str, np.ndarray]]]:
    max_gens = max(1, max_fes // max(1, pop))
    cps = make_checkpoints(max_fes, n_points)

    X_init, fit_init = unpack_init(X0)
    if X_init is None:
        X = init_population(dim, pop, lb, ub, rng)
    else:
        X = np.array(X_init, copy=True)

    if fit_init is None:
        fit = np.array([fun(x) for x in X], dtype=float)
    else:
        fit = np.array(fit_init, dtype=float)

    fes = pop
    best = float(np.min(fit))
    curve: List[float] = []

    cp_idx = 0
    while cp_idx < len(cps) and cps[cp_idx] <= 0:
        curve.append(best)
        cp_idx += 1

    n_regions_base = max(3, min(6, int(pop // 8)))
    elite_frac = 0.12
    reinit_frac = 0.15
    stagnation_patience = max(8, int(0.05 * max_gens))

    trigger_policy: AdaptiveRegionsTrigger | None = None
    _tm = (trigger_mode or "adaptive").strip().lower()
    if _tm in {"adaptive", "adaptive_v2", "adaptive_hys", "adaptive_hysteresis"}:
        trigger_policy = AdaptiveRegionsTrigger(
            lb=np.asarray(lb, dtype=float),
            ub=np.asarray(ub, dtype=float),
            max_gens=int(max_gens),
            patience=int(stagnation_patience),
        )

    step_mult = np.ones(n_regions_base, dtype=float)
    temp_mult = np.ones(n_regions_base, dtype=float)

    # Operator pool:
    # 0: ARL core
    # 1: DE/pbest
    # 2: Levy escape
    # 3: Local refine
    # 4: Elite-diagonal sampling (quasi-CMA flavor)
    n_ops = 5
    op_ema = np.full(n_ops, 0.20, dtype=float)
    op_ema_alpha = 0.25

    last_best = best
    stagn = 0

    span = float(ub - lb)
    span = span if np.isfinite(span) and span > 0 else 1.0

    # Diagnostics buffers (per gen)
    diag = None
    if record_diag:
        diag = {
            "gen": np.zeros(max_gens, dtype=int),
            "fes": np.zeros(max_gens, dtype=int),
            "best": np.zeros(max_gens, dtype=float),
            "use_regions": np.zeros(max_gens, dtype=int),
            "k": np.zeros(max_gens, dtype=int),
            "stagn": np.zeros(max_gens, dtype=int),
            "evr1": np.zeros(max_gens, dtype=float),
            "div_norm": np.zeros(max_gens, dtype=float),
            "rel_disp": np.zeros(max_gens, dtype=float),
            "ema_score": np.zeros(max_gens, dtype=float),
            "op_p0": np.zeros(max_gens, dtype=float),
            "op_p1": np.zeros(max_gens, dtype=float),
            "op_p2": np.zeros(max_gens, dtype=float),
            "op_p3": np.zeros(max_gens, dtype=float),
            "op_p4": np.zeros(max_gens, dtype=float),
            "op_succ0": np.zeros(max_gens, dtype=float),
            "op_succ1": np.zeros(max_gens, dtype=float),
            "op_succ2": np.zeros(max_gens, dtype=float),
            "op_succ3": np.zeros(max_gens, dtype=float),
            "op_succ4": np.zeros(max_gens, dtype=float),
        }

    for g in range(max_gens):
        if fes >= max_fes:
            break

        # embedding for regions
        if ablation == "noPCA":
            Z = X.copy()
            evr = np.array([0.0])
        else:
            out_dim = 2 if dim <= 2 else min(4, dim)
            Z, evr = pca_embed(X, out_dim=int(out_dim))

        if ablation == "noRegion":
            use_regions = False
        else:
            if trigger_policy is not None:
                use_regions = bool(trigger_policy.update(X, fit, evr, g, stagn))
            else:
                use_regions = bool(arl_regions_gate(trigger_mode, X, fit, evr, g, max_gens))

        # adaptive k when regions active
        k = 1
        div_norm = 0.0
        rel_disp = 0.0
        evr1 = float(evr[0]) if evr.size > 0 else 0.0
        ema_score = float(trigger_policy.ema_score) if trigger_policy is not None else 0.0
        if use_regions:
            # normalized diversity against box span
            eps = 1e-12
            div_norm = float(np.mean(np.std(X, axis=0)) / (span + eps))
            f_med = float(np.median(fit))
            rel_disp = float((np.median(np.abs(fit - f_med)) + eps) / (abs(f_med) + eps))

            # scale k with diversity (caps at n_regions_base)
            div_target = 0.08
            k_raw = int(np.round(n_regions_base * (div_norm / (div_target + 1e-12))))
            k = int(np.clip(k_raw, 2, n_regions_base))

        labels = kmeans_simple(Z, k, rng) if use_regions else np.zeros(pop, dtype=int)

        leaders = np.zeros(k, dtype=int)
        for j in range(k):
            idxs = np.where(labels == j)[0]
            if idxs.size == 0:
                leaders[j] = int(rng.integers(0, pop))
            else:
                leaders[j] = int(idxs[np.argmin(fit[idxs])])

        gbest_idx = int(np.argmin(fit))
        gbest = X[gbest_idx].copy()
        gbest_fit = float(fit[gbest_idx])

        counts = np.array([(labels == j).sum() for j in range(k)], dtype=float)
        dens = counts / (pop + 1e-12)

        t = g / max(1, max_gens)
        base_step = 0.55 * (1.0 - 0.90 * t)
        base_step = float(np.clip(base_step, 0.04, 0.75))
        base_temp = 0.35 * (1.0 - t) + 0.05

        pulse = 1.0
        if stagn >= stagnation_patience:
            pulse = 1.6
        elif stagn >= max(3, stagnation_patience // 2):
            pulse = 1.25

        if ablation == "noDensity":
            density_factor = np.ones(k, dtype=float)
        else:
            density_factor = (1.25 - dens[:k])
            density_factor = np.clip(density_factor, 0.65, 1.35)

        step_r = np.clip(base_step * density_factor * step_mult[:k], 0.03, 0.85)
        temp_r = np.clip(base_temp * temp_mult[:k] * pulse, 0.01, 0.60)

        elite_n = max(1, int(elite_frac * pop))
        elite_idx = np.argsort(fit)[:elite_n]
        elite_mask = np.zeros(pop, dtype=bool)
        elite_mask[elite_idx] = True

        # p-best pool for robust guidance (top 20%)
        p_pool = max(2, int(0.20 * pop))
        pbest_pool = np.argsort(fit)[:p_pool]

        def _op_probs(tt: float, stagn_: int, patience_: int, op_ema_: np.ndarray) -> np.ndarray:
            if tt < 0.30:
                base = np.array([0.30, 0.28, 0.22, 0.10, 0.10], dtype=float)
            elif tt < 0.70:
                base = np.array([0.38, 0.27, 0.12, 0.08, 0.15], dtype=float)
            else:
                base = np.array([0.32, 0.12, 0.04, 0.26, 0.26], dtype=float)

            if stagn_ >= patience_:
                base = np.array([0.20, 0.26, 0.20, 0.12, 0.22], dtype=float)

            score = op_ema_ / (float(np.mean(op_ema_)) + 1e-12)
            w = base * (0.70 + 0.30 * score)
            w = np.clip(w, 1e-6, None)
            w /= float(np.sum(w))
            return w

        op_probs = _op_probs(t, stagn, stagnation_patience, op_ema)

        att_op = np.zeros(n_ops, dtype=float)
        succ_op = np.zeros(n_ops, dtype=float)
        att_reg = np.zeros(n_regions_base, dtype=float)
        succ_reg = np.zeros(n_regions_base, dtype=float)

        X_new = X.copy()
        fit_new = fit.copy()

        for i in range(pop):
            if fes >= max_fes:
                break
            if elite_mask[i]:
                continue

            r = int(labels[i]) if use_regions else 0
            r = max(0, min(k - 1, r))

            leader = int(leaders[r])
            if ablation == "noLeader":
                x_ref = X[i]
                f_lead = float(fit[i])
            else:
                # leader mixed with p-best to reduce deception risk
                if rng.random() < 0.60:
                    x_ref = X[leader]
                    f_lead = float(fit[leader])
                else:
                    pb = int(rng.choice(pbest_pool))
                    x_ref = X[pb]
                    f_lead = float(fit[pb])

            x = X[i].copy()

            a, b = rng.integers(0, pop, size=2)
            while b == a:
                b = int(rng.integers(0, pop))
            diff = X[a] - X[b]

            op = int(rng.choice(n_ops, p=op_probs))
            att_op[op] += 1.0
            att_reg[r] += 1.0

            f_med = float(np.median(fit))
            rel_imp = (f_med - f_lead) / (abs(f_med) + 1e-12)
            rel_imp = float(np.clip(rel_imp, -1.0, 1.0))
            strength = (0.15 + 0.85 * max(0.0, rel_imp)) * (0.60 + 0.40 * (1.0 - t))
            attract = strength * (x_ref - x)

            if op == 0:
                x_trial = x + step_r[r] * attract + temp_r[r] * rng.normal(size=dim) + 0.15 * diff
                if use_regions and rng.random() < 0.10:
                    x_trial = x_trial + 0.08 * rng.normal(size=dim) * span

            elif op == 1:
                # DE-like current-to-pbest/1/bin (robust)
                F = 0.45 + 0.35 * rng.random()
                CR = 0.90
                pb = int(rng.choice(pbest_pool))
                x_pb = X[pb]
                v = x + F * (x_pb - x) + F * diff
                u = x.copy()
                jrand = int(rng.integers(0, dim))
                cross = rng.random(dim) < CR
                cross[jrand] = True
                u[cross] = v[cross]
                u = u + rng.normal(size=dim) * (0.02 * (1.0 - t)) * span
                x_trial = u

            elif op == 2:
                beta = 1.5
                L = levy_flight_step(dim, rng, beta=beta)
                scale = (0.10 * (1.0 - t) + 0.02) * span
                if stagn >= stagnation_patience:
                    scale *= 1.35
                x_trial = x + scale * L + 0.05 * diff

            elif op == 3:
                center = x_ref
                sigma = (0.05 * (1.0 - t) + 0.004) * span
                sigma *= float(1.0 / (1.0 + 2.0 * dens[r]))
                x_trial = center + rng.normal(size=dim) * sigma + 0.05 * diff

            else:
                # Elite-diagonal sampling (cheap quasi-CMA)
                E = X[elite_idx]
                mu = E.mean(axis=0)
                std = E.std(axis=0) + 1e-12
                scale = (0.35 * (1.0 - t) + 0.08) * (1.35 if stagn >= stagnation_patience else 1.0)
                x_trial = mu + rng.normal(size=dim) * (scale * std) + 0.05 * diff

            x_trial = reflect_box(x_trial, lb, ub)
            f_trial = float(fun(x_trial))
            fes += 1

            if f_trial <= fit[i]:
                X_new[i] = x_trial
                fit_new[i] = f_trial
                succ_op[op] += 1.0
                succ_reg[r] += 1.0

        if fes < max_fes and stagn >= max(3, stagnation_patience // 2):
            sigma_best = (0.02 * (1.0 - t) + 0.002) * span
            x_try = reflect_box(gbest + rng.normal(size=dim) * sigma_best, lb, ub)
            f_try = float(fun(x_try))
            fes += 1
            if f_try < gbest_fit - 1e-12:
                worst = int(np.argmax(fit_new))
                X_new[worst] = x_try
                fit_new[worst] = f_try

        X, fit = X_new, fit_new
        best = float(np.min(fit))

        if best < last_best - 1e-12:
            last_best = best
            stagn = 0
        else:
            stagn += 1

        for rr in range(k):
            sr = float(succ_reg[rr] / (att_reg[rr] + 1e-12))
            step_mult[rr] *= math.exp(0.65 * (sr - 0.20))
            step_mult[rr] = float(np.clip(step_mult[rr], 0.55, 2.20))
            temp_mult[rr] *= math.exp(0.55 * (sr - 0.15))
            temp_mult[rr] = float(np.clip(temp_mult[rr], 0.45, 2.60))

        sr_op = succ_op / (att_op + 1e-12)
        op_ema = (1.0 - op_ema_alpha) * op_ema + op_ema_alpha * sr_op
        op_ema = np.clip(op_ema, 1e-6, 1.0)

        if stagn >= stagnation_patience and (ablation is None) and (fes + pop <= max_fes):
            w = np.argsort(fit)[::-1]
            n_re = max(1, int(reinit_frac * pop))
            re_idx = w[:n_re]
            X[re_idx] = init_population(dim, n_re, lb, ub, rng)
            # evaluate strictly within budget
            for ii in range(n_re):
                if fes >= max_fes:
                    break
                fit[re_idx[ii]] = float(fun(X[re_idx[ii]]))
                fes += 1
            best = float(np.min(fit))
            last_best = min(last_best, best)
            stagn = 0

        while cp_idx < len(cps) and fes >= cps[cp_idx]:
            curve.append(best)
            cp_idx += 1

        if record_diag and diag is not None:
            diag["gen"][g] = g
            diag["fes"][g] = fes
            diag["best"][g] = best
            diag["use_regions"][g] = 1 if use_regions else 0
            diag["k"][g] = k
            diag["stagn"][g] = stagn
            diag["evr1"][g] = evr1
            diag["div_norm"][g] = float(div_norm)
            diag["rel_disp"][g] = float(rel_disp)
            diag["ema_score"][g] = float(ema_score)
            diag["op_p0"][g] = float(op_probs[0]); diag["op_p1"][g] = float(op_probs[1])
            diag["op_p2"][g] = float(op_probs[2]); diag["op_p3"][g] = float(op_probs[3])
            diag["op_p4"][g] = float(op_probs[4])
            for oi in range(n_ops):
                diag[f"op_succ{oi}"][g] = float(succ_op[oi] / (att_op[oi] + 1e-12))

    while cp_idx < len(cps):
        curve.append(best)
        cp_idx += 1

    if record_diag and diag is not None:
        used = int(min(len(diag["gen"]), max_gens))
        # trim to actual gens completed (first zero tail can exist if early break)
        last_nonzero = 0
        for i in range(used):
            if diag["fes"][i] > 0:
                last_nonzero = i
        last_nonzero = max(last_nonzero, 0)
        for k2 in list(diag.keys()):
            diag[k2] = diag[k2][:last_nonzero + 1]

    return cps.astype(int), np.array(curve, dtype=float), diag


# -----------------------------
# Baseline optimizers (strict FE-budget, reflect boundaries, cached init)
# -----------------------------

def run_DE(fun, lb, ub, dim, max_fes, pop, rng, X0=None, n_points=220):
    max_gens = max(1, max_fes // pop)
    cps = make_checkpoints(max_fes, n_points)

    X_init, fit_init = unpack_init(X0)
    if X_init is None:
        X = init_population(dim, pop, lb, ub, rng)
        fit = np.array([fun(x) for x in X], dtype=float)
    else:
        X = np.array(X_init, copy=True)
        fit = np.array([fun(x) for x in X], dtype=float) if fit_init is None else np.array(fit_init, dtype=float)

    fes = pop
    best = float(np.min(fit))

    F = 0.5
    CR = 0.9

    curve: List[float] = []
    cp_idx = 0
    while cp_idx < len(cps) and cps[cp_idx] <= 0:
        curve.append(best); cp_idx += 1

    for _ in range(max_gens):
        if fes >= max_fes:
            break
        for i in range(pop):
            if fes >= max_fes:
                break
            idxs = rng.choice(pop, size=3, replace=False)
            a, b, c = X[idxs]
            v = a + F * (b - c)
            jrand = int(rng.integers(0, dim))
            u = X[i].copy()
            cross = rng.random(dim) < CR
            cross[jrand] = True
            u[cross] = v[cross]
            u = reflect_box(u, lb, ub)

            fu = float(fun(u)); fes += 1
            if fu <= fit[i]:
                X[i] = u
                fit[i] = fu
                if fu < best:
                    best = fu

            while cp_idx < len(cps) and fes >= cps[cp_idx]:
                curve.append(best); cp_idx += 1

    while cp_idx < len(cps):
        curve.append(best); cp_idx += 1

    return cps.astype(int), np.array(curve, dtype=float)


def run_PSO(fun, lb, ub, dim, max_fes, pop, rng, X0=None, n_points=220):
    max_gens = max(1, max_fes // pop)
    cps = make_checkpoints(max_fes, n_points)

    X_init, fit_init = unpack_init(X0)
    if X_init is None:
        X = init_population(dim, pop, lb, ub, rng)
        fit = np.array([fun(x) for x in X], dtype=float)
    else:
        X = np.array(X_init, copy=True)
        fit = np.array([fun(x) for x in X], dtype=float) if fit_init is None else np.array(fit_init, dtype=float)

    V = rng.normal(scale=0.1, size=X.shape) * (ub - lb)

    fes = pop
    pbest = X.copy()
    pbest_fit = fit.copy()
    gbest_idx = int(np.argmin(fit))
    gbest = X[gbest_idx].copy()
    gbest_fit = float(fit[gbest_idx])

    w0, w1 = 0.85, 0.35
    c1, c2 = 1.5, 1.5

    curve: List[float] = []
    cp_idx = 0
    while cp_idx < len(cps) and cps[cp_idx] <= 0:
        curve.append(gbest_fit); cp_idx += 1

    for g in range(max_gens):
        if fes >= max_fes:
            break

        w = w0 + (w1 - w0) * (g / max(1, max_gens))
        r1 = rng.random(size=X.shape)
        r2 = rng.random(size=X.shape)
        V = w * V + c1 * r1 * (pbest - X) + c2 * r2 * (gbest[None, :] - X)
        X = reflect_box(X + V, lb, ub)

        for i in range(pop):
            if fes >= max_fes:
                break
            fi = float(fun(X[i])); fes += 1
            if fi < pbest_fit[i]:
                pbest_fit[i] = fi
                pbest[i] = X[i].copy()
                if fi < gbest_fit:
                    gbest_fit = fi
                    gbest = X[i].copy()

            while cp_idx < len(cps) and fes >= cps[cp_idx]:
                curve.append(gbest_fit); cp_idx += 1

    while cp_idx < len(cps):
        curve.append(gbest_fit); cp_idx += 1

    return cps.astype(int), np.array(curve, dtype=float)


def run_SA(fun, lb, ub, dim, max_fes, pop_unused, rng, X0=None, n_points=220):
    cps = make_checkpoints(max_fes, n_points)

    X_init, fit_init = unpack_init(X0)
    if X_init is None:
        x = rng.uniform(lb, ub, size=dim)
        fx = float(fun(x))
        fes = 1
    else:
        X = np.array(X_init, copy=True)
        if fit_init is None:
            fit0 = np.array([fun(xx) for xx in X], dtype=float)
        else:
            fit0 = np.array(fit_init, dtype=float)
        # start from best in the shared initial population (counts pop evals)
        ib = int(np.argmin(fit0))
        x = X[ib].copy()
        fx = float(fit0[ib])
        fes = int(X.shape[0])

    best = float(fx)

    T0, Tmin = 1.0, 1e-4

    curve: List[float] = []
    cp_idx = 0
    while cp_idx < len(cps) and cps[cp_idx] <= 0:
        curve.append(best); cp_idx += 1

    # Ensure strict budget:
    # fes already pop (or 1), now use remaining evaluations
    remaining = max(0, max_fes - fes)
    for t in range(1, remaining + 1):
        if fes >= max_fes:
            break
        frac = t / max(1, remaining)
        T = T0 * (Tmin / T0) ** frac
        step_scale = (ub - lb) * (0.15 * (1 - frac) + 0.02)
        x_new = reflect_box(x + rng.normal(size=dim) * step_scale, lb, ub)
        f_new = float(fun(x_new))
        fes += 1

        dE = f_new - fx
        if dE <= 0 or rng.random() < math.exp(-dE / (T + 1e-12)):
            x, fx = x_new, f_new
            if fx < best:
                best = fx

        while cp_idx < len(cps) and fes >= cps[cp_idx]:
            curve.append(best); cp_idx += 1

    while cp_idx < len(cps):
        curve.append(best); cp_idx += 1

    return cps.astype(int), np.array(curve, dtype=float)


def run_SSA(fun, lb, ub, dim, max_fes, pop, rng, X0=None, n_points=220):
    max_gens = max(1, max_fes // pop)
    cps = make_checkpoints(max_fes, n_points)

    X_init, fit_init = unpack_init(X0)
    if X_init is None:
        X = init_population(dim, pop, lb, ub, rng)
        fit = np.array([fun(x) for x in X], dtype=float)
    else:
        X = np.array(X_init, copy=True)
        fit = np.array([fun(x) for x in X], dtype=float) if fit_init is None else np.array(fit_init, dtype=float)

    fes = pop
    best = float(np.min(fit))

    PD = 0.2
    SD = 0.1

    curve: List[float] = []
    cp_idx = 0
    while cp_idx < len(cps) and cps[cp_idx] <= 0:
        curve.append(best); cp_idx += 1

    for _ in range(max_gens):
        if fes >= max_fes:
            break

        idx = np.argsort(fit)
        X = X[idx]
        fit = fit[idx]

        best = float(fit[0])
        worst = float(fit[-1])
        x_best = X[0].copy()

        n_prod = max(1, int(PD * pop))
        n_aware = max(1, int(SD * pop))

        r2 = rng.random()
        for i in range(n_prod):
            if r2 < 0.8:
                X[i] = X[i] * np.exp(-i / (rng.random() * max(1, max_gens)))
            else:
                X[i] = X[i] + rng.normal(size=dim)

        for i in range(n_prod, pop):
            if i > pop / 2:
                X[i] = rng.normal(size=dim) * np.exp((X[-1] - X[i]) / ((i + 1) ** 2))
            else:
                A = np.sign(rng.normal(size=dim))
                X[i] = x_best + np.abs(X[i] - x_best) * A

        aware_idx = rng.choice(pop, size=n_aware, replace=False)
        for i in aware_idx:
            if fit[i] > best:
                X[i] = x_best + rng.normal(size=dim) * np.abs(X[i] - x_best)
            else:
                X[i] = X[i] + (2 * rng.random() - 1) * (np.abs(X[i] - X[-1]) / (fit[i] - worst + 1e-12))

        X = reflect_box(X, lb, ub)

        # Strict FE evaluation (per individual)
        new_fit = fit.copy()
        for i in range(pop):
            if fes >= max_fes:
                break
            new_fit[i] = float(fun(X[i]))
            fes += 1
            if new_fit[i] < best:
                best = float(new_fit[i])

            while cp_idx < len(cps) and fes >= cps[cp_idx]:
                curve.append(best); cp_idx += 1

        fit = new_fit

    while cp_idx < len(cps):
        curve.append(best); cp_idx += 1

    return cps.astype(int), np.array(curve, dtype=float)


# -----------------------------
# Analysis and plotting
# -----------------------------

def compute_curves_metrics(
    cps: np.ndarray,
    curves: Dict[str, List[np.ndarray]],
    f_opt: float,
    auc_fracs: Tuple[float, ...] = (0.10, 0.25, 0.50),
) -> Dict[str, Dict[str, float]]:
    """Per-run metric computation then median aggregation."""
    metrics: Dict[str, Dict[str, float]] = {}
    x = cps.astype(float)
    x_norm = x / max(1.0, float(cps[-1]))
    frac_masks = {p: (x_norm <= p + 1e-15) for p in auc_fracs}

    for alg, runs in curves.items():
        Y = np.vstack(runs)  # (reps, T)
        # per-run log-gap
        gaps = np.maximum(Y - f_opt, 1e-12)
        gaps = np.where(np.isfinite(gaps), gaps, 1e12)
        log_gap_runs = np.log10(gaps)

        auc_full_runs = np.array([auc_trapz(log_gap_runs[i], x_norm) for i in range(log_gap_runs.shape[0])], dtype=float)
        out = {
            "final_gap": float(np.median(gaps[:, -1])),
            "final_obj": float(np.median(Y[:, -1])),
            "auc_log10_gap": float(np.median(auc_full_runs)),
        }

        for p in auc_fracs:
            mask = frac_masks[p]
            idx = np.where(mask)[0]
            if idx.size < 2:
                out[f"auc@{p:.2f}"] = float(np.median(auc_full_runs))
            else:
                auc_slice = np.array([auc_trapz(log_gap_runs[i, idx], x_norm[idx]) for i in range(log_gap_runs.shape[0])], dtype=float)
                out[f"auc@{p:.2f}"] = float(np.median(auc_slice))

        metrics[alg] = out

    return metrics


def ert_first_hit(cps: np.ndarray, curve: np.ndarray, f_opt: float, tol: float) -> int:
    max_fes = int(cps[-1])
    gap = np.array([safe_gap(v, f_opt) for v in curve], dtype=float)
    hit = np.where(gap <= tol)[0]
    if hit.size == 0:
        return max_fes + 1
    return int(cps[int(hit[0])])


def compute_ert(curves: Dict[str, List[np.ndarray]], cps: np.ndarray, f_opt: float, tol: float) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for alg, runs in curves.items():
        hits = np.array([ert_first_hit(cps, r, f_opt, tol) for r in runs], dtype=float)
        out[alg] = float(np.mean(hits))
    return out


def plot_convergence(
    title: str,
    cps: np.ndarray,
    curves: Dict[str, List[np.ndarray]],
    f_opt: float,
    outpath: str,
    alg_order: List[str],
    colors: Dict[str, str],
    xaxis: str,
    pop: int,
):
    if xaxis == "iter":
        x = cps.astype(float) / max(1.0, float(pop))
        x_label = "Iteration"
    else:
        x = cps.astype(float)
        x_label = "Function evaluations"

    fig, ax = plt.subplots(figsize=(10, 6))
    for alg in alg_order:
        runs = curves[alg]
        Y = np.vstack(runs)
        med = np.median(Y, axis=0)
        q1 = np.quantile(Y, 0.25, axis=0)
        q3 = np.quantile(Y, 0.75, axis=0)

        med_gap = np.array([safe_gap(v, f_opt) for v in med], dtype=float)
        q1_gap = np.array([safe_gap(v, f_opt) for v in q1], dtype=float)
        q3_gap = np.array([safe_gap(v, f_opt) for v in q3], dtype=float)

        ax.plot(x, med_gap, lw=2.6, color=colors.get(alg, NEUTRAL_GRAY), label=alg)
        ax.fill_between(x, q1_gap, q3_gap, color=colors.get(alg, NEUTRAL_GRAY), alpha=0.18, linewidth=0)

    ax.set_yscale("log")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Best-so-far gap to optimum")
    ax.set_title(title)
    ax.legend(frameon=False, loc="best", ncol=1)
    ax.margins(x=0.02)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def plot_rank_bar(title: str, avg_ranks: Dict[str, float], outpath: str, colors: Dict[str, str], highlight: str = "ARL"):
    items = sorted(avg_ranks.items(), key=lambda kv: kv[1])
    names = [k for k, _ in items]
    vals = [v for _, v in items]

    fig, ax = plt.subplots(figsize=(10, 5))
    bar_colors = [colors.get(n, "#7A7A7A") if n == highlight else "#7A7A7A" for n in names]
    ax.barh(names, vals, color=bar_colors, edgecolor="black", linewidth=1.0)
    ax.invert_yaxis()
    ax.set_xlabel("Average rank (lower is better)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def plot_ert_bar(title: str, ert_vals: Dict[str, float], outpath: str, colors: Dict[str, str], highlight: str = "ARL"):
    items = sorted(ert_vals.items(), key=lambda kv: kv[1])
    names = [k for k, _ in items]
    vals = [float(v) for _, v in items]

    fig, ax = plt.subplots(figsize=(10, 5))
    bar_colors = [colors.get(n, "#7A7A7A") if n == highlight else "#7A7A7A" for n in names]
    ax.barh(names, vals, color=bar_colors, edgecolor="black", linewidth=1.0)
    ax.invert_yaxis()
    ax.set_xlabel("ERT (function evaluations) to reach target tolerance")
    ax.set_title(title)

    vmin = max(1.0, float(np.min(vals)))
    vmax = float(np.max(vals))
    if vmax / vmin > 50.0:
        ax.set_xscale("log")

    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def plot_rank_diff_vs_ARL(title: str, avg_ranks: Dict[str, float], outpath: str, ref: str = "ARL"):
    r0 = float(avg_ranks[ref])
    items = sorted([(k, v - r0) for k, v in avg_ranks.items()], key=lambda kv: kv[1])
    names = [k for k, _ in items]
    diffs = [d for _, d in items]

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, n in enumerate(names):
        c = "#7A7A7A" if n != ref else NATURE.get(ref, "#D55E00")
        ax.barh(n, diffs[i], color=c, edgecolor="black", linewidth=1.0)
    ax.axvline(0.0, color="black", lw=1.2)
    ax.invert_yaxis()
    ax.set_xlabel("Average rank difference vs ARL (positive = worse than ARL)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def plot_winrate_matrix(title: str, winrate: np.ndarray, algs: List[str], outpath: str, cmap=None, annotate: bool = True):
    fig, ax = plt.subplots(figsize=(9, 7))
    cmap = make_cmap_nature() if cmap is None else cmap
    im = ax.imshow(winrate, vmin=0.0, vmax=1.0, cmap=cmap)

    ax.set_xticks(range(len(algs)))
    ax.set_xticklabels(algs, rotation=35, ha="right")
    ax.set_yticks(range(len(algs)))
    ax.set_yticklabels(algs)
    ax.set_title(title)

    if annotate:
        for i in range(len(algs)):
            for j in range(len(algs)):
                if i == j:
                    continue
                v = float(winrate[i, j])
                txt_color = "white" if v > 0.70 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=10, color=txt_color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Win rate")
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def pairwise_winrate_from_metric(per_func_metric: List[Dict[str, float]], algs: List[str]) -> np.ndarray:
    N = len(per_func_metric)
    W = np.zeros((len(algs), len(algs)), dtype=float)
    for m in per_func_metric:
        for i, ai in enumerate(algs):
            for j, aj in enumerate(algs):
                if i == j:
                    continue
                if m[ai] < m[aj]:
                    W[i, j] += 1
                elif np.isclose(m[ai], m[aj], rtol=0, atol=1e-15):
                    W[i, j] += 0.5
    W /= max(1, N)
    return W


def flatten_dict(d, prefix: str = "", sep: str = ".") -> List[Tuple[str, str]]:
    items: List[Tuple[str, str]] = []
    for k, v in d.items():
        key = f"{prefix}{sep}{k}" if prefix else str(k)
        if isinstance(v, dict):
            items.extend(flatten_dict(v, key, sep=sep))
        elif isinstance(v, (list, tuple, np.ndarray)):
            items.append((key, str(list(v))))
        else:
            items.append((key, str(v)))
    return items


def save_kv_csv(path: str, d: Dict):
    rows = flatten_dict(d)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["key", "value"])
        w.writerows(rows)


def save_matrix_csv(path: str, M: np.ndarray, labels: List[str]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([""] + labels)
        for i, lab in enumerate(labels):
            w.writerow([lab] + [f"{float(x):.6g}" for x in M[i]])


def save_csv_table(path: str, header: List[str], rows: List[List]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def save_per_function_metrics_csv(
    path: str,
    function: str,
    group: str,
    auc_log10_gap: Dict[str, float],
    final_gap: Dict[str, float],
    ert_eval_to_tol: Dict[str, float],
    ert_tol: float,
    avg_rank_auc: Dict[str, float],
    alg_order: List[str],
):
    header = ["function", "group", "algorithm", "auc_log10_gap", "final_gap", "ert_eval_to_tol", "ert_tol", "rank_auc"]
    rows = []
    for a in alg_order:
        rows.append([
            function, group, a,
            float(auc_log10_gap[a]),
            float(final_gap[a]),
            float(ert_eval_to_tol[a]),
            float(ert_tol),
            float(avg_rank_auc[a]),
        ])
    save_csv_table(path, header, rows)


# --- statistics (no scipy) ---

def gammainc_Q(a: float, x: float) -> float:
    if x < 0 or a <= 0:
        return 1.0
    if x == 0:
        return 1.0
    if x < a + 1.0:
        ap = a
        summ = 1.0 / a
        delt = summ
        for _ in range(1, 200):
            ap += 1.0
            delt *= x / ap
            summ += delt
            if abs(delt) < abs(summ) * 1e-12:
                break
        P = summ * math.exp(-x + a * math.log(x) - math.lgamma(a))
        return max(0.0, 1.0 - P)
    else:
        b = x + 1.0 - a
        c = 1.0 / 1e-30
        d = 1.0 / b
        h = d
        for i in range(1, 200):
            an = -i * (i - a)
            b += 2.0
            d = an * d + b
            if abs(d) < 1e-30:
                d = 1e-30
            c = b + an / c
            if abs(c) < 1e-30:
                c = 1e-30
            d = 1.0 / d
            delt = d * c
            h *= delt
            if abs(delt - 1.0) < 1e-12:
                break
        Q = h * math.exp(-x + a * math.log(x) - math.lgamma(a))
        return float(max(0.0, min(1.0, Q)))


def chi2_sf_approx(x: float, df: int) -> float:
    return float(gammainc_Q(df / 2.0, x / 2.0))


def friedman_test(ranks_per_func: List[Dict[str, float]]) -> Tuple[float, float]:
    algs = list(ranks_per_func[0].keys())
    k = len(algs)
    N = len(ranks_per_func)
    R = np.zeros(k, dtype=float)
    for d in ranks_per_func:
        R += np.array([d[a] for a in algs], dtype=float)
    Rbar = R / N
    chi2 = (12 * N / (k * (k + 1))) * float(np.sum((Rbar - (k + 1) / 2) ** 2))
    p = chi2_sf_approx(chi2, k - 1)
    return float(chi2), float(p)


def nemenyi_cd(k: int, N: int, alpha: float = 0.05) -> float:
    # alpha not used (fixed lookup at 0.05), consistent with your original
    q_table_005 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}
    q = q_table_005.get(k, 3.164 + 0.08 * (k - 10))
    return float(q * math.sqrt(k * (k + 1) / (6.0 * N)))


def plot_cd_diagram(title: str, avg_ranks: Dict[str, float], cd: float, outpath: str):
    items = sorted(avg_ranks.items(), key=lambda kv: kv[1])
    names = [k for k, _ in items]
    ranks = [v for _, v in items]

    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.set_title(title)
    ax.set_xlabel("Average rank (lower is better)")
    ax.set_yticks([])
    ax.set_xlim(1, max(ranks) + 0.6)
    ax.grid(True, axis="x")

    y0 = 0.5
    for n, r in zip(names, ranks):
        ax.scatter([r], [y0], s=140, color=NATURE.get(n, "#7A7A7A"), edgecolor="black", zorder=3)
        ax.text(r, y0 + 0.12, n, ha="center", va="bottom", fontsize=11)

    ax.plot([1.05, 1.05 + cd], [0.92, 0.92], lw=4, color="black")
    ax.text(1.05 + cd / 2, 0.98, f"CD = {cd:.3f}", ha="center", va="bottom", fontsize=12)

    ax.set_ylim(0.2, 1.15)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# ARL diagnostics (new; additive)
# -----------------------------

def _save_arl_diagnostics(diag_dir: str, tf_name: str, group_name: str, diag_runs: List[Dict[str, np.ndarray]]):
    """Aggregate diagnostics across reps (median), and output plots/CSV."""
    if not diag_runs:
        return
    ensure_dir(diag_dir)
    set_pub_style()

    # Align by minimum length across reps
    L = min(int(d["gen"].shape[0]) for d in diag_runs)
    keys = list(diag_runs[0].keys())
    stack = {k: np.vstack([d[k][:L] for d in diag_runs]) for k in keys if k not in {"gen"}}
    gen = diag_runs[0]["gen"][:L]

    # Median series
    med = {k: np.median(stack[k], axis=0) for k in stack.keys()}

    # Save CSV
    csv_path = os.path.join(diag_dir, f"{tf_name}_{group_name}_diagnostics.csv")
    header = ["gen"] + list(med.keys())
    rows = []
    for i in range(L):
        row = [int(gen[i])]
        for k in med.keys():
            row.append(float(med[k][i]))
        rows.append(row)
    save_csv_table(csv_path, header, rows)

    # Plot 1: trigger and k
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.plot(gen, med["use_regions"], lw=2.2, label="use_regions (median)")
    ax.plot(gen, med["k"], lw=2.2, label="k regions (median)")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Value")
    ax.set_title(f"{tf_name} | {group_name} | Trigger & region count")
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(diag_dir, f"{tf_name}_{group_name}_trigger_k.png"), bbox_inches="tight")
    plt.close(fig)

    # Plot 2: operator probabilities (median)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(gen, med["op_p0"], lw=2.0, label="op0 ARL")
    ax.plot(gen, med["op_p1"], lw=2.0, label="op1 DE/pbest")
    ax.plot(gen, med["op_p2"], lw=2.0, label="op2 Levy")
    ax.plot(gen, med["op_p3"], lw=2.0, label="op3 Local")
    ax.plot(gen, med["op_p4"], lw=2.0, label="op4 EliteDiag")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Probability")
    ax.set_title(f"{tf_name} | {group_name} | Operator mixture (median)")
    ax.legend(frameon=False, ncol=3, loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(diag_dir, f"{tf_name}_{group_name}_op_probs.png"), bbox_inches="tight")
    plt.close(fig)

    # Plot 3: diversity/dispersion/evr1/ema_score
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.plot(gen, med["div_norm"], lw=2.2, label="div_norm")
    ax.plot(gen, med["rel_disp"], lw=2.2, label="rel_disp")
    ax.plot(gen, med["evr1"], lw=2.2, label="evr1")
    ax.plot(gen, med["ema_score"], lw=2.2, label="ema_score")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Value")
    ax.set_title(f"{tf_name} | {group_name} | Trigger features (median)")
    ax.legend(frameon=False, ncol=2, loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(diag_dir, f"{tf_name}_{group_name}_trigger_features.png"), bbox_inches="tight")
    plt.close(fig)

    # Plot 4: operator success rates (median)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(gen, med["op_succ0"], lw=2.0, label="succ op0")
    ax.plot(gen, med["op_succ1"], lw=2.0, label="succ op1")
    ax.plot(gen, med["op_succ2"], lw=2.0, label="succ op2")
    ax.plot(gen, med["op_succ3"], lw=2.0, label="succ op3")
    ax.plot(gen, med["op_succ4"], lw=2.0, label="succ op4")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Success rate")
    ax.set_title(f"{tf_name} | {group_name} | Operator success (median)")
    ax.legend(frameon=False, ncol=3, loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(diag_dir, f"{tf_name}_{group_name}_op_success.png"), bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# Runner
# -----------------------------

@dataclass
class RunConfig:
    dim: int
    pop: int
    max_fes: int
    reps: int
    checkpoints: int
    suite_seed: int
    outdir: str
    alpha: float
    target_tol: float
    xaxis: str


def run_group(group_name: str, suite: List[TestFunction], algs: Dict[str, Callable], cfg: RunConfig) -> Dict:
    outdir = ensure_dir(os.path.join(cfg.outdir, group_name))
    set_pub_style()

    group_algs = list(algs.keys())
    if "ARL" in group_algs:
        group_algs = ["ARL"] + [a for a in group_algs if a != "ARL"]

    per_func_auc_metric: List[Dict[str, float]] = []
    per_func_auc10_metric: List[Dict[str, float]] = []
    per_func_auc25_metric: List[Dict[str, float]] = []
    per_func_auc50_metric: List[Dict[str, float]] = []
    per_func_rank_auc: List[Dict[str, float]] = []
    per_func_ert_metric: List[Dict[str, float]] = []

    # diagnostics store (ARL only)
    diag_root = ensure_dir(os.path.join(cfg.outdir, "ARL_Diagnostics", group_name))

    rows = []

    for tf in suite:
        func_dir = ensure_dir(os.path.join(outdir, tf.name))
        curves: Dict[str, List[np.ndarray]] = {a: [] for a in group_algs}
        cps_ref = None

        diag_runs: List[Dict[str, np.ndarray]] = []

        for rep in range(cfg.reps):
            rep_seed = (cfg.suite_seed * 1000003 + stable_hash32(tf.name) + rep * 977) & 0xFFFFFFFF
            rng0 = np.random.default_rng(rep_seed)
            X0 = init_population(tf.dim, cfg.pop, tf.lb, tf.ub, rng0)
            # cache initial fit (counts pop evals for each algorithm logically; reuse for speed/fairness)
            fit0 = np.array([tf.f(x) for x in X0], dtype=float)

            for alg_name, runner in algs.items():
                rng = np.random.default_rng((rep_seed + stable_hash32(alg_name)) & 0xFFFFFFFF)
                if alg_name == "ARL":
                    res = runner(tf.f, tf.lb, tf.ub, tf.dim, cfg.max_fes, cfg.pop, rng, X0=(X0, fit0), n_points=cfg.checkpoints)
                    if isinstance(res, tuple) and len(res) == 3:
                        cps, curve, diag = res
                        if diag is not None:
                            diag_runs.append(diag)
                    else:
                        cps, curve = res
                else:
                    cps, curve = runner(tf.f, tf.lb, tf.ub, tf.dim, cfg.max_fes, cfg.pop, rng, X0=(X0, fit0), n_points=cfg.checkpoints)

                if cps_ref is None:
                    cps_ref = cps
                curves[alg_name].append(curve)

        assert cps_ref is not None

        # Save ARL diagnostics (additive, not affecting existing outputs)
        diag_dir = ensure_dir(os.path.join(diag_root, tf.name))
        _save_arl_diagnostics(diag_dir, tf.name, group_name, diag_runs)

        metrics = compute_curves_metrics(cps_ref, curves, tf.f_opt)
        m_auc = {a: metrics[a]["auc_log10_gap"] for a in group_algs}
        m_auc10 = {a: metrics[a]["auc@0.10"] for a in group_algs}
        m_auc25 = {a: metrics[a]["auc@0.25"] for a in group_algs}
        m_auc50 = {a: metrics[a]["auc@0.50"] for a in group_algs}
        m_final_gap = {a: metrics[a]["final_gap"] for a in group_algs}
        m_ert = compute_ert(curves, cps_ref, tf.f_opt, cfg.target_tol)
        ranks = rank_lower_is_better(m_auc)

        per_func_auc_metric.append(m_auc)
        per_func_auc10_metric.append(m_auc10)
        per_func_auc25_metric.append(m_auc25)
        per_func_auc50_metric.append(m_auc50)
        per_func_rank_auc.append(ranks)
        per_func_ert_metric.append(m_ert)

        title = f"{tf.name}: {group_name} (median & IQR)"
        plot_convergence(
            title=title,
            cps=cps_ref,
            curves=curves,
            f_opt=tf.f_opt,
            outpath=os.path.join(func_dir, f"{tf.name}_{group_name}_convergence.png"),
            alg_order=group_algs,
            colors=NATURE,
            xaxis=cfg.xaxis,
            pop=cfg.pop,
        )

        save_per_function_metrics_csv(
            path=os.path.join(func_dir, f"{tf.name}_{group_name}_metrics.csv"),
            function=tf.name,
            group=group_name,
            auc_log10_gap=m_auc,
            final_gap=m_final_gap,
            ert_eval_to_tol=m_ert,
            ert_tol=cfg.target_tol,
            avg_rank_auc=ranks,
            alg_order=group_algs,
        )

        row = [tf.name]
        for a in group_algs:
            row += [m_auc[a], m_final_gap[a], m_ert[a], ranks[a]]
        rows.append(row)

    avg_ranks_auc = {a: float(np.mean([r[a] for r in per_func_rank_auc])) for a in group_algs}
    avg_ert = {a: float(np.mean([m[a] for m in per_func_ert_metric])) for a in group_algs}

    avg_ranks_auc10 = rank_lower_is_better({a: float(np.mean([m[a] for m in per_func_auc10_metric])) for a in group_algs})
    avg_ranks_auc25 = rank_lower_is_better({a: float(np.mean([m[a] for m in per_func_auc25_metric])) for a in group_algs})
    avg_ranks_auc50 = rank_lower_is_better({a: float(np.mean([m[a] for m in per_func_auc50_metric])) for a in group_algs})

    W = pairwise_winrate_from_metric(per_func_auc_metric, group_algs)
    chi2, p = friedman_test(per_func_rank_auc)
    cd = nemenyi_cd(k=len(group_algs), N=len(suite), alpha=cfg.alpha)

    plot_rank_bar(
        title=f"{group_name} | Average rank (AUC of log10 gap)",
        avg_ranks=avg_ranks_auc,
        outpath=os.path.join(outdir, f"{group_name}_avg_rank_auc.png"),
        colors=NATURE,
        highlight="ARL",
    )
    plot_ert_bar(
        title=f"{group_name} | Average ERT (tol={cfg.target_tol:g})",
        ert_vals=avg_ert,
        outpath=os.path.join(outdir, f"{group_name}_avg_ert_eval.png"),
        colors=NATURE,
        highlight="ARL",
    )
    plot_rank_bar(
        title=f"{group_name} | Average rank (AUC@0.10 of log10 gap)",
        avg_ranks=avg_ranks_auc10,
        outpath=os.path.join(outdir, f"{group_name}_avg_rank_auc10.png"),
        colors=NATURE,
        highlight="ARL",
    )
    plot_rank_bar(
        title=f"{group_name} | Average rank (AUC@0.25 of log10 gap)",
        avg_ranks=avg_ranks_auc25,
        outpath=os.path.join(outdir, f"{group_name}_avg_rank_auc25.png"),
        colors=NATURE,
        highlight="ARL",
    )
    plot_rank_bar(
        title=f"{group_name} | Average rank (AUC@0.50 of log10 gap)",
        avg_ranks=avg_ranks_auc50,
        outpath=os.path.join(outdir, f"{group_name}_avg_rank_auc50.png"),
        colors=NATURE,
        highlight="ARL",
    )

    plot_rank_diff_vs_ARL(
        title=f"{group_name} | Rank difference vs ARL (AUC)",
        avg_ranks=avg_ranks_auc,
        outpath=os.path.join(outdir, f"{group_name}_rank_diff_vs_ARL_auc.png"),
        ref="ARL",
    )
    plot_winrate_matrix(
        title=f"{group_name} | Win-rate matrix (AUC)",
        winrate=W,
        algs=group_algs,
        outpath=os.path.join(outdir, f"{group_name}_winrate_matrix_auc.png"),
        cmap=make_cmap_nature(),
        annotate=True,
    )
    plot_cd_diagram(
        title=f"{group_name} | CD diagram (Nemenyi, alpha={cfg.alpha})",
        avg_ranks=avg_ranks_auc,
        cd=cd,
        outpath=os.path.join(outdir, f"{group_name}_cd_diagram_auc.png"),
    )

    header = ["Function"]
    for a in group_algs:
        header += [f"{a}_AUC_log10_gap", f"{a}_Final_gap", f"{a}_ERT_eval(tol)", f"{a}_Rank_AUC"]
    save_csv_table(os.path.join(outdir, f"{group_name}_per_function_summary.csv"), header, rows)

    header2 = ["Function"]
    for a in group_algs:
        header2 += [f"{a}_AUC10", f"{a}_AUC25", f"{a}_AUC50"]
    rows2 = []
    for i, tf in enumerate(suite):
        r = [tf.name]
        for a in group_algs:
            r += [
                per_func_auc10_metric[i][a],
                per_func_auc25_metric[i][a],
                per_func_auc50_metric[i][a],
            ]
        rows2.append(r)
    save_csv_table(os.path.join(outdir, f"{group_name}_per_function_auc_slices.csv"), header2, rows2)

    save_kv_csv(os.path.join(outdir, f"{group_name}_overall_summary.csv"), {
        "group": group_name,
        "dim": cfg.dim,
        "pop": cfg.pop,
        "max_fes": cfg.max_fes,
        "reps": cfg.reps,
        "n_functions": len(suite),
        "ert_tol": cfg.target_tol,
        "friedman_chi2": chi2,
        "friedman_p_approx": p,
        "nemenyi_cd": cd,
        "alg_order": group_algs,
        "note": "Friedman p-value is a chi-square approximation (no scipy). Recompute with scipy for final manuscript if desired.",
    })
    save_csv_table(os.path.join(outdir, f"{group_name}_avg_ranks_auc.csv"),
                   ["algorithm", "avg_rank_auc"], [[a, avg_ranks_auc[a]] for a in group_algs])
    save_csv_table(os.path.join(outdir, f"{group_name}_avg_ert_eval.csv"),
                   ["algorithm", "avg_ert_eval"], [[a, avg_ert[a]] for a in group_algs])
    save_matrix_csv(os.path.join(outdir, f"{group_name}_winrate_matrix_auc.csv"), W, group_algs)
    save_csv_table(os.path.join(outdir, f"{group_name}_friedman_test.csv"), ["chi2", "p_approx"], [[chi2, p]])

    return {
        "avg_ranks_auc": avg_ranks_auc,
        "avg_ranks_auc10": avg_ranks_auc10,
        "avg_ranks_auc25": avg_ranks_auc25,
        "avg_ranks_auc50": avg_ranks_auc50,
        "winrate_auc": W.tolist(),
        "friedman": {"chi2": chi2, "p_approx": p},
        "cd": cd,
        "alg_order": group_algs,
    }


# -----------------------------
# FrameworkAdvantage (derived; unchanged in spirit)
# -----------------------------

def _parse_group_per_function_summary(path: str) -> Tuple[List[str], List[str], Dict[str, Dict[str, float]]]:
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    if len(header) < 2 or header[0].strip().lower() != "function":
        raise ValueError(f"Unexpected header in: {path}")

    algs: List[str] = []
    for h in header[1:]:
        if h.endswith("_AUC_log10_gap"):
            algs.append(h[: -len("_AUC_log10_gap")])

    if not algs:
        raise ValueError(f"Could not detect algorithms in: {path}")

    col_idx = {a: header.index(f"{a}_AUC_log10_gap") for a in algs}

    funcs: List[str] = []
    auc: Dict[str, Dict[str, float]] = {}
    for r in rows:
        if not r:
            continue
        fn = r[0]
        funcs.append(fn)
        auc[fn] = {}
        for a in algs:
            auc[fn][a] = float(r[col_idx[a]])
    return funcs, algs, auc


def _parse_group_auc_slices(path: str) -> Tuple[List[str], List[str], Dict[str, Dict[str, Tuple[float, float, float]]]]:
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    if len(header) < 2 or header[0].strip().lower() != "function":
        raise ValueError(f"Unexpected header in: {path}")

    algs: List[str] = []
    for h in header[1:]:
        if h.endswith("_AUC10"):
            algs.append(h[: -len("_AUC10")])

    if not algs:
        raise ValueError(f"Could not detect algorithms in: {path}")

    idx10 = {a: header.index(f"{a}_AUC10") for a in algs}
    idx25 = {a: header.index(f"{a}_AUC25") for a in algs}
    idx50 = {a: header.index(f"{a}_AUC50") for a in algs}

    funcs: List[str] = []
    slices: Dict[str, Dict[str, Tuple[float, float, float]]] = {}
    for r in rows:
        fn = r[0]
        funcs.append(fn)
        slices[fn] = {}
        for a in algs:
            slices[fn][a] = (float(r[idx10[a]]), float(r[idx25[a]]), float(r[idx50[a]]))
    return funcs, algs, slices


def _save_framework_advantage(
    outdir: str,
    funcs: List[str],
    baseline_algs: List[str],
    auc_full: Dict[str, Dict[str, float]],
    auc_slices: Dict[str, Dict[str, Tuple[float, float, float]]],
    ref_alg: str = "ARL",
):
    ensure_dir(outdir)
    set_pub_style()

    competitors = [a for a in baseline_algs if a != ref_alg]
    if not competitors:
        return

    delta_full = []
    for fn in funcs:
        arl = auc_full[fn][ref_alg]
        best_other = min(auc_full[fn][a] for a in competitors)
        delta_full.append((fn, arl - best_other))

    csv_path = os.path.join(outdir, "FrameworkAdvantage_delta_auc_full.csv")
    save_csv_table(
        csv_path,
        ["Function", "ARL_AUC", "BestBaseline_AUC", "Delta(ARL-BestBaseline)"],
        [[fn, auc_full[fn][ref_alg], min(auc_full[fn][a] for a in competitors), d] for fn, d in delta_full]
    )

    delta_full_sorted = sorted(delta_full, key=lambda x: x[1])
    fns = [x[0] for x in delta_full_sorted]
    ds = np.array([x[1] for x in delta_full_sorted], dtype=float)

    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.bar(range(len(fns)), ds, color=NATURE.get(ref_alg, "#D55E00"), edgecolor="black", linewidth=0.8)
    ax.axhline(0.0, color="black", lw=1.2)
    ax.set_xlim(-0.5, len(fns) - 0.5)
    ax.set_xticks(range(len(fns)))
    ax.set_xticklabels(fns, rotation=70, ha="right", fontsize=9)
    ax.set_ylabel("ΔAUC (ARL − best baseline)")
    ax.set_title("Framework advantage per function (ΔAUC; negative = ARL better)")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "FrameworkAdvantage_delta_auc_full_sorted_bar.png"), bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.hist(ds, bins=14, edgecolor="black", linewidth=0.8, color=NATURE.get(ref_alg, "#D55E00"), alpha=0.85)
    ax.axvline(0.0, color="black", lw=1.2)
    ax.set_xlabel("ΔAUC (ARL − best baseline)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of ΔAUC over functions")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "FrameworkAdvantage_delta_auc_full_hist.png"), bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.3, 4.2))
    ax.boxplot(ds, vert=True, widths=0.5, patch_artist=True,
               boxprops=dict(facecolor="#F3F3F3", edgecolor="black"),
               medianprops=dict(color="black", linewidth=1.5),
               whiskerprops=dict(color="black"),
               capprops=dict(color="black"))
    ax.axhline(0.0, color="black", lw=1.2)
    ax.set_ylabel("ΔAUC (ARL − best baseline)")
    ax.set_xticks([1])
    ax.set_xticklabels(["All functions"])
    ax.set_title("ΔAUC summary")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "FrameworkAdvantage_delta_auc_full_box.png"), bbox_inches="tight")
    plt.close(fig)

    csv_path2 = os.path.join(outdir, "FrameworkAdvantage_delta_auc_slices.csv")
    rows2 = []
    for fn in funcs:
        arl10, arl25, arl50 = auc_slices[fn][ref_alg]
        best10 = min(auc_slices[fn][a][0] for a in competitors)
        best25 = min(auc_slices[fn][a][1] for a in competitors)
        best50 = min(auc_slices[fn][a][2] for a in competitors)
        rows2.append([fn, arl10, best10, arl10 - best10, arl25, best25, arl25 - best25, arl50, best50, arl50 - best50])

    save_csv_table(
        csv_path2,
        ["Function",
         "ARL_AUC10", "BestBaseline_AUC10", "Delta10",
         "ARL_AUC25", "BestBaseline_AUC25", "Delta25",
         "ARL_AUC50", "BestBaseline_AUC50", "Delta50"],
        rows2
    )

    d10 = np.array([r[3] for r in rows2], dtype=float)
    d25 = np.array([r[6] for r in rows2], dtype=float)
    d50 = np.array([r[9] for r in rows2], dtype=float)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    xs = np.arange(3)
    meds = [float(np.median(d10)), float(np.median(d25)), float(np.median(d50))]
    iqr_low = [float(np.quantile(d10, 0.25)), float(np.quantile(d25, 0.25)), float(np.quantile(d50, 0.25))]
    iqr_hi = [float(np.quantile(d10, 0.75)), float(np.quantile(d25, 0.75)), float(np.quantile(d50, 0.75))]
    err_low = np.array(meds) - np.array(iqr_low)
    err_hi = np.array(iqr_hi) - np.array(meds)
    ax.bar(xs, meds, color=NATURE.get(ref_alg, "#D55E00"), edgecolor="black", linewidth=0.9)
    ax.errorbar(xs, meds, yerr=[err_low, err_hi], fmt="none", ecolor="black", elinewidth=1.2, capsize=6)
    ax.axhline(0.0, color="black", lw=1.2)
    ax.set_xticks(xs)
    ax.set_xticklabels(["AUC@0.10", "AUC@0.25", "AUC@0.50"])
    ax.set_ylabel("Median ΔAUC (ARL − best baseline)")
    ax.set_title("Early-/mid-budget framework advantage (median ± IQR)")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "FrameworkAdvantage_delta_auc_slices_bar.png"), bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.boxplot([d10, d25, d50], labels=["AUC@0.10", "AUC@0.25", "AUC@0.50"],
               patch_artist=True,
               boxprops=dict(facecolor="#F3F3F3", edgecolor="black"),
               medianprops=dict(color="black", linewidth=1.5),
               whiskerprops=dict(color="black"),
               capprops=dict(color="black"))
    ax.axhline(0.0, color="black", lw=1.2)
    ax.set_ylabel("ΔAUC (ARL − best baseline)")
    ax.set_title("ΔAUC distributions across budget slices")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "FrameworkAdvantage_delta_auc_slices_box.png"), bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# AblationContribution (new; derived, additive)
# -----------------------------

def _save_ablation_contribution(outdir: str, ablation_dir: str, ref_alg: str = "ARL"):
    """Compute ΔAUC(ablation - ARL). Positive means ablation worse (component helps)."""
    ensure_dir(outdir)
    set_pub_style()

    funcs, algs, auc_full = _parse_group_per_function_summary(os.path.join(ablation_dir, "Ablations_per_function_summary.csv"))
    # pick ablations only
    abls = [a for a in algs if a != ref_alg]
    if not abls:
        return

    rows = []
    for fn in funcs:
        arl = auc_full[fn][ref_alg]
        for a in abls:
            rows.append([fn, a, auc_full[fn][a], arl, auc_full[fn][a] - arl])

    save_csv_table(
        os.path.join(outdir, "AblationContribution_delta_auc_full.csv"),
        ["Function", "Ablation", "AUC_Ablation", "AUC_ARL", "Delta(Ablation-ARL)"],
        rows
    )

    # For each ablation: sorted bar of per-function deltas
    for a in abls:
        ds = [(fn, auc_full[fn][a] - auc_full[fn][ref_alg]) for fn in funcs]
        ds_sorted = sorted(ds, key=lambda x: x[1], reverse=True)
        fns = [x[0] for x in ds_sorted]
        vals = np.array([x[1] for x in ds_sorted], dtype=float)

        fig, ax = plt.subplots(figsize=(12, 4.8))
        ax.bar(range(len(fns)), vals, color="#7A7A7A", edgecolor="black", linewidth=0.8)
        ax.axhline(0.0, color="black", lw=1.2)
        ax.set_xlim(-0.5, len(fns) - 0.5)
        ax.set_xticks(range(len(fns)))
        ax.set_xticklabels(fns, rotation=70, ha="right", fontsize=9)
        ax.set_ylabel("ΔAUC (Ablation − ARL)")
        ax.set_title(f"Ablation contribution per function: {a} (positive = component helps)")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"AblationContribution_{a}_sorted_bar.png"), bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7.0, 4.5))
        ax.hist(vals, bins=14, edgecolor="black", linewidth=0.8, color="#7A7A7A", alpha=0.85)
        ax.axvline(0.0, color="black", lw=1.2)
        ax.set_xlabel("ΔAUC (Ablation − ARL)")
        ax.set_ylabel("Count")
        ax.set_title(f"ΔAUC distribution: {a}")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"AblationContribution_{a}_hist.png"), bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        ax.boxplot(vals, vert=True, widths=0.5, patch_artist=True,
                   boxprops=dict(facecolor="#F3F3F3", edgecolor="black"),
                   medianprops=dict(color="black", linewidth=1.5),
                   whiskerprops=dict(color="black"),
                   capprops=dict(color="black"))
        ax.axhline(0.0, color="black", lw=1.2)
        ax.set_ylabel("ΔAUC (Ablation − ARL)")
        ax.set_xticks([1])
        ax.set_xticklabels([a])
        ax.set_title(f"ΔAUC summary: {a}")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"AblationContribution_{a}_box.png"), bbox_inches="tight")
        plt.close(fig)


# -----------------------------
# CLI / main
# -----------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="ARL benchmark (FE-budget, publication-style plots).")
    ap.add_argument("--dim", type=int, default=10)
    ap.add_argument("--pop", type=int, default=40)
    ap.add_argument("--max_fes", type=int, default=8000)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--checkpoints", type=int, default=220)
    ap.add_argument("--n_funcs", type=int, default=30)
    ap.add_argument("--suite_seed", type=int, default=42)
    ap.add_argument("--outdir", type=str, default="outputs_arl_pub")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--target_tol", type=float, default=1e-8)
    ap.add_argument(
        "--xaxis",
        type=str,
        default="iter",
        choices=["iter", "fes"],
        help="X-axis for convergence plots: iter (=fes/pop) or fes.",
    )
    ap.add_argument(
        "--skip_trigger",
        action="store_true",
        help="Skip TriggerStudy group (adaptive vs always/never/two_phase).",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    outdir = ensure_dir(args.outdir)

    cfg = RunConfig(
        dim=args.dim,
        pop=args.pop,
        max_fes=args.max_fes,
        reps=args.reps,
        checkpoints=args.checkpoints,
        suite_seed=args.suite_seed,
        outdir=outdir,
        alpha=args.alpha,
        target_tol=args.target_tol,
        xaxis=args.xaxis,
    )

    suite = make_suite(cfg.dim, cfg.suite_seed, n_funcs=args.n_funcs)

    baselines = {
        "ARL": lambda fun, lb, ub, dim, max_fes, pop, rng, X0=None, n_points=220: run_ARL(
            fun, lb, ub, dim, max_fes, pop, rng, X0=X0, ablation=None, n_points=n_points, record_diag=True
        ),
        "DE": run_DE,
        "PSO": run_PSO,
        "SA": run_SA,
        "SSA": run_SSA,
    }

    triggers = {
        "ARL": lambda fun, lb, ub, dim, max_fes, pop, rng, X0=None, n_points=220: run_ARL(
            fun, lb, ub, dim, max_fes, pop, rng, X0=X0, ablation=None, trigger_mode="adaptive",
            n_points=n_points, record_diag=True
        ),
        "ARL_AlwaysRegion": lambda fun, lb, ub, dim, max_fes, pop, rng, X0=None, n_points=220: run_ARL(
            fun, lb, ub, dim, max_fes, pop, rng, X0=X0, ablation=None, trigger_mode="always",
            n_points=n_points, record_diag=False
        )[:2],
        "ARL_NeverRegion": lambda fun, lb, ub, dim, max_fes, pop, rng, X0=None, n_points=220: run_ARL(
            fun, lb, ub, dim, max_fes, pop, rng, X0=X0, ablation=None, trigger_mode="never",
            n_points=n_points, record_diag=False
        )[:2],
        "ARL_TwoPhase": lambda fun, lb, ub, dim, max_fes, pop, rng, X0=None, n_points=220: run_ARL(
            fun, lb, ub, dim, max_fes, pop, rng, X0=X0, ablation=None, trigger_mode="two_phase",
            n_points=n_points, record_diag=False
        )[:2],
    }

    ablations = {
        "ARL": lambda fun, lb, ub, dim, max_fes, pop, rng, X0=None, n_points=220: run_ARL(
            fun, lb, ub, dim, max_fes, pop, rng, X0=X0, ablation=None, n_points=n_points, record_diag=True
        ),
        "ARL_noPCA": lambda fun, lb, ub, dim, max_fes, pop, rng, X0=None, n_points=220: run_ARL(
            fun, lb, ub, dim, max_fes, pop, rng, X0=X0, ablation="noPCA", n_points=n_points, record_diag=False
        )[:2],
        "ARL_noDensity": lambda fun, lb, ub, dim, max_fes, pop, rng, X0=None, n_points=220: run_ARL(
            fun, lb, ub, dim, max_fes, pop, rng, X0=X0, ablation="noDensity", n_points=n_points, record_diag=False
        )[:2],
        "ARL_noLeader": lambda fun, lb, ub, dim, max_fes, pop, rng, X0=None, n_points=220: run_ARL(
            fun, lb, ub, dim, max_fes, pop, rng, X0=X0, ablation="noLeader", n_points=n_points, record_diag=False
        )[:2],
        "ARL_noRegion": lambda fun, lb, ub, dim, max_fes, pop, rng, X0=None, n_points=220: run_ARL(
            fun, lb, ub, dim, max_fes, pop, rng, X0=X0, ablation="noRegion", n_points=n_points, record_diag=False
        )[:2],
    }

    t0 = time.time()
    print(f"[INFO] n_funcs={len(suite)}, dim={cfg.dim}, pop={cfg.pop}, max_fes={cfg.max_fes}, reps={cfg.reps}")
    print(f"[INFO] outdir={outdir}")

    overall: Dict[str, Any] = {}
    overall["Baselines"] = run_group("Baselines", suite, baselines, cfg)
    overall["Ablations"] = run_group("Ablations", suite, ablations, cfg)
    if not args.skip_trigger:
        overall["TriggerStudy"] = run_group("TriggerStudy", suite, triggers, cfg)
    overall["runtime_seconds"] = float(time.time() - t0)

    # Derived summaries
    try:
        base_dir = os.path.join(outdir, "Baselines")
        funcs, algs, auc_full = _parse_group_per_function_summary(os.path.join(base_dir, "Baselines_per_function_summary.csv"))
        funcs2, algs2, auc_slices = _parse_group_auc_slices(os.path.join(base_dir, "Baselines_per_function_auc_slices.csv"))
        out_fw = ensure_dir(os.path.join(outdir, "FrameworkAdvantage"))
        _save_framework_advantage(out_fw, funcs, algs, auc_full, auc_slices, ref_alg="ARL")
    except Exception as e:
        print("[WARN] FrameworkAdvantage derivation skipped due to error:", str(e))

    try:
        ab_dir = os.path.join(outdir, "Ablations")
        out_ab = ensure_dir(os.path.join(outdir, "AblationContribution"))
        _save_ablation_contribution(out_ab, ab_dir, ref_alg="ARL")
    except Exception as e:
        print("[WARN] AblationContribution derivation skipped due to error:", str(e))

    save_kv_csv(os.path.join(outdir, "overall_run_summary.csv"), overall)
    print("[DONE] Finished. See outputs in:", outdir)


if __name__ == "__main__":
    main()
