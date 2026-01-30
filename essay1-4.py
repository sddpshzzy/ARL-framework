#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
essay1-4.py  (VB.4-PubPlotFix)
Engineering validation:
  - Kriging baseline: KED (Polynomial drift + Ridge) + Residual OK (NormalScore optional)
  - ARL model: HistGradientBoostingRegressor (with NaN-safe preprocessing)
  - RouteB Fusion:
      (1) ARL + alpha * OOF-Residual-Kriging  (alpha via GroupCV, cheap grid)
      (2) StackFusion: learn linear weights over [ARL, ARL+RK, KED] via GroupCV OOF stacking
          (no catastrophic nested loops; saves outputs even if interrupted)

Plotting upgrade (publication-grade):
  - Color-blind safe palette (Okabe–Ito)
  - Step-hist + median lines + two-panel zoom (makes subtle differences visible)
  - ECDF of absolute error (reviewer-friendly)
  - Error vs depth: scatter + binned median trend (clearer than raw clouds)
  - Scatter: metrics box (R2/RMSE/MAE/Bias) on figure

Inputs (default in current folder):
  - 定位1.xlsx
  - 实验数据.xlsx
  - 侧斜.xlsx
  - 岩性.xlsx (optional)

Outputs (under outdir):
  - metrics.csv
  - predictions_test.csv
  - diagnostics.json
  - scatter_R2_*.png
  - error_boxplot.png
  - error_hist_overlay.png
  - error_ecdf_abs.png
  - error_vs_depth.png
  - intermediate_*.csv (stage checkpoints)
"""

import os
import re
import json
import time
import argparse
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

SCRIPT_VERSION = "2026-01-23-VB.4-PubPlotFix (KED-Poly+NSK + OOF-RK + OOF-StackFusion + pub plots)"


# --------------------------- Config ---------------------------

@dataclass
class KrigingCfg:
    k_neighbors: int = 32
    anis_azimuth_deg: float = 45.0
    anis_ratio_xy: float = 3.0     # correlation longer in rotated x' -> distance divided by ratio
    z_scale: float = 2.0
    nugget_num: float = 1e-6       # numerical nugget for stability
    jitter: float = 1e-8
    max_range: Optional[float] = None  # optional radius cutoff in scaled distance
    model: str = "exponential"     # use exponential for robustness


# --------------------------- Utilities ---------------------------

def _ts():
    return time.strftime("%Y%m%d_%H%M%S")

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().replace("\u3000", " ").replace("\ufeff", "") for c in df.columns]
    return df

def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = list(df.columns)
    norm = {c: re.sub(r"\s+", "", str(c)) for c in cols}
    cand_norm = [re.sub(r"\s+", "", s) for s in candidates]
    for c in cols:
        if norm[c] in cand_norm:
            return c
    for c in cols:
        for s in candidates:
            if s in str(c):
                return c
    return None

def normalize_bhid(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df = df.copy()
    df[col] = df[col].astype(str).str.strip()
    df[col] = df[col].str.replace(r"^0+", "", regex=True)
    return df

def safe_float(x, default=np.nan):
    try:
        v = float(x)
        if np.isfinite(v):
            return v
        return default
    except Exception:
        return default

def _replace_nonfinite(arr: np.ndarray, fill_value: float) -> Tuple[np.ndarray, int]:
    arr = np.asarray(arr, dtype=float)
    bad = ~np.isfinite(arr)
    nbad = int(np.sum(bad))
    if nbad > 0:
        arr = arr.copy()
        arr[bad] = fill_value
    return arr, nbad

def summarize_series(name: str, s: np.ndarray) -> str:
    s = np.asarray(s, dtype=float)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return f"{name}: empty"
    qs = np.quantile(s, [0.01, 0.05, 0.50, 0.95, 0.99])
    return (f"{name} n={len(s)}\n"
            f"       mean={np.mean(s):.4g} std={np.std(s):.4g} min={np.min(s):.4g} max={np.max(s):.4g}\n"
            f"       q01={qs[0]:.4g} q05={qs[1]:.4g} q50={qs[2]:.4g} q95={qs[3]:.4g} q99={qs[4]:.4g}")

def save_json(path: str, obj: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def stage_save_dataframe(outdir: str, df: pd.DataFrame, name: str):
    path = os.path.join(outdir, f"intermediate_{name}.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")


# --------------------------- Normal Score Transform ---------------------------

class NormalScore:
    """
    Empirical normal score transform:
      y -> z ~ N(0,1) using rank-based mapping
      inverse uses empirical quantile mapping back to original scale.
    """
    def __init__(self, eps: float = 1e-6):
        self.eps = eps
        self.y_sorted = None

    def fit(self, y: np.ndarray):
        y = np.asarray(y, dtype=float)
        y = y[np.isfinite(y)]
        if y.size < 10:
            self.y_sorted = np.sort(y) if y.size > 0 else np.array([0.0])
        else:
            self.y_sorted = np.sort(y)
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        if self.y_sorted is None or self.y_sorted.size == 0:
            return np.zeros_like(y, dtype=float)

        ys = self.y_sorted
        idx = np.searchsorted(ys, y, side="left")
        n = ys.size
        p = (idx + 0.5) / n
        p = np.clip(p, self.eps, 1.0 - self.eps)

        from math import sqrt
        try:
            from numpy import erfinv
            z = sqrt(2.0) * erfinv(2.0 * p - 1.0)
        except Exception:
            z = np.log(p / (1.0 - p))
            z = (z - np.mean(z)) / (np.std(z) + 1e-9)

        z, _ = _replace_nonfinite(z.astype(float), 0.0)
        return z

    def inverse_transform(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        z, _ = _replace_nonfinite(z, 0.0)

        from math import sqrt
        try:
            from numpy import erf
            p = 0.5 * (1.0 + erf(z / sqrt(2.0)))
        except Exception:
            p = 1.0 / (1.0 + np.exp(-z))
        p = np.clip(p, self.eps, 1.0 - self.eps)

        ys = self.y_sorted
        n = ys.size
        q = p * (n - 1)
        lo = np.floor(q).astype(int)
        hi = np.ceil(q).astype(int)
        lo = np.clip(lo, 0, n - 1)
        hi = np.clip(hi, 0, n - 1)
        w = (q - lo).astype(float)
        y = (1.0 - w) * ys[lo] + w * ys[hi]
        y, _ = _replace_nonfinite(y, float(np.mean(ys)))
        return y


# --------------------------- Deviation -> XYZ ---------------------------

def _interp_angles(depths: np.ndarray, dev_df: pd.DataFrame, col_depth: str, col_az: str, col_dip: str):
    data = dev_df.sort_values(col_depth).copy()
    if data[col_depth].iloc[0] > 0:
        first = data.iloc[0].copy()
        first[col_depth] = 0.0
        data = pd.concat([pd.DataFrame([first]), data], ignore_index=True)

    az = np.interp(depths, data[col_depth].values, data[col_az].values) * np.pi / 180.0
    dip = np.interp(depths, data[col_depth].values, data[col_dip].values) * np.pi / 180.0
    return az, dip

def compute_xyz_from_deviation(collar_E, collar_N, collar_R, dev_df, depths,
                              col_depth: str, col_az: str, col_dip: str):
    az, dip = _interp_angles(depths, dev_df, col_depth, col_az, col_dip)
    d = np.diff(np.concatenate([[depths[0]], depths]))
    h = d * np.cos(dip)
    dz = -d * np.sin(dip)
    dx = h * np.sin(az)
    dy = h * np.cos(az)
    X = collar_E + np.cumsum(dx)
    Y = collar_N + np.cumsum(dy)
    Z = collar_R + np.cumsum(dz)
    return np.column_stack([X, Y, Z])


# --------------------------- Build samples ---------------------------

def build_samples(df_loc: pd.DataFrame,
                  df_exp: pd.DataFrame,
                  df_dev: pd.DataFrame,
                  df_lith: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, Dict]:

    df_loc = _clean_cols(df_loc)
    df_exp = _clean_cols(df_exp)
    df_dev = _clean_cols(df_dev)
    if df_lith is not None:
        df_lith = _clean_cols(df_lith)

    # BHID
    loc_bhid = _find_col(df_loc, ["工程号", "孔号", "钻孔号", "BHID"])
    exp_bhid = _find_col(df_exp, ["工程号", "孔号", "钻孔号", "BHID"])
    dev_bhid = _find_col(df_dev, ["工程号", "孔号", "钻孔号", "BHID"])
    if loc_bhid is None or exp_bhid is None or dev_bhid is None:
        raise KeyError(f"无法识别工程号列。loc={loc_bhid}, exp={exp_bhid}, dev={dev_bhid}")

    df_loc = normalize_bhid(df_loc, loc_bhid)
    df_exp = normalize_bhid(df_exp, exp_bhid)
    df_dev = normalize_bhid(df_dev, dev_bhid)

    # Collar coords
    col_E = _find_col(df_loc, ["开孔坐标E", "E", "东坐标", "X", "x"])
    col_N = _find_col(df_loc, ["开孔坐标N", "N", "北坐标", "Y", "y"])
    col_R = _find_col(df_loc, ["开孔坐标R", "R", "高程", "Z", "z"])
    if col_E is None or col_N is None or col_R is None:
        raise KeyError(f"定位表缺少坐标列：E={col_E}, N={col_N}, R={col_R}")

    # Experiment interval
    col_from = _find_col(df_exp, ["从", "起", "起点", "From", "from"])
    col_to   = _find_col(df_exp, ["至", "止", "终点", "To", "to"])
    col_tfe  = _find_col(df_exp, ["TFe", "全铁", "总铁", "品位", "tfe"])
    if col_from is None or col_to is None or col_tfe is None:
        raise KeyError(f"实验数据缺少区间/品位列：从={col_from}, 至={col_to}, TFe={col_tfe}")

    # Covariates: prioritize known variables; else numeric autodetect
    ignore_cols = {exp_bhid, col_from, col_to, col_tfe}
    preferred = ["FeO", "SFe", "磁性率", "FeO含量", "硫化铁", "磁化率"]
    cov_cols = [c for c in df_exp.columns if (c not in ignore_cols and str(c) in preferred)]
    if len(cov_cols) == 0:
        for c in df_exp.columns:
            if c in ignore_cols:
                continue
            s = pd.to_numeric(df_exp[c], errors="coerce")
            if np.isfinite(s).sum() > 50:
                cov_cols.append(c)

    # Deviation columns
    dev_depth = _find_col(df_dev, ["深度", "Depth", "depth"])
    dev_az    = _find_col(df_dev, ["方位角", "Azimuth", "az"])
    dev_dip   = _find_col(df_dev, ["倾角", "Dip", "dip"])
    if dev_depth is None or dev_az is None or dev_dip is None:
        raise KeyError(f"侧斜缺少列：深度={dev_depth}, 方位角={dev_az}, 倾角={dev_dip}")

    # Lithology columns
    lith_enabled = False
    lith_by = {}
    lith_bhid = lith_from = lith_to = lith_name = None
    if df_lith is not None:
        df_lith = _clean_cols(df_lith)
        lith_bhid = _find_col(df_lith, ["工程号", "孔号", "钻孔号", "BHID"])
        lith_from = _find_col(df_lith, ["从", "起", "From", "from"])
        lith_to   = _find_col(df_lith, ["至", "止", "To", "to"])
        lith_name = _find_col(df_lith, ["岩性", "岩性名称", "Lith", "lith"])
        if all([lith_bhid, lith_from, lith_to, lith_name]):
            df_lith = normalize_bhid(df_lith, lith_bhid)
            lith_by = {k: v.sort_values(lith_from).reset_index(drop=True)
                       for k, v in df_lith.groupby(lith_bhid)}
            lith_enabled = True

    loc_by = df_loc.set_index(loc_bhid)
    dev_by = {k: v.sort_values(dev_depth).reset_index(drop=True) for k, v in df_dev.groupby(dev_bhid)}
    exp_by = {k: v.sort_values(col_from).reset_index(drop=True) for k, v in df_exp.groupby(exp_bhid)}

    rows = []
    miss_exp = 0
    miss_dev = 0

    holes = list(loc_by.index)
    for bh in tqdm(holes, desc="Build samples", ncols=90):
        if bh not in exp_by or exp_by[bh].empty:
            miss_exp += 1
            continue

        collar = loc_by.loc[bh]
        E0 = safe_float(collar[col_E])
        N0 = safe_float(collar[col_N])
        R0 = safe_float(collar[col_R])
        if not (np.isfinite(E0) and np.isfinite(N0) and np.isfinite(R0)):
            continue

        dev_i = dev_by.get(bh)
        if dev_i is None or dev_i.empty:
            exp_i = exp_by[bh]
            maxd = float(np.nanmax(pd.to_numeric(exp_i[col_to], errors="coerce").values))
            dev_i = pd.DataFrame({
                dev_bhid: [bh, bh],
                dev_depth: [0.0, maxd],
                dev_az: [0.0, 0.0],
                dev_dip: [0.0, 0.0]
            })
            miss_dev += 1

        exp_i = exp_by[bh]
        lith_i = lith_by.get(bh) if lith_enabled else None

        for _, r in exp_i.iterrows():
            d0 = safe_float(r[col_from])
            d1 = safe_float(r[col_to])
            tfe = safe_float(r[col_tfe])
            if not (np.isfinite(d0) and np.isfinite(d1) and np.isfinite(tfe)):
                continue
            if d1 <= d0:
                continue
            dmid = 0.5 * (d0 + d1)

            xyz = compute_xyz_from_deviation(E0, N0, R0, dev_i, np.array([dmid], dtype=float),
                                             dev_depth, dev_az, dev_dip)[0]

            cov_vals = {}
            for c in cov_cols:
                cov_vals[f"Cov_{c}"] = safe_float(r.get(c, np.nan))

            lith_val = "Unknown"
            if lith_enabled and lith_i is not None and not lith_i.empty:
                a = pd.to_numeric(lith_i[lith_from], errors="coerce")
                b = pd.to_numeric(lith_i[lith_to], errors="coerce")
                mask = (a <= dmid) & (b >= dmid)
                if mask.any():
                    lith_val = str(lith_i.loc[mask, lith_name].iloc[0])

            rows.append({
                "BHID": str(bh),
                "Depth_mid": float(dmid),
                "E": float(xyz[0]),
                "N": float(xyz[1]),
                "R": float(xyz[2]),
                "TFe": float(tfe),
                "Lith": lith_val,
                **cov_vals
            })

    if len(rows) == 0:
        raise RuntimeError("未生成任何样点，请核对工程号匹配、列名、数值有效性。")

    df = pd.DataFrame(rows)
    st = {
        "n_samples": int(len(df)),
        "n_holes": int(df["BHID"].nunique()),
        "miss_exp_holes": int(miss_exp),
        "miss_dev_holes": int(miss_dev),
        "covariates": [c for c in df.columns if c.startswith("Cov_")],
        "lith_enabled": bool(lith_enabled)
    }
    return df, st


# --------------------------- Split ---------------------------

def split_by_hole(df: pd.DataFrame, seed: int = 42, test_ratio: float = 0.30):
    rng = np.random.RandomState(seed)
    holes = np.array(sorted(df["BHID"].unique()))
    rng.shuffle(holes)
    n_test = max(1, int(round(len(holes) * test_ratio)))
    test_holes = set(holes[:n_test].tolist())
    is_test = df["BHID"].isin(test_holes).values
    return df.loc[~is_test].reset_index(drop=True), df.loc[is_test].reset_index(drop=True)


# --------------------------- Variogram (exponential, robust heuristics) ---------------------------

def _anisotropic_rotate_scale(dx: np.ndarray, dy: np.ndarray, az_deg: float, ratio: float):
    az = np.deg2rad(float(az_deg))
    ca, sa = np.cos(az), np.sin(az)
    x = ca * dx + sa * dy
    y = -sa * dx + ca * dy
    ratio = max(float(ratio), 1e-6)
    x = x / ratio
    return x, y

def covariance_from_variogram(h: np.ndarray, psill: float, vrange: float, nugget: float):
    # C(h) = psill * exp(-h/range). nugget handled numerically via diag (cfg.nugget_num)
    h = np.asarray(h, dtype=float)
    vrange = max(float(vrange), 1e-6)
    psill = max(float(psill), 0.0)
    return psill * np.exp(-h / vrange)

def scaled_distance_to_all(train_xyz: np.ndarray, q_xyz: np.ndarray, cfg: KrigingCfg) -> np.ndarray:
    dx = train_xyz[:, 0] - q_xyz[0]
    dy = train_xyz[:, 1] - q_xyz[1]
    dz = (train_xyz[:, 2] - q_xyz[2]) / max(cfg.z_scale, 1e-6)
    xs, ys = _anisotropic_rotate_scale(dx, dy, cfg.anis_azimuth_deg, cfg.anis_ratio_xy)
    d = np.sqrt(xs*xs + ys*ys + dz*dz)
    d, _ = _replace_nonfinite(d, np.inf)
    return d

def fit_variogram_params_heuristic(train_xyz: np.ndarray, y: np.ndarray, cfg: KrigingCfg,
                                  seed: int = 42, n_pairs: int = 12000) -> Tuple[float, float, float, Dict]:
    """
    Heuristic fit:
      - nugget ~= median semivariance of closest 10% pairs
      - psill  ~= var(y) - nugget
      - range  ~= max(0.6*q90, q50) for robustness
    """
    rng = np.random.RandomState(seed)
    y = np.asarray(y, dtype=float)
    y, _ = _replace_nonfinite(y, float(np.nanmean(y)))
    var = float(np.var(y))

    n = train_xyz.shape[0]
    if n < 30:
        psill = max(var * 0.7, 1e-3)
        nugget = max(var * 0.1, 1e-6)
        vrange = 1.0
        return psill, vrange, nugget, {"note": "few points", "var": var}

    m = min(n_pairs, n * 10)
    ia = rng.randint(0, n, size=m)
    ib = rng.randint(0, n, size=m)

    hs = []
    gs = []
    for a, b in zip(ia, ib):
        if a == b:
            continue
        pa = train_xyz[a]; pb = train_xyz[b]
        dx = pa[0] - pb[0]
        dy = pa[1] - pb[1]
        dz = (pa[2] - pb[2]) / max(cfg.z_scale, 1e-6)
        xs, ys = _anisotropic_rotate_scale(dx, dy, cfg.anis_azimuth_deg, cfg.anis_ratio_xy)
        h = float(np.sqrt(xs*xs + ys*ys + dz*dz))
        g = 0.5 * float((y[a] - y[b])**2)
        if np.isfinite(h) and np.isfinite(g):
            hs.append(h); gs.append(g)

    hs = np.asarray(hs, dtype=float)
    gs = np.asarray(gs, dtype=float)
    if hs.size < 200:
        vrange = float(np.median(hs)) if hs.size > 0 else 1.0
        nugget = max(var * 0.1, 1e-6)
        psill = max(var - nugget, 1e-3)
        return psill, max(vrange, 1e-3), nugget, {"note": "few pairs", "var": var, "pairs": int(hs.size)}

    q10 = float(np.quantile(hs, 0.10))
    q50 = float(np.quantile(hs, 0.50))
    q90 = float(np.quantile(hs, 0.90))
    vrange = max(0.6 * q90, q50, 1e-3)

    small = hs <= q10
    if np.any(small):
        nugget = float(np.median(gs[small]))
        nugget = max(nugget, 1e-6)
    else:
        nugget = max(var * 0.1, 1e-6)

    psill = max(var - nugget, 1e-3)

    diag = {
        "var": var,
        "psill": psill,
        "range": vrange,
        "nugget": nugget,
        "pairs": int(hs.size),
        "hs_q10": q10, "hs_q50": q50, "hs_q90": q90,
    }
    return psill, vrange, nugget, diag


# --------------------------- Local Ordinary Kriging ---------------------------

def ok_predict_local(train_xyz: np.ndarray, train_y: np.ndarray,
                     query_xyz: np.ndarray,
                     cfg: KrigingCfg,
                     psill: float, vrange: float, nugget_struct: float,
                     verbose_name: str = "Kriging Predict") -> np.ndarray:
    """
    Local OK with kNN in anisotropic scaled distance.
    Numeric-stable:
      - covariance C(h) = psill*exp(-h/range)
      - adds cfg.nugget_num+jitter on diagonal
      - fallback to local mean if solve fails
    """
    train_y = np.asarray(train_y, dtype=float)
    ymean = float(np.nanmean(train_y))
    train_y, _ = _replace_nonfinite(train_y, ymean)

    out = np.zeros(query_xyz.shape[0], dtype=float)
    n = train_xyz.shape[0]

    for i in tqdm(range(query_xyz.shape[0]), desc=verbose_name, ncols=90):
        q = query_xyz[i]
        d = scaled_distance_to_all(train_xyz, q, cfg)

        if cfg.max_range is not None:
            mask = d <= cfg.max_range
            idx_all = np.where(mask)[0]
            if idx_all.size == 0:
                out[i] = ymean
                continue
            idx = idx_all[np.argsort(d[idx_all])[:cfg.k_neighbors]]
        else:
            idx = np.argsort(d)[:cfg.k_neighbors]

        pts = train_xyz[idx]
        vals = train_y[idx]
        m = pts.shape[0]
        if m < 3:
            out[i] = float(np.mean(vals))
            continue

        C = np.zeros((m, m), dtype=float)
        diag_val = (psill + nugget_struct) + cfg.nugget_num + cfg.jitter
        np.fill_diagonal(C, diag_val)

        for a in range(m):
            for b in range(a + 1, m):
                dx = pts[a, 0] - pts[b, 0]
                dy = pts[a, 1] - pts[b, 1]
                dz = (pts[a, 2] - pts[b, 2]) / max(cfg.z_scale, 1e-6)
                xs, ys = _anisotropic_rotate_scale(dx, dy, cfg.anis_azimuth_deg, cfg.anis_ratio_xy)
                h = float(np.sqrt(xs*xs + ys*ys + dz*dz))
                c = float(covariance_from_variogram(h, psill, vrange, nugget_struct))
                C[a, b] = c
                C[b, a] = c

        dx = pts[:, 0] - q[0]
        dy = pts[:, 1] - q[1]
        dz = (pts[:, 2] - q[2]) / max(cfg.z_scale, 1e-6)
        xs, ys = _anisotropic_rotate_scale(dx, dy, cfg.anis_azimuth_deg, cfg.anis_ratio_xy)
        hq = np.sqrt(xs*xs + ys*ys + dz*dz)
        cvec = covariance_from_variogram(hq, psill, vrange, nugget_struct).astype(float)

        A = np.zeros((m + 1, m + 1), dtype=float)
        A[:m, :m] = C
        A[:m, m] = 1.0
        A[m, :m] = 1.0

        bvec = np.zeros(m + 1, dtype=float)
        bvec[:m] = cvec
        bvec[m] = 1.0

        try:
            sol = np.linalg.solve(A, bvec)
            w = sol[:m]
            pred = float(np.dot(w, vals))
            if not np.isfinite(pred):
                pred = float(np.mean(vals))
            out[i] = pred
        except np.linalg.LinAlgError:
            out[i] = float(np.mean(vals))

    out, _ = _replace_nonfinite(out, ymean)
    return out


# --------------------------- Metrics ---------------------------

def calc_metrics(y_true, y_pred) -> Dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_pred, _ = _replace_nonfinite(y_pred, float(np.nanmean(y_true)))
    r2 = float(r2_score(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    denom = np.clip(np.abs(y_true), 1e-6, None)
    mape = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)
    bias = float(np.mean(y_pred - y_true))
    return {"R2": r2, "RMSE": rmse, "MAE": mae, "MAPE(%)": mape, "Bias": bias}


# --------------------------- Publication-grade Plot Style ---------------------------

def _set_pub_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",   # switch to 'Arial' if installed
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.linewidth": 1.0,
        "xtick.major.width": 0.9,
        "ytick.major.width": 0.9,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "figure.dpi": 200,
        "savefig.dpi": 450,
        "savefig.bbox": "tight"
    })

# Color-blind safe (Okabe–Ito)
_COLOR = {
    "KED": "#0072B2",        # blue
    "ARL": "#D55E00",        # vermillion
    "ARL+OOF-RK": "#009E73", # bluish green
    "StackFusion": "#CC79A7" # reddish purple
}
_LINE = {
    "KED": "-",
    "ARL": "-",
    "ARL+OOF-RK": "--",
    "StackFusion": "-."
}

def _metric_text(y_true, y_pred) -> str:
    m = calc_metrics(y_true, y_pred)
    return f"R²={m['R2']:.3f}  RMSE={m['RMSE']:.2f}  MAE={m['MAE']:.2f}  Bias={m['Bias']:.2f}"

def save_scatter(outdir: str, y_true: np.ndarray, y_pred: np.ndarray, title: str, fname: str):
    _set_pub_style()
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_pred, _ = _replace_nonfinite(y_pred, float(np.nanmean(y_true)))

    lo = float(np.quantile(np.concatenate([y_true, y_pred]), 0.01))
    hi = float(np.quantile(np.concatenate([y_true, y_pred]), 0.99))

    plt.figure(figsize=(5.6, 5.4))
    plt.scatter(y_true, y_pred, s=10, alpha=0.55, edgecolors="none")
    plt.plot([lo, hi], [lo, hi], linewidth=1.2, linestyle="--")
    plt.xlim(lo, hi); plt.ylim(lo, hi)
    plt.xlabel("Observed TFe")
    plt.ylabel("Predicted TFe")
    plt.title(title)

    txt = _metric_text(y_true, y_pred)
    plt.gca().text(0.02, 0.98, txt, transform=plt.gca().transAxes,
                   va="top", ha="left",
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, linewidth=0.8))
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, fname), dpi=450)
    plt.close()

def save_error_boxplot(outdir: str, err_dict: Dict[str, np.ndarray], fname="error_boxplot.png"):
    _set_pub_style()
    labels = list(err_dict.keys())
    data = [np.asarray(err_dict[k], dtype=float) for k in labels]

    plt.figure(figsize=(7.4, 3.9))
    bp = plt.boxplot(data, labels=labels, showfliers=False, widths=0.55, patch_artist=True)

    for patch, lab in zip(bp["boxes"], labels):
        patch.set_facecolor(_COLOR.get(lab, "#4D4D4D"))
        patch.set_alpha(0.55)
        patch.set_linewidth(1.0)
    for med in bp["medians"]:
        med.set_linewidth(1.4)

    plt.axhline(0.0, linestyle="--", linewidth=1.0, color="#333333")
    plt.ylabel("Error (Pred − Obs)")
    plt.title("Error distribution (boxplot)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, fname), dpi=450)
    plt.close()

def save_error_hist_overlay(outdir: str,
                            err_dict: Dict[str, np.ndarray],
                            hist_range=(-15, 15),
                            bins=60,
                            fname="error_hist_overlay.png"):
    """
    Step hist (not filled) + median lines + two-panel zoom (subtle differences visible).
    """
    _set_pub_style()
    labels = list(err_dict.keys())
    errs = {k: np.asarray(err_dict[k], dtype=float) for k in labels}

    fig = plt.figure(figsize=(7.6, 4.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.0, 1.0], hspace=0.08)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    x_min, x_max = hist_range
    edges = np.linspace(x_min, x_max, bins + 1)

    for k in labels:
        e = errs[k]
        e = e[np.isfinite(e)]
        color = _COLOR.get(k, "#4D4D4D")
        ls = _LINE.get(k, "-")

        ax1.hist(e, bins=edges, density=True, histtype="step", linewidth=1.6,
                 color=color, linestyle=ls, label=k)
        ax2.hist(e, bins=edges, density=True, histtype="step", linewidth=1.6,
                 color=color, linestyle=ls)

        med = float(np.median(e)) if e.size else 0.0
        ax1.axvline(med, color=color, linestyle=":", linewidth=1.2, alpha=0.9)

    ax1.axvline(0.0, color="#333333", linestyle="--", linewidth=1.1)
    ax2.axvline(0.0, color="#333333", linestyle="--", linewidth=1.1)

    ax1.set_ylabel("Density")
    ax1.set_title("Error distribution (step histogram) + medians")

    ax2.set_xlabel("Error (Pred − Obs)")
    ax2.set_ylabel("Density")
    ax2.set_ylim(0, ax1.get_ylim()[1] * 0.55)

    ax1.legend(frameon=True, framealpha=0.9, loc="upper right")
    plt.setp(ax1.get_xticklabels(), visible=False)

    plt.savefig(os.path.join(outdir, fname), dpi=450)
    plt.close()

def save_error_ecdf(outdir: str, err_dict: Dict[str, np.ndarray], fname="error_ecdf_abs.png"):
    """
    ECDF of absolute error: P(|e| <= t). Usually the clearest for reviewers.
    """
    _set_pub_style()
    plt.figure(figsize=(6.8, 4.4))

    for k, e in err_dict.items():
        e = np.asarray(e, dtype=float)
        e = e[np.isfinite(e)]
        ae = np.abs(e)
        ae.sort()
        y = np.linspace(0, 1, len(ae), endpoint=True)
        plt.plot(ae, y, linewidth=1.8,
                 color=_COLOR.get(k, "#4D4D4D"),
                 linestyle=_LINE.get(k, "-"),
                 label=k)

    plt.xlabel("|Error| (absolute)")
    plt.ylabel("ECDF  P(|e| ≤ t)")
    plt.title("Cumulative distribution of absolute errors")
    plt.legend(frameon=True, framealpha=0.9, loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, fname), dpi=450)
    plt.close()

def save_error_vs_depth(outdir: str, depth: np.ndarray, err_dict: Dict[str, np.ndarray], fname="error_vs_depth.png"):
    """
    Scatter + binned median trend (clear, robust, no extra deps).
    """
    _set_pub_style()
    depth = np.asarray(depth, dtype=float)

    plt.figure(figsize=(7.6, 4.4))
    ax = plt.gca()

    nbins = 25
    bins = np.linspace(np.nanmin(depth), np.nanmax(depth), nbins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])

    for k, e in err_dict.items():
        e = np.asarray(e, dtype=float)
        mask = np.isfinite(depth) & np.isfinite(e)
        d = depth[mask]; ee = e[mask]

        # downsample for readability
        if d.size > 4000:
            idx = np.random.RandomState(0).choice(np.arange(d.size), size=4000, replace=False)
            d_sc, e_sc = d[idx], ee[idx]
        else:
            d_sc, e_sc = d, ee

        ax.scatter(d_sc, e_sc, s=6, alpha=0.18,
                   color=_COLOR.get(k, "#4D4D4D"), edgecolors="none")

        # binned median trend
        med = np.full(nbins, np.nan)
        for i in range(nbins):
            m = (d >= bins[i]) & (d < bins[i + 1])
            if np.any(m):
                med[i] = np.median(ee[m])
        ax.plot(centers, med, linewidth=2.0,
                color=_COLOR.get(k, "#4D4D4D"),
                linestyle=_LINE.get(k, "-"),
                label=k)

    ax.axhline(0.0, linestyle="--", linewidth=1.0, color="#333333")
    ax.set_xlabel("Depth_mid")
    ax.set_ylabel("Error (Pred − Obs)")
    ax.set_title("Error vs depth (scatter + binned median trend)")
    ax.legend(frameon=True, framealpha=0.9, loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, fname), dpi=450)
    plt.close()


# --------------------------- ARL (your framework proxy) ---------------------------

def fit_arl(train_df: pd.DataFrame, seed: int = 42) -> Tuple[Pipeline, List[str], List[str]]:
    cov_cols = [c for c in train_df.columns if c.startswith("Cov_")]
    num_cols = ["E", "N", "R", "Depth_mid"] + cov_cols
    cat_cols = ["Lith"] if "Lith" in train_df.columns else []

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore"))
    ])
    pre = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols)
    ], remainder="drop")

    model = HistGradientBoostingRegressor(
        random_state=seed,
        learning_rate=0.05,
        max_depth=7,
        max_iter=1200,
        min_samples_leaf=25,
        l2_regularization=0.0,
        max_bins=255
    )

    pipe = Pipeline([
        ("pre", pre),
        ("model", model)
    ])
    X = train_df[num_cols + cat_cols]
    y = train_df["TFe"].values.astype(float)
    pipe.fit(X, y)
    return pipe, num_cols, cat_cols

def predict_arl(pipe: Pipeline, df: pd.DataFrame, num_cols: List[str], cat_cols: List[str]) -> np.ndarray:
    X = df[num_cols + cat_cols]
    pred = pipe.predict(X)
    pred, _ = _replace_nonfinite(pred, float(np.nanmean(df["TFe"].values)))
    return pred


# --------------------------- KED (Polynomial drift + Ridge) + Residual OK ---------------------------

def regression_kriging_ked_poly(train_df: pd.DataFrame,
                               test_df: pd.DataFrame,
                               cfg: KrigingCfg,
                               seed: int = 42,
                               degree: int = 2,
                               use_ns_transform: bool = True,
                               alpha_grid=(0.1, 1.0, 10.0, 50.0, 200.0)) -> Tuple[np.ndarray, Dict]:
    """
    Traditional KED baseline:
      drift: PolynomialFeatures(degree) + Ridge on [covariates + depth + xyz]
      residual: OK (optional NormalScore transform)
    """
    cov_cols = [c for c in train_df.columns if c.startswith("Cov_")]

    # keep only covariates with finite values
    good_cov = []
    for c in cov_cols:
        s = pd.to_numeric(train_df[c], errors="coerce")
        if np.isfinite(s).any():
            good_cov.append(c)
    dropped = list(sorted(set(cov_cols) - set(good_cov)))
    cov_cols = good_cov

    base_num = cov_cols + ["Depth_mid", "E", "N", "R"]
    cat_cols = ["Lith"] if "Lith" in train_df.columns else []

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("poly", PolynomialFeatures(degree=degree, include_bias=False))
    ])
    transformers = [("num", num_pipe, base_num)]
    if cat_cols:
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore"))
        ])
        transformers.append(("cat", cat_pipe, cat_cols))

    pre = ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.3)

    X = train_df[base_num + cat_cols].copy()
    y = train_df["TFe"].values.astype(float)
    groups = train_df["BHID"].astype(str).values
    gkf = GroupKFold(n_splits=5)

    best_alpha, best_rmse = None, np.inf
    for a in alpha_grid:
        pipe = Pipeline([
            ("pre", pre),
            ("ridge", Ridge(alpha=float(a), random_state=seed))
        ])
        rmses = []
        for tr, va in gkf.split(X, y, groups=groups):
            pipe.fit(X.iloc[tr], y[tr])
            pred = pipe.predict(X.iloc[va])
            pred, _ = _replace_nonfinite(pred, float(np.nanmean(y[tr])))
            rmses.append(float(np.sqrt(mean_squared_error(y[va], pred))))
        rmse = float(np.mean(rmses))
        if rmse < best_rmse:
            best_rmse = rmse
            best_alpha = float(a)

    drift = Pipeline([
        ("pre", pre),
        ("ridge", Ridge(alpha=best_alpha, random_state=seed))
    ])
    drift.fit(X, y)

    drift_train = drift.predict(X)
    drift_train, _ = _replace_nonfinite(drift_train, float(np.nanmean(y)))
    resid = y - drift_train
    resid = resid - float(np.mean(resid))

    ns = None
    resid_used = resid.copy()
    if use_ns_transform:
        ns = NormalScore().fit(resid_used)
        resid_used = ns.transform(resid_used)

    train_xyz = train_df[["E", "N", "R"]].values.astype(float)
    psill, vrange, nugget_struct, vdiag = fit_variogram_params_heuristic(train_xyz, resid_used, cfg, seed=seed + 11)

    test_xyz = test_df[["E", "N", "R"]].values.astype(float)
    r_hat_z = ok_predict_local(train_xyz, resid_used, test_xyz, cfg, psill, vrange, nugget_struct,
                               verbose_name="Kriging Predict (KED residual)")
    r_hat = ns.inverse_transform(r_hat_z) if (use_ns_transform and ns is not None) else r_hat_z

    Xt = test_df[base_num + cat_cols].copy()
    drift_test = drift.predict(Xt)
    drift_test, _ = _replace_nonfinite(drift_test, float(np.nanmean(y)))
    yhat = drift_test + r_hat
    yhat, _ = _replace_nonfinite(yhat, float(np.nanmean(y)))

    diag = {
        "drift": f"Poly(deg={degree})+Ridge(+Imputer+Scaler)",
        "best_alpha": best_alpha,
        "drift_groupcv_rmse": best_rmse,
        "use_ns_transform": bool(use_ns_transform),
        "dropped_all_nan_cov_cols": dropped,
        "variogram": {"model": cfg.model, **vdiag},
    }
    return yhat, diag


# --------------------------- OOF Residual Kriging for ARL + alpha selection ---------------------------

def arl_plus_oof_rk(train_df: pd.DataFrame,
                    test_df: pd.DataFrame,
                    cfg: KrigingCfg,
                    seed: int = 42,
                    alpha_grid=(0.0, 0.2, 0.35, 0.5, 0.75, 1.0),
                    use_ns_resid: bool = True) -> Tuple[np.ndarray, Dict, np.ndarray]:
    """
    Compute leakage-safe OOF residuals of ARL (GroupKFold by BHID),
    fit residual OK on full train OOF residuals, then choose alpha via GroupCV cheaply:
      y = y_arl + alpha * r_ok
    """
    y_train = train_df["TFe"].values.astype(float)
    groups = train_df["BHID"].astype(str).values
    gkf = GroupKFold(n_splits=5)

    arl_full, ncols_full, ccols_full = fit_arl(train_df, seed=seed)
    y_test_arl = predict_arl(arl_full, test_df, ncols_full, ccols_full)

    X_train_full = train_df[ncols_full + ccols_full]
    oof = np.zeros(len(train_df), dtype=float)

    for tr, va in gkf.split(X_train_full, y_train, groups=groups):
        tr_df = train_df.iloc[tr].reset_index(drop=True)
        va_df = train_df.iloc[va].reset_index(drop=True)
        pipe, ncols, ccols = fit_arl(tr_df, seed=seed + 17)
        oof[va] = predict_arl(pipe, va_df, ncols, ccols)

    oof, _ = _replace_nonfinite(oof, float(np.nanmean(y_train)))
    resid = y_train - oof
    resid = resid - float(np.mean(resid))

    ns = None
    resid_used = resid.copy()
    if use_ns_resid:
        ns = NormalScore().fit(resid_used)
        resid_used = ns.transform(resid_used)

    train_xyz = train_df[["E", "N", "R"]].values.astype(float)
    psill, vrange, nugget_struct, vdiag = fit_variogram_params_heuristic(train_xyz, resid_used, cfg, seed=seed + 23)

    test_xyz = test_df[["E", "N", "R"]].values.astype(float)
    r_test_z = ok_predict_local(train_xyz, resid_used, test_xyz, cfg, psill, vrange, nugget_struct,
                                verbose_name="Kriging Predict (OOF-RK residual)")
    r_test = ns.inverse_transform(r_test_z) if (use_ns_resid and ns is not None) else r_test_z

    def score_alpha(alpha: float) -> float:
        rmses = []
        for tr, va in gkf.split(train_df, y_train, groups=groups):
            tr_df = train_df.iloc[tr].reset_index(drop=True)
            va_df = train_df.iloc[va].reset_index(drop=True)

            pipe, ncols, ccols = fit_arl(tr_df, seed=seed + 31)
            y_va_arl = predict_arl(pipe, va_df, ncols, ccols)

            y_tr_pred = predict_arl(pipe, tr_df, ncols, ccols)
            r_tr = tr_df["TFe"].values.astype(float) - y_tr_pred
            r_tr = r_tr - float(np.mean(r_tr))

            ns2 = None
            r_tr_used = r_tr.copy()
            if use_ns_resid:
                ns2 = NormalScore().fit(r_tr_used)
                r_tr_used = ns2.transform(r_tr_used)

            tr_xyz = tr_df[["E", "N", "R"]].values.astype(float)
            va_xyz = va_df[["E", "N", "R"]].values.astype(float)
            ps, rg, ng, _ = fit_variogram_params_heuristic(tr_xyz, r_tr_used, cfg, seed=seed + 41)
            r_va_z = ok_predict_local(tr_xyz, r_tr_used, va_xyz, cfg, ps, rg, ng,
                                      verbose_name="Kriging Predict (alphaCV)")
            r_va = ns2.inverse_transform(r_va_z) if (use_ns_resid and ns2 is not None) else r_va_z

            y_hat = y_va_arl + alpha * r_va
            rmses.append(float(np.sqrt(mean_squared_error(va_df["TFe"].values.astype(float), y_hat))))
        return float(np.mean(rmses))

    best_alpha, best_rmse = None, np.inf
    for a in alpha_grid:
        rmse = score_alpha(float(a))
        if rmse < best_rmse:
            best_rmse = rmse
            best_alpha = float(a)

    y_fused = y_test_arl + best_alpha * r_test
    y_fused, _ = _replace_nonfinite(y_fused, float(np.nanmean(y_train)))

    diag = {
        "alpha_grid": list(alpha_grid),
        "best_alpha": best_alpha,
        "groupcv_rmse": best_rmse,
        "use_ns_resid": bool(use_ns_resid),
        "resid_variogram": {"model": cfg.model, **vdiag},
        "formula": "Fusion = ARL + alpha * OK(OOF_residual)"
    }
    return y_fused, diag, y_test_arl


# --------------------------- OOF Stacking (learn weights) ---------------------------

def oof_stack_fusion(train_df: pd.DataFrame,
                     test_df: pd.DataFrame,
                     y_arl_test: np.ndarray,
                     y_arlrk_test: np.ndarray,
                     y_ked_test: np.ndarray,
                     seed: int = 42) -> Tuple[np.ndarray, Dict]:
    """
    Learn linear weights on predictors:
      [ARL, ARL+RK, KED] using OOF (GroupKFold by BHID) on training.
    """
    y_train = train_df["TFe"].values.astype(float)
    groups = train_df["BHID"].astype(str).values
    gkf = GroupKFold(n_splits=5)

    oof_arl = np.zeros(len(train_df), dtype=float)
    oof_arlrk = np.zeros(len(train_df), dtype=float)
    oof_ked = np.zeros(len(train_df), dtype=float)

    if not hasattr(oof_stack_fusion, "_cfg"):
        raise RuntimeError("Internal: cfg not set for oof_stack_fusion. Call oof_stack_fusion._cfg = cfg in main.")
    cfg_fold = getattr(oof_stack_fusion, "_cfg")

    alpha_grid = (0.0, 0.35, 0.75, 1.0)

    for tr, va in gkf.split(train_df, y_train, groups=groups):
        tr_df = train_df.iloc[tr].reset_index(drop=True)
        va_df = train_df.iloc[va].reset_index(drop=True)

        pipe, ncols, ccols = fit_arl(tr_df, seed=seed + 101)
        oof_arl[va] = predict_arl(pipe, va_df, ncols, ccols)

        y_ked_va, _ = regression_kriging_ked_poly(
            tr_df, va_df, cfg=cfg_fold, seed=seed + 201,
            degree=2, use_ns_transform=True, alpha_grid=(0.1, 1.0, 10.0, 50.0)
        )
        oof_ked[va] = y_ked_va

        y_tr_pred = predict_arl(pipe, tr_df, ncols, ccols)
        r_tr = tr_df["TFe"].values.astype(float) - y_tr_pred
        r_tr = r_tr - float(np.mean(r_tr))

        ns = NormalScore().fit(r_tr)
        r_tr_z = ns.transform(r_tr)

        tr_xyz = tr_df[["E", "N", "R"]].values.astype(float)
        va_xyz = va_df[["E", "N", "R"]].values.astype(float)
        ps, rg, ng, _ = fit_variogram_params_heuristic(tr_xyz, r_tr_z, cfg_fold, seed=seed + 303)
        r_va_z = ok_predict_local(tr_xyz, r_tr_z, va_xyz, cfg_fold, ps, rg, ng,
                                  verbose_name="Kriging Predict (stackOOF)")
        r_va = ns.inverse_transform(r_va_z)

        y_va_arl = oof_arl[va]
        best_a = 0.0
        best_rmse = np.inf
        for a in alpha_grid:
            y_hat = y_va_arl + float(a) * r_va
            rmse = float(np.sqrt(mean_squared_error(va_df["TFe"].values.astype(float), y_hat)))
            if rmse < best_rmse:
                best_rmse = rmse
                best_a = float(a)
        oof_arlrk[va] = y_va_arl + best_a * r_va

    Xoof = np.column_stack([
        np.ones(len(train_df)),
        oof_arl,
        oof_arlrk,
        oof_ked
    ])
    ridge = Ridge(alpha=1.0, random_state=seed)
    ridge.fit(Xoof, y_train)

    Xtest = np.column_stack([
        np.ones(len(test_df)),
        y_arl_test,
        y_arlrk_test,
        y_ked_test
    ])
    y_stack = ridge.predict(Xtest)
    y_stack, _ = _replace_nonfinite(y_stack, float(np.nanmean(y_train)))

    w = ridge.coef_.astype(float)
    b = float(ridge.intercept_)

    diag = {
        "stack_model": "Ridge(alpha=1.0) on [ARL, ARL+RK, KED]",
        "coef": {"w_ARL": float(w[1]), "w_ARLRK": float(w[2]), "w_KED": float(w[3])},
        "intercept": b,
        "note": "Weights learned on OOF predictions (GroupKFold by BHID)."
    }
    return y_stack, diag


# --------------------------- Main ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loc", default="定位1.xlsx", help="定位表 xlsx")
    ap.add_argument("--exp", default="实验数据.xlsx", help="实验数据 xlsx")
    ap.add_argument("--dev", default="侧斜.xlsx", help="侧斜 xlsx")
    ap.add_argument("--lith", default="岩性.xlsx", help="岩性 xlsx (optional)")
    ap.add_argument("--outdir", default="", help="output directory (default auto)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test_ratio", type=float, default=0.30)

    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--anis_az", type=float, default=45.0)
    ap.add_argument("--anis_ratio", type=float, default=3.0)
    ap.add_argument("--z_scale", type=float, default=2.0)
    ap.add_argument("--max_range_q", type=float, default=0.0,
                    help="if >0, set max_range as quantile(q) of train pair distances (scaled). recommend 0.90~0.95")

    ap.add_argument("--ked_degree", type=int, default=2, help="Polynomial drift degree for KED (2 recommended)")
    ap.add_argument("--no_ns", action="store_true", help="Disable NormalScore transform for residual kriging")

    args = ap.parse_args()

    outdir = args.outdir.strip() or f"outputs_engineering_{_ts()}"
    ensure_dir(outdir)

    print(f"[INFO] SCRIPT_VERSION = {SCRIPT_VERSION}")
    print(f"[INFO] outdir = {outdir}")

    for fp in [args.loc, args.exp, args.dev]:
        if not os.path.exists(fp):
            raise FileNotFoundError(f"找不到 {fp}")

    df_loc = pd.read_excel(args.loc)
    df_exp = pd.read_excel(args.exp)
    df_dev = pd.read_excel(args.dev)

    df_lith = None
    if args.lith and os.path.exists(args.lith):
        df_lith = pd.read_excel(args.lith)
        print("[INFO] Lithology loaded and enabled.")
    else:
        print("[INFO] Lithology not found or disabled.")

    df, st = build_samples(df_loc, df_exp, df_dev, df_lith=df_lith)
    print(f"[INFO] 样点数：{st['n_samples']}；孔数：{st['n_holes']}；缺实验孔：{st['miss_exp_holes']}；缺侧斜孔（直孔补齐）：{st['miss_dev_holes']}")
    if st["covariates"]:
        print(f"[INFO] Covariates used: {[c.replace('Cov_', '') for c in st['covariates']]}")
    print("[INFO] " + summarize_series("Grade(raw)", df["TFe"].values).replace("\n", "\n[INFO] "))

    stage_save_dataframe(outdir, df.head(2000), "samples_head2000")

    df_train, df_test = split_by_hole(df, seed=args.seed, test_ratio=args.test_ratio)
    print(f"[INFO] Split: by_hole | train_holes={df_train['BHID'].nunique()} test_holes={df_test['BHID'].nunique()}")
    print("[INFO] " + summarize_series("Train Grade", df_train["TFe"].values).replace("\n", "\n[INFO] "))
    print("[INFO] " + summarize_series("Test  Grade", df_test["TFe"].values).replace("\n", "\n[INFO] "))

    cfg = KrigingCfg(
        k_neighbors=int(args.k),
        anis_azimuth_deg=float(args.anis_az),
        anis_ratio_xy=float(args.anis_ratio),
        z_scale=float(args.z_scale),
        max_range=None,
        model="exponential"
    )

    if args.max_range_q and args.max_range_q > 0:
        q = float(args.max_range_q)
        q = min(max(q, 0.5), 0.99)
        rng = np.random.RandomState(args.seed + 7)
        xyz = df_train[["E", "N", "R"]].values.astype(float)
        n = xyz.shape[0]
        m = min(8000, n * 2)
        ia = rng.randint(0, n, size=m)
        ib = rng.randint(0, n, size=m)
        ds = []
        for a, b in zip(ia, ib):
            if a == b:
                continue
            da = xyz[a]; db = xyz[b]
            dx = da[0] - db[0]
            dy = da[1] - db[1]
            dz = (da[2] - db[2]) / max(cfg.z_scale, 1e-6)
            xs, ys = _anisotropic_rotate_scale(dx, dy, cfg.anis_azimuth_deg, cfg.anis_ratio_xy)
            ds.append(float(np.sqrt(xs*xs + ys*ys + dz*dz)))
        ds = np.asarray(ds, dtype=float)
        cfg.max_range = float(np.quantile(ds, q))
        print(f"[INFO] Kriging max_range enabled: q={q} -> max_range={cfg.max_range:.3f} (scaled)")

    diagnostics = {
        "SCRIPT_VERSION": SCRIPT_VERSION,
        "KrigingCfg": cfg.__dict__,
        "st": st,
        "paths": {"loc": args.loc, "exp": args.exp, "dev": args.dev, "lith": args.lith}
    }

    y_true = df_test["TFe"].values.astype(float)

    y_ked = y_arl = y_arlrk = y_stack = None
    ked_diag = arlrk_diag = stack_diag = None

    try:
        y_ked, ked_diag = regression_kriging_ked_poly(
            df_train, df_test, cfg=cfg, seed=args.seed,
            degree=int(args.ked_degree),
            use_ns_transform=(not args.no_ns),
            alpha_grid=(0.1, 1.0, 10.0, 50.0, 200.0)
        )
        diagnostics["KED"] = ked_diag
        stage_save_dataframe(outdir, pd.DataFrame({"TFe": y_true, "Pred_KED": y_ked}), "ked_done")

        arl_pipe, ncols, ccols = fit_arl(df_train, seed=args.seed)
        y_arl = predict_arl(arl_pipe, df_test, ncols, ccols)
        stage_save_dataframe(outdir, pd.DataFrame({"TFe": y_true, "Pred_ARL": y_arl}), "arl_done")

        y_arlrk, arlrk_diag, _ = arl_plus_oof_rk(
            df_train, df_test, cfg=cfg, seed=args.seed,
            alpha_grid=(0.0, 0.2, 0.35, 0.5, 0.75, 1.0),
            use_ns_resid=True
        )
        diagnostics["ARL_OOF_RK"] = arlrk_diag
        stage_save_dataframe(outdir, pd.DataFrame({"TFe": y_true, "Pred_ARL_RK": y_arlrk}), "arlrk_done")

        oof_stack_fusion._cfg = cfg
        y_stack, stack_diag = oof_stack_fusion(
            df_train, df_test,
            y_arl_test=y_arl,
            y_arlrk_test=y_arlrk,
            y_ked_test=y_ked,
            seed=args.seed
        )
        diagnostics["STACK_FUSION"] = stack_diag
        stage_save_dataframe(outdir, pd.DataFrame({"TFe": y_true, "Pred_STACK": y_stack}), "stack_done")

    except KeyboardInterrupt:
        print("\n[WARN] KeyboardInterrupt received. Saving partial outputs...")
    except Exception as e:
        print(f"\n[ERROR] Exception occurred: {repr(e)}\nSaving partial outputs...")

    preds = df_test[["BHID", "Depth_mid", "E", "N", "R", "TFe"]].copy()
    if "Lith" in df_test.columns:
        preds["Lith"] = df_test["Lith"].astype(str)

    methods = []
    rows = []

    if y_ked is not None:
        preds["Pred_KED"] = y_ked
        mk = calc_metrics(y_true, y_ked)
        rows.append({"Method": "Kriging(KED-Poly+NSK)", **mk})
        methods.append("KED")

    if y_arl is not None:
        preds["Pred_ARL"] = y_arl
        ma = calc_metrics(y_true, y_arl)
        rows.append({"Method": "ARL_full", **ma})
        methods.append("ARL")

    if y_arlrk is not None:
        preds["Pred_ARL_OOF_RK"] = y_arlrk
        mf = calc_metrics(y_true, y_arlrk)
        rows.append({"Method": "RouteB_Fusion(ARL+OOF-RK)", **mf})
        methods.append("ARL+RK")

    if y_stack is not None:
        preds["Pred_StackFusion"] = y_stack
        ms = calc_metrics(y_true, y_stack)
        rows.append({"Method": "RouteB_StackFusion(ARL+OOF-RK+KED)", **ms})
        methods.append("STACK")

    preds.to_csv(os.path.join(outdir, "predictions_test.csv"), index=False, encoding="utf-8-sig")
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(outdir, "metrics.csv"), index=False, encoding="utf-8-sig")
    print(res)

    # plots
    try:
        if y_ked is not None:
            save_scatter(outdir, y_true, y_ked, "Kriging(KED-Poly+NSK)", "scatter_R2_ked.png")
        if y_arl is not None:
            save_scatter(outdir, y_true, y_arl, "ARL_full", "scatter_R2_arl.png")
        if y_arlrk is not None:
            save_scatter(outdir, y_true, y_arlrk, "RouteB_Fusion(ARL+OOF-RK)", "scatter_R2_arl_oof_rk.png")
        if y_stack is not None:
            save_scatter(outdir, y_true, y_stack, "RouteB_StackFusion", "scatter_R2_stackfusion.png")

        err_dict = {}
        if y_ked is not None:
            err_dict["KED"] = (y_ked - y_true)
        if y_arl is not None:
            err_dict["ARL"] = (y_arl - y_true)
        if y_arlrk is not None:
            err_dict["ARL+OOF-RK"] = (y_arlrk - y_true)
        if y_stack is not None:
            err_dict["StackFusion"] = (y_stack - y_true)

        if len(err_dict) >= 2:
            save_error_boxplot(outdir, err_dict, fname="error_boxplot.png")
            save_error_hist_overlay(outdir, err_dict, hist_range=(-15, 15), bins=60, fname="error_hist_overlay.png")
            save_error_ecdf(outdir, err_dict, fname="error_ecdf_abs.png")
            save_error_vs_depth(outdir, df_test["Depth_mid"].values.astype(float), err_dict, fname="error_vs_depth.png")
    except Exception as e:
        print(f"[WARN] Plotting failed: {repr(e)}")

    diagnostics["methods_available"] = methods
    if ked_diag is not None:
        diagnostics["KED"] = ked_diag
    if arlrk_diag is not None:
        diagnostics["ARL_OOF_RK"] = arlrk_diag
    if stack_diag is not None:
        diagnostics["STACK_FUSION"] = stack_diag

    save_json(os.path.join(outdir, "diagnostics.json"), diagnostics)

    print(f"[INFO] Done. Outputs saved to: {outdir}")


if __name__ == "__main__":
    main()
