#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
essay1-4.py
Engineering validation: Kriging (KED), ARL, ARL + Residual Kriging (OOF-RK)

Inputs (default in current folder):
  - 定位1.xlsx
  - 实验数据.xlsx
  - 侧斜.xlsx
  - 岩性.xlsx (optional)

Outputs:
  - metrics.csv
  - scatter_R2_*.png
  - error_boxplot.png
  - error_hist_overlay.png
  - error_vs_depth.png
  - predictions_test.csv
"""

import os
import re
import math
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
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


SCRIPT_VERSION = "2026-01-22-VB.2 (KED-NaNsafe + stable OK + OOF-RK + plots)"


# --------------------------- Config ---------------------------

@dataclass
class KrigingCfg:
    k_neighbors: int = 32               # local neighbors
    anis_azimuth_deg: float = 0.0       # rotation azimuth in XY plane
    anis_ratio_xy: float = 1.0          # stretch along rotated x-axis
    z_scale: float = 1.0                # scale Z distance
    nugget: float = 1e-6                # numerical nugget (for stability)
    jitter: float = 1e-8                # matrix jitter
    max_range: Optional[float] = None   # optional radius cutoff (scaled distance)
    # simple param bounds for "reasonable" semivariogram
    # NOTE: we use exponential model only for robustness
    model: str = "exponential"


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
    # fallback: contains match
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

def summarize_series(name: str, s: np.ndarray) -> str:
    s = np.asarray(s, dtype=float)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return f"{name}: empty"
    qs = np.quantile(s, [0.01, 0.05, 0.50, 0.95, 0.99])
    return (f"{name} n={len(s)}\n"
            f"       mean={np.mean(s):.4g} std={np.std(s):.4g} min={np.min(s):.4g} max={np.max(s):.4g}\n"
            f"       q01={qs[0]:.4g} q05={qs[1]:.4g} q50={qs[2]:.4g} q95={qs[3]:.4g} q99={qs[4]:.4g}")

def _replace_nonfinite(arr: np.ndarray, fill_value: float) -> Tuple[np.ndarray, int]:
    arr = np.asarray(arr, dtype=float)
    bad = ~np.isfinite(arr)
    nbad = int(np.sum(bad))
    if nbad > 0:
        arr = arr.copy()
        arr[bad] = fill_value
    return arr, nbad


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
    if df_lith is not None:
        lith_bhid = _find_col(df_lith, ["工程号", "孔号", "钻孔号", "BHID"])
        if lith_bhid is not None:
            df_lith = normalize_bhid(df_lith, lith_bhid)
        else:
            df_lith = None  # can't use

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

    # Covariates: numeric columns excluding identifiers and interval and TFe
    ignore_cols = {exp_bhid, col_from, col_to, col_tfe}
    cov_cols = []
    for c in df_exp.columns:
        if c in ignore_cols:
            continue
        # keep typical covariates used by you
        if str(c) in ["FeO", "SFe", "磁性率", "FeO含量", "硫化铁", "磁化率"]:
            cov_cols.append(c)
    # If not found, try numeric auto-detect
    if len(cov_cols) == 0:
        for c in df_exp.columns:
            if c in ignore_cols:
                continue
            s = pd.to_numeric(df_exp[c], errors="coerce")
            if np.isfinite(s).sum() > 0:
                cov_cols.append(c)

    # Deviation columns
    dev_depth = _find_col(df_dev, ["深度", "Depth", "depth"])
    dev_az    = _find_col(df_dev, ["方位角", "Azimuth", "az"])
    dev_dip   = _find_col(df_dev, ["倾角", "Dip", "dip"])
    if dev_depth is None or dev_az is None or dev_dip is None:
        raise KeyError(f"侧斜缺少列：深度={dev_depth}, 方位角={dev_az}, 倾角={dev_dip}")

    # Lithology columns
    lith_enabled = df_lith is not None
    if lith_enabled:
        lith_bhid = _find_col(df_lith, ["工程号", "孔号", "钻孔号", "BHID"])
        lith_from = _find_col(df_lith, ["从", "起", "From", "from"])
        lith_to   = _find_col(df_lith, ["至", "止", "To", "to"])
        lith_name = _find_col(df_lith, ["岩性", "岩性名称", "Lith", "lith"])
        if lith_bhid is None or lith_from is None or lith_to is None or lith_name is None:
            lith_enabled = False

    loc_by = df_loc.set_index(loc_bhid)
    dev_by = {k: v.sort_values(dev_depth).reset_index(drop=True) for k, v in df_dev.groupby(dev_bhid)}
    exp_by = {k: v.sort_values(col_from).reset_index(drop=True) for k, v in df_exp.groupby(exp_bhid)}
    lith_by = {}
    if lith_enabled:
        lith_by = {k: v.sort_values(lith_from).reset_index(drop=True) for k, v in df_lith.groupby(lith_bhid)}

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
            # straight hole fallback
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

            # covariates (from same experiment row)
            cov_vals = {}
            for c in cov_cols:
                cov_vals[f"Cov_{c}"] = safe_float(r.get(c, np.nan))

            # lithology assign by interval
            lith_val = "Unknown"
            if lith_enabled and lith_i is not None and not lith_i.empty:
                # find interval containing dmid
                mask = (pd.to_numeric(lith_i[lith_from], errors="coerce") <= dmid) & \
                       (pd.to_numeric(lith_i[lith_to], errors="coerce") >= dmid)
                if mask.any():
                    lith_val = str(lith_i.loc[mask, lith_name].iloc[0])
                else:
                    lith_val = "Unknown"

            rows.append({
                "BHID": bh,
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
        "covariates": [c for c in df.columns if c.startswith("Cov_")]
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


# --------------------------- Variogram (robust, simple) ---------------------------

def variogram_exponential(h: np.ndarray, psill: float, vrange: float, nugget: float):
    # gamma(h) = nugget + psill * (1 - exp(-h / range))
    h = np.asarray(h, dtype=float)
    vrange = max(float(vrange), 1e-6)
    psill = max(float(psill), 0.0)
    nugget = max(float(nugget), 0.0)
    return nugget + psill * (1.0 - np.exp(-h / vrange))

def covariance_from_variogram(h: np.ndarray, psill: float, vrange: float, nugget: float):
    # C(h) = (psill + nugget) - gamma(h)
    gamma = variogram_exponential(h, psill, vrange, nugget)
    return (psill + nugget) - gamma


# --------------------------- Anisotropic scaled distance ---------------------------

def anisotropic_transform_xy(dx: np.ndarray, dy: np.ndarray, az_deg: float, ratio: float):
    # rotate by az then stretch x' by ratio (ratio>=1 means longer correlation in x'? we scale distance)
    az = np.deg2rad(float(az_deg))
    ca, sa = np.cos(az), np.sin(az)
    x = ca * dx + sa * dy
    y = -sa * dx + ca * dy
    # distance scaling: if correlation longer in x, distance along x should be divided by ratio
    ratio = max(float(ratio), 1e-6)
    x_scaled = x / ratio
    return x_scaled, y

def scaled_distance(p: np.ndarray, q: np.ndarray, cfg: KrigingCfg) -> float:
    dx = q[0] - p[0]
    dy = q[1] - p[1]
    dz = (q[2] - p[2]) / max(cfg.z_scale, 1e-6)
    xs, ys = anisotropic_transform_xy(dx, dy, cfg.anis_azimuth_deg, cfg.anis_ratio_xy)
    return float(np.sqrt(xs*xs + ys*ys + dz*dz))


# --------------------------- Local Ordinary Kriging ---------------------------

def ok_predict_local(train_xyz: np.ndarray, train_y: np.ndarray,
                     query_xyz: np.ndarray,
                     cfg: KrigingCfg,
                     psill: float, vrange: float, nugget: float) -> np.ndarray:
    """
    Local OK with k-nearest neighbors (in anisotropic scaled distance).
    NaN-safe with jitter and fallbacks.
    """
    n = train_xyz.shape[0]
    ymean = float(np.nanmean(train_y))

    out = np.zeros(query_xyz.shape[0], dtype=float)

    # precompute for speed (still fine for n~7k, query~2k)
    for i in tqdm(range(query_xyz.shape[0]), desc="Kriging Predict", ncols=90):
        q = query_xyz[i]
        # distances to all train
        dists = np.zeros(n, dtype=float)
        for j in range(n):
            dists[j] = scaled_distance(train_xyz[j], q, cfg)

        if cfg.max_range is not None:
            mask = dists <= cfg.max_range
            idx_all = np.where(mask)[0]
            if idx_all.size == 0:
                out[i] = ymean
                continue
            idx = idx_all[np.argsort(dists[idx_all])][:cfg.k_neighbors]
        else:
            idx = np.argsort(dists)[:cfg.k_neighbors]

        pts = train_xyz[idx]
        vals = train_y[idx]
        vals, _ = _replace_nonfinite(vals, ymean)

        m = pts.shape[0]
        if m < 3:
            out[i] = float(np.mean(vals))
            continue

        # build OK system: [C 1; 1^T 0] [w; mu] = [c; 1]
        # covariance uses (psill+nugget) - gamma(h); add tiny jitter on diag
        C = np.zeros((m, m), dtype=float)
        for a in range(m):
            C[a, a] = (psill + nugget) + cfg.nugget + cfg.jitter
            for b in range(a+1, m):
                h = scaled_distance(pts[a], pts[b], cfg)
                c = covariance_from_variogram(h, psill, vrange, nugget)
                C[a, b] = c
                C[b, a] = c

        cvec = np.zeros(m, dtype=float)
        for a in range(m):
            h = scaled_distance(pts[a], q, cfg)
            cvec[a] = covariance_from_variogram(h, psill, vrange, nugget)

        A = np.zeros((m+1, m+1), dtype=float)
        A[:m, :m] = C
        A[:m, m] = 1.0
        A[m, :m] = 1.0
        A[m, m] = 0.0

        b = np.zeros(m+1, dtype=float)
        b[:m] = cvec
        b[m] = 1.0

        try:
            sol = np.linalg.solve(A, b)
            w = sol[:m]
            pred = float(np.dot(w, vals))
            if not np.isfinite(pred):
                pred = float(np.mean(vals))
            out[i] = pred
        except np.linalg.LinAlgError:
            out[i] = float(np.mean(vals))

    out, _ = _replace_nonfinite(out, ymean)
    return out


# --------------------------- Simple robust "fit" for variogram params ---------------------------

def fit_variogram_params(train_xyz: np.ndarray, y: np.ndarray, cfg: KrigingCfg,
                         seed: int = 42, n_pairs: int = 6000) -> Tuple[float, float, float, Dict]:
    """
    Very robust, low-cost fitting:
      - sample pairs
      - estimate sill from variance
      - set range from median distance of informative pairs
      - nugget from small-distance semivariance
    This avoids unstable optimization that created NaN / range=0 in your logs.
    """
    rng = np.random.RandomState(seed)
    y = np.asarray(y, dtype=float)
    y, _ = _replace_nonfinite(y, float(np.nanmean(y)))
    var = float(np.var(y))
    psill = max(var * 0.6, 1e-3)
    nugget = max(var * 0.1, 1e-6)

    n = train_xyz.shape[0]
    if n < 10:
        return psill, 1.0, nugget, {"note": "too few points"}

    # sample pairs
    idx_a = rng.randint(0, n, size=n_pairs)
    idx_b = rng.randint(0, n, size=n_pairs)
    hs = []
    gs = []
    for a, b in zip(idx_a, idx_b):
        if a == b:
            continue
        h = scaled_distance(train_xyz[a], train_xyz[b], cfg)
        g = 0.5 * (y[a] - y[b])**2
        if np.isfinite(h) and np.isfinite(g):
            hs.append(h)
            gs.append(g)
    hs = np.asarray(hs, dtype=float)
    gs = np.asarray(gs, dtype=float)
    if hs.size < 50:
        vrange = float(np.median(hs)) if hs.size > 0 else 1.0
        return psill, max(vrange, 1e-3), nugget, {"note": "few pairs"}

    # range: median of distances excluding extreme tail
    vrange = float(np.quantile(hs, 0.5))
    vrange = max(vrange, 1e-3)

    # nugget: median semivariance of the smallest 10% distances
    small_mask = hs <= np.quantile(hs, 0.10)
    if np.any(small_mask):
        nugget = float(np.median(gs[small_mask]))
        nugget = max(nugget, 1e-6)

    # psill: clamp to remaining variance scale
    psill = max(var - nugget, 1e-3)

    diag = {
        "var": var,
        "psill": psill,
        "range": vrange,
        "nugget": nugget,
        "pairs": int(hs.size),
        "hs_q50": float(np.quantile(hs, 0.5)),
        "hs_q90": float(np.quantile(hs, 0.9)),
    }
    return psill, vrange, nugget, diag


# --------------------------- Metrics & Plots ---------------------------

def calc_metrics(y_true, y_pred) -> Dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_pred, _ = _replace_nonfinite(y_pred, float(np.nanmean(y_true)))
    r2 = float(r2_score(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    # MAPE with safe denom
    denom = np.clip(np.abs(y_true), 1e-6, None)
    mape = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)
    bias = float(np.mean(y_pred - y_true))
    return {"R2": r2, "RMSE": rmse, "MAE": mae, "MAPE(%)": mape, "Bias": bias}

def save_scatter(outdir: str, y_true: np.ndarray, y_pred: np.ndarray, title: str, fname: str):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_pred, _ = _replace_nonfinite(y_pred, float(np.nanmean(y_true)))

    r2 = r2_score(y_true, y_pred)
    plt.figure(figsize=(6.0, 6.0))
    plt.scatter(y_true, y_pred, s=10, alpha=0.6)
    mn = float(min(np.min(y_true), np.min(y_pred)))
    mx = float(max(np.max(y_true), np.max(y_pred)))
    plt.plot([mn, mx], [mn, mx], linewidth=1)
    plt.xlabel("Observed TFe")
    plt.ylabel("Predicted TFe")
    plt.title(f"{title} | R2={r2:.3f}")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, fname), dpi=220)
    plt.close()

def save_error_boxplot(outdir: str, err_dict: Dict[str, np.ndarray], fname="error_boxplot.png"):
    plt.figure(figsize=(7.2, 4.0))
    labels = list(err_dict.keys())
    data = [err_dict[k] for k in labels]
    plt.boxplot(data, tick_labels=labels, showfliers=False)
    plt.ylabel("Prediction Error (Pred - Obs)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, fname), dpi=220)
    plt.close()

def save_error_hist_overlay(outdir: str, err_dict: Dict[str, np.ndarray],
                            hist_range=(-25, 25), bins=40, fname="error_hist_overlay.png"):
    plt.figure(figsize=(7.2, 4.0))
    for k, e in err_dict.items():
        e = np.asarray(e, dtype=float)
        e = e[np.isfinite(e)]
        plt.hist(e, bins=bins, range=hist_range, alpha=0.45, density=True, label=k)
    plt.xlabel("Prediction Error (Pred - Obs)")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, fname), dpi=220)
    plt.close()

def save_error_vs_depth(outdir: str, depth: np.ndarray, err_dict: Dict[str, np.ndarray], fname="error_vs_depth.png"):
    plt.figure(figsize=(7.2, 4.0))
    for k, e in err_dict.items():
        plt.scatter(depth, e, s=6, alpha=0.35, label=k)
    plt.xlabel("Depth_mid")
    plt.ylabel("Prediction Error (Pred - Obs)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, fname), dpi=220)
    plt.close()


# --------------------------- ARL (your "framework" proxy) ---------------------------

def fit_arl(train_df: pd.DataFrame, seed: int = 42) -> Tuple[Pipeline, List[str], List[str]]:
    cov_cols = [c for c in train_df.columns if c.startswith("Cov_")]
    num_cols = ["E", "N", "R", "Depth_mid"] + cov_cols
    cat_cols = ["Lith"] if "Lith" in train_df.columns else []

    # Numeric: impute + standardize
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    # Cat: impute + OHE
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore"))
    ])

    pre = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols)
    ], remainder="drop")

    # HistGradientBoosting is robust and handles nonlinearities well
    model = HistGradientBoostingRegressor(
        random_state=seed,
        learning_rate=0.05,
        max_depth=6,
        max_iter=800,
        l2_regularization=0.0,
        min_samples_leaf=30,
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


# --------------------------- KED / Regression-Kriging baseline ---------------------------

def regression_kriging_ked(train_df: pd.DataFrame,
                           test_df: pd.DataFrame,
                           cfg: KrigingCfg,
                           seed: int = 42,
                           use_xyz_in_drift: bool = True,
                           alpha_grid=(0.1, 1.0, 10.0, 50.0, 200.0)) -> Tuple[np.ndarray, Dict]:
    """
    Traditional geostatistics baseline that can be competitive:
      drift: Ridge on covariates + depth (+ optional xyz)
      residual: OK
    NaN-safe by design.
    """
    cov_cols = [c for c in train_df.columns if c.startswith("Cov_")]

    # drop covariate columns that are entirely NaN in training
    good_cov = []
    for c in cov_cols:
        s = pd.to_numeric(train_df[c], errors="coerce")
        if np.isfinite(s).any():
            good_cov.append(c)
    dropped = list(sorted(set(cov_cols) - set(good_cov)))
    cov_cols = good_cov

    num_cols = cov_cols + ["Depth_mid"]
    if use_xyz_in_drift:
        num_cols += ["E", "N", "R"]
    cat_cols = ["Lith"] if "Lith" in train_df.columns else []

    transformers = []
    if num_cols:
        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("num", num_pipe, num_cols))
    if cat_cols:
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ])
        transformers.append(("cat", cat_pipe, cat_cols))

    pre = ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.3)

    X = train_df[num_cols + cat_cols].copy()
    y = train_df["TFe"].values.astype(float)
    groups = train_df["BHID"].astype(str).values
    gkf = GroupKFold(n_splits=5)

    best_alpha = None
    best_rmse = np.inf
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

    # fit variogram for residuals
    train_xyz = train_df[["E", "N", "R"]].values.astype(float)
    psill, vrange, nugget, vdiag = fit_variogram_params(train_xyz, resid, cfg, seed=seed+11)

    # residual OK
    tr_res = train_df.copy()
    tr_res["RESID"] = resid
    yres_hat = ok_predict_local(
        train_xyz=tr_res[["E", "N", "R"]].values.astype(float),
        train_y=tr_res["RESID"].values.astype(float),
        query_xyz=test_df[["E", "N", "R"]].values.astype(float),
        cfg=cfg,
        psill=psill, vrange=vrange, nugget=nugget
    )

    Xt = test_df[num_cols + cat_cols].copy()
    drift_test = drift.predict(Xt)
    drift_test, _ = _replace_nonfinite(drift_test, float(np.nanmean(y)))
    yhat = drift_test + yres_hat
    yhat, _ = _replace_nonfinite(yhat, float(np.nanmean(y)))

    diag = {
        "drift_model": "Ridge(+Imputer)",
        "best_alpha": best_alpha,
        "drift_groupcv_rmse": best_rmse,
        "use_xyz_in_drift": use_xyz_in_drift,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "dropped_all_nan_cov_cols": dropped,
        "resid_variogram": {"model": cfg.model, **vdiag},
    }
    return yhat, diag


# --------------------------- OOF Residual Kriging Fusion for ARL ---------------------------

def oof_residual_kriging_fusion(train_df: pd.DataFrame,
                               test_df: pd.DataFrame,
                               cfg: KrigingCfg,
                               seed: int = 42,
                               alpha_grid=(0.0, 0.25, 0.5, 0.75, 1.0)) -> Tuple[np.ndarray, Dict, np.ndarray]:
    """
    Fit ARL on train, compute OOF residuals (GroupKFold by BHID), krige residuals,
    then fuse with alpha: y = y_arl + alpha * y_resid_kriged
    alpha is selected by GroupCV on train (leakage-safe).
    """
    # Fit ARL once
    arl_pipe, num_cols, cat_cols = fit_arl(train_df, seed=seed)

    y_train = train_df["TFe"].values.astype(float)
    groups = train_df["BHID"].astype(str).values
    gkf = GroupKFold(n_splits=5)

    X_train = train_df[num_cols + cat_cols]
    # OOF predictions
    oof = np.zeros(len(train_df), dtype=float)
    for tr, va in gkf.split(X_train, y_train, groups=groups):
        sub_pipe, _, _ = fit_arl(train_df.iloc[tr].reset_index(drop=True), seed=seed+17)
        pred = predict_arl(sub_pipe, train_df.iloc[va].reset_index(drop=True), num_cols, cat_cols)
        oof[va] = pred

    oof, _ = _replace_nonfinite(oof, float(np.nanmean(y_train)))
    resid = y_train - oof
    resid = resid - float(np.mean(resid))

    # variogram on residuals
    train_xyz = train_df[["E", "N", "R"]].values.astype(float)
    psill, vrange, nugget, vdiag = fit_variogram_params(train_xyz, resid, cfg, seed=seed+23)

    # predict residuals on train (for alpha selection) and on test
    # To keep leakage-safe for alpha selection, we do GroupCV scoring where residual kriging also re-fit inside folds.
    def score_alpha(alpha: float) -> float:
        rmses = []
        for tr, va in gkf.split(train_df, y_train, groups=groups):
            tr_df = train_df.iloc[tr].reset_index(drop=True)
            va_df = train_df.iloc[va].reset_index(drop=True)

            # ARL refit on tr, predict va
            tr_pipe, ncols, ccols = fit_arl(tr_df, seed=seed+31)
            pred_va_arl = predict_arl(tr_pipe, va_df, ncols, ccols)

            # OOF residuals on tr: for simplicity, use in-fold residuals (still leakage-safe w.r.t va)
            pred_tr_arl = predict_arl(tr_pipe, tr_df, ncols, ccols)
            r_tr = (tr_df["TFe"].values.astype(float) - pred_tr_arl)
            r_tr = r_tr - float(np.mean(r_tr))

            # krige residual from tr to va
            tr_xyz = tr_df[["E", "N", "R"]].values.astype(float)
            va_xyz = va_df[["E", "N", "R"]].values.astype(float)
            ps, rg, ng, _ = fit_variogram_params(tr_xyz, r_tr, cfg, seed=seed+41)

            r_va_hat = ok_predict_local(tr_xyz, r_tr, va_xyz, cfg, ps, rg, ng)
            yhat = pred_va_arl + alpha * r_va_hat
            rmses.append(float(np.sqrt(mean_squared_error(va_df["TFe"].values.astype(float), yhat))))
        return float(np.mean(rmses))

    best_alpha = None
    best_rmse = np.inf
    for a in alpha_grid:
        rmse = score_alpha(float(a))
        if rmse < best_rmse:
            best_rmse = rmse
            best_alpha = float(a)

    # final predictions
    y_test_arl = predict_arl(arl_pipe, test_df, num_cols, cat_cols)
    test_xyz = test_df[["E", "N", "R"]].values.astype(float)
    y_test_resid = ok_predict_local(train_xyz, resid, test_xyz, cfg, psill, vrange, nugget)
    y_test_fused = y_test_arl + best_alpha * y_test_resid
    y_test_fused, _ = _replace_nonfinite(y_test_fused, float(np.nanmean(y_train)))

    diag = {
        "arl_model": "HistGradientBoostingRegressor",
        "alpha_grid": list(alpha_grid),
        "best_alpha": best_alpha,
        "groupcv_rmse": best_rmse,
        "resid_variogram": {"model": cfg.model, **vdiag},
        "note": "Fusion = ARL + alpha * OK(residual)"
    }
    return y_test_fused, diag, y_test_arl


# --------------------------- Main ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loc", default="定位1.xlsx", help="定位表 xlsx")
    ap.add_argument("--exp", default="实验数据.xlsx", help="实验数据 xlsx")
    ap.add_argument("--dev", default="侧斜.xlsx", help="侧斜 xlsx")
    ap.add_argument("--lith", default="岩性.xlsx", help="岩性 xlsx (optional; if not exists will be ignored)")
    ap.add_argument("--outdir", default="", help="output directory (default auto)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test_ratio", type=float, default=0.30)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--anis_az", type=float, default=45.0)
    ap.add_argument("--anis_ratio", type=float, default=3.0)
    ap.add_argument("--z_scale", type=float, default=2.0)
    ap.add_argument("--max_range_q", type=float, default=0.0,
                    help="if >0, set max_range as quantile(q) of train pair distances (scaled)")
    args = ap.parse_args()

    outdir = args.outdir.strip()
    if outdir == "":
        outdir = f"outputs_engineering_{_ts()}"
    ensure_dir(outdir)

    print(f"[INFO] SCRIPT_VERSION = {SCRIPT_VERSION}")
    print(f"[INFO] outdir = {outdir}")

    # Load excel
    if not os.path.exists(args.loc):
        raise FileNotFoundError(f"找不到 {args.loc}")
    if not os.path.exists(args.exp):
        raise FileNotFoundError(f"找不到 {args.exp}")
    if not os.path.exists(args.dev):
        raise FileNotFoundError(f"找不到 {args.dev}")

    df_loc = pd.read_excel(args.loc)
    df_exp = pd.read_excel(args.exp)
    df_dev = pd.read_excel(args.dev)

    df_lith = None
    if args.lith and os.path.exists(args.lith):
        df_lith = pd.read_excel(args.lith)
        print("[INFO] Lithology loaded and enabled.")
    else:
        print("[INFO] Lithology not found or disabled.")

    # Build samples
    df, st = build_samples(df_loc, df_exp, df_dev, df_lith=df_lith)
    print(f"[INFO] 样点数：{st['n_samples']}；孔数：{st['n_holes']}；缺实验孔：{st['miss_exp_holes']}；缺侧斜孔（直孔补齐）：{st['miss_dev_holes']}")
    if st["covariates"]:
        print(f"[INFO] Covariates used: {[c.replace('Cov_', '') for c in st['covariates']]}")

    y_all = df["TFe"].values.astype(float)
    print("[INFO] " + summarize_series("Grade(raw)", y_all).replace("\n", "\n[INFO] "))

    # Split
    df_train, df_test = split_by_hole(df, seed=args.seed, test_ratio=args.test_ratio)
    print(f"[INFO] Split: by_hole | train_holes={df_train['BHID'].nunique()} test_holes={df_test['BHID'].nunique()}")
    print("[INFO] " + summarize_series("Train Grade", df_train["TFe"].values).replace("\n", "\n[INFO] "))
    print("[INFO] " + summarize_series("Test  Grade", df_test["TFe"].values).replace("\n", "\n[INFO] "))

    # Kriging cfg
    cfg = KrigingCfg(
        k_neighbors=int(args.k),
        anis_azimuth_deg=float(args.anis_az),
        anis_ratio_xy=float(args.anis_ratio),
        z_scale=float(args.z_scale),
        nugget=1e-6,
        jitter=1e-8,
        max_range=None,
        model="exponential"
    )

    # optional max_range from distance quantile
    if args.max_range_q and args.max_range_q > 0:
        q = float(args.max_range_q)
        q = min(max(q, 0.5), 0.99)
        # sample distances
        rng = np.random.RandomState(args.seed + 7)
        xyz = df_train[["E", "N", "R"]].values.astype(float)
        n = xyz.shape[0]
        m = min(5000, n * 2)
        idx_a = rng.randint(0, n, size=m)
        idx_b = rng.randint(0, n, size=m)
        ds = []
        for a, b in zip(idx_a, idx_b):
            if a == b:
                continue
            ds.append(scaled_distance(xyz[a], xyz[b], cfg))
        ds = np.asarray(ds, dtype=float)
        cfg.max_range = float(np.quantile(ds, q))
        print(f"[INFO] Kriging max_range enabled: q={q} -> max_range={cfg.max_range:.3f} (scaled)")

    # ----------------- Method 1: Traditional Kriging baseline (KED) -----------------
    yk, ked_diag = regression_kriging_ked(df_train, df_test, cfg=cfg, seed=args.seed, use_xyz_in_drift=True)
    mk = calc_metrics(df_test["TFe"].values, yk)

    # ----------------- Method 2: ARL -----------------
    arl_pipe, ncols, ccols = fit_arl(df_train, seed=args.seed)
    ya = predict_arl(arl_pipe, df_test, ncols, ccols)
    ma = calc_metrics(df_test["TFe"].values, ya)

    # ----------------- Method 3: ARL + OOF Residual Kriging (RouteB Fusion) -----------------
    y_fuse, fdiag, y_arl_for_diag = oof_residual_kriging_fusion(df_train, df_test, cfg=cfg, seed=args.seed)
    mf = calc_metrics(df_test["TFe"].values, y_fuse)

    # report
    res = pd.DataFrame([
        {"Method": "Kriging(KED)", **mk},
        {"Method": "ARL_full", **ma},
        {"Method": "RouteB_Fusion(ARL+OOF-RK)", **mf},
    ])
    print(res)

    # Save metrics
    res.to_csv(os.path.join(outdir, "metrics.csv"), index=False, encoding="utf-8-sig")

    # Save predictions
    out_pred = df_test[["BHID", "Depth_mid", "E", "N", "R", "TFe", "Lith"]].copy()
    out_pred["Pred_Kriging_KED"] = yk
    out_pred["Pred_ARL"] = ya
    out_pred["Pred_Fusion"] = y_fuse
    out_pred.to_csv(os.path.join(outdir, "predictions_test.csv"), index=False, encoding="utf-8-sig")

    # Plots
    y_true = df_test["TFe"].values.astype(float)
    save_scatter(outdir, y_true, yk, "Kriging(KED)", "scatter_R2_kriging_ked.png")
    save_scatter(outdir, y_true, ya, "ARL_full", "scatter_R2_arl.png")
    save_scatter(outdir, y_true, y_fuse, "RouteB_Fusion(ARL+OOF-RK)", "scatter_R2_fusion.png")

    err_dict = {
        "Kriging(KED)": (yk - y_true),
        "ARL": (ya - y_true),
        "Fusion": (y_fuse - y_true),
    }
    save_error_boxplot(outdir, err_dict, fname="error_boxplot.png")
    save_error_hist_overlay(outdir, err_dict, hist_range=(-25, 25), bins=40, fname="error_hist_overlay.png")
    save_error_vs_depth(outdir, df_test["Depth_mid"].values.astype(float), err_dict, fname="error_vs_depth.png")

    # Save diagnostics
    diag = {
        "SCRIPT_VERSION": SCRIPT_VERSION,
        "KED": ked_diag,
        "FUSION": fdiag,
        "KrigingCfg": cfg.__dict__
    }
    with open(os.path.join(outdir, "diagnostics.txt"), "w", encoding="utf-8") as f:
        f.write(str(diag))

    print(f"[INFO] Done. Outputs saved to: {outdir}")


if __name__ == "__main__":
    main()
