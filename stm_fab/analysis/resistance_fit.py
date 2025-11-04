# stm_fab/analysis/resistance_fit.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
from scipy.optimize import curve_fit
from scipy import interpolate

def model_invT(T_K, R0, T0, Rc):
    return R0 * np.exp(T0 / T_K) + Rc

def model_expT(T_K, a, b, c):
    return a * np.exp(b * T_K) + c

@dataclass
class FitResult:
    model_name: str
    params: Tuple[float, ...]
    cov: Optional[np.ndarray]
    r2: float
    rmse: float
    n: int
    notes: str = ""

def _initial_guesses_invT(T_K: np.ndarray, R: np.ndarray) -> Tuple[float, float, float]:
    Rc0 = np.percentile(R, 5) - 0.05 * (np.percentile(R, 95) - np.percentile(R, 5))
    Rc0 = min(Rc0, np.min(R) - 1e-6)
    R_adj = np.maximum(R - Rc0, 1e-9)
    x = 1.0 / T_K
    y = np.log(R_adj)
    A = np.vstack([np.ones_like(x), x]).T
    try:
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        lnR0, T0 = beta
        R0 = np.exp(lnR0)
        if not np.isfinite(R0) or R0 <= 0: R0 = np.median(R_adj)
        if not np.isfinite(T0) or T0 <= 0: T0 = 5000.0
    except Exception:
        R0 = np.median(R_adj)
        T0 = 5000.0
    return float(R0), float(T0), float(Rc0)

def _fit_model(model_name: str, T_K: np.ndarray, R: np.ndarray,
               bounds: Optional[Tuple[Tuple[float, ...], Tuple[float, ...]]] = None) -> FitResult:
    if model_name == "invT":
        p0 = _initial_guesses_invT(T_K, R)
        f = model_invT
        k = 3
        if bounds is None:
            bounds = ((1e-12, 1.0, -np.inf), (np.inf, 2.0e4, np.inf))
    elif model_name == "expT":
        a0 = max(1e-9, np.percentile(R, 20))
        b0 = 1e-3 / max(1.0, np.median(T_K))
        c0 = np.percentile(R, 5)
        p0 = (a0, b0, c0)
        f = model_expT
        k = 3
        if bounds is None:
            bounds = ((-np.inf, -1e2, -np.inf), (np.inf, 1e2, np.inf))
    else:
        raise ValueError("Unknown model name")

    popt, pcov = curve_fit(f, T_K, R, p0=p0, bounds=bounds, maxfev=100000)
    R_hat = f(T_K, *popt)
    resid = R - R_hat
    n = len(R)
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((R - np.mean(R))**2)) if n > 1 else 0.0
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    rmse = float(np.sqrt(ss_res / n)) if n > 0 else np.nan
    return FitResult(model_name=model_name, params=tuple(popt), cov=pcov, r2=r2, rmse=rmse, n=n)

def fit_resistance_vs_temperature(
    df: pd.DataFrame,
    time_col: str = "Time",
    temp_col_candidates=("Pyro_T","Pyro T","Pyro_Temp"),
    R_col_candidates=("TDK_R","TDK R"),
    V_col_candidates=("TDK_V","TDK V"),
    I_col_candidates=("TDK_I","TDK I"),
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    prefer_model: str = "invT",
    try_both_and_choose: bool = False,
    allow_negative_Rc: bool = True
) -> Dict[str, Any]:
    cols = {c.lower().replace(" ", "_"): c for c in df.columns}
    def pick(col_names):
        for name in col_names:
            key = name.lower().replace(" ", "_")
            if key in cols:
                return cols[key]
        return None

    time_col = pick((time_col,)) or pick(("Time_s","X_Value","X Value","time"))
    temp_col = pick(temp_col_candidates)
    R_col = pick(R_col_candidates)
    V_col = pick(V_col_candidates)
    I_col = pick(I_col_candidates)

    if temp_col is None: raise ValueError("Temperature column not found")
    if I_col is None: raise ValueError("Current column (TDK_I) not found")
    if R_col is None and V_col is None: raise ValueError("Need TDK_R or TDK_V to compute R")

    d = df.copy()
    if time_col and (start_time is not None or end_time is not None):
        mask = np.ones(len(d), dtype=bool)
        if start_time is not None: mask &= d[time_col] >= start_time
        if end_time is not None: mask &= d[time_col] <= end_time
        d = d.loc[mask]

    T_C = d[temp_col].to_numpy(dtype=float)
    T_K = T_C + 273.15
    I = d[I_col].to_numpy(dtype=float)

    if R_col is not None:
        R = d[R_col].to_numpy(dtype=float)
    else:
        V = d[V_col].to_numpy(dtype=float)
        with np.errstate(divide='ignore', invalid='ignore'):
            R = V / I

    good = np.isfinite(T_K) & np.isfinite(R) & np.isfinite(I) & (T_K > 0) & (I != 0)
    T_K = T_K[good]; R = R[good]; I = I[good]
    if len(T_K) < 10: raise ValueError("Insufficient points in selected region for fitting")

    results = []
    if prefer_model == "invT":
        results.append(_fit_model("invT", T_K, R))
        if try_both_and_choose:
            try: results.append(_fit_model("expT", T_K, R))
            except: pass
    else:
        results.append(_fit_model("expT", T_K, R))
        if try_both_and_choose:
            try: results.append(_fit_model("invT", T_K, R))
            except: pass

    best = None; best_score = np.inf
    for r in results:
        rss = (r.rmse**2) * r.n
        k = len(r.params); n = r.n
        aic = n*np.log(rss / max(n,1)) + 2*k if rss > 0 else -np.inf
        if aic < best_score:
            best_score = aic; best = r
    if best is None: raise RuntimeError("No valid fit produced")

    sort_idx = np.argsort(R)
    R_sorted = R[sort_idx]; I_sorted = I[sort_idx]
    R_unique, unique_idx = np.unique(R_sorted, return_index=True)
    I_unique = I_sorted[unique_idx]
    I_of_R = interpolate.interp1d(R_unique, I_unique, kind='linear', bounds_error=False, fill_value="extrapolate")

    fig1, ax1 = plt.subplots(1, 2, figsize=(12, 5))
    if time_col:
        t = d.loc[good, time_col].to_numpy()
        ax1[0].plot(t, T_C[good], 'r-', label='Pyro T (°C)')
        ax1[0].set_xlabel("Time (s)"); ax1[0].set_ylabel("T (°C)", color='r'); ax1[0].grid(True, alpha=0.3)
        ax0b = ax1[0].twinx()
        ax0b.plot(t, R, 'b-', label='R (Ω)')
        ax0b.set_ylabel("R (Ω)", color='b')
    else:
        ax1[0].plot(T_C, R, 'k.', ms=3, alpha=0.6)
        ax1[0].set_xlabel("T (°C)"); ax1[0].set_ylabel("R (Ω)"); ax1[0].set_title("Scatter R vs T")

    Tmin, Tmax = np.nanmin(T_K), np.nanmax(T_K)
    Tgrid = np.linspace(Tmin, Tmax, 400)
    if best.model_name == "invT":
        Rfit = model_invT(Tgrid, *best.params)
        model_label = "R = R0·exp(T0/T) + Rc"
    else:
        Rfit = model_expT(Tgrid, *best.params)
        model_label = "R = a·exp(b·T) + c"
    ax1[1].plot(T_K, R, 'k.', ms=3, alpha=0.6, label="Data")
    ax1[1].plot(Tgrid, Rfit, 'r-', lw=2, label=f"Fit ({model_label})")
    ax1[1].set_xlabel("T (K)"); ax1[1].set_ylabel("R (Ω)"); ax1[1].grid(True, alpha=0.3); ax1[1].legend(loc='best')
    fig1.tight_layout()

    return {
        "fit": best,
        "I_of_R": I_of_R,
        "figures": {"overview_and_fit": fig1},
        "data_used": {"T_K": T_K, "R": R, "I": I, "time": d.loc[good, time_col].to_numpy() if time_col else None}
    }

def current_for_temperature(fit_pack: Dict[str, Any], T_target_C: float) -> Dict[str, float]:
    T_target_K = T_target_C + 273.15
    fit = fit_pack["fit"]; I_of_R = fit_pack["I_of_R"]
    if fit.model_name == "invT":
        R_target = model_invT(T_target_K, *fit.params)
    else:
        R_target = model_expT(T_target_K, *fit.params)
    I_target = float(I_of_R(R_target))
    return {"T_target_C": T_target_C, "R_target_ohm": float(R_target), "I_target_A": I_target}
