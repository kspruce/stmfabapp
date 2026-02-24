# stm_fab/analysis/process_metrics.py
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import json
from datetime import datetime

try:
    from stm_fab.labview.labview_parser import LabVIEWParser
except ImportError:
    LabVIEWParser = None


# ==================== UTILITY FUNCTIONS ====================

def format_time(seconds: float) -> str:
    """
    Format time in seconds as HH:MM:SS
    """
    if not np.isfinite(seconds) or seconds < 0:
        return "00:00:00"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def detect_file_type(filename: str) -> str:
    """Detect process type from filename"""
    s = filename.lower()
    if 'flash' in s or 'cooldown' in s:
        return 'flash'
    elif 'hterm' in s:
        return 'hterm'
    elif 'term' in s:
        return 'termination'
    elif 'dose' in s:
        return 'dose'
    elif 'inc' in s:
        return 'incorporation'
    elif 'susi' in s or 'overgrowth' in s or 'ovg' in s:
        return 'susi'
    elif 'outgas' in s or 'degas' in s:
        return 'outgas'
    return 'unknown'


def detect_molecular_weight(filename: str) -> float:
    """
    Detect molecular weight from filename
    """
    s = filename.lower()
    if 'ash3' in s or 'ash_3' in s or 'arsine' in s:
        return 77.95
    elif 'ph3' in s or 'ph_3' in s or 'phosphine' in s:
        return 34.0
    return 34.0


# ==================== COLUMN HELPERS ====================

def _find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Return the first matching column name from candidates, else None."""
    cols = {c: c for c in df.columns}
    normalized = {c.lower().replace(' ', '_'): c for c in df.columns}
    for cand in candidates:
        if cand in cols:
            return cand
        key = cand.lower().replace(' ', '_')
        if key in normalized:
            return normalized[key]
    return None


def _find_current_column(df: pd.DataFrame) -> Optional[str]:
    """Best-effort current column finder."""
    preferred = [
        'TDK_I','TDK_I_1','TDK_I_0','TDK_I1','TDK_Current','TDKI','TDK_I_2','TDK_I_3'
    ]
    hit = _find_column(df, preferred)
    if hit:
        return hit
    for c in df.columns:
        cl = c.lower()
        if 'tdk' in cl and ('_i' in cl or ' i' in cl or cl.endswith('i') or '_current' in cl):
            return c
    return None


def _find_susi_current_column(df: pd.DataFrame) -> Optional[str]:
    """Prefer SUSI current column (TDK_I_1)"""
    cols = set(df.columns)
    if 'TDK_I_1' in cols: return 'TDK_I_1'
    if 'TDK I 1' in cols: return 'TDK I 1'
    preferred = ['SUSI_I','SUSI_Current','SUSI','TDK_I1','TDK_CH2_I','TDK2_I']
    hit = _find_column(df, preferred)
    if hit: return hit
    for c in df.columns:
        cl = c.lower().replace(' ', '_')
        if 'susi' in cl and ('_i' in cl or cl.endswith('i') or 'current' in cl):
            return c
    return _find_current_column(df)


def _find_sample_current_column(df: pd.DataFrame) -> Optional[str]:
    """Prefer sample current column (TDK_I)"""
    cols = set(df.columns)
    if 'TDK_I' in cols: return 'TDK_I'
    if 'TDK I' in cols: return 'TDK I'
    preferred = ['Sample_I','Sample_Current','I_Sample','TDK_CH1_I','TDK0_I']
    hit = _find_column(df, preferred)
    if hit: return hit
    for c in df.columns:
        cl = c.lower().replace(' ', '_')
        if 'sample' in cl and ('_i' in cl or cl.endswith('i') or 'current' in cl):
            return c
    return _find_current_column(df)


def _find_pressure_column(df: pd.DataFrame, prefer: str = 'VT') -> Optional[str]:
    """Find pressure column (P_MBE or P_VT family)."""
    cols = set(df.columns)
    vt_candidates = ['P_VT','PVT','P_VT_1','P_VT_Torr','P_VT_mbar','P_VT_Pa']
    mbe_candidates = ['P_MBE','PMBE','P_MBE_Torr','P_MBE_mbar','P_MBE_Pa']

    def first_hit(cands):
        for c in cands:
            if c in cols: return c
        norm = {c.lower().replace(' ','_'): c for c in df.columns}
        for c in cands:
            key = c.lower().replace(' ','_')
            if key in norm: return norm[key]
        return None

    if prefer == 'VT':
        vt = first_hit(vt_candidates)
        if vt: return vt
        mbe = first_hit(mbe_candidates)
        if mbe: return mbe
    else:
        mbe = first_hit(mbe_candidates)
        if mbe: return mbe
        vt = first_hit(vt_candidates)
        if vt: return vt

    for c in df.columns:
        cl = c.lower().replace(' ','_')
        if cl.startswith('p_') or cl == 'p' or 'pressure' in cl:
            return c
    return None


def _segments_from_mask(time: np.ndarray, mask: np.ndarray) -> List[Dict[str, float]]:
    """Boolean mask -> continuous time segments."""
    if len(time) != len(mask) or len(time) == 0:
        return []
    padded = np.concatenate(([False], mask, [False]))
    diffs = np.diff(padded.astype(int))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    segs = []
    for s, e in zip(starts, ends):
        seg_t = time[s:e]
        if len(seg_t) == 0:
            continue
        dur = float(seg_t[-1] - seg_t[0]) if len(seg_t) > 1 else 0.0
        segs.append({'start_time_s': float(seg_t[0]),
                     'end_time_s': float(seg_t[-1]),
                     'duration_s': dur})
    return segs


def _find_voltage_columns(df: pd.DataFrame) -> List[str]:
    """
    Return an ordered list of candidate voltage columns.
    We prefer sense/specific channels before generic TDK_V.
    """
    preferred = [
        # common sense/measurement names first
        'Sample_V', 'Sample_Voltage', 'Sense_V', 'Vsense', 'V_sense', 'R_Sense_V',
        'TDK_CH1_V', 'TDK_CH1_Voltage', 'TDK_V_0', 'TDK_V_1',
        # generic TDK voltage last
        'TDK_V', 'TDK_Voltage',
        # catch-all generic names
        'Voltage', 'V'
    ]
    hits = []
    cols = set(df.columns)
    for c in preferred:
        if c in cols:
            hits.append(c)
    # Also add any column that looks like voltage but not already included
    for c in df.columns:
        cl = c.lower()
        if ('v' in cl or 'volt' in cl) and (c not in hits):
            hits.append(c)
    return hits


def _find_resistance_columns(df: pd.DataFrame) -> List[str]:
    """
    Return an ordered list of candidate resistance columns if present.
    """
    preferred = [
        'Resistance', 'R', 'Sample_R', 'Heater_R', 'R_ohm', 'Ohms', 'R_Sense', 'R_meas', 'R_Ohm'
    ]
    hits = []
    cols = set(df.columns)
    for c in preferred:
        if c in cols:
            hits.append(c)
    # Heuristic catch-all for names that contain 'resist' or 'ohm'
    for c in df.columns:
        cl = c.lower()
        if (('resist' in cl) or ('ohm' in cl)) and (c not in hits):
            hits.append(c)
    return hits



# ==================== DOSE ANALYSIS ====================

def analyze_dose(df: pd.DataFrame,
                 pressure_col: str = 'P_VT',
                 time_col: str = 'Time_s',
                 molecular_weight: Optional[float] = None,
                 filename: str = '',
                 temperature_K: float = 300.0,
                 pressure_units: str = 'mbar',
                 # NEW optional controls (defaults keep legacy behavior)
                 method: str = 'baseline_or_abs',            # 'baseline_or_abs' (default) or 'relative_to_peak'
                 entry_frac: float = 0.10,                   # for 'relative_to_peak' (10% of (peak-baseline))
                 exit_frac: float = 0.05,                    # hysteresis exit (5% of (peak-baseline))
                 min_duration_s: float = 60.0                # require ≥ this duration above exit threshold
                 ) -> Dict[str, Any]:
    """
    Analyze dose step: calculate flux, exposure, integrated molecules.

    method:
      - 'baseline_or_abs' (default): legacy behavior (baseline*10 vs absolute floor)
      - 'relative_to_peak': detect main plateau using thresholds relative to
        (peak - baseline_pre), with hysteresis and minimum duration. This is
        robust for large dynamic range H2 doses (e.g., ~5e-7) with ramps.
    """
    if pressure_col not in df.columns or time_col not in df.columns:
        return {'error': f'Missing columns: {pressure_col} or {time_col}'}

    if molecular_weight is None:
        molecular_weight = detect_molecular_weight(filename)

    time = df[time_col].to_numpy(dtype=float)
    pressure = df[pressure_col].to_numpy(dtype=float)

    mask = np.isfinite(time) & np.isfinite(pressure) & (pressure > 0)
    time = time[mask]; pressure = pressure[mask]
    if len(time) < 2:
        return {'error': 'Insufficient valid data points'}

    # Normalize units -> Torr internally
    if pressure_units == 'mbar':
        pressure_torr = pressure * 0.75006
    elif pressure_units == 'Pa':
        pressure_torr = pressure * 0.00750062
    else:
        # assume Torr
        pressure_torr = pressure

    # Legacy method: baseline or absolute floor
    if method == 'baseline_or_abs':
        baseline = float(np.min(pressure_torr))
        thr_from_baseline = baseline * 10.0
        abs_thr = 5e-10 * 0.75006  # absolute floor in Torr
        dose_thr = min(thr_from_baseline, abs_thr)

        dose_mask = pressure_torr > dose_thr
        if not np.any(dose_mask):
            return {
                'baseline_pressure_torr': float(baseline),
                'peak_pressure_torr': float(np.max(pressure_torr)),
                'dose_threshold_torr': float(dose_thr),
                'dose_detected': False,
                'dose_duration_s': 0.0,
                'exposure_langmuirs': 0.0,
                'integrated_dose_cm2': 0.0,
                'molecular_weight_gmol': float(molecular_weight),
                'threshold_method': 'baseline_or_abs'
            }

        i0 = int(np.where(dose_mask)[0][0])
        i1 = int(np.where(dose_mask)[0][-1])
        dose_time = time[i0:i1+1]
        dose_pressure_torr = pressure_torr[i0:i1+1]

    else:
        # Robust method: thresholds relative to (peak - pre-peak baseline), with hysteresis
        # 1) Peak index/time/value
        peak_idx = int(np.argmax(pressure_torr))
        peak_val = float(pressure_torr[peak_idx])

        # 2) Pre-peak baseline: median of the earlier portion (avoid ramp influence)
        n = len(pressure_torr)
        pre_len = max(min(int(0.1 * n), peak_idx), 30)  # first 10% or up to peak, at least 30 samples if possible
        if pre_len >= 5:
            baseline_pre = float(np.median(pressure_torr[:pre_len]))
        else:
            baseline_pre = float(np.min(pressure_torr))

        dyn = max(peak_val - baseline_pre, 0.0)
        if dyn <= 0:
            return {
                'baseline_pressure_torr': float(baseline_pre),
                'peak_pressure_torr': float(peak_val),
                'dose_detected': False,
                'dose_duration_s': 0.0,
                'exposure_langmuirs': 0.0,
                'integrated_dose_cm2': 0.0,
                'molecular_weight_gmol': float(molecular_weight),
                'threshold_method': 'relative_to_peak',
                'note': 'No dynamic range between baseline and peak'
            }

        entry_thr = baseline_pre + entry_frac * dyn
        exit_thr  = baseline_pre + exit_frac  * dyn

        above_exit = pressure_torr >= exit_thr

        # 3) Find the contiguous segment above exit_thr that CONTAINS the peak (robust to ramps/noise)
        padded = np.concatenate(([False], above_exit, [False]))
        diffs = np.diff(padded.astype(int))
        starts = np.where(diffs == 1)[0]
        ends   = np.where(diffs == -1)[0]

        seg_with_peak = None
        for s, e in zip(starts, ends):
            if s <= peak_idx < e:
                seg_with_peak = (s, e)  # indices in 'time' space
                break

        if seg_with_peak is None:
            # If nothing contains the peak, no robust dose plateau found
            return {
                'baseline_pressure_torr': float(baseline_pre),
                'peak_pressure_torr': float(peak_val),
                'dose_detected': False,
                'dose_duration_s': 0.0,
                'exposure_langmuirs': 0.0,
                'integrated_dose_cm2': 0.0,
                'molecular_weight_gmol': float(molecular_weight),
                'threshold_method': 'relative_to_peak',
                'note': 'No contiguous segment above exit threshold contains the peak'
            }

        i_start_exit, i_end_exit = seg_with_peak

        # 4) Enforce minimum duration on plateau segment
        if (time[i_end_exit-1] - time[i_start_exit]) < float(min_duration_s):
            # Short blip; treat as not a valid dose
            return {
                'baseline_pressure_torr': float(baseline_pre),
                'peak_pressure_torr': float(peak_val),
                'dose_detected': False,
                'dose_duration_s': 0.0,
                'exposure_langmuirs': 0.0,
                'integrated_dose_cm2': 0.0,
                'molecular_weight_gmol': float(molecular_weight),
                'threshold_method': 'relative_to_peak',
                'note': f'Segment above exit threshold shorter than {min_duration_s:.1f} s'
            }

        # 5) Use the exit-threshold segment as the dose window (hysteresis ensures clean edges)
        i0, i1 = i_start_exit, i_end_exit - 1  # 'ends' are exclusive index
        dose_time = time[i0:i1+1]
        dose_pressure_torr = pressure_torr[i0:i1+1]

    # Common metrics using the chosen window (legacy or robust)
    dose_start, dose_end = float(dose_time[0]), float(dose_time[-1])
    dose_duration = float(dose_end - dose_start)

    exposure_torr_s = float(np.trapz(dose_pressure_torr, dose_time))
    exposure_L = float(exposure_torr_s * 1e6)

    sqrt_MT = float(np.sqrt(molecular_weight * temperature_K))
    flux = 3.513e22 * dose_pressure_torr / sqrt_MT
    integrated = float(np.trapz(flux, dose_time))

    return {
        'baseline_pressure_torr': float(np.min(pressure_torr)),
        'peak_pressure_torr': float(np.max(pressure_torr)),
        'dose_detected': True,
        'dose_start_time_s': dose_start,
        'dose_end_time_s': dose_end,
        'dose_duration_s': dose_duration,
        'mean_dose_pressure_torr': float(np.mean(dose_pressure_torr)),
        'exposure_langmuirs': exposure_L,
        'molecular_weight_gmol': float(molecular_weight),
        'temperature_K': float(temperature_K),
        'integrated_dose_cm2': integrated,
        'mean_flux_cm2s': float(np.mean(flux)),
        'dose_threshold_torr': float(locals().get('exit_thr', np.nan)),  # for reference
        'threshold_method': method
    }


# ==================== FLASH ANALYSIS ====================

def analyze_flash(df: pd.DataFrame,
                  temp_col: str = 'Pyro_T',
                  time_col: str = 'Time_s',
                  flash_threshold_C: float = 1000.0,
                  exit_threshold_C: float = 1000.0) -> Dict[str, Any]:
    """
    Analyze flash step: count flashes, measure duration and peak temperature
    """
    if temp_col not in df.columns or time_col not in df.columns:
        return {'error': f'Missing columns: {temp_col} or {time_col}'}

    time = df[time_col].to_numpy(dtype=float)
    temp = df[temp_col].to_numpy(dtype=float)
    mask = np.isfinite(time) & np.isfinite(temp)
    time = time[mask]; temp = temp[mask]
    if len(time) < 2:
        return {'error': 'Insufficient valid data points'}

    in_flash = False
    flash_events = []
    cur = None

    for t, T in zip(time, temp):
        if not in_flash and T > flash_threshold_C:
            in_flash = True
            cur = {'start_time_s': float(t), 'start_temp_C': float(T),
                   'peak_temp_C': float(T), 'peak_time_s': float(t)}
        elif in_flash:
            if T > cur['peak_temp_C']:
                cur['peak_temp_C'] = float(T); cur['peak_time_s'] = float(t)
            if T < exit_threshold_C:
                cur['end_time_s'] = float(t); cur['end_temp_C'] = float(T)
                cur['duration_s'] = cur['end_time_s'] - cur['start_time_s']
                flash_events.append(cur); in_flash = False; cur = None

    if in_flash and cur is not None:
        cur['end_time_s'] = float(time[-1]); cur['end_temp_C'] = float(temp[-1])
        cur['duration_s'] = cur['end_time_s'] - cur['start_time_s']
        flash_events.append(cur)

    total_flash_time = sum(f['duration_s'] for f in flash_events)
    peak_temp = max((f['peak_temp_C'] for f in flash_events), default=0.0)
    return {'flash_count': len(flash_events),
            'total_flash_time_s': float(total_flash_time),
            'peak_temperature_C': float(peak_temp),
            'flashes': flash_events}


# ==================== OVERGROWTH ANALYSIS (CUSTOM RULES) ====================

def _median_or_none(vals: np.ndarray) -> Optional[float]:
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return None
    return float(np.median(vals))


def _estimate_phase_temperature(current_vals: np.ndarray,
                                voltage_vals: Optional[np.ndarray],
                                calibration: Dict[str, Any],
                                resistance_vals: Optional[np.ndarray] = None,
                                voltage_label: Optional[str] = None,
                                resistance_label: Optional[str] = None) -> Tuple[Optional[float], Dict[str, Any]]:
    """
    Estimate median temperature (°C) for a phase using calibration with robust selection.
    Returns (temperature_C, info_dict), where info_dict may include:
      - temperature_source: description of which source was used (resistance column or V/I)
      - calibration_note: any note/warning about range or inputs
    Supports:
      - model == 'T(I)_polynomial' (dict coefficients {a0,a1,...}) returns K -> °C
      - model == 'polynomial' (list-like coeffs) °C directly
      - model == 'R(T)_exponential' with a,b,c
        Tries resistance_vals (if provided), else V/I using absolute values.
    """
    info: Dict[str, Any] = {}
    if calibration is None:
        return None, {'calibration_note': 'No calibration provided'}

    model = str(calibration.get('model', '')).strip()

    # Clean current
    I_all = np.asarray(current_vals, dtype=float)
    I_all = I_all[np.isfinite(I_all)]
    if I_all.size == 0:
        return None, {'calibration_note': 'No valid current samples in phase'}

    # Optional temp range for scoring candidate choices
    tmin_K = calibration.get('calibration_range', {}).get('temp_min', None)
    tmax_K = calibration.get('calibration_range', {}).get('temp_max', None)
    tmin_C = (tmin_K - 273.15) if isinstance(tmin_K, (int, float)) else None
    tmax_C = (tmax_K - 273.15) if isinstance(tmax_K, (int, float)) else None

    # 1) T(I)_polynomial -> Kelvin then to °C and clamp
    if model == 'T(I)_polynomial':
        coeffs_dict = calibration.get('coefficients', {})
        if not isinstance(coeffs_dict, dict) or len(coeffs_dict) == 0:
            return None, {'calibration_note': 'Calibration coefficients missing for T(I)_polynomial'}
        keys_sorted = sorted(
            coeffs_dict.keys(),
            key=lambda k: int(k[1:]) if isinstance(k, str) and k.startswith('a') and k[1:].isdigit() else 0
        )
        coeffs = [coeffs_dict[k] for k in keys_sorted]
        poly = np.poly1d(coeffs[::-1])
        T_K = poly(I_all)
        T_C = float(np.median(T_K) - 273.15)
        T_C = max(0.0, T_C)
        info['temperature_source'] = 'T(I)_polynomial @ median(I)'
        return T_C, info

    # 2) polynomial: °C directly
    if model == 'polynomial':
        coeffs = calibration.get('coefficients') or calibration.get('fit_coefficients')
        if not coeffs:
            return None, {'calibration_note': 'Calibration coefficients missing for polynomial'}
        T_C = float(np.polyval(coeffs, float(np.median(I_all))))
        T_C = max(0.0, T_C)
        info['temperature_source'] = 'polynomial(I) @ median(I)'
        return T_C, info

    # 3) R(T)_exponential: try resistance column first, else V/I (abs)
    if model == 'R(T)_exponential':
        a = calibration.get('coefficients', {}).get('a', None)
        b = calibration.get('coefficients', {}).get('b', None)
        c = calibration.get('coefficients', {}).get('c', None)
        if not all(isinstance(x, (int, float)) for x in [a, b, c]) or b == 0:
            return None, {'calibration_note': 'Invalid a/b/c for R(T)_exponential'}

        def temp_from_R(R_arr: np.ndarray) -> Optional[float]:
            # Domain: (R - c)/a > 0
            X = (R_arr - c) / a
            valid = np.isfinite(X) & (X > 0)
            if not np.any(valid):
                return None
            T_K = (1.0 / b) * np.log(X[valid])
            T_C_vals = T_K - 273.15
            T_C_vals = T_C_vals[np.isfinite(T_C_vals)]
            if T_C_vals.size == 0:
                return None
            T_C_vals = np.clip(T_C_vals, 0.0, None)  # no negative temperatures
            return float(np.median(T_C_vals))

        # Build candidate list [(label, R_array)]
        candidates: List[Tuple[str, np.ndarray]] = []

        # Candidate 1: Provided resistance array
        if resistance_vals is not None:
            Rv = np.asarray(resistance_vals, dtype=float)
            Rv = Rv[np.isfinite(Rv)]
            if Rv.size > 0:
                candidates.append((resistance_label or 'ResistanceColumn', Rv))

        # Candidate 2+: V/I for each voltage candidate provided by caller
        if voltage_vals is not None:
            Vv = np.asarray(voltage_vals, dtype=float)
            # Compute using absolute values to avoid sign artifacts
            I_abs = np.abs(I_all)
            V_abs = np.abs(Vv[np.isfinite(Vv)])
            # Align lengths (caller should pass the segment-sliced arrays)
            n = min(I_abs.size, V_abs.size)
            if n > 0:
                R_vi = V_abs[:n] / I_abs[:n]
                R_vi = R_vi[np.isfinite(R_vi)]
                if R_vi.size > 0:
                    candidates.append((voltage_label or 'V_over_I', R_vi))

        # Score and pick best candidate
        best: Tuple[str, float] = ("", float('inf'))  # (label, score)
        best_T: Optional[float] = None
        best_label: Optional[str] = None

        def score_temp(Tc: float) -> float:
            # 0 if within [tmin_C, tmax_C], else distance to nearest boundary
            if not isinstance(Tc, (int, float)):
                return float('inf')
            if (tmin_C is None) or (tmax_C is None):
                # if no range provided, prefer higher temps over near-zero
                return abs(Tc - 300.0)
            if tmin_C <= Tc <= tmax_C:
                return 0.0
            # distance to boundary
            if Tc < tmin_C:
                return float(tmin_C - Tc)
            return float(Tc - tmax_C)

        for label, R_arr in candidates:
            Tc = temp_from_R(R_arr)
            if Tc is None:
                continue
            s = score_temp(Tc)
            if s < best[1]:
                best = (label, s)
                best_T = Tc
                best_label = label

        if best_T is not None:
            info['temperature_source'] = f"R(T) via {best_label}"
            return best_T, info

        # If nothing worked, explain why
        note = "Unable to compute temperature: no valid resistance candidate in domain"
        if (resistance_vals is None) and (voltage_vals is None):
            note = "R(T) calibration provided, but neither resistance nor voltage data available"
        return None, {'calibration_note': note}

    # Unknown model
    return None, {'calibration_note': f"Unknown calibration model: {model}"}


def analyze_overgrowth(df: pd.DataFrame,
                       current_col: str = 'TDK_I',
                       time_col: str = 'Time_s',
                       calibration: Optional[Dict[str, Any]] = None,
                       locking_layer_rate_ML_min: Optional[float] = None,
                       lte_rate_ML_min: Optional[float] = None,
                       thresholds: Optional[Dict[str, float]] = None,
                       auto_detect_thresholds: bool = False) -> Dict[str, Any]:
    """
    Overgrowth analysis using explicit rules:
      - RT (locking layer): current ≤ 0.02 A
      - RTA: current > 0.10 A (spike)
      - LTE: 0.02 A < current ≤ 0.10 A (first segment after RTA)
      - RT is exactly 10 ML ending at the RTA spike start (requires locking_layer_rate_ML_min)
    Fallback to quantile thresholds if RTA/LTE are not found.
    """
    if current_col not in df.columns or time_col not in df.columns:
        return {'error': f'Missing columns: {current_col} or {time_col}'}

    time = df[time_col].to_numpy(dtype=float)
    current = df[current_col].to_numpy(dtype=float)
    mask = np.isfinite(time) & np.isfinite(current)
    time = time[mask]; current = current[mask]
    if len(time) < 2:
        return {'error': 'Insufficient valid data points'}

    # Quantiles for diagnostics/fallback
    q33 = float(np.percentile(current, 33))
    q66 = float(np.percentile(current, 66))

    # Voltage (for R(T) calibration) - use the first available candidate
    voltage_cols = _find_voltage_columns(df)
    voltage_col = voltage_cols[0] if voltage_cols else None
    voltage = df[voltage_col].to_numpy(dtype=float)[mask] if voltage_col else None

    # Default thresholds (your rules)
    RT_MAX = 0.02
    RTA_MIN = 0.10
    LTE_MIN = RT_MAX
    LTE_MAX = RTA_MIN
    thresholds_source = 'default'

    # 1) Detect RTA spike (first contiguous run > 0.10A)
    above_rta = current > RTA_MIN
    rta_detected = bool(np.any(above_rta))
    rta_metrics = {'detected': False}
    i_rta_start = None
    i_rta_end = None
    if rta_detected:
        idxs = np.where(above_rta)[0]
        i_rta_start = int(idxs[0])
        # extend to end of contiguous block
        end = i_rta_start
        while end + 1 < len(current) and current[end + 1] > RTA_MIN:
            end += 1
        i_rta_end = end
        t_start = time[i_rta_start]; t_end = time[i_rta_end]
        cur_vals = current[i_rta_start:i_rta_end+1]
        rta_metrics = {
            'detected': True,
            'start_time_s': float(t_start),
            'end_time_s': float(t_end),
            'duration_s': float(t_end - t_start),
            'median_current_A': float(np.median(cur_vals)),
            'mean_current_A': float(np.mean(cur_vals)),
            'max_current_A': float(np.max(cur_vals))
        }
    else:
        thresholds_source = 'fallback_auto'

    # 2) LTE: first contiguous segment after RTA where 0.02 < I ≤ 0.10
    lte_metrics = {'detected': False}
    if rta_detected and (i_rta_end is not None) and (i_rta_end + 1 < len(current)):
        after_I = current[i_rta_end+1:]
        after_t = time[i_rta_end+1:]
        lte_mask = (after_I > LTE_MIN) & (after_I <= LTE_MAX)
        if np.any(lte_mask):
            idxs = np.where(lte_mask)[0]
            start_rel = int(idxs[0])
            end_rel = start_rel
            while end_rel + 1 < len(lte_mask) and lte_mask[end_rel + 1]:
                end_rel += 1
            i_lte_start = i_rta_end + 1 + start_rel
            i_lte_end = i_rta_end + 1 + end_rel
            t_start = time[i_lte_start]; t_end = time[i_lte_end]
            cur_vals = current[i_lte_start:i_lte_end+1]
            lte_metrics = {
                'detected': True,
                'start_time_s': float(t_start),
                'end_time_s': float(t_end),
                'duration_s': float(t_end - t_start),
                'median_current_A': float(np.median(cur_vals)),
                'mean_current_A': float(np.mean(cur_vals)),
                'number_of_segments': 1
            }
            # LTE deposition if rate provided
            if lte_rate_ML_min is not None:
                dur_min = (t_end - t_start) / 60.0
                ML = lte_rate_ML_min * dur_min
                nm = ML * 0.136
                lte_metrics['deposited_ML'] = float(ML)
                lte_metrics['deposited_nm'] = float(nm)
                lte_metrics['deposition_rate_ML_min'] = float(lte_rate_ML_min)

    # 3) RT: exactly 10 ML before RTA spike start (requires locking rate)
    rt_metrics = {'detected': False}
    if rta_detected and locking_layer_rate_ML_min is not None:
        t10_min = 10.0 / float(locking_layer_rate_ML_min)
        t10_s = t10_min * 60.0
        rt_end_t = time[i_rta_start]
        rt_start_t = rt_end_t - t10_s
        # Collect current samples within [rt_start_t, rt_end_t], regardless of their exact I
        within = (time >= rt_start_t) & (time <= rt_end_t)
        cur_vals = current[within]
        rt_metrics = {
            'detected': True,
            'estimated': True,  # per spec: fixed 10 ML window by definition
            'start_time_s': float(rt_start_t),
            'end_time_s': float(rt_end_t),
            'duration_s': float(t10_s),
            'median_current_A': float(np.median(cur_vals)) if len(cur_vals) > 0 else None,
            'mean_current_A': float(np.mean(cur_vals)) if len(cur_vals) > 0 else None,
            'deposited_ML': 10.0,
            'deposited_nm': float(10.0 * 0.136),
            'deposition_rate_ML_min': float(locking_layer_rate_ML_min)
        }
    elif rta_detected and locking_layer_rate_ML_min is None:
        # No rate -> fall back to natural low-current region before spike (best effort)
        pre_mask = time < time[i_rta_start]
        pre_time = time[pre_mask]; pre_I = current[pre_mask]
        if len(pre_time) > 1:
            low_mask = pre_I <= RT_MAX
            segs = _segments_from_mask(pre_time, low_mask)
            if segs:
                longest = max(segs, key=lambda s: s['duration_s'])
                vals = pre_I[(pre_time >= longest['start_time_s']) & (pre_time <= longest['end_time_s'])]
                rt_metrics = {
                    'detected': True,
                    'start_time_s': float(longest['start_time_s']),
                    'end_time_s': float(longest['end_time_s']),
                    'duration_s': float(longest['duration_s']),
                    'median_current_A': float(np.median(vals)) if len(vals) > 0 else None,
                    'mean_current_A': float(np.mean(vals)) if len(vals) > 0 else None
                }

    # 4) Fallback (quantile) if RTA or LTE not detected
    fallback_used = False
    if not rta_metrics.get('detected') or not lte_metrics.get('detected'):
        # Use quantile-based thresholds as a backup
        auto_RT = 0.8 * q33
        auto_RTA = q66
        auto_LTE_MIN, auto_LTE_MAX = auto_RT, auto_RTA
        # RTA
        above_auto = current >= auto_RTA
        if np.any(above_auto) and not rta_metrics.get('detected'):
            fallback_used = True
            thresholds_source = 'fallback_auto'
            idxs = np.where(above_auto)[0]
            i_rta_start = int(idxs[0])
            end = i_rta_start
            while end + 1 < len(current) and current[end + 1] >= auto_RTA:
                end += 1
            i_rta_end = end
            t_start = time[i_rta_start]; t_end = time[i_rta_end]
            cur_vals = current[i_rta_start:i_rta_end+1]
            rta_metrics = {
                'detected': True,
                'start_time_s': float(t_start),
                'end_time_s': float(t_end),
                'duration_s': float(t_end - t_start),
                'median_current_A': float(np.median(cur_vals)),
                'mean_current_A': float(np.mean(cur_vals)),
                'max_current_A': float(np.max(cur_vals))
            }
        # LTE after updated RTA
        if rta_metrics.get('detected') and not lte_metrics.get('detected') and (i_rta_end is not None):
            after_I = current[i_rta_end+1:]
            after_t = time[i_rta_end+1:]
            mask_lte = (after_I > auto_LTE_MIN) & (after_I < auto_LTE_MAX)
            if np.any(mask_lte):
                fallback_used = True
                thresholds_source = 'fallback_auto'
                idxs = np.where(mask_lte)[0]
                start_rel = int(idxs[0])
                end_rel = start_rel
                while end_rel + 1 < len(mask_lte) and mask_lte[end_rel + 1]:
                    end_rel += 1
                i_lte_start = i_rta_end + 1 + start_rel
                i_lte_end = i_rta_end + 1 + end_rel
                t_start = time[i_lte_start]; t_end = time[i_lte_end]
                cur_vals = current[i_lte_start:i_lte_end+1]
                lte_metrics = {
                    'detected': True,
                    'start_time_s': float(t_start),
                    'end_time_s': float(t_end),
                    'duration_s': float(t_end - t_start),
                    'median_current_A': float(np.median(cur_vals)),
                    'mean_current_A': float(np.mean(cur_vals)),
                    'number_of_segments': 1
                }
                if lte_rate_ML_min is not None:
                    dur_min = (t_end - t_start) / 60.0
                    ML = lte_rate_ML_min * dur_min
                    nm = ML * 0.136
                    lte_metrics['deposited_ML'] = float(ML)
                    lte_metrics['deposited_nm'] = float(nm)
                    lte_metrics['deposition_rate_ML_min'] = float(lte_rate_ML_min)

        # RT via fallback: if still no RT and rate provided, keep fixed 10 ML window before RTA
        if rta_metrics.get('detected') and not rt_metrics.get('detected') and locking_layer_rate_ML_min is not None:
            t10_min = 10.0 / float(locking_layer_rate_ML_min)
            t10_s = t10_min * 60.0
            rt_end_t = rta_metrics['start_time_s']
            rt_start_t = rt_end_t - t10_s
            within = (time >= rt_start_t) & (time <= rt_end_t)
            cur_vals = current[within]
            rt_metrics = {
                'detected': True,
                'estimated': True,
                'start_time_s': float(rt_start_t),
                'end_time_s': float(rt_end_t),
                'duration_s': float(t10_s),
                'median_current_A': float(np.median(cur_vals)) if len(cur_vals) > 0 else None,
                'mean_current_A': float(np.mean(cur_vals)) if len(cur_vals) > 0 else None,
                'deposited_ML': 10.0,
                'deposited_nm': float(10.0 * 0.136),
                'deposition_rate_ML_min': float(locking_layer_rate_ML_min)
            }

    # 5) Temperatures: use calibration if provided (smart source selection for R(T))
    if calibration is not None:
        # Prepare candidate columns once
        res_cols = _find_resistance_columns(df)
        volt_cols = _find_voltage_columns(df)

        def phase_arrays(start_s: float, end_s: float):
            sel = (time >= start_s) & (time <= end_s)
            I_seg = current[sel]
            # Build resistance candidate from the first available resistance column (segment)
            R_seg = None
            R_label = None
            for rc in res_cols:
                R_all = df[rc].to_numpy(dtype=float)[mask]  # align to mask used earlier
                R_seg_candidate = R_all[sel]
                if np.any(np.isfinite(R_seg_candidate)):
                    R_seg = R_seg_candidate
                    R_label = rc
                    break
            # If no direct resistance, we will try voltage candidates later
            return I_seg, R_seg, R_label, sel

        # RT
        if rt_metrics.get('detected'):
            I_seg, R_seg, R_label, sel = phase_arrays(rt_metrics['start_time_s'], rt_metrics['end_time_s'])

            # Try direct resistance first
            T_C, info = _estimate_phase_temperature(
                I_seg, None, calibration,
                resistance_vals=R_seg,
                resistance_label=R_label
            )
            # If not obtained and R(T) model, try voltage candidates V/I
            if (T_C is None) and calibration.get('model') == 'R(T)_exponential':
                best_T = None
                best_info = {}
                # Try each voltage candidate for this phase and pick the best
                for vc in volt_cols:
                    V_all = df[vc].to_numpy(dtype=float)[mask]
                    V_seg = V_all[sel]
                    Tc_try, info_try = _estimate_phase_temperature(
                        I_seg, V_seg, calibration,
                        resistance_vals=None,
                        voltage_label=vc
                    )
                    if Tc_try is None:
                        continue
                    # Choose first in-range (or best toward center ~300°C)
                    if best_T is None:
                        best_T, best_info = Tc_try, info_try
                    else:
                        # prefer closer to calibration range center if available
                        best_T, best_info = Tc_try, info_try
                        break
                if best_T is not None:
                    T_C, info = best_T, best_info

            if T_C is not None:
                rt_metrics['median_temperature_C'] = float(T_C)
                rt_metrics['median_temperature_K'] = float(T_C + 273.15)
            rt_metrics.update({k: v for k, v in info.items() if v})

        # RTA
        if rta_metrics.get('detected'):
            I_seg, R_seg, R_label, sel = phase_arrays(rta_metrics['start_time_s'], rta_metrics['end_time_s'])

            T_C, info = _estimate_phase_temperature(
                I_seg, None, calibration,
                resistance_vals=R_seg,
                resistance_label=R_label
            )
            if (T_C is None) and calibration.get('model') == 'R(T)_exponential':
                best_T = None
                best_info = {}
                for vc in volt_cols:
                    V_all = df[vc].to_numpy(dtype=float)[mask]
                    V_seg = V_all[sel]
                    Tc_try, info_try = _estimate_phase_temperature(
                        I_seg, V_seg, calibration,
                        resistance_vals=None,
                        voltage_label=vc
                    )
                    if Tc_try is None:
                        continue
                    if best_T is None:
                        best_T, best_info = Tc_try, info_try
                    else:
                        best_T, best_info = Tc_try, info_try
                        break
                if best_T is not None:
                    T_C, info = best_T, best_info

            if T_C is not None:
                rta_metrics['median_temperature_C'] = float(T_C)
                rta_metrics['median_temperature_K'] = float(T_C + 273.15)
            rta_metrics.update({k: v for k, v in info.items() if v})

        # LTE
        if lte_metrics.get('detected'):
            I_seg, R_seg, R_label, sel = phase_arrays(lte_metrics['start_time_s'], lte_metrics['end_time_s'])

            T_C, info = _estimate_phase_temperature(
                I_seg, None, calibration,
                resistance_vals=R_seg,
                resistance_label=R_label
            )
            if (T_C is None) and calibration.get('model') == 'R(T)_exponential':
                best_T = None
                best_info = {}
                for vc in volt_cols:
                    V_all = df[vc].to_numpy(dtype=float)[mask]
                    V_seg = V_all[sel]
                    Tc_try, info_try = _estimate_phase_temperature(
                        I_seg, V_seg, calibration,
                        resistance_vals=None,
                        voltage_label=vc
                    )
                    if Tc_try is None:
                        continue
                    if best_T is None:
                        best_T, best_info = Tc_try, info_try
                    else:
                        best_T, best_info = Tc_try, info_try
                        break
                if best_T is not None:
                    T_C, info = best_T, best_info

            if T_C is not None:
                lte_metrics['median_temperature_C'] = float(T_C)
                lte_metrics['median_temperature_K'] = float(T_C + 273.15)
            lte_metrics.update({k: v for k, v in info.items() if v})

    # 6) Totals and result packaging
    total_ML = 0.0; total_nm = 0.0; dep_calc = False
    for p in (rt_metrics, lte_metrics):
        if p.get('detected') and p.get('deposited_ML') is not None:
            total_ML += p['deposited_ML']; total_nm += p['deposited_nm']; dep_calc = True

    # Total duration: from RT_start (10 ML before RTA) to LTE end; fallback to spans if missing
    if rt_metrics.get('detected') and lte_metrics.get('detected'):
        total_duration_s = float(lte_metrics['end_time_s'] - rt_metrics['start_time_s'])
    elif rta_metrics.get('detected') and rt_metrics.get('detected'):
        total_duration_s = float(rta_metrics['end_time_s'] - rt_metrics['start_time_s'])
    else:
        total_duration_s = float(time[-1] - time[0])

    result = {
        'phase_thresholds': {'RT_max_A': RT_MAX, 'RTA_min_A': RTA_MIN, 'LTE_min_A': LTE_MIN, 'LTE_max_A': LTE_MAX},
        'phase_thresholds_source': thresholds_source,
        'thresholds_auto_fallback': bool(fallback_used),
        'current_q33': q33,
        'current_q66': q66,
        'RT_growth': rt_metrics,
        'RTA_anneal': rta_metrics,
        'LTE_growth': lte_metrics,
        'total_duration_s': float(total_duration_s),
        'total_duration_min': float(total_duration_s/60.0),
        'total_duration_formatted': format_time(total_duration_s),
        'total_deposited_ML': float(total_ML) if dep_calc else None,
        'total_deposited_nm': float(total_nm) if dep_calc else None
    }
    return result


# ==================== SUSI ANALYSIS ====================

def analyze_susi(df: pd.DataFrame,
                 current_col: str = 'TDK_I_1',
                 time_col: str = 'Time_s',
                 threshold_A: Optional[float] = None) -> Dict[str, Any]:
    """
    SUSI operation metrics on the SUSI current (not sample current).
    """
    if current_col not in df.columns:
        detected = _find_susi_current_column(df) or _find_current_column(df)
        if detected:
            current_col = detected

    if current_col not in df.columns or time_col not in df.columns:
        return {'error': f'Missing columns: {current_col} or {time_col}'}

    time = df[time_col].to_numpy(dtype=float)
    current = df[current_col].to_numpy(dtype=float)
    mask = np.isfinite(time) & np.isfinite(current)
    time = time[mask]; current = current[mask]
    if len(time) < 2:
        return {'error': 'Insufficient valid data points'}

    if threshold_A is None:
        top_decile = np.percentile(current, 90)
        top_currents = current[current > top_decile]
        threshold_A = 0.8 * (np.median(top_currents) if len(top_currents) > 0 else np.max(current))

    operating_mask = current > threshold_A
    if not np.any(operating_mask):
        return {
            'operating_threshold_A': float(threshold_A),
            'operating_time_detected': False,
            'total_operating_time_s': 0.0,
            'mean_operating_current_A': 0.0
        }

    padded = np.concatenate(([False], operating_mask, [False]))
    diffs = np.diff(padded.astype(int))
    starts = np.where(diffs == 1)[0]; ends = np.where(diffs == -1)[0]

    segments = []
    for s, e in zip(starts, ends):
        seg_t = time[s:e]; seg_i = current[s:e]
        dur = seg_t[-1] - seg_t[0] if len(seg_t) > 1 else 0.0
        segments.append({
            'start_time_s': float(seg_t[0]),
            'end_time_s': float(seg_t[-1]),
            'duration_s': float(dur),
            'mean_current_A': float(np.mean(seg_i))
        })

    operating_current = current[operating_mask]
    total_operating_time = sum(s['duration_s'] for s in segments)

    return {
        'operating_threshold_A': float(threshold_A),
        'operating_time_detected': True,
        'total_operating_time_s': float(total_operating_time),
        'mean_operating_current_A': float(np.mean(operating_current)),
        'peak_current_A': float(np.max(operating_current)),
        'number_of_segments': len(segments),
        'segments': segments
    }


# ==================== OUTGAS ANALYSIS ====================

def analyze_outgas(df: pd.DataFrame,
                   pressure_col: Optional[str] = None,
                   time_col: str = 'Time_s',
                   pressure_units: str = 'mbar',
                   stabilization_window_s: float = 300.0,
                   stabilization_threshold: float = 0.05,
                   return_to_baseline_factor: float = 1.2,
                   slope_fraction_threshold: float = 0.05) -> Dict[str, Any]:
    """
    Outgas/degas analysis: base/peak pressure, time to stabilize.
    Improvements:
      - Stabilization is sought only AFTER the peak pressure.
      - Stabilization requires:
          (a) low variation (std/mean < stabilization_threshold),
          (b) window mean near the final baseline target (<= baseline_target * factor),
          (c) flat window (fractional change across window <= slope_fraction_threshold).
      - Baseline target: robust post-peak 'tail' median when possible, else global min.
      - Returns time-to-stabilize from start (compat) and from peak (new).
    """
    if time_col not in df.columns:
        return {'error': f'Missing column: {time_col}'}

    chosen_col = pressure_col
    if chosen_col is None or chosen_col not in df.columns:
        chosen_col = _find_pressure_column(df, prefer='MBE')
        if chosen_col is None:
            return {'error': 'No suitable pressure column found (expected P_MBE or P_VT)'}

    preferred_mbe = _find_pressure_column(df, prefer='MBE')
    warning = None
    if preferred_mbe and chosen_col != preferred_mbe:
        warning = f"Preferred MBE gauge not found. Using '{chosen_col}' instead."

    time = df[time_col].to_numpy(dtype=float)
    pressure_raw = df[chosen_col].to_numpy(dtype=float)
    mask = np.isfinite(time) & np.isfinite(pressure_raw) & (pressure_raw > 0)
    time = time[mask]; pressure_raw = pressure_raw[mask]
    if len(time) < 2:
        return {'error': 'Insufficient valid data points'}

    # Convert to Torr internally for consistent metrics
    if pressure_units == 'mbar':
        pressure_torr = pressure_raw * 0.75006
    elif pressure_units == 'Pa':
        pressure_torr = pressure_raw * 0.00750062
    else:
        pressure_torr = pressure_raw

    # Basic metrics
    peak_idx = int(np.argmax(pressure_torr))
    peak_torr = float(pressure_torr[peak_idx])
    peak_time = float(time[peak_idx])

    base_global_torr = float(np.min(pressure_torr))  # global min (robust-ish if long)
    final_torr = float(pressure_torr[-1])

    # Build a robust "tail baseline" from the last chunk of data after the peak
    # Use the last max(300 s, 10% of series) window (or as much as available)
    dt = float(np.median(np.diff(time))) if len(time) > 1 else 0.0
    tail_seconds = max(300.0, 0.10 * (time[-1] - time[0]))  # tunable
    tail_size = int(round(tail_seconds / dt)) if dt > 0 else 0
    tail_size = max(tail_size, 30)  # make it at least some samples if dt small
    if tail_size >= len(pressure_torr):
        tail_size = max(5, len(pressure_torr) // 3)

    tail_start = max(len(pressure_torr) - tail_size, peak_idx + 1)
    if tail_start < len(pressure_torr):
        tail_baseline_torr = float(np.median(pressure_torr[tail_start:]))
    else:
        # If no room for tail, fall back to global min
        tail_baseline_torr = base_global_torr

    # Choose a conservative baseline target (prefer actual post-peak floor):
    baseline_target_torr = min(tail_baseline_torr, base_global_torr)

    # Sliding window AFTER the peak to detect stabilization
    stabilized = False
    stabilized_time_abs = None
    stabilized_time_from_peak = None

    window_size = int(round(stabilization_window_s / dt)) if dt > 0 else 0
    if window_size < 5:
        window_size = 5

    start_idx = max(peak_idx + 1, window_size)  # ensure full window and after-peak

    for i in range(start_idx, len(pressure_torr)):
        win_p = pressure_torr[i - window_size:i]
        win_t = time[i - window_size:i]
        if win_p.size < 5:
            continue

        mean_p = float(np.mean(win_p))
        std_p = float(np.std(win_p))
        cov_ok = (mean_p > 0) and ((std_p / mean_p) < stabilization_threshold)

        # Require near-baseline (within factor)
        near_baseline_ok = mean_p <= (baseline_target_torr * return_to_baseline_factor)

        # Flatness: fractional change across window
        # slope [Torr/s] from linear fit (robust enough for this use)
        try:
            slope, intercept = np.polyfit(win_t, win_p, 1)
            frac_change = abs(slope) * (win_t[-1] - win_t[0]) / max(mean_p, 1e-99)
        except Exception:
            frac_change = 0.0
        flat_ok = frac_change <= slope_fraction_threshold

        # Optional: also ensure the last point itself is near baseline
        last_ok = win_p[-1] <= (baseline_target_torr * return_to_baseline_factor)

        if cov_ok and near_baseline_ok and flat_ok and last_ok:
            stabilized = True
            stabilized_time_abs = float(time[i])
            stabilized_time_from_peak = float(time[i] - peak_time)
            break


    t0 = float(time[0])

    result = {
        'pressure_column_used': chosen_col,
        'base_pressure_torr': base_global_torr,
        'base_pressure_mbar': float(base_global_torr / 0.75006),
        'peak_pressure_torr': peak_torr,
        'peak_pressure_mbar': float(peak_torr / 0.75006),
        'final_pressure_torr': final_torr,
        'final_pressure_mbar': float(final_torr / 0.75006),
        'total_duration_s': float(time[-1] - time[0]),
        'pressure_drop_factor': float(peak_torr / base_global_torr) if base_global_torr > 0 else 0.0,
        'stabilization_detected': stabilized,
        # Absolute (from start) for backward-compatibility with your UI:
        'time_to_stabilize_s': float(stabilized_time_abs - t0) if stabilized else None,
        # New: time after peak
        'time_to_stabilize_from_peak_s': stabilized_time_from_peak if stabilized else None,
        # References used
        'baseline_reference_torr': float(baseline_target_torr),
        'peak_time_s': peak_time,
    }

    # NEW: formatted HH:MM:SS strings
    result['peak_time_hms'] = format_time(peak_time - t0)
    if stabilized:
        result['time_to_stabilize_hms'] = format_time(result['time_to_stabilize_s'])
        if result.get('time_to_stabilize_from_peak_s') is not None:
            result['time_to_stabilize_from_peak_hms'] = format_time(result['time_to_stabilize_from_peak_s'])
    else:
        result['time_to_stabilize_hms'] = None
        result['time_to_stabilize_from_peak_hms'] = None

    if not stabilized:
        result['stabilization_note'] = 'Pressure did not stabilize within measurement'
    if warning:
        result['pressure_column_warning'] = warning

    return result




# ==================== TERMINATION ANALYSIS ====================
def _infer_pressure_units_from_name(colname: str) -> str:
    name = colname.lower()
    if 'torr' in name:
        return 'Torr'
    if 'mbar' in name:
        return 'mbar'
    if 'pa' in name:
        return 'Pa'
    return 'mbar'  # default used across the codebase


def analyze_termination(df: pd.DataFrame,
                        pressure_col: Optional[str] = None,
                        time_col: str = 'Time_s',
                        molecular_weight: Optional[float] = None,
                        filename: str = '',
                        temperature_K: float = 300.0,
                        pressure_units: str = 'mbar') -> Dict[str, Any]:
    """
    H-termination analysis: prefers P_MBE gauge; extends dose metrics and adds mbar fields.
    Uses robust dose detection relative to peak to avoid ramp/shoulder mis-detection.
    """
    if molecular_weight is None:
        molecular_weight = 2.016  # H2

    chosen_col = pressure_col
    if chosen_col is None or chosen_col not in df.columns:
        chosen_col = _find_pressure_column(df, prefer='MBE')
        if chosen_col is None:
            return {'error': 'No suitable pressure column found (expected P_MBE or P_VT)'}

    # Infer units from column name if available
    inferred_units = _infer_pressure_units_from_name(chosen_col)

    dose = analyze_dose(
        df, chosen_col, time_col, molecular_weight, filename, temperature_K,
        pressure_units=inferred_units,
        method='relative_to_peak',  # <-- robust mode for H2 termination
        entry_frac=0.10,
        exit_frac=0.05,
        min_duration_s=60.0
    )
    dose['pressure_column_used'] = chosen_col

    # Decorate with H-termination-specific fields
    if dose.get('dose_detected'):
        dose['termination_type'] = 'H-termination'
        dose['exposure_time_s'] = dose['dose_duration_s']
        dose['average_pressure_torr'] = dose.get('mean_dose_pressure_torr')

        # Also provide mbar values for UI titles
        if 'peak_pressure_torr' in dose:
            dose['peak_pressure_mbar'] = float(dose['peak_pressure_torr'] / 0.75006)
        if 'baseline_pressure_torr' in dose:
            dose['baseline_pressure_mbar'] = float(dose['baseline_pressure_torr'] / 0.75006)
        if 'mean_dose_pressure_torr' in dose:
            dose['mean_dose_pressure_mbar'] = float(dose['mean_dose_pressure_torr'] / 0.75006)

    return dose

# ==================== MAIN DISPATCH ====================

def analyze_process_file(file_path: str,
                         calibration: Optional[Dict[str, Any]] = None,
                         locking_layer_rate_ML_min: Optional[float] = None,
                         lte_rate_ML_min: Optional[float] = None,
                         auto_detect_thresholds: bool = False,
                         thresholds: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """
    Analyze a LabVIEW process file and return comprehensive metrics.
    """
    path = Path(file_path)
    if not path.exists():
        return {'error': f'File not found: {file_path}'}

    try:
        if LabVIEWParser is None:
            raise ImportError("LabVIEWParser not available")
        parser = LabVIEWParser(str(path))
        parsed = parser.parse()
        df = parsed['data']
        file_type = detect_file_type(path.name)
    except Exception as e:
        return {'error': f'Failed to parse file: {str(e)}'}

    # Normalize
    df.columns = [c.strip().replace(' ', '_') for c in df.columns]
    if 'Time' in df.columns and 'Time_s' not in df.columns:
        df.rename(columns={'Time': 'Time_s'}, inplace=True)

    result = {
        'filename': path.name,
        'filepath': str(path.absolute()),
        'file_type': file_type,
        'timestamp': datetime.now().isoformat()
    }

    # Dispatch
    if file_type == 'dose':
        metrics = analyze_dose(df, filename=path.name)
    elif file_type in ('termination', 'hterm'):
        metrics = analyze_termination(df, pressure_col=None, filename=path.name)
    elif file_type == 'flash':
        metrics = analyze_flash(df)
    elif file_type == 'outgas':
        pressure_col = _find_pressure_column(df, prefer='MBE')
        metrics = analyze_outgas(df, pressure_col=pressure_col)
    elif file_type == 'susi':
        if 'Time' in df.columns and 'Time_s' not in df.columns:
            df.rename(columns={'Time': 'Time_s'}, inplace=True)
        cols = set(df.columns)
        susi_current_col = 'TDK_I_1' if 'TDK_I_1' in cols else (_find_susi_current_column(df) or 'TDK_I_1')
        sample_current_col = 'TDK_I' if 'TDK_I' in cols else (_find_sample_current_column(df) or 'TDK_I')

        susi_metrics = analyze_susi(df, current_col=susi_current_col, time_col='Time_s')
        overgrowth_metrics = analyze_overgrowth(
            df,
            current_col=sample_current_col,
            time_col='Time_s',
            calibration=calibration,
            locking_layer_rate_ML_min=locking_layer_rate_ML_min,
            lte_rate_ML_min=lte_rate_ML_min,
            thresholds=thresholds,
            auto_detect_thresholds=auto_detect_thresholds
        )
        # Choose a single representative voltage column (first candidate) for reporting purposes
        _voltage_cols = _find_voltage_columns(df)
        _voltage_col_used = _voltage_cols[0] if _voltage_cols else None

        metrics = {
            'columns_used': {
                'susi_current_col': susi_current_col,
                'sample_current_col': sample_current_col,
                'time_col': 'Time_s',
                'voltage_col': _voltage_col_used
            },
            'rates_used': {
                'locking_layer_rate_ML_min': locking_layer_rate_ML_min,
                'lte_rate_ML_min': lte_rate_ML_min
            },
            'susi': susi_metrics,
            'overgrowth': overgrowth_metrics
        }

        if susi_current_col == sample_current_col:
            metrics['column_warning'] = (
                "SUSI and sample current columns resolved to the same name "
                f"('{susi_current_col}'). Expected SUSI='TDK_I_1', SAMPLE='TDK_I'."
            )
    else:
        metrics = {'info': 'No specialized analysis for this file type'}

    result['metrics'] = metrics
    return result


# ==================== BATCH EXPORT ====================

def export_metrics_for_folder(input_directory: str,
                              output_path: Optional[str] = None,
                              calibration: Optional[Dict[str, Any]] = None,
                              locking_layer_rate_ML_min: Optional[float] = None,
                              lte_rate_ML_min: Optional[float] = None) -> List[Dict[str, Any]]:
    """
    Analyze all LabVIEW .txt files in a folder and optionally save JSON.
    """
    input_dir = Path(input_directory)
    if not input_dir.exists():
        raise ValueError(f"Directory not found: {input_directory}")

    txt_files = list(input_dir.glob("*.txt"))
    if not txt_files:
        raise ValueError(f"No .txt files found in {input_directory}")

    results = []
    for txt_file in txt_files:
        print(f"Analyzing: {txt_file.name}")
        result = analyze_process_file(
            str(txt_file),
            calibration=calibration,
            locking_layer_rate_ML_min=locking_layer_rate_ML_min,
            lte_rate_ML_min=lte_rate_ML_min
        )
        results.append(result)

    if output_path:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved metrics summary to: {output_path}")

    return results


# ==================== CALIBRATION-INTEGRATED ANALYSIS ====================

def analyze_with_active_calibration(file_path: str,
                                    calibration: Optional[Dict[str, Any]] = None,
                                    calibration_db_path: Optional[str] = None,
                                    override_rates: Optional[Tuple[float, float]] = None) -> Dict[str, Any]:
    """
    Analyze overgrowth file using active SUSI calibration (rates) from DB, or override rates.
    """
    try:
        from stm_fab.analysis.susi_calibration import SUSICalibrationManager

        manager = SUSICalibrationManager(calibration_db_path)
        active_cal = manager.get_active_calibration()

        if active_cal is None and override_rates is None:
            print("⚠ Warning: No active SUSI calibration found.")
            return analyze_process_file(file_path, calibration=calibration)

        if override_rates:
            locking_rate, lte_rate = override_rates
            print(f"Using override rates: Locking={locking_rate:.3f}, LTE={lte_rate:.3f} ML/min")
        else:
            locking_rate = active_cal.get_rate_ML_min('locking_layer')
            lte_rate = active_cal.get_rate_ML_min('lte')
            print(f"Using active calibration (ID {active_cal.calibration_id}) from {active_cal.date}:")
            print(f"  Locking Layer: {locking_rate:.3f} ML/min")
            print(f"  LTE:           {lte_rate:.3f} ML/min")

        result = analyze_process_file(
            file_path,
            calibration=calibration,
            locking_layer_rate_ML_min=locking_rate,
            lte_rate_ML_min=lte_rate
        )

        if override_rates is None and active_cal:
            result['susi_calibration_used'] = {
                'calibration_id': active_cal.calibration_id,
                'date': active_cal.date,
                'sample_name': active_cal.sample_name,
                'method': active_cal.method
            }
        return result

    except ImportError:
        print("⚠ Warning: susi_calibration module not available")
        if override_rates:
            locking_rate, lte_rate = override_rates
            return analyze_process_file(
                file_path,
                calibration=calibration,
                locking_layer_rate_ML_min=locking_rate,
                lte_rate_ML_min=lte_rate
            )
        else:
            return analyze_process_file(file_path, calibration=calibration)
