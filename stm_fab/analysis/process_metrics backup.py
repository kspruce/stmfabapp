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

    Args:
        seconds: Time in seconds

    Returns:
        Formatted string HH:MM:SS
    """
    if not np.isfinite(seconds) or seconds < 0:
        return "00:00:00"

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


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

    Returns:
        34.0 for PH3, 77.95 for AsH3, 34.0 as default
    """
    s = filename.lower()
    if 'ash3' in s or 'ash_3' in s or 'arsine' in s:
        return 77.95
    elif 'ph3' in s or 'ph_3' in s or 'phosphine' in s:
        return 34.0
    else:
        # Default to PH3
        return 34.0


# ==================== COLUMN HELPERS ====================

def _find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Return the first matching column name from candidates, else None."""
    cols = {c: c for c in df.columns}
    normalized = {c.lower().replace(' ', '_'): c for c in df.columns}
    for cand in candidates:
        key = cand.lower().replace(' ', '_')
        if cand in cols:
            return cand
        if key in normalized:
            return normalized[key]
    return None


def _find_current_column(df: pd.DataFrame) -> Optional[str]:
    """
    Generic finder for a current-like column (best-effort).
    """
    preferred = [
        'TDK_I',
        'TDK_I_1',
        'TDK_I_0',
        'TDK_I1',
        'TDK_Current',
        'TDKI',
        'TDK_I_2',
        'TDK_I_3'
    ]
    hit = _find_column(df, preferred)
    if hit is not None:
        return hit

    # Heuristic fallback
    for c in df.columns:
        cl = c.lower()
        if 'tdk' in cl and ('_i' in cl or ' i' in cl or cl.endswith('i') or '_current' in cl):
            return c
    return None


def _segments_from_mask(time: np.ndarray, mask: np.ndarray) -> List[Dict[str, float]]:
    """
    Convert a boolean mask into contiguous time segments.
    Returns list of dicts with start/end/duration seconds.
    """
    if len(time) != len(mask) or len(time) == 0:
        return []

    padded = np.concatenate(([False], mask, [False]))
    diffs = np.diff(padded.astype(int))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]

    segments = []
    for s, e in zip(starts, ends):
        seg_t = time[s:e]
        if len(seg_t) == 0:
            continue
        duration = float(seg_t[-1] - seg_t[0]) if len(seg_t) > 1 else 0.0
        segments.append({
            'start_time_s': float(seg_t[0]),
            'end_time_s': float(seg_t[-1]),
            'duration_s': duration
        })
    return segments


def _find_susi_current_column(df: pd.DataFrame) -> Optional[str]:
    """
    Find the SUSI (source) current column.
    Your mapping: SUSI current = 'TDK I 1' -> normalized to 'TDK_I_1'.
    Prefers exact normalized match first, then fallbacks.
    """
    cols = set(df.columns)

    # 1) Prefer exact normalized column
    if 'TDK_I_1' in cols:
        return 'TDK_I_1'

    # 2) If normalization was skipped for some reason, try the spaced variant
    if 'TDK I 1' in cols:
        return 'TDK I 1'

    # 3) Common variants or explicit SUSI names
    preferred = [
        'SUSI_I', 'SUSI_Current', 'SUSI',
        'TDK_I1', 'TDK_CH2_I', 'TDK2_I'
    ]
    hit = _find_column(df, preferred)
    if hit is not None:
        return hit

    # 4) Heuristic: any column with 'susi' and current-ish naming
    for c in df.columns:
        cl = c.lower().replace(' ', '_')
        if 'susi' in cl and ('_i' in cl or cl.endswith('i') or 'current' in cl):
            return c

    # 5) Last resort: generic finder
    return _find_current_column(df)


def _find_sample_current_column(df: pd.DataFrame) -> Optional[str]:
    """
    Find the SAMPLE current column (used for overgrowth phase detection/temperature).
    Your mapping: sample current = 'TDK I' -> normalized to 'TDK_I'.
    Prefers exact normalized match first, then fallbacks.
    """
    cols = set(df.columns)

    # 1) Prefer exact normalized column
    if 'TDK_I' in cols:
        return 'TDK_I'

    # 2) If normalization was skipped, try spaced variant
    if 'TDK I' in cols:
        return 'TDK I'

    # 3) Common variants or explicit sample names
    preferred = [
        'Sample_I', 'Sample_Current', 'I_Sample',
        'TDK_CH1_I', 'TDK0_I'
    ]
    hit = _find_column(df, preferred)
    if hit is not None:
        return hit

    # 4) Heuristic: anything with 'sample' and current-ish
    for c in df.columns:
        cl = c.lower().replace(' ', '_')
        if 'sample' in cl and ('_i' in cl or cl.endswith('i') or 'current' in cl):
            return c

    # 5) Generic fallback
    return _find_current_column(df)

def _find_pressure_column(df: pd.DataFrame, prefer: str = 'VT') -> Optional[str]:
    """
    Find the pressure column, preferring VT (P_VT) over MBE if requested.

    prefer:
      - 'VT': prefer VT gauge columns (P_VT, PVT, P_VT_1, etc.)
      - 'MBE': prefer MBE columns (P_MBE, PMBE, etc.)

    Returns:
      The chosen column name or None if not found.
    """
    cols = set(df.columns)

    vt_candidates = [
        'P_VT', 'PVT', 'P_VT_1', 'P_VT_Torr', 'P_VT_mbar', 'P_VT_Pa'
    ]
    mbe_candidates = [
        'P_MBE', 'PMBE', 'P_MBE_Torr', 'P_MBE_mbar', 'P_MBE_Pa'
    ]

    def first_hit(cands):
        for c in cands:
            if c in cols:
                return c
        # relaxed normalized search
        norm = {c.lower().replace(' ', '_'): c for c in df.columns}
        for c in cands:
            key = c.lower().replace(' ', '_')
            if key in norm:
                return norm[key]
        return None

    if prefer.upper() == 'VT':
        hit = first_hit(vt_candidates)
        if hit:
            return hit
        # fallback to MBE if VT missing
        hit = first_hit(mbe_candidates)
        if hit:
            return hit
    else:
        hit = first_hit(mbe_candidates)
        if hit:
            return hit
        hit = first_hit(vt_candidates)
        if hit:
            return hit

    # very last resort: any column that looks like a pressure
    for c in df.columns:
        cl = c.lower().replace(' ', '_')
        if cl.startswith('p_') or cl == 'p' or 'pressure' in cl:
            return c

    return None


# ==================== DOSE ANALYSIS ====================

def analyze_dose(df: pd.DataFrame,
                 pressure_col: str = 'P_VT',
                 time_col: str = 'Time_s',
                 molecular_weight: Optional[float] = None,
                 filename: str = '',
                 temperature_K: float = 300.0,
                 pressure_units: str = 'mbar') -> Dict[str, Any]:
    """
    Analyze dose step: calculate flux, exposure, integrated molecules

    Molecular flux formula:
    Flux [molecules/cm²/s] ≈ 3.513e22 × P[Torr] / sqrt(M[g/mol] × T[K])
    """
    if pressure_col not in df.columns or time_col not in df.columns:
        return {'error': f'Missing columns: {pressure_col} or {time_col}'}

    # Auto-detect molecular weight if not provided
    if molecular_weight is None:
        molecular_weight = detect_molecular_weight(filename)

    time = df[time_col].to_numpy(dtype=float)
    pressure = df[pressure_col].to_numpy(dtype=float)

    # Remove invalid points
    mask = np.isfinite(time) & np.isfinite(pressure) & (pressure > 0)
    time = time[mask]
    pressure = pressure[mask]

    if len(time) < 2:
        return {'error': 'Insufficient valid data points'}

    # Convert pressure to Torr if needed
    if pressure_units == 'mbar':
        pressure_torr = pressure * 0.75006  # mbar to Torr
    elif pressure_units == 'Pa':
        pressure_torr = pressure * 0.00750062  # Pa to Torr
    else:
        pressure_torr = pressure

    # Calculate baseline (minimum)
    baseline_pressure = np.min(pressure_torr)

    # Define dose threshold: 10x baseline OR 5e-10 mbar (3.75e-10 Torr), whichever is lower
    threshold_from_baseline = baseline_pressure * 10
    absolute_threshold = 5e-10 * 0.75006  # Convert 5e-10 mbar to Torr
    dose_threshold = min(threshold_from_baseline, absolute_threshold)

    # Find dose period
    dose_mask = pressure_torr > dose_threshold

    if not np.any(dose_mask):
        return {
            'baseline_pressure_torr': float(baseline_pressure),
            'peak_pressure_torr': float(np.max(pressure_torr)),
            'dose_threshold_torr': float(dose_threshold),
            'dose_detected': False,
            'dose_duration_s': 0.0,
            'exposure_langmuirs': 0.0,
            'integrated_dose_cm2': 0.0,
            'molecular_weight_gmol': float(molecular_weight)
        }

    dose_time = time[dose_mask]
    dose_pressure_torr = pressure_torr[dose_mask]

    dose_start = dose_time[0]
    dose_end = dose_time[-1]
    dose_duration = dose_end - dose_start

    # Calculate exposure in Langmuirs (1 L = 1e-6 Torr·s)
    exposure_torr_s = np.trapz(dose_pressure_torr, dose_time)
    exposure_langmuirs = exposure_torr_s * 1e6

    # Calculate molecular flux and integrated dose
    sqrt_MT = np.sqrt(molecular_weight * temperature_K)
    flux_array = 3.513e22 * dose_pressure_torr / sqrt_MT  # molecules/cm²/s
    integrated_dose = np.trapz(flux_array, dose_time)  # molecules/cm²

    return {
        'baseline_pressure_torr': float(baseline_pressure),
        'peak_pressure_torr': float(np.max(dose_pressure_torr)),
        'dose_threshold_torr': float(dose_threshold),
        'dose_detected': True,
        'dose_duration_s': float(dose_duration),
        'dose_start_time_s': float(dose_start),
        'dose_end_time_s': float(dose_end),
        'mean_dose_pressure_torr': float(np.mean(dose_pressure_torr)),
        'exposure_langmuirs': float(exposure_langmuirs),
        'molecular_weight_gmol': float(molecular_weight),
        'temperature_K': float(temperature_K),
        'integrated_dose_cm2': float(integrated_dose),
        'mean_flux_cm2s': float(np.mean(flux_array))
    }


# ==================== FLASH ANALYSIS ====================

def analyze_flash(df: pd.DataFrame,
                  temp_col: str = 'Pyro_T',
                  time_col: str = 'Time_s',
                  flash_threshold_C: float = 1000.0,
                  exit_threshold_C: float = 1000.0) -> Dict[str, Any]:
    """
    Analyze flash step: count flashes, measure duration and peak temperature

    Flash detection:
    - Sample held at 900°C
    - Flash events are >1000°C
    - Enters flash when T > 1000°C
    - Exits flash when T < 1000°C
    """
    if temp_col not in df.columns or time_col not in df.columns:
        return {'error': f'Missing columns: {temp_col} or {time_col}'}

    time = df[time_col].to_numpy(dtype=float)
    temp = df[temp_col].to_numpy(dtype=float)

    # Remove invalid points
    mask = np.isfinite(time) & np.isfinite(temp)
    time = time[mask]
    temp = temp[mask]

    if len(time) < 2:
        return {'error': 'Insufficient valid data points'}

    # Detect flash events
    in_flash = False
    flash_events = []
    current_flash = None

    for i, (t, T) in enumerate(zip(time, temp)):
        if not in_flash and T > flash_threshold_C:
            # Enter flash
            in_flash = True
            current_flash = {
                'start_time': t,
                'start_index': i,
                'peak_temp': T,
                'peak_time': t,
                'peak_index': i
            }
        elif in_flash:
            # Update peak if higher
            if T > current_flash['peak_temp']:
                current_flash['peak_temp'] = T
                current_flash['peak_time'] = t
                current_flash['peak_index'] = i

            # Check for exit
            if T < exit_threshold_C:
                # Exit flash
                current_flash['end_time'] = t
                current_flash['end_index'] = i
                current_flash['duration'] = t - current_flash['start_time']
                flash_events.append(current_flash)
                in_flash = False
                current_flash = None

    # If still in flash at end
    if in_flash and current_flash is not None:
        current_flash['end_time'] = time[-1]
        current_flash['end_index'] = len(time) - 1
        current_flash['duration'] = time[-1] - current_flash['start_time']
        flash_events.append(current_flash)

    # Compile results
    if not flash_events:
        return {
            'flash_count': 0,
            'total_flash_time_s': 0.0,
            'peak_temperature_C': float(np.max(temp)) if len(temp) > 0 else 0.0,
            'flashes': []
        }

    total_flash_time = sum(f['duration'] for f in flash_events)
    peak_temp_overall = max(f['peak_temp'] for f in flash_events)

    return {
        'flash_count': len(flash_events),
        'total_flash_time_s': float(total_flash_time),
        'peak_temperature_C': float(peak_temp_overall),
        'flash_threshold_C': float(flash_threshold_C),
        'exit_threshold_C': float(exit_threshold_C),
        'flashes': [
            {
                'flash_number': i + 1,
                'start_time_s': float(f['start_time']),
                'end_time_s': float(f['end_time']),
                'duration_s': float(f['duration']),
                'peak_temp_C': float(f['peak_temp']),
                'peak_time_s': float(f['peak_time'])
            }
            for i, f in enumerate(flash_events)
        ]
    }


# ==================== OVERGROWTH ANALYSIS ====================

def analyze_overgrowth(df: pd.DataFrame,
                       current_col: str = 'TDK_I',
                       time_col: str = 'Time_s',
                       calibration: Optional[Dict[str, Any]] = None,
                       locking_layer_rate_ML_min: Optional[float] = None,
                       lte_rate_ML_min: Optional[float] = None,
                       thresholds: Optional[Dict[str, float]] = None,
                       auto_detect_thresholds: bool = False) -> Dict[str, Any]:
    """
    Overgrowth analysis with three phases:
    - Locking Layer (RT): estimated as 10 ML ending at RTA start (if rate provided),
      otherwise threshold-based detection.
    - RTA: current > 0.10 A (default)
    - LTE: 0.03–0.10 A (default)

    Enhancements:
    - Robust contiguous segmentation per phase.
    - Optional auto-thresholding (off by default).
    - Deposition using: 10 ML = 1.36 nm → 1 ML = 0.136 nm.
    - Adds current_q33/current_q66 for UI display.
    - For RT: if locking_layer_rate_ML_min is available and RTA start is detected,
      assume RT is exactly 10 ML ending at RTA start and back-calculate its duration.
      Mark RT metrics as 'estimated': True.
    """
    if time_col not in df.columns or current_col not in df.columns:
        return {'error': f'Missing columns: {current_col} or {time_col}'}

    time = df[time_col].to_numpy(dtype=float)
    current = df[current_col].to_numpy(dtype=float)

    mask_valid = np.isfinite(time) & np.isfinite(current)
    time = time[mask_valid]
    current = current[mask_valid]

    if len(time) < 2:
        return {'error': 'Insufficient valid data points'}

    # Quantiles for context and optional auto thresholds
    q10 = float(np.percentile(current, 10))
    q33 = float(np.percentile(current, 33))
    q50 = float(np.percentile(current, 50))
    q66 = float(np.percentile(current, 66))
    q90 = float(np.percentile(current, 90))

    # Default thresholds
    RT_THRESHOLD = 0.01
    RTA_THRESHOLD = 0.10
    LTE_MIN = 0.03
    LTE_MAX = 0.10

    if auto_detect_thresholds:
        # Conservative auto rules that still respect your defaults
        RT_THRESHOLD = min(RT_THRESHOLD, q10 * 1.2)
        RTA_THRESHOLD = max(RTA_THRESHOLD, q90 * 0.8)
        LTE_MIN = max(LTE_MIN, q33 * 0.9)
        LTE_MAX = min(LTE_MAX, q66 * 1.1)

    if thresholds:
        RT_THRESHOLD = thresholds.get('RT_max_A', RT_THRESHOLD)
        RTA_THRESHOLD = thresholds.get('RTA_min_A', RTA_THRESHOLD)
        LTE_MIN = thresholds.get('LTE_min_A', LTE_MIN)
        LTE_MAX = thresholds.get('LTE_max_A', LTE_MAX)

    # Masks based on thresholds
    rt_mask = current < RT_THRESHOLD
    rta_mask = current > RTA_THRESHOLD
    lte_mask = (current >= LTE_MIN) & (current < LTE_MAX)

    # Segments from masks
    rt_segments = _segments_from_mask(time, rt_mask)
    rta_segments = _segments_from_mask(time, rta_mask)
    lte_segments = _segments_from_mask(time, lte_mask)

    ML_TO_NM = 0.136  # 10 ML = 1.36 nm

    def summarize_phase(mask: np.ndarray,
                        segments: List[Dict[str, float]],
                        growth_rate: Optional[float],
                        phase_name: str) -> Dict[str, Any]:
        detected = bool(np.any(mask))
        if not detected:
            return {
                'detected': False,
                'duration_s': 0.0,
                'duration_min': 0.0,
                'duration_formatted': format_time(0.0),
                'median_current_A': 0.0,
                'mean_current_A': 0.0,
                'min_current_A': None,
                'max_current_A': None,
                'segments': [],
                'start_time_s': None,
                'end_time_s': None,
                'start_time_min': None,
                'end_time_min': None,
                'deposition_rate_ML_min': growth_rate if growth_rate is not None else None,
                'deposited_ML': None,
                'deposited_nm': None
            }

        phase_time = time[mask]
        phase_current = current[mask]
        duration_s = float(sum(seg['duration_s'] for seg in segments))
        duration_min = duration_s / 60.0
        median_I = float(np.median(phase_current))
        mean_I = float(np.mean(phase_current))

        if segments:
            start_s = float(segments[0]['start_time_s'])
            end_s = float(segments[-1]['end_time_s'])
        else:
            start_s = float(phase_time[0])
            end_s = float(phase_time[-1])

        metrics: Dict[str, Any] = {
            'detected': True,
            'duration_s': duration_s,
            'duration_min': duration_min,
            'duration_formatted': format_time(duration_s),
            'median_current_A': median_I,
            'mean_current_A': mean_I,
            'min_current_A': float(np.min(phase_current)),
            'max_current_A': float(np.max(phase_current)),
            'segments': segments,
            'start_time_s': start_s,
            'end_time_s': end_s,
            'start_time_min': start_s / 60.0,
            'end_time_min': end_s / 60.0
        }

        # Temperature at median current if calibration provided
        if calibration is not None:
            try:
                T_C = apply_calibration(median_I, calibration)
                metrics['median_temperature_C'] = float(T_C)
                metrics['median_temperature_K'] = float(T_C + 273.15)
            except Exception as e:
                metrics['calibration_error'] = str(e)

        # Deposition if growth rate provided
        if growth_rate is not None and duration_s > 0:
            deposited_ML = float(growth_rate * duration_min)
            metrics['deposition_rate_ML_min'] = float(growth_rate)
            metrics['deposited_ML'] = deposited_ML
            metrics['deposited_nm'] = float(deposited_ML * ML_TO_NM)
        else:
            metrics['deposition_rate_ML_min'] = None if growth_rate is None else float(growth_rate)
            metrics['deposited_ML'] = None
            metrics['deposited_nm'] = None

        return metrics

    # Start with standard summaries
    rt_metrics = summarize_phase(rt_mask, rt_segments, locking_layer_rate_ML_min, 'RT')
    rta_metrics = summarize_phase(rta_mask, rta_segments, 0.0, 'RTA')  # negligible growth
    lte_metrics = summarize_phase(lte_mask, lte_segments, lte_rate_ML_min, 'LTE')

    # --- Apply the RT = 10 ML assumption if possible ---
    rta_start_s = None
    if rta_segments:
        rta_start_s = float(rta_segments[0]['start_time_s'])

    if rta_start_s is not None and locking_layer_rate_ML_min and locking_layer_rate_ML_min > 0:
        # Duration required to deposit 10 ML at locking-layer rate
        rt_required_min = 10.0 / float(locking_layer_rate_ML_min)
        rt_required_s = rt_required_min * 60.0

        # Estimated RT start is 'rt_required_s' before RTA start
        rt_est_start_s = rta_start_s - rt_required_s
        rt_est_end_s = rta_start_s

        # Build stats from available data within this window
        in_window = (time >= rt_est_start_s) & (time <= rt_est_end_s) & np.isfinite(current)
        phase_current = current[in_window]
        phase_time = time[in_window]

        if len(phase_time) > 0:
            median_I = float(np.median(phase_current))
            mean_I = float(np.mean(phase_current))
            min_I = float(np.min(phase_current))
            max_I = float(np.max(phase_current))
        else:
            median_I = 0.0
            mean_I = 0.0
            min_I = None
            max_I = None

        rt_est_metrics: Dict[str, Any] = {
            'detected': True,
            'estimated': True,
            'estimation_method': '10 ML before RTA at locking-layer rate',
            'duration_s': float(rt_required_s),
            'duration_min': float(rt_required_min),
            'duration_formatted': format_time(rt_required_s),
            'median_current_A': median_I,
            'mean_current_A': mean_I,
            'min_current_A': min_I,
            'max_current_A': max_I,
            'segments': [{
                'start_time_s': float(rt_est_start_s),
                'end_time_s': float(rt_est_end_s),
                'duration_s': float(rt_required_s)
            }],
            'start_time_s': float(rt_est_start_s),
            'end_time_s': float(rt_est_end_s),
            'start_time_min': float(rt_est_start_s / 60.0),
            'end_time_min': float(rt_est_end_s / 60.0),
            'deposition_rate_ML_min': float(locking_layer_rate_ML_min),
            'deposited_ML': 10.0,
            'deposited_nm': float(10.0 * ML_TO_NM)
        }

        # Temperature at median current if calibration provided
        if calibration is not None and len(phase_time) > 0:
            try:
                T_C = apply_calibration(median_I, calibration)
                rt_est_metrics['median_temperature_C'] = float(T_C)
                rt_est_metrics['median_temperature_K'] = float(T_C + 273.15)
            except Exception as e:
                rt_est_metrics['calibration_error'] = str(e)

        # Override RT metrics with the estimated block
        rt_metrics = rt_est_metrics

    # Totals (with possible RT override)
    total_ML = 0.0
    total_nm = 0.0
    deposition_calculated = False
    for p in (rt_metrics, rta_metrics, lte_metrics):
        if p['detected'] and (p.get('deposited_ML') is not None):
            total_ML += p['deposited_ML']
            total_nm += p['deposited_nm']
            deposition_calculated = True

    total_duration_s = float(time[-1] - time[0])
    total_duration_min = total_duration_s / 60.0

    result = {
        'phase_thresholds': {
            'RT_max_A': RT_THRESHOLD,
            'RTA_min_A': RTA_THRESHOLD,
            'LTE_min_A': LTE_MIN,
            'LTE_max_A': LTE_MAX
        },
        'current_q33': q33,
        'current_q66': q66,
        'RT_growth': rt_metrics,
        'RTA_anneal': rta_metrics,
        'LTE_growth': lte_metrics,
        'total_duration_s': total_duration_s,
        'total_duration_min': total_duration_min,
        'total_duration_formatted': format_time(total_duration_s),
        'total_deposited_ML': float(total_ML) if deposition_calculated else None,
        'total_deposited_nm': float(total_nm) if deposition_calculated else None
    }

    if locking_layer_rate_ML_min is None or lte_rate_ML_min is None:
        result['note'] = (
            'Growth rates not provided - deposition amounts not calculated for one or more phases. '
            'Provide locking_layer_rate_ML_min and lte_rate_ML_min for deposition calculation.'
        )

    return result


def apply_calibration(current: float, calibration: Dict[str, Any]) -> float:
    """
    Apply temperature calibration to current value

    Supports two calibration formats:
    1. Polynomial: {'model': 'polynomial', 'coefficients': [a0, a1, a2, ...]}
       T(I) = a0 + a1*I + a2*I^2 + a3*I^3 + ...
    2. Legacy ti_fitter: {'model': 'exp'/'pow'/'log', 'params': [...]}
    """
    model = calibration.get('model', 'polynomial')

    if model == 'polynomial':
        # Polynomial calibration: T = a0 + a1*I + a2*I^2 + ...
        coeffs = calibration.get('coefficients', calibration.get('fit_coefficients', []))
        if not coeffs:
            raise ValueError("No coefficients found in polynomial calibration")

        # Coefficients are in descending order [a3, a2, a1, a0]
        # Use numpy polyval for evaluation
        temp_C = np.polyval(coeffs, current)
        return float(temp_C)

    else:
        # Legacy ti_fitter models - return temperature in Kelvin
        try:
            from stm_fab.analysis.ti_fitter import model_exp, model_pow, model_log

            params = calibration.get('params', [])

            if model == 'exp':
                T_K = model_exp(current, *params)
            elif model == 'pow':
                T_K = model_pow(current, *params)
            elif model == 'log':
                T_K = model_log(current, *params)
            else:
                raise ValueError(f"Unknown calibration model: {model}")

            # Convert Kelvin to Celsius
            return float(T_K - 273.15)

        except ImportError:
            raise ImportError("ti_fitter module not available for legacy calibration")


# ==================== SUSI ANALYSIS ====================

def analyze_susi(df: pd.DataFrame,
                 current_col: str = 'TDK_I_1',
                 time_col: str = 'Time_s',
                 threshold_A: Optional[float] = None) -> Dict[str, Any]:
    """
    Analyze SUSI operation: measure runtime at SUSI operating current (not sample current).
    Auto-finds SUSI column (prefers TDK_I_1) if the given column is missing.
    """
    if current_col not in df.columns:
        detected = _find_susi_current_column(df) or _find_current_column(df)
        if detected:
            current_col = detected

    if current_col not in df.columns or time_col not in df.columns:
        return {'error': f'Missing columns: {current_col} or {time_col}'}

    time = df[time_col].to_numpy(dtype=float)
    current = df[current_col].to_numpy(dtype=float)

    # Remove invalid points
    mask = np.isfinite(time) & np.isfinite(current)
    time = time[mask]
    current = current[mask]

    if len(time) < 2:
        return {'error': 'Insufficient valid data points'}

    # Auto threshold if not provided
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

    # Segment contiguous operating periods
    padded = np.concatenate(([False], operating_mask, [False]))
    diffs = np.diff(padded.astype(int))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]

    segments = []
    for s, e in zip(starts, ends):
        seg_t = time[s:e]
        seg_i = current[s:e]
        duration = seg_t[-1] - seg_t[0] if len(seg_t) > 1 else 0.0
        segments.append({
            'start_time_s': float(seg_t[0]),
            'end_time_s': float(seg_t[-1]),
            'duration_s': float(duration),
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
                   stabilization_threshold: float = 0.05) -> Dict[str, Any]:
    """
    Analyze outgas/degas step: measure base pressure, peak pressure, time to stabilize.

    Prefers VT gauge (P_VT) and only falls back to MBE if VT is not present.
    Returns 'pressure_column_used' and a warning if it had to fall back.
    """
    # Resolve columns
    if time_col not in df.columns:
        return {'error': f'Missing column: {time_col}'}

    chosen_col = pressure_col
    if chosen_col is None or chosen_col not in df.columns:
        chosen_col = _find_pressure_column(df, prefer='VT')
        if chosen_col is None:
            return {'error': 'No suitable pressure column found (expected P_VT or P_MBE)'}

    # Detect if we fell back to MBE, for transparency
    preferred_vt = _find_pressure_column(df, prefer='VT')
    warning = None
    if preferred_vt and chosen_col != preferred_vt:
        warning = f"Preferred VT gauge not found. Using '{chosen_col}' instead."

    # Extract arrays
    time = df[time_col].to_numpy(dtype=float)
    pressure = df[chosen_col].to_numpy(dtype=float)

    # Remove invalid points
    mask = np.isfinite(time) & np.isfinite(pressure) & (pressure > 0)
    time = time[mask]
    pressure = pressure[mask]

    if len(time) < 2:
        return {'error': 'Insufficient valid data points'}

    # Convert to Torr for standardization
    if pressure_units == 'mbar':
        pressure_torr = pressure * 0.75006
    elif pressure_units == 'Pa':
        pressure_torr = pressure * 0.00750062
    else:
        # assume already Torr
        pressure_torr = pressure

    # Calculate key metrics
    base_pressure = np.min(pressure_torr)
    peak_pressure = np.max(pressure_torr)
    final_pressure = pressure_torr[-1]

    # Detect time to stabilization
    stabilized_time = None
    stabilized = False

    dt = np.median(np.diff(time)) if len(time) > 1 else 0
    window_size = int(stabilization_window_s / dt) if dt > 0 else 0
    if window_size < 5:
        window_size = 5  # Minimum window size

    for i in range(window_size, len(pressure_torr)):
        window = pressure_torr[i - window_size:i]
        mean_p = np.mean(window)
        std_p = np.std(window)
        if mean_p > 0 and (std_p / mean_p) < stabilization_threshold:
            stabilized_time = time[i]
            stabilized = True
            break

    result = {
        'pressure_column_used': chosen_col,
        'base_pressure_torr': float(base_pressure),
        'base_pressure_mbar': float(base_pressure / 0.75006),
        'peak_pressure_torr': float(peak_pressure),
        'peak_pressure_mbar': float(peak_pressure / 0.75006),
        'final_pressure_torr': float(final_pressure),
        'final_pressure_mbar': float(final_pressure / 0.75006),
        'total_duration_s': float(time[-1] - time[0]),
        'pressure_drop_factor': float(peak_pressure / base_pressure) if base_pressure > 0 else 0.0,
        'stabilization_detected': stabilized
    }

    if stabilized:
        result['time_to_stabilize_s'] = float(stabilized_time - time[0])
    else:
        result['time_to_stabilize_s'] = None
        result['stabilization_note'] = 'Pressure did not stabilize within measurement'

    if warning:
        result['pressure_column_warning'] = warning

    return result

# ==================== TERMINATION ANALYSIS ====================

def analyze_termination(df: pd.DataFrame,
                        pressure_col: str = 'P_VT',
                        time_col: str = 'Time_s',
                        molecular_weight: Optional[float] = None,
                        filename: str = '',
                        temperature_K: float = 300.0,
                        pressure_units: str = 'mbar') -> Dict[str, Any]:
    """
    Analyze H-termination step: exposure time, average dose pressure, duration

    This is essentially a dose analysis with additional termination-specific metrics.
    """
    # For H-termination, use H2 molecular weight if not specified
    if molecular_weight is None:
        molecular_weight = 2.016  # H2

    # Use dose analysis as base
    dose_metrics = analyze_dose(df, pressure_col, time_col, molecular_weight,
                                filename, temperature_K, pressure_units)

    # Add termination-specific interpretation
    if dose_metrics.get('dose_detected'):
        dose_metrics['termination_type'] = 'H-termination'
        dose_metrics['exposure_time_s'] = dose_metrics['dose_duration_s']
        dose_metrics['average_pressure_torr'] = dose_metrics['mean_dose_pressure_torr']
        dose_metrics['h2_exposure_langmuirs'] = dose_metrics['exposure_langmuirs']

    return dose_metrics


# ==================== MAIN ANALYSIS DISPATCHER ====================

def analyze_process_file(file_path: str,
                         calibration: Optional[Dict[str, Any]] = None,
                         locking_layer_rate_ML_min: Optional[float] = None,
                         lte_rate_ML_min: Optional[float] = None,
                         auto_detect_thresholds: bool = False,
                         thresholds: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """
    Analyze a LabVIEW process file and return comprehensive metrics

    Args:
        file_path: Path to LabVIEW .txt file
        calibration: Optional temperature calibration for overgrowth analysis
        locking_layer_rate_ML_min: Growth rate for locking layer in ML/min
        lte_rate_ML_min: Growth rate for LTE in ML/min
        auto_detect_thresholds: Enable automatic thresholding for overgrowth phases
        thresholds: Optional dict override for thresholds keys:
                    {'RT_max_A':..., 'RTA_min_A':..., 'LTE_min_A':..., 'LTE_max_A':...}
    """
    path = Path(file_path)

    if not path.exists():
        return {'error': f'File not found: {file_path}'}

    # Parse file
    try:
        if LabVIEWParser is None:
            raise ImportError("LabVIEWParser not available")
        parser = LabVIEWParser(str(path))
        parsed = parser.parse()
        df = parsed['data']
        file_type = detect_file_type(path.name)
    except Exception as e:
        return {'error': f'Failed to parse file: {str(e)}'}

    # Normalize column names
    df.columns = [c.strip().replace(' ', '_') for c in df.columns]
    if 'Time' in df.columns and 'Time_s' not in df.columns:
        df.rename(columns={'Time': 'Time_s'}, inplace=True)

    result = {
        'filename': path.name,
        'filepath': str(path.absolute()),
        'file_type': file_type,
        'timestamp': datetime.now().isoformat()
    }

    # Dispatch to appropriate analyzer
    if file_type == 'dose':
        metrics = analyze_dose(df, filename=path.name)
    elif file_type == 'termination' or file_type == 'hterm':
        metrics = analyze_termination(df, filename=path.name)
    elif file_type == 'flash':
        metrics = analyze_flash(df)
    elif file_type == 'outgas':
        # Prefer VT gauge; fallback to MBE if missing
        pressure_col = _find_pressure_column(df, prefer='VT')
        metrics = analyze_outgas(df, pressure_col=pressure_col)

    elif file_type == 'susi':
        # Ensure Time_s exists
        if 'Time' in df.columns and 'Time_s' not in df.columns:
            df.rename(columns={'Time': 'Time_s'}, inplace=True)

        cols = set(df.columns)

        # Prefer exact normalized names if present, else fallback to finders
        susi_current_col = 'TDK_I_1' if 'TDK_I_1' in cols else (_find_susi_current_column(df) or 'TDK_I_1')
        sample_current_col = 'TDK_I' if 'TDK_I' in cols else (_find_sample_current_column(df) or 'TDK_I')

        # Run analyses
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

        metrics = {
            'columns_used': {
                'susi_current_col': susi_current_col,
                'sample_current_col': sample_current_col,
                'time_col': 'Time_s'
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


# ==================== BATCH PROCESSING ====================

def export_metrics_for_folder(input_directory: str,
                              output_path: Optional[str] = None,
                              calibration: Optional[Dict[str, Any]] = None,
                              locking_layer_rate_ML_min: Optional[float] = None,
                              lte_rate_ML_min: Optional[float] = None) -> List[Dict[str, Any]]:
    """
    Analyze all LabVIEW files in a folder and export metrics

    Args:
        input_directory: Directory containing LabVIEW .txt files
        output_path: Optional path to save JSON summary
        calibration: Optional temperature calibration
        locking_layer_rate_ML_min: Growth rate for locking layer in ML/min
        lte_rate_ML_min: Growth rate for LTE in ML/min

    Returns:
        List of analysis results
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

    # Export to JSON if path provided
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved metrics summary to: {output_path}")

    return results


# ==================== SUSI CALIBRATION INTEGRATION ====================

def analyze_with_active_calibration(file_path: str,
                                    calibration: Optional[Dict[str, Any]] = None,
                                    calibration_db_path: Optional[str] = None,
                                    override_rates: Optional[Tuple[float, float]] = None) -> Dict[str, Any]:
    """
    Analyze overgrowth file using the active SUSI calibration from database

    This function automatically retrieves the current active growth rates from
    the calibration database, so you don't need to specify them manually.
    """
    try:
        from stm_fab.analysis.susi_calibration import SUSICalibrationManager

        # Get active calibration
        manager = SUSICalibrationManager(calibration_db_path)
        active_cal = manager.get_active_calibration()

        if active_cal is None and override_rates is None:
            print("⚠ Warning: No active SUSI calibration found.")
            print("  Add a calibration using: python susi_calibration.py add ...")
            print("  Or provide rates manually with override_rates parameter")
            # Analyze without rates
            return analyze_process_file(file_path, calibration=calibration)

        # Use override if provided, otherwise use active calibration
        if override_rates:
            locking_rate, lte_rate = override_rates
            print(f"Using override rates: Locking={locking_rate:.3f}, LTE={lte_rate:.3f} ML/min")
        else:
            locking_rate = active_cal.get_rate_ML_min('locking_layer')
            lte_rate = active_cal.get_rate_ML_min('lte')
            print(f"Using active calibration (ID {active_cal.calibration_id}) from {active_cal.date}:")
            print(f"  Locking Layer: {locking_rate:.3f} ML/min")
            print(f"  LTE:           {lte_rate:.3f} ML/min")

        # Analyze with rates
        result = analyze_process_file(
            file_path,
            calibration=calibration,
            locking_layer_rate_ML_min=locking_rate,
            lte_rate_ML_min=lte_rate
        )

        # Add calibration info to result
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
        print("  Using manual rates if provided...")
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
