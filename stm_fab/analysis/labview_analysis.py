# stm_fab/analysis/labview_analysis.py
from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from stm_fab.analysis.process_metrics import (
    detect_file_type,
    analyze_flash,
    analyze_dose,
    analyze_outgas,
    analyze_termination,
    analyze_overgrowth,
    analyze_susi,
)

try:
    from stm_fab.labview.labview_parser import LabVIEWParser
except ImportError:
    LabVIEWParser = None


# =========================
# Data loading and helpers
# =========================

def load_labview_df(file_path: str) -> pd.DataFrame:
    """
    Load a LabVIEW .txt file into a normalized DataFrame:
    - Normalizes columns (spaces -> underscores)
    - Ensures Time_s column exists (renames Time -> Time_s if needed)
    """
    if LabVIEWParser is None:
        raise ImportError("LabVIEWParser not available")

    parser = LabVIEWParser(file_path)
    parsed = parser.parse()
    if 'data' not in parsed or not isinstance(parsed['data'], pd.DataFrame):
        raise ValueError("Parsed LabVIEW file missing 'data' DataFrame")

    df = parsed['data'].copy()

    # Normalize column names
    df.columns = [c.strip().replace(' ', '_') for c in df.columns]

    # Ensure Time_s
    if 'Time_s' not in df.columns:
        if 'Time' in df.columns:
            df.rename(columns={'Time': 'Time_s'}, inplace=True)
        else:
            # Fallback: create a simple time base if missing
            df['Time_s'] = np.arange(len(df), dtype=float)

    return df


def _find_pressure_column(df: pd.DataFrame, prefer: str = 'VT') -> Optional[str]:
    """
    Simple pressure column finder (independent utility for plotting):
    prefer: 'VT' or 'MBE'
    """
    cols = set(df.columns)

    vt_candidates = ['P_VT', 'PVT', 'P_VT_1', 'P_VT_Torr', 'P_VT_mbar', 'P_VT_Pa']
    mbe_candidates = ['P_MBE', 'PMBE', 'P_MBE_Torr', 'P_MBE_mbar', 'P_MBE_Pa']

    def hit(cands):
        for c in cands:
            if c in cols:
                return c
        # relaxed
        norm = {c.lower().replace(' ', '_'): c for c in df.columns}
        for c in cands:
            key = c.lower().replace(' ', '_')
            if key in norm:
                return norm[key]
        return None

    if prefer == 'VT':
        vt = hit(vt_candidates)
        if vt:
            return vt
        mbe = hit(mbe_candidates)
        if mbe:
            return mbe
    else:
        mbe = hit(mbe_candidates)
        if mbe:
            return mbe
        vt = hit(vt_candidates)
        if vt:
            return vt

    # very last resort
    for c in df.columns:
        cl = c.lower().replace(' ', '_')
        if cl.startswith('p_') or cl == 'p' or 'pressure' in cl:
            return c
    return None


def _fmt_eng(x: float, unit: str = "") -> str:
    if x is None or not isinstance(x, (int, float)):
        return "N/A"
    return f"{x:.2e}{(' ' + unit) if unit else ''}"


def _annot(ax: plt.Axes, text: str, xy: Tuple[float, float], color: str = "black", fontsize: int = 9):
    ax.annotate(
        text, xy=xy, xytext=(5, 5), textcoords="offset points",
        fontsize=fontsize, color=color,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, alpha=0.8)
    )


# ===============
# Plotting funcs
# ===============

def plot_flash(df: pd.DataFrame,
               time_col: str = 'Time_s',
               temp_col_candidates: Optional[List[str]] = None) -> List[plt.Figure]:
    """
    Plot flash data (temperature vs time) with default thresholds:
    - Horizontal line at 1000 C
    - Annotate peak temperature
    """
    if temp_col_candidates is None:
        temp_col_candidates = ['Pyro_T', 'Pyro_T_', 'PyroT', 'Pyro_Temp', 'Pyro_Temperature']

    temp_col = None
    for c in temp_col_candidates:
        if c in df.columns:
            temp_col = c
            break
    if temp_col is None:
        # fall back: first column containing 'pyro' and 't'
        for c in df.columns:
            cl = c.lower().replace(' ', '_')
            if 'pyro' in cl and ('t' in cl or 'temp' in cl):
                temp_col = c
                break
    if temp_col is None:
        raise ValueError("No temperature column found for flash plot")

    metrics = analyze_flash(df, temp_col=temp_col, time_col=time_col)

    t = df[time_col].to_numpy(dtype=float)
    T = df[temp_col].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, T, 'r-', lw=1.4, label=temp_col)

    # Draw threshold line
    ax.axhline(1000.0, color='gray', ls='--', lw=1, label='1000 °C')

    # Annotate peak
    if 'peak_temperature_C' in metrics and isinstance(metrics['peak_temperature_C'], (int, float)):
        peak = metrics['peak_temperature_C']
        idx = int(np.nanargmax(T)) if np.isfinite(np.nanmax(T)) else None
        x_peak = float(t[idx]) if idx is not None else float(t[np.argmax(T)])
        _annot(ax, f"Peak: {peak:.1f} °C", (x_peak, peak), color='red')

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title("Flash: Temperature vs Time")
    ax.grid(True, alpha=0.3)
    ax.legend()

    return [fig]


def plot_dose(df: pd.DataFrame,
              time_col: str = 'Time_s',
              prefer_pressure: str = 'VT') -> List[plt.Figure]:
    """
    Plot dose data: pressure vs time, shading the detected dose interval.
    """
    pcol = _find_pressure_column(df, prefer=prefer_pressure) or _find_pressure_column(df, prefer='MBE')
    if pcol is None:
        raise ValueError("No pressure column found for dose plotting")

    metrics = analyze_dose(df, pressure_col=pcol, time_col=time_col)

    t = df[time_col].to_numpy(dtype=float)
    P = df[pcol].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, P, 'b-', lw=1.3, label=pcol)

    # Shade dose period
    if metrics.get('dose_detected'):
        t0 = metrics.get('dose_start_time_s')
        t1 = metrics.get('dose_end_time_s')
        if isinstance(t0, (int, float)) and isinstance(t1, (int, float)):
            ax.axvspan(t0, t1, color='orange', alpha=0.25, label='Dose')

        # Show basic annotations
        ax.set_title(f"Dose: {metrics.get('exposure_langmuirs', 0):.2f} L, "
                     f"Peak: {_fmt_eng(metrics.get('peak_pressure_torr'), 'Torr')}")
    else:
        ax.set_title("Dose: Not detected")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"{pcol} (raw units)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    return [fig]


def plot_outgas(df: pd.DataFrame,
                time_col: str = 'Time_s') -> List[plt.Figure]:
    """
    Plot outgas: pressure vs time (P_MBE preferred), show stabilization if detected.
    """
    pcol = _find_pressure_column(df, prefer='MBE') or _find_pressure_column(df, prefer='VT')
    if pcol is None:
        raise ValueError("No pressure column found for outgas plotting")

    metrics = analyze_outgas(df, pressure_col=pcol, time_col=time_col)

    t = df[time_col].to_numpy(dtype=float)
    P = df[pcol].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, P, 'k-', lw=1.3, label=pcol)

    # Annotate stabilization
    if metrics.get('stabilization_detected'):
        t_stab = metrics.get('time_to_stabilize_s')
        if isinstance(t_stab, (int, float)):
            ax.axvline(t_stab + float(t[0]), color='green', ls='--', lw=1, label='Stabilized')
            _annot(ax, f"Stabilized @ {t_stab:.1f} s", (t_stab + t[0], float(P[-1])), color='green')

    ax.set_title(f"Outgas: Base {_fmt_eng(metrics.get('base_pressure_mbar'), 'mbar')}, "
                 f"Peak {_fmt_eng(metrics.get('peak_pressure_mbar'), 'mbar')}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"{pcol} (raw units)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    return [fig]


def plot_termination(df: pd.DataFrame,
                     time_col: str = 'Time_s') -> List[plt.Figure]:
    """
    Plot H-termination: P_MBE vs time and shade dose interval as detected.
    """
    # Use analyze_termination (prefers P_MBE internally)
    metrics = analyze_termination(df, pressure_col=None, time_col=time_col)

    # pick a pressure column to plot
    pcol = metrics.get('pressure_column_used') or _find_pressure_column(df, prefer='MBE') or _find_pressure_column(df, prefer='VT')
    if pcol is None:
        raise ValueError("No pressure column found for H-termination plotting")

    t = df[time_col].to_numpy(dtype=float)
    P = df[pcol].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, P, 'b-', lw=1.3, label=pcol)

    # Shade termination period if detected as dose
    if metrics.get('dose_detected'):
        t0 = metrics.get('dose_start_time_s')
        t1 = metrics.get('dose_end_time_s')
        if isinstance(t0, (int, float)) and isinstance(t1, (int, float)):
            ax.axvspan(t0, t1, color='orange', alpha=0.25, label='Termination Dose')
        ax.set_title(f"H-Term: Peak {_fmt_eng(metrics.get('peak_pressure_mbar'), 'mbar')}, "
                     f"Exposure {metrics.get('exposure_langmuirs', 0):.2f} L")
    else:
        ax.set_title("H-Termination: No dose detected")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"{pcol} (raw units)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    return [fig]


def plot_susi(df: pd.DataFrame,
              time_col: str = 'Time_s',
              susi_current_col_candidates: Optional[List[str]] = None,
              sample_current_col_candidates: Optional[List[str]] = None,
              calibration: Optional[Dict[str, Any]] = None,
              locking_layer_rate_ML_min: Optional[float] = None,
              lte_rate_ML_min: Optional[float] = None,
              thresholds: Optional[Dict[str, float]] = None,
              auto_detect_thresholds: bool = False) -> List[plt.Figure]:
    """
    Plot SUSI-related data:
      - Figure 1: Sample current vs time, shaded RT/RTA/LTE as determined by analyze_overgrowth
      - Figure 2: SUSI current vs time, shaded operating segments from analyze_susi
    This will respect the current overgrowth rules implemented in process_metrics.
    """
    if susi_current_col_candidates is None:
        susi_current_col_candidates = ['TDK_I_1', 'TDK_I1', 'TDK2_I', 'SUSI_I', 'SUSI_Current', 'TDK_CH2_I']
    if sample_current_col_candidates is None:
        sample_current_col_candidates = ['TDK_I', 'TDK0_I', 'Sample_I', 'Sample_Current', 'TDK_CH1_I']

    # Resolve columns
    susi_col = None
    for c in susi_current_col_candidates:
        if c in df.columns:
            susi_col = c
            break
    if susi_col is None:
        # fallback: heuristic
        for c in df.columns:
            cl = c.lower().replace(' ', '_')
            if ('susi' in cl or 'tdk_i_1' in cl) and ('i' in cl):
                susi_col = c
                break
    if susi_col is None:
        # last resort
        susi_col = 'TDK_I_1' if 'TDK_I_1' in df.columns else None

    sample_col = None
    for c in sample_current_col_candidates:
        if c in df.columns:
            sample_col = c
            break
    if sample_col is None:
        # fallback heuristic
        for c in df.columns:
            cl = c.lower().replace(' ', '_')
            if ('sample' in cl or 'tdk_i' == cl) and ('i' in cl):
                sample_col = c
                break
    if sample_col is None:
        sample_col = 'TDK_I' if 'TDK_I' in df.columns else None

    if sample_col is None:
        raise ValueError("Sample current column not found for SUSI/overgrowth plotting")

    # Compute metrics
    ovg = analyze_overgrowth(
        df,
        current_col=sample_col,
        time_col=time_col,
        calibration=calibration,
        locking_layer_rate_ML_min=locking_layer_rate_ML_min,
        lte_rate_ML_min=lte_rate_ML_min,
        thresholds=thresholds,
        auto_detect_thresholds=auto_detect_thresholds
    )
    susi_metrics = None
    if susi_col is not None:
        susi_metrics = analyze_susi(df, current_col=susi_col, time_col=time_col)

    figs: List[plt.Figure] = []

    # Figure 1: Sample current + phases
    t = df[time_col].to_numpy(dtype=float)
    I = df[sample_col].to_numpy(dtype=float)

    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(t, I, 'g-', lw=1.3, label=f"{sample_col}")

    # Shade RT
    rt = ovg.get('RT_growth', {}) if isinstance(ovg, dict) else {}
    if rt.get('detected'):
        ax1.axvspan(rt['start_time_s'], rt['end_time_s'], color='limegreen', alpha=0.25, label='RT (10 ML)')
        _annot(ax1,
               f"RT: {rt.get('duration_s', 0)/60.0:.1f} min\n"
               f"I~{(rt.get('median_current_A') or 0):.4f} A\n"
               f"T~{(rt.get('median_temperature_C') or float('nan')):.0f}°C",
               (rt['start_time_s'], (np.nanmedian(I) if np.isfinite(np.nanmedian(I)) else 0.0)),
               color='green')

    # Shade RTA
    rta = ovg.get('RTA_anneal', {}) if isinstance(ovg, dict) else {}
    if rta.get('detected'):
        ax1.axvspan(rta['start_time_s'], rta['end_time_s'], color='orange', alpha=0.25, label='RTA')
        _annot(ax1,
               f"RTA: {rta.get('duration_s', 0)/60.0:.1f} min\n"
               f"I~{(rta.get('median_current_A') or 0):.4f} A\n"
               f"T~{(rta.get('median_temperature_C') or float('nan')):.0f}°C",
               (rta['start_time_s'], (np.nanmedian(I) if np.isfinite(np.nanmedian(I)) else 0.0)),
               color='darkorange')

    # Shade LTE
    lte = ovg.get('LTE_growth', {}) if isinstance(ovg, dict) else {}
    if lte.get('detected'):
        ax1.axvspan(lte['start_time_s'], lte['end_time_s'], color='purple', alpha=0.25, label='LTE')
        _annot(ax1,
               f"LTE: {lte.get('duration_s', 0)/60.0:.1f} min\n"
               f"I~{(lte.get('median_current_A') or 0):.4f} A\n"
               f"T~{(lte.get('median_temperature_C') or float('nan')):.0f}°C",
               (lte['start_time_s'], (np.nanmedian(I) if np.isfinite(np.nanmedian(I)) else 0.0)),
               color='purple')

    # Threshold lines (from ovg thresholds)
    thr = ovg.get('phase_thresholds', {}) if isinstance(ovg, dict) else {}
    if thr:
        if isinstance(thr.get('RT_max_A'), (int, float)):
            ax1.axhline(thr['RT_max_A'], color='gray', ls='--', lw=0.8, label=f"RT max {thr['RT_max_A']:.3f} A")
        if isinstance(thr.get('RTA_min_A'), (int, float)):
            ax1.axhline(thr['RTA_min_A'], color='gray', ls=':', lw=0.8, label=f"RTA min {thr['RTA_min_A']:.3f} A")

    ax1.set_title("Overgrowth (Sample Current) with RT / RTA / LTE")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Sample Current (A)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', ncol=2)
    figs.append(fig1)

    # Figure 2: SUSI current + operation
    if susi_col is not None and susi_metrics and susi_metrics.get('operating_time_detected'):
        It = df[susi_col].to_numpy(dtype=float)
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.plot(t, It, 'r-', lw=1.3, label=f"{susi_col}")

        # Shade operating segments
        for seg in susi_metrics.get('segments', []):
            ax2.axvspan(seg['start_time_s'], seg['end_time_s'], color='red', alpha=0.18, label='SUSI Operating')

        ax2.set_title("SUSI Operation (Source Current)")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("SUSI Current (A)")
        ax2.grid(True, alpha=0.3)
        # Avoid duplicate legend entries
        handles, labels = ax2.get_legend_handles_labels()
        uniq = dict(zip(labels, handles))
        ax2.legend(uniq.values(), uniq.keys(), loc='upper right')
        figs.append(fig2)

    return figs


# =========================
# Save/export plot helpers
# =========================

def save_figures(figs: List[plt.Figure],
                 out_dir: str,
                 base_name: str,
                 dpi: int = 200) -> List[str]:
    """
    Save a list of matplotlib figures as numbered PNG files
    Returns list of file paths.
    """
    out = []
    os.makedirs(out_dir, exist_ok=True)
    for i, fig in enumerate(figs, 1):
        fname = f"{base_name}_{i:02d}.png"
        fpath = str(Path(out_dir) / fname)
        fig.savefig(fpath, dpi=dpi, bbox_inches='tight')
        out.append(fpath)
        plt.close(fig)
    return out


def export_plots_for_folder(input_dir: str,
                            output_dir: str,
                            dpi: int = 200) -> Dict[str, List[str]]:
    """
    Batch export plots for all .txt files in a folder.
    Returns a dict: {filename: [list of created png files]}
    """
    results: Dict[str, List[str]] = {}
    in_path = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    files = list(in_path.glob("*.txt"))
    for fp in files:
        try:
            df = load_labview_df(str(fp))
            ftype = detect_file_type(fp.name)
            figs: List[plt.Figure] = []

            if ftype == 'flash':
                figs = plot_flash(df)
            elif ftype in ('termination', 'hterm'):
                figs = plot_termination(df)
            elif ftype == 'dose':
                figs = plot_dose(df)
            elif ftype == 'outgas':
                figs = plot_outgas(df)
            elif ftype == 'susi':
                figs = plot_susi(df)
            else:
                # Unknown: attempt a basic plot for inspection
                fig, ax = plt.subplots(figsize=(10, 3))
                t = df['Time_s'].to_numpy(dtype=float)
                ax.plot(t, t*0, 'k-')
                ax.set_title(f"Unknown type: {fp.name}")
                figs = [fig]

            saved = save_figures(figs, str(out_path), base_name=fp.stem, dpi=dpi)
            results[fp.name] = saved

        except Exception as e:
            # record empty or error
            results[fp.name] = []
            print(f"Failed to export plots for {fp.name}: {e}")

    return results
