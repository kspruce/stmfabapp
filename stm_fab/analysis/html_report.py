# stm_fab/analysis/html_report.py
from __future__ import annotations
from pathlib import Path
from datetime import datetime
import io
import base64
import json
from typing import Dict, Any, List, Tuple, Optional

import matplotlib.pyplot as plt

from stm_fab.analysis.process_metrics import detect_file_type
from stm_fab.analysis.labview_analysis import (
    load_labview_df, plot_flash, plot_termination, plot_susi, plot_dose, plot_outgas
)


def _fig_to_base64(fig, dpi: int = 150) -> str:
    """Convert a Matplotlib figure to a base64 PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode('ascii')
    plt.close(fig)
    return data


def _plots_for_file(file_path: str, dpi: int = 150) -> Tuple[str, List[Dict[str, str]]]:
    """
    Produce plots for one file, return (detected_type, list_of_images),
    where images are dicts: {"label": ..., "b64": ...}
    """
    p = Path(file_path)
    df = load_labview_df(str(p))
    ftype = detect_file_type(p.name)

    # Use your existing plotting functions
    if ftype == 'flash':
        figs = plot_flash(df)
    elif ftype in ['termination', 'hterm']:
        figs = plot_termination(df)
    elif ftype == 'dose':
        figs = plot_dose(df)
    elif ftype == 'outgas':
        figs = plot_outgas(df)
    elif ftype == 'susi':
        figs = plot_susi(df)
    else:
        figs = []  # Unknown type -> no plots

    images = []
    for i, fig in enumerate(figs):
        b64 = _fig_to_base64(fig, dpi=dpi)
        images.append({"label": f"Plot {i+1}", "b64": b64})
    return ftype, images


def _format_time_hms(seconds: float) -> str:
    """Format seconds as HH:MM:SS"""
    if seconds is None or not isinstance(seconds, (int, float)) or seconds < 0:
        return "00:00:00"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _format_time_ms(seconds: float) -> str:
    """Format seconds as MM:SS"""
    if seconds is None or not isinstance(seconds, (int, float)) or seconds < 0:
        return "00:00"
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"


def _fmt_num(v, fmt=".2f", suffix=""):
    return f"{format(v, fmt)}{suffix}" if isinstance(v, (int, float)) else "N/A"


def _fmt_mbar(v):
    return f"{v:.2e} mbar" if isinstance(v, (int, float)) else "N/A"


def _extract_summary_metrics(results_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Extract summary metrics in the order: outgas, flash, hterm, dose, incorporation, overgrowth"""
    files = results_dict.get("files", [])
    
    summary = {
        'outgas': {},
        'flash': {},
        'hterm': {},
        'doses': [],
        'incorporation': {},
        'overgrowth': {}
    }
    
    for entry in files:
        ftype = entry.get('file_type', 'unknown')
        metrics = entry.get('metrics', {}) or {}
        
        if ftype == 'outgas':
            summary['outgas'] = {
                'base_pressure': metrics.get('base_pressure_mbar'),
                'time_to_stabilize': metrics.get('time_to_stabilize_s')
            }
        
        elif ftype == 'flash':
            summary['flash'] = {
                'flash_count': metrics.get('flash_count', 0),
                'total_time': metrics.get('total_flash_time_s', 0),
                'peak_temp': metrics.get('peak_temperature_C')
            }
        
        elif ftype in ['hterm', 'termination']:
            # Prefer mbar if present; otherwise convert from Torr
            peak_mbar = metrics.get('peak_pressure_mbar')
            if peak_mbar is None:
                peak_torr = metrics.get('peak_pressure_torr')
                if isinstance(peak_torr, (int, float)):
                    peak_mbar = peak_torr / 0.75006
            duration = metrics.get('dose_duration_s') if metrics.get('dose_duration_s') is not None else metrics.get('total_duration_s')
            summary['hterm'] = {
                'peak_pressure': peak_mbar,
                'duration': duration
            }
        
        elif ftype == 'dose':
            mw = metrics.get('molecular_weight_gmol', 34.0)
            mol = 'AsH₃' if isinstance(mw, (int, float)) and mw > 50 else 'PH₃'
            summary['doses'].append({
                'gas_type': metrics.get('gas_type', mol),
                'duration': metrics.get('dose_duration_s'),
                'exposure': metrics.get('exposure_langmuirs')
            })
        
        elif ftype in ['inc', 'incorporation']:
            summary['incorporation'] = {
                'temperature': metrics.get('median_temperature_C'),
                'duration': metrics.get('duration_s')
            }
        
        elif ftype == 'susi':
            # Nested structure: metrics['susi'] and metrics['overgrowth']
            susi_metrics = metrics.get('susi', {}) or {}
            ovg = metrics.get('overgrowth', {}) or {}
            rt = ovg.get('RT_growth', {}) if isinstance(ovg, dict) else {}
            rta = ovg.get('RTA_anneal', {}) if isinstance(ovg, dict) else {}
            lte = ovg.get('LTE_growth', {}) if isinstance(ovg, dict) else {}
            
            total_si = 0.0
            if isinstance(rt.get('deposited_nm'), (int, float)): total_si += rt.get('deposited_nm', 0.0)
            if isinstance(lte.get('deposited_nm'), (int, float)): total_si += lte.get('deposited_nm', 0.0)
            
            summary['overgrowth'] = {
                'susi_current': susi_metrics.get('mean_operating_current_A'),
                'susi_time': susi_metrics.get('total_operating_time_s'),
                'total_silicon_nm': total_si,
                'total_duration': ovg.get('total_duration_s'),
                'total_deposited_nm': ovg.get('total_deposited_nm'),
                'rt': rt,
                'rta': rta,
                'lte': lte
            }
    
    return summary


def _render_outgas_section(metrics: Dict[str, Any]) -> str:
    """Render outgas file section"""
    if not metrics:
        return "<p><em>No data</em></p>"
    
    base_p = metrics.get('base_pressure_mbar', None)
    peak_p = metrics.get('peak_pressure_mbar', None)
    stab_time = metrics.get('time_to_stabilize_s')
    
    html = "<div class='metric-section'>"
    html += "<h4>Key Metrics</h4>"
    html += "<table class='metrics-table'>"
    html += f"<tr><td>Base Pressure</td><td>{_fmt_mbar(base_p)}</td></tr>"
    html += f"<tr><td>Peak Pressure</td><td>{_fmt_mbar(peak_p)}</td></tr>"
    html += f"<tr><td>Time to Stabilize</td><td>{_format_time_ms(stab_time) if isinstance(stab_time, (int, float)) else 'N/A'}</td></tr>"
    html += "</table>"
    html += "</div>"
    
    return html


def _render_flash_section(metrics: Dict[str, Any]) -> str:
    """Render flash file section"""
    if not metrics:
        return "<p><em>No data</em></p>"
    
    flash_count = metrics.get('flash_count', 0)
    total_time = metrics.get('total_flash_time_s', 0)
    peak_temp = metrics.get('peak_temperature_C', None)
    flashes = metrics.get('flashes', [])
    
    html = "<div class='metric-section'>"
    html += "<h4>Key Metrics</h4>"
    html += "<table class='metrics-table'>"
    html += f"<tr><td>Flash Count</td><td>{int(flash_count)}</td></tr>"
    html += f"<tr><td>Total Flash Time</td><td>{_format_time_hms(total_time)}</td></tr>"
    html += f"<tr><td>Peak Temperature</td><td>{_fmt_num(peak_temp, '.1f', ' °C')}</td></tr>"
    html += "</table>"
    
    if flashes:
        html += "<h4>Individual Flashes</h4>"
        html += "<table class='metrics-table'>"
        html += "<tr><th>#</th><th>Duration</th><th>Peak Temp (°C)</th></tr>"
        for i, flash in enumerate(flashes, 1):
            dur = flash.get('duration_s', 0)
            peak = flash.get('peak_temp_C', 0)
            html += f"<tr><td>{i}</td><td>{dur:.1f} s</td><td>{peak:.1f}</td></tr>"
        html += "</table>"
    
    html += "</div>"
    return html


def _render_hterm_section(metrics: Dict[str, Any]) -> str:
    """Render H-termination file section (using P_MBE preference)"""
    if not metrics:
        return "<p><em>No data</em></p>"
    
    peak_mbar = metrics.get('peak_pressure_mbar', None)
    if peak_mbar is None:
        peak_torr = metrics.get('peak_pressure_torr', None)
        if isinstance(peak_torr, (int, float)):
            peak_mbar = peak_torr / 0.75006
    
    duration = metrics.get('dose_duration_s', None)
    if duration is None:
        duration = metrics.get('total_duration_s', None)
    exposure = metrics.get('exposure_langmuirs', None)
    avg_p_torr = metrics.get('mean_dose_pressure_torr', None)
    avg_p_mbar = metrics.get('mean_dose_pressure_mbar', None)
    if avg_p_mbar is None and isinstance(avg_p_torr, (int, float)):
        avg_p_mbar = avg_p_torr / 0.75006
    
    html = "<div class='metric-section'>"
    html += "<h4>Key Metrics</h4>"
    html += "<table class='metrics-table'>"
    html += f"<tr><td>Peak Pressure (P_MBE)</td><td>{_fmt_mbar(peak_mbar)}</td></tr>"
    html += f"<tr><td>Termination Duration</td><td>{_format_time_hms(duration) if isinstance(duration, (int, float)) else 'N/A'}</td></tr>"
    html += f"<tr><td>H₂ Exposure</td><td>{_fmt_num(exposure, '.2f', ' L')}</td></tr>"
    html += f"<tr><td>Average Pressure</td><td>{_fmt_mbar(avg_p_mbar)}</td></tr>"
    col_used = metrics.get('pressure_column_used')
    if col_used:
        html += f"<tr><td>Pressure Column</td><td>{col_used}</td></tr>"
    html += "</table>"
    html += "</div>"
    return html


def _render_dose_section(metrics: Dict[str, Any]) -> str:
    """Render dose file section"""
    if not metrics:
        return "<p><em>No data</em></p>"
    
    mw = metrics.get('molecular_weight_gmol', 34.0)
    molecule = 'AsH₃' if isinstance(mw, (int, float)) and mw > 50 else 'PH₃'
    duration = metrics.get('dose_duration_s')
    exposure = metrics.get('exposure_langmuirs')
    integrated = metrics.get('integrated_dose_cm2')
    
    html = "<div class='metric-section'>"
    html += "<h4>Key Metrics</h4>"
    html += "<table class='metrics-table'>"
    html += f"<tr><td>Molecule</td><td>{molecule} ({mw:.2f} g/mol)</td></tr>" if isinstance(mw, (int, float)) else f"<tr><td>Molecule</td><td>{molecule}</td></tr>"
    html += f"<tr><td>Dose Duration</td><td>{_format_time_hms(duration) if isinstance(duration, (int, float)) else 'N/A'}</td></tr>"
    html += f"<tr><td>Exposure</td><td>{_fmt_num(exposure, '.2f', ' L')}</td></tr>"
    html += f"<tr><td>Integrated Dose</td><td>{_fmt_num(integrated, '.2e', ' molecules/cm²')}</td></tr>"
    html += "</table>"
    html += "</div>"
    return html


def _render_overgrowth_section(metrics: Dict[str, Any]) -> str:
    """
    Render overgrowth file section (SUSI + RT/RTA/LTE) with complete details:
    - Inputs (columns used, rates)
    - Thresholds (source and values)
    - Overall totals
    - SUSI operation
    - RT, RTA, LTE phases with duration, deposition, currents, temperatures
    """
    if not metrics:
        return "<p><em>No data</em></p>"

    susi_metrics = metrics.get('susi', {}) or {}
    ovg = metrics.get('overgrowth', {}) or {}

    if not isinstance(ovg, dict) or not ovg:
        return "<p><em>No overgrowth data</em></p>"

    rt = ovg.get('RT_growth', {}) or {}
    rta = ovg.get('RTA_anneal', {}) or {}
    lte = ovg.get('LTE_growth', {}) or {}

    # Inputs
    cols_used = metrics.get('columns_used', {}) or {}
    rates_used = metrics.get('rates_used', {}) or {}

    # Thresholds
    thr = ovg.get('phase_thresholds', {}) or {}
    thr_source = ovg.get('phase_thresholds_source', 'unknown')
    thr_fb = ovg.get('thresholds_auto_fallback', False)

    def fmt_duration(sec):
        return _format_time_hms(sec) if isinstance(sec, (int, float)) else "N/A"

    html = "<div class='metric-section'>"

    # Inputs (columns, rates)
    html += "<h4>Analysis Inputs</h4>"
    html += "<table class='metrics-table'>"
    if cols_used:
        html += f"<tr><td>SUSI Current Column</td><td>{cols_used.get('susi_current_col', 'N/A')}</td></tr>"
        html += f"<tr><td>Sample Current Column</td><td>{cols_used.get('sample_current_col', 'N/A')}</td></tr>"
        if cols_used.get('voltage_col'):
            html += f"<tr><td>Voltage Column</td><td>{cols_used.get('voltage_col')}</td></tr>"
    if rates_used:
        lr = rates_used.get('locking_layer_rate_ML_min', None)
        lte_r = rates_used.get('lte_rate_ML_min', None)
        html += f"<tr><td>Locking Layer Rate</td><td>{_fmt_num(lr, '.3f', ' ML/min')}</td></tr>"
        html += f"<tr><td>LTE Rate</td><td>{_fmt_num(lte_r, '.3f', ' ML/min')}</td></tr>"
    html += "</table>"

    # Thresholds used
    html += "<h4>Phase Thresholds</h4>"
    html += "<table class='metrics-table'>"
    html += f"<tr><td>Source</td><td>{thr_source}{' (fallback auto)' if thr_fb else ''}</td></tr>"
    html += f"<tr><td>RT max (A)</td><td>{_fmt_num(thr.get('RT_max_A'), '.4f', ' A')}</td></tr>"
    html += f"<tr><td>RTA min (A)</td><td>{_fmt_num(thr.get('RTA_min_A'), '.4f', ' A')}</td></tr>"
    html += f"<tr><td>LTE min (A)</td><td>{_fmt_num(thr.get('LTE_min_A'), '.4f', ' A')}</td></tr>"
    html += f"<tr><td>LTE max (A)</td><td>{_fmt_num(thr.get('LTE_max_A'), '.4f', ' A')}</td></tr>"
    html += "</table>"

    # Overall totals
    html += "<h4>Overall Deposition</h4>"
    html += "<table class='metrics-table'>"
    total_dur = ovg.get('total_duration_s')
    total_nm = ovg.get('total_deposited_nm')
    total_ml = ovg.get('total_deposited_ML')
    html += f"<tr><td>Total Duration</td><td>{fmt_duration(total_dur)}</td></tr>"
    if isinstance(total_ml, (int, float)) and isinstance(total_nm, (int, float)):
        html += f"<tr><td>Total Deposited</td><td>{total_ml:.1f} ML ({total_nm:.2f} nm)</td></tr>"
    else:
        html += "<tr><td>Total Deposited</td><td>N/A (no rates provided)</td></tr>"
    html += "</table>"

    # SUSI operation
    html += "<h4>SUSI Operation</h4>"
    html += "<table class='metrics-table'>"
    html += f"<tr><td>Mean Current</td><td>{_fmt_num(susi_metrics.get('mean_operating_current_A'), '.4f', ' A')}</td></tr>"
    html += f"<tr><td>Total Operating Time</td><td>{fmt_duration(susi_metrics.get('total_operating_time_s'))}</td></tr>"
    html += "</table>"

    # RT Growth (Locking Layer)
    html += "<h4>RT Growth (Locking Layer)</h4>"
    html += "<table class='metrics-table'>"
    if rt.get('detected'):
        html += f"<tr><td>Duration</td><td>{fmt_duration(rt.get('duration_s'))}</td></tr>"
        if isinstance(rt.get('deposited_ML'), (int, float)) and isinstance(rt.get('deposited_nm'), (int, float)):
            html += f"<tr><td>Deposited</td><td>{rt['deposited_ML']:.1f} ML ({rt['deposited_nm']:.2f} nm)</td></tr>"
        rate = rt.get('deposition_rate_ML_min')
        html += f"<tr><td>Deposition Rate</td><td>{_fmt_num(rate, '.2f', ' ML/min')}</td></tr>"
        html += f"<tr><td>Median Current</td><td>{_fmt_num(rt.get('median_current_A'), '.4f', ' A')}</td></tr>"
        html += f"<tr><td>Median Temperature</td><td>{_fmt_num(rt.get('median_temperature_C'), '.1f', ' °C')}</td></tr>"
        if rt.get('estimated'):
            html += f"<tr><td colspan='2'><em>Note: RT set to fixed 10 ML ending at RTA start</em></td></tr>"
        if rt.get('calibration_note'):
            html += f"<tr><td colspan='2'><em>{rt.get('calibration_note')}</em></td></tr>"
    else:
        html += "<tr><td colspan='2'>Not detected</td></tr>"
    html += "</table>"

    # RTA Anneal
    html += "<h4>RTA Anneal</h4>"
    html += "<table class='metrics-table'>"
    if rta.get('detected'):
        html += f"<tr><td>Duration</td><td>{fmt_duration(rta.get('duration_s'))}</td></tr>"
        html += f"<tr><td>Median Current</td><td>{_fmt_num(rta.get('median_current_A'), '.4f', ' A')}</td></tr>"
        html += f"<tr><td>Median Temperature</td><td>{_fmt_num(rta.get('median_temperature_C'), '.1f', ' °C')}</td></tr>"
        html += f"<tr><td>Max Current</td><td>{_fmt_num(rta.get('max_current_A'), '.4f', ' A')}</td></tr>"
        if rta.get('calibration_note'):
            html += f"<tr><td colspan='2'><em>{rta.get('calibration_note')}</em></td></tr>"
    else:
        html += "<tr><td colspan='2'>Not detected</td></tr>"
    html += "</table>"

    # LTE Growth
    html += "<h4>LTE Growth</h4>"
    html += "<table class='metrics-table'>"
    if lte.get('detected'):
        html += f"<tr><td>Duration</td><td>{fmt_duration(lte.get('duration_s'))}</td></tr>"
        if isinstance(lte.get('deposited_ML'), (int, float)) and isinstance(lte.get('deposited_nm'), (int, float)):
            html += f"<tr><td>Deposited</td><td>{lte['deposited_ML']:.1f} ML ({lte['deposited_nm']:.2f} nm)</td></tr>"
        rate = lte.get('deposition_rate_ML_min')
        html += f"<tr><td>Deposition Rate</td><td>{_fmt_num(rate, '.2f', ' ML/min')}</td></tr>"
        html += f"<tr><td>Median Current</td><td>{_fmt_num(lte.get('median_current_A'), '.4f', ' A')}</td></tr>"
        html += f"<tr><td>Median Temperature</td><td>{_fmt_num(lte.get('median_temperature_C'), '.1f', ' °C')}</td></tr>"
        if lte.get('calibration_note'):
            html += f"<tr><td colspan='2'><em>{lte.get('calibration_note')}</em></td></tr>"
    else:
        html += "<tr><td colspan='2'>Not detected</td></tr>"
    html += "</table>"

    html += "</div>"
    return html


def build_device_report(folder_path: str,
                        results_dict: Dict[str, Any],
                        output_html: str,
                        device_meta: Optional[Dict[str, str]] = None,
                        dpi: int = 150,
                        inline_images: bool = True) -> Dict[str, str]:
    """
    Build device report with custom formatting
    """
    folder = Path(folder_path)
    output_html = Path(output_html)
    
    # Extract summary metrics
    summary = _extract_summary_metrics(results_dict)
    
    # Collect and order files
    files_ordered = []
    process_order = ['outgas', 'flash', 'hterm', 'dose', 'incorporation', 'susi']
    
    for ptype in process_order:
        for entry in results_dict.get("files", []):
            ftype = entry.get('file_type', 'unknown')
            if (ptype == 'susi' and ftype in ['susi', 'overgrowth']) or ftype == ptype:
                files_ordered.append(entry)
    
    # Add any remaining files not in the order
    for entry in results_dict.get("files", []):
        if entry not in files_ordered:
            files_ordered.append(entry)
    
    # Process each file
    per_file = []
    for entry in files_ordered:
        fname = entry.get('filename', 'unknown')
        fpath = entry.get('filepath', None)
        ftype = entry.get('file_type', 'unknown')
        status = "SUCCESS" if 'error' not in entry else f"ERROR: {entry['error']}"
        metrics = entry.get('metrics', {}) or {}
        
        # Get plots
        images = []
        if fpath and 'error' not in entry:
            try:
                detected_type, images = _plots_for_file(fpath, dpi=dpi)
                ftype = detected_type or ftype
            except Exception as e:
                status = f"ERROR (plot): {e}"
        
        # Render metrics HTML based on type
        if ftype == 'outgas':
            metrics_html = _render_outgas_section(metrics)
        elif ftype == 'flash':
            metrics_html = _render_flash_section(metrics)
        elif ftype in ['hterm', 'termination']:
            metrics_html = _render_hterm_section(metrics)
        elif ftype == 'dose':
            metrics_html = _render_dose_section(metrics)
        elif ftype in ['susi', 'overgrowth']:
            metrics_html = _render_overgrowth_section(metrics)
        else:
            metrics_html = f"<pre>{json.dumps(metrics, indent=2)}</pre>"
        
        per_file.append({
            "filename": fname,
            "file_type": ftype,
            "status": status,
            "metrics_html": metrics_html,
            "images": images
        })
    
    # Build HTML
    html = _build_html(
        device_name=(device_meta or {}).get("device_name") or folder.name,
        device_notes=(device_meta or {}).get("notes") or "",
        folder_path=str(folder.resolve()),
        summary=summary,
        per_file=per_file
    )
    
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(html, encoding="utf-8")
    return {"html_path": str(output_html)}


def _build_html(device_name: str, device_notes: str, folder_path: str, 
                summary: Dict[str, Any], per_file: List[Dict[str, Any]]) -> str:
    """Build the complete HTML document"""
    
    def fmt_num_or_na(v, fmt=".2f", suffix=""):
        return f"{format(v, fmt)}{suffix}" if isinstance(v, (int, float)) else "N/A"
    
    # Build summary section
    summary_html = "<div class='summary-section'>"
    summary_html += "<h2>Fabrication Summary</h2>"
    
    # Outgas
    if summary.get('outgas'):
        og = summary['outgas']
        bp = og.get('base_pressure')
        tts = og.get('time_to_stabilize')
        summary_html += "<h3>Outgas</h3><table class='summary-table'>"
        summary_html += f"<tr><td>Base Pressure</td><td>{fmt_num_or_na(bp, '.2e', ' mbar')}</td></tr>"
        summary_html += f"<tr><td>Time to Stabilize</td><td>{_format_time_hms(tts) if isinstance(tts, (int, float)) else 'N/A'}</td></tr>"
        summary_html += "</table>"
    
    # Flash
    if summary.get('flash'):
        fl = summary['flash']
        pt = fl.get('peak_temp')
        summary_html += "<h3>Flash</h3><table class='summary-table'>"
        summary_html += f"<tr><td>Total Flash Count</td><td>{int(fl.get('flash_count', 0))}</td></tr>"
        summary_html += f"<tr><td>Total Flash Time</td><td>{_format_time_hms(fl.get('total_time', 0))}</td></tr>"
        summary_html += f"<tr><td>Peak Flash Temp</td><td>{fmt_num_or_na(pt, '.1f', ' °C')}</td></tr>"
        summary_html += "</table>"
    
    # H-Termination
    if summary.get('hterm'):
        ht = summary['hterm']
        pp = ht.get('peak_pressure')
        dur = ht.get('duration')
        summary_html += "<h3>H-Termination</h3><table class='summary-table'>"
        summary_html += f"<tr><td>Peak Pressure (P_MBE)</td><td>{fmt_num_or_na(pp, '.2e', ' mbar')}</td></tr>"
        summary_html += f"<tr><td>Termination Duration</td><td>{_format_time_hms(dur) if isinstance(dur, (int, float)) else 'N/A'}</td></tr>"
        summary_html += "</table>"
    
    # Doses
    if summary.get('doses'):
        summary_html += "<h3>Doses</h3>"
        for i, dose in enumerate(summary['doses'], 1):
            dur = dose.get('duration')
            exp = dose.get('exposure')
            summary_html += f"<h4>Dose {i}</h4><table class='summary-table'>"
            summary_html += f"<tr><td>Type</td><td>{dose.get('gas_type', 'Unknown')}</td></tr>"
            summary_html += f"<tr><td>Time</td><td>{_format_time_hms(dur) if isinstance(dur, (int, float)) else 'N/A'}</td></tr>"
            summary_html += f"<tr><td>Exposure</td><td>{fmt_num_or_na(exp, '.2f', ' L')}</td></tr>"
            summary_html += "</table>"
    
    # Overgrowth (from SUSI/overgrowth file)
    if summary.get('overgrowth'):
        ovg = summary['overgrowth']
        total_dur = ovg.get('total_duration')
        total_nm = ovg.get('total_deposited_nm')
        summary_html += "<h3>Overgrowth</h3><table class='summary-table'>"
        summary_html += f"<tr><td>Total Duration</td><td>{_format_time_hms(total_dur) if isinstance(total_dur, (int, float)) else 'N/A'}</td></tr>"
        summary_html += f"<tr><td>Total Silicon Deposited</td><td>{fmt_num_or_na(total_nm, '.2f', ' nm')}</td></tr>"
        # RT
        rt = ovg.get('rt', {}) or {}
        if rt.get('detected'):
            summary_html += "<tr><td colspan='2'><strong>RT Growth (Locking Layer)</strong></td></tr>"
            summary_html += f"<tr><td>  Duration</td><td>{_format_time_hms(rt.get('duration_s')) if isinstance(rt.get('duration_s'), (int, float)) else 'N/A'}</td></tr>"
            if isinstance(rt.get('deposited_nm'), (int, float)):
                summary_html += f"<tr><td>  Deposited</td><td>{rt.get('deposited_nm'):.2f} nm</td></tr>"
            summary_html += f"<tr><td>  Current</td><td>{fmt_num_or_na(rt.get('median_current_A'), '.4f', ' A')}</td></tr>"
        # RTA
        rta = ovg.get('rta', {}) or {}
        if rta.get('detected'):
            summary_html += "<tr><td colspan='2'><strong>RTA Anneal</strong></td></tr>"
            summary_html += f"<tr><td>  Duration</td><td>{_format_time_hms(rta.get('duration_s')) if isinstance(rta.get('duration_s'), (int, float)) else 'N/A'}</td></tr>"
            summary_html += f"<tr><td>  Current</td><td>{fmt_num_or_na(rta.get('median_current_A'), '.4f', ' A')}</td></tr>"
            summary_html += f"<tr><td>  Temperature</td><td>{fmt_num_or_na(rta.get('median_temperature_C'), '.1f', ' °C')}</td></tr>"
        # LTE
        lte = ovg.get('lte', {}) or {}
        if lte.get('detected'):
            summary_html += "<tr><td colspan='2'><strong>LTE Growth</strong></td></tr>"
            summary_html += f"<tr><td>  Duration</td><td>{_format_time_hms(lte.get('duration_s')) if isinstance(lte.get('duration_s'), (int, float)) else 'N/A'}</td></tr>"
            if isinstance(lte.get('deposited_nm'), (int, float)):
                summary_html += f"<tr><td>  Deposited</td><td>{lte.get('deposited_nm'):.2f} nm</td></tr>"
            summary_html += f"<tr><td>  Current</td><td>{fmt_num_or_na(lte.get('median_current_A'), '.4f', ' A')}</td></tr>"
            summary_html += f"<tr><td>  Temperature</td><td>{fmt_num_or_na(lte.get('median_temperature_C'), '.1f', ' °C')}</td></tr>"
        summary_html += "</table>"
    
    summary_html += "</div>"
    
    # Build file sections
    files_html = "<div class='files-section'><h2>Process Files</h2>"
    for f in per_file:
        files_html += f"<div class='file-card'>"
        files_html += f"<h3>{f['filename']}</h3>"
        files_html += f"<div class='file-type'>Type: <strong>{f['file_type'].upper()}</strong></div>"
        files_html += f"<div class='file-status'>{f['status']}</div>"
        files_html += f"{f['metrics_html']}"
        
        # Plots
        if f['images']:
            files_html += "<div class='plots'>"
            for img in f['images']:
                files_html += f"<img src='data:image/png;base64,{img['b64']}' alt='Plot' />"
            files_html += "</div>"
        
        files_html += "</div>"
    files_html += "</div>"
    
    # Complete HTML
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8" />
    <title>Device Report: {device_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .header {{ background: white; padding: 20px; margin-bottom: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ margin: 0 0 10px 0; color: #333; }}
        .device-info {{ color: #666; font-size: 14px; margin: 5px 0; }}
        .summary-section {{ background: white; padding: 20px; margin-bottom: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .summary-section h2 {{ color: #2c3e50; margin-top: 0; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        .summary-section h3 {{ color: #34495e; margin-top: 20px; margin-bottom: 10px; font-size: 16px; }}
        .summary-section h4 {{ color: #7f8c8d; margin: 10px 0 5px 0; font-size: 14px; }}
        .summary-table {{ width: 100%; margin-bottom: 15px; border-collapse: collapse; }}
        .summary-table td {{ padding: 6px 10px; border-bottom: 1px solid #ecf0f1; }}
        .summary-table td:first-child {{ font-weight: 500; color: #555; width: 40%; }}
        .summary-table td:last-child {{ color: #333; }}
        .files-section h2 {{ color: #2c3e50; margin: 30px 0 20px 0; }}
        .file-card {{ background: white; padding: 20px; margin-bottom: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .file-card h3 {{ margin: 0 0 10px 0; color: #2c3e50; }}
        .file-type {{ display: inline-block; background: #3498db; color: white; padding: 4px 12px; border-radius: 4px; font-size: 12px; margin: 5px 0; }}
        .file-status {{ color: #27ae60; font-size: 12px; margin: 5px 0; }}
        .metric-section {{ margin: 20px 0; }}
        .metric-section h4 {{ color: #34495e; margin: 15px 0 10px 0; font-size: 15px; border-bottom: 1px solid #ecf0f1; padding-bottom: 5px; }}
        .metrics-table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
        .metrics-table td, .metrics-table th {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #ecf0f1; }}
        .metrics-table th {{ background: #f8f9fa; font-weight: 600; color: #555; }}
        .metrics-table td:first-child {{ font-weight: 500; color: #555; width: 40%; }}
        .plots {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 15px; margin-top: 20px; }}
        .plots img {{ width: 100%; border: 1px solid #ddd; border-radius: 4px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Device Characterization Report</h1>
        <div class="device-info"><strong>Device:</strong> {device_name}</div>
        <div class="device-info"><strong>Folder:</strong> {folder_path}</div>
        <div class="device-info"><strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>
        {f'<div class="device-info"><strong>Notes:</strong> {device_notes}</div>' if device_notes else ''}
    </div>
    
    {summary_html}
    {files_html}
</body>
</html>"""
    
    return html
