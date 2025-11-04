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
    if seconds is None or seconds < 0:
        return "00:00:00"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _format_time_ms(seconds: float) -> str:
    """Format seconds as MM:SS"""
    if seconds is None or seconds < 0:
        return "00:00"
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"


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
        metrics = entry.get('metrics', {})
        
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
        
        elif ftype == 'hterm':
            summary['hterm'] = {
                'peak_pressure': metrics.get('peak_pressure_mbar'),
                'duration': metrics.get('total_duration_s')
            }
        
        elif ftype == 'dose':
            summary['doses'].append({
                'gas_type': metrics.get('gas_type', 'Unknown'),
                'duration': metrics.get('dose_duration_s'),
                'exposure': metrics.get('exposure_langmuirs')
            })
        
        elif ftype in ['inc', 'incorporation']:
            summary['incorporation'] = {
                'temperature': metrics.get('median_temperature_C'),
                'duration': metrics.get('duration_s')
            }
        
        elif ftype == 'susi':
            ovg = metrics.get('overgrowth', {})
            rt = ovg.get('locking_layer', {}) if isinstance(ovg, dict) else {}
            rta = ovg.get('rta', {}) if isinstance(ovg, dict) else {}
            lte = ovg.get('lte', {}) if isinstance(ovg, dict) else {}
            
            total_si = 0
            if rt.get('deposited_nm'): total_si += rt.get('deposited_nm', 0)
            if lte.get('deposited_nm'): total_si += lte.get('deposited_nm', 0)
            
            summary['overgrowth'] = {
                'susi_current': metrics.get('mean_operating_current_A'),
                'susi_time': metrics.get('total_operating_time_s'),
                'total_silicon_nm': total_si,
                'rt': rt,
                'rta': rta,
                'lte': lte
            }
    
    return summary


def _render_outgas_section(metrics: Dict[str, Any]) -> str:
    """Render outgas file section"""
    if not metrics:
        return "<p><em>No data</em></p>"
    
    base_p = metrics.get('base_pressure_mbar', 'N/A')
    peak_p = metrics.get('peak_pressure_mbar', 'N/A')
    stab_time = metrics.get('time_to_stabilize_s')
    
    html = "<div class='metric-section'>"
    html += "<h4>Key Metrics</h4>"
    html += "<table class='metrics-table'>"
    html += f"<tr><td>Base Pressure</td><td>{base_p:.2e} mbar</td></tr>" if isinstance(base_p, float) else f"<tr><td>Base Pressure</td><td>{base_p}</td></tr>"
    html += f"<tr><td>Peak Pressure</td><td>{peak_p:.2e} mbar</td></tr>" if isinstance(peak_p, float) else f"<tr><td>Peak Pressure</td><td>{peak_p}</td></tr>"
    html += f"<tr><td>Time to Stabilize</td><td>{_format_time_ms(stab_time)}</td></tr>" if stab_time else "<tr><td>Time to Stabilize</td><td>N/A</td></tr>"
    html += "</table>"
    html += "</div>"
    
    return html


def _render_flash_section(metrics: Dict[str, Any]) -> str:
    """Render flash file section"""
    if not metrics:
        return "<p><em>No data</em></p>"
    
    flash_count = metrics.get('flash_count', 0)
    total_time = metrics.get('total_flash_time_s', 0)
    peak_temp = metrics.get('peak_temperature_C', 'N/A')
    flashes = metrics.get('flashes', [])
    
    html = "<div class='metric-section'>"
    html += "<h4>Key Metrics</h4>"
    html += "<table class='metrics-table'>"
    html += f"<tr><td>Flash Count</td><td>{flash_count}</td></tr>"
    html += f"<tr><td>Total Flash Time</td><td>{_format_time_hms(total_time)}</td></tr>"
    html += f"<tr><td>Peak Temperature</td><td>{peak_temp:.1f} °C</td></tr>" if isinstance(peak_temp, (int, float)) else f"<tr><td>Peak Temperature</td><td>{peak_temp}</td></tr>"
    html += "</table>"
    
    # Individual flashes table
    if flashes:
        html += "<h4>Individual Flashes</h4>"
        html += "<table class='metrics-table'>"
        html += "<tr><th>Flash #</th><th>Duration (s)</th><th>Peak Temp (°C)</th></tr>"
        for i, flash in enumerate(flashes, 1):
            dur = flash.get('duration_s', 0)
            temp = flash.get('peak_temp_C', 'N/A')
            temp_str = f"{temp:.1f}" if isinstance(temp, (int, float)) else str(temp)
            html += f"<tr><td>{i}</td><td>{dur:.1f}</td><td>{temp_str}</td></tr>"
        html += "</table>"
    
    html += "</div>"
    return html


def _render_hterm_section(metrics: Dict[str, Any]) -> str:
    """Render hydrogen termination section"""
    if not metrics:
        return "<p><em>No data</em></p>"
    
    base_p = metrics.get('base_pressure_mbar', 'N/A')
    peak_p = metrics.get('peak_pressure_mbar', 'N/A')
    duration = metrics.get('total_duration_s')
    
    html = "<div class='metric-section'>"
    html += "<h4>Key Metrics</h4>"
    html += "<table class='metrics-table'>"
    html += f"<tr><td>Base Pressure</td><td>{base_p:.2e} mbar</td></tr>" if isinstance(base_p, float) else f"<tr><td>Base Pressure</td><td>{base_p}</td></tr>"
    html += f"<tr><td>Peak Pressure</td><td>{peak_p:.2e} mbar</td></tr>" if isinstance(peak_p, float) else f"<tr><td>Peak Pressure</td><td>{peak_p}</td></tr>"
    html += f"<tr><td>Dose Duration</td><td>{_format_time_hms(duration)}</td></tr>" if duration else "<tr><td>Dose Duration</td><td>N/A</td></tr>"
    html += "</table>"
    html += "</div>"
    
    return html


def _render_dose_section(metrics: Dict[str, Any]) -> str:
    """Render dose section"""
    if not metrics:
        return "<p><em>No data</em></p>"
    
    gas_type = metrics.get('gas_type', 'Unknown')
    base_p = metrics.get('base_pressure_mbar', 'N/A')
    peak_p = metrics.get('peak_pressure_mbar', 'N/A')
    duration = metrics.get('dose_duration_s')
    exposure = metrics.get('exposure_langmuirs', 'N/A')
    integrated = metrics.get('integrated_dose_molecules', 'N/A')
    flux = metrics.get('mean_flux_molecules_cm2_s', 'N/A')
    
    html = "<div class='metric-section'>"
    html += "<h4>Key Metrics</h4>"
    html += "<table class='metrics-table'>"
    html += f"<tr><td>Gas Type</td><td>{gas_type}</td></tr>"
    html += f"<tr><td>Base Pressure</td><td>{base_p:.2e} mbar</td></tr>" if isinstance(base_p, float) else f"<tr><td>Base Pressure</td><td>{base_p}</td></tr>"
    html += f"<tr><td>Peak Pressure</td><td>{peak_p:.2e} mbar</td></tr>" if isinstance(peak_p, float) else f"<tr><td>Peak Pressure</td><td>{peak_p}</td></tr>"
    html += f"<tr><td>Dose Duration</td><td>{_format_time_ms(duration)}</td></tr>" if duration else "<tr><td>Dose Duration</td><td>N/A</td></tr>"
    html += f"<tr><td>Exposure</td><td>{exposure:.2f} L</td></tr>" if isinstance(exposure, (int, float)) else f"<tr><td>Exposure</td><td>{exposure}</td></tr>"
    html += f"<tr><td>Integrated Dose</td><td>{integrated:.2e} molecules</td></tr>" if isinstance(integrated, float) else f"<tr><td>Integrated Dose</td><td>{integrated}</td></tr>"
    html += f"<tr><td>Mean Flux</td><td>{flux:.2e} molec/cm²/s</td></tr>" if isinstance(flux, float) else f"<tr><td>Mean Flux</td><td>{flux}</td></tr>"
    html += "</table>"
    html += "</div>"
    
    return html


def _render_overgrowth_section(metrics: Dict[str, Any]) -> str:
    """Render overgrowth (SUSI) section"""
    if not metrics:
        return "<p><em>No data</em></p>"
    
    ovg = metrics.get('overgrowth', {})
    if not isinstance(ovg, dict):
        return "<p><em>No overgrowth data</em></p>"
    
    rt = ovg.get('locking_layer', {})
    rta = ovg.get('rta', {})
    lte = ovg.get('lte', {})
    
    html = "<div class='metric-section'>"
    html += "<h4>Deposition Rate</h4>"
    html += "<table class='metrics-table'>"
    
    rate = rt.get('deposition_rate_ML_min') or lte.get('deposition_rate_ML_min')
    html += f"<tr><td>Rate</td><td>{rate:.3f} ML/min</td></tr>" if rate else "<tr><td>Rate</td><td>N/A</td></tr>"
    html += "</table>"
    
    # SUSI metrics
    html += "<h4>SUSI Operation</h4>"
    html += "<table class='metrics-table'>"
    susi_current = metrics.get('peak_current_A', 'N/A')
    susi_time = metrics.get('total_operating_time_s')
    html += f"<tr><td>Peak Current</td><td>{susi_current:.3f} A</td></tr>" if isinstance(susi_current, (int, float)) else f"<tr><td>Peak Current</td><td>{susi_current}</td></tr>"
    html += f"<tr><td>Total Operating Time</td><td>{_format_time_hms(susi_time)}</td></tr>" if susi_time else "<tr><td>Total Operating Time</td><td>N/A</td></tr>"
    html += "</table>"
    
    # RT Growth
    if rt.get('detected'):
        html += "<h4>RT Growth (Locking Layer)</h4>"
        html += "<table class='metrics-table'>"
        html += f"<tr><td>Duration</td><td>{_format_time_hms(rt.get('duration_s', 0))}</td></tr>"
        html += f"<tr><td>Deposited</td><td>{rt.get('deposited_ML', 0):.2f} ML ({rt.get('deposited_nm', 0):.3f} nm)</td></tr>"
        html += "</table>"
    
    # RTA Anneal
    if rta.get('detected'):
        html += "<h4>RTA Anneal</h4>"
        html += "<table class='metrics-table'>"
        html += f"<tr><td>Duration</td><td>{rta.get('duration_s', 0):.1f} s</td></tr>"
        html += f"<tr><td>Max Current</td><td>{rta.get('max_current_A', 0):.3f} A</td></tr>"
        
        temp_c = rta.get('median_temperature_C')
        if temp_c:
            html += f"<tr><td>Temperature</td><td>{temp_c:.1f} °C</td></tr>"
        else:
            html += "<tr><td>Temperature</td><td>N/A (no calibration)</td></tr>"
        html += "</table>"
    
    # LTE
    if lte.get('detected'):
        html += "<h4>LTE Growth</h4>"
        html += "<table class='metrics-table'>"
        html += f"<tr><td>Total Duration</td><td>{_format_time_hms(lte.get('duration_s', 0))}</td></tr>"
        
        mean_i = lte.get('mean_current_A')
        html += f"<tr><td>Mean Current</td><td>{mean_i:.2f} A</td></tr>" if mean_i else "<tr><td>Mean Current</td><td>N/A</td></tr>"
        
        temp_c = lte.get('median_temperature_C')
        if temp_c:
            html += f"<tr><td>Temperature</td><td>{temp_c:.1f} °C</td></tr>"
        else:
            html += "<tr><td>Temperature</td><td>N/A (no calibration)</td></tr>"
        
        html += f"<tr><td>Total Deposited</td><td>{lte.get('deposited_ML', 0):.2f} ML ({lte.get('deposited_nm', 0):.3f} nm)</td></tr>"
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
        elif ftype == 'hterm':
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
    
    # Build summary section
    summary_html = "<div class='summary-section'>"
    summary_html += "<h2>Fabrication Summary</h2>"
    
    # Outgas
    if summary['outgas']:
        summary_html += "<h3>Outgas</h3><table class='summary-table'>"
        og = summary['outgas']
        bp = og.get('base_pressure')
        summary_html += f"<tr><td>Base Pressure</td><td>{bp:.2e} mbar</td></tr>" if bp else "<tr><td>Base Pressure</td><td>N/A</td></tr>"
        tts = og.get('time_to_stabilize')
        summary_html += f"<tr><td>Time to Stabilize</td><td>{_format_time_hms(tts)}</td></tr>" if tts else "<tr><td>Time to Stabilize</td><td>N/A</td></tr>"
        summary_html += "</table>"
    
    # Flash
    if summary['flash']:
        summary_html += "<h3>Flash</h3><table class='summary-table'>"
        fl = summary['flash']
        summary_html += f"<tr><td>Total Flash Count</td><td>{fl.get('flash_count', 0)}</td></tr>"
        summary_html += f"<tr><td>Total Flash Time</td><td>{_format_time_hms(fl.get('total_time', 0))}</td></tr>"
        pt = fl.get('peak_temp')
        summary_html += f"<tr><td>Peak Flash Temp</td><td>{pt:.1f} °C</td></tr>" if pt else "<tr><td>Peak Flash Temp</td><td>N/A</td></tr>"
        summary_html += "</table>"
    
    # Termination
    if summary['hterm']:
        summary_html += "<h3>H-Termination</h3><table class='summary-table'>"
        ht = summary['hterm']
        pp = ht.get('peak_pressure')
        summary_html += f"<tr><td>Peak Pressure (P_MBE)</td><td>{pp:.2e} mbar</td></tr>" if pp else "<tr><td>Peak Pressure</td><td>N/A</td></tr>"
        dur = ht.get('duration')
        summary_html += f"<tr><td>Termination Duration</td><td>{_format_time_hms(dur)}</td></tr>" if dur else "<tr><td>Duration</td><td>N/A</td></tr>"
        summary_html += "</table>"
    
    # Doses
    if summary['doses']:
        summary_html += "<h3>Doses</h3>"
        for i, dose in enumerate(summary['doses'], 1):
            summary_html += f"<h4>Dose {i}</h4><table class='summary-table'>"
            summary_html += f"<tr><td>Type</td><td>{dose.get('gas_type', 'Unknown')}</td></tr>"
            dur = dose.get('duration')
            summary_html += f"<tr><td>Time</td><td>{_format_time_hms(dur)}</td></tr>" if dur else "<tr><td>Time</td><td>N/A</td></tr>"
            exp = dose.get('exposure')
            summary_html += f"<tr><td>Exposure</td><td>{exp:.2f} L</td></tr>" if exp else "<tr><td>Exposure</td><td>N/A</td></tr>"
            summary_html += "</table>"
    
    # Incorporation
    if summary['incorporation']:
        summary_html += "<h3>Incorporation</h3><table class='summary-table'>"
        inc = summary['incorporation']
        temp = inc.get('temperature')
        summary_html += f"<tr><td>Temperature</td><td>{temp:.1f} °C</td></tr>" if temp else "<tr><td>Temperature</td><td>N/A</td></tr>"
        dur = inc.get('duration')
        summary_html += f"<tr><td>Time</td><td>{dur:.1f} s</td></tr>" if dur else "<tr><td>Time</td><td>N/A</td></tr>"
        summary_html += "</table>"
    
    # Overgrowth
    if summary['overgrowth']:
        summary_html += "<h3>Overgrowth</h3><table class='summary-table'>"
        ovg = summary['overgrowth']
        
        curr = ovg.get('susi_current')
        summary_html += f"<tr><td>SUSI Operating Current</td><td>{curr:.3f} A</td></tr>" if curr else "<tr><td>SUSI Current</td><td>N/A</td></tr>"
        
        time = ovg.get('susi_time')
        summary_html += f"<tr><td>SUSI Operating Time</td><td>{_format_time_hms(time)}</td></tr>" if time else "<tr><td>SUSI Time</td><td>N/A</td></tr>"
        
        si = ovg.get('total_silicon_nm')
        summary_html += f"<tr><td>Total Silicon Deposited</td><td>{si:.3f} nm</td></tr>" if si else "<tr><td>Silicon Deposited</td><td>N/A</td></tr>"
        
        summary_html += "</table>"
    
    summary_html += "</div>"
    
    # Build file sections
    files_html = "<div class='files-section'><h2>Process Files</h2>"
    for f in per_file:
        files_html += f"<div class='file-card'>"
        files_html += f"<h3>{f['filename']}</h3>"
        files_html += f"<div class='file-type'>Type: <strong>{f['file_type'].upper()}</strong></div>"
        files_html += f"<div class='file-status'>{f['status']}</div>"
        files_html += f['metrics_html']
        
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