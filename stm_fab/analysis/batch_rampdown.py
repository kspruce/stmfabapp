# batchprocess_backend.py
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks

from stm_fab.labview.labview_parser import LabVIEWParser

# -----------------------
# Helpers and extraction
# -----------------------

def _col(df: pd.DataFrame, *cands: str) -> Optional[str]:
    """Return the first matching column name (case/underscore tolerant)."""
    norm = {c.lower().replace(' ', '_'): c for c in df.columns}
    for cand in cands:
        k = cand.lower().replace(' ', '_')
        if k in norm:
            return norm[k]
    return None

def _dynamic_savgol(series: np.ndarray) -> np.ndarray:
    n = len(series)
    if n < 7:
        return series
    # odd window <= min(51, n-1) and reasonably sized
    win = min(51, (n // 10) * 2 + 1)
    if win < 7: win = 7
    if win >= n: win = (n//2)*2 - 1 if (n//2)*2 - 1 >= 5 else 5
    return savgol_filter(series, window_length=win, polyorder=3)

def _filter_power_outliers(temperature: np.ndarray, power: np.ndarray, current: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Port of your outlier filtering (rolling median + targeted region)."""
    df = pd.DataFrame({'temperature': temperature, 'power': power, 'current': current})
    df = df.sort_values('temperature').copy()
    df = df[df['power'] > 0]  # remove zero/negative power
    # Targeted removal >15W and 400–500C
    df = df[~((df['power'] > 15) & (df['temperature'] > 400) & (df['temperature'] < 500))]

    # Rolling median/std
    w = 5
    df['power_median'] = df['power'].rolling(window=w, center=True).median()
    df['power_std'] = df['power'].rolling(window=w, center=True).std()
    df['power_median'] = df['power_median'].fillna(df['power'])
    df['power_std'] = df['power_std'].fillna(df['power'].std() if df['power'].std() > 0 else 1.0)

    # 3-sigma
    mask = np.abs(df['power'] - df['power_median']) <= 3 * df['power_std']
    df = df[mask]
    return df['temperature'].to_numpy(), df['power'].to_numpy(), df['current'].to_numpy()

def _closest_at_power(power: np.ndarray, temperature: np.ndarray, current: np.ndarray, target: float, tol_frac: float = 0.1):
    if len(power) == 0:
        return None, None
    idx = int(np.argmin(np.abs(power - target)))
    if np.abs(power[idx] - target) > tol_frac * target:
        return None, None
    return float(temperature[idx]), float(current[idx])

def _extract_date_and_sample(filename: str):
    """Optional: parse 'DD-MM-YYYY SAMPLENAME' from filename if present."""
    import re
    from datetime import datetime
    base = Path(filename).stem
    m = re.match(r'(\d{2}-\d{2}-\d{4})\s+(.+)', base)
    if m:
        try:
            return datetime.strptime(m.group(1), '%d-%m-%Y'), m.group(2)
        except:
            return None, base
    return None, base

# -----------------------
# Core per-file analysis
# -----------------------

def process_file(file_path: str) -> Dict[str, Any]:
    """Parse file with LabVIEWParser, isolate final ramp-down, compute metrics and figures later."""
    try:
        parser = LabVIEWParser(file_path)
        parsed = parser.parse()
        df = parsed['data']
        filename = Path(file_path).name
        date_obj, sample_name = _extract_date_and_sample(filename)

        # Get columns
        c_time = _col(df, 'Time', 'Time_s', 'X_Value')
        c_pow = _col(df, 'TDK P')  # matches original BatchProcess column index 1
        c_cur = _col(df, 'TDK I')  # index 5
        c_tmp = _col(df, 'Pyro T') # index 7

        if not all([c_time, c_pow, c_cur, c_tmp]):
            raise ValueError("Required columns not found (Time, TDK P, TDK I, Pyro T)")

        power_full = pd.to_numeric(df[c_pow], errors='coerce').to_numpy()
        current_full = pd.to_numeric(df[c_cur], errors='coerce').to_numpy()
        temperature_full = pd.to_numeric(df[c_tmp], errors='coerce').to_numpy()
        time_full = pd.to_numeric(df[c_time], errors='coerce').to_numpy()

        mask = np.isfinite(power_full) & np.isfinite(current_full) & np.isfinite(temperature_full) & np.isfinite(time_full)
        power_full, current_full, temperature_full, time_full = power_full[mask], current_full[mask], temperature_full[mask], time_full[mask]

        if len(power_full) < 10:
            raise ValueError("Too few valid data points")

        max_power = float(np.max(power_full))
        max_current = float(np.max(current_full))
        max_temp = float(np.max(temperature_full))

        # Smooth + peaks
        current_smooth = _dynamic_savgol(current_full)
        peak_height = 0.4 * np.nanmax(current_smooth) if np.nanmax(current_smooth) > 0 else None
        if peak_height and peak_height > 0:
            peaks, _ = find_peaks(current_smooth, height=peak_height)
        else:
            peaks = np.array([])

        start_idx = int(peaks[-1]) if len(peaks) else int(np.nanargmax(current_smooth))
        start_idx = max(0, min(start_idx, len(current_full)-1))

        peak_current = float(current_full[start_idx])
        power_at_peak = float(power_full[start_idx])
        temp_at_peak = float(temperature_full[start_idx])

        # Ramp-down slices
        power_ramp = power_full[start_idx:]
        current_ramp = current_full[start_idx:]
        temp_ramp = temperature_full[start_idx:]

        # Filter T < 900°C
        m_lt900 = temp_ramp < 900
        if not np.any(m_lt900):
            return {'filename': filename, 'sample_name': sample_name, 'date': date_obj, 'success': False,
                    'error': 'No data points < 900°C after ramp-down start'}

        power_f = power_ramp[m_lt900]
        current_f = current_ramp[m_lt900]
        temp_f = temp_ramp[m_lt900]

        # Outlier filtering on power
        temp_f, power_f, current_f = _filter_power_outliers(temp_f, power_f, current_f)

        # Markers at 2/3/4 W
        t2, i2 = _closest_at_power(power_f, temp_f, current_f, 2.0)
        t3, i3 = _closest_at_power(power_f, temp_f, current_f, 3.0)
        t4, i4 = _closest_at_power(power_f, temp_f, current_f, 4.0)

        return {
            'filename': filename,
            'sample_name': sample_name,
            'date': date_obj,
            'power': power_f,
            'current': current_f,
            'temperature': temp_f,
            'highest_current': peak_current,
            'power_at_max': power_at_peak,
            'temp_at_max': temp_at_peak,
            'max_power': max_power,
            'max_current': max_current,
            'max_temp': max_temp,
            'temp_at_2w': t2, 'current_at_2w': i2,
            'temp_at_3w': t3, 'current_at_3w': i3,
            'temp_at_4w': t4, 'current_at_4w': i4,
            'success': True
        }
    except Exception as e:
        return {'filename': Path(file_path).name, 'sample_name': Path(file_path).stem,
                'date': None, 'success': False, 'error': str(e)}

# -----------------------
# Figure generators
# -----------------------

def figure_per_file(result: Dict[str, Any]) -> plt.Figure:
    """Two-panel figure: Power vs Temp and Current vs Temp with markers."""
    if not result.get('success'):
        raise ValueError(f"Result not successful for {result.get('filename')}")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Power vs T
    ax1.plot(result['temperature'], result['power'], 'b-')
    ax1.set_title('Power vs. Temperature (Ramp-Down, T < 900°C)')
    ax1.set_xlabel('Temperature (°C)')
    ax1.set_ylabel('Power (W)')
    ax1.grid(True)

    # Markers
    if result['temp_at_2w'] is not None:
        ax1.plot(result['temp_at_2w'], 2.0, 'g*', markersize=10, label=f'2W ({result["temp_at_2w"]:.1f}°C)')
    if result['temp_at_3w'] is not None:
        ax1.plot(result['temp_at_3w'], 3.0, 'r*', markersize=10, label=f'3W ({result["temp_at_3w"]:.1f}°C)')
    if result['temp_at_4w'] is not None:
        ax1.plot(result['temp_at_4w'], 4.0, 'c*', markersize=10, label=f'4W ({result["temp_at_4w"]:.1f}°C)')
    ax1.legend()

    # Current vs T
    ax2.plot(result['temperature'], result['current'], 'r-')
    ax2.set_title('Current vs. Temperature (Ramp-Down, T < 900°C)')
    ax2.set_xlabel('Temperature (°C)')
    ax2.set_ylabel('Current (A)')
    ax2.grid(True)

    # Start of ramp-down marker
    ax2.plot(result['temp_at_max'], result['highest_current'], 'ko', markersize=8,
             label=f'Ramp Start ({result["highest_current"]:.3f} A)')
    if result['temp_at_2w'] is not None:
        ax2.plot(result['temp_at_2w'], result['current_at_2w'], 'g*', markersize=10)
    if result['temp_at_3w'] is not None:
        ax2.plot(result['temp_at_3w'], result['current_at_3w'], 'r*', markersize=10)
    if result['temp_at_4w'] is not None:
        ax2.plot(result['temp_at_4w'], result['current_at_4w'], 'c*', markersize=10)
    ax2.legend()

    fig.tight_layout()
    return fig

def _colorblind_palette():
    return ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e',
            '#e6ab02', '#a6761d', '#666666', '#332288', '#88CCEE',
            '#44AA99', '#117733', '#999933', '#DDCC77']

def figure_comparison(sorted_results: List[Dict[str,Any]], plot_type: str, temp_range: Optional[Tuple[float,float]]=None, title_suffix:str="") -> plt.Figure:
    """Multi-sample line plot (power or current vs temperature), full range or zoomed."""
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = _colorblind_palette()
    line_styles = ['-', '--', '-.', ':']
    markers = ['o','s','^','v','D','*','p','h','+','x']

    range_text = f"({temp_range[0]}°C < T < {temp_range[1]}°C)" if temp_range else "(Ramp-Down, T < 900°C)"
    fig.suptitle(f"{'Power' if plot_type=='power' else 'Current'} vs. Temperature Comparison {range_text}{title_suffix}", fontsize=16)

    count = 0
    for i, r in enumerate(sorted_results):
        if not r.get('success'):
            continue
        temp = r['temperature']
        data = r['power'] if plot_type == 'power' else r['current']
        if temp_range:
            m = (temp >= temp_range[0]) & (temp <= temp_range[1])
            if not np.any(m): 
                continue
            temp = temp[m]
            data = data[m]
        if len(temp) == 0:
            continue

        color = colors[i % len(colors)]
        ls = line_styles[(i // len(colors)) % len(line_styles)]
        mk = markers[i % len(markers)]
        markevery = max(1, len(temp)//10)
        label = f"{(r['date'].strftime('%d-%m-%Y')+' ') if r['date'] else ''}{r['sample_name']}"
        ax.plot(temp, data, color=color, linestyle=ls, marker=mk, markevery=markevery, markersize=6, label=label)
        count += 1

    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Power (W)" if plot_type == 'power' else "Current (A)")
    ax.grid(True)
    if temp_range:
        ax.set_xlim(temp_range)
    if count > 10:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout(rect=[0,0.03,0.85,0.95])
    else:
        ax.legend()
        plt.tight_layout(rect=[0,0.03,1,0.95])
    return fig

def figures_fixed_power_bars(sorted_results: List[Dict[str,Any]], power_level: float) -> Tuple[plt.Figure, plt.Figure]:
    """Bar charts: temperature at N W and current at N W across samples."""
    labels, dates, temps, currents = [], [], [], []
    for r in sorted_results:
        if not r.get('success'):
            continue
        tk = f"temp_at_{int(power_level)}w"
        ik = f"current_at_{int(power_level)}w"
        if r.get(tk) is None:
            continue
        label = f"{(r['date'].strftime('%d-%m-%Y')+' ') if r['date'] else ''}{r['sample_name']}"
        labels.append(label)
        dates.append(r['date'] or pd.Timestamp('1900-01-01').to_pydatetime())
        temps.append(r[tk])
        currents.append(r[ik])

    if not labels:
        return None, None

    idx = np.argsort(dates)
    labels = [labels[i] for i in idx]
    temps = [temps[i] for i in idx]
    currents = [currents[i] for i in idx]

    colors = plt.cm.viridis(np.linspace(0, 0.9, len(labels)))

    # Temperature bars
    figT, axT = plt.subplots(figsize=(12, 8))
    bars = axT.bar(range(len(labels)), temps, color=colors)
    for i, b in enumerate(bars):
        h = b.get_height()
        axT.text(b.get_x()+b.get_width()/2., h+5, f'{temps[i]:.1f}°C', ha='center', va='bottom', fontsize=9)
    axT.set_title(f'Temperature at {power_level}W Comparison', fontsize=16)
    axT.set_ylabel('Temperature (°C)')
    axT.set_xticks(range(len(labels)))
    axT.set_xticklabels(labels, rotation=45 if len(labels)>5 else 0, ha='right' if len(labels)>5 else 'center')
    axT.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Current bars
    figI, axI = plt.subplots(figsize=(12, 8))
    bars = axI.bar(range(len(labels)), currents, color=colors)
    for i, b in enumerate(bars):
        h = b.get_height()
        axI.text(b.get_x()+b.get_width()/2., h+0.02, f'{currents[i]:.3f}A', ha='center', va='bottom', fontsize=9)
    axI.set_title(f'Current at {power_level}W Comparison', fontsize=16)
    axI.set_ylabel('Current (A)')
    axI.set_xticks(range(len(labels)))
    axI.set_xticklabels(labels, rotation=45 if len(labels)>5 else 0, ha='right' if len(labels)>5 else 'center')
    axI.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    return figT, figI

# -----------------------
# Folder-level API
# -----------------------

def process_folder(input_directory: str) -> List[Dict[str,Any]]:
    files = list(Path(input_directory).glob("*.txt"))
    results = [process_file(str(fp)) for fp in files]
    return results

def generate_comparison_figures(results: List[Dict[str,Any]]) -> Dict[str, plt.Figure]:
    # Chronological sort (Unknown dates last)
    sorted_results = sorted(results, key=lambda x: (x['date'] is None, x['date'] or pd.Timestamp.max.to_pydatetime()))
    figs = {}

    figs['power_full'] = figure_comparison(sorted_results, 'power')
    figs['current_full'] = figure_comparison(sorted_results, 'current')

    figs['power_zoom_300_450'] = figure_comparison(sorted_results, 'power', temp_range=(300, 450), title_suffix=" - Zoomed View")
    figs['current_zoom_300_450'] = figure_comparison(sorted_results, 'current', temp_range=(300, 450), title_suffix=" - Zoomed View")

    for W in (2.0, 3.0, 4.0):
        fT, fI = figures_fixed_power_bars(sorted_results, W)
        if fT is not None:
            figs[f'temp_at_{int(W)}W'] = fT
            figs[f'current_at_{int(W)}W'] = fI
    return figs

def export_summary_excel(results: List[Dict[str,Any]], output_path: str):
    """Write a multi-sheet Excel similar to your original BatchProcess."""
    summary_rows = []
    for r in results:
        if not r.get('success'):
            continue
        summary_rows.append({
            'Date': r['date'].strftime('%Y-%m-%d') if r['date'] else 'Unknown',
            'Sample Name': r['sample_name'],
            'Current at 2W (A)': r['current_at_2w'] if r['current_at_2w'] is not None else 'N/A',
            'Temperature at 2W (C)': r['temp_at_2w'] if r['temp_at_2w'] is not None else 'N/A',
            'Current at 3W (A)': r['current_at_3w'] if r['current_at_3w'] is not None else 'N/A',
            'Temperature at 3W (C)': r['temp_at_3w'] if r['temp_at_3w'] is not None else 'N/A',
            'Current at 4W (A)': r['current_at_4w'] if r['current_at_4w'] is not None else 'N/A',
            'Temperature at 4W (C)': r['temp_at_4w'] if r['temp_at_4w'] is not None else 'N/A',
            'Max Current (A)': r['max_current'],
            'Max Temperature (C)': r['max_temp'],
            'Max Power (W)': r['max_power'],
        })

    if not summary_rows:
        raise ValueError("No successful results to export")

    excel_path = Path(output_path)
    excel_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(str(excel_path), engine='openpyxl') as writer:
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name='Summary', index=False)

        # Detailed sheets per file
        # Keep chronological order
        sorted_results = sorted(results, key=lambda x: (x['date'] is None, x['date'] or pd.Timestamp.max.to_pydatetime()))
        for r in sorted_results:
            if not r.get('success'):
                continue
            detail = pd.DataFrame({
                'Temperature (C)': r['temperature'],
                'Power (W)': r['power'],
                'Current (A)': r['current']
            })
            sheet = ((r['date'].strftime('%Y-%m-%d') + ' ') if r['date'] else '') + r['sample_name']
            detail.to_excel(writer, sheet_name=sheet[:31], index=False)
