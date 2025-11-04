# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 13:56:54 2025
Updated on Tue Jul 22, 2025 to intelligently find sample names within folder names.

@author: kspruce
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
import numpy as np
from matplotlib.ticker import FuncFormatter

# --- Parameter Database ---
SAMPLE_PARAMS = {
    'amersham': {'damping': 5805.23, 'amplitude': 0.00729211, 'offset': 1.86249},
    'anerly': {'damping': 5150.07, 'amplitude': 0.0211297, 'offset': 0.404944},
    'archway': {'damping': 5137.52, 'amplitude': 0.0154253, 'offset': 0.454775},
    'bermondsey': {'damping': 6413.15, 'amplitude': 0.00316631, 'offset': 2.07127},
    'croxley': {'damping': 4971.48, 'amplitude': 0.0400215, 'offset': -0.728179},
    'denmark_hill': {'damping': 4964.75, 'amplitude': 0.0354644, 'offset': -0.815681},
    'fulham_broadway': {'damping': 4985.62, 'amplitude': 0.0192559, 'offset': 0.134058},
    'gospel_oak': {'damping': 4947.78, 'amplitude': 0.0461436, 'offset': -1.02412},
    'kilburn': {'damping': 5143.8, 'amplitude': 0.0249835, 'offset': -0.359991},
    'kingsbury': {'damping': 4612.45, 'amplitude': 0.050053, 'offset': -0.955029},
    'london_bridge': {'damping': 5402.72, 'amplitude': 0.0112197, 'offset': 1.0664},
    'neasden': {'damping': 5039.32, 'amplitude': 0.0312678, 'offset': -0.0224321},
    'langley': {'damping': 4773.31, 'amplitude': 0.0486393, 'offset': -1.12888},
    'nine_elms': {'damping': 4746.05, 'amplitude': 0.0372163, 'offset': -0.513126},
    'parsons_green': {'damping': 4921.23, 'amplitude': 0.0197741, 'offset': 0.154148},
    'poplar': {'damping': 4978.16, 'amplitude': 0.0212957, 'offset': 0.382278},
    'putney_bridge': {'damping': 4961.88, 'amplitude': 0.0213655, 'offset': 0.37462},
    'slough': {'damping': 5190.33, 'amplitude': 0.0283826, 'offset': -0.193577}
}

# --- Helper Functions ---

def plot_and_save(df, time_col, data_col, y_label, title, base_filename, output_dir, timestamp, use_log_scale=False):
    """A generic function to create, save, and close a single plot."""
    try:
        if not all(col in df.columns for col in [time_col, data_col]):
            raise KeyError(f"Missing one of required columns: {time_col}, {data_col}")

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df[time_col], df[data_col], label=y_label)
        
        ax.set_title(title)
        ax.set_xlabel('Time (min)')
        ax.set_ylabel(y_label)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        if use_log_scale:
            ax.set_yscale('log')
        
        def seconds_to_minutes(x, pos):
            return f'{x / 60:.1f}'
        ax.xaxis.set_major_formatter(FuncFormatter(seconds_to_minutes))
        
        clean_title = title.replace(' ', '_').replace('(', '').replace(')', '')
        filename = f"{base_filename}_{clean_title}_{timestamp}.png"
        save_path = os.path.join(output_dir, filename)
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  -> Saved to {os.path.relpath(save_path)}")

    except KeyError as e:
        print(f"  - Skipped '{title}' plot for {base_filename}: {e}")
    except Exception as e:
        print(f"  - An error occurred while creating '{title}' plot for {base_filename}: {e}")

def trim_idle_data(df, filter_params):
    """Trims trailing idle data based on the last time a current threshold was exceeded."""
    if not filter_params['trim'] or 'TDK_I' not in df.columns:
        return df

    threshold = filter_params['threshold']
    padding = filter_params['padding']
    
    active_indices = df.index[df['TDK_I'] > threshold].tolist()
    
    if not active_indices:
        print("  - Warning: No active current found above threshold. Skipping trim.")
        return df
        
    last_active_index = active_indices[-1]
    last_active_time = df.loc[last_active_index, 'Time_s']
    
    cutoff_time = last_active_time + padding
    
    original_rows = len(df)
    df_trimmed = df[df['Time_s'] <= cutoff_time].copy()
    trimmed_rows = len(df_trimmed)
    
    print(f"  - Data trimmed from {original_rows} to {trimmed_rows} rows (cutoff at {cutoff_time:.1f}s).")
    return df_trimmed

# --- Main Analysis Function ---

def analyze_data(file_path, analysis_type, output_dir, timestamp, temp_params, filter_params):
    """
    Reads, analyzes, trims, and exports plots based on the chosen analysis type.
    """
    try:
        # --- 1. Find header and read data ---
        header_end_count = 0
        header_lines_to_skip = 0
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if '***End_of_Header***' in line:
                    header_end_count += 1
                if header_end_count == 2:
                    header_lines_to_skip = i + 1
                    break
        
        if header_lines_to_skip == 0:
            print(f"Error: Could not find data start in {file_path}.")
            return

        df = pd.read_csv(file_path, sep='\t', skiprows=header_lines_to_skip, decimal='.')

        # --- 2. Data Cleaning and Preparation ---
        df.columns = [col.strip().replace(' ', '_') for col in df.columns]
        df.rename(columns={'X_Value': 'Time_s'}, inplace=True)

        cols_to_convert = ['Time_s', 'Pyro_T', 'P_MBE', 'P_VT', 'TDK_I', 'TDK_V', 'TDK_I_1', 'TDK_P_1']
        
        for col in cols_to_convert:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df.dropna(subset=[col for col in df.columns if col in cols_to_convert], inplace=True)
        
        base_filename = os.path.splitext(os.path.basename(file_path))[0]
        print(f"\n>>> Processing '{base_filename}' for '{analysis_type}' analysis...")

        # --- 3. Trim trailing idle data ---
        df = trim_idle_data(df, filter_params)

        # --- 4. Optional: Calculate Estimated Temperature ---
        if temp_params.get('calculate', False):
            if 'TDK_V' in df.columns and 'TDK_I' in df.columns:
                df['Estimated_Temp'] = np.nan
                zero_current_mask = df['TDK_I'] == 0
                df.loc[zero_current_mask, 'Estimated_Temp'] = 298.15
                
                active_current_mask = df['TDK_I'] > 0
                if active_current_mask.any():
                    resistance = df.loc[active_current_mask, 'TDK_V'] / df.loc[active_current_mask, 'TDK_I']
                    log_arg = (resistance - temp_params['offset']) / temp_params['amplitude']
                    valid_log_mask = log_arg > 0
                    if valid_log_mask.any():
                        calculated_temps = temp_params['damping'] / np.log(log_arg[valid_log_mask])
                        df.loc[calculated_temps.index, 'Estimated_Temp'] = calculated_temps
                
                print("  - Estimated Temperature calculated (with room temp for zero current).")
            else:
                print("  - Warning: TDK_V or TDK_I columns not found. Cannot calculate temperature.")

        # --- 5. Generate and Save Plots Based on Analysis Type ---
        if analysis_type == 'Flash':
            plot_and_save(df, 'Time_s', 'Pyro_T', 'Pyro Temp (°C)', 'Pyro vs Time', base_filename, output_dir, timestamp)
            plot_and_save(df, 'Time_s', 'P_MBE', 'Pressure (mbar)', 'P MBE vs Time', base_filename, output_dir, timestamp, use_log_scale=True)
            plot_and_save(df, 'Time_s', 'TDK_I', 'Sample Current (A)', 'Sample Current vs Time', base_filename, output_dir, timestamp)
            if temp_params.get('calculate', False) and 'Estimated_Temp' in df.columns:
                plot_and_save(df, 'Time_s', 'Estimated_Temp', 'Estimated Temp (K)', 'Estimated Temperature vs Time', base_filename, output_dir, timestamp)

        elif analysis_type == 'Termination':
            plot_and_save(df, 'Time_s', 'TDK_I', 'Sample Current (A)', 'Sample Current vs Time', base_filename, output_dir, timestamp)
            plot_and_save(df, 'Time_s', 'P_MBE', 'Pressure (mbar)', 'P MBE vs Time', base_filename, output_dir, timestamp, use_log_scale=True)
            if temp_params.get('calculate', False) and 'Estimated_Temp' in df.columns:
                plot_and_save(df, 'Time_s', 'Estimated_Temp', 'Estimated Temp (K)', 'Estimated Temperature vs Time', base_filename, output_dir, timestamp)

        elif analysis_type == 'SUSI':
            plot_and_save(df, 'Time_s', 'TDK_I', 'Sample Current (A)', 'Sample Current vs Time', base_filename, output_dir, timestamp)
            plot_and_save(df, 'Time_s', 'TDK_I_1', 'SUSI Current (A)', 'SUSI Current vs Time', base_filename, output_dir, timestamp)
            plot_and_save(df, 'Time_s', 'TDK_P_1', 'SUSI Power (W)', 'SUSI Power vs Time', base_filename, output_dir, timestamp)
            plot_and_save(df, 'Time_s', 'P_MBE', 'Pressure (mbar)', 'P MBE vs Time', base_filename, output_dir, timestamp, use_log_scale=True)
            if temp_params.get('calculate', False) and 'Estimated_Temp' in df.columns:
                plot_and_save(df, 'Time_s', 'Estimated_Temp', 'Estimated Temp (K)', 'Estimated Temperature vs Time', base_filename, output_dir, timestamp)
        
        elif analysis_type == 'Dose':
            plot_and_save(df, 'Time_s', 'P_VT', 'VT Pressure (mbar)', 'VT Pressure vs Time', base_filename, output_dir, timestamp, use_log_scale=True)
            
        elif analysis_type == 'Outgas':
            plot_and_save(df, 'Time_s', 'P_MBE', 'Pressure (mbar)', 'P MBE vs Time', base_filename, output_dir, timestamp, use_log_scale=True)

    except Exception as e:
        print(f"An unexpected error occurred while processing {file_path}: {e}")

def get_user_choice_for_file(filename):
    """Prompts the user to select an analysis type for a specific file."""
    print(f"\n[!] Could not automatically determine analysis type for '{filename}'.")
    print("    Please choose an analysis type for this file:")
    print("    1: Flash Analysis")
    print("    2: Termination Analysis")
    print("    3: SUSI Analysis")
    print("    4: Dose Analysis")
    print("    5: Outgas Analysis")
    print("    6: Skip this file")
    
    analysis_map = {'1': 'Flash', '2': 'Termination', '3': 'SUSI', '4': 'Dose', '5': 'Outgas', '6': 'Skip'}
    choice = ''
    while choice not in analysis_map:
        choice = input("    Enter your choice (1-6): ")
        if choice not in analysis_map:
            print("    Invalid input. Please enter 1, 2, 3, 4, 5, or 6.")
    return analysis_map[choice]

def get_temp_params_for_folder(folder_path):
    """Prompts user for temperature calculation settings for an unrecognized folder."""
    print(f"\n--- Configuring Temperature Estimation for UNKNOWN folder: '{os.path.relpath(folder_path)}' ---")
    
    params = {'calculate': False}
    
    choice_temp = input("Calculate estimated temperature for files in this folder? [Y/n]: ").lower().strip()
    if choice_temp == '' or choice_temp == 'y':
        params['calculate'] = True
        try:
            print("    Please provide the parameters for this sample.")
            params['damping'] = float(input("    Enter Damping value: "))
            params['offset'] = float(input("    Enter Offset value: "))
            params['amplitude'] = float(input("    Enter Amplitude value: "))
        except ValueError:
            print("    Invalid input. Temperature calculation will be skipped for this folder.")
            params['calculate'] = False
    return params

# --- Main Execution Block ---
if __name__ == '__main__':
    # --- Setup for export ---
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_output_dir = 'exports'
    os.makedirs(base_output_dir, exist_ok=True)
    print(f"Output will be saved to the '{base_output_dir}' folder with timestamp '{timestamp}'.")

    # --- Prompt for Data Trimming (once for the whole run) ---
    filter_params = {'trim': False, 'threshold': 0.05, 'padding': 20.0}
    choice_trim = input("\nDo you want to automatically trim idle data from the end of files? [Y/n]: ").lower().strip()
    if choice_trim == '' or choice_trim == 'y':
        filter_params['trim'] = True
        print(f"\nDefault trimming parameters: Current Threshold = {filter_params['threshold']} A, Padding = {filter_params['padding']} s.")
        choice_custom_trim = input("Use custom trimming parameters? [y/N]: ").lower().strip()
        if choice_custom_trim == 'y':
            try:
                filter_params['threshold'] = float(input("Enter current threshold (A): "))
                filter_params['padding'] = float(input("Enter padding time (s): "))
            except ValueError:
                print("Invalid input. Using default trimming parameters.")

    # --- Recursively find and process all .txt files ---
    current_directory = '.'
    files_found = False
    folder_temp_params = {}

    for root, dirs, files in os.walk(current_directory):
        if base_output_dir in dirs:
            dirs.remove(base_output_dir)
            
        txt_files_in_dir = [f for f in files if f.endswith('.txt')]
        if not txt_files_in_dir:
            continue

        # --- Per-Folder Configuration ---
        if root not in folder_temp_params:
            found_sample_key = None
            folder_name_lower = os.path.basename(root).lower()
            
            # Check if any known sample name is in the folder name
            for sample_key in SAMPLE_PARAMS.keys():
                search_term = sample_key.replace('_', ' ')
                if search_term in folder_name_lower:
                    found_sample_key = sample_key
                    break
            
            if found_sample_key:
                print(f"\n--- Found and loaded parameters for sample '{found_sample_key}' in folder '{os.path.relpath(root)}' ---")
                params = SAMPLE_PARAMS[found_sample_key].copy()
                params['calculate'] = True
                folder_temp_params[root] = params
            else:
                folder_temp_params[root] = get_temp_params_for_folder(root)
        
        current_temp_params = folder_temp_params[root]

        for filename in txt_files_in_dir:
            files_found = True
            analysis_type = None
            lower_filename = filename.lower()

            # Rule-based analysis selection
            if 'flash' in lower_filename or 'cooldown' in lower_filename:
                analysis_type = 'Flash'
            elif 'term' in lower_filename:
                analysis_type = 'Termination'
            elif any(keyword in lower_filename for keyword in ['susi', 'ovg', 'overgrowth']):
                analysis_type = 'SUSI'
            elif 'dose' in lower_filename:
                analysis_type = 'Dose'
            elif 'outgas' in lower_filename:
                analysis_type = 'Outgas'
            
            if analysis_type is None:
                analysis_type = get_user_choice_for_file(filename)

            if analysis_type == 'Skip':
                print(f"  -> Skipping file: {filename}")
                continue

            relative_path = os.path.relpath(root, current_directory)
            output_subdir = os.path.join(base_output_dir, relative_path)
            os.makedirs(output_subdir, exist_ok=True)

            full_path = os.path.join(root, filename)
            analyze_data(full_path, analysis_type, output_subdir, timestamp, current_temp_params, filter_params)

    if not files_found:
        print("\nNo .txt files found in the current directory or its subdirectories.")

    print("\n>>> All files processed.")
