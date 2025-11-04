"""
labview_parser_fixed.py - Fixed LabVIEW parser for your specific file format

Corrected to handle the actual column structure in your files:
X_Value, TDK P, P MBE, P VT, TDK V, TDK I, TDK R, Pyro T, Untitled, TDK P 1, TDK V 1, TDK I 1, TDK R 1, Comment
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import re


class LabVIEWParser:
    """
    Parser for LabVIEW measurement text files from STM system
    Fixed for your specific file format
    """

    # File type detection patterns
    FILE_TYPE_PATTERNS = {
        'degas': r'OUTGAS|outgas|Outgas',
        'flash': r'Flash|FLASH|flash',
        'hterm': r'HTERM|hterm|H[_-]?term|HTERMINATE',
        'dose': r'dose|DOSE|PH3|AsH3',
        'incorporation': r'Inc|INC|incorporation',
        'overgrowth': r'Overgrowth|OVERGROWTH|overgrowth|Vergrowth'
    }

    def __init__(self, filepath):
        """
        Initialize parser with LabVIEW file

        Args:
            filepath: Path to LabVIEW measurement file
        """
        self.filepath = Path(filepath)
        self.filename = self.filepath.name
        self.file_type = self._detect_file_type()

        self.header = {}
        self.channels = {}
        self.data = None

        # Fixed channel names for your file format
        self.channel_names = [
            'TDK_P', 'P_MBE', 'P_VT', 'TDK_V', 'TDK_I', 'TDK_R', 
            'Pyro_T', 'Untitled', 'TDK_P_1', 'TDK_V_1', 'TDK_I_1', 'TDK_R_1', 'Comment'
        ]

    def _detect_file_type(self):
        """Detect process type from filename"""
        filename_lower = self.filename.lower()

        for process_type, pattern in self.FILE_TYPE_PATTERNS.items():
            if re.search(pattern, filename_lower, re.IGNORECASE):
                return process_type

        return 'unknown'

    def parse(self):
        """
        Parse the LabVIEW file and extract all data

        Returns:
            dict: Parsed data with header, channels, and metrics
        """
        with open(self.filepath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        # Parse header section
        self._parse_header(lines)

        # Parse data section
        self._parse_data(lines)

        # Calculate metrics
        metrics = self._calculate_metrics()

        return {
            'filename': self.filename,
            'filepath': str(self.filepath),
            'file_type': self.file_type,
            'header': self.header,
            'channels': self.channels,
            'data': self.data,
            'metrics': metrics
        }

    def _parse_header(self, lines):
        """Extract header information"""
        in_header = True

        for idx, line in enumerate(lines):
            if '***End_of_Header***' in line and in_header:
                in_header = False
                continue

            if in_header and '\t' in line:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    key, value = parts[0], parts[1]
                    self.header[key] = value

    def _parse_data(self, lines):
        """Parse numerical data section"""
        data_start_idx = None

        # Find where data starts (line with X_Value)
        for idx, line in enumerate(lines):
            if 'X_Value' in line:
                # Update channel names from actual file
                parts = line.strip().split('\t')
                if len(parts) >= 2:  # At least X_Value + one channel
                    self.channel_names = parts[1:]  # Skip X_Value
                
                data_start_idx = idx + 1
                break

        if data_start_idx is None:
            raise ValueError("Could not find data section")

        # Read data into pandas DataFrame
        data_lines = lines[data_start_idx:]

        # Parse into columns
        data_rows = []
        num_expected_columns = len(self.channel_names) + 1  # channels + time
        
        for line in data_lines:
            if line.strip() and '\t' in line:
                try:
                    # Clean the line and split
                    cleaned_line = line.strip().rstrip('\n')
                    parts = cleaned_line.split('\t')
                    
                    # Convert to float, handle Inf values
                    row_data = []
                    for i, part in enumerate(parts[:num_expected_columns]):
                        if part == 'Inf':
                            row_data.append(np.inf)
                        elif part == '-Inf':
                            row_data.append(-np.inf)
                        elif part.strip() == '':  # Empty string
                            row_data.append(np.nan)
                        else:
                            try:
                                row_data.append(float(part))
                            except ValueError:
                                row_data.append(np.nan)
                    
                    # Pad with NaN if row is shorter than expected
                    while len(row_data) < num_expected_columns:
                        row_data.append(np.nan)
                    
                    data_rows.append(row_data)
                    
                except Exception as e:
                    # Skip malformed lines
                    continue

        if not data_rows:
            raise ValueError("No valid data found")

        # Create DataFrame with correct column count
        column_names = ['Time'] + self.channel_names
        self.data = pd.DataFrame(data_rows, columns=column_names)

        # Store individual channel data
        for channel in self.channel_names:
            if channel in self.data.columns:
                self.channels[channel] = self.data[channel].values

    def _calculate_metrics(self):
        """Calculate metrics from parsed data"""
        metrics = {}

        if self.data is None or len(self.data) == 0:
            return metrics

        try:
            # Basic timing metrics
            time_data = self.data['Time'].values
            if len(time_data) > 1:
                metrics['duration_seconds'] = float(time_data[-1] - time_data[0])
            else:
                metrics['duration_seconds'] = 0.0

            # Temperature metrics (from Pyro T column)
            if 'Pyro T' in self.data.columns:
                temp_data = self.data['Pyro T'].values
                # Filter out infinite values
                valid_temp = temp_data[np.isfinite(temp_data)]
                
                if len(valid_temp) > 0:
                    metrics['peak_temperature'] = float(np.max(valid_temp))
                    metrics['min_temperature'] = float(np.min(valid_temp))
                    metrics['mean_temperature'] = float(np.mean(valid_temp))
                else:
                    metrics['peak_temperature'] = 0.0
                    metrics['min_temperature'] = 0.0
                    metrics['mean_temperature'] = 0.0

            # Pressure metrics (from P MBE column)
            if 'P MBE' in self.data.columns:
                pressure_data = self.data['P MBE'].values
                valid_pressure = pressure_data[np.isfinite(pressure_data)]
                
                if len(valid_pressure) > 0:
                    metrics['peak_pressure'] = float(np.max(valid_pressure))
                    metrics['base_pressure'] = float(np.min(valid_pressure))
                else:
                    metrics['peak_pressure'] = 0.0
                    metrics['base_pressure'] = 0.0

            # Calculate thermal budget
            if 'Pyro T' in self.data.columns:
                temperature = self.data['Pyro T'].values
                time = self.data['Time'].values
                
                # Remove NaN and infinite values
                valid_mask = np.isfinite(temperature) & np.isfinite(time)
                temperature = temperature[valid_mask]
                time = time[valid_mask]

                if len(temperature) >= 2:
                    # Baseline temperature (room temp)
                    baseline_temp = 23.0  # °C
                    
                    # Calculate effective temperature above baseline
                    effective_temp = np.maximum(temperature - baseline_temp, 0)
                    
                    # Trapezoidal integration
                    thermal_budget = np.trapz(effective_temp, time)
                    metrics['thermal_budget'] = float(thermal_budget)
                else:
                    metrics['thermal_budget'] = 0.0

            # Process-specific metrics
            if self.file_type == 'flash':
                self._calculate_flash_metrics(metrics)
            elif self.file_type == 'dose':
                self._calculate_dose_metrics(metrics)

        except Exception as e:
            print(f"Warning: Error calculating metrics for {self.filename}: {e}")
            metrics['error'] = str(e)

        return metrics

    def _calculate_flash_metrics(self, metrics):
        """Calculate flash-specific metrics"""
        if 'Pyro T' not in self.data.columns:
            return

        temp_data = self.data['Pyro T'].values
        time_data = self.data['Time'].values

        # Find flash duration (time above 500°C)
        valid_mask = np.isfinite(temp_data)
        temp_valid = temp_data[valid_mask]
        time_valid = time_data[valid_mask]

        if len(temp_valid) > 0:
            hot_mask = temp_valid > 500
            if np.any(hot_mask):
                flash_times = time_valid[hot_mask]
                metrics['flash_duration'] = float(flash_times[-1] - flash_times[0])

            # Calculate heating and cooling rates
            peak_idx = np.argmax(temp_valid)
            if peak_idx > 10 and peak_idx < len(temp_valid) - 10:
                # Heating rate (around 200-400°C range)
                low_temp_mask = temp_valid > 200
                mid_temp_mask = temp_valid < 400
                heating_mask = low_temp_mask & mid_temp_mask
                
                if np.sum(heating_mask) > 5:
                    heating_times = time_valid[heating_mask]
                    heating_temps = temp_valid[heating_mask]
                    if len(heating_times) > 2:
                        heating_rate = np.polyfit(heating_times, heating_temps, 1)[0]
                        metrics['heating_rate'] = float(heating_rate)

                # Cooling rate (around 400-200°C range during cooling)
                cooling_mask = (temp_valid < 400) & (temp_valid > 200) & (np.arange(len(temp_valid)) > peak_idx)
                
                if np.sum(cooling_mask) > 5:
                    cooling_times = time_valid[cooling_mask]
                    cooling_temps = temp_valid[cooling_mask]
                    if len(cooling_times) > 2:
                        cooling_rate = np.polyfit(cooling_times, cooling_temps, 1)[0]
                        metrics['cooling_rate'] = float(cooling_rate)

    def _calculate_dose_metrics(self, metrics):
        """Calculate dose-specific metrics"""
        if 'P MBE' not in self.data.columns:
            return

        pressure_data = self.data['P MBE'].values
        time_data = self.data['Time'].values

        # Find dose duration (time above baseline pressure)
        valid_mask = np.isfinite(pressure_data)
        pressure_valid = pressure_data[valid_mask]
        time_valid = time_data[valid_mask]

        if len(pressure_valid) > 0:
            base_pressure = np.min(pressure_valid)
            dose_threshold = base_pressure * 10  # 10x baseline
            
            dose_mask = pressure_valid > dose_threshold
            if np.any(dose_mask):
                dose_times = time_valid[dose_mask]
                metrics['dose_duration'] = float(dose_times[-1] - dose_times[0])
                
                # Average dose pressure
                dose_pressures = pressure_valid[dose_mask]
                metrics['dose_pressure'] = float(np.mean(dose_pressures))
                
                # Calculate exposure in Langmuirs (1 Langmuir = 1e-6 Torr·s)
                if metrics['dose_duration'] > 0 and metrics['dose_pressure'] > 0:
                    exposure_torr_s = metrics['dose_pressure'] * metrics['dose_duration']
                    metrics['exposure_langmuirs'] = exposure_torr_s * 1e6  # Convert to Langmuirs

    def generate_summary(self):
        """Generate human-readable summary"""
        if not hasattr(self, 'metrics') or not self.metrics:
            return "No data available"

        summary_lines = [
            f"LabVIEW File: {self.filename}",
            f"File Type: {self.file_type}",
            f"Duration: {self.metrics['duration_seconds']:.1f} seconds"
        ]

        if 'peak_temperature' in self.metrics:
            summary_lines.extend([
                f"Peak Temperature: {self.metrics['peak_temperature']:.1f} °C",
                f"Min Temperature: {self.metrics['min_temperature']:.1f} °C",
                f"Mean Temperature: {self.metrics['mean_temperature']:.1f} °C"
            ])

        if 'thermal_budget' in self.metrics:
            summary_lines.append(f"Thermal Budget: {self.metrics['thermal_budget']:.2e} °C·s")

        if 'peak_pressure' in self.metrics:
            summary_lines.extend([
                f"Peak Pressure: {self.metrics['peak_pressure']:.2e} Torr",
                f"Base Pressure: {self.metrics['base_pressure']:.2e} Torr"
            ])

        # File-specific summaries
        if self.file_type == 'flash':
            if 'flash_duration' in self.metrics:
                summary_lines.append(f"Flash Duration: {self.metrics['flash_duration']:.1f} s")
            if 'heating_rate' in self.metrics:
                summary_lines.append(f"Heating Rate: {self.metrics['heating_rate']:.1f} °C/s")
            if 'cooling_rate' in self.metrics:
                summary_lines.append(f"Cooling Rate: {self.metrics['cooling_rate']:.1f} °C/s")

        elif self.file_type == 'dose':
            if 'dose_duration' in self.metrics:
                summary_lines.append(f"Dose Duration: {self.metrics['dose_duration']:.1f} s")
            if 'dose_pressure' in self.metrics:
                summary_lines.append(f"Dose Pressure: {self.metrics['dose_pressure']:.2e} Torr")
            if 'exposure_langmuirs' in self.metrics:
                summary_lines.append(f"Total Exposure: {self.metrics['exposure_langmuirs']:.1f} Langmuirs")

        return '\n'.join(summary_lines)


# Test function
def test_parser():
    """Test the parser with your uploaded files"""
    test_files = [
        '20240414 PH3DOSE.txt',
        '20240419_OUTGAS.txt', 
        '20240423 Flash1.txt',
        '20240423 HTERMINATE.txt',
        '20240424 OVergrowth.txt',
        '20240425 AsH3 Dose.txt'
    ]
    
    for test_file in test_files:
        if Path(test_file).exists():
            try:
                parser = LabVIEWParser(test_file)
                result = parser.parse()
                
                print(f"\n{test_file}:")
                print(f"  File Type: {result['file_type']}")
                print(f"  Duration: {result['metrics']['duration_seconds']:.1f} seconds")
                
                if 'peak_temperature' in result['metrics']:
                    print(f"  Peak Temp: {result['metrics']['peak_temperature']:.1f} °C")
                    
                if 'thermal_budget' in result['metrics']:
                    print(f"  Thermal Budget: {result['metrics']['thermal_budget']:.2e} °C·s")
                    
            except Exception as e:
                print(f"  Error: {e}")


if __name__ == "__main__":
    test_parser()