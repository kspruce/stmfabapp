# stm_fab/analysis/cooldown_analysis.py
"""
cooldown_analysis.py - Cooldown curve extraction and temperature calibration

Extracts cooldown curves from flash files and creates temperature-current calibrations
Based on Phase 5 of the Enhancement Roadmap
"""

import numpy as np
from scipy import interpolate, optimize
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, Callable
import json


def load_calibration_from_json(json_path: str) -> Dict[str, Any]:
    """
    Load temperature calibration from JSON file
    
    Args:
        json_path: Path to calibration JSON file
        
    Returns:
        Dictionary with calibration data including:
        - model: Model type ('R(T)_exponential', 'T(I)_polynomial', etc)
        - coefficients: Model coefficients
        - r_squared: Fit quality
        - rmse: Root mean squared error
        - calibration_range: Valid range for calibration
        - temperature_function: Callable that computes T from I or R
    """
    with open(json_path, 'r') as f:
        cal_data = json.load(f)
    
    model = cal_data.get('model', 'unknown')
    coeffs = cal_data.get('coefficients', {})
    
    # Create temperature estimation function based on model type
    if model == 'R(T)_exponential':
        # R = a * exp(b / T) + c
        # Need to invert to get T(R): T = (1/b) * ln((R - c) / a)
        a = coeffs['a']
        b = coeffs['b']
        c = coeffs['c']
        
        def temp_from_resistance(R: float) -> float:
            """Calculate temperature (K) from resistance (Ω)"""
            if R <= c:
                return np.nan
            log_arg = (R - c) / a
            if log_arg <= 0:
                return np.nan
            return b / np.log(log_arg)
        
        # For current-based estimation, need V/I = R
        def temp_from_VI(V: float, I: float) -> float:
            """Calculate temperature (K) from voltage and current"""
            if I == 0:
                return np.nan
            R = V / I
            return temp_from_resistance(R)
        
        cal_data['temperature_from_resistance'] = temp_from_resistance
        cal_data['temperature_from_VI'] = temp_from_VI
        
        # Also provide vectorized versions
        def temp_from_resistance_vec(R: np.ndarray) -> np.ndarray:
            T = np.full_like(R, np.nan, dtype=float)
            valid_mask = R > c
            
            if not np.any(valid_mask):
                return T
            
            log_arg = (R[valid_mask] - c) / a
            positive_log = log_arg > 0
            
            if np.any(positive_log):
                T[valid_mask][positive_log] = b / np.log(log_arg[positive_log])
            
            return T
        
        def temp_from_VI_vec(V: np.ndarray, I: np.ndarray) -> np.ndarray:
            R = np.full_like(V, np.nan, dtype=float)
            valid = I != 0
            R[valid] = V[valid] / I[valid]
            return temp_from_resistance_vec(R)
        
        cal_data['temperature_from_resistance_vec'] = temp_from_resistance_vec
        cal_data['temperature_from_VI_vec'] = temp_from_VI_vec
        
    elif model == 'T(I)_polynomial':
        # T = a0 + a1*I + a2*I^2 + ...
        poly_coeffs = list(coeffs.values())
        poly_func = np.poly1d(poly_coeffs[::-1])  # Reverse for poly1d convention
        
        def temp_from_current(I: float) -> float:
            """Calculate temperature (K) from current (A)"""
            return float(poly_func(I))
        
        def temp_from_current_vec(I: np.ndarray) -> np.ndarray:
            """Calculate temperature (K) from current array (A)"""
            return poly_func(I)
        
        cal_data['temperature_from_current'] = temp_from_current
        cal_data['temperature_from_current_vec'] = temp_from_current_vec
    
    else:
        raise ValueError(f"Unknown calibration model: {model}")
    
    return cal_data


class CooldownAnalyzer:
    """
    Analyzes cooldown curves from flash files and creates temperature calibrations
    """
    
    def __init__(self, calibration_json: Optional[str] = None):
        """
        Initialize cooldown analyzer
        
        Args:
            calibration_json: Optional path to existing calibration JSON
        """
        self.cooldown_data = None
        self.calibration_curve = None
        self.fit_coefficients = None
        self.external_calibration = None
        
        if calibration_json:
            self.load_calibration(calibration_json)
    
    def load_calibration(self, json_path: str):
        """Load calibration from JSON file"""
        self.external_calibration = load_calibration_from_json(json_path)
    
    def estimate_temperature_from_current(self, current: np.ndarray, 
                                         voltage: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Estimate temperature from current (and optionally voltage) using loaded calibration
        
        Args:
            current: Current array (A)
            voltage: Optional voltage array (V) - required for R(T) models
            
        Returns:
            Temperature array (K), or array of NaN if no calibration loaded
        """
        if self.external_calibration is None:
            return np.full_like(current, np.nan, dtype=float)
        
        model = self.external_calibration.get('model', 'unknown')
        
        if model == 'R(T)_exponential':
            if voltage is None:
                raise ValueError("Voltage required for R(T) calibration model")
            temp_func = self.external_calibration['temperature_from_VI_vec']
            return temp_func(voltage, current)
        
        elif model == 'T(I)_polynomial':
            temp_func = self.external_calibration['temperature_from_current_vec']
            return temp_func(current)
        
        else:
            return np.full_like(current, np.nan, dtype=float)
    
    def extract_cooldown_curve(self, temperature_array: np.ndarray, 
                               current_array: np.ndarray,
                               time_array: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract cooldown portion from flash data
        
        Args:
            temperature_array: Temperature data (°C)
            current_array: Current data (A)
            time_array: Time data (s)
            
        Returns:
            Dictionary with cooldown data
        """
        # Remove NaN values
        valid_mask = ~(np.isnan(temperature_array) | np.isnan(current_array) | np.isnan(time_array))
        temp = temperature_array[valid_mask]
        current = current_array[valid_mask]
        time = time_array[valid_mask]
        
        if len(temp) < 10:
            raise ValueError("Insufficient data points for cooldown analysis")
        
        # Find peak temperature (start of cooldown)
        peak_idx = np.argmax(temp)
        
        # Extract cooldown portion (from peak onwards)
        cooldown_temp = temp[peak_idx:]
        cooldown_current = current[peak_idx:]
        cooldown_time = time[peak_idx:]
        
        # Ensure monotonically decreasing temperature for interpolation
        # Sometimes there are small fluctuations, so we smooth slightly
        if len(cooldown_temp) > 10:
            cooldown_temp_smooth = savgol_filter(cooldown_temp, 
                                                window_length=min(11, len(cooldown_temp)//2*2+1), 
                                                polyorder=3)
        else:
            cooldown_temp_smooth = cooldown_temp
        
        self.cooldown_data = {
            'temperature': cooldown_temp,
            'temperature_smooth': cooldown_temp_smooth,
            'current': cooldown_current,
            'time': cooldown_time,
            'peak_temp': temp[peak_idx],
            'peak_current': current[peak_idx],
            'peak_time': time[peak_idx]
        }
        
        return self.cooldown_data
    
    def create_calibration(self, min_current: float = 0.03, 
                          max_current: float = 0.30,
                          num_points: int = 100) -> Dict[str, Any]:
        """
        Create temperature-current calibration curve
        
        Args:
            min_current: Minimum current for calibration range (A)
            max_current: Maximum current for calibration range (A)
            num_points: Number of interpolation points
            
        Returns:
            Calibration dictionary with interpolation function and fit
        """
        if self.cooldown_data is None:
            raise ValueError("Must extract cooldown curve first")
        
        temp = self.cooldown_data['temperature_smooth']
        current = self.cooldown_data['current']
        
        # Remove duplicates and sort by current (for interpolation)
        sorted_indices = np.argsort(current)
        current_sorted = current[sorted_indices]
        temp_sorted = temp[sorted_indices]
        
        # Remove duplicate current values (keep first occurrence)
        unique_current, unique_indices = np.unique(current_sorted, return_index=True)
        unique_temp = temp_sorted[unique_indices]
        
        # Filter to calibration range
        in_range = (unique_current >= min_current) & (unique_current <= max_current)
        cal_current = unique_current[in_range]
        cal_temp = unique_temp[in_range]
        
        if len(cal_current) < 5:
            raise ValueError("Insufficient data points in calibration range")
        
        # Create interpolation function: Temperature as function of Current
        # Since cooldown is monotonic, this should work well
        interp_func = interpolate.interp1d(cal_current, cal_temp, 
                                           kind='cubic', 
                                           fill_value='extrapolate',
                                           bounds_error=False)
        
        # Create uniform current grid for calibration
        cal_current_grid = np.linspace(min_current, max_current, num_points)
        cal_temp_grid = interp_func(cal_current_grid)
        
        # Also fit polynomial: T(I) = a0 + a1*I + a2*I^2 + ... 
        # This gives us an analytical expression
        poly_degree = 3  # Cubic fit usually works well
        fit_coeffs = np.polyfit(cal_current, cal_temp, poly_degree)
        poly_func = np.poly1d(fit_coeffs)
        
        # Calculate fit quality
        cal_temp_fit = poly_func(cal_current)
        residuals = cal_temp - cal_temp_fit
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((cal_temp - np.mean(cal_temp))**2)
        r_squared = 1 - (ss_res / ss_tot)
        rmse = np.sqrt(np.mean(residuals**2))
        
        self.calibration_curve = {
            'current': cal_current_grid,
            'temperature': cal_temp_grid,
            'interp_function': interp_func,
            'poly_function': poly_func,
            'fit_coefficients': fit_coeffs.tolist(),
            'r_squared': r_squared,
            'rmse': rmse,
            'current_range': (min_current, max_current),
            'data_points': {
                'current': cal_current,
                'temperature': cal_temp
            }
        }
        
        self.fit_coefficients = fit_coeffs
        
        return self.calibration_curve
    
    def save_calibration(self, output_path: str, source_file: str = None):
        """
        Save calibration to JSON file
        
        Args:
            output_path: Path to save calibration JSON
            source_file: Optional source file name for metadata
        """
        if self.calibration_curve is None:
            raise ValueError("No calibration curve to save")
        
        calibration_dict = {
            'source_file': source_file,
            'model': 'T(I)_polynomial',
            'coefficients': {
                f'a{i}': float(coeff) 
                for i, coeff in enumerate(self.fit_coefficients[::-1])
            },
            'r_squared': float(self.calibration_curve['r_squared']),
            'rmse': float(self.calibration_curve['rmse']),
            'calibration_range': {
                'current_min': float(self.calibration_curve['current_range'][0]),
                'current_max': float(self.calibration_curve['current_range'][1]),
                'temp_min': float(np.min(self.calibration_curve['data_points']['temperature'])),
                'temp_max': float(np.max(self.calibration_curve['data_points']['temperature']))
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(calibration_dict, f, indent=2)
        
        print(f"Calibration saved to {output_path}")
    
    def plot_cooldown(self, figsize=(12, 4)):
        """
        Plot cooldown curve
        
        Args:
            figsize: Figure size tuple
            
        Returns:
            Figure object
        """
        if self.cooldown_data is None:
            raise ValueError("No cooldown data available")
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Time vs Temperature
        axes[0].plot(self.cooldown_data['time'], self.cooldown_data['temperature'], 'b-', alpha=0.5, label='Raw')
        axes[0].plot(self.cooldown_data['time'], self.cooldown_data['temperature_smooth'], 'r-', label='Smoothed')
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Temperature (°C)')
        axes[0].set_title('Cooldown: Temperature vs Time')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Time vs Current
        axes[1].plot(self.cooldown_data['time'], self.cooldown_data['current'], 'g-')
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Current (A)')
        axes[1].set_title('Cooldown: Current vs Time')
        axes[1].grid(True, alpha=0.3)
        
        # Temperature vs Current
        axes[2].plot(self.cooldown_data['current'], self.cooldown_data['temperature'], 'b.', alpha=0.5, markersize=2)
        axes[2].set_xlabel('Current (A)')
        axes[2].set_ylabel('Temperature (°C)')
        axes[2].set_title('Temperature vs Current')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_calibration(self, figsize=(10, 6)):
        """
        Plot calibration curve with fit
        
        Args:
            figsize: Figure size tuple
            
        Returns:
            Figure object
        """
        if self.calibration_curve is None:
            raise ValueError("No calibration curve available")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        cal = self.calibration_curve
        data = cal['data_points']
        
        # Plot data points
        ax.plot(data['current'], data['temperature'], 'bo', label='Data', markersize=4, alpha=0.6)
        
        # Plot interpolation
        ax.plot(cal['current'], cal['temperature'], 'r-', label='Interpolation', linewidth=2)
        
        # Plot polynomial fit
        poly_fit = cal['poly_function'](cal['current'])
        ax.plot(cal['current'], poly_fit, 'g--', label=f'Poly Fit (R²={cal["r_squared"]:.4f})', linewidth=2)
        
        ax.set_xlabel('Current (A)', fontsize=12)
        ax.set_ylabel('Temperature (°C)', fontsize=12)
        ax.set_title('Temperature Calibration Curve', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add fit equation as text
        coeffs = cal['fit_coefficients']
        equation = f'T = {coeffs[0]:.2e}'
        for i, c in enumerate(coeffs[1:], 1):
            equation += f' + {c:.2e}·I^{i}'
        
        ax.text(0.05, 0.95, equation, transform=ax.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        return fig
    
    def get_temperature_for_current(self, current_value: float) -> float:
        """
        Get calibrated temperature for a given current
        
        Args:
            current_value: Current in Amperes
            
        Returns:
            Temperature in °C
        """
        if self.calibration_curve is None:
            raise ValueError("No calibration curve available")
        
        return float(self.calibration_curve['interp_function'](current_value))


# Convenience function for loading calibration
def load_temperature_calibration(json_path: str) -> Dict[str, Any]:
    """
    Load temperature calibration from JSON file (convenience wrapper)
    
    Args:
        json_path: Path to calibration JSON file
        
    Returns:
        Calibration dictionary
    """
    return load_calibration_from_json(json_path)
