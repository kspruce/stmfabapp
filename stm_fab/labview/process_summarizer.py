"""
process_summarizer.py - Generate human-readable summaries from LabVIEW process data

Creates formatted summaries for different process types based on parsed data
Part of Phase 2 in the Enhancement Roadmap
"""

from typing import Dict, Any, List
from datetime import datetime, timedelta


class ProcessSummarizer:
    """
    Generates formatted summaries for different process types
    """
    
    @staticmethod
    def format_time(seconds: float) -> str:
        """Format seconds into human-readable time"""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f} min"
        else:
            hours = seconds / 3600
            return f"{hours:.2f} hours"
    
    @staticmethod
    def format_pressure(torr: float) -> str:
        """Format pressure with appropriate units"""
        if torr >= 1e-3:
            return f"{torr:.2e} Torr"
        elif torr >= 1e-6:
            return f"{torr * 1e3:.2f} mTorr"
        elif torr >= 1e-9:
            return f"{torr * 1e6:.2f} µTorr"
        else:
            return f"{torr:.2e} Torr"
    
    def generate_summary(self, parsed_data: Dict[str, Any]) -> str:
        """
        Generate summary based on file type
        
        Args:
            parsed_data: Dictionary from LabVIEW parser
            
        Returns:
            Formatted summary string
        """
        file_type = parsed_data.get('file_type', 'unknown')
        
        # Route to appropriate summarizer
        summarizers = {
            'degas': self.summarize_degas,
            'flash': self.summarize_flash,
            'hterm': self.summarize_hterm,
            'dose': self.summarize_dose,
            'incorporation': self.summarize_incorporation,
            'overgrowth': self.summarize_overgrowth,
        }
        
        summarizer = summarizers.get(file_type, self.summarize_generic)
        return summarizer(parsed_data)
    
    def summarize_degas(self, data: Dict[str, Any]) -> str:
        """Generate summary for outgassing/degas process"""
        metrics = data['metrics']
        header = data.get('header', {})
        
        lines = []
        lines.append("╔" + "═" * 78 + "╗")
        lines.append("║" + " OUTGASSING / DEGAS SUMMARY".center(78) + "║")
        lines.append("╠" + "═" * 78 + "╣")
        lines.append("║" + " " * 78 + "║")
        
        # Basic info
        if 'Date' in header:
            lines.append(f"║  Date: {header['Date']:<69}║")
        lines.append(f"║  Duration: {self.format_time(metrics.get('duration_seconds', 0)):<65}║")
        lines.append("║" + " " * 78 + "║")
        
        # Temperature info
        if 'peak_temperature' in metrics:
            lines.append(f"║  Peak Temperature: {metrics['peak_temperature']:.1f} °C{' ' * 48}║")
        if 'average_temperature' in metrics:
            lines.append(f"║  Average Temperature: {metrics['average_temperature']:.1f} °C{' ' * 45}║")
        
        # Thermal budget
        if 'thermal_budget' in metrics:
            tb = metrics['thermal_budget']
            lines.append(f"║  Thermal Budget: {tb:.2e} °C·s{' ' * 44}║")
        
        lines.append("║" + " " * 78 + "║")
        
        # Pressure info
        if 'base_pressure' in metrics:
            lines.append(f"║  Final Base Pressure: {self.format_pressure(metrics['base_pressure']):<57}║")
        if 'peak_pressure' in metrics:
            lines.append(f"║  Peak Pressure: {self.format_pressure(metrics['peak_pressure']):<61}║")
        
        lines.append("║" + " " * 78 + "║")
        
        # Cycles (if detected)
        if 'num_cycles' in metrics:
            lines.append(f"║  Number of Cycles: {metrics['num_cycles']}{' ' * 57}║")
        
        # Quality flags
        quality = metrics.get('quality_flags', {})
        lines.append("║  Status:{' ' * 69}║")
        
        status_checks = [
            ('pressure_stable', 'Pressure Stable'),
            ('no_anomalies', 'No Anomalies Detected'),
        ]
        
        for key, label in status_checks:
            if key in quality:
                status = "✓" if quality[key] else "✗"
                lines.append(f"║    {status} {label}{' ' * (71 - len(label))}║")
        
        lines.append("║" + " " * 78 + "║")
        lines.append("╚" + "═" * 78 + "╝")
        
        return "\n".join(lines)
    
    def summarize_flash(self, data: Dict[str, Any]) -> str:
        """Generate summary for flash cleaning process"""
        metrics = data['metrics']
        header = data.get('header', {})
        
        lines = []
        lines.append("╔" + "═" * 78 + "╗")
        lines.append("║" + " FLASH CLEANING & COOLDOWN SUMMARY".center(78) + "║")
        lines.append("╠" + "═" * 78 + "╣")
        lines.append("║" + " " * 78 + "║")
        
        # Basic info
        if 'Date' in header:
            lines.append(f"║  Date: {header['Date']:<69}║")
        lines.append(f"║  Duration: {self.format_time(metrics.get('duration_seconds', 0)):<65}║")
        lines.append("║" + " " * 78 + "║")
        
        # Temperature profile
        if 'peak_temperature' in metrics:
            lines.append(f"║  Peak Temperature: {metrics['peak_temperature']:.1f} °C{' ' * 48}║")
        
        if 'flash_duration' in metrics:
            lines.append(f"║  Time at Peak: {self.format_time(metrics['flash_duration']):<61}║")
        
        if 'heating_rate' in metrics:
            lines.append(f"║  Heating Rate: {metrics['heating_rate']:.1f} °C/s{' ' * 52}║")
        
        if 'cooling_rate' in metrics:
            lines.append(f"║  Cooling Rate: {metrics['cooling_rate']:.1f} °C/s{' ' * 52}║")
        
        lines.append("║" + " " * 78 + "║")
        
        # Thermal budget
        if 'thermal_budget' in metrics:
            tb = metrics['thermal_budget']
            lines.append(f"║  Thermal Budget Contribution: {tb:.2e} °C·s{' ' * 35}║")
        
        lines.append("║" + " " * 78 + "║")
        
        # Pressure info
        if 'base_pressure' in metrics:
            lines.append(f"║  Base Pressure: {self.format_pressure(metrics['base_pressure']):<59}║")
        if 'peak_pressure' in metrics:
            lines.append(f"║  Peak Pressure: {self.format_pressure(metrics['peak_pressure']):<59}║")
        
        lines.append("║" + " " * 78 + "║")
        
        # Quality indicators
        quality = metrics.get('quality_flags', {})
        lines.append("║  Quality Indicators:{' ' * 57}║")
        
        status_checks = [
            ('pressure_stable', 'Pressure Remained Stable'),
            ('temp_overshoot', 'Temperature Overshoot', True),  # Inverted (bad if True)
            ('cooling_normal', 'Cooling Rate Normal'),
        ]
        
        for item in status_checks:
            key = item[0]
            label = item[1]
            inverted = item[2] if len(item) > 2 else False
            
            if key in quality:
                value = quality[key]
                if inverted:
                    status = "✗" if value else "✓"
                else:
                    status = "✓" if value else "✗"
                lines.append(f"║    {status} {label}{' ' * (71 - len(label))}║")
        
        lines.append("║" + " " * 78 + "║")
        lines.append("╚" + "═" * 78 + "╝")
        
        return "\n".join(lines)
    
    def summarize_hterm(self, data: Dict[str, Any]) -> str:
        """Generate summary for hydrogen termination"""
        metrics = data['metrics']
        header = data.get('header', {})
        
        lines = []
        lines.append("╔" + "═" * 78 + "╗")
        lines.append("║" + " HYDROGEN TERMINATION SUMMARY".center(78) + "║")
        lines.append("╠" + "═" * 78 + "╣")
        lines.append("║" + " " * 78 + "║")
        
        # Basic info
        if 'Date' in header:
            lines.append(f"║  Date: {header['Date']:<69}║")
        lines.append(f"║  Total Duration: {self.format_time(metrics.get('duration_seconds', 0)):<61}║")
        lines.append("║" + " " * 78 + "║")
        
        # Process conditions
        if 'process_temperature' in metrics:
            lines.append(f"║  Process Temperature: {metrics['process_temperature']:.1f} °C (~330°C target){' ' * 32}║")
        
        if 'exposure_pressure' in metrics:
            lines.append(f"║  H₂ Exposure Pressure: {self.format_pressure(metrics['exposure_pressure']):<54}║")
        
        if 'exposure_duration' in metrics:
            lines.append(f"║  Exposure Duration: {self.format_time(metrics['exposure_duration']):<57}║")
        
        lines.append("║" + " " * 78 + "║")
        
        # Dose calculation (if available)
        if 'total_dose_langmuir' in metrics:
            lines.append(f"║  Total H₂ Dose: {metrics['total_dose_langmuir']:.1f} L (Langmuirs){' ' * 40}║")
        
        # Thermal budget
        if 'thermal_budget' in metrics:
            tb = metrics['thermal_budget']
            lines.append(f"║  Thermal Budget: {tb:.2e} °C·s{' ' * 44}║")
        
        lines.append("║" + " " * 78 + "║")
        
        # Pressure recovery
        if 'base_pressure_before' in metrics and 'base_pressure_after' in metrics:
            bp_before = metrics['base_pressure_before']
            bp_after = metrics['base_pressure_after']
            lines.append(f"║  Base Pressure Recovery:{' ' * 54}║")
            lines.append(f"║    Before: {self.format_pressure(bp_before):<64}║")
            lines.append(f"║    After:  {self.format_pressure(bp_after):<64}║")
        
        lines.append("║" + " " * 78 + "║")
        
        # Quality checks
        quality = metrics.get('quality_flags', {})
        lines.append("║  Process Quality:{' ' * 61}║")
        
        status_checks = [
            ('temperature_stable', 'Temperature Stable'),
            ('pressure_stable', 'Pressure Stable During Exposure'),
            ('pressure_recovered', 'Base Pressure Recovered'),
        ]
        
        for key, label in status_checks:
            if key in quality:
                status = "✓" if quality[key] else "✗"
                lines.append(f"║    {status} {label}{' ' * (71 - len(label))}║")
        
        lines.append("║" + " " * 78 + "║")
        lines.append("╚" + "═" * 78 + "╝")
        
        return "\n".join(lines)
    
    def summarize_dose(self, data: Dict[str, Any]) -> str:
        """Generate summary for PH3/AsH3 dosing"""
        metrics = data['metrics']
        header = data.get('header', {})
        
        # Detect gas type from filename or data
        filename = data.get('filename', '').lower()
        if 'ph3' in filename:
            gas = 'PH₃ (Phosphine)'
        elif 'ash3' in filename:
            gas = 'AsH₃ (Arsine)'
        else:
            gas = 'Dopant Gas'
        
        lines = []
        lines.append("╔" + "═" * 78 + "╗")
        lines.append("║" + f" DOPANT DOSING SUMMARY - {gas}".center(78) + "║")
        lines.append("╠" + "═" * 78 + "╣")
        lines.append("║" + " " * 78 + "║")
        
        # Basic info
        if 'Date' in header:
            lines.append(f"║  Date: {header['Date']:<69}║")
        lines.append(f"║  Total Duration: {self.format_time(metrics.get('duration_seconds', 0)):<61}║")
        lines.append("║" + " " * 78 + "║")
        
        # Dose parameters
        if 'dose_pressure' in metrics:
            lines.append(f"║  Dose Pressure: {self.format_pressure(metrics['dose_pressure']):<60}║")
            
            # Standard dose is 5e-9 Torr for 10 minutes
            target_pressure = 5e-9
            actual_pressure = metrics['dose_pressure']
            if abs(actual_pressure - target_pressure) / target_pressure < 0.2:
                lines.append(f"║    ✓ Within 20% of standard (5×10⁻⁹ Torr){' ' * 43}║")
            else:
                lines.append(f"║    ⚠ Deviates from standard (5×10⁻⁹ Torr){' ' * 43}║")
        
        if 'dose_duration' in metrics:
            lines.append(f"║  Dose Duration: {self.format_time(metrics['dose_duration']):<60}║")
            
            # Check against standard 10 minutes = 600 seconds
            if metrics['dose_duration'] >= 600:
                lines.append(f"║    ✓ Meets minimum 10 minute requirement{' ' * 44}║")
            else:
                lines.append(f"║    ⚠ Below standard 10 minute dose{' ' * 49}║")
        
        lines.append("║" + " " * 78 + "║")
        
        # Total dose in Langmuirs
        if 'total_dose_langmuir' in metrics:
            lines.append(f"║  Total Dose: {metrics['total_dose_langmuir']:.1f} L (Langmuirs){' ' * 45}║")
        
        # Temperature during dose
        if 'dose_temperature' in metrics:
            lines.append(f"║  Temperature During Dose: {metrics['dose_temperature']:.1f} °C{' ' * 43}║")
        
        lines.append("║" + " " * 78 + "║")
        
        # Pressure recovery
        if 'base_pressure_before' in metrics and 'base_pressure_after' in metrics:
            lines.append(f"║  Base Pressure:{' ' * 63}║")
            lines.append(f"║    Before Dose: {self.format_pressure(metrics['base_pressure_before']):<58}║")
            lines.append(f"║    After Dose:  {self.format_pressure(metrics['base_pressure_after']):<58}║")
        
        lines.append("║" + " " * 78 + "║")
        
        # Quality indicators
        quality = metrics.get('quality_flags', {})
        lines.append("║  Dose Quality:{' ' * 64}║")
        
        status_checks = [
            ('dose_pressure_reached', 'Target Pressure Achieved'),
            ('dose_time_sufficient', 'Sufficient Exposure Time'),
            ('pressure_stable', 'Pressure Stable During Dose'),
            ('pressure_recovered', 'Base Pressure Recovered'),
        ]
        
        for key, label in status_checks:
            if key in quality:
                status = "✓" if quality[key] else "✗"
                lines.append(f"║    {status} {label}{' ' * (71 - len(label))}║")
        
        lines.append("║" + " " * 78 + "║")
        lines.append("╚" + "═" * 78 + "╝")
        
        return "\n".join(lines)
    
    def summarize_incorporation(self, data: Dict[str, Any]) -> str:
        """Generate summary for dopant incorporation anneal"""
        metrics = data['metrics']
        header = data.get('header', {})
        
        lines = []
        lines.append("╔" + "═" * 78 + "╗")
        lines.append("║" + " DOPANT INCORPORATION ANNEAL SUMMARY".center(78) + "║")
        lines.append("╠" + "═" * 78 + "╣")
        lines.append("║" + " " * 78 + "║")
        
        # Basic info
        if 'Date' in header:
            lines.append(f"║  Date: {header['Date']:<69}║")
        lines.append(f"║  Total Duration: {self.format_time(metrics.get('duration_seconds', 0)):<61}║")
        lines.append("║" + " " * 78 + "║")
        
        # Anneal parameters
        if 'anneal_temperature' in metrics:
            lines.append(f"║  Anneal Temperature: {metrics['anneal_temperature']:.1f} °C (~350°C target){' ' * 33}║")
        
        if 'anneal_duration' in metrics:
            lines.append(f"║  Time at Temperature: {self.format_time(metrics['anneal_duration']):<54}║")
        
        if 'heating_rate' in metrics:
            lines.append(f"║  Heating Rate: {metrics['heating_rate']:.2f} °C/s{' ' * 51}║")
        
        if 'cooling_rate' in metrics:
            lines.append(f"║  Cooling Rate: {metrics['cooling_rate']:.2f} °C/s{' ' * 51}║")
        
        lines.append("║" + " " * 78 + "║")
        
        # Thermal budget
        if 'thermal_budget' in metrics:
            tb = metrics['thermal_budget']
            lines.append(f"║  Thermal Budget Contribution: {tb:.2e} °C·s{' ' * 35}║")
            lines.append(f"║    ⚠ This is typically the largest thermal budget contribution{' ' * 24}║")
        
        lines.append("║" + " " * 78 + "║")
        
        # Pressure stability
        if 'base_pressure' in metrics:
            lines.append(f"║  Base Pressure: {self.format_pressure(metrics['base_pressure']):<59}║")
        
        if 'peak_pressure' in metrics:
            lines.append(f"║  Peak Pressure: {self.format_pressure(metrics['peak_pressure']):<59}║")
        
        lines.append("║" + " " * 78 + "║")
        
        # Quality checks
        quality = metrics.get('quality_flags', {})
        lines.append("║  Process Quality:{' ' * 61}║")
        
        status_checks = [
            ('temperature_stable', 'Temperature Stable at Setpoint'),
            ('pressure_stable', 'Pressure Remained Stable'),
            ('no_anomalies', 'No Process Anomalies'),
        ]
        
        for key, label in status_checks:
            if key in quality:
                status = "✓" if quality[key] else "✗"
                lines.append(f"║    {status} {label}{' ' * (71 - len(label))}║")
        
        lines.append("║" + " " * 78 + "║")
        lines.append("╚" + "═" * 78 + "╝")
        
        return "\n".join(lines)
    
    def summarize_overgrowth(self, data: Dict[str, Any]) -> str:
        """Generate summary for epitaxial overgrowth"""
        metrics = data['metrics']
        header = data.get('header', {})
        
        lines = []
        lines.append("╔" + "═" * 78 + "╗")
        lines.append("║" + " EPITAXIAL OVERGROWTH SUMMARY".center(78) + "║")
        lines.append("╠" + "═" * 78 + "╣")
        lines.append("║" + " " * 78 + "║")
        
        # Basic info
        if 'Date' in header:
            lines.append(f"║  Date: {header['Date']:<69}║")
        lines.append(f"║  Total Duration: {self.format_time(metrics.get('duration_seconds', 0)):<61}║")
        lines.append("║" + " " * 78 + "║")
        
        # Growth parameters
        if 'growth_temperature' in metrics:
            lines.append(f"║  Growth Temperature: {metrics['growth_temperature']:.1f} °C{' ' * 48}║")
        
        if 'growth_duration' in metrics:
            lines.append(f"║  Growth Duration: {self.format_time(metrics['growth_duration']):<58}║")
        
        if 'growth_rate' in metrics:
            lines.append(f"║  Estimated Growth Rate: {metrics['growth_rate']:.3f} nm/min{' ' * 42}║")
        
        if 'estimated_thickness' in metrics:
            lines.append(f"║  Estimated Thickness: {metrics['estimated_thickness']:.1f} nm{' ' * 47}║")
        
        lines.append("║" + " " * 78 + "║")
        
        # Thermal budget
        if 'thermal_budget' in metrics:
            tb = metrics['thermal_budget']
            lines.append(f"║  Thermal Budget Contribution: {tb:.2e} °C·s{' ' * 35}║")
        
        lines.append("║" + " " * 78 + "║")
        
        # Pressure
        if 'growth_pressure' in metrics:
            lines.append(f"║  Growth Pressure: {self.format_pressure(metrics['growth_pressure']):<57}║")
        
        if 'base_pressure_after' in metrics:
            lines.append(f"║  Final Base Pressure: {self.format_pressure(metrics['base_pressure_after']):<53}║")
        
        lines.append("║" + " " * 78 + "║")
        
        # Quality indicators
        quality = metrics.get('quality_flags', {})
        lines.append("║  Growth Quality:{' ' * 62}║")
        
        status_checks = [
            ('temperature_stable', 'Temperature Stable'),
            ('pressure_stable', 'Pressure Stable'),
            ('growth_uniform', 'Growth Rate Uniform'),
        ]
        
        for key, label in status_checks:
            if key in quality:
                status = "✓" if quality[key] else "✗"
                lines.append(f"║    {status} {label}{' ' * (71 - len(label))}║")
        
        lines.append("║" + " " * 78 + "║")
        lines.append("╚" + "═" * 78 + "╝")
        
        return "\n".join(lines)
    
    def summarize_generic(self, data: Dict[str, Any]) -> str:
        """Generate generic summary for unknown process types"""
        metrics = data['metrics']
        header = data.get('header', {})
        filename = data.get('filename', 'Unknown')
        
        lines = []
        lines.append("╔" + "═" * 78 + "╗")
        lines.append("║" + " PROCESS SUMMARY".center(78) + "║")
        lines.append("╠" + "═" * 78 + "╣")
        lines.append("║" + " " * 78 + "║")
        lines.append(f"║  File: {filename:<69}║")
        
        if 'Date' in header:
            lines.append(f"║  Date: {header['Date']:<69}║")
        
        lines.append(f"║  Duration: {self.format_time(metrics.get('duration_seconds', 0)):<65}║")
        lines.append("║" + " " * 78 + "║")
        
        # Include any metrics available
        for key, value in metrics.items():
            if key not in ['duration_seconds', 'quality_flags']:
                if isinstance(value, float):
                    lines.append(f"║  {key}: {value:.2e}{' ' * (67 - len(key))}║")
                else:
                    val_str = str(value)[:60]
                    lines.append(f"║  {key}: {val_str}{' ' * (67 - len(key) - len(val_str))}║")
        
        lines.append("║" + " " * 78 + "║")
        lines.append("╚" + "═" * 78 + "╝")
        
        return "\n".join(lines)


# Example usage
if __name__ == '__main__':
    summarizer = ProcessSummarizer()
    
    # Example: Flash process
    flash_data = {
        'filename': '20231002_Flash_3.txt',
        'file_type': 'flash',
        'header': {
            'Date': '2023-10-02',
            'Operator': 'vtstmuhv'
        },
        'metrics': {
            'duration_seconds': 4543.2,
            'peak_temperature': 1579.8,
            'flash_duration': 127.5,
            'heating_rate': 15.3,
            'cooling_rate': -8.1,
            'thermal_budget': 2.45e6,
            'base_pressure': 1.2e-10,
            'peak_pressure': 8.5e-10,
            'quality_flags': {
                'pressure_stable': True,
                'temp_overshoot': False,
                'cooling_normal': True
            }
        }
    }
    
    print(summarizer.generate_summary(flash_data))
    print("\n")
    
    # Example: Dose process
    dose_data = {
        'filename': '20231004_PH3_dose_1.txt',
        'file_type': 'dose',
        'header': {
            'Date': '2023-10-04'
        },
        'metrics': {
            'duration_seconds': 720,
            'dose_pressure': 5.2e-9,
            'dose_duration': 600,
            'dose_temperature': 23.5,
            'total_dose_langmuir': 3900,
            'base_pressure_before': 1.1e-10,
            'base_pressure_after': 1.3e-10,
            'quality_flags': {
                'dose_pressure_reached': True,
                'dose_time_sufficient': True,
                'pressure_stable': True,
                'pressure_recovered': True
            }
        }
    }
    
    print(summarizer.generate_summary(dose_data))
