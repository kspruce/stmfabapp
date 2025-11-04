"""
thermal_budget.py - Thermal budget calculation and tracking


"""

import numpy as np
from datetime import datetime


class ThermalBudgetCalculator:
    """
    Calculate and track thermal budget for device fabrication
    """

    # Standard baseline temperature (room temp)
    BASELINE_TEMP = 23.0  # °C

    # Which processes should be counted toward the effective total
    COUNTED_PROCESSES = {'incorporation', 'overgrowth'}

    # Typical contribution percentages (for reference)
    TYPICAL_CONTRIBUTIONS = {
        'degas': 0.15,          # 15%
        'flash': 0.08,          # 8%
        'hterm': 0.02,          # 2%
        'incorporation': 0.45,  # 45%
        'overgrowth': 0.30,     # 30%
    }

    def __init__(self):
        self.contributions = {}
        # gross_total sums all contributions
        self.gross_total = 0.0
        # effective_total sums only COUNTED_PROCESSES
        self.effective_total = 0.0
        # For backward compatibility: total_budget mirrors effective_total
        self.total_budget = 0.0

    def calculate_from_labview_data(self, temperature, time):
        """
        Calculate thermal budget from temperature and time arrays

        Args:
            temperature: Temperature array in °C
            time: Time array in seconds

        Returns:
            thermal_budget: Thermal budget in °C·s
        """
        # Ensure arrays are numpy arrays
        temperature = np.array(temperature)
        time = np.array(time)

        # Remove NaN values
        valid_mask = ~np.isnan(temperature) & ~np.isnan(time)
        temperature = temperature[valid_mask]
        time = time[valid_mask]

        if len(temperature) < 2:
            return 0.0

        # Calculate effective temperature (above baseline)
        effective_temp = np.maximum(temperature - self.BASELINE_TEMP, 0)

        # Trapezoidal integration
        thermal_budget = np.trapz(effective_temp, time)

        return float(thermal_budget)

    def add_contribution(self, process_type, thermal_budget):
        """
        Add a process contribution to the total thermal budget

        Args:
            process_type: Type of process ('degas', 'flash', etc.)
            thermal_budget: Thermal budget contribution in °C·s
        """
        if process_type not in self.contributions:
            self.contributions[process_type] = 0.0

        self.contributions[process_type] += thermal_budget
        self._recalculate_totals()

    def _recalculate_totals(self):
        """Recalculate gross and effective totals"""
        self.gross_total = sum(self.contributions.values())
        self.effective_total = sum(
            v for k, v in self.contributions.items()
            if k in self.COUNTED_PROCESSES
        )
        # Keep compatibility
        self.total_budget = self.effective_total

    def get_status(self, warning_threshold=3.0e8, critical_threshold=4.0e8, use_effective=True):
        """
        Get status based on thresholds

        Args:
            warning_threshold: Warning level in °C·s
            critical_threshold: Critical level in °C·s
            use_effective: If True, compare counted (effective) total; else gross

        Returns:
            status: 'normal', 'warning', or 'critical'
        """
        total = self.effective_total if use_effective else self.gross_total
        if total >= critical_threshold:
            return 'critical'
        elif total >= warning_threshold:
            return 'warning'
        else:
            return 'normal'

    def get_contribution_percentages(self, percent_of='gross'):
        """
        Calculate percentage contribution of each process

        Returns:
            dict: Process type -> percentage
        """
        if percent_of not in ('gross', 'effective'):
            percent_of = 'gross'
        denom = self.gross_total if percent_of == 'gross' else self.effective_total
        if denom <= 0:
            return {}

        percentages = {}
        for process_type, contribution in self.contributions.items():
            if denom > 0:
                percentages[process_type] = (contribution / denom) * 100
            else:
                percentages[process_type] = 0.0

        return percentages

    def compare_to_typical(self):
        """
        Compare current contributions to typical values (based on gross percentages)

        Returns:
            dict: Process type -> deviation from typical (%)
        """
        current_percentages = self.get_contribution_percentages(percent_of='gross')
        deviations = {}

        for process_type, typical_pct in self.TYPICAL_CONTRIBUTIONS.items():
            current_pct = current_percentages.get(process_type, 0.0)
            typical_pct_scaled = typical_pct * 100
            deviation = current_pct - typical_pct_scaled
            deviations[process_type] = deviation

        return deviations

    def generate_report(self):
        """
        Generate a text report of thermal budget

        Returns:
            str: Formatted report
        """
        report = []
        report.append("=" * 60)
        report.append("THERMAL BUDGET REPORT")
        report.append("=" * 60)
        report.append(f"Effective Total (counted): {self.effective_total:.2e} °C·s")
        report.append(f"Gross Total (all)     : {self.gross_total:.2e} °C·s")
        report.append(f"Status (effective)    : {self.get_status(use_effective=True)}")
        report.append("")
        report.append("Process Contributions:")
        report.append("-" * 60)

        percentages_gross = self.get_contribution_percentages(percent_of='gross')
        for process_type, contribution in sorted(self.contributions.items()):
            pct = percentages_gross.get(process_type, 0.0)
            typical_pct = self.TYPICAL_CONTRIBUTIONS.get(process_type, 0) * 100
            counted_marker = "✓" if process_type in self.COUNTED_PROCESSES else " "
            report.append(f"  {counted_marker} {process_type:15s}: {contribution:12.2e} °C·s ({pct:5.1f}% | typical: {typical_pct:5.1f}%)")

        report.append("")
        report.append("Deviations from Typical:")
        report.append("-" * 60)

        deviations = self.compare_to_typical()
        for process_type, deviation in sorted(deviations.items()):
            indicator = "↑" if deviation > 0 else "↓" if deviation < 0 else "="
            report.append(f"  {process_type:15s}: {indicator} {abs(deviation):5.1f}%")

        return "\n".join(report)


# Example usage
if __name__ == '__main__':
    calculator = ThermalBudgetCalculator()

    # Simulate adding process contributions
    calculator.add_contribution('degas', 4.5e5)
    calculator.add_contribution('flash', 2.4e5)
    calculator.add_contribution('hterm', 6.0e4)
    calculator.add_contribution('incorporation', 1.35e6)
    calculator.add_contribution('overgrowth', 9.0e5)

    print(calculator.generate_report())
