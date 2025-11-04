"""
quality_checks.py - Quality check framework for STM fabrication

Implements automated and manual quality checks for device fabrication steps
Based on Phase 4 of the Enhancement Roadmap
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class CheckType(Enum):
    """Type of quality check"""
    AUTOMATED = "automated"
    MANUAL = "manual"
    VISUAL = "visual"


class CheckCategory(Enum):
    """Importance category of check"""
    CRITICAL = "critical"      # Must pass to proceed
    IMPORTANT = "important"    # Should pass, but can be waived
    INFORMATIONAL = "informational"  # For documentation only


@dataclass
class QualityCheck:
    """Individual quality check definition"""
    check_id: str
    name: str
    description: str
    check_type: CheckType
    category: CheckCategory
    expected_value: Optional[str] = None
    tolerance: Optional[float] = None
    
    def __post_init__(self):
        # Convert string enums if needed
        if isinstance(self.check_type, str):
            self.check_type = CheckType(self.check_type)
        if isinstance(self.category, str):
            self.category = CheckCategory(self.category)


@dataclass
class CheckResult:
    """Result of a quality check"""
    check: QualityCheck
    passed: bool
    actual_value: Optional[str] = None
    notes: str = ""
    checked_by: str = ""
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class QualityCheckManager:
    """
    Manages quality checks for fabrication steps
    """
    
    def __init__(self):
        self.checks_registry = {}
        self._initialize_standard_checks()
    
    def _initialize_standard_checks(self):
        """Initialize standard quality checks for process steps (not STM patterning)"""
        
        # ============ THERMAL BUDGET CHECKS ============
        self.register_check(QualityCheck(
            check_id="thermal_budget_ok",
            name="Thermal Budget Within Limits",
            description="Cumulative thermal budget has not exceeded critical threshold",
            check_type=CheckType.AUTOMATED,
            category=CheckCategory.CRITICAL,
            expected_value="< 4.0e6 °C·s"
        ))
        
        self.register_check(QualityCheck(
            check_id="thermal_budget_warning",
            name="Thermal Budget Below Warning Level",
            description="Cumulative thermal budget below warning threshold",
            check_type=CheckType.AUTOMATED,
            category=CheckCategory.IMPORTANT,
            expected_value="< 3.0e6 °C·s"
        ))
        
        # ============ DEGAS PROCESS CHECKS ============
        self.register_check(QualityCheck(
            check_id="degas_temp_reached",
            name="Degas Temperature Reached",
            description="Sample reached target degas temperature (600-650°C)",
            check_type=CheckType.AUTOMATED,
            category=CheckCategory.CRITICAL,
            expected_value="600-650°C"
        ))
        
        self.register_check(QualityCheck(
            check_id="degas_pressure_stable",
            name="Pressure Stable During Degas",
            description="Base pressure remained stable during degas",
            check_type=CheckType.AUTOMATED,
            category=CheckCategory.IMPORTANT
        ))
        
        self.register_check(QualityCheck(
            check_id="degas_time_sufficient",
            name="Degas Time Sufficient",
            description="Degas held at temperature for sufficient time",
            check_type=CheckType.AUTOMATED,
            category=CheckCategory.IMPORTANT,
            expected_value="> 30 minutes"
        ))
        
        # ============ FLASH PROCESS CHECKS ============
        self.register_check(QualityCheck(
            check_id="flash_peak_temp",
            name="Flash Peak Temperature Correct",
            description="Peak temperature within acceptable range (1550-1600°C)",
            check_type=CheckType.AUTOMATED,
            category=CheckCategory.CRITICAL,
            expected_value="1550-1600°C"
        ))
        
        self.register_check(QualityCheck(
            check_id="flash_no_overshoot",
            name="No Temperature Overshoot",
            description="Temperature did not significantly overshoot target",
            check_type=CheckType.AUTOMATED,
            category=CheckCategory.IMPORTANT,
            tolerance=0.03  # ±3%
        ))
        
        self.register_check(QualityCheck(
            check_id="flash_cooldown_smooth",
            name="Cooldown Curve Smooth",
            description="No anomalies in cooldown curve",
            check_type=CheckType.VISUAL,
            category=CheckCategory.IMPORTANT
        ))
        
        self.register_check(QualityCheck(
            check_id="flash_pressure_recovery",
            name="Pressure Recovered After Flash",
            description="Base pressure returned to normal after flash",
            check_type=CheckType.AUTOMATED,
            category=CheckCategory.IMPORTANT,
            expected_value="< 2e-10 Torr"
        ))
        
        # ============ H-TERMINATION CHECKS ============
        self.register_check(QualityCheck(
            check_id="hterm_temp_correct",
            name="H-Termination Temperature Correct",
            description="Temperature held at 330°C for H-termination",
            check_type=CheckType.AUTOMATED,
            category=CheckCategory.CRITICAL,
            expected_value="330°C ±10°C",
            tolerance=0.03  # ±10°C at 330°C
        ))
        
        self.register_check(QualityCheck(
            check_id="hterm_time_sufficient",
            name="H-Termination Time Sufficient",
            description="Exposure time sufficient for complete termination",
            check_type=CheckType.AUTOMATED,
            category=CheckCategory.CRITICAL,
            expected_value="> 10 minutes"
        ))
        
        self.register_check(QualityCheck(
            check_id="hterm_pressure_ok",
            name="H2 Pressure Appropriate",
            description="Hydrogen pressure in correct range",
            check_type=CheckType.AUTOMATED,
            category=CheckCategory.IMPORTANT,
            expected_value="1-5 Torr"
        ))
        
        # ============ DOSE CHECKS ============
        self.register_check(QualityCheck(
            check_id="dose_pressure_reached",
            name="Dose Pressure Reached",
            description="Target dose pressure achieved",
            check_type=CheckType.AUTOMATED,
            category=CheckCategory.CRITICAL,
            expected_value="5e-9 Torr"
        ))
        
        self.register_check(QualityCheck(
            check_id="dose_time_sufficient",
            name="Dose Time Sufficient",
            description="Dose time meets minimum requirement",
            check_type=CheckType.AUTOMATED,
            category=CheckCategory.CRITICAL,
            expected_value="> 10 minutes"
        ))
        
        self.register_check(QualityCheck(
            check_id="dose_stability",
            name="Dose Pressure Stable",
            description="Pressure remained stable during dose",
            check_type=CheckType.AUTOMATED,
            category=CheckCategory.IMPORTANT
        ))
        
        # ============ INCORPORATION CHECKS ============
        self.register_check(QualityCheck(
            check_id="inc_temp_correct",
            name="Incorporation Temperature Correct",
            description="Temperature held at 350°C for incorporation",
            check_type=CheckType.AUTOMATED,
            category=CheckCategory.CRITICAL,
            expected_value="350°C ±10°C",
            tolerance=0.03
        ))
        
        self.register_check(QualityCheck(
            check_id="inc_time_sufficient",
            name="Incorporation Time Sufficient",
            description="Annealing time sufficient for incorporation",
            check_type=CheckType.AUTOMATED,
            category=CheckCategory.CRITICAL,
            expected_value="> 30 minutes"
        ))
        
        self.register_check(QualityCheck(
            check_id="inc_no_overheat",
            name="No Overheating During Incorporation",
            description="Temperature did not exceed safe limits",
            check_type=CheckType.AUTOMATED,
            category=CheckCategory.CRITICAL,
            expected_value="< 380°C"
        ))
        
        # ============ OVERGROWTH CHECKS ============
        self.register_check(QualityCheck(
            check_id="overgrowth_temp_profile",
            name="Overgrowth Temperature Profile Correct",
            description="Temperature ramp and hold followed target profile",
            check_type=CheckType.AUTOMATED,
            category=CheckCategory.CRITICAL
        ))
        
        self.register_check(QualityCheck(
            check_id="overgrowth_no_spikes",
            name="No Temperature Spikes",
            description="No sudden temperature spikes during overgrowth",
            check_type=CheckType.AUTOMATED,
            category=CheckCategory.CRITICAL
        ))
        
        self.register_check(QualityCheck(
            check_id="overgrowth_pressure_stable",
            name="Pressure Stable During Overgrowth",
            description="Base pressure stable during overgrowth",
            check_type=CheckType.AUTOMATED,
            category=CheckCategory.IMPORTANT
        ))
        
        self.register_check(QualityCheck(
            check_id="overgrowth_thermal_contrib",
            name="Overgrowth Thermal Contribution Acceptable",
            description="Thermal budget contribution from overgrowth within limits",
            check_type=CheckType.AUTOMATED,
            category=CheckCategory.IMPORTANT,
            expected_value="< 1.2e6 °C·s"
        ))
        
        # ============ GENERAL PROCESS CHECKS ============
        self.register_check(QualityCheck(
            check_id="process_no_anomalies",
            name="No Process Anomalies",
            description="No unexpected events during process",
            check_type=CheckType.AUTOMATED,
            category=CheckCategory.CRITICAL
        ))
        
        self.register_check(QualityCheck(
            check_id="process_data_complete",
            name="Process Data Complete",
            description="All required process data recorded",
            check_type=CheckType.AUTOMATED,
            category=CheckCategory.IMPORTANT
        ))
        
        # ============ POST-PROCESS VISUAL CHECKS ============
        self.register_check(QualityCheck(
            check_id="surface_quality_maintained",
            name="Surface Quality Maintained",
            description="Surface remains clean and flat after process",
            check_type=CheckType.VISUAL,
            category=CheckCategory.IMPORTANT
        ))
        
        self.register_check(QualityCheck(
            check_id="no_visible_damage",
            name="No Visible Damage",
            description="No visible damage or contamination on sample",
            check_type=CheckType.VISUAL,
            category=CheckCategory.CRITICAL
        ))
    
    def register_check(self, check: QualityCheck):
        self.register_check(QualityCheck(
            check_id="thermal_budget_ok",
            name="Thermal Budget Within Limits",
            description="Cumulative thermal budget below warning threshold",
            check_type=CheckType.AUTOMATED,
            category=CheckCategory.IMPORTANT
        ))
    
    def register_check(self, check: QualityCheck):
        """Register a quality check in the system"""
        self.checks_registry[check.check_id] = check
    
    def get_check(self, check_id: str) -> Optional[QualityCheck]:
        """Get a quality check by ID"""
        return self.checks_registry.get(check_id)
    
    def get_checks_for_step(self, step_name: str) -> List[QualityCheck]:
        """
        Get relevant quality checks for a specific fabrication step
        
        Args:
            step_name: Name of the fabrication step
            
        Returns:
            List of applicable quality checks
        """
        # Map step names to check IDs
        step_check_mapping = {
            'Nominal Design Reference': [
                'stm_scan_complete', 'stm_image_quality', 'stm_target_visible'
            ],
            'alignment': [
                'stm_scan_complete', 'stm_image_quality', 'surface_clean'
            ],
            'patterning': [
                'stm_scan_complete', 'stm_image_quality', 'stm_target_visible',
                'pattern_complete', 'pattern_clean', 'pattern_dimensions'
            ],
            'dose': [
                'dose_pressure_reached', 'dose_time_sufficient', 
                'process_pressure_stable', 'process_no_anomalies'
            ],
            'post_dose': [
                'stm_scan_complete', 'stm_image_quality', 'stm_target_visible',
                'pattern_complete'
            ],
            'thermal_process': [
                'process_temp_stable', 'process_pressure_stable',
                'process_no_anomalies', 'thermal_budget_ok'
            ]
        }
        
        # Determine step category
        step_lower = step_name.lower()
        if 'dose' in step_lower and 'pre' not in step_lower:
            category = 'dose'
        elif 'post' in step_lower or 'after' in step_lower:
            category = 'post_dose'
        elif any(term in step_lower for term in ['degas', 'flash', 'incorporation', 'overgrowth']):
            category = 'thermal_process'
        elif any(term in step_lower for term in ['pattern', 'dot', 'gate', 'source', 'drain']):
            category = 'patterning'
        elif any(term in step_lower for term in ['alignment', 'step edge', 'taper']):
            category = 'alignment'
        else:
            category = 'Nominal Design Reference'
        
        # Get check IDs for this category
        check_ids = step_check_mapping.get(category, [
            'stm_scan_complete', 'stm_image_quality'
        ])
        
        # Return corresponding checks
        return [self.checks_registry[check_id] for check_id in check_ids 
                if check_id in self.checks_registry]
    
    def perform_automated_check(self, check: QualityCheck, 
                               actual_value: Any,
                               expected_value: Optional[Any] = None,
                               operator: str = "system") -> CheckResult:
        """
        Perform an automated quality check
        
        Args:
            check: QualityCheck to perform
            actual_value: Measured/actual value
            expected_value: Expected value (overrides check.expected_value if provided)
            operator: Name of operator running check
            
        Returns:
            CheckResult with pass/fail status
        """
        if expected_value is None:
            expected_value = check.expected_value
        
        passed = False
        notes = ""
        
        # Type-specific comparison logic
        if isinstance(actual_value, (int, float)) and isinstance(expected_value, (int, float)):
            # Numerical comparison with tolerance
            tolerance = check.tolerance or 0.05  # Default 5%
            diff = abs(actual_value - expected_value)
            max_diff = abs(expected_value * tolerance)
            
            passed = diff <= max_diff
            notes = f"Measured: {actual_value}, Expected: {expected_value} ±{tolerance*100}%"
        
        elif isinstance(actual_value, bool):
            # Boolean check
            passed = actual_value == True
            notes = f"Check result: {actual_value}"
        
        else:
            # String comparison
            passed = str(actual_value).lower() == str(expected_value).lower()
            notes = f"Actual: {actual_value}, Expected: {expected_value}"
        
        return CheckResult(
            check=check,
            passed=passed,
            actual_value=str(actual_value),
            notes=notes,
            checked_by=operator,
            timestamp=datetime.now()
        )
    
    def create_manual_check_result(self, check: QualityCheck,
                                   passed: bool,
                                   actual_value: str = "",
                                   notes: str = "",
                                   operator: str = "") -> CheckResult:
        """
        Create a manual check result
        
        Args:
            check: QualityCheck being performed
            passed: Whether the check passed
            actual_value: Measured/observed value
            notes: Additional notes
            operator: Name of operator performing check
            
        Returns:
            CheckResult
        """
        return CheckResult(
            check=check,
            passed=passed,
            actual_value=actual_value,
            notes=notes,
            checked_by=operator,
            timestamp=datetime.now()
        )
    
    def generate_check_report(self, results: List[CheckResult]) -> str:
        """
        Generate a formatted report of quality check results
        
        Args:
            results: List of CheckResult objects
            
        Returns:
            Formatted text report
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("QUALITY CHECK REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Group by category
        critical = [r for r in results if r.check.category == CheckCategory.CRITICAL]
        important = [r for r in results if r.check.category == CheckCategory.IMPORTANT]
        informational = [r for r in results if r.check.category == CheckCategory.INFORMATIONAL]
        
        def format_results(title: str, results_list: List[CheckResult]):
            if not results_list:
                return []
            
            lines = [f"\n{title}:", "-" * 80]
            for result in results_list:
                status = "✓ PASS" if result.passed else "✗ FAIL"
                lines.append(f"{status} | {result.check.name}")
                if result.actual_value:
                    lines.append(f"       Value: {result.actual_value}")
                if result.notes:
                    lines.append(f"       Notes: {result.notes}")
                lines.append("")
            return lines
        
        report_lines.extend(format_results("CRITICAL CHECKS", critical))
        report_lines.extend(format_results("IMPORTANT CHECKS", important))
        report_lines.extend(format_results("INFORMATIONAL CHECKS", informational))
        
        # Summary
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        failed = total - passed
        
        report_lines.append("=" * 80)
        report_lines.append("SUMMARY")
        report_lines.append("=" * 80)
        report_lines.append(f"Total Checks: {total}")
        report_lines.append(f"Passed: {passed}")
        report_lines.append(f"Failed: {failed}")
        
        # Overall status
        critical_failed = sum(1 for r in critical if not r.passed)
        if critical_failed > 0:
            report_lines.append(f"\n⚠️  WARNING: {critical_failed} CRITICAL CHECK(S) FAILED")
            report_lines.append("    Action required before proceeding")
        else:
            report_lines.append(f"\n✓ All critical checks passed")
        
        return "\n".join(report_lines)
    
    def get_check_summary_dict(self, results: List[CheckResult]) -> Dict[str, Any]:
        """
        Get a dictionary summary of check results
        
        Args:
            results: List of CheckResult objects
            
        Returns:
            Dictionary with summary statistics
        """
        critical = [r for r in results if r.check.category == CheckCategory.CRITICAL]
        important = [r for r in results if r.check.category == CheckCategory.IMPORTANT]
        
        return {
            'total_checks': len(results),
            'passed': sum(1 for r in results if r.passed),
            'failed': sum(1 for r in results if not r.passed),
            'critical_checks': len(critical),
            'critical_passed': sum(1 for r in critical if r.passed),
            'critical_failed': sum(1 for r in critical if not r.passed),
            'important_checks': len(important),
            'important_passed': sum(1 for r in important if r.passed),
            'important_failed': sum(1 for r in important if not r.passed),
            'all_critical_passed': all(r.passed for r in critical),
            'overall_status': 'pass' if all(r.passed for r in critical) else 'fail'
        }


# Example usage
if __name__ == '__main__':
    # Initialize quality check manager
    qc_manager = QualityCheckManager()
    
    # Get checks for a patterning step
    checks = qc_manager.get_checks_for_step("50nm Quantum Dot Patterning")
    
    print("Quality Checks for Quantum Dot Patterning:")
    print("=" * 60)
    for check in checks:
        print(f"\n{check.name}")
        print(f"  Type: {check.check_type.value}")
        print(f"  Category: {check.category.value}")
        print(f"  Description: {check.description}")
    
    # Simulate performing checks
    results = []
    
    # Automated check
    stm_complete_check = qc_manager.get_check('stm_scan_complete')
    result1 = qc_manager.perform_automated_check(
        stm_complete_check,
        actual_value=True,
        operator="User"
    )
    results.append(result1)
    
    # Manual check
    pattern_check = qc_manager.get_check('pattern_complete')
    result2 = qc_manager.create_manual_check_result(
        pattern_check,
        passed=True,
        notes="All quantum dots visible and well-defined",
        operator="User"
    )
    results.append(result2)
    
    # Generate report
    print("\n" + qc_manager.generate_check_report(results))
    
    # Get summary
    summary = qc_manager.get_check_summary_dict(results)
    print(f"\nOverall Status: {summary['overall_status']}")