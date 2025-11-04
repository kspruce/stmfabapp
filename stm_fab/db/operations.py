"""
database_operations.py - Helper functions for database CRUD operations

Provides convenient functions for working with the STM fabrication database
"""

from sqlalchemy.orm import Session
from sqlalchemy import desc, func
from typing import List, Optional, Dict, Any
from datetime import datetime
import json

from stm_fab.db.models import (
    Sample, Device, FabricationStep, STMScan, ProcessStep,
    ThermalBudget, CooldownCalibration, QualityCheck as DBQualityCheck,
    ProcessMetrics
)


class DatabaseOperations:
    """Helper class for database operations"""
    
    def __init__(self, session: Session):
        self.session = session
    
    # ==================== SAMPLE OPERATIONS ====================
    
    def create_sample(self, sample_name: str, substrate_type: str = "Si(100)",
                      labview_folder_path: Optional[str] = None,
                      scan_folder_path: Optional[str] = None,
                      **kwargs) -> Sample:
        sample = Sample(
            sample_name=sample_name,
            substrate_type=substrate_type,
            labview_folder_path=labview_folder_path,
            scan_folder_path=scan_folder_path,
            **kwargs
        )
        self.session.add(sample)
        self.session.commit()
        # Create associated thermal budget record
        thermal_budget = ThermalBudget(sample_id=sample.sample_id)
        self.session.add(thermal_budget)
        self.session.commit()
        return sample

    def get_sample_paths(self, sample_id: int) -> Dict[str, Optional[str]]:
        """Return labview and scan folder paths for a sample."""
        sample = self.get_sample_by_id(sample_id)
        if not sample:
            raise ValueError(f"Sample {sample_id} not found")
        return {
            "labview_folder_path": getattr(sample, "labview_folder_path", None),
            "scan_folder_path": getattr(sample, "scan_folder_path", None),
        }

    def set_sample_paths(self, sample_id: int,
                         labview_folder_path: Optional[str] = None,
                         scan_folder_path: Optional[str] = None) -> Sample:
        """Update one or both path fields for a sample."""
        sample = self.get_sample_by_id(sample_id)
        if not sample:
            raise ValueError(f"Sample {sample_id} not found")

        changed = False
        if labview_folder_path is not None and labview_folder_path != sample.labview_folder_path:
            sample.labview_folder_path = labview_folder_path
            changed = True
        if scan_folder_path is not None and scan_folder_path != sample.scan_folder_path:
            sample.scan_folder_path = scan_folder_path
            changed = True

        if changed:
            self.session.commit()
        return sample
    
    def get_sample_by_name(self, sample_name: str) -> Optional[Sample]:
        """Get sample by name"""
        return self.session.query(Sample).filter_by(sample_name=sample_name).first()
    
    def get_sample_by_id(self, sample_id: int) -> Optional[Sample]:
        """Get sample by ID"""
        return self.session.query(Sample).filter_by(sample_id=sample_id).first()
    
    def list_samples(self, status: Optional[str] = None) -> List[Sample]:
        """
        List all samples, optionally filtered by status
        
        Args:
            status: Optional status filter ('active', 'complete', 'failed')
            
        Returns:
            List of Sample objects
        """
        query = self.session.query(Sample)
        if status:
            query = query.filter_by(status=status)
        return query.order_by(desc(Sample.creation_date)).all()
    
    def update_sample(self, sample_id: int, **kwargs) -> Sample:
        """Update sample parameters"""
        sample = self.get_sample_by_id(sample_id)
        if not sample:
            raise ValueError(f"Sample {sample_id} not found")
        
        for key, value in kwargs.items():
            if hasattr(sample, key):
                setattr(sample, key, value)
        
        self.session.commit()
        return sample
    
    def rename_sample(self, sample_id: int, new_name: str) -> Sample:
        """Rename a sample"""
        # Check if new name already exists
        existing = self.get_sample_by_name(new_name)
        if existing and existing.sample_id != sample_id:
            raise ValueError(f"Sample name '{new_name}' already exists")
        
        sample = self.get_sample_by_id(sample_id)
        if not sample:
            raise ValueError(f"Sample {sample_id} not found")
        
        old_name = sample.sample_name
        sample.sample_name = new_name
        self.session.commit()
        
        return sample
    
    def delete_sample(self, sample_id: int, force: bool = False) -> bool:
        """
        Delete a sample (and all associated data if force=True)
        
        Args:
            sample_id: Sample ID to delete
            force: If True, delete even if there are associated devices
            
        Returns:
            True if deleted successfully
        """
        sample = self.get_sample_by_id(sample_id)
        if not sample:
            raise ValueError(f"Sample {sample_id} not found")
        
        # Check for associated devices
        devices = self.list_devices(sample_id=sample_id)
        if devices and not force:
            raise ValueError(
                f"Sample has {len(devices)} associated device(s). "
                "Use force=True to delete anyway (this will delete all devices)."
            )
        
        # Explicitly delete related records to avoid constraint violations
        # 1. Delete thermal budget
        thermal_budget = self.get_thermal_budget(sample_id)
        if thermal_budget:
            self.session.delete(thermal_budget)
        
        # 2. Delete cooldown calibrations
        from database_models import CooldownCalibration
        calibrations = self.session.query(CooldownCalibration).filter_by(sample_id=sample_id).all()
        for cal in calibrations:
            self.session.delete(cal)
        
        # 3. Delete process steps
        from database_models import ProcessStep
        process_steps = self.session.query(ProcessStep).filter_by(sample_id=sample_id).all()
        for step in process_steps:
            self.session.delete(step)
        
        # 4. Delete devices (cascade will handle fabrication steps and scans)
        for device in devices:
            self.session.delete(device)
        
        # 5. Finally delete the sample
        self.session.delete(sample)
        self.session.commit()
        return True
    
    # ==================== DEVICE OPERATIONS ====================
    
    def create_device(self, device_name: str, sample_id: int, **kwargs) -> Device:
        """
        Create a new device
        
        Args:
            device_name: Unique device name
            sample_id: ID of parent sample
            **kwargs: Additional device parameters
            
        Returns:
            Created Device object
        """
        device = Device(
            device_name=device_name,
            sample_id=sample_id,
            fabrication_start=datetime.now(),
            **kwargs
        )
        self.session.add(device)
        self.session.commit()
        return device
    
    def get_device_by_name(self, device_name: str) -> Optional[Device]:
        """Get device by name"""
        return self.session.query(Device).filter_by(device_name=device_name).first()
    
    def get_device_by_id(self, device_id: int) -> Optional[Device]:
        """Get device by ID"""
        return self.session.query(Device).filter_by(device_id=device_id).first()
    
    def list_devices(self, sample_id: Optional[int] = None,
                    status: Optional[str] = None) -> List[Device]:
        """
        List devices, optionally filtered
        
        Args:
            sample_id: Optional sample ID filter
            status: Optional status filter
            
        Returns:
            List of Device objects
        """
        query = self.session.query(Device)
        if sample_id:
            query = query.filter_by(sample_id=sample_id)
        if status:
            query = query.filter_by(overall_status=status)
        return query.order_by(desc(Device.fabrication_start)).all()
    
    def update_device_completion(self, device_id: int) -> Device:
        """Update device completion percentage"""
        device = self.get_device_by_id(device_id)
        if not device:
            raise ValueError(f"Device {device_id} not found")
        
        device.completion_percentage = device.calculate_completion()
        self.session.commit()
        return device
    
    def rename_device(self, device_id: int, new_name: str) -> Device:
        """Rename a device"""
        # Check if new name already exists
        existing = self.get_device_by_name(new_name)
        if existing and existing.device_id != device_id:
            raise ValueError(f"Device name '{new_name}' already exists")
        
        device = self.get_device_by_id(device_id)
        if not device:
            raise ValueError(f"Device {device_id} not found")
        
        old_name = device.device_name
        device.device_name = new_name
        self.session.commit()
        
        return device
    
    def delete_device(self, device_id: int) -> bool:
        """
        Delete a device and all associated fabrication steps/scans
        
        Args:
            device_id: Device ID to delete
            
        Returns:
            True if deleted successfully
        """
        device = self.get_device_by_id(device_id)
        if not device:
            raise ValueError(f"Device {device_id} not found")
        
        self.session.delete(device)
        self.session.commit()
        return True
    
    # ==================== FABRICATION STEP OPERATIONS ====================
    
    def add_fabrication_step(self, device_id: int, step_number: int,
                            step_name: str, **kwargs) -> FabricationStep:
        """
        Add a fabrication step to a device
        
        Args:
            device_id: Parent device ID
            step_number: Step number in sequence
            step_name: Name of the step
            **kwargs: Additional step parameters
            
        Returns:
            Created FabricationStep object
        """
        step = FabricationStep(
            device_id=device_id,
            step_number=step_number,
            step_name=step_name,
            timestamp=datetime.now(),
            **kwargs
        )
        self.session.add(step)
        self.session.commit()
        
        # Update device completion
        self.update_device_completion(device_id)
        
        return step
    
    def get_device_steps(self, device_id: int) -> List[FabricationStep]:
        """Get all steps for a device"""
        return (self.session.query(FabricationStep)
                .filter_by(device_id=device_id)
                .order_by(FabricationStep.step_number)
                .all())
    
    def update_step_status(self, step_id: int, status: str, **kwargs) -> FabricationStep:
        """Update fabrication step status"""
        step = self.session.query(FabricationStep).filter_by(step_id=step_id).first()
        if not step:
            raise ValueError(f"Step {step_id} not found")
        
        step.status = status
        for key, value in kwargs.items():
            if hasattr(step, key):
                setattr(step, key, value)
        
        self.session.commit()
        
        # Update device completion
        self.update_device_completion(step.device_id)
        
        return step
    
    # ==================== STM SCAN OPERATIONS ====================
    
    def add_stm_scan(self, step_id: int, filename: str, filepath: str,
                    metadata: Dict[str, Any], **kwargs) -> STMScan:
        """
        Add STM scan to a fabrication step
        
        Args:
            step_id: Parent fabrication step ID
            filename: Scan filename
            filepath: Full path to scan file
            metadata: Parsed metadata dictionary
            **kwargs: Additional scan parameters
            
        Returns:
            Created STMScan object
        """
        scan = STMScan(
            step_id=step_id,
            filename=filename,
            filepath=filepath,
            metadata_json=metadata,
            scan_date=metadata.get('scan_date'),
            bias_voltage=metadata.get('bias_voltage'),
            setpoint_current=metadata.get('setpoint_current'),
            scan_size_x=metadata.get('scan_size_x'),
            scan_size_y=metadata.get('scan_size_y'),
            pixels_x=metadata.get('pixels_x'),
            pixels_y=metadata.get('pixels_y'),
            scan_speed=metadata.get('scan_speed'),
            acquisition_time=metadata.get('acquisition_time'),
            **kwargs
        )
        self.session.add(scan)
        self.session.commit()
        return scan
    
    def get_step_scans(self, step_id: int) -> List[STMScan]:
        """Get all STM scans for a step"""
        return (self.session.query(STMScan)
                .filter_by(step_id=step_id)
                .order_by(STMScan.scan_date)
                .all())
    
    # ==================== PROCESS STEP OPERATIONS ====================
    
    def add_process_step(self, sample_id: int, process_type: str,
                        labview_file_path: str, parsed_data: Dict[str, Any],
                        **kwargs) -> ProcessStep:
        """
        Add a process step (degas, flash, etc.) with LabVIEW data
        
        Args:
            sample_id: Parent sample ID
            process_type: Type of process
            labview_file_path: Path to LabVIEW file
            parsed_data: Parsed LabVIEW data
            **kwargs: Additional process parameters
            
        Returns:
            Created ProcessStep object
        """
        metrics = parsed_data.get('metrics', {})
        
        process = ProcessStep(
            sample_id=sample_id,
            process_type=process_type,
            labview_file_path=labview_file_path,
            parameters_json=parsed_data,
            max_temperature=metrics.get('peak_temperature'),
            thermal_budget_contribution=metrics.get('thermal_budget', 0),
            peak_pressure=metrics.get('peak_pressure'),
            base_pressure=metrics.get('base_pressure'),
            **kwargs
        )
        self.session.add(process)
        self.session.commit()
        
        # Update thermal budget
        if metrics.get('thermal_budget'):
            self.update_thermal_budget(sample_id, process_type, 
                                     metrics['thermal_budget'])
        
        return process
    
    def get_sample_processes(self, sample_id: int) -> List[ProcessStep]:
        """Get all process steps for a sample"""
        return (self.session.query(ProcessStep)
                .filter_by(sample_id=sample_id)
                .order_by(ProcessStep.start_time)
                .all())
    
    # ==================== THERMAL BUDGET OPERATIONS ====================
    
    def get_thermal_budget(self, sample_id: int) -> Optional[ThermalBudget]:
        """Get thermal budget for a sample"""
        return self.session.query(ThermalBudget).filter_by(sample_id=sample_id).first()
    
    def update_thermal_budget(self, sample_id: int, process_type: str,
                             contribution: float) -> ThermalBudget:
        """
        Update thermal budget for a sample
        
        Args:
            sample_id: Sample ID
            process_type: Type of process contributing
            contribution: Thermal budget contribution (°C·s)
            
        Returns:
            Updated ThermalBudget object
        """
        budget = self.get_thermal_budget(sample_id)
        if not budget:
            budget = ThermalBudget(sample_id=sample_id)
            self.session.add(budget)
        
        # Add to appropriate contribution
        if process_type == 'degas':
            budget.degas_contribution += contribution
        elif process_type == 'flash':
            budget.flash_contribution += contribution
        elif process_type == 'hterm':
            budget.hterm_contribution += contribution
        elif process_type == 'incorporation':
            budget.incorporation_contribution += contribution
        elif process_type == 'overgrowth':
            budget.overgrowth_contribution += contribution
        else:
            budget.other_contribution += contribution
        
        # Recalculate total
        budget.calculate_total()
        self.session.commit()
        
        return budget
    
    # ==================== COOLDOWN CALIBRATION OPERATIONS ====================
    
    def add_cooldown_calibration(self, sample_id: int, 
                                 labview_file_path: str,
                                 calibration_data: Dict[str, Any]) -> CooldownCalibration:
        """
        Add cooldown calibration for a sample
        
        Args:
            sample_id: Sample ID
            labview_file_path: Path to flash file used
            calibration_data: Calibration data dictionary
            
        Returns:
            Created CooldownCalibration object
        """
        cal = CooldownCalibration(
            sample_id=sample_id,
            labview_file_path=labview_file_path,
            curve_data_json=calibration_data.get('data_points'),
            fit_coefficients=calibration_data.get('fit_coefficients'),
            r_squared=calibration_data.get('r_squared'),
            rmse=calibration_data.get('rmse')
        )
        
        # Calculate standard setpoints if available
        if 'standard_setpoints' in calibration_data:
            setpoints = calibration_data['standard_setpoints']
            if 'h_termination' in setpoints:
                cal.current_at_330C = setpoints['h_termination'].get('current')
            if 'incorporation' in setpoints:
                cal.current_at_350C = setpoints['incorporation'].get('current')
        
        self.session.add(cal)
        self.session.commit()
        return cal
    
    def get_latest_calibration(self, sample_id: int) -> Optional[CooldownCalibration]:
        """Get most recent cooldown calibration for a sample"""
        return (self.session.query(CooldownCalibration)
                .filter_by(sample_id=sample_id)
                .order_by(desc(CooldownCalibration.calibration_date))
                .first())
    
    # ==================== QUALITY CHECK OPERATIONS ====================
    
    def add_quality_check(self, step_id: int, check_name: str,
                         passed: bool, **kwargs) -> DBQualityCheck:
        """
        Add quality check result to a step
        
        Args:
            step_id: Fabrication step ID
            check_name: Name of the check
            passed: Whether check passed
            **kwargs: Additional check parameters
            
        Returns:
            Created QualityCheck object
        """
        check = DBQualityCheck(
            step_id=step_id,
            check_name=check_name,
            passed=passed,
            check_timestamp=datetime.now(),
            **kwargs
        )
        self.session.add(check)
        self.session.commit()
        
        # Update step's quality_check_passed flag
        step = self.session.query(FabricationStep).filter_by(step_id=step_id).first()
        if step:
            # Check if all quality checks passed
            all_checks = self.get_step_quality_checks(step_id)
            step.quality_check_passed = all(c.passed for c in all_checks)
            self.session.commit()
        
        return check
    
    def get_step_quality_checks(self, step_id: int) -> List[DBQualityCheck]:
        """Get all quality checks for a step"""
        return (self.session.query(DBQualityCheck)
                .filter_by(step_id=step_id)
                .order_by(DBQualityCheck.check_timestamp)
                .all())
    
    # ==================== QUERY AND ANALYSIS ====================
    
    def get_device_summary(self, device_id: int) -> Dict[str, Any]:
        """
        Get comprehensive summary of a device
        
        Args:
            device_id: Device ID
            
        Returns:
            Summary dictionary
        """
        device = self.get_device_by_id(device_id)
        if not device:
            raise ValueError(f"Device {device_id} not found")
        
        steps = self.get_device_steps(device_id)
        
        summary = {
            'device_name': device.device_name,
            'sample_name': device.sample.sample_name,
            'status': device.overall_status,
            'completion': device.completion_percentage,
            'start_date': device.fabrication_start,
            'end_date': device.fabrication_end,
            'operator': device.operator,
            'total_steps': len(steps),
            'completed_steps': sum(1 for s in steps if s.status == 'complete'),
            'failed_steps': sum(1 for s in steps if s.status == 'failed'),
            'total_scans': sum(len(self.get_step_scans(s.step_id)) for s in steps)
        }
        
        return summary
    
    def get_sample_summary(self, sample_id: int) -> Dict[str, Any]:
        """
        Get comprehensive summary of a sample
        
        Args:
            sample_id: Sample ID
            
        Returns:
            Summary dictionary
        """
        sample = self.get_sample_by_id(sample_id)
        if not sample:
            raise ValueError(f"Sample {sample_id} not found")
        
        devices = self.list_devices(sample_id=sample_id)
        processes = self.get_sample_processes(sample_id)
        thermal_budget = self.get_thermal_budget(sample_id)
        
        summary = {
            'sample_name': sample.sample_name,
            'substrate_type': sample.substrate_type,
            'status': sample.status,
            'creation_date': sample.creation_date,
            'total_devices': len(devices),
            'completed_devices': sum(1 for d in devices if d.overall_status == 'complete'),
            'active_devices': sum(1 for d in devices if d.overall_status == 'in_progress'),
            'total_processes': len(processes),
            'thermal_budget': thermal_budget.total_budget if thermal_budget else 0,
            'thermal_budget_status': thermal_budget.status() if thermal_budget else 'unknown',
            'labview_folder_path': getattr(sample, 'labview_folder_path', None),
            'scan_folder_path': getattr(sample, 'scan_folder_path', None),
        }
        
        return summary
    
    def search_devices(self, query: str) -> List[Device]:
        """
        Search for devices by name
        
        Args:
            query: Search query
            
        Returns:
            List of matching devices
        """
        return (self.session.query(Device)
                .filter(Device.device_name.contains(query))
                .order_by(desc(Device.fabrication_start))
                .all())
    
    def get_recent_activity(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent fabrication activity
        
        Args:
            limit: Maximum number of items to return
            
        Returns:
            List of activity dictionaries
        """
        # Get recent devices
        recent_devices = (self.session.query(Device)
                         .order_by(desc(Device.fabrication_start))
                         .limit(limit)
                         .all())
        
        activity = []
        for device in recent_devices:
            activity.append({
                'type': 'device_created',
                'timestamp': device.fabrication_start,
                'device_name': device.device_name,
                'sample_name': device.sample.sample_name,
                'status': device.overall_status
            })
        
        return sorted(activity, key=lambda x: x['timestamp'], reverse=True)
    
    # ==================== PROCESS METRICS OPERATIONS ====================
    
    def add_process_metrics(self, process_step_id: int, file_path: str, 
                           file_type: str, metrics_dict: Dict[str, Any]):
        """
        Add computed process metrics for a process step
        
        Args:
            process_step_id: ProcessStep ID to attach metrics to
            file_path: Path to LabVIEW file
            file_type: Type of file (dose, flash, susi, etc.)
            metrics_dict: Dict of computed metrics
            
        Returns:
            Created ProcessMetrics object
        """
        from stm_fab.db.models import ProcessMetrics
        
        # Check if metrics already exist
        existing = (self.session.query(ProcessMetrics)
                    .filter_by(process_step_id=process_step_id)
                    .first())
        
        if existing:
            # Update existing
            existing.file_path = file_path
            existing.file_type = file_type
            existing.metrics_json = metrics_dict
            existing.updated_at = datetime.now()
            self.session.commit()
            return existing
        else:
            # Create new
            metrics = ProcessMetrics(
                process_step_id=process_step_id,
                file_path=file_path,
                file_type=file_type,
                metrics_json=metrics_dict
            )
            self.session.add(metrics)
            self.session.commit()
            return metrics
    
    def get_metrics_for_step(self, process_step_id: int):
        """Get metrics for a process step"""
        from stm_fab.db.models import ProcessMetrics
        
        return (self.session.query(ProcessMetrics)
                .filter_by(process_step_id=process_step_id)
                .first())
    
    def find_process_step_by_file(self, sample_id: int, file_path: str):
        """
        Find a ProcessStep by sample ID and file path
        
        Args:
            sample_id: Sample ID
            file_path: LabVIEW file path
            
        Returns:
            ProcessStep if found, None otherwise
        """
        return (self.session.query(ProcessStep)
                .filter_by(sample_id=sample_id, labview_file_path=file_path)
                .first())


# Standalone helper functions for common operations

def get_or_create_sample(session: Session, sample_name: str, 
                        **kwargs) -> Sample:
    """Get existing sample or create new one"""
    ops = DatabaseOperations(session)
    sample = ops.get_sample_by_name(sample_name)
    if not sample:
        sample = ops.create_sample(sample_name, **kwargs)
    return sample


def get_or_create_device(session: Session, device_name: str,
                        sample_name: str, **kwargs) -> Device:
    """Get existing device or create new one"""
    ops = DatabaseOperations(session)
    device = ops.get_device_by_name(device_name)
    if not device:
        # Get or create sample
        sample = get_or_create_sample(session, sample_name)
        device = ops.create_device(device_name, sample.sample_id, **kwargs)
    return device


if __name__ == '__main__':
    # Example usage
    from database_models import init_database
    
    session = init_database('sqlite:///test_db.db')
    ops = DatabaseOperations(session)
    
    # Create sample
    sample = ops.create_sample(
        sample_name='TEST_SAMPLE_001',
        substrate_type='Si(100)',
        supplier='UniversityWafer',
        doping_level=1e15
    )
    print(f"Created: {sample}")
    
    # Create device
    device = ops.create_device(
        device_name='QD_001',
        sample_id=sample.sample_id,
        operator='User'
    )
    print(f"Created: {device}")
    
    # Add fabrication steps
    for i in range(3):
        step = ops.add_fabrication_step(
            device_id=device.device_id,
            step_number=i+1,
            step_name=f'Step {i+1}',
            purpose='Test step'
        )
        print(f"Added: {step}")
    
    # Get summary
    summary = ops.get_device_summary(device.device_id)
    print(f"\nDevice Summary: {summary}")