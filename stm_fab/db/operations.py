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

from stm_fab.db.operations_enhanced import FabricationStepOperations
from datetime import datetime
from pathlib import Path

# Import BMR models - add these to your imports in operations.py
from stm_fab.db.models import BatchManufacturingRecord, BMRStep

class DatabaseOperations(FabricationStepOperations):
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


    def search_samples(self, query: str, status: Optional[str] = None) -> List[Sample]:
        q = self.session.query(Sample)
        if query:
            q = q.filter(Sample.sample_name.contains(query))
        if status:
            q = q.filter_by(status=status)
        return q.order_by(desc(Sample.creation_date)).all()


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

    def get_device_by_name_in_sample(self, sample_id: int, device_name: str) -> Optional[Device]:
        return (self.session.query(Device)
                .filter(Device.sample_id == sample_id, Device.device_name == device_name)
                .first())
   
    def update_device(self, device_id: int, **kwargs) -> Device:
        """
        Update device fields safely.
        Allowed keys: 'fabrication_start', 'fabrication_end', 'overall_status',
                      'operator', 'nominal_design_ref', 'notes', 'completion_percentage'
        """
        device = self.get_device_by_id(device_id)
        if not device:
            raise ValueError(f"Device {device_id} not found")

        allowed = {
            "fabrication_start", "fabrication_end", "overall_status",
            "operator", "nominal_design_ref", "notes", "completion_percentage"
        }
        changed = False
        for key, value in kwargs.items():
            if key in allowed:
                setattr(device, key, value)
                changed = True

        if changed:
            self.session.commit()
        return device
        
    
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
        """Rename a sample (names are allowed to be non-unique)."""
        sample = self.get_sample_by_id(sample_id)
        if not sample:
            raise ValueError(f"Sample {sample_id} not found")
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
        #from database_models import CooldownCalibration
        calibrations = self.session.query(CooldownCalibration).filter_by(sample_id=sample_id).all()
        for cal in calibrations:
            self.session.delete(cal)
        
        # 3. Delete process steps
        #from database_models import ProcessStep
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
        # Global uniqueness check
        existing = self.get_device_by_name(device_name)
        if existing:
            raise ValueError(f"Device name '{device_name}' already exists (globally)")

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
        device = self.get_device_by_id(device_id)
        if not device:
            raise ValueError(f"Device {device_id} not found")

        exists = (self.session.query(Device)
                  .filter(Device.device_name == new_name, Device.device_id != device_id)
                  .first())
        if exists:
            raise ValueError(f"Device name '{new_name}' already exists (globally)")

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

    def initialize_device_steps(self, device_id: int,
                               step_definitions: List[Dict[str, Any]],
                               overwrite: bool = False):
        """Initialize all fabrication steps for a device"""
        from stm_fab.db.models import FabricationStep
        
        if overwrite:
            # Delete existing steps
            existing = self.get_device_steps(device_id)
            for step in existing:
                self.session.delete(step)
            self.session.commit()
        
        # Create new steps
        created_steps = []
        for step_def in step_definitions:
            step = FabricationStep(
                device_id=device_id,
                step_number=step_def.get('step_num', 0),
                step_name=step_def.get('name', 'Unnamed Step'),
                purpose=step_def.get('purpose', ''),
                requires_scan=step_def.get('requires_scan', True),
                status='pending',
                operator="Unknown",
                notes=step_def.get('note', ''),
                timestamp=datetime.now()
            )
            self.session.add(step)
            created_steps.append(step)
        
        self.session.commit()
        return created_steps

    def get_step_completion_stats(self, device_id: int):
        """Get completion statistics for device steps"""
        steps = self.get_device_steps(device_id)
        
        if not steps:
            return {
                'total_steps': 0,
                'completed': 0,
                'in_progress': 0,
                'pending': 0,
                'failed': 0,
                'skipped': 0,
                'completion_percentage': 0.0
            }
        
        stats = {
            'total_steps': len(steps),
            'completed': sum(1 for s in steps if s.status == 'complete'),
            'in_progress': sum(1 for s in steps if s.status == 'in_progress'),
            'pending': sum(1 for s in steps if s.status == 'pending'),
            'failed': sum(1 for s in steps if s.status == 'failed'),
            'skipped': sum(1 for s in steps if s.status == 'skipped'),
        }
        
        stats['completion_percentage'] = (stats['completed'] / stats['total_steps']) * 100
        
        return stats




    # Add these methods to your DatabaseOperations class (around line 350, after get_step_by_id)

    def update_step_status(self, step_id: int, status: str,
                          notes: Optional[str] = None):
        """Update the status of a fabrication step"""
        from stm_fab.db.models import FabricationStep
        
        step = self.get_step_by_id(step_id)
        if not step:
            raise ValueError(f"Step {step_id} not found")
        
        step.status = status
        step.timestamp = datetime.now()
        
        if notes:
            if step.notes:
                step.notes += f"\n{notes}"
            else:
                step.notes = notes
        
        self.session.commit()
        return step

    def delete_fabrication_step(self, step_id: int) -> bool:
        """Delete a fabrication step and all associated scans"""
        from stm_fab.db.models import FabricationStep
        
        step = self.get_step_by_id(step_id)
        if not step:
            raise ValueError(f"Step {step_id} not found")
        
        self.session.delete(step)
        self.session.commit()
        return True
    
    def update_fabrication_step(self, step_id: int, **kwargs) -> FabricationStep:
        """
        Update a fabrication step
        
        Args:
            step_id: Step ID to update
            **kwargs: Fields to update (status, notes, operator, etc.)
            
        Returns:
            Updated FabricationStep object
        """
        from stm_fab.db.models import FabricationStep
        
        step = self.get_step_by_id(step_id)
        if not step:
            raise ValueError(f"Step {step_id} not found")
        
        # Update allowed fields
        for key, value in kwargs.items():
            if hasattr(step, key):
                setattr(step, key, value)
        
        self.session.commit()
        
        # Update device completion if status changed
        if 'status' in kwargs:
            self.update_device_completion(step.device_id)
        
        return step

    def get_step_scans(self, step_id: int):
        """Get all STM scans for a fabrication step"""
        from stm_fab.db.models import STMScan
        
        return (self.session.query(STMScan)
                .filter_by(step_id=step_id)
                .order_by(STMScan.scan_date)
                .all())

    def get_scan_by_id(self, scan_id: int):
        """Get an STM scan by ID"""
        from stm_fab.db.models import STMScan
        
        return self.session.query(STMScan).filter_by(scan_id=scan_id).first()

    def delete_stm_scan(self, scan_id: int) -> bool:
        """Delete an STM scan"""
        from stm_fab.db.models import STMScan
        
        scan = self.get_scan_by_id(scan_id)
        if not scan:
            raise ValueError(f"Scan {scan_id} not found")
        
        self.session.delete(scan)
        self.session.commit()
        return True

    def _extract_float(self, value: Any) -> Optional[float]:
        """Safely extract float from various types"""
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                val = value.strip().split()[0]
                return float(val)
            except:
                return None
        return None

    def _extract_int(self, value: Any) -> Optional[int]:
        """Safely extract int from various types"""
        if value is None:
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str):
            try:
                return int(value)
            except:
                return None
        return None

    def get_step_scan_summary(self, step_id: int) -> Dict[str, Any]:
        """Get summary of scans for a fabrication step"""
        scans = self.get_step_scans(step_id)
        
        return {
            'scan_count': len(scans),
            'filenames': [s.filename for s in scans],
            'scan_dates': [s.scan_date.isoformat() if s.scan_date else None 
                          for s in scans],
            'has_images': sum(1 for s in scans if s.image_data is not None)
        }
    
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
    
    # ==================== REPORT MANAGEMENT ====================
    
    def update_device_report(self, device_id: int, report_path: str) -> bool:
        """
        Link a fabrication report to a device
        
        Args:
            device_id: Device ID
            report_path: Path to HTML report file
            
        Returns:
            True if successful, False if device not found
        """
        device = self.get_device_by_id(device_id)
        if device:
            device.report_path = report_path
            device.report_generated_date = datetime.now()
            self.session.commit()
            return True
        return False
    
    
    # Add after the last existing method in DatabaseOperations class

    def create_bmr(self, device_id: int, batch_number: str, operator: str,
                   process_type: str = "SET", **kwargs):
        """Create a new BMR record linked to a device"""
        from stm_fab.db.models import BatchManufacturingRecord
        
        device = self.get_device_by_id(device_id)
        if not device:
            raise ValueError(f"Device {device_id} not found")
        
        existing = self.session.query(BatchManufacturingRecord)\
            .filter_by(batch_number=batch_number).first()
        if existing:
            raise ValueError(f"Batch number '{batch_number}' already exists")
        
        bmr = BatchManufacturingRecord(
            device_id=device_id,
            batch_number=batch_number,
            operator=operator,
            process_type=process_type,
            start_date=kwargs.get('start_date'),
            target_completion=kwargs.get('target_completion'),
            status='in_progress'
        )
        
        self.session.add(bmr)
        self.session.commit()
        return bmr
    
    
    def get_bmr_by_id(self, bmr_id: int):
        """Get BMR by ID"""
        from stm_fab.db.models import BatchManufacturingRecord
        return self.session.query(BatchManufacturingRecord)\
            .filter_by(bmr_id=bmr_id).first()
    
    
    def get_bmr_for_device(self, device_id: int, active_only: bool = True):
        """Get the most recent BMR for a device"""
        from stm_fab.db.models import BatchManufacturingRecord
        
        query = self.session.query(BatchManufacturingRecord)\
            .filter_by(device_id=device_id)
        
        if active_only:
            query = query.filter_by(status='in_progress')
        
        return query.order_by(desc(BatchManufacturingRecord.created_at)).first()
    
    
    def list_bmrs_for_device(self, device_id: int):
        """Get all BMRs for a device"""
        from stm_fab.db.models import BatchManufacturingRecord
        
        return self.session.query(BatchManufacturingRecord)\
            .filter_by(device_id=device_id)\
            .order_by(desc(BatchManufacturingRecord.created_at)).all()
    
    
    def update_bmr(self, bmr_id: int, **kwargs):
        """Update BMR metadata"""
        bmr = self.get_bmr_by_id(bmr_id)
        if not bmr:
            raise ValueError(f"BMR {bmr_id} not found")
        
        allowed = {
            'status', 'operator', 'start_date', 'completion_date', 
            'target_completion', 'json_file_path', 'pdf_file_path',
            'qc_approved', 'qc_approved_by', 'qc_approved_date',
            'pi_approved', 'pi_approved_by', 'pi_approved_date'
        }
        
        changed = False
        for key, value in kwargs.items():
            if key in allowed:
                setattr(bmr, key, value)
                changed = True
        
        if changed:
            self.session.commit()
        
        return bmr
    
    
    def link_bmr_pdf(self, bmr_id: int, pdf_path: str):
        """Link a PDF file to a BMR record"""
        bmr = self.get_bmr_by_id(bmr_id)
        if not bmr:
            raise ValueError(f"BMR {bmr_id} not found")
        
        bmr.pdf_file_path = pdf_path
        self.session.commit()
        return bmr
    
    
    def load_bmr_to_json(self, bmr_id: int):
        """Load BMR data from database into JSON-compatible dictionary"""
        bmr = self.get_bmr_by_id(bmr_id)
        if not bmr:
            raise ValueError(f"BMR {bmr_id} not found")
        
        device = bmr.device
        metadata = {
            'batch_number': bmr.batch_number,
            'device_id': device.device_name if device else '',
            'sample_name': device.sample.sample_name if device and device.sample else '',
            'operator': bmr.operator,
            'process_type': bmr.process_type,
            'start_date': bmr.start_date.isoformat() if bmr.start_date else None,
            'completion_date': bmr.completion_date.isoformat() if bmr.completion_date else None,
            'target_completion': bmr.target_completion.isoformat() if bmr.target_completion else None,
            'status': bmr.status
        }
        
        steps = []
        for bmr_step in bmr.bmr_steps:
            # Get parameters which may contain extra UI fields
            params = bmr_step.get_parameters()
            
            # Extract UI-specific fields if they exist in parameters
            step_initials = params.pop('step_initials', '')
            step_initial_time = params.pop('step_initial_time', None)
            quality_check_results = params.pop('quality_check_results', {})
            
            step_data = {
                'step_number': bmr_step.step_number,
                'step_name': bmr_step.step_name,
                'status': bmr_step.status,
                'operator_initials': bmr_step.operator_initials or '',
                'start_time': bmr_step.start_time.isoformat() if bmr_step.start_time else None,
                'end_time': bmr_step.end_time.isoformat() if bmr_step.end_time else None,
                'parameters': params,  # Now without the UI fields
                'quality_notes': bmr_step.quality_notes or '',
                'pre_check_pass': bmr_step.pre_check_pass,
                'post_check_pass': bmr_step.post_check_pass,
                'deviations': bmr_step.get_deviations(),
                'corrective_actions': bmr_step.get_corrective_actions(),
                'labview_file': bmr_step.labview_file or '',
                'sxm_scans': bmr_step.get_sxm_scans(),
                'verified_by': bmr_step.verified_by or '',
                'verified_time': bmr_step.verified_time.isoformat() if bmr_step.verified_time else None,
                'step_initials': step_initials,
                'step_initial_time': step_initial_time,
                'quality_check_results': quality_check_results
            }
            steps.append(step_data)
        
        # Try to load step_definitions from original JSON file
        step_definitions = None
        if bmr.json_file_path:
            try:
                from pathlib import Path
                json_path = Path(bmr.json_file_path)
                if json_path.exists():
                    with open(json_path, 'r', encoding='utf-8') as f:
                        orig_data = json.load(f)
                        step_definitions = orig_data.get('step_definitions')
            except:
                pass
        
        result = {
            'metadata': metadata,
            'steps': steps
        }
        
        if step_definitions:
            result['step_definitions'] = step_definitions
        
        return result
    
    
    def create_bmr_step(self, bmr_id: int, step_number: int, step_name: str, **kwargs):
        """Create a new BMR step"""
        from stm_fab.db.models import BMRStep
        
        bmr_step = BMRStep(
            bmr_id=bmr_id,
            step_number=step_number,
            step_name=step_name,
            **kwargs
        )
        self.session.add(bmr_step)
        self.session.commit()
    
    
    def delete_bmr(self, bmr_id: int):
        """Delete a BMR record and all its steps (cascade)"""
        bmr = self.get_bmr_by_id(bmr_id)
        if not bmr:
            raise ValueError(f"BMR {bmr_id} not found")
        
        self.session.delete(bmr)
        self.session.commit()
    
    
    def import_bmr_from_json(self, device_id: int, json_path: str, overwrite: bool = False):
        """
        Import BMR from JSON file and link to device
        
        Args:
            device_id: Device to link BMR to
            json_path: Path to JSON file
            overwrite: If True, delete existing BMR with same batch_number
        
        Returns:
            Created BMR object
        """
        from stm_fab.db.models import BatchManufacturingRecord, BMRStep
        from pathlib import Path
        
        # Load JSON
        json_file = Path(json_path)
        if not json_file.exists():
            raise FileNotFoundError(f"JSON file not found: {json_path}")
        
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        metadata = data.get('metadata', {})
        steps_data = data.get('steps', [])
        
        batch_number = metadata.get('batch_number', '')
        if not batch_number:
            raise ValueError("JSON missing batch_number in metadata")
        
        # Check for existing BMR with same batch number
        existing = self.session.query(BatchManufacturingRecord)\
            .filter_by(batch_number=batch_number).first()
        
        if existing:
            if overwrite:
                self.delete_bmr(existing.bmr_id)
            else:
                raise ValueError(
                    f"BMR with batch_number '{batch_number}' already exists. "
                    f"Use overwrite=True to replace it."
                )
        
        # Parse dates
        def parse_date(date_str):
            if date_str:
                try:
                    return datetime.fromisoformat(date_str)
                except:
                    return None
            return None
        
        # Create BMR
        bmr = BatchManufacturingRecord(
            device_id=device_id,
            batch_number=batch_number,
            operator=metadata.get('operator', ''),
            process_type=metadata.get('process_type', 'SET'),
            start_date=parse_date(metadata.get('start_date')),
            completion_date=parse_date(metadata.get('completion_date')),
            target_completion=parse_date(metadata.get('target_completion')),
            status=metadata.get('status', 'in_progress'),
            json_file_path=str(json_path)
        )
        
        self.session.add(bmr)
        self.session.flush()  # Get bmr_id without committing
        
        # Create steps
        for step_data in steps_data:
            # Include UI fields in parameters for storage
            params = dict(step_data.get('parameters', {}))
            params['step_initials'] = step_data.get('step_initials', '')
            params['step_initial_time'] = step_data.get('step_initial_time')
            params['quality_check_results'] = step_data.get('quality_check_results', {})
            
            bmr_step = BMRStep(
                bmr_id=bmr.bmr_id,
                step_number=step_data.get('step_number', 0),
                step_name=step_data.get('step_name', ''),
                status=step_data.get('status', 'not_started'),
                operator_initials=step_data.get('operator_initials', ''),
                start_time=parse_date(step_data.get('start_time')),
                end_time=parse_date(step_data.get('end_time')),
                quality_notes=step_data.get('quality_notes', ''),
                pre_check_pass=step_data.get('pre_check_pass', False),
                post_check_pass=step_data.get('post_check_pass', False),
                labview_file=step_data.get('labview_file', ''),
                verified_by=step_data.get('verified_by', ''),
                verified_time=parse_date(step_data.get('verified_time'))
            )
            
            # Set JSON fields using helper methods
            bmr_step.set_parameters(params)  # Now includes UI fields
            bmr_step.set_deviations(step_data.get('deviations', []))
            bmr_step.set_corrective_actions(step_data.get('corrective_actions', []))
            bmr_step.set_sxm_scans(step_data.get('sxm_scans', []))
            
            self.session.add(bmr_step)
        
        self.session.commit()
        return bmr
        return bmr_step


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
    """Get existing device by globally unique device_name, or create under given sample."""
    ops = DatabaseOperations(session)

    # If a device with this name already exists anywhere, return it
    device = ops.get_device_by_name(device_name)
    if device:
        return device

    # Otherwise, create it under the given sample (first match or new)
    sample = get_or_create_sample(session, sample_name)
    return ops.create_device(device_name, sample.sample_id, **kwargs)



if __name__ == '__main__':
    from stm_fab.db.models import init_database

    session = init_database('sqlite:///test_db.db')
    ops = DatabaseOperations(session)

    # This will return the first sample with that name or create a new one if none exist.
    sample = get_or_create_sample(
        session,
        sample_name='TEST_SAMPLE_001',
        substrate_type='Si(100)',
        supplier='UniversityWafer',
        doping_level=1e15
    )
    print(f"Sample: {sample}")

    # Global uniqueness: returns existing device anywhere, or creates new one under 'sample'
    device = get_or_create_device(
        session,
        device_name='QD_001',
        sample_name=sample.sample_name,
        operator='User'
    )
    print(f"Device: {device}")

    existing_steps = ops.get_device_steps(device.device_id)
    if not existing_steps:
        for i in range(3):
            step = ops.add_fabrication_step(
                device_id=device.device_id,
                step_number=i + 1,
                step_name=f'Step {i + 1}',
                purpose='Test step'
            )
            print(f"Added: {step}")

    summary = ops.get_device_summary(device.device_id)
    print(f"\nDevice Summary: {summary}")