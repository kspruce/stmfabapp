"""
database_operations_enhanced.py - Enhanced database operations for fabrication steps

Add these methods to your DatabaseOperations class in operations.py
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pathlib import Path
import json


class FabricationStepOperations:
    """
    Enhanced database operations for fabrication steps
    
    These methods should be added to your DatabaseOperations class
    """
    
    # ==================== FABRICATION STEP OPERATIONS ====================
    
    def add_fabrication_step(self, device_id: int, step_number: int,
                            step_name: str, purpose: str = "",
                            requires_scan: bool = True, status: str = "pending",
                            operator: Optional[str] = None,
                            notes: str = "") -> 'FabricationStep':
        """
        Add a fabrication step to a device
        
        Args:
            device_id: Device ID
            step_number: Step number in sequence
            step_name: Name of the step
            purpose: Purpose/description of step
            requires_scan: Whether this step requires STM scan
            status: Step status (pending, in_progress, complete, skipped, failed)
            operator: Operator performing the step
            notes: Additional notes
            
        Returns:
            Created FabricationStep object
        """
        from stm_fab.db.models import FabricationStep
        
        step = FabricationStep(
            device_id=device_id,
            step_number=step_number,
            step_name=step_name,
            purpose=purpose,
            requires_scan=requires_scan,
            status=status,
            operator=operator or "Unknown",
            notes=notes,
            timestamp=datetime.now()
        )
        self.session.add(step)
        self.session.commit()
        return step
    
    def get_device_steps(self, device_id: int) -> List['FabricationStep']:
        """
        Get all fabrication steps for a device, ordered by step number
        
        Args:
            device_id: Device ID
            
        Returns:
            List of FabricationStep objects
        """
        from stm_fab.db.models import FabricationStep
        
        return (self.session.query(FabricationStep)
                .filter_by(device_id=device_id)
                .order_by(FabricationStep.step_number)
                .all())
    
    def get_step_by_id(self, step_id: int) -> Optional['FabricationStep']:
        """Get a fabrication step by ID"""
        from stm_fab.db.models import FabricationStep
        
        return self.session.query(FabricationStep).filter_by(step_id=step_id).first()
    ##
    def update_fabrication_step(
        self,
        step_id: int,
        *,
        step_name=None,
        purpose=None,
        notes=None,
        operator=None,
        requires_scan=None,
        status=None
    ):
        step = self.get_step_by_id(step_id)
        if not step:
            raise ValueError(f"Fabrication step {step_id} not found")

        is_complete = step.status == "complete"

        # HARD LOCK once complete
        if is_complete:
            if step_name is not None:
                raise ValueError("Cannot edit step name after completion")
            if purpose is not None:
                raise ValueError("Cannot edit purpose after completion")
            if requires_scan is not None:
                raise ValueError("Cannot change scan requirement after completion")

        # Apply allowed changes
        if step_name is not None:
            step.step_name = step_name

        if purpose is not None:
            step.purpose = purpose

        if requires_scan is not None:
            step.requires_scan = requires_scan

        # 🧾 Append-only notes (audit trail)
        if notes:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M")
            user = operator or step.operator or "Unknown"
            step.notes = (step.notes or "") + f"\n[{ts}] ({user}) {notes}"

        if status is not None:
            step.status = status

        if operator is not None:
            step.operator = operator

        step.timestamp = datetime.now()
        self.session.commit()
        return step

    def update_step_status(self, step_id: int, status: str,
                          notes: Optional[str] = None) -> 'FabricationStep':
        """
        Update the status of a fabrication step
        
        Args:
            step_id: Step ID
            status: New status (pending, in_progress, complete, skipped, failed)
            notes: Optional notes to append
            
        Returns:
            Updated FabricationStep object
        """
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
        """
        Delete a fabrication step and all associated scans
        
        Args:
            step_id: Step ID to delete
            
        Returns:
            True if deleted successfully
        """
        step = self.get_step_by_id(step_id)
        if not step:
            raise ValueError(f"Step {step_id} not found")
        
        # Associated scans will be deleted by cascade
        self.session.delete(step)
        self.session.commit()
        return True
    
    def initialize_device_steps(self, device_id: int,
                               step_definitions: List[Dict[str, Any]],
                               overwrite: bool = False) -> List['FabricationStep']:
        """
        Initialize all fabrication steps for a device from step definitions
        
        Args:
            device_id: Device ID
            step_definitions: List of step definition dictionaries
            overwrite: If True, delete existing steps first
            
        Returns:
            List of created FabricationStep objects
        """
        if overwrite:
            # Delete existing steps
            existing = self.get_device_steps(device_id)
            for step in existing:
                self.session.delete(step)
            self.session.commit()
        
        # Create new steps
        created_steps = []
        for step_def in step_definitions:
            step = self.add_fabrication_step(
                device_id=device_id,
                step_number=step_def.get('step_num', 0),
                step_name=step_def.get('name', 'Unnamed Step'),
                purpose=step_def.get('purpose', ''),
                requires_scan=step_def.get('requires_scan', True),
                status='pending',
                notes=step_def.get('note', '')
            )
            created_steps.append(step)
        
        return created_steps
    
    # ==================== STM SCAN OPERATIONS ====================
    
    def add_stm_scan(self, step_id: int, filename: str, filepath: str,
                     metadata: Dict[str, Any],
                     image_data: Optional[str] = None) -> 'STMScan':
        """
        Add an STM scan to a fabrication step
        
        Args:
            step_id: FabricationStep ID
            filename: Scan filename
            filepath: Full path to scan file
            metadata: Dictionary of scan metadata
            image_data: Optional Base64-encoded image
            
        Returns:
            Created STMScan object
        """
        from stm_fab.db.models import STMScan
        
        # Extract key parameters from metadata
        bias_voltage = metadata.get('bias_V')
        if isinstance(bias_voltage, str):
            try:
                bias_voltage = float(bias_voltage)
            except:
                bias_voltage = None
        
        scan = STMScan(
            step_id=step_id,
            filename=filename,
            filepath=filepath,
            scan_date=datetime.now(),
            bias_voltage=bias_voltage,
            setpoint_current=self._extract_float(metadata.get('setpoint_current')),
            scan_size_x=self._extract_float(metadata.get('scan_width_nm')),
            scan_size_y=self._extract_float(metadata.get('scan_height_nm')),
            pixels_x=self._extract_int(metadata.get('pixels_x')),
            pixels_y=self._extract_int(metadata.get('pixels_y')),
            scan_speed=self._extract_float(metadata.get('scan_speed_nm_s')),
            acquisition_time=self._extract_float(metadata.get('acq_time_s')),
            metadata_json=metadata,
            image_data=image_data
        )
        
        self.session.add(scan)
        self.session.commit()
        return scan
    
    def get_step_scans(self, step_id: int) -> List['STMScan']:
        """
        Get all STM scans for a fabrication step
        
        Args:
            step_id: FabricationStep ID
            
        Returns:
            List of STMScan objects
        """
        from stm_fab.db.models import STMScan
        
        return (self.session.query(STMScan)
                .filter_by(step_id=step_id)
                .order_by(STMScan.scan_date)
                .all())
    
    def get_scan_by_id(self, scan_id: int) -> Optional['STMScan']:
        """Get an STM scan by ID"""
        from stm_fab.db.models import STMScan
        
        return self.session.query(STMScan).filter_by(scan_id=scan_id).first()
    
    def delete_stm_scan(self, scan_id: int) -> bool:
        """
        Delete an STM scan
        
        Args:
            scan_id: Scan ID to delete
            
        Returns:
            True if deleted successfully
        """
        scan = self.get_scan_by_id(scan_id)
        if not scan:
            raise ValueError(f"Scan {scan_id} not found")
        
        self.session.delete(scan)
        self.session.commit()
        return True
    
    def get_device_scan_count(self, device_id: int) -> int:
        """Get total number of scans for a device"""
        from stm_fab.db.models import FabricationStep, STMScan
        
        count = (self.session.query(STMScan)
                .join(FabricationStep)
                .filter(FabricationStep.device_id == device_id)
                .count())
        return count
    
    def update_scan_metadata(self, scan_id: int,
                            metadata: Dict[str, Any]) -> 'STMScan':
        """
        Update metadata for an STM scan
        
        Args:
            scan_id: Scan ID
            metadata: New metadata dictionary
            
        Returns:
            Updated STMScan object
        """
        scan = self.get_scan_by_id(scan_id)
        if not scan:
            raise ValueError(f"Scan {scan_id} not found")
        
        scan.metadata_json = metadata
        
        # Update extracted fields if present
        if 'bias_V' in metadata:
            scan.bias_voltage = self._extract_float(metadata['bias_V'])
        if 'scan_speed_nm_s' in metadata:
            scan.scan_speed = self._extract_float(metadata['scan_speed_nm_s'])
        
        self.session.commit()
        return scan
    
    # ==================== HELPER METHODS ====================
    
    def _extract_float(self, value: Any) -> Optional[float]:
        """Safely extract float from various types"""
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                # Remove units if present
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
    
    # ==================== SUMMARY & STATISTICS ====================
    
    def get_step_completion_stats(self, device_id: int) -> Dict[str, Any]:
        """
        Get completion statistics for a device's fabrication steps
        
        Args:
            device_id: Device ID
            
        Returns:
            Dictionary with completion statistics
        """
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
    
    def get_step_scan_summary(self, step_id: int) -> Dict[str, Any]:
        """
        Get summary of scans for a fabrication step
        
        Args:
            step_id: Step ID
            
        Returns:
            Dictionary with scan summary
        """
        scans = self.get_step_scans(step_id)
        
        return {
            'scan_count': len(scans),
            'filenames': [s.filename for s in scans],
            'scan_dates': [s.scan_date.isoformat() if s.scan_date else None 
                          for s in scans],
            'has_images': sum(1 for s in scans if s.image_data is not None)
        }


# Example integration with existing DatabaseOperations class:
"""
# Add these methods to your existing DatabaseOperations class in operations.py:

from database_operations_enhanced import FabricationStepOperations

class DatabaseOperations(FabricationStepOperations):
    # Your existing methods...
    pass
"""
