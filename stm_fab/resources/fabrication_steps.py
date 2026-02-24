"""
fabrication_steps.py - Fabrication Step Management Module

Provides:
- Standard SET fabrication protocol (14 steps)
- Custom step creation and management
- Step status tracking
- Integration with database models
"""

from typing import List, Dict, Optional, Any
from datetime import datetime
from pathlib import Path

# Standard SET (Single Electron Transistor) Fabrication Protocol
STANDARD_SET_STEPS = [
    {
        'step_num': 1,
        'name': 'Nominal Design Reference',
        'purpose': 'Document GDS design for fabrication reference',
        'requires_scan': True,
        'category': 'design'
    },
    {
        'step_num': 2,
        'name': '12μm Step Edge Mapping',
        'purpose': 'Map step edge structure for alignment and future device location',
        'requires_scan': True,
        'category': 'alignment'
    },
    {
        'step_num': 3,
        'name': '5μm Alignment Layer Area',
        'purpose': 'Verify terrace cleanliness and identify suitable fabrication area',
        'requires_scan': True,
        'category': 'alignment'
    },
    {
        'step_num': 4,
        'name': '2.5μm Alignment Layer Tapers',
        'purpose': 'Create tapers connecting μm scale to nm scale structures',
        'requires_scan': True,
        'category': 'patterning'
    },
    {
        'step_num': 5,
        'name': '700nm Inner Region Patterning',
        'purpose': 'Pattern source, drain, and gates with gaps for quantum dot',
        'requires_scan': True,
        'category': 'patterning'
    },
    {
        'step_num': 6,
        'name': '50nm Quantum Dot Patterning',
        'purpose': 'Pattern quantum dots with correct dimer row gaps',
        'requires_scan': True,
        'category': 'patterning'
    },
    {
        'step_num': 7,
        'name': '50nm Pre-Dose Tunnel Region (Dual Bias)',
        'purpose': 'Document tunnel region before XH3 dosing',
        'requires_scan': True,
        'category': 'characterization',
        'multiple': True
    },
    {
        'step_num': 8,
        'name': '500nm Pre-Dose Full Device (45° Dual Bias)',
        'purpose': 'Full inner device documentation before dosing',
        'requires_scan': True,
        'category': 'characterization',
        'multiple': True
    },
    {
        'step_num': 9,
        'name': 'Contact Leg Verification',
        'purpose': 'Ensure complete connection and full desorption for each contact leg',
        'requires_scan': True,
        'category': 'verification',
        'multiple': True,
        'note': 'Repeat for each leg: 4-6 total'
    },
    {
        'step_num': 10,
        'name': 'XH3 DOSING STEP (NO STM)',
        'purpose': 'Saturation dose, 10 mins, 5×10⁻⁹ Torr',
        'requires_scan': False,
        'category': 'processing',
        'special_fields': ['elog_entry', 'time_started', 'time_completed', 'dose_confirmed']
    },
    {
        'step_num': 11,
        'name': '500nm Post-Dose Full Device (45° Dual Bias)',
        'purpose': 'Document full inner device after XH3 dosing',
        'requires_scan': True,
        'category': 'characterization',
        'multiple': True
    },
    {
        'step_num': 12,
        'name': '50nm Post-Dose Tunnel Region (Dual Bias)',
        'purpose': 'High-resolution documentation of tunnel region after dosing',
        'requires_scan': True,
        'category': 'characterization',
        'multiple': True
    },
    {
        'step_num': 13,
        'name': 'Dopant Incorporation Imaging (Optional)',
        'purpose': 'Image device after dopant incorporation process',
        'requires_scan': True,
        'category': 'characterization',
        'optional': True,
        'multiple': True
    },
    {
        'step_num': 14,
        'name': 'Overgrowth Imaging (200nm, 100nm, 30nm)',
        'purpose': 'Document device at multiple scales during/after overgrowth',
        'requires_scan': True,
        'category': 'characterization',
        'multiple': True,
        'note': 'Multiple scans at different scales'
    }
]


class StepDefinition:
    """
    Represents a fabrication step definition (template for actual steps)
    """
    
    def __init__(self, step_num: int, name: str, purpose: str = "",
                 requires_scan: bool = True, category: str = "general",
                 optional: bool = False, multiple: bool = False,
                 note: str = "", special_fields: Optional[List[str]] = None):
        self.step_num = step_num
        self.name = name
        self.purpose = purpose
        self.requires_scan = requires_scan
        self.category = category
        self.optional = optional
        self.multiple = multiple
        self.note = note
        self.special_fields = special_fields or []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'step_num': self.step_num,
            'name': self.name,
            'purpose': self.purpose,
            'requires_scan': self.requires_scan,
            'category': self.category,
            'optional': self.optional,
            'multiple': self.multiple,
            'note': self.note,
            'special_fields': self.special_fields
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StepDefinition':
        """Create from dictionary"""
        return cls(
            step_num=data.get('step_num', 0),
            name=data.get('name', ''),
            purpose=data.get('purpose', ''),
            requires_scan=data.get('requires_scan', True),
            category=data.get('category', 'general'),
            optional=data.get('optional', False),
            multiple=data.get('multiple', False),
            note=data.get('note', ''),
            special_fields=data.get('special_fields', [])
        )
    
    def __repr__(self):
        return f"<StepDefinition(num={self.step_num}, name='{self.name}')>"


class FabricationStepsManager:
    """
    Manages fabrication step definitions and protocols
    """
    
    def __init__(self):
        self.standard_steps = [StepDefinition.from_dict(s) for s in STANDARD_SET_STEPS]
        self.custom_protocols = {}  # protocol_name -> list of StepDefinitions
    
    def get_standard_steps(self) -> List[StepDefinition]:
        """Get the standard SET fabrication steps"""
        return self.standard_steps.copy()
    
    def create_custom_protocol(self, protocol_name: str,
                               steps: List[Dict[str, Any]]) -> List[StepDefinition]:
        """
        Create a custom fabrication protocol
        
        Args:
            protocol_name: Name for this custom protocol
            steps: List of step definition dictionaries
            
        Returns:
            List of StepDefinition objects
        """
        step_defs = [StepDefinition.from_dict(s) for s in steps]
        self.custom_protocols[protocol_name] = step_defs
        return step_defs
    
    def get_protocol(self, protocol_name: str) -> Optional[List[StepDefinition]]:
        """Get a custom protocol by name"""
        if protocol_name == "STANDARD_SET":
            return self.get_standard_steps()
        return self.custom_protocols.get(protocol_name)
    
    def add_custom_step(self, step_num: int, name: str, **kwargs) -> StepDefinition:
        """
        Create a single custom step definition
        
        Args:
            step_num: Step number
            name: Step name
            **kwargs: Additional step parameters
            
        Returns:
            StepDefinition object
        """
        return StepDefinition(step_num=step_num, name=name, **kwargs)
    
    def export_protocol(self, protocol_name: str) -> Optional[List[Dict[str, Any]]]:
        """
        Export protocol to dictionary format for saving
        
        Args:
            protocol_name: Name of protocol to export
            
        Returns:
            List of step dictionaries or None if not found
        """
        steps = self.get_protocol(protocol_name)
        if steps:
            return [s.to_dict() for s in steps]
        return None
    
    def get_step_by_number(self, step_num: int,
                          protocol: str = "STANDARD_SET") -> Optional[StepDefinition]:
        """
        Get a specific step by number from a protocol
        
        Args:
            step_num: Step number to retrieve
            protocol: Protocol name (default: STANDARD_SET)
            
        Returns:
            StepDefinition or None if not found
        """
        steps = self.get_protocol(protocol)
        if steps:
            for step in steps:
                if step.step_num == step_num:
                    return step
        return None
    
    def get_step_categories(self, protocol: str = "STANDARD_SET") -> List[str]:
        """Get all unique categories in a protocol"""
        steps = self.get_protocol(protocol)
        if steps:
            return sorted(list(set(s.category for s in steps)))
        return []
    
    def get_steps_by_category(self, category: str,
                             protocol: str = "STANDARD_SET") -> List[StepDefinition]:
        """Get all steps in a specific category"""
        steps = self.get_protocol(protocol)
        if steps:
            return [s for s in steps if s.category == category]
        return []
    
    def validate_step_sequence(self, steps: List[StepDefinition]) -> List[str]:
        """
        Validate a sequence of steps for completeness and ordering
        
        Returns:
            List of validation warnings/errors
        """
        issues = []
        
        # Check for duplicate step numbers
        step_nums = [s.step_num for s in steps]
        if len(step_nums) != len(set(step_nums)):
            issues.append("Duplicate step numbers found")
        
        # Check for sequential numbering
        sorted_nums = sorted(step_nums)
        expected = list(range(1, len(sorted_nums) + 1))
        if sorted_nums != expected:
            issues.append(f"Step numbers not sequential: {sorted_nums}")
        
        # Check for steps without names
        for step in steps:
            if not step.name.strip():
                issues.append(f"Step {step.step_num} has no name")
        
        return issues


# Convenience function to get the standard protocol
def get_standard_set_steps() -> List[StepDefinition]:
    """Quick access to standard SET steps"""
    manager = FabricationStepsManager()
    return manager.get_standard_steps()


if __name__ == '__main__':
    # Example usage
    manager = FabricationStepsManager()
    
    # Get standard steps
    standard = manager.get_standard_steps()
    print(f"Standard SET protocol has {len(standard)} steps")
    
    # Print first 3 steps
    for step in standard[:3]:
        print(f"  Step {step.step_num}: {step.name}")
        print(f"    Purpose: {step.purpose}")
        print(f"    Requires scan: {step.requires_scan}")
    
    # Create custom protocol
    custom_steps = [
        {'step_num': 1, 'name': 'Custom Step 1', 'purpose': 'Test step'},
        {'step_num': 2, 'name': 'Custom Step 2', 'purpose': 'Another test'},
    ]
    custom_protocol = manager.create_custom_protocol("TEST_PROTOCOL", custom_steps)
    print(f"\nCreated custom protocol with {len(custom_protocol)} steps")
    
    # Get steps by category
    patterning_steps = manager.get_steps_by_category('patterning')
    print(f"\nPatterning steps: {len(patterning_steps)}")
    for step in patterning_steps:
        print(f"  - {step.name}")
