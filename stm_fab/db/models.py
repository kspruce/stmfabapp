"""
database_models.py - SQLAlchemy ORM models for STM fabrication system

Based on the Technical Implementation Guide provided in the roadmap

ENHANCED VERSION:
- Added cascade='all, delete-orphan' to all Sample relationships
- This ensures proper deletion order when deleting samples
- Prevents IntegrityError: NOT NULL constraint violations

If upgrading from the original version, you'll need to recreate your database
or the enhanced database_operations.py will handle deletions explicitly.
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, ForeignKey, JSON, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()


class Sample(Base):
    """
    Top-level sample entity - represents a physical substrate
    """
    __tablename__ = 'samples'

    sample_id = Column(Integer, primary_key=True)
    sample_name = Column(String(100), unique=True, nullable=False, index=True)
    creation_date = Column(DateTime, default=datetime.utcnow)
    substrate_type = Column(String(50))  # e.g., "Si(100)", "GaAs(100)"
    supplier = Column(String(100))
    doping_level = Column(Float)  # cm^-3
    resistivity = Column(Float)  # Ohm-cm
    status = Column(String(20), default='active')  # active, complete, failed
    notes = Column(Text)
    
    # NEW: paths linked to the sample
    labview_folder_path = Column(String(500))  # Folder containing LabVIEW .txt files
    scan_folder_path = Column(String(500))     # Folder containing .sxm scans

    # Relationships
    devices = relationship('Device', back_populates='sample', cascade='all, delete-orphan')
    process_steps = relationship('ProcessStep', back_populates='sample', cascade='all, delete-orphan')
    thermal_budget = relationship('ThermalBudget', back_populates='sample', uselist=False, cascade='all, delete-orphan')
    cooldown_calibrations = relationship('CooldownCalibration', back_populates='sample', cascade='all, delete-orphan')

    def __repr__(self):
        return f"<Sample(name='{self.sample_name}', status='{self.status}')>"


class Device(Base):
    """
    Individual device on a sample - represents a fabrication attempt
    """
    __tablename__ = 'devices'

    device_id = Column(Integer, primary_key=True)
    sample_id = Column(Integer, ForeignKey('samples.sample_id'), nullable=False)
    device_name = Column(String(100), unique=True, nullable=False, index=True)
    nominal_design_ref = Column(String(200))  # Link to GDS file or design doc
    fabrication_start = Column(DateTime)
    fabrication_end = Column(DateTime)
    completion_percentage = Column(Float, default=0.0)
    overall_status = Column(String(20), default='in_progress')  # in_progress, complete, failed
    operator = Column(String(100))
    notes = Column(Text)

    # Relationships
    sample = relationship('Sample', back_populates='devices')
    fabrication_steps = relationship('FabricationStep', back_populates='device', 
                                    cascade='all, delete-orphan', order_by='FabricationStep.step_number')

    def __repr__(self):
        return f"<Device(name='{self.device_name}', status='{self.overall_status}')>"

    def calculate_completion(self):
        """Calculate completion percentage based on completed steps"""
        if not self.fabrication_steps:
            return 0.0
        completed = sum(1 for step in self.fabrication_steps 
                       if step.status == 'complete')
        return (completed / len(self.fabrication_steps)) * 100


class FabricationStep(Base):
    """
    Individual fabrication step within a device workflow
    """
    __tablename__ = 'fabrication_steps'

    step_id = Column(Integer, primary_key=True)
    device_id = Column(Integer, ForeignKey('devices.device_id'), nullable=False)
    step_number = Column(Integer, nullable=False)
    step_name = Column(String(200), nullable=False)
    purpose = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)
    operator = Column(String(100))
    status = Column(String(20), default='pending')  # pending, complete, skipped, failed
    notes = Column(Text)
    quality_check_passed = Column(Boolean)
    requires_scan = Column(Boolean, default=True)

    # Relationships
    device = relationship('Device', back_populates='fabrication_steps')
    stm_scans = relationship('STMScan', back_populates='fabrication_step', 
                            cascade='all, delete-orphan')
    quality_checks = relationship('QualityCheck', back_populates='fabrication_step')

    def __repr__(self):
        return f"<FabricationStep(device='{self.device_id}', step={self.step_number}, status='{self.status}')>"


class STMScan(Base):
    """
    STM scan file metadata and parameters
    """
    __tablename__ = 'stm_scans'

    scan_id = Column(Integer, primary_key=True)
    step_id = Column(Integer, ForeignKey('fabrication_steps.step_id'), nullable=False)
    filename = Column(String(200), nullable=False)
    filepath = Column(String(500), nullable=False)
    scan_date = Column(DateTime)

    # Scan parameters
    bias_voltage = Column(Float)  # V
    setpoint_current = Column(Float)  # A
    scan_size_x = Column(Float)  # m
    scan_size_y = Column(Float)  # m
    pixels_x = Column(Integer)
    pixels_y = Column(Integer)
    scan_speed = Column(Float)  # nm/s
    acquisition_time = Column(Float)  # s

    # Complete metadata as JSON
    metadata_json = Column(JSON)

    # Image data (base64 encoded) - optional, can store separately
    image_data = Column(Text)  # Base64 encoded PNG

    # Relationships
    fabrication_step = relationship('FabricationStep', back_populates='stm_scans')

    def __repr__(self):
        return f"<STMScan(filename='{self.filename}', date='{self.scan_date}')>"


class ProcessStep(Base):
    """
    Process steps (degas, flash, dose, etc.) with LabVIEW file links
    """
    __tablename__ = 'process_steps'

    process_id = Column(Integer, primary_key=True)
    sample_id = Column(Integer, ForeignKey('samples.sample_id'), nullable=False)
    process_type = Column(String(50), nullable=False)  # degas, flash, hterm, dose, inc, overgrowth
    labview_file_path = Column(String(500))
    start_time = Column(DateTime)
    end_time = Column(DateTime)

    # Process metrics
    max_temperature = Column(Float)  # °C
    thermal_budget_contribution = Column(Float)  # °C·s
    peak_pressure = Column(Float)  # Torr
    base_pressure = Column(Float)  # Torr

    # Complete process data as JSON
    parameters_json = Column(JSON)
    summary = Column(Text)

    # Quality flags
    pressure_stable = Column(Boolean)
    temperature_stable = Column(Boolean)
    completed_successfully = Column(Boolean)

    # Relationships
    sample = relationship('Sample', back_populates='process_steps')

    def __repr__(self):
        return f"<ProcessStep(sample='{self.sample_id}', type='{self.process_type}')>"


class ThermalBudget(Base):
    """
    Cumulative thermal budget tracking for a sample
    """
    __tablename__ = 'thermal_budgets'

    budget_id = Column(Integer, primary_key=True)
    sample_id = Column(Integer, ForeignKey('samples.sample_id'), nullable=False, unique=True)

    # Individual contributions (°C·s)
    degas_contribution = Column(Float, default=0.0)
    flash_contribution = Column(Float, default=0.0)
    hterm_contribution = Column(Float, default=0.0)
    incorporation_contribution = Column(Float, default=0.0)
    overgrowth_contribution = Column(Float, default=0.0)
    other_contribution = Column(Float, default=0.0)

    # Total
    total_budget = Column(Float, default=0.0)

    # Thresholds and warnings
    warning_threshold = Column(Float, default=3.0e6)  # °C·s
    critical_threshold = Column(Float, default=4.0e6)  # °C·s

    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    sample = relationship('Sample', back_populates='thermal_budget')

    def calculate_total(self):
        """Recalculate total thermal budget"""
        self.total_budget = (
            self.degas_contribution +
            self.flash_contribution +
            self.hterm_contribution +
            self.incorporation_contribution +
            self.overgrowth_contribution +
            self.other_contribution
        )
        return self.total_budget

    def status(self):
        """Get warning status"""
        if self.total_budget >= self.critical_threshold:
            return 'critical'
        elif self.total_budget >= self.warning_threshold:
            return 'warning'
        else:
            return 'normal'


class CooldownCalibration(Base):
    """
    Temperature vs. current calibration from cooldown curves
    """
    __tablename__ = 'cooldown_calibrations'

    calibration_id = Column(Integer, primary_key=True)
    sample_id = Column(Integer, ForeignKey('samples.sample_id'), nullable=False)
    calibration_date = Column(DateTime, default=datetime.utcnow)
    labview_file_path = Column(String(500))

    # Curve data stored as JSON arrays
    # Format: {'current': [0.03, 0.05, ...], 'temperature': [1580, 1450, ...]}
    curve_data_json = Column(JSON)

    # Polynomial fit coefficients [a0, a1, a2, ...]
    # T(I) = a0 + a1*I + a2*I^2 + ...
    fit_coefficients = Column(JSON)

    # Fit quality metrics
    r_squared = Column(Float)
    rmse = Column(Float)

    # Common temperature lookups (cached)
    current_at_330C = Column(Float)  # For H-termination
    current_at_350C = Column(Float)  # For incorporation

    notes = Column(Text)

    # Relationships
    sample = relationship('Sample', back_populates='cooldown_calibrations')

    def get_current_for_temperature(self, target_temp):
        """
        Calculate required current for target temperature using polynomial fit

        Args:
            target_temp: Temperature in Celsius

        Returns:
            current: Required current in Amps
        """
        import numpy as np
        if not self.fit_coefficients:
            raise ValueError("No calibration data available")

        # Use polynomial to get current from temperature
        # Need to solve for I given T: T = a0 + a1*I + a2*I^2 + ...
        # This is an inverse problem - can use root finding or interpolation

        # Simpler approach: use interpolation on original data
        curve_data = self.curve_data_json
        temp_array = np.array(curve_data['temperature'])
        current_array = np.array(curve_data['current'])

        # Interpolate
        current = np.interp(target_temp, temp_array[::-1], current_array[::-1])
        return float(current)


class QualityCheck(Base):
    """
    Quality check records for fabrication steps
    """
    __tablename__ = 'quality_checks'

    check_id = Column(Integer, primary_key=True)
    step_id = Column(Integer, ForeignKey('fabrication_steps.step_id'))
    check_name = Column(String(200), nullable=False)
    check_type = Column(String(50))  # automated, manual
    category = Column(String(50))  # critical, important, informational

    passed = Column(Boolean)
    checked_by = Column(String(100))
    check_timestamp = Column(DateTime, default=datetime.utcnow)

    # Check results
    expected_value = Column(String(200))
    actual_value = Column(String(200))
    notes = Column(Text)

    # Relationships
    fabrication_step = relationship('FabricationStep', back_populates='quality_checks')

    def __repr__(self):
        status = "✓" if self.passed else "✗"
        return f"<QualityCheck({status} {self.check_name})>"


class ProcessMetrics(Base):
    """
    Detailed process metrics computed from LabVIEW files
    
    Stores analysis results like:
    - Dose: flux, exposure, integrated molecules
    - Flash: count, durations, peak temperatures
    - Overgrowth: RT/RTA/LTE phase breakdown
    - SUSI: operating time, currents
    """
    __tablename__ = 'process_metrics'
    
    metrics_id = Column(Integer, primary_key=True)
    process_step_id = Column(Integer, ForeignKey('process_steps.process_id'), nullable=False)
    
    # File info
    file_path = Column(String(500))
    file_type = Column(String(50))  # dose, flash, susi, etc.
    
    # Complete metrics as JSON
    metrics_json = Column(JSON)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<ProcessMetrics(process_step={self.process_step_id}, type='{self.file_type}')>"


# Database initialization
def init_database(db_url='sqlite:///stm_fab_records.db'):
    """Initialize database and create all tables"""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    engine = create_engine(db_url, echo=False)
    Base.metadata.create_all(engine)

    Session = sessionmaker(bind=engine)
    return Session()


# Example usage
if __name__ == '__main__':
    # Create database
    session = init_database()

    # Create a sample
    sample = Sample(
        sample_name='Si_2025_Q4_001',
        substrate_type='Si(100)',
        supplier='UniversityWafer',
        doping_level=1e15,
        resistivity=10.0,
        notes='High-quality substrate for quantum dot fabrication'
    )
    session.add(sample)
    session.commit()

    print(f"Created: {sample}")