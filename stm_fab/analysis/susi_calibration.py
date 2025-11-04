# stm_fab/analysis/susi_calibration.py
"""
SUSI Growth Rate Calibration Management

Stores and retrieves calibrated growth rates for locking layer and LTE phases.
Calibrations are stored with metadata (date, sample, method) for traceability.
"""

import json
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import pandas as pd


@dataclass
class SUSICalibration:
    """Data class for a SUSI calibration entry"""
    calibration_id: Optional[int]
    date: str  # ISO format: YYYY-MM-DD
    sample_name: str
    method: str  # 'SIMS', 'MASK', 'STM', 'XRR', etc.
    locking_layer_rate: float  # ML/min or nm/min
    lte_rate: float  # ML/min or nm/min
    rate_units: str  # 'ML/min' or 'nm/min'
    notes: str = ""
    created_timestamp: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    

    def get_rate_ML_min(self, phase: str) -> float:
        """
        Get rate in ML/min for a given phase. Uses 1 ML = 0.136 nm.
        """
        if phase == 'locking_layer':
            rate = self.locking_layer_rate
        elif phase == 'lte':
            rate = self.lte_rate
        else:
            raise ValueError(f"Unknown phase: {phase}")

        if self.rate_units == 'nm/min':
            # 10 ML = 1.36 nm ⇒ 1 ML = 0.136 nm ⇒ ML/min = (nm/min) / 0.136
            rate = rate / 0.136

        return rate



class SUSICalibrationManager:
    """
    Manages SUSI growth rate calibrations
    
    Stores calibrations in SQLite database with full history and metadata.
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize calibration manager
        
        Args:
            db_path: Path to SQLite database file
                    Default: ~/.stm_fab/susi_calibrations.db
        """
        if db_path is None:
            # Default to user's home directory
            home = Path.home()
            db_dir = home / '.stm_fab'
            db_dir.mkdir(parents=True, exist_ok=True)
            db_path = str(db_dir / 'susi_calibrations.db')
        
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS calibrations (
                calibration_id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                sample_name TEXT NOT NULL,
                method TEXT NOT NULL,
                locking_layer_rate REAL NOT NULL,
                lte_rate REAL NOT NULL,
                rate_units TEXT NOT NULL,
                notes TEXT,
                created_timestamp TEXT NOT NULL,
                is_active INTEGER DEFAULT 1
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_calibration(self, 
                       date: str,
                       sample_name: str,
                       method: str,
                       locking_layer_rate: float,
                       lte_rate: float,
                       rate_units: str = 'ML/min',
                       notes: str = "",
                       set_as_active: bool = True) -> int:
        """
        Add a new calibration to the database
        
        Args:
            date: Calibration date (YYYY-MM-DD)
            sample_name: Sample identifier
            method: Measurement method ('SIMS', 'MASK', 'STM', 'XRR', etc.)
            locking_layer_rate: Growth rate for locking layer
            lte_rate: Growth rate for LTE
            rate_units: 'ML/min' or 'nm/min'
            notes: Optional notes
            set_as_active: If True, set this as the active calibration
            
        Returns:
            calibration_id of the new entry
        """
        # Validate inputs
        if rate_units not in ['ML/min', 'nm/min']:
            raise ValueError(f"rate_units must be 'ML/min' or 'nm/min', got '{rate_units}'")
        
        # Validate method
        valid_methods = ['SIMS', 'MASK', 'STM', 'XRR', 'TEM', 'RHEED', 'OTHER']
        if method.upper() not in valid_methods:
            print(f"Warning: method '{method}' not in standard list {valid_methods}")
        
        created_timestamp = datetime.now().isoformat()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # If setting as active, deactivate all others first
        if set_as_active:
            cursor.execute('UPDATE calibrations SET is_active = 0')
        
        # Insert new calibration
        cursor.execute('''
            INSERT INTO calibrations 
            (date, sample_name, method, locking_layer_rate, lte_rate, 
             rate_units, notes, created_timestamp, is_active)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (date, sample_name, method.upper(), locking_layer_rate, lte_rate,
              rate_units, notes, created_timestamp, 1 if set_as_active else 0))
        
        calibration_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        print(f"✓ Added calibration ID {calibration_id}")
        if set_as_active:
            print(f"  Set as active calibration")
        
        return calibration_id
    
    def get_active_calibration(self) -> Optional[SUSICalibration]:
        """
        Get the currently active calibration
        
        Returns:
            SUSICalibration object or None if no active calibration
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM calibrations 
            WHERE is_active = 1
            ORDER BY created_timestamp DESC
            LIMIT 1
        ''')
        
        row = cursor.fetchone()
        conn.close()
        
        if row is None:
            return None
        
        return self._row_to_calibration(row)
    
    def get_calibration_by_id(self, calibration_id: int) -> Optional[SUSICalibration]:
        """Get a specific calibration by ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM calibrations WHERE calibration_id = ?', 
                      (calibration_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row is None:
            return None
        
        return self._row_to_calibration(row)
    
    def get_latest_calibration(self) -> Optional[SUSICalibration]:
        """Get the most recent calibration (by date)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM calibrations 
            ORDER BY date DESC, created_timestamp DESC
            LIMIT 1
        ''')
        
        row = cursor.fetchone()
        conn.close()
        
        if row is None:
            return None
        
        return self._row_to_calibration(row)
    
    def get_calibration_history(self, limit: Optional[int] = None) -> List[SUSICalibration]:
        """
        Get calibration history
        
        Args:
            limit: Maximum number of entries (None for all)
            
        Returns:
            List of calibrations, newest first
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = 'SELECT * FROM calibrations ORDER BY date DESC, created_timestamp DESC'
        if limit:
            query += f' LIMIT {limit}'
        
        cursor.execute(query)
        rows = cursor.fetchall()
        conn.close()
        
        return [self._row_to_calibration(row) for row in rows]
    
    def set_active_calibration(self, calibration_id: int):
        """Set a specific calibration as active"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Deactivate all
        cursor.execute('UPDATE calibrations SET is_active = 0')
        
        # Activate specified
        cursor.execute('UPDATE calibrations SET is_active = 1 WHERE calibration_id = ?',
                      (calibration_id,))
        
        if cursor.rowcount == 0:
            conn.close()
            raise ValueError(f"Calibration ID {calibration_id} not found")
        
        conn.commit()
        conn.close()
        
        print(f"✓ Set calibration ID {calibration_id} as active")
    
    def delete_calibration(self, calibration_id: int):
        """Delete a calibration entry"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM calibrations WHERE calibration_id = ?',
                      (calibration_id,))
        
        if cursor.rowcount == 0:
            conn.close()
            raise ValueError(f"Calibration ID {calibration_id} not found")
        
        conn.commit()
        conn.close()
        
        print(f"✓ Deleted calibration ID {calibration_id}")
    
    def get_calibrations_by_method(self, method: str) -> List[SUSICalibration]:
        """Get all calibrations using a specific method"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM calibrations 
            WHERE method = ?
            ORDER BY date DESC
        ''', (method.upper(),))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [self._row_to_calibration(row) for row in rows]
    
    def export_to_json(self, output_path: str):
        """Export all calibrations to JSON file"""
        calibrations = self.get_calibration_history()
        
        data = {
            'export_date': datetime.now().isoformat(),
            'calibrations': [cal.to_dict() for cal in calibrations]
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✓ Exported {len(calibrations)} calibrations to {output_path}")
    
    def import_from_json(self, input_path: str):
        """Import calibrations from JSON file"""
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        count = 0
        for cal_dict in data['calibrations']:
            # Skip calibration_id to let DB auto-generate
            self.add_calibration(
                date=cal_dict['date'],
                sample_name=cal_dict['sample_name'],
                method=cal_dict['method'],
                locking_layer_rate=cal_dict['locking_layer_rate'],
                lte_rate=cal_dict['lte_rate'],
                rate_units=cal_dict['rate_units'],
                notes=cal_dict.get('notes', ''),
                set_as_active=False
            )
            count += 1
        
        print(f"✓ Imported {count} calibrations from {input_path}")
    
    def get_summary_dataframe(self) -> pd.DataFrame:
        """Get calibration history as pandas DataFrame"""
        calibrations = self.get_calibration_history()
        
        if not calibrations:
            return pd.DataFrame()
        
        data = []
        for cal in calibrations:
            data.append({
                'ID': cal.calibration_id,
                'Date': cal.date,
                'Sample': cal.sample_name,
                'Method': cal.method,
                'Locking Rate': f"{cal.locking_layer_rate:.3f}",
                'LTE Rate': f"{cal.lte_rate:.3f}",
                'Units': cal.rate_units,
                'Active': '✓' if cal.calibration_id == self.get_active_calibration().calibration_id else '',
                'Notes': cal.notes[:30] + '...' if len(cal.notes) > 30 else cal.notes
            })
        
        return pd.DataFrame(data)
    
    def print_summary(self, limit: int = 10):
        """Print a formatted summary of calibrations"""
        calibrations = self.get_calibration_history(limit=limit)
        active = self.get_active_calibration()
        
        print("=" * 100)
        print("SUSI CALIBRATION HISTORY")
        print("=" * 100)
        print("")
        
        if not calibrations:
            print("No calibrations found.")
            return
        
        if active:
            print(f"Active Calibration: ID {active.calibration_id} from {active.date}")
            print(f"  Locking Layer: {active.locking_layer_rate:.3f} {active.rate_units}")
            print(f"  LTE:           {active.lte_rate:.3f} {active.rate_units}")
            print("")
        
        print("Recent Calibrations:")
        print("-" * 100)
        print(f"{'ID':<5} {'Date':<12} {'Sample':<20} {'Method':<8} "
              f"{'Lock Rate':<12} {'LTE Rate':<12} {'Units':<10} {'Active':<7}")
        print("-" * 100)
        
        for cal in calibrations[:limit]:
            is_active = '✓' if active and cal.calibration_id == active.calibration_id else ''
            print(f"{cal.calibration_id:<5} {cal.date:<12} {cal.sample_name:<20} {cal.method:<8} "
                  f"{cal.locking_layer_rate:<12.3f} {cal.lte_rate:<12.3f} "
                  f"{cal.rate_units:<10} {is_active:<7}")
        
        if len(calibrations) > limit:
            print(f"\n... and {len(calibrations) - limit} more")
        
        print("=" * 100)
    
    def _row_to_calibration(self, row: Tuple) -> SUSICalibration:
        """Convert database row to SUSICalibration object"""
        return SUSICalibration(
            calibration_id=row[0],
            date=row[1],
            sample_name=row[2],
            method=row[3],
            locking_layer_rate=row[4],
            lte_rate=row[5],
            rate_units=row[6],
            notes=row[7] if row[7] else "",
            created_timestamp=row[8]
        )


# ==================== CONVENIENCE FUNCTIONS ====================

def add_susi_calibration(date: str,
                        sample_name: str,
                        method: str,
                        locking_layer_rate: float,
                        lte_rate: float,
                        rate_units: str = 'ML/min',
                        notes: str = "",
                        db_path: Optional[str] = None) -> int:
    """
    Convenience function to add a calibration
    
    Example:
        add_susi_calibration(
            date='2025-11-01',
            sample_name='Sample_A_001',
            method='SIMS',
            locking_layer_rate=0.5,
            lte_rate=1.2,
            notes='Post-maintenance calibration'
        )
    """
    manager = SUSICalibrationManager(db_path)
    return manager.add_calibration(
        date=date,
        sample_name=sample_name,
        method=method,
        locking_layer_rate=locking_layer_rate,
        lte_rate=lte_rate,
        rate_units=rate_units,
        notes=notes,
        set_as_active=True
    )


def get_current_rates(db_path: Optional[str] = None) -> Optional[Tuple[float, float]]:
    """
    Get current active calibration rates in ML/min
    
    Returns:
        Tuple of (locking_layer_rate, lte_rate) in ML/min, or None if no active calibration
    """
    manager = SUSICalibrationManager(db_path)
    cal = manager.get_active_calibration()
    
    if cal is None:
        return None
    
    return (cal.get_rate_ML_min('locking_layer'), 
            cal.get_rate_ML_min('lte'))


def show_calibrations(db_path: Optional[str] = None):
    """Show calibration history"""
    manager = SUSICalibrationManager(db_path)
    manager.print_summary()


# ==================== COMMAND LINE INTERFACE ====================

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("SUSI Calibration Manager")
        print("")
        print("Usage:")
        print("  python susi_calibration.py add <date> <sample> <method> <lock_rate> <lte_rate> [units] [notes]")
        print("  python susi_calibration.py show")
        print("  python susi_calibration.py active")
        print("  python susi_calibration.py set-active <id>")
        print("  python susi_calibration.py export <filename.json>")
        print("  python susi_calibration.py import <filename.json>")
        print("")
        print("Examples:")
        print("  python susi_calibration.py add 2025-11-01 Sample_A SIMS 0.5 1.2")
        print("  python susi_calibration.py add 2025-11-01 Sample_B MASK 0.48 1.15 ML/min 'Post maintenance'")
        print("  python susi_calibration.py show")
        sys.exit(1)
    
    command = sys.argv[1]
    manager = SUSICalibrationManager()
    
    if command == 'add':
        if len(sys.argv) < 7:
            print("Error: add requires: date sample method lock_rate lte_rate [units] [notes]")
            sys.exit(1)
        
        date = sys.argv[2]
        sample = sys.argv[3]
        method = sys.argv[4]
        lock_rate = float(sys.argv[5])
        lte_rate = float(sys.argv[6])
        units = sys.argv[7] if len(sys.argv) > 7 else 'ML/min'
        notes = sys.argv[8] if len(sys.argv) > 8 else ''
        
        manager.add_calibration(date, sample, method, lock_rate, lte_rate, units, notes)
    
    elif command == 'show':
        manager.print_summary(limit=20)
    
    elif command == 'active':
        cal = manager.get_active_calibration()
        if cal:
            print(f"Active Calibration: ID {cal.calibration_id}")
            print(f"  Date:          {cal.date}")
            print(f"  Sample:        {cal.sample_name}")
            print(f"  Method:        {cal.method}")
            print(f"  Locking Layer: {cal.locking_layer_rate:.3f} {cal.rate_units}")
            print(f"  LTE:           {cal.lte_rate:.3f} {cal.rate_units}")
            if cal.notes:
                print(f"  Notes:         {cal.notes}")
        else:
            print("No active calibration set")
    
    elif command == 'set-active':
        if len(sys.argv) < 3:
            print("Error: set-active requires calibration ID")
            sys.exit(1)
        manager.set_active_calibration(int(sys.argv[2]))
    
    elif command == 'export':
        if len(sys.argv) < 3:
            print("Error: export requires filename")
            sys.exit(1)
        manager.export_to_json(sys.argv[2])
    
    elif command == 'import':
        if len(sys.argv) < 3:
            print("Error: import requires filename")
            sys.exit(1)
        manager.import_from_json(sys.argv[2])
    
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
