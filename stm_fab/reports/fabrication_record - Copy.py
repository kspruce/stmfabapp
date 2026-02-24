"""
STM Hydrogen Desorption Lithography - Fabrication Record Generator

This script automates the documentation of STM (Scanning Tunneling Microscope) 
fabrication processes for quantum dot devices. It generates comprehensive HTML 
reports with embedded images and metadata from STM scan files.

Key Features:
- Parses .sxm files (Nanonis STM format) using pySPM library
- Extracts detailed scan metadata (dimensions, speed, bias voltage, etc.)
- Generates structured fabrication records following a step-by-step protocol
- Creates self-contained HTML reports with Base64-encoded images
- Supports both standard fabrication workflows and custom step definitions
"""

import pySPM  # Library for parsing Nanonis STM files
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import os
from pathlib import Path
from datetime import datetime
import base64
from io import BytesIO
import argparse

from importlib.resources import files

# --------------------------------------------------------------------------------------
# Fabrication step definitions (Standard SET)
# --------------------------------------------------------------------------------------
# Define the standard fabrication protocol for quantum dot devices
# Each step represents a specific stage in the fabrication process
FABRICATION_STEPS = [
    {
        'step_num': 1,
        'name': 'Nominal Design Reference',
        'purpose': 'Document GDS design for fabrication reference',
        'requires_scan': True  # Indicates this step needs an STM scan
    },
    {
        'step_num': 2,
        'name': '12μm Step Edge Mapping',
        'purpose': 'Map step edge structure for alignment and future device location',
        'requires_scan': True
    },
    {
        'step_num': 3,
        'name': '5μm Alignment Layer Area',
        'purpose': 'Verify terrace cleanliness and identify suitable fabrication area',
        'requires_scan': True
    },
    {
        'step_num': 4,
        'name': '2.5μm Alignment Layer Tapers',
        'purpose': 'Create tapers connecting μm scale to nm scale structures',
        'requires_scan': True
    },
    {
        'step_num': 5,
        'name': '700nm Inner Region Patterning',
        'purpose': 'Pattern source, drain, and gates with gaps for quantum dot',
        'requires_scan': True
    },
    {
        'step_num': 6,
        'name': '50nm Quantum Dot Patterning',
        'purpose': 'Pattern quantum dots with correct dimer row gaps',
        'requires_scan': True
    },
    {
        'step_num': 7,
        'name': '50nm Pre-Dose Tunnel Region (Dual Bias)',
        'purpose': 'Document tunnel region before XH3 dosing',
        'requires_scan': True
    },
    {
        'step_num': 8,
        'name': '500nm Pre-Dose Full Device (45° Dual Bias)',
        'purpose': 'Full inner device documentation before dosing',
        'requires_scan': True
    },
    {
        'step_num': 9,
        'name': 'Contact Leg Verification',
        'purpose': 'Ensure complete connection and full desorption for each contact leg',
        'requires_scan': True,
        'multiple': True,  # Indicates this step can have multiple scans
        'note': 'Repeat for each leg: 4-6 total'
    },
    {
        'step_num': 10,
        'name': 'XH3 DOSING STEP (NO STM)',
        'purpose': 'Saturation dose, 10 mins, 5×10⁻⁹ Torr',
        'requires_scan': False,  # This is a process step without imaging
        'special_fields': ['elog_entry', 'time_started', 'time_completed', 'dose_confirmed']
    },
    {
        'step_num': 11,
        'name': '500nm Post-Dose Full Device (45° Dual Bias)',
        'purpose': 'Document full inner device after XH3 dosing',
        'requires_scan': True
    },
    {
        'step_num': 12,
        'name': '50nm Post-Dose Tunnel Region (Dual Bias)',
        'purpose': 'High-resolution documentation of tunnel region after dosing',
        'requires_scan': True
    },
    {
        'step_num': 13,
        'name': 'Dopant Incorporation Imaging (Optional)',
        'purpose': 'Image device after dopant incorporation process',
        'requires_scan': True,
        'optional': True  # Marks this step as optional in the workflow
    },
    {
        'step_num': 14,
        'name': 'Overgrowth Imaging (200nm, 100nm, 30nm)',
        'purpose': 'Document device at multiple scales during/after overgrowth',
        'requires_scan': True,
        'multiple': True,
        'note': 'Multiple scans at different scales'
    }
]

# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------
def load_gwyddion_colormap(colormap_resource='Gwyddionnet.pymap'):
    """
    Load a Gwyddion colormap from a .pymap file for STM image visualization.
    
    Gwyddion is a popular SPM (Scanning Probe Microscopy) data visualization tool.
    This function loads custom colormaps compatible with Gwyddion's format.
    
    Args:
        colormap_file: Path to .pymap file containing RGB colormap data
        
    Returns:
        matplotlib colormap object (either custom Gwyddion or default viridis)
    """
    # Check if the colormap file exists
    try:
        colormap_path = files('stm_fab.resources').joinpath(colormap_resource)
        if not colormap_path.is_file():
            print(f"Warning: Resource '{colormap_resource}' not found. Using default colormap.")
            return plt.cm.viridis
        raw_rgb = np.genfromtxt(str(colormap_path), skip_header=1)
        return ListedColormap(raw_rgb)
    except Exception as e:
        print(f"Warning: Error loading colormap: {e}. Using default colormap.")
        return plt.cm.viridis


def _unwrap_nested(value):
    """
    Unwrap nested list structures from pySPM/Nanonis header values.
    
    The pySPM library often returns metadata in nested list formats that need
    to be flattened to extract the actual values.
    
    Examples:
        [['256', '256']] -> ['256', '256']
        [['-2E+0']] -> '-2E+0'
        '100E-12' -> '100E-12'
    
    Args:
        value: Potentially nested list/tuple structure
        
    Returns:
        Unwrapped value (scalar or simple list)
    """
    v = value
    # Unwrap nested lists/tuples until we reach non-nested data
    while isinstance(v, (list, tuple)) and len(v) > 0 and isinstance(v[0], (list, tuple)):
        v = v[0]
    # If it's a single-element list/tuple, extract the element
    if isinstance(v, (list, tuple)) and len(v) == 1:
        v = v[0]
    return v


def _to_float(val, default=None):
    """
    Convert various string formats to float values.
    
    Handles scientific notation, values with units (V, A, nm, etc.), and nested structures
    from STM metadata. This is essential for parsing Nanonis header values.
    
    Args:
        val: Value to convert (string, number, or nested structure)
        default: Value to return if conversion fails
        
    Returns:
        float value or default if conversion fails
    """
    if val is None:
        return default
    
    # First unwrap any nested structure
    v = _unwrap_nested(val)
    
    # If already a number, convert to float
    if isinstance(v, (int, float)):
        return float(v)
    
    # Handle string values with units
    if isinstance(v, str):
        s = v.strip()
        # Remove common unit suffixes (A=Ampere, V=Volt, s=second, m=meter, etc.)
        for suffix in ['A', 'V', 's', 'm/s', 'm', 'nm', 'um', 'μm', 'µm']:
            if s.endswith(suffix):
                s = s[: -len(suffix)].strip()
        # Normalize micro symbol variants
        s = s.replace('µ', 'u').replace('μ', 'u')
        try:
            return float(s)
        except:
            return default
    
    # Handle list/tuple by recursively trying first element
    if isinstance(v, (list, tuple)) and len(v) >= 1:
        return _to_float(v[0], default=default)
    
    return default


# --------------------------------------------------------------------------------------
# Metadata extraction
# --------------------------------------------------------------------------------------
def parse_sxm_metadata(sxm_file):
    """
    Extract normalized metadata from an SXM file using pySPM library.
    
    SXM is the native file format for Nanonis STM systems. This function parses
    the header and extracts key imaging parameters.
    
    Metadata extracted includes:
    - Image dimensions (pixels and physical size in nm)
    - Scan speed
    - Bias voltage
    - Acquisition time
    - Z-range (height variation)
    
    Args:
        sxm_file: Path to .sxm file
        
    Returns:
        Dictionary containing normalized metadata fields
    """
    try:
        # Load the SXM file using pySPM
        sxm = pySPM.SXM(sxm_file)
        header = sxm.header

        md = {}  # Metadata dictionary to populate

        # -------------------------------------------------------------------------
        # Extract pixel dimensions
        # -------------------------------------------------------------------------
        # Try Nanonis-specific keys first, then fall back to generic keys
        px_val = header.get('Scan>pixels/line') or header.get('SCAN_PIXELS') or 0
        py_val = header.get('Scan>lines') or 0

        px = None  # pixels in x direction
        py = None  # pixels in y direction
        
        # SCAN_PIXELS can be nested like [['256','256']]
        if isinstance(px_val, (list, tuple)):
            pv = _unwrap_nested(px_val)
            if isinstance(pv, (list, tuple)) and len(pv) >= 2:
                px = _to_float(pv[0], default=None)
                py = _to_float(pv[1], default=None)
        
        # If not found in nested structure, try direct conversion
        if px is None:
            px = _to_float(px_val, default=None)
        if py is None:
            py = _to_float(py_val, default=None)

        # Convert to integers (pixel counts should be whole numbers)
        if isinstance(px, float):
            px = int(round(px))
        if isinstance(py, float):
            py = int(round(py))

        md['pixels_x'] = px if px is not None else 'N/A'
        md['pixels_y'] = py if py is not None else 'N/A'
        # Pixel ratio can indicate non-square scans
        md['pixel_ratio'] = (px / py) if (isinstance(px, int) and isinstance(py, int) and py != 0) else 'N/A'

        # -------------------------------------------------------------------------
        # Extract physical scan size (in nanometers)
        # -------------------------------------------------------------------------
        scan_x_nm = None
        scan_y_nm = None
        
        # Scan>Scanfield contains dimensions like "1.5e-07;1.5e-07" (in meters)
        scan_range = header.get('Scan>Scanfield', '')
        if isinstance(scan_range, str) and ';' in scan_range:
            parts = [p.strip() for p in scan_range.split(';')]
            if len(parts) >= 2:
                x_m = _to_float(parts[0], default=None)  # meters
                y_m = _to_float(parts[1], default=None)
                if x_m is not None:
                    scan_x_nm = x_m * 1e9  # Convert meters to nanometers
                if y_m is not None:
                    scan_y_nm = y_m * 1e9

        # Fallback to SCAN_RANGE field if Scanfield not found
        if scan_x_nm is None or scan_y_nm is None:
            scan_range_alt = header.get('SCAN_RANGE')
            if scan_range_alt is not None:
                sr = _unwrap_nested(scan_range_alt)
                if isinstance(sr, (list, tuple)) and len(sr) >= 2:
                    x_m_alt = _to_float(sr[0], default=None)
                    y_m_alt = _to_float(sr[1], default=None)
                    if scan_x_nm is None and x_m_alt is not None:
                        scan_x_nm = x_m_alt * 1e9
                    if scan_y_nm is None and y_m_alt is not None:
                        scan_y_nm = y_m_alt * 1e9

        md['scan_width_nm'] = f"{scan_x_nm:.2f}" if scan_x_nm else 'N/A'
        md['scan_height_nm'] = f"{scan_y_nm:.2f}" if scan_y_nm else 'N/A'

        # -------------------------------------------------------------------------
        # Calculate scan speed (nm/s)
        # -------------------------------------------------------------------------
        # Scan speed is calculated as: width / (total_time / number_of_lines)
        # This gives the tip velocity across each scan line
        scan_time_s = _to_float(header.get('ACQ_TIME'), default=None)
        if scan_x_nm and scan_time_s and isinstance(py, int) and py > 0:
            # Speed = distance per line / time per line
            md['scan_speed_nm_s'] = f"{(scan_x_nm / (scan_time_s / py)):.2f}"
        else:
            md['scan_speed_nm_s'] = 'N/A'

        # -------------------------------------------------------------------------
        # Extract bias voltage
        # -------------------------------------------------------------------------
        bias_v = _to_float(header.get('Bias>Bias (V)') or header.get('BIAS'), default=None)
        md['bias_V'] = f"{bias_v:.3f}" if bias_v is not None else 'N/A'

        # -------------------------------------------------------------------------
        # Extract acquisition time
        # -------------------------------------------------------------------------
        if scan_time_s:
            minutes = int(scan_time_s // 60)
            seconds = int(scan_time_s % 60)
            md['acq_time'] = f"{minutes}m {seconds}s"
        else:
            md['acq_time'] = 'N/A'

        # -------------------------------------------------------------------------
        # Extract Z-range (height variation in the scan)
        # -------------------------------------------------------------------------
        z_range_m = _to_float(header.get('Scan>Z-Controller>Z range') or header.get('Z-CONTROLLER>Z range'), default=None)
        if z_range_m is not None:
            md['z_range_nm'] = f"{z_range_m * 1e9:.2f}"
        else:
            md['z_range_nm'] = 'N/A'

        # -------------------------------------------------------------------------
        # Calculate pixel pitch (physical size per pixel)
        # -------------------------------------------------------------------------
        if scan_x_nm and isinstance(px, int) and px > 0:
            md['pixel_pitch_nm'] = f"{scan_x_nm / px:.3f}"
        else:
            md['pixel_pitch_nm'] = 'N/A'

        return md

    except Exception as e:
        # If parsing fails, return error message
        print(f"  Warning: Could not parse metadata for {sxm_file}: {e}")
        return {'error': str(e)}


# --------------------------------------------------------------------------------------
# Image generation
# --------------------------------------------------------------------------------------
def generate_image_base64(sxm_file, colormap=None, dpi=150):
    """
    Generate a Base64-encoded PNG image from an SXM file.
    
    This creates a publication-quality visualization of the STM topography data
    that can be embedded directly in HTML reports.
    
    Args:
        sxm_file: Path to .sxm file
        colormap: matplotlib colormap to use (default: viridis)
        dpi: Resolution for the output image
        
    Returns:
        Base64-encoded string of the PNG image, or None if generation fails
    """
    try:
        # Load the STM data
        sxm = pySPM.SXM(sxm_file)
        # Get the topography channel (Z data)
        channel = sxm.get_channel('Z')
        # Convert to numpy array
        data = channel.pixels

        # Set default colormap if none provided
        if colormap is None:
            colormap = plt.cm.viridis

        # Create figure with no padding/borders for clean embedding
        fig, ax = plt.subplots(figsize=(6, 6), dpi=dpi)
        
        # Display the STM image
        im = ax.imshow(data, cmap=colormap, origin='lower', interpolation='nearest')
        
        # Add colorbar to show height scale
        plt.colorbar(im, ax=ax, label='Height (arb. units)')
        
        # Remove axis ticks for cleaner appearance
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Set title with filename
        ax.set_title(Path(sxm_file).name, fontsize=10)

        # Save figure to memory buffer instead of disk
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)  # Free memory
        
        # Encode image as Base64 for HTML embedding
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        
        return img_base64

    except Exception as e:
        print(f"  Warning: Could not generate image for {sxm_file}: {e}")
        return None


# --------------------------------------------------------------------------------------
# Main generator class
# --------------------------------------------------------------------------------------
class FabricationRecordGenerator:
    """
    Main class for generating fabrication records from STM data.
    
    This class orchestrates the entire workflow:
    1. Scans a folder for .sxm files
    2. Interactively associates files with fabrication steps
    3. Extracts metadata and generates images
    4. Produces a comprehensive HTML report
    
    The interactive session guides users through the fabrication protocol,
    allowing them to assign scans to specific steps or skip optional steps.
    """
    
    def __init__(self, scan_folder, device_name, output_dir, step_defs=None):
        """
        Initialize the fabrication record generator.
        
        Args:
            scan_folder: Path to folder containing .sxm files
            device_name: Identifier for the device being documented
            output_dir: Where to save the final HTML report
            step_defs: List of step definitions (defaults to FABRICATION_STEPS)
        """
        self.scan_folder = Path(scan_folder)
        self.device_name = device_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Use provided steps or default to standard fabrication protocol
        self.step_defs = step_defs if step_defs else FABRICATION_STEPS
        
        # Storage for completed step data
        self.steps_data = []
        
        # Load custom colormap for STM visualization
        self.colormap = load_gwyddion_colormap()
        
        # Scan for available .sxm files
        self.available_sxm_files = self._scan_for_sxm_files()

    def _scan_for_sxm_files(self):
        """
        Recursively find all .sxm files in the scan folder.
        
        Returns:
            Sorted list of Path objects for .sxm files
        """
        sxm_files = list(self.scan_folder.rglob('*.sxm'))
        # Sort alphabetically for consistent presentation
        sxm_files.sort()
        print(f"Found {len(sxm_files)} .sxm files in {self.scan_folder}")
        return sxm_files

    def _list_available_files(self):
        """
        Display available .sxm files with index numbers for user selection.
        """
        print("\nAvailable .sxm files:")
        for idx, f in enumerate(self.available_sxm_files, start=1):
            # Show relative path for easier identification
            rel_path = f.relative_to(self.scan_folder)
            print(f"  [{idx}] {rel_path}")

    def _select_file_interactive(self, prompt="Select file index: "):
        """
        Prompt user to select a file by its index number.
        
        Args:
            prompt: Text to display when asking for file selection
            
        Returns:
            Selected Path object or None if user cancels
        """
        while True:
            choice = input(prompt).strip()
            
            # Allow user to skip or cancel
            if choice.lower() in ['', 'skip', 's']:
                return None
            
            try:
                idx = int(choice)
                # Validate index is in valid range
                if 1 <= idx <= len(self.available_sxm_files):
                    return self.available_sxm_files[idx - 1]
                else:
                    print(f"  Invalid index. Choose 1-{len(self.available_sxm_files)}")
            except ValueError:
                print("  Invalid input. Enter a number or 'skip'")

    def run_interactive_session(self):
        """
        Interactive session to document each fabrication step.
        
        This guides the user through the fabrication protocol, prompting them to:
        - Associate .sxm files with steps that require scans
        - Enter process parameters for non-scan steps (e.g., dosing conditions)
        - Add notes or skip optional steps
        
        The session maintains state in self.steps_data for later report generation.
        """
        print(f"\n{'='*70}")
        print(f"Fabrication Record for Device: {self.device_name}")
        print(f"{'='*70}")
        print("Instructions:")
        print("  - For STM steps: select file index or type 'skip' to skip step")
        print("  - For Process steps: enter requested parameters")
        print("  - Optional steps can be skipped")
        print(f"{'='*70}\n")

        # Iterate through each defined fabrication step
        for step_def in self.step_defs:
            step_num = step_def['step_num']
            step_name = step_def['name']
            purpose = step_def['purpose']
            requires_scan = step_def.get('requires_scan', True)
            is_optional = step_def.get('optional', False)
            is_multiple = step_def.get('multiple', False)

            print(f"\n{'─'*70}")
            print(f"Step {step_num}: {step_name}")
            print(f"Purpose: {purpose}")
            
            # Show additional context if available
            if 'note' in step_def:
                print(f"Note: {step_def['note']}")
            if is_optional:
                print("(OPTIONAL)")
            print(f"{'─'*70}")

            # Handle steps that require STM scans
            if requires_scan:
                if is_multiple:
                    # Allow multiple scans for this step (e.g., contact leg verification)
                    scans_for_step = []
                    print(f"This step can contain multiple scans.")
                    self._list_available_files()
                    
                    while True:
                        sel = self._select_file_interactive(
                            f"Select file for Step {step_num} (or Enter to finish): "
                        )
                        if sel is None:
                            break
                        scans_for_step.append(sel)
                        print(f"  Added: {sel.name}")
                    
                    # Process all selected scans
                    if scans_for_step:
                        step_record = {
                            'step_num': step_num,
                            'step_name': step_name,
                            'purpose': purpose,
                            'scans': []
                        }
                        for scan_file in scans_for_step:
                            scan_data = self._process_scan(scan_file)
                            step_record['scans'].append(scan_data)
                        self.steps_data.append(step_record)
                        print(f"  Recorded {len(scans_for_step)} scan(s) for Step {step_num}")
                    elif not is_optional:
                        # Non-optional steps must have data
                        print(f"  Step {step_num} SKIPPED (no scans added)")
                    else:
                        print(f"  Optional step {step_num} skipped")
                else:
                    # Single scan per step (most common case)
                    self._list_available_files()
                    sel = self._select_file_interactive(f"Select file for Step {step_num}: ")
                    
                    if sel is not None:
                        # Process the scan and store results
                        scan_data = self._process_scan(sel)
                        step_record = {
                            'step_num': step_num,
                            'step_name': step_name,
                            'purpose': purpose,
                            'scans': [scan_data]
                        }
                        self.steps_data.append(step_record)
                        print(f"  Step {step_num} recorded")
                    elif not is_optional:
                        print(f"  Step {step_num} SKIPPED")
                    else:
                        print(f"  Optional step {step_num} skipped")

            else:
                # Handle process steps (no scan required, e.g., chemical dosing)
                print(f"This is a process step (no STM scan).")
                
                process_params = {}
                
                # Check if specific fields are predefined for this step
                if 'special_fields' in step_def:
                    for field in step_def['special_fields']:
                        val = input(f"  {field}: ").strip()
                        process_params[field] = val if val else 'N/A'
                else:
                    # Free-form parameter entry
                    print("  Enter any process parameters (or press Enter to skip):")
                    while True:
                        param_name = input("    Parameter name (or Enter to finish): ").strip()
                        if not param_name:
                            break
                        param_value = input(f"    {param_name} = ").strip()
                        process_params[param_name] = param_value if param_value else 'N/A'

                # Allow user to add general notes
                notes = input("  Additional notes (optional): ").strip()
                
                # Store process step data
                step_record = {
                    'step_num': step_num,
                    'step_name': step_name,
                    'purpose': purpose,
                    'process_params': process_params,
                    'notes': notes if notes else None
                }
                self.steps_data.append(step_record)
                print(f"  Step {step_num} process parameters recorded")

        print(f"\n{'='*70}")
        print("Interactive session complete. All steps documented.")
        print(f"{'='*70}\n")

    def _process_scan(self, sxm_file):
        """
        Extract metadata and generate image for a single scan file.
        
        Args:
            sxm_file: Path to .sxm file
            
        Returns:
            Dictionary containing filename, metadata, and Base64 image
        """
        print(f"  Processing: {sxm_file.name}...", end=' ')
        
        # Extract all metadata from the scan
        metadata = parse_sxm_metadata(sxm_file)
        
        # Generate embedded image
        img_b64 = generate_image_base64(sxm_file, colormap=self.colormap)
        
        print("✓")
        
        return {
            'filename': sxm_file.name,
            'metadata': metadata,
            'image_base64': img_b64
        }

    def generate_html_report(self):
        """
        Generate a self-contained HTML report with all data and images.
        
        The report includes:
        - Device information and timestamp
        - Each fabrication step with associated scans
        - Embedded STM images (Base64-encoded)
        - Detailed metadata tables
        - Professional styling with CSS
        
        Returns:
            HTML string ready to write to file
        """
        # Start building HTML document
        html_parts = []
        
        # HTML header with embedded CSS
        html_parts.append("""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Fabrication Record - """ + self.device_name + """</title>
    <style>
        /* Global styling */
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
            color: #333;
        }
        
        /* Header styling */
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        .header h1 {
            margin: 0 0 10px 0;
            font-size: 2.5em;
        }
        
        .header .device-info {
            font-size: 1.2em;
            opacity: 0.95;
        }
        
        /* Step container styling */
        .step {
            background: white;
            margin-bottom: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        
        /* Step header with number and title */
        .step-header {
            background: #667eea;
            color: white;
            padding: 20px;
            font-size: 1.5em;
            font-weight: bold;
        }
        
        .step-purpose {
            background: #f8f9fa;
            padding: 15px 20px;
            border-left: 4px solid #667eea;
            font-style: italic;
            color: #555;
        }
        
        /* Scan container within a step */
        .scan {
            padding: 20px;
            border-top: 1px solid #e0e0e0;
        }
        
        .scan:first-child {
            border-top: none;
        }
        
        .scan h4 {
            color: #667eea;
            margin-top: 0;
        }
        
        /* Image styling */
        .scan-image {
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 4px;
            margin: 15px 0;
            display: block;
        }
        
        /* Metadata table styling */
        .metadata-table {
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
            background: white;
        }
        
        .metadata-table th {
            background: #f8f9fa;
            padding: 12px;
            text-align: left;
            font-weight: 600;
            color: #667eea;
            border-bottom: 2px solid #667eea;
        }
        
        .metadata-table td {
            padding: 10px 12px;
            border-bottom: 1px solid #e0e0e0;
        }
        
        .metadata-table tr:last-child td {
            border-bottom: none;
        }
        
        .metadata-table tr:hover {
            background: #f8f9fa;
        }
        
        /* Process step styling */
        .process-step {
            background: #fff3cd;
            padding: 20px;
            border-left: 4px solid #ffc107;
        }
        
        .process-params {
            background: white;
            padding: 15px;
            border-radius: 4px;
            margin-top: 10px;
        }
        
        /* Footer styling */
        .footer {
            text-align: center;
            color: #666;
            margin-top: 40px;
            padding: 20px;
            border-top: 1px solid #ddd;
        }
    </style>
</head>
<body>
""")

        # Report header with device name and generation timestamp
        html_parts.append(f"""
    <div class="header">
        <h1>STM Fabrication Record</h1>
        <div class="device-info">
            <strong>Device:</strong> {self.device_name}<br>
            <strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
    </div>
""")

        # Generate content for each documented step
        for step_record in self.steps_data:
            step_num = step_record['step_num']
            step_name = step_record['step_name']
            purpose = step_record['purpose']

            # Step header
            html_parts.append(f"""
    <div class="step">
        <div class="step-header">
            Step {step_num}: {step_name}
        </div>
        <div class="step-purpose">
            <strong>Purpose:</strong> {purpose}
        </div>
""")

            # Handle STM scan steps
            if 'scans' in step_record:
                for scan_data in step_record['scans']:
                    filename = scan_data['filename']
                    metadata = scan_data['metadata']
                    img_b64 = scan_data['image_base64']

                    html_parts.append(f"""
        <div class="scan">
            <h4>📊 Scan: {filename}</h4>
""")

                    # Embed STM image if available
                    if img_b64:
                        html_parts.append(f"""
            <img src="data:image/png;base64,{img_b64}" alt="{filename}" class="scan-image">
""")

                    # Generate metadata table
                    if metadata and 'error' not in metadata:
                        html_parts.append("""
            <table class="metadata-table">
                <thead>
                    <tr>
                        <th>Parameter</th>
                        <th>Value</th>
                    </tr>
                </thead>
                <tbody>
""")
                        # Display each metadata field
                        for key, value in metadata.items():
                            # Format key for display (convert snake_case to Title Case)
                            display_key = key.replace('_', ' ').title()
                            html_parts.append(f"""
                    <tr>
                        <td>{display_key}</td>
                        <td>{value}</td>
                    </tr>
""")
                        html_parts.append("""
                </tbody>
            </table>
""")

                    html_parts.append("""
        </div>
""")

            # Handle process steps (no scans)
            elif 'process_params' in step_record:
                process_params = step_record['process_params']
                notes = step_record.get('notes')

                html_parts.append("""
        <div class="process-step">
            <h4>⚙️ Process Parameters</h4>
""")

                # Display process parameters
                if process_params:
                    html_parts.append("""
            <div class="process-params">
""")
                    for param_name, param_value in process_params.items():
                        html_parts.append(f"""
                <strong>{param_name}:</strong> {param_value}<br>
""")
                    html_parts.append("""
            </div>
""")

                # Display additional notes if provided
                if notes:
                    html_parts.append(f"""
            <p><strong>Notes:</strong> {notes}</p>
""")

                html_parts.append("""
        </div>
""")

            html_parts.append("""
    </div>
""")

        # Report footer
        html_parts.append("""
    <div class="footer">
        <p>Generated by STM Fabrication Record Generator</p>
        <p><em>This report contains embedded images and can be viewed offline</em></p>
    </div>
</body>
</html>
""")

        # Combine all parts into final HTML string
        return ''.join(html_parts)

    def save_report(self, output_format='html'):
        """
        Save the fabrication record to a file.
        
        Args:
            output_format: Format for output (currently only 'html' supported)
            
        Returns:
            Path to the saved report file
        """
        # Currently only HTML output is implemented
        if output_format != 'html':
            print(f"Note: Only HTML output is implemented in this script. Falling back to HTML.")
            output_format = 'html'

        # Generate the HTML content
        report_content = self.generate_html_report()

        # Create timestamped filename to avoid overwrites
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{self.device_name}_fabrication_record_{timestamp}.html"
        filepath = self.output_dir / filename

        # Write to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report_content)

        print(f"\n✓ Fabrication record saved: {filepath}")
        print(f"  → Open in browser to view with embedded images")

        return filepath


# --------------------------------------------------------------------------------------
# Build a fully custom step set (interactive)
# --------------------------------------------------------------------------------------
def build_custom_steps_interactive():
    """
    Interactively build a custom fabrication step list.
    
    This allows users to define their own protocols instead of using the standard
    SET protocol. Useful for non-standard devices or experimental processes.
    
    For each step, the user defines:
    - Type: STM (requires scan) or Process (no scan)
    - Title and Purpose
    - For Process steps, optional custom parameter fields
    
    Returns:
        List of step definition dictionaries compatible with FabricationRecordGenerator
    """
    steps = []
    index = 1
    
    print("\nCreate your custom step set. You can add as many steps as you like.")
    
    while True:
        print("\n" + "-"*60)
        print(f"Define Step {index}")
        
        # Ask for step type
        step_type = input("  Step Type [STM/Process] (default: STM): ").strip().lower() or 'stm'
        is_scan = (step_type == 'stm')

        # Get step title (required)
        title = input("  Step Title: ").strip()
        while not title:
            title = input("  Step Title (required): ").strip()

        # Get step purpose/description
        purpose = input("  Step Purpose/Description: ").strip() or 'N/A'

        # Build step definition dictionary
        step_def = {
            'step_num': index,
            'name': title,
            'purpose': purpose,
            'requires_scan': is_scan
        }

        if is_scan:
            # Optional: allow multiple scans for this STM step
            multi = input("  Will this step contain multiple scans? (y/n, default n): ").strip().lower()
            if multi == 'y':
                step_def['multiple'] = True
        else:
            # Process step: optionally predefine parameter fields
            predef = input("  Predefine parameter fields to prompt? (y/n): ").strip().lower()
            if predef == 'y':
                fields = []
                while True:
                    f = input("    Field name (e.g., 'elog_entry'), or Enter to stop: ").strip()
                    if not f:
                        break
                    fields.append(f)
                if fields:
                    step_def['special_fields'] = fields

        steps.append(step_def)

        # Ask if user wants to add more steps
        another = input("\nAdd another step? (y/n): ").strip().lower()
        if another != 'y':
            break
        index += 1

    print(f"\nCustom step set created with {len(steps)} steps.")
    return steps


# --------------------------------------------------------------------------------------
# Interactive configuration for device, folders, and step set
# --------------------------------------------------------------------------------------
def interactive_setup(project_root):
    """
    Interactively configure the fabrication record generation.
    
    Prompts user for:
    - Run mode: Standard SET protocol or fully custom steps
    - Device name (identifier for the device)
    - Scan folder location (date-based or full path)
    - Output folder (where to save HTML report)
    
    Args:
        project_root: Root directory of the project (used for default paths)
        
    Returns:
        Tuple of (scan_folder, device_name, output_dir, step_defs)
    """
    # Define default paths relative to project root
    default_data_root = project_root / 'stm_data'
    default_output_dir = project_root / 'output'

    print("\n=== Interactive Setup ===")
    
    # -------------------------------------------------------------------------
    # Choose fabrication protocol
    # -------------------------------------------------------------------------
    print("Step Set Options:")
    print("  1) Standard SET (predefined steps)")
    print("  2) Fully Custom device (define your own steps)")
    choice = input("Select [1/2] (default 1): ").strip() or '1'
    
    if choice == '2':
        # Build custom protocol interactively
        step_defs = build_custom_steps_interactive()
    else:
        # Use standard predefined protocol
        step_defs = FABRICATION_STEPS

    # -------------------------------------------------------------------------
    # Get device name
    # -------------------------------------------------------------------------
    device_name = input("\nDevice Name (identifier): ").strip()
    while not device_name:
        device_name = input("Device Name is required: ").strip()

    # -------------------------------------------------------------------------
    # Select scan folder
    # -------------------------------------------------------------------------
    print("\nScan Folder Options:")
    print(f"  1) Date under default data root ({default_data_root})")
    print("  2) Provide full path to scans")
    sc_choice = input("Select [1/2] (default 1): ").strip() or '1'

    if sc_choice == '2':
        # User provides full path
        sf = input("Enter full path to folder containing .sxm files: ").strip()
        scan_folder = Path(sf)
    else:
        # Date-based organization under default data root
        date_str = input("Enter date folder (e.g., 2025-10-03): ").strip()
        while not date_str:
            date_str = input("Date is required (e.g., 2025-10-03): ").strip()
        scan_folder = default_data_root / date_str

    # Validate scan folder exists
    while not scan_folder.exists():
        print(f"  ! Path not found: {scan_folder}")
        sc_choice = input("Try again? y/n (default y): ").strip().lower() or 'y'
        if sc_choice != 'y':
            raise SystemExit("Aborted by user.")
        # Retry selection
        return interactive_setup(project_root)

    # -------------------------------------------------------------------------
    # Select output folder
    # -------------------------------------------------------------------------
    print(f"\nDefault output folder: {default_output_dir}")
    out_custom = input("Use a different output folder? (y/n, default n): ").strip().lower() or 'n'
    
    if out_custom == 'y':
        outp = input("Enter full path for output folder: ").strip()
        output_dir = Path(outp)
    else:
        output_dir = default_output_dir
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    return scan_folder, device_name, output_dir, step_defs


# --------------------------------------------------------------------------------------
# Command Line Interface (CLI)
# --------------------------------------------------------------------------------------
def main():
    """
    Main entry point for the script.
    
    Supports both command-line arguments and interactive mode:
    - CLI mode: All parameters provided as arguments
    - Interactive mode: User is prompted for missing parameters
    
    The script will generate an HTML fabrication record based on the provided
    or interactively selected configuration.
    """
    # Determine project root from script location
    # Expected structure: <project_root>/scripts/this_file.py
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    # Set up command-line argument parser
    parser = argparse.ArgumentParser(
        description='STM Hydrogen Desorption Lithography - Fabrication Record Generator'
    )
    
    # All arguments are optional; missing ones trigger interactive mode
    parser.add_argument('--scan-folder', '-s', 
                       help='Full path to folder containing .sxm scan files')
    parser.add_argument('--date', '-D', 
                       help='Date subfolder under <project_root>/stm_data, e.g., 2025-10-03')
    parser.add_argument('--device-name', '-d', 
                       help='Device name/identifier')
    parser.add_argument('--output-dir', '-o', 
                       help='Output directory (default: <project_root>/output)')
    parser.add_argument('--standard', action='store_true', 
                       help='Use standard steps without prompting for a custom set')
    parser.add_argument('--format', '-f', choices=['html'], default='html', 
                       help='Output format (only html supported)')
    
    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # Determine if interactive mode is needed
    # -------------------------------------------------------------------------
    # Interactive mode is triggered if any key configuration is missing
    need_interactive = any([
        args.device_name is None,
        (args.scan_folder is None and args.date is None),
        (args.output_dir is None),
        (not args.standard)  # if not explicitly requesting standard, allow interactive menu
    ])

    if need_interactive:
        # Run interactive setup to get missing configuration
        scan_folder, device_name, output_dir, step_defs = interactive_setup(project_root)
    else:
        # Use command-line arguments
        # Resolve scan folder from either full path or date
        if args.scan_folder:
            scan_folder = Path(args.scan_folder)
        else:
            scan_folder = (project_root / 'stm_data' / args.date)
        
        # Validate scan folder exists
        if not scan_folder.exists():
            print(f"Error: Scan folder not found: {scan_folder}")
            return

        device_name = args.device_name
        output_dir = Path(args.output_dir) if args.output_dir else (project_root / 'output')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Use standard protocol
        step_defs = FABRICATION_STEPS

    # -------------------------------------------------------------------------
    # Create generator and run
    # -------------------------------------------------------------------------
    generator = FabricationRecordGenerator(
        scan_folder=scan_folder,
        device_name=device_name,
        output_dir=output_dir,
        step_defs=step_defs
    )

    # Run interactive session to document each step
    generator.run_interactive_session()

    # Generate and save final HTML report
    print("\nGenerating fabrication record...")
    generator.save_report(output_format='html')

    print("\n✓ Fabrication record generation complete!")


# --------------------------------------------------------------------------------------
# Database Integration Function
# --------------------------------------------------------------------------------------
def create_fabrication_record(session, device_id: int, output_path: str = None):
    """
    Create a fabrication record from database for a specific device.
    
    This function reads all fabrication steps and scans from the database
    and generates an HTML report without requiring interactive input.
    
    Args:
        session: SQLAlchemy session
        device_id: ID of the device to generate report for
        output_path: Optional specific path for output file
        
    Returns:
        Path to the generated HTML report
    """
    from stm_fab.db.models import Device, FabricationStep, STMScan
    
    # Get device from database
    device = session.query(Device).filter_by(device_id=device_id).first()
    if not device:
        raise ValueError(f"Device with ID {device_id} not found")
    
    # Get all fabrication steps for this device
    steps = session.query(FabricationStep).filter_by(device_id=device_id).order_by(FabricationStep.step_number).all()
    
    if not steps:
        raise ValueError(f"No fabrication steps found for device {device.device_name}")
    
    # Get device info
    device_name = device.device_name
    sample_name = device.sample.sample_name if device.sample else "Unknown"
    
    # Determine output directory
    if output_path:
        output_file = Path(output_path)
        output_dir = output_file.parent
    else:
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = output_dir / f"{device_name}_fabrication_record_{timestamp}.html"
    
    # Load colormap
    colormap = load_gwyddion_colormap()
    
    # Build steps data from database
    steps_data = []
    
    for step in steps:
        step_dict = {
            'step_num': step.step_number,
            'name': step.step_name,
            'purpose': step.purpose or '',
            'status': step.status,
            'timestamp': step.timestamp,
            'operator': step.operator or '',
            'notes': step.notes or '',
            'scans': []
        }
        
        # Get all scans for this step
        scans = session.query(STMScan).filter_by(step_id=step.step_id).all()
        
        for scan in scans:
            try:
                # Check if we have image data stored
                if scan.image_data:
                    # Use stored base64 image
                    image_base64 = scan.image_data
                elif scan.filepath and Path(scan.filepath).exists():
                    # Generate image from file
                    image_base64 = generate_image_base64(scan.filepath, colormap=colormap)
                else:
                    image_base64 = None
                
                scan_dict = {
                    'filename': scan.filename,
                    'filepath': scan.filepath,
                    'metadata': scan.metadata_json or {},
                    'image_base64': image_base64,
                    'scan_date': scan.scan_date
                }
                step_dict['scans'].append(scan_dict)
                
            except Exception as e:
                print(f"Warning: Could not process scan {scan.filename}: {e}")
                continue
        
        steps_data.append(step_dict)
    
    # Generate HTML report
    html_content = _generate_html_from_steps_data(
        device_name=device_name,
        sample_name=sample_name,
        steps_data=steps_data,
        device=device
    )
    
    # Write to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✓ Fabrication record saved: {output_file}")
    
    return str(output_file)


def _generate_html_from_steps_data(device_name: str, sample_name: str, steps_data: list, device=None):
    """
    Generate HTML report from structured steps data (used by database integration).
    
    Args:
        device_name: Name of the device
        sample_name: Name of the sample
        steps_data: List of step dictionaries with scan data
        device: Optional Device object for additional metadata
        
    Returns:
        HTML string
    """
    # Build HTML header
    html_parts = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        "    <meta charset='UTF-8'>",
        f"    <title>Fabrication Record - {device_name}</title>",
        "    <style>",
        "        body { font-family: 'Segoe UI', Arial, sans-serif; margin: 40px; background: #f5f5f5; }",
        "        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 40px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }",
        "        h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }",
        "        h2 { color: #34495e; margin-top: 30px; }",
        "        .metadata { background: #ecf0f1; padding: 15px; border-radius: 5px; margin: 20px 0; }",
        "        .metadata p { margin: 5px 0; }",
        "        .step { margin: 30px 0; padding: 20px; border: 1px solid #bdc3c7; border-radius: 5px; background: #fafafa; }",
        "        .step-header { font-size: 1.3em; font-weight: bold; color: #2980b9; margin-bottom: 10px; }",
        "        .step-purpose { font-style: italic; color: #7f8c8d; margin: 10px 0; }",
        "        .step-status { display: inline-block; padding: 4px 12px; border-radius: 3px; font-size: 0.9em; font-weight: bold; margin: 10px 0; }",
        "        .status-complete { background: #2ecc71; color: white; }",
        "        .status-pending { background: #f39c12; color: white; }",
        "        .status-in_progress { background: #3498db; color: white; }",
        "        .status-failed { background: #e74c3c; color: white; }",
        "        .status-skipped { background: #95a5a6; color: white; }",
        "        .scan-image { max-width: 800px; margin: 15px 0; border: 1px solid #ddd; border-radius: 3px; }",
        "        .scan-metadata { background: #fff; padding: 10px; margin: 10px 0; border-left: 3px solid #3498db; font-size: 0.9em; }",
        "        .scan-metadata table { width: 100%; border-collapse: collapse; }",
        "        .scan-metadata td { padding: 4px 8px; border-bottom: 1px solid #ecf0f1; }",
        "        .scan-metadata td:first-child { font-weight: bold; color: #34495e; width: 200px; }",
        "        .notes { background: #fff3cd; border-left: 4px solid #ffc107; padding: 10px; margin: 10px 0; }",
        "        .timestamp { color: #7f8c8d; font-size: 0.9em; }",
        "        .no-scans { color: #95a5a6; font-style: italic; }",
        "    </style>",
        "</head>",
        "<body>",
        "    <div class='container'>",
        f"        <h1>Fabrication Record: {device_name}</h1>",
        "        <div class='metadata'>",
        f"            <p><strong>Sample:</strong> {sample_name}</p>",
        f"            <p><strong>Device:</strong> {device_name}</p>",
    ]
    
    # Add device metadata if available
    if device:
        if device.fabrication_start:
            html_parts.append(f"            <p><strong>Start Date:</strong> {device.fabrication_start.strftime('%Y-%m-%d %H:%M')}</p>")
        if device.operator:
            html_parts.append(f"            <p><strong>Operator:</strong> {device.operator}</p>")
        if device.overall_status:
            html_parts.append(f"            <p><strong>Status:</strong> {device.overall_status}</p>")
        if device.completion_percentage is not None:
            html_parts.append(f"            <p><strong>Completion:</strong> {device.completion_percentage:.0f}%</p>")
    
    html_parts.extend([
        f"            <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>",
        "        </div>",
        "        <h2>Fabrication Steps</h2>",
    ])
    
    # Add each step
    for step_data in steps_data:
        status_class = f"status-{step_data.get('status', 'pending')}"
        
        html_parts.extend([
            "        <div class='step'>",
            f"            <div class='step-header'>Step {step_data['step_num']}: {step_data['name']}</div>",
            f"            <div class='step-status {status_class}'>{step_data.get('status', 'pending').upper()}</div>",
        ])
        
        if step_data.get('purpose'):
            html_parts.append(f"            <div class='step-purpose'>{step_data['purpose']}</div>")
        
        if step_data.get('timestamp'):
            timestamp_str = step_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
            html_parts.append(f"            <div class='timestamp'>Completed: {timestamp_str}</div>")
        
        if step_data.get('notes'):
            html_parts.append(f"            <div class='notes'><strong>Notes:</strong> {step_data['notes']}</div>")
        
        # Add scans
        scans = step_data.get('scans', [])
        if scans:
            html_parts.append(f"            <p><strong>Scans ({len(scans)}):</strong></p>")
            
            for scan in scans:
                html_parts.append(f"            <h3>{scan['filename']}</h3>")
                
                if scan.get('image_base64'):
                    html_parts.append(f"            <img src='data:image/png;base64,{scan['image_base64']}' class='scan-image' />")
                
                # Add metadata table
                metadata = scan.get('metadata', {})
                if metadata:
                    html_parts.extend([
                        "            <div class='scan-metadata'>",
                        "                <table>",
                    ])
                    
                    for key, value in metadata.items():
                        html_parts.append(f"                    <tr><td>{key}</td><td>{value}</td></tr>")
                    
                    html_parts.extend([
                        "                </table>",
                        "            </div>",
                    ])
        else:
            html_parts.append("            <p class='no-scans'>No scans recorded for this step</p>")
        
        html_parts.append("        </div>")
    
    # Close HTML
    html_parts.extend([
        "    </div>",
        "</body>",
        "</html>"
    ])
    
    return '\n'.join(html_parts)


# --------------------------------------------------------------------------------------
# Script entry point
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()