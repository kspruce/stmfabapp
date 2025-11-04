#!/usr/bin/env python3
"""
Convert auto-heater TDK data files to Flash.txt format.

This script converts LabVIEW auto-heater files (8 columns) to the standard 
Flash.txt format (12 columns) by adding missing columns with appropriate values.
"""

import sys
import argparse
from pathlib import Path


def parse_autoheater_file(input_path):
    """Parse the auto-heater file and extract header and data sections."""
    with open(input_path, 'r') as f:
        lines = f.readlines()
    
    # Find the two headers
    first_header_end = None
    second_header_end = None
    
    for i, line in enumerate(lines):
        if '***End_of_Header***' in line:
            if first_header_end is None:
                first_header_end = i
            else:
                second_header_end = i
                break
    
    if first_header_end is None or second_header_end is None:
        raise ValueError("Could not find header markers in file")
    
    # Split into sections
    first_header = lines[:first_header_end + 1]
    middle_section = lines[first_header_end + 1:second_header_end + 1]
    data_section = lines[second_header_end + 1:]
    
    return first_header, middle_section, data_section


def convert_to_flash_format(first_header, middle_section, data_section):
    """Convert 8-column format to 12-column Flash.txt format."""
    output_lines = []
    
    # First header stays mostly the same
    output_lines.extend(first_header)
    output_lines.append('\t\n')  # blank line after first header
    
    # Process middle section (channels, samples, date, time, etc.)
    for line in middle_section:
        if line.startswith('Channels'):
            # Change from 12 to 12 (already correct, but let's be explicit)
            output_lines.append('Channels\t12\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\n')
        elif line.startswith('Samples'):
            # Extend samples line to 12 columns
            parts = line.rstrip('\n').split('\t')
            # Keep first element 'Samples', then add twelve '1' values
            new_line = parts[0] + '\t' + '\t'.join(['1'] * 12) + '\t\n'
            output_lines.append(new_line)
        elif line.startswith('Date') and not line.startswith('Date\t2'):
            # This is the date row in the channel info
            parts = line.rstrip('\n').split('\t')
            if len(parts) > 1:
                date_val = parts[1]
                new_line = parts[0] + '\t' + '\t'.join([date_val] * 12) + '\t\n'
                output_lines.append(new_line)
            else:
                output_lines.append(line)
        elif line.startswith('Time') and not line.startswith('Time\t'):
            # This is the time row in the channel info
            parts = line.rstrip('\n').split('\t')
            if len(parts) > 1:
                time_val = parts[1]
                new_line = parts[0] + '\t' + '\t'.join([time_val] * 12) + '\t\n'
                output_lines.append(new_line)
            else:
                output_lines.append(line)
        elif line.startswith('Y_Unit_Label'):
            # Extend Y_Unit_Label: A, mbar, mbar, V, A, A, C, (blank), A, V, A, A
            parts = line.rstrip('\n').split('\t')
            # Original: A, mbar, mbar, V, A, A, (blank), (blank)
            # Need to add: C, (blank), A, V, A, A
            new_line = parts[0] + '\tA\tmbar\tmbar\tV\tA\tA\tC\t\tA\tV\tA\tA\t\n'
            output_lines.append(new_line)
        elif line.startswith('X_Dimension'):
            # Extend to 12 'Time' values
            parts = line.rstrip('\n').split('\t')
            new_line = parts[0] + '\t' + '\t'.join(['Time'] * 12) + '\t\n'
            output_lines.append(new_line)
        elif line.startswith('X0'):
            # Extend to 12 zeros
            parts = line.rstrip('\n').split('\t')
            if len(parts) > 1:
                zero_val = parts[1]
                new_line = parts[0] + '\t' + '\t'.join([zero_val] * 12) + '\t\n'
                output_lines.append(new_line)
            else:
                output_lines.append(line)
        elif line.startswith('Delta_X'):
            # Extend to 12 '1.000000' values
            parts = line.rstrip('\n').split('\t')
            if len(parts) > 1:
                delta_val = parts[1]
                new_line = parts[0] + '\t' + '\t'.join([delta_val] * 12) + '\t\n'
                output_lines.append(new_line)
            else:
                output_lines.append(line)
        else:
            output_lines.append(line)
    
    # Process data section
    for i, line in enumerate(data_section):
        line = line.rstrip('\n')
        if not line.strip():
            output_lines.append('\n')
            continue
            
        parts = line.split('\t')
        
        if i == 0:
            # Header row: X_Value, TDK P, P MBE, P VT, TDK V, TDK I, TDK R, Pyro T, Untitled, TDK P 1, TDK V 1, TDK I 1, TDK R 1, Comment
            # Original has: X_Value, TDK P, P MBE, P VT, TDK V, TDK I, TDK R, Untitled, Untitled 1, Comment
            # We need to add: Pyro T, Untitled, TDK P 1, TDK V 1, TDK I 1, TDK R 1
            new_line = 'X_Value\tTDK P\tP MBE\tP VT\tTDK V\tTDK I\tTDK R\tPyro T\tUntitled\tTDK P 1\tTDK V 1\tTDK I 1\tTDK R 1\tComment\n'
            output_lines.append(new_line)
        else:
            # Data rows: pad with additional columns
            # Auto-heater format has extra tab-delimited empty columns
            # Parse: empty, TDK P, empty, P MBE, empty, P VT, empty, TDK V, empty, TDK I, empty, TDK R, empty, Untitled, empty, Untitled 1, empty, Comment
            # Target: X_Value, TDK P, P MBE, P VT, TDK V, TDK I, TDK R, Pyro T, Untitled, TDK P 1, TDK V 1, TDK I 1, TDK R 1, Comment
            
            # The auto-heater has alternating empty columns, so actual data is at odd indices
            # Extract values skipping empty columns
            if len(parts) >= 19:
                x_value = parts[0].strip()  # Index 0 (empty initially)
                tdk_p = parts[1].strip()    # Index 1
                p_mbe = parts[3].strip()    # Index 3
                p_vt = parts[5].strip()     # Index 5
                tdk_v = parts[7].strip()    # Index 7
                tdk_i = parts[9].strip()    # Index 9
                tdk_r = parts[11].strip()   # Index 11
                untitled_col13 = parts[13].strip()  # Index 13 - first "Untitled"
                untitled_col15 = parts[15].strip()  # Index 15 - second "Untitled 1"
                comment = parts[17].strip() if len(parts) > 17 else ''  # Index 17
            else:
                # Fallback if format is different
                while len(parts) < 19:
                    parts.append('')
                x_value = parts[0].strip()
                tdk_p = parts[1].strip()
                p_mbe = parts[3].strip() if len(parts) > 3 else ''
                p_vt = parts[5].strip() if len(parts) > 5 else ''
                tdk_v = parts[7].strip() if len(parts) > 7 else ''
                tdk_i = parts[9].strip() if len(parts) > 9 else ''
                tdk_r = parts[11].strip() if len(parts) > 11 else ''
                untitled_col13 = parts[13].strip() if len(parts) > 13 else ''
                untitled_col15 = parts[15].strip() if len(parts) > 15 else ''
                comment = parts[17].strip() if len(parts) > 17 else ''
            
            # Build new row with additional columns
            # Pyro T column: use the first Untitled column value (TDK R value from autoheater, col 13)
            pyro_t = untitled_col13
            
            # New Untitled column: use the second Untitled column value (col 15)
            untitled_new = untitled_col15
            
            # TDK P 1, TDK V 1, TDK I 1, TDK R 1: set to defaults (0.0 for P/V/I, Inf for R)
            tdk_p_1 = '0.000000'
            tdk_v_1 = '0.013000'  # Match Flash.txt default
            tdk_i_1 = '0.000000'
            tdk_r_1 = 'Inf'
            
            new_line = '\t'.join([
                x_value, tdk_p, p_mbe, p_vt, tdk_v, tdk_i, tdk_r,
                pyro_t, untitled_new, tdk_p_1, tdk_v_1, tdk_i_1, tdk_r_1
            ]) + '\t' + comment + '\n'
            
            output_lines.append(new_line)
    
    return output_lines


def convert_file(input_path, output_path=None):
    """Convert a single auto-heater file to Flash format."""
    input_path = Path(input_path)
    
    if output_path is None:
        # Create output filename by adding '_flash_format' before extension
        output_path = input_path.parent / f"{input_path.stem}_flash_format{input_path.suffix}"
    else:
        output_path = Path(output_path)
    
    print(f"Converting: {input_path}")
    print(f"Output to: {output_path}")
    
    try:
        first_header, middle_section, data_section = parse_autoheater_file(input_path)
        output_lines = convert_to_flash_format(first_header, middle_section, data_section)
        
        with open(output_path, 'w') as f:
            f.writelines(output_lines)
        
        print(f"✓ Successfully converted!")
        return True
        
    except Exception as e:
        print(f"✗ Error converting file: {e}")
        return False


def batch_convert(input_dir, output_dir=None, pattern="Auto-heater*.txt"):
    """Convert all matching files in a directory."""
    input_dir = Path(input_dir)
    
    if output_dir is None:
        output_dir = input_dir
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    files = list(input_dir.glob(pattern))
    
    if not files:
        print(f"No files matching '{pattern}' found in {input_dir}")
        return 0
    
    print(f"Found {len(files)} file(s) to convert")
    print("-" * 60)
    
    success_count = 0
    for input_file in files:
        output_file = output_dir / f"{input_file.stem}_flash_format{input_file.suffix}"
        if convert_file(input_file, output_file):
            success_count += 1
        print("-" * 60)
    
    print(f"\nConversion complete: {success_count}/{len(files)} files successfully converted")
    return success_count


def main():
    parser = argparse.ArgumentParser(
        description='Convert auto-heater TDK data files to Flash.txt format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert a single file
  python convert_autoheater_to_flash_format.py input.txt
  
  # Convert with custom output name
  python convert_autoheater_to_flash_format.py input.txt -o output.txt
  
  # Batch convert all auto-heater files in a directory
  python convert_autoheater_to_flash_format.py -d /path/to/data/
  
  # Batch convert with custom output directory
  python convert_autoheater_to_flash_format.py -d /path/to/data/ --output-dir /path/to/output/
        """
    )
    
    parser.add_argument('input', nargs='?', help='Input file path')
    parser.add_argument('-o', '--output', help='Output file path')
    parser.add_argument('-d', '--directory', help='Process all matching files in directory')
    parser.add_argument('--output-dir', help='Output directory for batch processing')
    parser.add_argument('-p', '--pattern', default='Auto-heater*.txt',
                       help='File pattern for batch processing (default: Auto-heater*.txt)')
    
    args = parser.parse_args()
    
    if args.directory:
        batch_convert(args.directory, args.output_dir, args.pattern)
    elif args.input:
        convert_file(args.input, args.output)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
