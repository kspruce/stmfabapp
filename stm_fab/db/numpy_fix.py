"""
Database Operations NumPy Fix

This module provides a wrapper/patch for your database operations
to ensure all NumPy arrays are converted to JSON-serializable types
BEFORE being passed to SQLAlchemy.

Usage:
------
1. Import this at the top of your main GUI file:
   from db_ops_numpy_fix import ensure_json_serializable

2. Wrap any data before passing to add_process_step:
   data = ensure_json_serializable(parsed_data)
   self.db_ops.add_process_step(..., parsed_data=data)

OR

3. Monkey-patch your DatabaseOperations class (see bottom of file)
"""

import numpy as np
from typing import Any, Dict, List, Union
import json


def ensure_json_serializable(obj: Any) -> Any:
    """
    Recursively convert NumPy arrays and other non-serializable objects to Python types.
    
    This function handles:
    - numpy.ndarray → list
    - numpy integer types → int
    - numpy floating types → float
    - numpy bool → bool
    - nested dicts and lists
    - other objects → attempt str() conversion
    
    Args:
        obj: Any Python object that needs to be JSON serializable
        
    Returns:
        JSON-serializable version of the object
    """
    if obj is None:
        return None
    
    # Handle NumPy types
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, np.complexfloating):
        return {'real': float(obj.real), 'imag': float(obj.imag)}
    
    # Handle Python built-in containers
    elif isinstance(obj, dict):
        return {key: ensure_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [ensure_json_serializable(item) for item in obj]
    elif isinstance(obj, set):
        return [ensure_json_serializable(item) for item in obj]
    
    # Handle basic Python types (already serializable)
    elif isinstance(obj, (str, int, float, bool)):
        return obj
    
    # For anything else, try to convert to string
    else:
        try:
            # Try to see if it's already JSON serializable
            json.dumps(obj)
            return obj
        except (TypeError, ValueError):
            # If not, convert to string representation
            return str(obj)


def validate_json_serializable(obj: Any, path: str = "root") -> None:
    """
    Validate that an object is JSON serializable and raise informative errors.
    
    Args:
        obj: Object to validate
        path: Current path in object tree (for error messages)
        
    Raises:
        TypeError: If object contains non-serializable data with location info
    """
    try:
        json.dumps(obj)
    except (TypeError, ValueError) as e:
        if isinstance(obj, dict):
            for key, value in obj.items():
                try:
                    json.dumps(value)
                except:
                    raise TypeError(
                        f"Non-serializable data at '{path}.{key}': "
                        f"{type(value).__name__} = {repr(value)[:100]}"
                    ) from e
        else:
            raise TypeError(
                f"Non-serializable data at '{path}': "
                f"{type(obj).__name__} = {repr(obj)[:100]}"
            ) from e


class DatabaseOperationsWrapper:
    """
    Wrapper class that automatically converts NumPy arrays before database operations.
    
    Usage:
    ------
    # Wrap your existing db_ops
    from db_ops_numpy_fix import DatabaseOperationsWrapper
    
    db_ops_safe = DatabaseOperationsWrapper(db_ops)
    
    # Now use db_ops_safe instead of db_ops
    db_ops_safe.add_process_step(sample_id=1, parsed_data=data_with_numpy)
    """
    
    def __init__(self, db_ops):
        """
        Args:
            db_ops: Your existing DatabaseOperations instance
        """
        self._db_ops = db_ops
    
    def __getattr__(self, name):
        """Delegate all other methods to the wrapped db_ops"""
        return getattr(self._db_ops, name)
    
    def add_process_step(self, sample_id, process_type, labview_file_path, 
                        parsed_data=None, **kwargs):
        """
        Wrapped version of add_process_step that ensures JSON serialization.
        
        This automatically converts any NumPy arrays in parsed_data before
        passing to the database.
        """
        # Convert parsed_data to ensure it's JSON serializable
        if parsed_data is not None:
            parsed_data = ensure_json_serializable(parsed_data)
        
        # Convert any other kwargs that might contain numpy
        kwargs = ensure_json_serializable(kwargs)
        
        # Call the original method
        return self._db_ops.add_process_step(
            sample_id=sample_id,
            process_type=process_type,
            labview_file_path=labview_file_path,
            parsed_data=parsed_data,
            **kwargs
        )


def patch_database_operations(db_ops_class):
    """
    Monkey-patch a DatabaseOperations class to automatically handle NumPy conversion.
    
    Usage:
    ------
    # At the top of your main script, after importing DatabaseOperations:
    from db_operations import DatabaseOperations
    from db_ops_numpy_fix import patch_database_operations
    
    patch_database_operations(DatabaseOperations)
    
    # Now all instances will automatically convert NumPy
    db_ops = DatabaseOperations(db_path)
    """
    original_add_process_step = db_ops_class.add_process_step
    
    def patched_add_process_step(self, sample_id, process_type, labview_file_path,
                                 parsed_data=None, **kwargs):
        """Patched version that converts NumPy before calling original"""
        if parsed_data is not None:
            parsed_data = ensure_json_serializable(parsed_data)
        
        kwargs = ensure_json_serializable(kwargs)
        
        return original_add_process_step(
            self, sample_id, process_type, labview_file_path,
            parsed_data=parsed_data, **kwargs
        )
    
    # Replace the method
    db_ops_class.add_process_step = patched_add_process_step
    
    print("✓ DatabaseOperations patched for NumPy array handling")


# ============================================================================
# DIAGNOSTIC FUNCTIONS
# ============================================================================

def find_numpy_arrays(obj: Any, path: str = "root", max_depth: int = 10) -> List[str]:
    """
    Find all NumPy arrays in a nested data structure.
    
    Useful for debugging to see where NumPy arrays are hiding.
    
    Args:
        obj: Object to search
        path: Current path (for reporting)
        max_depth: Maximum recursion depth
        
    Returns:
        List of paths where NumPy arrays were found
    """
    if max_depth <= 0:
        return []
    
    found = []
    
    if isinstance(obj, np.ndarray):
        found.append(f"{path} (shape: {obj.shape}, dtype: {obj.dtype})")
    elif isinstance(obj, dict):
        for key, value in obj.items():
            found.extend(find_numpy_arrays(value, f"{path}.{key}", max_depth - 1))
    elif isinstance(obj, (list, tuple)):
        for i, item in enumerate(obj):
            found.extend(find_numpy_arrays(item, f"{path}[{i}]", max_depth - 1))
    
    return found


def test_conversion(test_data: Any) -> bool:
    """
    Test if data can be successfully converted and serialized.
    
    Args:
        test_data: Data to test
        
    Returns:
        True if conversion successful, False otherwise
    """
    try:
        # Convert
        converted = ensure_json_serializable(test_data)
        
        # Try to serialize
        json.dumps(converted)
        
        print("✓ Conversion successful")
        return True
        
    except Exception as e:
        print(f"✗ Conversion failed: {e}")
        
        # Show where NumPy arrays are
        arrays = find_numpy_arrays(test_data)
        if arrays:
            print("\nNumPy arrays found at:")
            for arr_path in arrays:
                print(f"  - {arr_path}")
        
        return False


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example 1: Test conversion of various NumPy types
    print("=== Example 1: Testing NumPy Conversion ===")
    
    test_data = {
        'array': np.array([1, 2, 3]),
        'nested': {
            'matrix': np.array([[1, 2], [3, 4]]),
            'scalar': np.int64(42),
            'float': np.float32(3.14)
        },
        'mixed': [1, 2.0, np.array([5, 6])]
    }
    
    print("Original data types:")
    print(f"  array: {type(test_data['array'])}")
    print(f"  nested.matrix: {type(test_data['nested']['matrix'])}")
    
    converted = ensure_json_serializable(test_data)
    
    print("\nConverted data types:")
    print(f"  array: {type(converted['array'])}")
    print(f"  nested.matrix: {type(converted['nested']['matrix'])}")
    
    # Verify it's JSON serializable
    json_str = json.dumps(converted, indent=2)
    print("\nJSON output (first 200 chars):")
    print(json_str[:200])
    
    # Example 2: Find NumPy arrays in nested structure
    print("\n\n=== Example 2: Finding NumPy Arrays ===")
    
    complex_data = {
        'metadata': {
            'scan_params': {
                'pixels': np.array([512, 512]),
                'range': np.array([10.0, 10.0])
            }
        },
        'data': np.random.rand(10, 10)
    }
    
    arrays_found = find_numpy_arrays(complex_data)
    print(f"Found {len(arrays_found)} NumPy arrays:")
    for path in arrays_found:
        print(f"  {path}")
