# check_file_format.py
"""
Quick script to check the format of your files and debug the header issue
"""

import pandas as pd

def check_file_headers(filepath):
    """Check the headers and format of a file"""
    print(f"=== Checking file: {filepath} ===")
    
    # Read first few lines to see the raw format
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    print("First 5 lines of file:")
    for i, line in enumerate(lines[:5]):
        print(f"Line {i}: {repr(line)}")  # repr shows tabs, spaces, etc.
    
    # Find header line
    header_idx = None
    for i, line in enumerate(lines):
        if 'x [m]' in line and 'y [m]' in line:
            header_idx = i
            print(f"\nHeader found at line {i}: {repr(line)}")
            break
    
    if header_idx is None:
        print("ERROR: No header line found!")
        
        # Try to find any line with 'x' or 'y'
        print("Looking for lines with 'x' or 'y':")
        for i, line in enumerate(lines[:10]):
            if 'x' in line.lower() or 'y' in line.lower():
                print(f"Line {i}: {repr(line)}")
        return
    
    # Try to load with pandas
    try:
        df = pd.read_csv(filepath, skiprows=header_idx)
        print(f"\nPandas loading successful!")
        print(f"Columns found: {list(df.columns)}")
        print(f"Data shape: {df.shape}")
        print(f"First few rows:")
        print(df.head())
        
    except Exception as e:
        print(f"\nERROR loading with pandas: {e}")
        
        # Try different separators
        print("Trying different separators...")
        for sep_name, sep_char in [("comma", ","), ("tab", "\t"), ("space", " "), ("semicolon", ";")]:
            try:
                df_test = pd.read_csv(filepath, skiprows=header_idx, sep=sep_char)
                print(f"  {sep_name} separator: SUCCESS - {df_test.shape}, columns: {list(df_test.columns)}")
            except:
                print(f"  {sep_name} separator: FAILED")

def compare_files(original_piv_file, averaged_file):
    """Compare original PIV file with averaged file"""
    print("\n" + "="*60)
    print("COMPARING ORIGINAL vs AVERAGED FILES")
    print("="*60)
    
    print("\nORIGINAL PIV FILE:")
    check_file_headers(original_piv_file)
    
    print("\nAVERAGED FILE:")
    check_file_headers(averaged_file)

if __name__ == "__main__":
    # Check your averaged file
    averaged_file = "averaged_piv_steady_state.txt"
    
    # Also check an original PIV file for comparison
    original_piv_file = "PIV/PIVs_txts/p1/PIVlab_0901.txt"  # Update this path
    
    # First check just the averaged file
    check_file_headers(averaged_file)
    
    # If you have an original file, compare them
    try:
        compare_files(original_piv_file, averaged_file)
    except FileNotFoundError as e:
        print(f"\nCould not find original file for comparison: {e}")
        print("Update the 'original_piv_file' path in the script to compare formats")