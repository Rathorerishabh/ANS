"""
HRV Analysis Main Runner
-----------------------
This script runs the complete HRV analysis pipeline 
by calling the three component scripts in sequence.

It's designed to minimize memory usage by only running
one step at a time and cleaning up between steps.
"""

import os
import sys
import subprocess
import argparse
import gc
import time

def ensure_dir(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def run_full_analysis(file_path, column_name=None, fs=150, chunk_size=None, 
                     signal_type='ppg', correction_level='medium', output_dir='hrv_output'):
    """
    Run the complete HRV analysis pipeline
    
    Parameters:
    -----------
    file_path : str
        Path to the input file (CSV or Excel)
    column_name : str
        Name of the column containing signal data
    fs : float
        Sampling frequency in Hz
    chunk_size : int or None
        Size of chunks to process (for large files)
    signal_type : str
        Type of signal: 'ppg' or 'ecg'
    correction_level : str
        Artifact correction level
    output_dir : str
        Directory to save output files
        
    Returns:
    --------
    bool
        Success status
    """
    start_time = time.time()
    print(f"Starting complete HRV analysis pipeline for {file_path}")
    
    # Create output directory
    ensure_dir(output_dir)
    
    # Step 1: Run signal processor
    print("\n\n" + "="*80)
    print("STEP 1: SIGNAL PROCESSING AND BEAT DETECTION")
    print("="*80)
    
    cmd = [
        "python", "1_signal_processor.py",
        file_path,
        "--fs", str(fs),
        "--signal_type", signal_type,
        "--output_dir", output_dir
    ]
    
    if column_name:
        cmd.extend(["--column_name", column_name])
    
    if chunk_size:
        cmd.extend(["--chunk_size", str(chunk_size)])
    
    try:
        # Run signal processor script
        process = subprocess.run(cmd, check=True)
        
        # Force garbage collection
        gc.collect()
        
        # Check if output file exists
        intervals_path = os.path.join(output_dir, 'raw_intervals.csv')
        if not os.path.exists(intervals_path):
            print(f"Error: {intervals_path} not found. Step 1 may have failed.")
            return False
        
    except subprocess.CalledProcessError as e:
        print(f"Error running signal processor: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error in Step 1: {e}")
        return False
    
    # Step 2: Run interval processor
    print("\n\n" + "="*80)
    print("STEP 2: INTERVAL PROCESSING AND ARTIFACT CORRECTION")
    print("="*80)
    
    cmd = [
        "python", "2_interval_processor.py",
        intervals_path,
        "--correction", correction_level,
        "--output_dir", output_dir
    ]
    
    try:
        # Run interval processor script
        process = subprocess.run(cmd, check=True)
        
        # Force garbage collection
        gc.collect()
        
        # Check if output file exists
        corrected_intervals_path = os.path.join(output_dir, 'corrected_intervals.csv')
        if not os.path.exists(corrected_intervals_path):
            print(f"Error: {corrected_intervals_path} not found. Step 2 may have failed.")
            return False
        
    except subprocess.CalledProcessError as e:
        print(f"Error running interval processor: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error in Step 2: {e}")
        return False
    
    # Step 3: Run HRV analyzer
    print("\n\n" + "="*80)
    print("STEP 3: ADVANCED HRV ANALYSIS")
    print("="*80)
    
    cmd = [
        "python", "3_hrv_analyzer.py",
        corrected_intervals_path,
        "--output_dir", output_dir
    ]
    
    try:
        # Run HRV analyzer script
        process = subprocess.run(cmd, check=True)
        
        # Force garbage collection
        gc.collect()
        
        # Check if output file exists
        hrv_params_path = os.path.join(output_dir, 'hrv_parameters.csv')
        if not os.path.exists(hrv_params_path):
            print(f"Warning: {hrv_params_path} not found. Step 3 may have completed with issues.")
        
    except subprocess.CalledProcessError as e:
        print(f"Error running HRV analyzer: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error in Step 3: {e}")
        return False
    
    # Print summary
    end_time = time.time()
    duration = end_time - start_time
    
    print("\n\n" + "="*80)
    print(f"HRV ANALYSIS COMPLETE - Total time: {duration:.1f} seconds")
    print("="*80)
    print(f"Results saved in: {output_dir}")
    
    # List output files
    print("\nOutput files:")
    try:
        for file in sorted(os.listdir(output_dir)):
            file_path = os.path.join(output_dir, file)
            file_size = os.path.getsize(file_path) / 1024  # KB
            print(f"  - {file} ({file_size:.1f} KB)")
    except Exception as e:
        print(f"Error listing output files: {e}")
    
    return True

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run complete HRV analysis pipeline')
    parser.add_argument('file_path', help='Path to the file containing signal data')
    parser.add_argument('--column_name', help='Name of the column containing signal data')
    parser.add_argument('--fs', type=float, default=150, help='Sampling frequency in Hz (default: 150)')
    parser.add_argument('--signal_type', default='ppg', choices=['ppg', 'ecg'], 
                       help='Type of signal: ppg or ecg (default: ppg)')
    parser.add_argument('--correction', default='medium', 
                       choices=['none', 'very_low', 'low', 'medium', 'strong', 'very_strong'],
                       help='Artifact correction level (default: medium)')
    parser.add_argument('--output_dir', default='hrv_output', help='Output directory (default: hrv_output)')
    parser.add_argument('--chunk_size', type=int, help='Process in chunks of this size (for large files)')
    
    args = parser.parse_args()
    
    # Run the pipeline
    run_full_analysis(
        args.file_path,
        column_name=args.column_name,
        fs=args.fs,
        chunk_size=args.chunk_size,
        signal_type=args.signal_type,
        correction_level=args.correction,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main()