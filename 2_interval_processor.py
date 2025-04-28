"""
HRV Analysis Part 2: Interval Processor
--------------------------------------
This script takes the beat intervals from Part 1, applies artifact correction,
and performs basic HRV analysis.

This is the second step in the HRV processing pipeline.
"""

import numpy as np
import pandas as pd
import scipy.signal as signal
import matplotlib.pyplot as plt
import os
import gc
import time
import argparse

def ensure_dir(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)

class IntervalProcessor:
    """
    Second step of HRV analysis pipeline: interval processing and artifact correction
    """
    
    def __init__(self, intervals_path, output_dir='hrv_output'):
        """
        Initialize the interval processor
        
        Parameters:
        -----------
        intervals_path : str
            Path to CSV file containing intervals from step 1
        output_dir : str
            Directory to save output files
        """
        self.intervals_path = intervals_path
        self.output_dir = output_dir
        ensure_dir(self.output_dir)
        
        # Define thresholds for different correction levels (in seconds)
        self.correction_thresholds = {
            'very_low': 0.45,
            'low': 0.35,
            'medium': 0.25,
            'strong': 0.15,
            'very_strong': 0.05,
            'none': 999  # No correction
        }
    
    def load_intervals(self):
        """
        Load RR intervals from CSV file
        
        Returns:
        --------
        tuple
            (beat_times, intervals, heart_rates) or None if error
        """
        try:
            start_time = time.time()
            print(f"Loading intervals from {self.intervals_path}...")
            
            # Load the data
            df = pd.read_csv(self.intervals_path)
            
            # Extract data based on column names
            if 'interval' in df.columns:
                intervals = df['interval'].values
                beat_times = df['beat_time'].values if 'beat_time' in df.columns else None
                heart_rates = df['heart_rate'].values if 'heart_rate' in df.columns else None
            elif 'RR_Interval' in df.columns:
                # Convert from ms to seconds if needed
                if df['RR_Interval'].mean() > 100:  # Probably in ms
                    intervals = df['RR_Interval'].values / 1000
                else:
                    intervals = df['RR_Interval'].values
                beat_times = df['Beat_Time'].values if 'Beat_Time' in df.columns else None
                heart_rates = df['Heart_Rate'].values if 'Heart_Rate' in df.columns else None
            else:
                raise ValueError("Could not find interval column in the file")
            
            # Recalculate heart rate if it's not in the file
            if heart_rates is None and intervals is not None:
                heart_rates = 60 / intervals
            
            print(f"Loaded {len(intervals)} intervals")
            print(f"Loading completed in {time.time() - start_time:.2f} seconds")
            
            return beat_times, intervals, heart_rates
            
        except Exception as e:
            print(f"Error loading intervals: {e}")
            return None, None, None
    
    def apply_physiological_filter(self, beat_times, intervals, heart_rates, min_hr=30, max_hr=200):
        """
        Filter intervals based on physiological constraints
        
        Parameters:
        -----------
        beat_times : np.ndarray
            Times of each beat
        intervals : np.ndarray
            RR intervals
        heart_rates : np.ndarray
            Heart rates
        min_hr : float
            Minimum physiologically possible heart rate (BPM)
        max_hr : float
            Maximum physiologically possible heart rate (BPM)
            
        Returns:
        --------
        tuple
            (filtered_beat_times, filtered_intervals, filtered_heart_rates)
        """
        try:
            print("Applying physiological constraints...")
            
            # Convert heart rate limits to interval limits
            min_interval = 60 / max_hr  # Convert max HR to min interval
            max_interval = 60 / min_hr  # Convert min HR to max interval
            
            # Create mask for valid intervals
            valid_mask = (intervals >= min_interval) & (intervals <= max_interval)
            
            # Apply mask to all arrays
            filtered_intervals = intervals[valid_mask]
            filtered_beat_times = beat_times[valid_mask] if beat_times is not None else None
            filtered_heart_rates = heart_rates[valid_mask] if heart_rates is not None else None
            
            # Report results
            excluded_count = len(intervals) - len(filtered_intervals)
            if excluded_count > 0:
                print(f"Excluded {excluded_count} intervals ({excluded_count/len(intervals)*100:.1f}%) outside physiological range")
                print(f"Remaining intervals: {len(filtered_intervals)}")
            else:
                print("All intervals are within physiological range")
            
            return filtered_beat_times, filtered_intervals, filtered_heart_rates
            
        except Exception as e:
            print(f"Error in physiological filtering: {e}")
            return beat_times, intervals, heart_rates
    
    def correct_artifacts(self, beat_times, intervals, correction_level='medium'):
        """
        Apply Kubios-inspired artifact correction
        
        Parameters:
        -----------
        beat_times : np.ndarray
            Times of each beat
        intervals : np.ndarray
            RR intervals
        correction_level : str
            Level of correction to apply
            
        Returns:
        --------
        tuple
            (corrected_intervals, artifacts_indices)
        """
        try:
            start_time = time.time()
            print(f"Applying {correction_level} artifact correction...")
            
            # Get threshold for the selected correction level
            if correction_level not in self.correction_thresholds:
                print(f"Unknown correction level: {correction_level}. Using 'medium'.")
                correction_level = 'medium'
            
            threshold = self.correction_thresholds[correction_level]
            
            # If correction level is 'none', return original intervals
            if correction_level == 'none':
                print("No artifact correction applied")
                return intervals, []
            
            # Adjust threshold based on mean heart rate as described in the articles
            mean_interval = np.mean(intervals)
            if mean_interval < 0.6:  # High heart rate (>100 BPM)
                threshold *= 0.8
                print(f"Adjusted threshold for high heart rate: {threshold:.3f}s")
            elif mean_interval > 1.0:  # Low heart rate (<60 BPM)
                threshold *= 1.2
                print(f"Adjusted threshold for low heart rate: {threshold:.3f}s")
            
            # Create corrected intervals array
            corrected_intervals = intervals.copy()
            
            # Apply median filter to get reference values
            window_size = min(11, len(intervals))
            if window_size % 2 == 0:
                window_size -= 1  # Ensure odd window size
            
            median_intervals = signal.medfilt(intervals, window_size)
            
            # Find artifacts based on deviation from local median
            deviations = np.abs(intervals - median_intervals)
            artifacts = np.where(deviations > threshold)[0]
            
            # Apply correction
            if len(artifacts) > 0:
                corrected_intervals[artifacts] = median_intervals[artifacts]
            
            # Report results
            print(f"Found {len(artifacts)} artifacts out of {len(intervals)} intervals ({len(artifacts)/len(intervals)*100:.1f}%)")
            print(f"Artifact correction completed in {time.time() - start_time:.2f} seconds")
            
            return corrected_intervals, artifacts
            
        except Exception as e:
            print(f"Error during artifact correction: {e}")
            return intervals, []
    
    def calculate_basic_hrv(self, intervals):
        """
        Calculate basic time-domain HRV parameters
        
        Parameters:
        -----------
        intervals : np.ndarray
            RR intervals in seconds
            
        Returns:
        --------
        dict
            Dictionary containing HRV parameters
        """
        try:
            print("Calculating basic HRV parameters...")
            
            # Initialize results dictionary
            hrv_params = {}
            
            # Mean RR interval (ms)
            hrv_params['mean_rr_ms'] = np.mean(intervals) * 1000
            
            # Mean heart rate (BPM)
            hrv_params['mean_hr_bpm'] = 60 / np.mean(intervals)
            
            # Standard deviation of NN intervals (SDNN) (ms)
            hrv_params['sdnn_ms'] = np.std(intervals, ddof=1) * 1000
            
            # Root mean square of successive differences (RMSSD) (ms)
            rr_diff = np.diff(intervals)
            hrv_params['rmssd_ms'] = np.sqrt(np.mean(rr_diff**2)) * 1000
            
            # NN50: number of pairs of successive intervals differing by more than 50 ms
            nn50 = np.sum(np.abs(rr_diff) > 0.05)  # 50 ms = 0.05 s
            hrv_params['nn50_count'] = int(nn50)
            
            # pNN50: percentage of NN50
            hrv_params['pnn50_percent'] = (nn50 / len(rr_diff)) * 100 if len(rr_diff) > 0 else 0
            
            # Poincaré plot parameters
            # SD1: standard deviation perpendicular to the line of identity
            sd1 = np.std(np.diff(intervals) / np.sqrt(2), ddof=1) * 1000  # ms
            hrv_params['sd1_ms'] = sd1
            
            # SD2: standard deviation along the line of identity
            # Formula is 2*SDNN² - 0.5*SDSD², where SDSD is approximated by RMSSD
            sd2 = np.sqrt(2 * (hrv_params['sdnn_ms']/1000)**2 - 0.5 * (hrv_params['rmssd_ms']/1000)**2) * 1000  # ms
            hrv_params['sd2_ms'] = sd2
            
            # SD2/SD1 ratio
            hrv_params['sd2_sd1_ratio'] = sd2 / sd1 if sd1 > 0 else 0
            
            # Simplified PNS and SNS indices
            # Standardize against population values (simplified version)
            mean_rr_std = (hrv_params['mean_rr_ms'] - 900) / 140  # Standardized mean RR
            rmssd_std = (hrv_params['rmssd_ms'] - 40) / 30  # Standardized RMSSD
            sd1_std = (hrv_params['sd1_ms'] - 30) / 20  # Standardized SD1
            
            # Calculate PNS index (simplified version)
            hrv_params['pns_index'] = (mean_rr_std + rmssd_std + sd1_std) / 3
            
            # SNS index based on Mean HR and SD2/SD1 ratio
            mean_hr_std = (hrv_params['mean_hr_bpm'] - 70) / 15  # Standardized mean HR
            sd_ratio_std = (hrv_params['sd2_sd1_ratio'] - 2) / 1  # Standardized SD2/SD1 ratio
            
            # Calculate SNS index (simplified version)
            hrv_params['sns_index'] = (mean_hr_std + sd_ratio_std) / 2
            
            return hrv_params
            
        except Exception as e:
            print(f"Error calculating HRV parameters: {e}")
            return {}
    
    def plot_intervals(self, beat_times, intervals, corrected_intervals, artifacts, hrv_params):
        """
        Create plots of intervals with artifacts
        
        Parameters:
        -----------
        beat_times : np.ndarray
            Times of each beat
        intervals : np.ndarray
            Original RR intervals
        corrected_intervals : np.ndarray
            Corrected RR intervals
        artifacts : list
            Indices of artifacts
        hrv_params : dict
            HRV parameters
            
        Returns:
        --------
        str
            Path to saved plot
        """
        try:
            print("Creating interval plots...")
            
            plt.figure(figsize=(12, 8), dpi=100)
            
            # Plot intervals
            plt.subplot(2, 1, 1)
            plt.plot(beat_times, intervals * 1000, 'b-', label='Original Intervals')
            
            # Plot corrected intervals
            plt.plot(beat_times, corrected_intervals * 1000, 'g-', label='Corrected Intervals')
            
            # Highlight artifacts
            if len(artifacts) > 0:
                artifact_times = beat_times[artifacts]
                artifact_values = intervals[artifacts] * 1000
                plt.plot(artifact_times, artifact_values, 'rx', markersize=8, label='Artifacts')
            
            plt.title('RR Intervals with Artifact Correction')
            plt.xlabel('Time (s)')
            plt.ylabel('RR Interval (ms)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Plot Poincaré plot (current RR interval vs next RR interval)
            plt.subplot(2, 1, 2)
            
            # Create Poincaré plot
            rr_n = corrected_intervals[:-1] * 1000  # Current RR intervals (ms)
            rr_n1 = corrected_intervals[1:] * 1000  # Next RR intervals (ms)
            
            plt.scatter(rr_n, rr_n1, s=10, c='b', alpha=0.5)
            plt.plot([np.min(rr_n), np.max(rr_n)], [np.min(rr_n), np.max(rr_n)], 'k--', alpha=0.5)
            
            # Add SD1 and SD2 ellipse if we have these values
            if 'sd1_ms' in hrv_params and 'sd2_ms' in hrv_params:
                sd1 = hrv_params['sd1_ms']
                sd2 = hrv_params['sd2_ms']
                
                # Add text with SD1 and SD2 values
                plt.text(0.05, 0.95, f"SD1: {sd1:.1f} ms\nSD2: {sd2:.1f} ms\nSD2/SD1: {hrv_params['sd2_sd1_ratio']:.2f}",
                         transform=plt.gca().transAxes, verticalalignment='top',
                         bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
            
            plt.title('Poincaré Plot')
            plt.xlabel('RR$_n$ (ms)')
            plt.ylabel('RR$_{n+1}$ (ms)')
            plt.axis('equal')
            plt.grid(True, alpha=0.3)
            
            # Adjust layout and save
            plt.tight_layout()
            plot_path = os.path.join(self.output_dir, 'intervals_plot.png')
            plt.savefig(plot_path, dpi=100)
            plt.close()
            
            print(f"Plot saved to {plot_path}")
            return plot_path
            
        except Exception as e:
            print(f"Error creating plot: {e}")
            return None
    
    def save_results(self, beat_times, intervals, corrected_intervals, artifacts, hrv_params):
        """
        Save results to CSV files
        
        Parameters:
        -----------
        beat_times : np.ndarray
            Times of each beat
        intervals : np.ndarray
            Original RR intervals
        corrected_intervals : np.ndarray
            Corrected RR intervals
        artifacts : list
            Indices of artifacts
        hrv_params : dict
            HRV parameters
            
        Returns:
        --------
        dict
            Paths to output files
        """
        output_files = {}
        
        try:
            print("Saving results...")
            
            # Save corrected intervals
            intervals_df = pd.DataFrame({
                'beat_time': beat_times,
                'original_interval_ms': intervals * 1000,  # Convert to ms
                'corrected_interval_ms': corrected_intervals * 1000,  # Convert to ms
                'original_hr_bpm': 60 / intervals,
                'corrected_hr_bpm': 60 / corrected_intervals,
                'is_artifact': np.zeros(len(intervals), dtype=int)
            })
            
            # Mark artifacts
            if len(artifacts) > 0:
                intervals_df.loc[artifacts, 'is_artifact'] = 1
            
            intervals_path = os.path.join(self.output_dir, 'corrected_intervals.csv')
            intervals_df.to_csv(intervals_path, index=False)
            output_files['corrected_intervals'] = intervals_path
            
            # Save HRV parameters
            if hrv_params:
                params_df = pd.DataFrame([hrv_params])
                params_path = os.path.join(self.output_dir, 'basic_hrv_params.csv')
                params_df.to_csv(params_path, index=False)
                output_files['basic_hrv_params'] = params_path
            
            print(f"Results saved to {self.output_dir}")
            return output_files
            
        except Exception as e:
            print(f"Error saving results: {e}")
            return output_files
    
    def process_and_save(self, correction_level='medium', min_hr=30, max_hr=200):
        """
        Run the complete interval processing pipeline
        
        Parameters:
        -----------
        correction_level : str
            Artifact correction level
        min_hr : float
            Minimum heart rate (BPM)
        max_hr : float
            Maximum heart rate (BPM)
            
        Returns:
        --------
        dict
            Paths to output files
        """
        output_files = {}
        
        # Step 1: Load intervals
        beat_times, intervals, heart_rates = self.load_intervals()
        
        if intervals is None:
            return None
        
        # Step 2: Apply physiological filter
        beat_times, intervals, heart_rates = self.apply_physiological_filter(
            beat_times, intervals, heart_rates, min_hr=min_hr, max_hr=max_hr
        )
        
        # Check if we have enough intervals to continue
        if len(intervals) < 10:
            print("Not enough valid intervals for analysis")
            return None
        
        # Step 3: Correct artifacts
        corrected_intervals, artifacts = self.correct_artifacts(
            beat_times, intervals, correction_level=correction_level
        )
        
        # Step 4: Calculate basic HRV parameters
        hrv_params = self.calculate_basic_hrv(corrected_intervals)
        
        # Step 5: Create plots
        plot_path = self.plot_intervals(beat_times, intervals, corrected_intervals, artifacts, hrv_params)
        if plot_path:
            output_files['intervals_plot'] = plot_path
        
        # Step 6: Save results
        output_files.update(self.save_results(
            beat_times, intervals, corrected_intervals, artifacts, hrv_params
        ))
        
        print("\nInterval processing complete!")
        print(f"- Total intervals processed: {len(intervals)}")
        print(f"- Artifacts corrected: {len(artifacts)}")
        print(f"- Mean heart rate: {hrv_params.get('mean_hr_bpm', 0):.1f} BPM")
        print(f"- SDNN: {hrv_params.get('sdnn_ms', 0):.1f} ms")
        print(f"- RMSSD: {hrv_params.get('rmssd_ms', 0):.1f} ms")
        print(f"- Output files saved in: {self.output_dir}")
        
        return output_files

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process RR intervals and correct artifacts')
    parser.add_argument('intervals_path', help='Path to CSV file containing intervals from step 1')
    parser.add_argument('--correction', default='medium', 
                       choices=['none', 'very_low', 'low', 'medium', 'strong', 'very_strong'],
                       help='Artifact correction level (default: medium)')
    parser.add_argument('--output_dir', default='hrv_output', help='Output directory (default: hrv_output)')
    parser.add_argument('--min_hr', type=float, default=30, help='Minimum heart rate in BPM (default: 30)')
    parser.add_argument('--max_hr', type=float, default=200, help='Maximum heart rate in BPM (default: 200)')
    
    args = parser.parse_args()
    
    # Process the intervals
    processor = IntervalProcessor(args.intervals_path, output_dir=args.output_dir)
    processor.process_and_save(correction_level=args.correction, min_hr=args.min_hr, max_hr=args.max_hr)

if __name__ == "__main__":
    main()