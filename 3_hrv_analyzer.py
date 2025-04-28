"""
HRV Analysis Part 3: Advanced HRV Analysis
-----------------------------------------
This script performs advanced HRV analysis including frequency domain and
nonlinear parameters calculation from corrected intervals.

This is the third and final step in the HRV processing pipeline.
"""

import numpy as np
import pandas as pd
import scipy.signal as signal
from scipy import interpolate
import matplotlib.pyplot as plt
import os
import gc
import time
import json
import argparse

def ensure_dir(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)

class HRVAnalyzer:
    """
    Third step of HRV analysis pipeline: advanced HRV analysis
    """
    
    def __init__(self, intervals_path, output_dir='hrv_output'):
        """
        Initialize the HRV analyzer
        
        Parameters:
        -----------
        intervals_path : str
            Path to CSV file containing corrected intervals
        output_dir : str
            Directory to save output files
        """
        self.intervals_path = intervals_path
        self.output_dir = output_dir
        ensure_dir(self.output_dir)
        
        # Frequency bands
        self.vlf_band = (0.0033, 0.04)  # VLF band (0.0033-0.04 Hz)
        self.lf_band = (0.04, 0.15)     # LF band (0.04-0.15 Hz)
        self.hf_band = (0.15, 0.4)      # HF band (0.15-0.4 Hz)
        
        # Detrending parameter for smoothness priors
        self.detrend_lambda = 500
    
    def load_intervals(self):
        """
        Load corrected RR intervals from CSV file
        
        Returns:
        --------
        tuple
            (beat_times, intervals) or None if error
        """
        try:
            start_time = time.time()
            print(f"Loading intervals from {self.intervals_path}...")
            
            # Load the data
            df = pd.read_csv(self.intervals_path)
            
            # Extract data based on column names
            if 'corrected_interval_ms' in df.columns:
                # Convert from ms to seconds
                intervals = df['corrected_interval_ms'].values / 1000
                beat_times = df['beat_time'].values if 'beat_time' in df.columns else None
            elif 'Corrected_RR_Interval' in df.columns:
                # Convert from ms to seconds if needed
                if df['Corrected_RR_Interval'].mean() > 100:  # Probably in ms
                    intervals = df['Corrected_RR_Interval'].values / 1000
                else:
                    intervals = df['Corrected_RR_Interval'].values
                beat_times = df['Beat_Time'].values if 'Beat_Time' in df.columns else None
            elif 'corrected_interval' in df.columns:
                intervals = df['corrected_interval'].values
                beat_times = df['beat_time'].values if 'beat_time' in df.columns else None
            else:
                # Try to find any column that might contain intervals
                for col in df.columns:
                    if 'interval' in col.lower() or 'rr' in col.lower():
                        print(f"Using column: {col}")
                        # Check if values are in ms or seconds
                        if df[col].mean() > 100:  # Probably in ms
                            intervals = df[col].values / 1000
                        else:
                            intervals = df[col].values
                        
                        # Look for a time column
                        time_cols = [c for c in df.columns if 'time' in c.lower() or 'beat' in c.lower()]
                        beat_times = df[time_cols[0]].values if time_cols else None
                        break
                else:
                    raise ValueError("Could not find interval column in the file")
            
            print(f"Loaded {len(intervals)} corrected intervals")
            print(f"Loading completed in {time.time() - start_time:.2f} seconds")
            
            return beat_times, intervals
            
        except Exception as e:
            print(f"Error loading intervals: {e}")
            return None, None
    
    def _interpolate_intervals(self, beat_times, intervals, fs_interp=4):
        """
        Interpolate RR intervals to get evenly sampled time series
        
        Parameters:
        -----------
        beat_times : np.ndarray
            Times of each beat
        intervals : np.ndarray
            RR intervals
        fs_interp : float
            Interpolation frequency in Hz
            
        Returns:
        --------
        tuple
            (regular_time, interpolated_intervals)
        """
        try:
            # Create time points from beat times
            if beat_times is None:
                # If beat times not provided, create from cumulative sum of intervals
                beat_times = np.cumsum(np.insert(intervals, 0, 0))[:-1]
            
            # Create regular time axis for interpolation
            regular_time = np.arange(beat_times[0], beat_times[-1], 1/fs_interp)
            
            # Interpolate intervals
            f_interp = interpolate.interp1d(
                beat_times, intervals, kind='cubic', 
                bounds_error=False, fill_value='extrapolate'
            )
            regular_intervals = f_interp(regular_time)
            
            return regular_time, regular_intervals
            
        except Exception as e:
            print(f"Error in interpolation: {e}")
            return None, None
    
    def _smoothness_priors_detrend(self, signal_data, lam=500):
        """
        Simplified smoothness priors detrending
        
        Parameters:
        -----------
        signal_data : np.ndarray
            Signal to detrend
        lam : float
            Regularization parameter
            
        Returns:
        --------
        np.ndarray
            Detrended signal
        """
        try:
            # For very large signals, use simple detrending to save memory
            if len(signal_data) > 10000:
                return signal.detrend(signal_data)
            
            N = len(signal_data)
            
            # Create second-order difference matrix
            D = np.zeros((N-2, N))
            for i in range(N-2):
                D[i, i] = 1
                D[i, i+1] = -2
                D[i, i+2] = 1
            
            # Calculate (I + λ²D'D)⁻¹
            I = np.eye(N)
            DTD = D.T @ D
            H = np.linalg.inv(I + lam * lam * DTD)
            
            # Free memory
            del D, DTD
            gc.collect()
            
            # Calculate trend: z_trend = Hz
            trend = H @ signal_data
            
            # Return detrended signal: z_stat = z - z_trend
            return signal_data - trend
            
        except MemoryError:
            print("Memory error in detrending. Using simpler method.")
            return signal.detrend(signal_data)
        except Exception as e:
            print(f"Error in detrending: {e}. Using simpler method.")
            return signal.detrend(signal_data)
    
    def calculate_frequency_domain(self, beat_times, intervals, fs_interp=4):
        """
        Calculate frequency-domain HRV parameters
        
        Parameters:
        -----------
        beat_times : np.ndarray
            Times of each beat
        intervals : np.ndarray
            RR intervals
        fs_interp : float
            Interpolation frequency in Hz
            
        Returns:
        --------
        dict
            Dictionary containing frequency-domain HRV parameters
        """
        try:
            print("Calculating frequency-domain parameters...")
            
            # Check if we have enough intervals
            if len(intervals) < 30:
                print("Not enough intervals for frequency analysis (need at least 30)")
                return {}
            
            # Interpolate intervals to get evenly sampled data
            regular_time, regular_intervals = self._interpolate_intervals(
                beat_times, intervals, fs_interp=fs_interp
            )
            
            if regular_intervals is None:
                return {}
            
            # Apply detrending to remove slow trends
            detrended_intervals = self._smoothness_priors_detrend(
                regular_intervals, lam=self.detrend_lambda
            )
            
            # Calculate power spectral density using Welch's method
            # Use a smaller window size to reduce memory usage
            nperseg = min(256, len(detrended_intervals))
            if nperseg % 2 != 0:
                nperseg -= 1
            
            f, psd = signal.welch(
                detrended_intervals,
                fs=fs_interp,
                nperseg=nperseg,
                noverlap=nperseg//2,
                scaling='density'
            )
            
            # Calculate power in each band
            vlf_power = np.sum(psd[(f >= self.vlf_band[0]) & (f < self.vlf_band[1])])
            lf_power = np.sum(psd[(f >= self.lf_band[0]) & (f < self.lf_band[1])])
            hf_power = np.sum(psd[(f >= self.hf_band[0]) & (f < self.hf_band[1])])
            
            # Adjust for frequency resolution
            freq_res = f[1] - f[0]
            vlf_power *= freq_res
            lf_power *= freq_res
            hf_power *= freq_res
            
            # Total power
            total_power = vlf_power + lf_power + hf_power
            
            # Initialize results dictionary
            freq_params = {}
            
            # Store absolute powers (ms²)
            freq_params['vlf_power_ms2'] = vlf_power * 1e6
            freq_params['lf_power_ms2'] = lf_power * 1e6
            freq_params['hf_power_ms2'] = hf_power * 1e6
            freq_params['total_power_ms2'] = total_power * 1e6
            
            # Store normalized units for LF and HF
            if lf_power + hf_power > 0:
                freq_params['lf_nu'] = (lf_power / (lf_power + hf_power)) * 100
                freq_params['hf_nu'] = (hf_power / (lf_power + hf_power)) * 100
            else:
                freq_params['lf_nu'] = 0
                freq_params['hf_nu'] = 0
            
            # LF/HF ratio
            freq_params['lf_hf_ratio'] = lf_power / hf_power if hf_power > 0 else 0
            
            # Store frequencies for plotting
            freq_params['frequencies'] = f
            freq_params['psd'] = psd
            
            print("Frequency-domain parameters calculated")
            return freq_params
            
        except Exception as e:
            print(f"Error calculating frequency-domain parameters: {e}")
            return {}
    
    def calculate_nonlinear_params(self, intervals):
        """
        Calculate nonlinear HRV parameters
        
        Parameters:
        -----------
        intervals : np.ndarray
            RR intervals
            
        Returns:
        --------
        dict
            Dictionary containing nonlinear HRV parameters
        """
        try:
            print("Calculating nonlinear parameters...")
            
            # Initialize results dictionary
            nonlinear_params = {}
            
            # Poincaré plot parameters
            # SD1: standard deviation perpendicular to the line of identity
            sd1 = np.std(np.diff(intervals) / np.sqrt(2), ddof=1) * 1000  # ms
            nonlinear_params['sd1_ms'] = sd1
            
            # SD2: standard deviation along the line of identity
            # Using the formula: SD2² = 2*SDNN² - 0.5*SD1²
            sdnn = np.std(intervals, ddof=1) * 1000  # ms
            sd2 = np.sqrt(2 * sdnn**2 - 0.5 * sd1**2)
            nonlinear_params['sd2_ms'] = sd2
            
            # SD2/SD1 ratio
            nonlinear_params['sd2_sd1_ratio'] = sd2 / sd1 if sd1 > 0 else 0
            
            # Simplified Sample Entropy (if we have enough intervals)
            if len(intervals) >= 100:
                try:
                    # Sample entropy with m=2, r=0.2*SDNN
                    r = 0.2 * np.std(intervals)
                    sample_entropy = self._calculate_sample_entropy(intervals, m=2, r=r)
                    nonlinear_params['sample_entropy'] = sample_entropy
                except Exception as e:
                    print(f"Error calculating sample entropy: {e}")
            
            # Simplified Detrended Fluctuation Analysis (DFA)
            if len(intervals) >= 100:
                try:
                    # Calculate short-term (Alpha1) and long-term (Alpha2) scaling exponents
                    alpha1, alpha2 = self._calculate_dfa(intervals)
                    nonlinear_params['dfa_alpha1'] = alpha1
                    nonlinear_params['dfa_alpha2'] = alpha2
                except Exception as e:
                    print(f"Error calculating DFA: {e}")
            
            print("Nonlinear parameters calculated")
            return nonlinear_params
            
        except Exception as e:
            print(f"Error calculating nonlinear parameters: {e}")
            return {}
    
    def _calculate_sample_entropy(self, intervals, m=2, r=0.2):
        """
        Calculate sample entropy for the intervals
        
        Parameters:
        -----------
        intervals : np.ndarray
            RR intervals
        m : int
            Embedding dimension
        r : float
            Tolerance
            
        Returns:
        --------
        float
            Sample entropy value
        """
        try:
            # This is a simplified implementation of sample entropy
            # It may not be as accurate as the full algorithm but uses less memory
            
            # For very large datasets, use a random subset to save memory
            if len(intervals) > 5000:
                print("Using a subset of data for sample entropy calculation")
                np.random.seed(42)  # For reproducibility
                idx = np.random.choice(len(intervals), 5000, replace=False)
                intervals = intervals[np.sort(idx)]
            
            # Create embedding vectors of length m and m+1
            def create_templates(series, m):
                templates = []
                for i in range(len(series) - m + 1):
                    templates.append(series[i:i+m])
                return np.array(templates)
            
            # Count matches
            def count_matches(templates, r):
                N = len(templates)
                count = 0
                for i in range(N):
                    for j in range(i+1, N):  # Avoid self-matches
                        if np.max(np.abs(templates[i] - templates[j])) <= r:
                            count += 1
                return count
            
            # Calculate number of matches for m and m+1
            templates_m = create_templates(intervals, m)
            templates_m1 = create_templates(intervals, m+1)
            
            # Count matches for both templates
            count_m = count_matches(templates_m, r)
            count_m1 = count_matches(templates_m1, r)
            
            # Calculate sample entropy
            if count_m > 0:
                return -np.log(count_m1 / count_m)
            else:
                return float('nan')
            
        except Exception as e:
            print(f"Error in sample entropy calculation: {e}")
            return float('nan')
    
    def _calculate_dfa(self, intervals, scales=None):
        """
        Calculate Detrended Fluctuation Analysis (DFA) parameters
        
        Parameters:
        -----------
        intervals : np.ndarray
            RR intervals
        scales : np.ndarray or None
            Scales to use for DFA
            
        Returns:
        --------
        tuple
            (alpha1, alpha2) - short and long-term scaling exponents
        """
        try:
            # Use a subset of the data if it's very large
            if len(intervals) > 5000:
                print("Using a subset of data for DFA calculation")
                intervals = intervals[:5000]
            
            # Default scales if not provided
            if scales is None:
                scales = np.logspace(np.log10(4), np.log10(min(64, len(intervals)//4)), 10, dtype=int)
            
            # Calculate cumulative sum of mean-subtracted intervals
            y = np.cumsum(intervals - np.mean(intervals))
            
            # Calculate fluctuation for each scale
            fluct = np.zeros(len(scales))
            
            for i, scale in enumerate(scales):
                # Number of segments
                n_segments = len(y) // scale
                
                if n_segments == 0:
                    continue
                
                # Calculate local trend for each segment
                err = np.zeros(n_segments)
                
                for j in range(n_segments):
                    segment = y[j*scale:(j+1)*scale]
                    segment_time = np.arange(scale)
                    # Fit linear trend to segment
                    p = np.polyfit(segment_time, segment, 1)
                    # Calculate error
                    trend = np.polyval(p, segment_time)
                    err[j] = np.sqrt(np.mean((segment - trend)**2))
                
                # Calculate RMS fluctuation across all segments
                fluct[i] = np.sqrt(np.mean(err**2))
            
            # Calculate scaling exponents using log-log linear regression
            # Alpha1: short-term fluctuations (usually scales 4-16)
            # Alpha2: long-term fluctuations (usually scales 16-64)
            
            # Determine split point between short and long term
            split_idx = np.searchsorted(scales, 16)
            if split_idx == 0:
                split_idx = len(scales) // 2
            
            # Calculate short-term exponent (alpha1)
            if split_idx > 1:
                log_scales1 = np.log10(scales[:split_idx])
                log_fluct1 = np.log10(fluct[:split_idx])
                alpha1 = np.polyfit(log_scales1, log_fluct1, 1)[0]
            else:
                alpha1 = float('nan')
            
            # Calculate long-term exponent (alpha2)
            if len(scales) - split_idx > 1:
                log_scales2 = np.log10(scales[split_idx:])
                log_fluct2 = np.log10(fluct[split_idx:])
                alpha2 = np.polyfit(log_scales2, log_fluct2, 1)[0]
            else:
                alpha2 = float('nan')
            
            return alpha1, alpha2
            
        except Exception as e:
            print(f"Error in DFA calculation: {e}")
            return float('nan'), float('nan')
    
    def calculate_time_domain(self, intervals):
        """
        Calculate time-domain HRV parameters
        
        Parameters:
        -----------
        intervals : np.ndarray
            RR intervals
            
        Returns:
        --------
        dict
            Dictionary containing time-domain HRV parameters
        """
        try:
            print("Calculating time-domain parameters...")
            
            # Initialize results dictionary
            time_params = {}
            
            # Mean RR interval (ms)
            time_params['mean_rr_ms'] = np.mean(intervals) * 1000
            
            # Mean heart rate (BPM)
            time_params['mean_hr_bpm'] = 60 / np.mean(intervals)
            
            # Standard deviation of NN intervals (SDNN) (ms)
            time_params['sdnn_ms'] = np.std(intervals, ddof=1) * 1000
            
            # Root mean square of successive differences (RMSSD) (ms)
            rr_diff = np.diff(intervals)
            time_params['rmssd_ms'] = np.sqrt(np.mean(rr_diff**2)) * 1000
            
            # NN50: number of pairs of successive intervals differing by more than 50 ms
            nn50 = np.sum(np.abs(rr_diff) > 0.05)  # 50 ms = 0.05 s
            time_params['nn50_count'] = int(nn50)
            
            # pNN50: percentage of NN50
            time_params['pnn50_percent'] = (nn50 / len(rr_diff)) * 100 if len(rr_diff) > 0 else 0
            
            print("Time-domain parameters calculated")
            return time_params
            
        except Exception as e:
            print(f"Error calculating time-domain parameters: {e}")
            return {}
    
    def plot_frequency_domain(self, freq_params):
        """
        Create frequency-domain plot
        
        Parameters:
        -----------
        freq_params : dict
            Dictionary containing frequency-domain parameters
            
        Returns:
        --------
        str
            Path to saved plot
        """
        try:
            if not freq_params or 'frequencies' not in freq_params or 'psd' not in freq_params:
                print("No frequency-domain data to plot")
                return None
            
            print("Creating frequency-domain plot...")
            
            f = freq_params['frequencies']
            psd = freq_params['psd']
            
            plt.figure(figsize=(10, 6), dpi=100)
            
            # Plot PSD
            plt.semilogy(f, psd * 1e6, 'b-', linewidth=2)  # Convert to ms²/Hz
            
            # Highlight frequency bands
            vlf_mask = (f >= self.vlf_band[0]) & (f < self.vlf_band[1])
            lf_mask = (f >= self.lf_band[0]) & (f < self.lf_band[1])
            hf_mask = (f >= self.hf_band[0]) & (f < self.hf_band[1])
            
            plt.fill_between(f[vlf_mask], 0, psd[vlf_mask] * 1e6, color='gray', alpha=0.3, label='VLF')
            plt.fill_between(f[lf_mask], 0, psd[lf_mask] * 1e6, color='blue', alpha=0.3, label='LF')
            plt.fill_between(f[hf_mask], 0, psd[hf_mask] * 1e6, color='green', alpha=0.3, label='HF')
            
            # Add band limits
            plt.axvline(x=self.vlf_band[1], color='gray', linestyle='--', alpha=0.6)
            plt.axvline(x=self.lf_band[1], color='gray', linestyle='--', alpha=0.6)
            
            plt.title('HRV Power Spectral Density')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('PSD (ms²/Hz)')
            plt.xlim([0, 0.5])  # Limit x-axis to 0.5 Hz
            plt.legend()
            plt.grid(True, which='both', alpha=0.3)
            
            # Add text with frequency-domain parameters
            info_text = (
                f"VLF: {freq_params['vlf_power_ms2']:.1f} ms²\n"
                f"LF: {freq_params['lf_power_ms2']:.1f} ms² ({freq_params['lf_nu']:.1f} n.u.)\n"
                f"HF: {freq_params['hf_power_ms2']:.1f} ms² ({freq_params['hf_nu']:.1f} n.u.)\n"
                f"LF/HF: {freq_params['lf_hf_ratio']:.2f}"
            )
            
            plt.text(0.05, 0.95, info_text, transform=plt.gca().transAxes,
                     verticalalignment='top', horizontalalignment='left',
                     bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
            
            # Save plot
            plt.tight_layout()
            plot_path = os.path.join(self.output_dir, 'frequency_domain_plot.png')
            plt.savefig(plot_path, dpi=100)
            plt.close()
            
            print(f"Plot saved to {plot_path}")
            return plot_path
            
        except Exception as e:
            print(f"Error creating frequency-domain plot: {e}")
            return None
    
    def plot_summary(self, time_params, freq_params, nonlinear_params):
        """
        Create a summary plot with key HRV parameters
        
        Parameters:
        -----------
        time_params : dict
            Time-domain parameters
        freq_params : dict
            Frequency-domain parameters
        nonlinear_params : dict
            Nonlinear parameters
            
        Returns:
        --------
        str
            Path to saved plot
        """
        try:
            print("Creating summary plot...")
            
            plt.figure(figsize=(12, 8), dpi=100)
            
            # Time-domain parameters
            plt.subplot(2, 2, 1)
            params = ['sdnn_ms', 'rmssd_ms', 'pnn50_percent']
            values = [time_params.get(p, 0) for p in params]
            labels = ['SDNN', 'RMSSD', 'pNN50']
            
            plt.bar(labels, values, color=['blue', 'green', 'orange'])
            plt.title('Time-Domain Parameters')
            plt.ylabel('Value')
            plt.grid(True, axis='y', alpha=0.3)
            
            # Add mean HR and RR
            info_text = (
                f"Mean HR: {time_params.get('mean_hr_bpm', 0):.1f} BPM\n"
                f"Mean RR: {time_params.get('mean_rr_ms', 0):.1f} ms"
            )
            plt.text(0.05, 0.95, info_text, transform=plt.gca().transAxes,
                     verticalalignment='top', horizontalalignment='left',
                     bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
            
            # Frequency-domain parameters
            plt.subplot(2, 2, 2)
            if freq_params:
                params = ['vlf_power_ms2', 'lf_power_ms2', 'hf_power_ms2']
                values = [freq_params.get(p, 0) for p in params]
                labels = ['VLF', 'LF', 'HF']
                
                plt.bar(labels, values, color=['gray', 'blue', 'green'])
                plt.title('Frequency-Domain Powers')
                plt.ylabel('Power (ms²)')
                
                # Add LF/HF and normalized units
                info_text = (
                    f"LF/HF: {freq_params.get('lf_hf_ratio', 0):.2f}\n"
                    f"LF n.u.: {freq_params.get('lf_nu', 0):.1f}\n"
                    f"HF n.u.: {freq_params.get('hf_nu', 0):.1f}"
                )
                plt.text(0.05, 0.95, info_text, transform=plt.gca().transAxes,
                         verticalalignment='top', horizontalalignment='left',
                         bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
            else:
                plt.text(0.5, 0.5, "No frequency-domain data available",
                         transform=plt.gca().transAxes, ha='center', va='center')
            
            plt.grid(True, axis='y', alpha=0.3)
            
            # Nonlinear parameters
            plt.subplot(2, 2, 3)
            if nonlinear_params:
                params = ['sd1_ms', 'sd2_ms']
                values = [nonlinear_params.get(p, 0) for p in params]
                labels = ['SD1', 'SD2']
                
                plt.bar(labels, values, color=['red', 'purple'])
                plt.title('Poincaré Plot Parameters')
                plt.ylabel('Value (ms)')
                
                # Add SD2/SD1 ratio
                info_text = f"SD2/SD1: {nonlinear_params.get('sd2_sd1_ratio', 0):.2f}"
                
                # Add sample entropy if available
                if 'sample_entropy' in nonlinear_params:
                    info_text += f"\nSampEn: {nonlinear_params.get('sample_entropy', 0):.3f}"
                
                # Add DFA if available
                if 'dfa_alpha1' in nonlinear_params:
                    info_text += (
                        f"\nDFA α1: {nonlinear_params.get('dfa_alpha1', 0):.3f}\n"
                        f"DFA α2: {nonlinear_params.get('dfa_alpha2', 0):.3f}"
                    )
                
                plt.text(0.05, 0.95, info_text, transform=plt.gca().transAxes,
                         verticalalignment='top', horizontalalignment='left',
                         bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
            else:
                plt.text(0.5, 0.5, "No nonlinear data available",
                         transform=plt.gca().transAxes, ha='center', va='center')
            
            plt.grid(True, axis='y', alpha=0.3)
            
            # Autonomic balance (PNS/SNS) indicators
            plt.subplot(2, 2, 4)
            
            # Estimate PNS index from RMSSD and SD1
            rmssd = time_params.get('rmssd_ms', 0)
            sd1 = nonlinear_params.get('sd1_ms', 0) if nonlinear_params else 0
            mean_rr = time_params.get('mean_rr_ms', 0)
            
            # Standardize values
            rmssd_std = (rmssd - 40) / 30  # Standardized RMSSD
            sd1_std = (sd1 - 30) / 20      # Standardized SD1
            mean_rr_std = (mean_rr - 900) / 140  # Standardized mean RR
            
            # Calculate PNS index
            pns_index = (mean_rr_std + rmssd_std + sd1_std) / 3
            
            # Estimate SNS index from HR and SD2/SD1
            hr = time_params.get('mean_hr_bpm', 0)
            sd_ratio = nonlinear_params.get('sd2_sd1_ratio', 0) if nonlinear_params else 0
            
            # Standardize values
            hr_std = (hr - 70) / 15  # Standardized HR
            sd_ratio_std = (sd_ratio - 2) / 1  # Standardized SD2/SD1
            
            # Calculate SNS index
            sns_index = (hr_std + sd_ratio_std) / 2
            
            # Plot PNS and SNS indices
            plt.bar(['PNS Index', 'SNS Index'], [pns_index, sns_index], 
                    color=['green', 'red'])
            plt.title('Autonomic Balance')
            plt.ylabel('Index Value')
            plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            plt.ylim([-3, 3])  # Limit y-axis
            plt.grid(True, axis='y', alpha=0.3)
            
            # Add interpretation text
            if pns_index > 0 and sns_index < 0:
                balance_text = "Parasympathetic Dominance"
            elif pns_index < 0 and sns_index > 0:
                balance_text = "Sympathetic Dominance"
            else:
                balance_text = "Balanced Autonomic Activity"
            
            plt.text(0.5, 0.95, balance_text, transform=plt.gca().transAxes,
                     verticalalignment='top', horizontalalignment='center',
                     bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
            
            # Add title to the entire figure
            plt.tight_layout()
            plt.subplots_adjust(top=0.9)
            plt.suptitle('HRV Analysis Summary', fontsize=16)
            
            # Save plot
            plot_path = os.path.join(self.output_dir, 'hrv_summary_plot.png')
            plt.savefig(plot_path, dpi=100)
            plt.close()
            
            print(f"Plot saved to {plot_path}")
            return plot_path
            
        except Exception as e:
            print(f"Error creating summary plot: {e}")
            return None
    
    def run_analysis(self):
        """
        Run the complete HRV analysis
        
        Returns:
        --------
        dict
            Combined HRV parameters
        """
        # Step 1: Load intervals
        beat_times, intervals = self.load_intervals()
        
        if intervals is None or len(intervals) < 10:
            print("Not enough valid intervals for analysis")
            return None
        
        # Step 2: Calculate time-domain parameters
        time_params = self.calculate_time_domain(intervals)
        
        # Step 3: Calculate frequency-domain parameters
        freq_params = self.calculate_frequency_domain(beat_times, intervals)
        
        # Step 4: Calculate nonlinear parameters
        nonlinear_params = self.calculate_nonlinear_params(intervals)
        
        # Step 5: Create plots
        self.plot_frequency_domain(freq_params)
        self.plot_summary(time_params, freq_params, nonlinear_params)
        
        # Step 6: Combine all parameters
        all_params = {}
        all_params.update(time_params)
        all_params.update({k: v for k, v in freq_params.items() 
                           if k not in ['frequencies', 'psd']})  # Exclude arrays
        all_params.update(nonlinear_params)
        
        # Calculate PNS and SNS indices
        all_params['pns_index'] = time_params.get('pns_index', (
            ((time_params.get('mean_rr_ms', 0) - 900) / 140) +  # Standardized mean RR
            ((time_params.get('rmssd_ms', 0) - 40) / 30) +     # Standardized RMSSD
            ((nonlinear_params.get('sd1_ms', 0) - 30) / 20)    # Standardized SD1
        ) / 3)
        
        all_params['sns_index'] = time_params.get('sns_index', (
            ((time_params.get('mean_hr_bpm', 0) - 70) / 15) +  # Standardized mean HR
            ((nonlinear_params.get('sd2_sd1_ratio', 0) - 2) / 1)  # Standardized SD2/SD1
        ) / 2)
        
        # Save results
        self.save_results(all_params)
        
        return all_params
    
    def save_results(self, all_params):
        """
        Save results to files
        
        Parameters:
        -----------
        all_params : dict
            Dictionary with all HRV parameters
        """
        try:
            print("Saving results...")
            
            # Save as CSV
            df = pd.DataFrame([all_params])
            csv_path = os.path.join(self.output_dir, 'hrv_parameters.csv')
            df.to_csv(csv_path, index=False)
            
            # Save as JSON (more readable)
            json_path = os.path.join(self.output_dir, 'hrv_parameters.json')
            with open(json_path, 'w') as f:
                json.dump(all_params, f, indent=4)
            
            print(f"Results saved to {csv_path} and {json_path}")
            
        except Exception as e:
            print(f"Error saving results: {e}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Advanced HRV analysis')
    parser.add_argument('intervals_path', help='Path to CSV file containing corrected intervals')
    parser.add_argument('--output_dir', default='hrv_output', help='Output directory (default: hrv_output)')
    
    args = parser.parse_args()
    
    # Run the analysis
    analyzer = HRVAnalyzer(args.intervals_path, output_dir=args.output_dir)
    analyzer.run_analysis()

if __name__ == "__main__":
    main()