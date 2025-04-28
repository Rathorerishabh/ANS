"""
HRV Analysis Part 1: Signal Processor
------------------------------------
This script loads PPG/ECG signal data, preprocesses it, detects beats,
and saves the beat times and intervals to CSV files.

This is the first step in the HRV processing pipeline.
"""

import numpy as np
import pandas as pd
import scipy.signal as signal
import matplotlib.pyplot as plt
import os
import gc
import time
from pywt import wavedec, waverec, threshold
import argparse

def ensure_dir(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)

class SignalProcessor:
    """
    First step of HRV analysis pipeline: signal processing and beat detection
    """
    
    def __init__(self, file_path, fs=150, signal_type='ppg', output_dir='hrv_output'):
        """
        Initialize the signal processor
        
        Parameters:
        -----------
        file_path : str
            Path to the file containing signal data
        fs : float
            Sampling frequency in Hz
        signal_type : str
            Type of signal: 'ppg' or 'ecg'
        output_dir : str
            Directory to save output files
        """
        self.file_path = file_path
        self.fs = fs
        self.signal_type = signal_type.lower()
        self.output_dir = output_dir
        
        # Create output directory
        ensure_dir(self.output_dir)
        
        # Processing parameters
        self.wavelet_family = 'db4'
        self.wavelet_level = 3
        self.filter_low = 0.5   # Hz - lower cutoff for bandpass filter
        self.filter_high = 8.0  # Hz - upper cutoff for bandpass filter
        self.filter_order = 4
        self.peak_distance = 0.5  # Seconds
        self.peak_prominence = 0.3  # Relative prominence
        self.detrend_lambda = 500  # For smoothness priors detrending
        
    def load_data(self, column_name=None, chunk_size=None):
        """
        Load data from file
        
        Parameters:
        -----------
        column_name : str or None
            Column name to use from dataframe
        chunk_size : int or None
            Size of chunks to process at once (for large files)
            
        Returns:
        --------
        tuple
            (raw_signal, time_axis) or None if error
        """
        try:
            start_time = time.time()
            print(f"Loading data from {self.file_path}...")
            
            # Determine file type
            file_ext = os.path.splitext(self.file_path)[1].lower()
            
            if chunk_size is not None and file_ext == '.csv':
                # For very large files, load in chunks
                print(f"Loading large file in chunks of {chunk_size} rows")
                return self._load_in_chunks(column_name, chunk_size)
            
            if file_ext == '.csv':
                if column_name:
                    df = pd.read_csv(self.file_path, usecols=[column_name])
                    raw_signal = df[column_name].values
                else:
                    df = pd.read_csv(self.file_path)
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        raw_signal = df[numeric_cols[0]].values
                        column_name = numeric_cols[0]
                    else:
                        raise ValueError("No numeric columns found in the file")
                
            elif file_ext in ['.xls', '.xlsx']:
                if column_name:
                    df = pd.read_excel(self.file_path, usecols=[column_name])
                    raw_signal = df[column_name].values
                else:
                    df = pd.read_excel(self.file_path)
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        raw_signal = df[numeric_cols[0]].values
                        column_name = numeric_cols[0]
                    else:
                        raise ValueError("No numeric columns found in the file")
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")
            
            # Create time axis
            time_axis = np.arange(0, len(raw_signal) / self.fs, 1 / self.fs)[:len(raw_signal)]
            
            # Print info
            duration = len(raw_signal) / self.fs
            print(f"Loaded {len(raw_signal)} samples ({duration:.2f} seconds)")
            print(f"Data loading completed in {time.time() - start_time:.2f} seconds")
            
            return raw_signal, time_axis, column_name
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return None, None, None
    
    def _load_in_chunks(self, column_name, chunk_size):
        """
        Load and process large files in chunks
        
        This is only the setup function; actual processing happens in process_signal_chunks
        """
        try:
            # First, determine the column to use if not specified
            if column_name is None:
                # Read just the first chunk to find numeric columns
                first_chunk = pd.read_csv(self.file_path, nrows=100)
                numeric_cols = first_chunk.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    column_name = numeric_cols[0]
                else:
                    raise ValueError("No numeric columns found in the file")
            
            # Don't actually load data here, just return the column name
            # The actual loading will happen in process_signal_chunks
            return None, None, column_name
            
        except Exception as e:
            print(f"Error setting up chunk processing: {e}")
            return None, None, None
    
    def process_signal(self, raw_signal, time_axis=None):
        """
        Preprocess the signal data
        
        Parameters:
        -----------
        raw_signal : np.ndarray
            Raw signal data
        time_axis : np.ndarray or None
            Time axis for the signal
            
        Returns:
        --------
        np.ndarray
            Processed signal
        """
        start_time = time.time()
        print("Preprocessing signal...")
        
        try:
            # Step 1: Wavelet denoising
            coeffs = wavedec(raw_signal, self.wavelet_family, level=self.wavelet_level)
            sigma = np.median(np.abs(coeffs[-3])) / 0.6745
            uthresh = sigma * np.sqrt(2 * np.log(len(raw_signal)))
            coeffs[1:] = [threshold(c, value=uthresh, mode='soft') for c in coeffs[1:]]
            denoised_signal = waverec(coeffs, self.wavelet_family)[:len(raw_signal)]
            
            # Free memory
            del coeffs
            gc.collect()
            
            # Step 2: Bandpass filtering
            nyquist = 0.5 * self.fs
            low = self.filter_low / nyquist
            high = self.filter_high / nyquist
            b, a = signal.butter(self.filter_order, [low, high], btype='band')
            filtered_signal = signal.filtfilt(b, a, denoised_signal)
            
            # Free memory
            del denoised_signal
            gc.collect()
            
            # Step 3: Detrending using smoothness priors
            detrended_signal = self._smoothness_priors_detrend(filtered_signal, lam=self.detrend_lambda)
            
            # Free memory
            del filtered_signal
            gc.collect()
            
            print(f"Preprocessing completed in {time.time() - start_time:.2f} seconds")
            
            return detrended_signal
            
        except Exception as e:
            print(f"Error during preprocessing: {e}")
            return None
    
    def _smoothness_priors_detrend(self, signal_data, lam=500):
        """
        Implement the smoothness priors detrending method
        
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
        # Use a simpler detrending method for efficiency
        # This is a simplified version of the Kubios method
        try:
            # For very large signals, use simple detrending to save memory
            if len(signal_data) > 50000:
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
    
    def detect_beats(self, processed_signal):
        """
        Detect beats in the processed signal
        
        Parameters:
        -----------
        processed_signal : np.ndarray
            Processed signal data
            
        Returns:
        --------
        tuple
            (beat_peaks, peak_times, intervals)
        """
        start_time = time.time()
        print("Detecting beats...")
        
        try:
            if self.signal_type == 'ppg':
                # For PPG, use adaptive peak detection
                min_distance = int(self.peak_distance * self.fs)
                
                peaks, _ = signal.find_peaks(
                    processed_signal,
                    distance=min_distance,
                    prominence=self.peak_prominence * np.std(processed_signal)
                )
                
            elif self.signal_type == 'ecg':
                # For ECG, use Pan-Tompkins inspired algorithm
                # First, enhance QRS complex
                diff_signal = np.diff(processed_signal)
                squared_signal = diff_signal ** 2
                
                # Use moving average to smooth
                window_size = int(0.08 * self.fs)  # 80 ms window
                if window_size % 2 == 0:
                    window_size += 1  # Ensure odd size
                smoothed_signal = signal.medfilt(squared_signal, window_size)
                
                # Detect peaks on the transformed signal
                min_distance = int(self.peak_distance * self.fs)
                peaks, _ = signal.find_peaks(
                    smoothed_signal,
                    distance=min_distance,
                    height=0.3 * np.mean(smoothed_signal)
                )
                
                # Refine peak detection to get the actual R-peak
                refined_peaks = []
                search_window = int(0.05 * self.fs)  # 50 ms
                
                for peak in peaks:
                    start = max(0, peak - search_window)
                    end = min(len(processed_signal) - 1, peak + search_window)
                    
                    # Find the maximum within the window
                    max_idx = start + np.argmax(processed_signal[start:end+1])
                    refined_peaks.append(max_idx)
                
                peaks = np.array(refined_peaks)
            
            # Convert peak indices to times
            peak_times = peaks / self.fs
            
            # Calculate intervals
            intervals = np.diff(peak_times)
            
            print(f"Detected {len(peaks)} beats")
            if len(intervals) > 0:
                print(f"Mean heart rate: {60 / np.mean(intervals):.1f} BPM")
            print(f"Beat detection completed in {time.time() - start_time:.2f} seconds")
            
            return peaks, peak_times, intervals
            
        except Exception as e:
            print(f"Error during beat detection: {e}")
            return None, None, None
    
    def process_and_save(self, column_name=None, chunk_size=None):
        """
        Process signal data and save results
        
        Parameters:
        -----------
        column_name : str or None
            Column name to use
        chunk_size : int or None
            Size of chunks to process (for large files)
            
        Returns:
        --------
        dict
            Paths to output files
        """
        output_files = {}
        
        # Check if we need to process in chunks
        if chunk_size is not None:
            return self.process_signal_chunks(column_name, chunk_size)
        
        # Regular processing for smaller files
        raw_signal, time_axis, column_name = self.load_data(column_name)
        
        if raw_signal is None:
            return None
        
        # Process signal
        processed_signal = self.process_signal(raw_signal, time_axis)
        
        if processed_signal is None:
            return None
        
        # Free memory
        del raw_signal
        gc.collect()
        
        # Detect beats
        peaks, peak_times, intervals = self.detect_beats(processed_signal)
        
        if peaks is None:
            return None
        
        # Save a preview plot
        try:
            self.save_preview_plot(processed_signal, peaks, time_axis)
            output_files['preview_plot'] = os.path.join(self.output_dir, 'signal_preview.png')
        except Exception as e:
            print(f"Warning: Could not save preview plot: {e}")
        
        # Free more memory
        del processed_signal
        gc.collect()
        
        # Save beat peaks
        try:
            peaks_df = pd.DataFrame({
                'peak_index': peaks,
                'peak_time': peak_times
            })
            peaks_path = os.path.join(self.output_dir, 'beat_peaks.csv')
            peaks_df.to_csv(peaks_path, index=False)
            output_files['beat_peaks'] = peaks_path
            print(f"Beat peaks saved to {peaks_path}")
        except Exception as e:
            print(f"Error saving beat peaks: {e}")
        
        # Save intervals
        try:
            if len(intervals) > 0:
                intervals_df = pd.DataFrame({
                    'beat_time': peak_times[1:],
                    'interval': intervals,
                    'heart_rate': 60 / intervals
                })
                intervals_path = os.path.join(self.output_dir, 'raw_intervals.csv')
                intervals_df.to_csv(intervals_path, index=False)
                output_files['raw_intervals'] = intervals_path
                print(f"Raw intervals saved to {intervals_path}")
        except Exception as e:
            print(f"Error saving intervals: {e}")
        
        print("\nSignal processing complete!")
        print(f"- Total samples processed: {len(time_axis)}")
        print(f"- Total beats detected: {len(peaks)}")
        print(f"- Output files saved in: {self.output_dir}")
        
        return output_files
    
    def process_signal_chunks(self, column_name, chunk_size):
        """
        Process signal data in chunks for large files
        
        Parameters:
        -----------
        column_name : str
            Column name to use
        chunk_size : int
            Size of chunks to process
            
        Returns:
        --------
        dict
            Paths to output files
        """
        output_files = {}
        
        try:
            print(f"Processing file in chunks of {chunk_size} rows...")
            
            # Process file in chunks
            chunks = pd.read_csv(self.file_path, usecols=[column_name], chunksize=chunk_size)
            
            all_peaks = []
            all_peak_times = []
            total_rows = 0
            row_offset = 0
            
            # Process each chunk
            for chunk_num, chunk in enumerate(chunks):
                # Force garbage collection
                gc.collect()
                
                # Get the signal data for this chunk
                signal_chunk = chunk[column_name].values
                chunk_length = len(signal_chunk)
                
                # Process signal
                processed_chunk = self.process_signal(signal_chunk)
                
                if processed_chunk is None:
                    continue
                
                # Detect beats in this chunk
                chunk_peaks, chunk_peak_times, _ = self.detect_beats(processed_chunk)
                
                if chunk_peaks is None:
                    continue
                
                # Adjust peak indices for the overall signal
                adjusted_peaks = chunk_peaks + row_offset
                all_peaks.extend(adjusted_peaks.tolist())
                all_peak_times.extend(chunk_peak_times.tolist())
                
                # Update row offset for the next chunk
                row_offset += chunk_length
                total_rows += chunk_length
                
                # Status update
                print(f"Processed chunk {chunk_num+1}: {total_rows} rows, {len(all_peaks)} beats found")
                
                # Free memory
                del signal_chunk, processed_chunk, chunk_peaks, chunk_peak_times
                gc.collect()
            
            # Convert to arrays
            all_peaks = np.array(all_peaks)
            all_peak_times = np.array(all_peak_times)
            
            # Calculate intervals
            intervals = np.diff(all_peak_times)
            
            # Save beat peaks
            peaks_df = pd.DataFrame({
                'peak_index': all_peaks,
                'peak_time': all_peak_times
            })
            peaks_path = os.path.join(self.output_dir, 'beat_peaks.csv')
            peaks_df.to_csv(peaks_path, index=False)
            output_files['beat_peaks'] = peaks_path
            
            # Save intervals
            if len(intervals) > 0:
                intervals_df = pd.DataFrame({
                    'beat_time': all_peak_times[1:],
                    'interval': intervals,
                    'heart_rate': 60 / intervals
                })
                intervals_path = os.path.join(self.output_dir, 'raw_intervals.csv')
                intervals_df.to_csv(intervals_path, index=False)
                output_files['raw_intervals'] = intervals_path
            
            print("\nChunk processing complete!")
            print(f"- Total samples processed: {total_rows}")
            print(f"- Total beats detected: {len(all_peaks)}")
            print(f"- Output files saved in: {self.output_dir}")
            
            return output_files
            
        except Exception as e:
            print(f"Error in chunk processing: {e}")
            return None
    
    def save_preview_plot(self, processed_signal, peaks, time_axis, max_time=30):
        """
        Save a preview plot of the signal and detected beats
        
        Parameters:
        -----------
        processed_signal : np.ndarray
            Processed signal data
        peaks : np.ndarray
            Indices of detected beats
        time_axis : np.ndarray
            Time axis for the signal
        max_time : float
            Maximum time to plot (seconds)
        """
        try:
            # Create a preview plot showing only a portion of the signal
            plt.figure(figsize=(10, 5), dpi=100)
            
            # Limit to first max_time seconds
            max_samples = min(int(max_time * self.fs), len(processed_signal))
            plot_time = time_axis[:max_samples]
            plot_signal = processed_signal[:max_samples]
            
            # Find peaks within the plot range
            plot_peaks = peaks[peaks < max_samples]
            
            # Plot the signal
            plt.plot(plot_time, plot_signal, 'b-', label='Processed Signal')
            
            # Plot the detected beats
            if len(plot_peaks) > 0:
                plt.plot(plot_peaks / self.fs, plot_signal[plot_peaks], 'ro', label='Detected Beats')
            
            plt.title(f'{self.signal_type.upper()} Signal with Detected Beats (First {max_time} seconds)')
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Save the plot
            plt.tight_layout()
            preview_path = os.path.join(self.output_dir, 'signal_preview.png')
            plt.savefig(preview_path, dpi=100)
            plt.close()
            
        except Exception as e:
            print(f"Error creating preview plot: {e}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process PPG/ECG signal data')
    parser.add_argument('file_path', help='Path to the file containing signal data')
    parser.add_argument('--column_name', help='Name of the column containing signal data')
    parser.add_argument('--fs', type=float, default=150, help='Sampling frequency in Hz (default: 150)')
    parser.add_argument('--signal_type', default='ppg', choices=['ppg', 'ecg'], 
                       help='Type of signal: ppg or ecg (default: ppg)')
    parser.add_argument('--output_dir', default='hrv_output', help='Output directory (default: hrv_output)')
    parser.add_argument('--chunk_size', type=int, help='Process in chunks of this size (for large files)')
    
    args = parser.parse_args()
    
    # Process the signal
    processor = SignalProcessor(
        args.file_path,
        fs=args.fs,
        signal_type=args.signal_type,
        output_dir=args.output_dir
    )
    
    processor.process_and_save(args.column_name, args.chunk_size)

if __name__ == "__main__":
    main()