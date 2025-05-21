# -*- coding: utf-8 -*-
"""
Created on Tue May 20 10:52:54 2025

@author: quiks and Gemini Pro 2.5

Instructions for Input Data Structure:
1. Project Folder: Create a main project folder. This is your "Input Data Directory".
2. Replicate Folders: Inside the Project Folder, create a separate sub-folder for each 
   of your experimental replicates (e.g., "Replicate1", "SensorA"). 
3. The number of these sub-folders determines 'N'.
4. Data Files (.DTA): Within each replicate folder:
   - Baseline/Initial File: MUST be named '0 (1).DTA'. This is treated as sequence point 1.
   - Subsequent Files: Named '0 (2).DTA', '0 (3).DTA', and so on. The number in 
     parentheses is the sequence/day value.
   - Fallback: If the '0 (X).DTA' pattern is not found, the script will attempt to parse 
     older 'X.DTA' naming (where 0.DTA becomes sequence 1, 1.DTA becomes sequence 2, etc.).

"""
import os
import logging
import threading
import tkinter as tk
from tkinter import filedialog, scrolledtext, ttk, messagebox
import queue 
import re 
import warnings
import deareis 
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm as MPL_LogNorm 
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use('Agg') 
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in log10")


# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------

def sanitize_filename(name_str, max_len=40):
    """Sanitizes a string to be used as part of a filename."""
    if not name_str: return ""
    sane_name = re.sub(r'[\\/*?:"<>|]', "", name_str)
    sane_name = re.sub(r'\s+', "_", sane_name)
    sane_name = re.sub(r'_+', "_", sane_name)
    return sane_name[:max_len].strip('_')

def extract_day_from_filename(filename):
    """
    Extracts sequence number. 
    Primary: '0 (X).DTA' -> X
    Fallback: 'X.DTA' -> X (if X=0, maps to 1; if X>0, maps to X)
    """
    basename = os.path.basename(filename)
    # New preferred format: "0 (X).DTA" or "File 0 (X).DTA" etc.
    match_new = re.search(r'\((\d+)\)\.dta$', basename, re.IGNORECASE)
    if match_new:
        return int(match_new.group(1))
    
    # Fallback to old format: "X.DTA"
    match_old = re.search(r'^(\d+)\.dta$', basename, re.IGNORECASE)
    if match_old:
        day_val_old = int(match_old.group(1))
        # Map old 0.DTA to sequence 1, 1.DTA to sequence 1 (if it's the first), 2.DTA to 2 etc.
        # The Day0 norm will look for sequence 1 as baseline.
        # If 0.DTA is present, it will be treated as baseline.
        # If only 1.DTA is present (as first file), it will be baseline.
        return day_val_old + 1 if day_val_old == 0 else day_val_old
        
    logging.warning(f"Could not extract day/sequence from filename: {filename}. Assigning -1 (will be skipped).")
    return -1 

def detect_num_replicates(input_dir, gui_replicates_fallback):
    """Counts subdirectories in input_dir. Falls back to GUI value."""
    try:
        subdirectories = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
        n_detected = len(subdirectories)
        if n_detected > 0:
            logging.info(f"Detected N={n_detected} replicate folders.")
            return n_detected
        else:
            logging.info(f"No replicate subfolders detected. Using N={gui_replicates_fallback} from GUI.")
            return gui_replicates_fallback
    except Exception as e:
        logging.warning(f"Error detecting replicates: {e}. Using N={gui_replicates_fallback} from GUI.")
        return gui_replicates_fallback

def apply_global_day0_normalization(df_input, N_replicates_deprecated, day_col_name='day', replicate_col_name='replicate_id', stop_event=None):
    """
    Applies Day0 normalization. Baseline is sequence point 1 (from '0 (1).DTA' or equivalent).
    """
    if df_input.empty: logging.warning("Day0 norm skipped: Input DF empty."); return df_input
    if stop_event and stop_event.is_set(): return df_input
    logging.info(f"Applying Day0 normalization (baseline is sequence point 1)...")
    
    df_with_rep_id = df_input.copy()
    if replicate_col_name not in df_with_rep_id.columns:
        logging.error(f"Day0 Norm Error: '{replicate_col_name}' column missing. Cannot perform per-replicate Day0 normalization."); return df_input

    all_norm_dfs = []
    unique_replicates = df_with_rep_id[replicate_col_name].unique()
    data_cols = [col for col in df_with_rep_id.columns if col not in [day_col_name, replicate_col_name]]

    for rep_id in unique_replicates:
        if stop_event and stop_event.is_set(): break
        rep_df = df_with_rep_id[df_with_rep_id[replicate_col_name] == rep_id].sort_values(by=day_col_name)
        if rep_df.empty: continue
        
        baseline_day_val_target = 1 # Sequence point 1 (from '0 (1).DTA' or mapped from '0.DTA')
        baseline_row_s = rep_df[rep_df[day_col_name] == baseline_day_val_target]
        
        actual_baseline_day_used = -1
        if baseline_row_s.empty: 
            min_day_in_rep = rep_df[day_col_name].min()
            if min_day_in_rep != -1 : # Check if any valid day was parsed
                baseline_row_s = rep_df[rep_df[day_col_name] == min_day_in_rep]
                actual_baseline_day_used = min_day_in_rep
                logging.warning(f"Day0 Norm: Replicate '{rep_id}' missing sequence '1'. Using its min valid day '{min_day_in_rep}' as baseline.")
            else:
                logging.warning(f"Day0 Norm: Replicate '{rep_id}' has no valid day numbers. Skipping Day0 norm for this replicate.")
                all_norm_dfs.append(rep_df) # Keep original data for this replicate
                continue
        else:
            actual_baseline_day_used = baseline_day_val_target
        
        if baseline_row_s.empty : # Still no baseline (e.g. all days were -1 or rep_df was empty after filter)
             logging.warning(f"No baseline could be determined for replicate '{rep_id}'. Skipping Day0 norm.")
             all_norm_dfs.append(rep_df)
             continue

        baseline_row = baseline_row_s.iloc[0]
        
        norm_rep_df_rows = []
        for _, row in rep_df.iterrows():
            if stop_event and stop_event.is_set(): break
            if row[day_col_name] == actual_baseline_day_used: continue 
            if row[day_col_name] < actual_baseline_day_used and row[day_col_name] != -1 : continue # Should not happen with proper sorting/filtering

            new_row = row.copy()
            for d_col in data_cols:
                try: new_row[d_col] = float(row[d_col]) - float(baseline_row[d_col])
                except (ValueError, TypeError): new_row[d_col] = np.nan 
            norm_rep_df_rows.append(new_row)
        
        if stop_event and stop_event.is_set(): break
        if norm_rep_df_rows: all_norm_dfs.append(pd.DataFrame(norm_rep_df_rows))
        elif not rep_df[rep_df[day_col_name] != actual_baseline_day_used].empty : # If only baseline day existed, no rows added.
            # This path means only baseline day existed, so after removing it, it's empty for this replicate.
            pass

    if stop_event and stop_event.is_set(): return df_input
    if not all_norm_dfs: 
        logging.warning("Day0 normalization resulted in no data (e.g., only baseline days existed across all replicates).")
        return pd.DataFrame(columns=df_input.columns) 
    
    result_df = pd.concat(all_norm_dfs, ignore_index=True)
    logging.info(f"Day0 normalization applied. Original rows: {len(df_input)}, Result rows: {len(result_df)}.")
    return result_df

def normalize_heatmap_data(data_for_heatmap, strategy, frequency_values=None, param_name_for_title="", stop_event=None):
    """Prepares data for heatmap plotting and returns data + optional norm object."""
    if stop_event and stop_event.is_set(): return data_for_heatmap, None
    if data_for_heatmap.empty: return data_for_heatmap, None
    logging.info(f"Applying heatmap data prep: '{strategy}' for {param_name_for_title or 'data'}")
    
    try: df_float = data_for_heatmap.astype(float)
    except Exception as e: logging.warning(f"Could not convert data to float for norm ({param_name_for_title}): {e}. Using Raw."); return data_for_heatmap.fillna(0.0), None

    norm_for_plotting = None 

    if strategy == "Logarithmic Color Scale":
        # Prepare data for LogNorm: must be positive. Add a small epsilon.
        # LogNorm object itself handles the scaling for the colormap.
        scaled_df = df_float.copy()
        if scaled_df.min().min() <= 0: # Check if any value is not positive
            scaled_df = scaled_df - scaled_df.min().min() + 1e-9 # Shift to be positive, add epsilon
        scaled_df[scaled_df <= 0] = 1e-9 # Ensure strictly positive after any shift
        norm_for_plotting = MPL_LogNorm(clip=True)
        logging.info(f"Prepared data for Logarithmic Color Scale. Min after prep: {scaled_df.min().min()}")
        return scaled_df, norm_for_plotting
    
    if strategy == "Raw Values": return df_float.fillna(0.0), None
    
    scaled_df = df_float.copy() 
    if strategy == "Per Parameter/Timeline (Column-wise)":
        for col in scaled_df.columns:
            if stop_event and stop_event.is_set(): return data_for_heatmap, None
            series = scaled_df[col]; min_v, max_v = series.min(), series.max()
            if pd.isna(min_v) or pd.isna(max_v) or (max_v == min_v): scaled_df[col] = 0.5 if not pd.isna(min_v) else np.nan
            else: # Avoid division by zero if max_v is very close to 0.99*min_v
                denominator = max_v - (0.99 * min_v)
                if np.isclose(denominator, 0): scaled_df[col] = 0.5
                else: scaled_df[col] = (series - (0.99 * min_v)) / denominator
        return scaled_df.fillna(0.0), None
    
    elif strategy == "Global Max Scaling":
        if scaled_df.empty or scaled_df.size == 0: return scaled_df.fillna(0.0), None
        global_min_val = scaled_df.min().min(); global_max_val = scaled_df.max().max()
        if pd.isna(global_min_val) or pd.isna(global_max_val) or (global_max_val == global_min_val): return scaled_df.fillna(0.5), None
        denominator = global_max_val - (0.99 * global_min_val)
        if np.isclose(denominator,0): return (scaled_df / global_max_val if not np.isclose(global_max_val,0) else scaled_df*0).fillna(0.0), None
        return ((scaled_df - (0.99 * global_min_val)) / denominator).fillna(0.0), None

    elif strategy == "Frequency Sections (L/M/H)":
        if frequency_values is None or not isinstance(frequency_values, (list, np.ndarray)) or len(frequency_values) == 0 or len(frequency_values) != len(scaled_df.columns):
            logging.warning("Freq values issue for 'Freq Sections'. Falling back to 'Per Timeline'.")
            return normalize_heatmap_data(scaled_df, "Per Parameter/Timeline (Column-wise)", None, param_name_for_title, stop_event)
        num_freqs = len(frequency_values)
        if num_freqs < 3: logging.warning("Too few freqs for 'Freq Sections'. Fallback."); return normalize_heatmap_data(scaled_df, "Per Parameter/Timeline (Column-wise)", None, param_name_for_title, stop_event)
        n_high = num_freqs // 3; n_mid = num_freqs // 3; n_low = num_freqs - n_high - n_mid
        sections_indices = []
        current_idx = 0
        if n_high > 0: sections_indices.append((current_idx, current_idx + n_high)); current_idx += n_high
        if n_mid > 0: sections_indices.append((current_idx, current_idx + n_mid)); current_idx += n_mid
        if n_low > 0: sections_indices.append((current_idx, current_idx + n_low))
        
        temp_scaled_df = scaled_df.copy() 
        for start_idx, end_idx in sections_indices:
            if stop_event and stop_event.is_set(): return data_for_heatmap, None
            section_cols = temp_scaled_df.columns[start_idx:end_idx]
            if section_cols.empty: continue
            section_data = temp_scaled_df[section_cols]
            if section_data.empty or section_data.size == 0: continue
            sec_min = section_data.min().min(); sec_max = section_data.max().max()
            if pd.isna(sec_min) or pd.isna(sec_max) or (sec_max == sec_min): 
                temp_scaled_df[section_cols] = 0.5 
                continue
            denominator = sec_max - (0.99 * sec_min)
            if np.isclose(denominator,0): temp_scaled_df[section_cols] = 0.5
            else: temp_scaled_df[section_cols] = (section_data - (0.99 * sec_min)) / denominator
        return temp_scaled_df.fillna(0.0), None

    logging.warning(f"Unknown heatmap norm strategy: '{strategy}'. Using Raw Values."); return df_float.fillna(0.0), None

def get_drt_enum_mapping(): # From user's GitHub
    """Returns DRT enum mappings."""
    return {
        "DRTMethod": {"MRQ Fit": deareis.DRTMethod.MRQ_FIT, "Tikhonov NNLS": deareis.DRTMethod.TR_NNLS, "Tikhonov RBF": deareis.DRTMethod.TR_RBF, "Bayesian Hilbert": deareis.DRTMethod.BHT},
        "DRTMode": {"Imaginary": deareis.DRTMode.IMAGINARY, "Real": deareis.DRTMode.REAL, "Complex": deareis.DRTMode.COMPLEX},
        "RBFType": {"Gaussian": deareis.RBFType.GAUSSIAN, "C0 Matern": deareis.RBFType.C0_MATERN, "C2 Matern": deareis.RBFType.C2_MATERN, "Cauchy": deareis.RBFType.CAUCHY, "Inverse Quadratic": deareis.RBFType.INVERSE_QUADRATIC},
        "RBFShape": {"FWHM": deareis.RBFShape.FWHM, "Factor": deareis.RBFShape.FACTOR}
    }

def format_plot_title(custom_prefix, auto_suffix, include_auto_suffix_bool): # From user's GitHub
    """Formats plot title based on custom prefix and auto suffix options."""
    if custom_prefix and include_auto_suffix_bool:
        return f"{custom_prefix} - {auto_suffix}"
    elif custom_prefix:
        return custom_prefix
    elif include_auto_suffix_bool: # Ensure auto_suffix is not None or empty if only this is true
        return auto_suffix if auto_suffix else "Plot"
    return "EIS Analysis Plot"

def get_spaced_ticks_and_labels(data_axis_values, custom_labels_str, max_ticks=15): # From user's GitHub, adapted
    """Helper for spacing out x or y tick labels for heatmaps."""
    custom_labels = [label.strip() for label in custom_labels_str.split(',') if label.strip()] if custom_labels_str else []
    data_axis_len = len(data_axis_values)

    if data_axis_len == 0: return np.array([]), []

    if not custom_labels: # Default: use a subset of actual data_axis_values
        if data_axis_len <= max_ticks:
            ticks_indices = np.arange(data_axis_len)
        else:
            ticks_indices = np.linspace(0, data_axis_len - 1, max_ticks, dtype=int)
        # Ensure values from data_axis_values are used for labels
        labels = [f"{data_axis_values[i]:.1f}" if isinstance(data_axis_values[i], float) else str(data_axis_values[i]) for i in ticks_indices]
        return ticks_indices, labels

    num_custom_labels = len(custom_labels)
    if num_custom_labels >= data_axis_len: 
        return np.arange(data_axis_len), custom_labels[:data_axis_len]
    else: 
        return np.linspace(0, data_axis_len -1 , num_custom_labels, dtype=int), custom_labels

# --- CSV Processing Helper ---
def structure_and_save_transposed_csv(df_day_vs_param, output_filepath, analysis_type_hint):
    """Transposes DataFrame, structures first columns, and saves to CSV."""
    if df_day_vs_param.empty:
        logging.warning(f"Cannot create transposed CSV for {output_filepath}, input DataFrame is empty.")
        return

    df_transposed = df_day_vs_param.T.copy() # Transpose
    df_transposed.index.name = 'Original_Param_Name' # Name the index before reset
    df_transposed = df_transposed.reset_index()
    
    new_cols_df = None
    first_cols_order = []

    if analysis_type_hint == "A1_Raw": # Zmod_100.0Hz or Zmod_1.00E-02Hz
        split_data = df_transposed['Original_Param_Name'].str.rsplit('_', n=1, expand=True)
        if split_data.shape[1] == 2:
            new_cols_df = pd.DataFrame({
                'Parameter_Type': split_data[0],
                'Frequency_Hz': split_data[1].str.replace('Hz', '', regex=False)
            })
            first_cols_order = ['Parameter_Type', 'Frequency_Hz']
    elif analysis_type_hint == "A2_DRT": # tau_1.000E-03
        split_data = df_transposed['Original_Param_Name'].str.rsplit('_', n=1, expand=True)
        if split_data.shape[1] == 2:
             new_cols_df = pd.DataFrame({
                'Metric': split_data[0].replace('tau', 'Gamma', regex=False), 
                'Tau_s': split_data[1]
            })
             first_cols_order = ['Metric', 'Tau_s']
    elif analysis_type_hint == "A3_ECM": # R0_R, L_L etc.
        split_data = df_transposed['Original_Param_Name'].str.rsplit('_', n=1, expand=True)
        if split_data.shape[1] == 2:
            new_cols_df = pd.DataFrame({
                'Component': split_data[0],
                'Parameter': split_data[1]
            })
            first_cols_order = ['Component', 'Parameter']
    elif analysis_type_hint == "A4_Peak": # Index is 'Freq Max' or 'Value'
        # For A4, df_day_vs_param's columns were 'Freq Max', 'Value'.
        # After transpose, 'Original_Param_Name' holds these.
        new_cols_df = pd.DataFrame({'Metric': df_transposed['Original_Param_Name']})
        first_cols_order = ['Metric']
    
    if new_cols_df is not None and not new_cols_df.empty:
        # Ensure no duplicate columns if Original_Param_Name was already one of the new column names
        existing_cols_to_drop = [col for col in first_cols_order if col in df_transposed.columns]
        if existing_cols_to_drop:
            df_transposed = df_transposed.drop(columns=existing_cols_to_drop)

        df_structured = pd.concat([new_cols_df, df_transposed.drop(columns=['Original_Param_Name'])], axis=1)
        # Reorder columns: new structured columns first, then the day columns
        day_columns = [col for col in df_structured.columns if col not in first_cols_order]
        df_structured = df_structured[first_cols_order + day_columns]
    else: 
        df_structured = df_transposed # Fallback if parsing failed, save basic transpose
        logging.warning(f"Could not parse param names for structured CSV: {output_filepath}. Saving basic transpose.")

    try:
        df_structured.to_csv(output_filepath, index=False)
        logging.info(f"Saved structured transposed CSV to: {output_filepath}")
    except Exception as e:
        logging.error(f"Error saving structured transposed CSV {output_filepath}: {e}")

# ---------------------------------------------------------
# ANALYSIS FUNCTIONS 
# ---------------------------------------------------------

def run_analysis_1(input_dir, output_dir, N_replicates_from_gui, 
                   apply_day0_norm_global, average_replicates_global, heatmap_norm_strategy,
                   custom_title_prefix, include_auto_titles, custom_x_labels_str, custom_x_axis_title, 
                   selected_palette, 
                   a1_norm_stddev_heatmap_global, 
                   stop_event):
    logging.info("Starting Analysis Type 1: Raw Variables Heatmaps...")
    if stop_event.is_set(): return "Analysis Cancelled at Start (A1)"
    N_replicates = detect_num_replicates(input_dir, N_replicates_from_gui)
    if N_replicates == 0: N_replicates = 1 # Default to 1 if no subfolders and fallback is 0

    all_files_data_list = []
    processed_one_file_for_cols = False
    dynamic_param_column_names = [] 
    common_freqs_hz_a1 = np.array([]) 

    replicate_folders = sorted([d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))])
    if not replicate_folders: replicate_folders = [None] # Process input_dir itself

    for rep_idx, replicate_folder_name in enumerate(replicate_folders):
        if stop_event.is_set(): return "Analysis Cancelled (A1 Rep Loop)"
        current_path = os.path.join(input_dir, replicate_folder_name) if replicate_folder_name else input_dir
        current_replicate_id = replicate_folder_name if replicate_folder_name else f"rep{rep_idx}"
        try:
            dta_files_unsorted = [f for f in os.listdir(current_path) if f.lower().endswith('.dta') and os.path.isfile(os.path.join(current_path, f))]
            dta_files = sorted(dta_files_unsorted, key=lambda f_name: extract_day_from_filename(f_name)) # Sort by extracted day
        except FileNotFoundError: logging.warning(f"A1: Path not found: {current_path}."); continue

        for filename in dta_files:
            if stop_event.is_set(): return "Analysis Cancelled (A1 File Loop)"
            filepath = os.path.join(current_path, filename)
            day_val = extract_day_from_filename(filename)
            if day_val == -1 : logging.warning(f"A1: Skipping {filename}, invalid name for day extraction."); continue
            
            logging.debug(f"A1: Processing {filepath} for day/seq: {day_val}")
            try:
                data_eis_list = deareis.parse_data(filepath)
                if not data_eis_list: logging.warning(f"A1: Could not parse {filepath}"); continue
                eis_obj = data_eis_list[0]
                bode = eis_obj.get_bode_data(); nyq = eis_obj.get_nyquist_data()
                if bode is None or nyq is None or not bode[0].size: logging.warning(f"A1: No Bode/Nyq in {filepath}"); continue

                file_data_row = {'day': day_val, 'replicate_id': current_replicate_id}
                freqs_hz, Zmod_vals, negZphz_vals = bode[0], bode[1], bode[2]
                Zreal_vals, Zimag_vals_from_nyq = nyq[0], nyq[1] # nyq[1] is -Zimag
                
                # We need Zimag (positive imaginary part for positive frequencies)
                # If nyq[1] is -Zimag (typical), then Zimag = -nyq[1]
                Zimag_vals = -Zimag_vals_from_nyq 

                if not processed_one_file_for_cols or common_freqs_hz_a1.size == 0 : 
                    common_freqs_hz_a1 = freqs_hz.copy() # Capture common frequencies from first valid file

                Creal_calc, Cimg_calc, Cmod_calc = [], [], []
                for i, f_hz in enumerate(freqs_hz):
                    Cr_val, Ci_val, Cm_val = np.nan, np.nan, np.nan
                    if f_hz != 0 and Zmod_vals[i] !=0:
                        # Updated Creal: Creal = -Zimag / (2*pi*f*Zmod^2)
                        Cr_val = -Zimag_vals[i] / (2 * np.pi * f_hz * Zmod_vals[i]**2)
                        # Cimag: Cimag = Zreal / (2*pi*f*Zmod^2)
                        Ci_val = Zreal_vals[i] / (2 * np.pi * f_hz * Zmod_vals[i]**2)
                        if not (np.isnan(Cr_val) or np.isnan(Ci_val)): Cm_val = np.sqrt(Cr_val**2 + Ci_val**2)
                    Creal_calc.append(Cr_val); Cimg_calc.append(Ci_val); Cmod_calc.append(Cm_val)
                
                all_param_series = {'Zmod': Zmod_vals, 'negZphz': negZphz_vals, 'Zreal': Zreal_vals, 
                                    'Zimag': Zimag_vals, 
                                    'Creal': np.array(Creal_calc), 'Cimg': np.array(Cimg_calc), 
                                    'Cmod': np.array(Cmod_calc)}

                if not processed_one_file_for_cols and common_freqs_hz_a1.size > 0:
                    dynamic_param_column_names = []
                    for param_name_iter in ['Zmod', '-Zphz', 'Zreal', 'Zimag', 'Creal', 'Cimg', 'Cmod']:
                        for freq_val_iter in common_freqs_hz_a1:
                            freq_str_iter = f"{freq_val_iter:.2E}" if freq_val_iter > 1000 or freq_val_iter < 0.01 else f"{freq_val_iter:.3f}"
                            dynamic_param_column_names.append(f"{param_name_iter}_{freq_str_iter}Hz")
                    processed_one_file_for_cols = True
                
                for param_key_ordered, param_values_current_file in all_param_series.items():
                    for freq_idx_common, freq_val_common in enumerate(common_freqs_hz_a1):
                        freq_str_common_fmt = f"{freq_val_common:.2E}" if freq_val_common > 1000 or freq_val_common < 0.01 else f"{freq_val_common:.3f}"
                        col_name_expected = f"{param_key_ordered}_{freq_str_common_fmt}Hz"
                        
                        current_file_freq_idx_match = np.where(np.isclose(freqs_hz, freq_val_common))[0]
                        if current_file_freq_idx_match.size > 0:
                            idx_in_file_data = current_file_freq_idx_match[0]
                            if idx_in_file_data < len(param_values_current_file):
                                file_data_row[col_name_expected] = param_values_current_file[idx_in_file_data]
                            else: file_data_row[col_name_expected] = np.nan
                        else: file_data_row[col_name_expected] = np.nan 
                all_files_data_list.append(file_data_row)
            except Exception as e: logging.error(f"A1 file processing error for {filepath}: {e}", exc_info=True)

    if not all_files_data_list: return "Error (A1): No data files could be processed or all had invalid names."
    
    # Create master DataFrame
    master_cols = ['day', 'replicate_id'] + (dynamic_param_column_names if processed_one_file_for_cols else [])
    raw_data_df = pd.DataFrame(all_files_data_list)
    for col in master_cols: 
        if col not in raw_data_df.columns: raw_data_df[col] = np.nan # Ensure all expected columns exist
    raw_data_df = raw_data_df[master_cols] # Enforce column order

    processed_df_for_means = raw_data_df.copy() 
    processed_df_for_std = raw_data_df.copy()   

    if apply_day0_norm_global:
        logging.info("A1: Applying global Day0 normalization...")
        processed_df_for_means = apply_global_day0_normalization(processed_df_for_means, N_replicates, 'day', 'replicate_id', stop_event)
        # Note: Applying Day0 to data for StdDev calc might not always be desired, depends on interpretation
        processed_df_for_std = apply_global_day0_normalization(processed_df_for_std, N_replicates, 'day', 'replicate_id', stop_event) 
        if stop_event.is_set() or processed_df_for_means.empty: return "Cancelled or Empty after Day0 (A1)"

    data_cols_for_agg = [col for col in raw_data_df.columns if col not in ['day', 'replicate_id']]
    dfs_for_main_heatmaps = {} 
    
    if average_replicates_global:
        if N_replicates > 0 and not processed_df_for_means.empty and data_cols_for_agg:
            # Average across replicates for each day
            # Ensure 'day' is numeric for groupby, and handle cases where day might be non-unique after processing
            processed_df_for_means['day'] = pd.to_numeric(processed_df_for_means['day'], errors='coerce')
            avg_df = processed_df_for_means.groupby('day')[data_cols_for_agg].mean()
            dfs_for_main_heatmaps["averaged"] = avg_df.sort_index() # Sort by day
            logging.info("Applied global replicate averaging for A1.")
        else: 
            dfs_for_main_heatmaps["averaged"] = processed_df_for_means.set_index('day')[data_cols_for_agg] if 'day' in processed_df_for_means and data_cols_for_agg else pd.DataFrame()
    else: 
        if 'replicate_id' in processed_df_for_means.columns:
            for rep_id, group_df in processed_df_for_means.groupby('replicate_id'):
                if stop_event.is_set(): return "Cancelled during A1 per-replicate split"
                if not group_df.empty and 'day' in group_df.columns and data_cols_for_agg:
                    group_df['day'] = pd.to_numeric(group_df['day'], errors='coerce')
                    # If a replicate has multiple entries for the same day (should not happen with file naming), average them.
                    unique_day_group = group_df.groupby('day')[data_cols_for_agg].mean().sort_index()
                    dfs_for_main_heatmaps[rep_id] = unique_day_group
    
    if stop_event.is_set(): return "Analysis Cancelled (A1)"
    
    # --- Standard Deviation Heatmap ---
    # (Using processed_df_for_std, which might have Day0 norm applied if global option was true)
    if N_replicates > 1 and 'replicate_id' in processed_df_for_std.columns and not processed_df_for_std.empty and data_cols_for_agg:
        # Calculate StdDev across replicates for each day and each parameter_freq
        std_dev_by_day_param = processed_df_for_std.groupby('day')[data_cols_for_agg].std()
        
        if not std_dev_by_day_param.empty:
            # Create heatmap: Avg StdDev over days, for Freq vs. ParamType
            std_dev_records = []
            for col_name_std in data_cols_for_agg: 
                try:
                    param_t_std, freq_s_hz_std = col_name_std.rsplit('_',1)
                    mean_std_for_param_freq = std_dev_by_day_param[col_name_std].mean() 
                    std_dev_records.append({'param_type': param_t_std, 'freq_str': freq_s_hz_std.replace('Hz',''), 'mean_std_dev': mean_std_for_param_freq})
                except ValueError: logging.warning(f"A1 StdDev: Could not split param_freq: {col_name_std}"); continue
                
            if std_dev_records:
                std_pivot_df = pd.DataFrame(std_dev_records).pivot_table(index='freq_str', columns='param_type', values='mean_std_dev')
                try: # Sort frequency index numerically
                    std_pivot_df.index = pd.to_numeric(std_pivot_df.index)
                    std_pivot_df = std_pivot_df.sort_index(ascending=False)
                    std_pivot_df.index = [f"{idx:.2E}" if idx > 1000 or idx < 0.01 else f"{idx:.3f}" for idx in std_pivot_df.index]
                except: logging.warning("A1: Could not sort StDev heatmap freqs numerically.")
                
                std_dev_heatmap_to_plot = std_pivot_df
                norm_obj_stddev = None
                if a1_norm_stddev_heatmap_global: # Apply chosen global strategy or LogNorm
                    std_dev_heatmap_to_plot, norm_obj_stddev = normalize_heatmap_data(std_pivot_df, heatmap_norm_strategy, 
                                                                   param_name_for_title="A1_StdDevs", stop_event=stop_event)

                if not std_dev_heatmap_to_plot.empty and (not stop_event or not stop_event.is_set()):
                    plt.figure(figsize=(10, max(6, len(std_dev_heatmap_to_plot.index)*0.3)))
                    title_std_auto = f"Avg Replicate StdDev (Norm: {heatmap_norm_strategy if a1_norm_stddev_heatmap_global else 'Raw Values'}) - A1"
                    title_std = format_plot_title(custom_title_prefix, title_std_auto, include_auto_titles)
                    
                    sns.heatmap(std_dev_heatmap_to_plot.fillna(0), cmap=selected_palette, fmt=".2E", annot=False, norm=norm_obj_stddev)
                    plt.title(title_std); plt.xlabel("Parameter Type"); plt.ylabel("Frequency (Hz)"); plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0); plt.tight_layout()
                    fname_std_suffix = sanitize_filename(custom_title_prefix)
                    plt.savefig(os.path.join(output_dir, f"A1_avg_replicate_std_dev_heatmap{'_' if fname_std_suffix else ''}{fname_std_suffix}.png")); plt.close()
                    logging.info("Saved A1 Avg Replicate StdDev heatmap.")
    if stop_event.is_set(): return "Analysis Cancelled (A1)"

    # --- Generate Main Heatmaps ---
    if not dfs_for_main_heatmaps: logging.warning("A1: No data available for main heatmaps."); return "A1 Completed (No main heatmap data)."

    unique_param_types_a1 = ['Zmod', '-Zphz', 'Zreal', 'Zimag', 'Creal', 'Cimg', 'Cmod']
    # Freq labels for Y-axis of heatmap, ensure they match columns of processed_data_for_plot.T
    # This will be derived from the actual columns of heatmap_actual_data later.

    for replicate_key, data_for_heatmap_day_vs_param_freq in dfs_for_main_heatmaps.items():
        if stop_event.is_set(): return f"Cancelled before heatmaps for {replicate_key} (A1)"
        if data_for_heatmap_day_vs_param_freq.empty: continue
        
        title_prefix_for_file = sanitize_filename(custom_title_prefix)
        csv_filename_base = f"A1_main_data{'_' + title_prefix_for_file if title_prefix_for_file else ''}"
        csv_filename_base += f"_rep_{sanitize_filename(replicate_key)}" if replicate_key != "averaged" else "_averaged"
        
        # Save structured transposed CSV
        structure_and_save_transposed_csv(data_for_heatmap_day_vs_param_freq, 
                                          os.path.join(output_dir, f"{csv_filename_base}_transposed.csv"),
                                          analysis_type_hint="A1_Raw")

        for param_type in unique_param_types_a1:
            if stop_event.is_set(): return f"Cancelled plotting {param_type} for {replicate_key} (A1)"
            
            # Select columns for the current parameter type (e.g., all Zmod_FreqHz columns)
            cols_for_this_param_type = [col for col in data_for_heatmap_day_vs_param_freq.columns if col.startswith(param_type + "_")]
            if not cols_for_this_param_type: continue
            
            heatmap_actual_data = data_for_heatmap_day_vs_param_freq[cols_for_this_param_type]
            
            # Extract frequencies for this param_type to pass to normalize_heatmap_data and for y-labels
            # And sort columns by frequency for consistent y-axis on heatmap
            current_param_freq_tuples = [] # (float_freq, col_name)
            for col_name in cols_for_this_param_type:
                try:
                    freq_part = col_name.split('_')[-1].replace('Hz','')
                    current_param_freq_tuples.append((float(freq_part), col_name))
                except ValueError: # Handle scientific notation if not caught by float directly
                    try: current_param_freq_tuples.append((float(re.sub(r'[^\d.Ee+-]', '', freq_part)), col_name))
                    except: logging.warning(f"A1: Could not parse freq from {col_name} for sorting/labeling."); continue
            
            current_param_freq_tuples.sort(key=lambda x: x[0], reverse=True) # Sort high to low freq for heatmap top to bottom
            sorted_cols_for_param = [x[1] for x in current_param_freq_tuples]
            current_param_freq_floats_for_norm = [x[0] for x in current_param_freq_tuples] # Freqs for norm function
            
            if not sorted_cols_for_param: continue
            heatmap_actual_data_sorted = heatmap_actual_data[sorted_cols_for_param]
            
            # Data prep for heatmap (includes LogNorm handling)
            processed_data_for_plot, norm_object_for_plot = normalize_heatmap_data(
                heatmap_actual_data_sorted, heatmap_norm_strategy, 
                frequency_values=current_param_freq_floats_for_norm, 
                param_name_for_title=f"{param_type} ({replicate_key})", stop_event=stop_event
            )
            if stop_event.is_set() or processed_data_for_plot.empty: continue

            plt.figure(figsize=(10, max(6, len(current_param_freq_floats_for_norm)*0.25)))
            auto_title_suffix = f'{param_type} ({replicate_key}, Norm: {heatmap_norm_strategy}) - A1'
            plot_title = format_plot_title(custom_title_prefix, auto_title_suffix, include_auto_titles)
            xlabel_main = custom_x_axis_title if custom_x_axis_title else "Sequence Point"
            
            x_ticks_indices, x_tick_labels = get_spaced_ticks_and_labels(processed_data_for_plot.index.values, custom_x_labels_str)
            
            # Y-axis (freqs) labels from sorted frequencies
            y_labels_plot = [f"{f:.2E}" if f > 1000 or f < 0.01 else f"{f:.3f}" for f in current_param_freq_floats_for_norm]

            ax = sns.heatmap(processed_data_for_plot.T, cmap=selected_palette, fmt=".2f", 
                             yticklabels=y_labels_plot, xticklabels=False, norm=norm_object_for_plot)
            if x_ticks_indices.size > 0:
                ax.set_xticks(x_ticks_indices + 0.5) 
                ax.set_xticklabels(x_tick_labels, rotation=45, ha="right")
            
            plt.title(plot_title); plt.xlabel(xlabel_main); plt.ylabel('Frequency (Hz)'); plt.yticks(rotation=0); plt.tight_layout()
            # Use the consistent base name for heatmap files
            hm_filename = f"A1_heatmap_{param_type.replace('/','_')}_{csv_filename_base.replace('A1_main_data_', '')}.png"
            plt.savefig(os.path.join(output_dir, hm_filename)); plt.close()
            logging.info(f"Saved A1 heatmap: {hm_filename}")

    return "Analysis 1 (Raw Vars Heatmaps) completed."

# ---------------------------------------------------------
# ANALYSIS FUNCTION 2: DRT 
# ---------------------------------------------------------
def run_analysis_drt(input_dir, output_dir, N_replicates_from_gui,
                     apply_day0_norm_global, average_replicates_global, heatmap_norm_strategy,
                     custom_title_prefix, include_auto_titles, custom_x_labels_str, custom_x_axis_title,
                     selected_palette, # New
                     drt_method_enum, drt_mode_enum, lambda_values_list,
                     rbf_type_enum, rbf_shape_enum, rbf_size_float,
                     fit_penalty_int, include_inductance_bool, num_attempts_int,
                     mrq_fit_cdc_string, num_drt_procs, stop_event):
    logging.info("Starting Analysis Type 2 (DRT Analysis)...")
    if stop_event.is_set(): return "DRT Cancelled at Start"
    N_replicates = detect_num_replicates(input_dir, N_replicates_from_gui)
    if N_replicates == 0: N_replicates = 1
    if not os.path.isdir(input_dir): return "Error (A2): Input directory not found."
    if not os.path.isdir(output_dir): os.makedirs(output_dir, exist_ok=True)

    first_file_processed_for_taus_overall = False
    common_tau_values_overall = np.array([])

    replicate_folders = sorted([d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))])
    if not replicate_folders: replicate_folders = [None] # Process input_dir itself if no subfolders

    for lambda_val_idx, lambda_val in enumerate(lambda_values_list):
        if stop_event.is_set(): return f"DRT Cancelled before lambda {lambda_val}"
        logging.info(f"Processing DRT for Lambda ({lambda_val_idx+1}/{len(lambda_values_list)}): {lambda_val}")

        all_files_drt_data_current_lambda = []
        processed_a_file_this_lambda = False # To check if any data was processed for this lambda

        for rep_idx, replicate_folder_name in enumerate(replicate_folders):
            if stop_event.is_set(): break # from replicate loop
            current_path = os.path.join(input_dir, replicate_folder_name) if replicate_folder_name else input_dir
            current_replicate_id = replicate_folder_name if replicate_folder_name else f"rep{rep_idx}"
            try:
                dta_files_unsorted = [f for f in os.listdir(current_path) if f.lower().endswith('.dta') and os.path.isfile(os.path.join(current_path, f))]
                dta_files = sorted(dta_files_unsorted, key=lambda f_name: extract_day_from_filename(f_name))
            except FileNotFoundError: logging.warning(f"A2 DRT: Path not found: {current_path}. Skipping."); continue

            for filename in dta_files:
                if stop_event.is_set(): break # from file loop
                filepath = os.path.join(current_path, filename)
                day_val = extract_day_from_filename(filename)
                if day_val == -1: logging.warning(f"A2 DRT: Skipping {filename}, invalid name for day extraction."); continue

                logging.debug(f"A2 DRT: Processing {filepath} for day/seq: {day_val}, Lambda: {lambda_val}")
                try:
                    eis_data_list = deareis.parse_data(filepath)
                    if not eis_data_list: logging.warning(f"A2 DRT: Could not parse {filepath}"); continue

                    prelim_fit_settings = deareis.FitSettings(mrq_fit_cdc_string or "R(QR)", method='AUTO', weight='AUTO', max_nfev=10000)
                    prelim_ecm_fit_obj = deareis.fit_circuit(eis_data_list[0], prelim_fit_settings)

                    drt_settings_dict = {
                        'method': drt_method_enum, 'mode': drt_mode_enum, 'lambda_value': float(lambda_val),
                        'rbf_type': rbf_type_enum, 'derivative_order': fit_penalty_int, 'rbf_shape': rbf_shape_enum,
                        'shape_coeff': rbf_size_float, 'inductance': include_inductance_bool, 'num_attempts': num_attempts_int,
                        'credible_intervals': False, 'timeout': 120, 'num_samples': 100, 'maximum_symmetry': 0.5,
                        'fit': prelim_ecm_fit_obj, 'gaussian_width': 0.5, 'num_per_decade': 10
                    }
                    drt_current_settings = deareis.DRTSettings(**drt_settings_dict)
                    drt_results = deareis.calculate_drt(eis_data_list[0], drt_current_settings, num_procs=num_drt_procs)

                    if not drt_results: logging.warning(f"A2 DRT: DRT calculation failed for {filepath}"); continue

                    gammas = drt_results.get_gammas()[0]; taus = drt_results.get_time_constants()
                    if not first_file_processed_for_taus_overall:
                        common_tau_values_overall = np.round(taus, 6); first_file_processed_for_taus_overall = True
                    if len(gammas) != len(common_tau_values_overall) and first_file_processed_for_taus_overall:
                        logging.warning(f"A2 DRT: Tau/Gamma length mismatch for {filepath}. Expected {len(common_tau_values_overall)}, got {len(gammas)}. Skipping."); continue
                    
                    all_files_drt_data_current_lambda.append({'day': day_val, 'replicate_id': current_replicate_id, 'gammas': gammas})
                    processed_a_file_this_lambda = True
                except Exception as e_calc: logging.error(f"A2 DRT: Error during DRT calculation for {filepath} (Lambda {lambda_val}): {e_calc}", exc_info=True)
            if stop_event.is_set(): break # from file loop
        if stop_event.is_set(): break # from replicate loop
        
        if stop_event.is_set() or not processed_a_file_this_lambda or not first_file_processed_for_taus_overall:
            logging.warning(f"A2 DRT: No DRT data successfully processed or common taus not established for lambda {lambda_val}. Skipping this lambda."); continue

        tau_col_names = [f"tau_{tau_val:.3E}" for tau_val in common_tau_values_overall]
        gamma_records = []
        for r_dict in all_files_drt_data_current_lambda:
            if len(r_dict['gammas']) == len(common_tau_values_overall): # Ensure consistency before creating record
                 gamma_records.append({'day': r_dict['day'], 'replicate_id': r_dict['replicate_id'], 
                                  **{tau_col_names[i]: r_dict['gammas'][i] for i in range(len(common_tau_values_overall))}})
            else:
                logging.warning(f"A2 DRT: Dropping record for day {r_dict['day']}, rep {r_dict['replicate_id']} due to gamma/tau mismatch post-processing.")


        if not gamma_records: logging.warning(f"A2 DRT: No valid gamma records to form DataFrame for lambda {lambda_val}."); continue
        
        raw_drt_df_lambda = pd.DataFrame(gamma_records)
        processed_df_lambda = raw_drt_df_lambda.copy()

        if apply_day0_norm_global:
            processed_df_lambda = apply_global_day0_normalization(processed_df_lambda, N_replicates, 'day', 'replicate_id', stop_event)
            if stop_event.is_set() or processed_df_lambda.empty:
                logging.warning(f"A2 DRT: Data empty or cancelled after Day0 norm for lambda {lambda_val}."); continue
        
        data_cols_drt = [col for col in processed_df_lambda.columns if col not in ['day', 'replicate_id']]
        dfs_for_drt_heatmaps_this_lambda = {}

        if average_replicates_global:
            if N_replicates > 0 and not processed_df_lambda.empty and data_cols_drt:
                processed_df_lambda['day'] = pd.to_numeric(processed_df_lambda['day'], errors='coerce')
                avg_df_drt = processed_df_lambda.groupby('day')[data_cols_drt].mean().sort_index()
                dfs_for_drt_heatmaps_this_lambda["averaged"] = avg_df_drt
            else: 
                dfs_for_drt_heatmaps_this_lambda["averaged"] = processed_df_lambda.set_index('day')[data_cols_drt] if 'day' in processed_df_lambda and data_cols_drt else pd.DataFrame()
        else: 
             if 'replicate_id' in processed_df_lambda.columns:
                for rep_id_drt, group_df_drt in processed_df_lambda.groupby('replicate_id'):
                    if stop_event.is_set(): break
                    if not group_df_drt.empty and 'day' in group_df_drt.columns and data_cols_drt:
                        group_df_drt['day'] = pd.to_numeric(group_df_drt['day'], errors='coerce')
                        unique_day_group_drt = group_df_drt.groupby('day')[data_cols_drt].mean().sort_index()
                        dfs_for_drt_heatmaps_this_lambda[rep_id_drt] = unique_day_group_drt
        if stop_event.is_set(): continue # To next lambda if cancelled

        for replicate_key_drt, data_for_heatmap_drt in dfs_for_drt_heatmaps_this_lambda.items():
            if stop_event.is_set() or data_for_heatmap_drt.empty: continue
            
            title_prefix_for_file = sanitize_filename(custom_title_prefix)
            csv_sfx_drt = f"_L{str(lambda_val).replace('.','_')}"
            csv_sfx_drt += f"_{title_prefix_for_file}" if title_prefix_for_file else ""
            csv_sfx_drt += f"_rep_{sanitize_filename(replicate_key_drt)}" if replicate_key_drt != "averaged" else "_averaged"
            
            structure_and_save_transposed_csv(data_for_heatmap_drt,
                                              os.path.join(output_dir, f"A2_DRT_data{csv_sfx_drt}_transposed.csv"),
                                              analysis_type_hint="A2_DRT")

            processed_drt_plot, norm_obj_drt = normalize_heatmap_data(
                data_for_heatmap_drt, heatmap_norm_strategy, 
                frequency_values=common_tau_values_overall, # Pass taus as 'frequency_values' if strategy uses them
                param_name_for_title=f"DRT (L {lambda_val}, {replicate_key_drt})", stop_event=stop_event
            )
            if stop_event.is_set() or processed_drt_plot.empty: continue
            
            # Y-axis (taus) label capping
            num_taus_total = len(common_tau_values_overall)
            max_y_ticks_drt = 20 
            y_tick_indices = np.linspace(0, num_taus_total - 1, min(num_taus_total, max_y_ticks_drt), dtype=int) if num_taus_total > 0 else np.array([])
            ytick_labels_drt_subset = [f"{common_tau_values_overall[i]:.1E}" for i in y_tick_indices] if y_tick_indices.size > 0 else False

            plt.figure(figsize=(10, 8))
            drt_method_name = drt_method_enum.name if hasattr(drt_method_enum, "name") else str(drt_method_enum)
            drt_mode_name = drt_mode_enum.name if hasattr(drt_mode_enum, "name") else str(drt_mode_enum)
            auto_title_sfx_drt = f'DRT (L {lambda_val}, {replicate_key_drt}, {drt_method_name}, {drt_mode_name}, N: {heatmap_norm_strategy})'
            plot_title_drt = format_plot_title(custom_title_prefix, auto_title_sfx_drt, include_auto_titles)
            xlabel_drt = custom_x_axis_title if custom_x_axis_title else "Sequence Point"
            
            x_ticks_indices_drt, x_tick_labels_drt = get_spaced_ticks_and_labels(processed_drt_plot.index.values, custom_x_labels_str)

            ax = sns.heatmap(processed_drt_plot.T, cmap=selected_palette, fmt=".2f", 
                             xticklabels=False, yticklabels=False, norm=norm_obj_drt)
            if ytick_labels_drt_subset: 
                ax.set_yticks(y_tick_indices + 0.5) # Center ticks on cells
                ax.set_yticklabels(ytick_labels_drt_subset, rotation=0)
            if x_ticks_indices_drt.size > 0:
                ax.set_xticks(x_ticks_indices_drt + 0.5)
                ax.set_xticklabels(x_tick_labels_drt, rotation=45, ha="right")
            
            plt.title(plot_title_drt); plt.xlabel(xlabel_drt); plt.ylabel('τ (s)'); plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"A2_drt_heatmap{csv_sfx_drt}.png")); plt.close()
            logging.info(f"Saved DRT heatmap for Lambda {lambda_val}, {replicate_key_drt}")
            if stop_event.is_set(): break # from replicate_key loop
        if stop_event.is_set(): break # from lambda_val loop

    if stop_event.is_set(): return "DRT Analysis Cancelled."
    return "Analysis Type 2 (DRT Analysis) completed."


# ---------------------------------------------------------
# ANALYSIS FUNCTION 3: ECM Fitting 
# ---------------------------------------------------------
def run_analysis_ecm_fitting(input_dir, output_dir, N_replicates_from_gui, 
                             apply_day0_norm_global, average_replicates_global, heatmap_norm_strategy,
                             custom_title_prefix, include_auto_titles, custom_x_labels_str, custom_x_axis_title,
                             selected_palette,
                             ecm_strings_list, fit_method_str, fit_weight_str, fit_max_nfev_int, stop_event):
    logging.info("Starting Analysis Type 3 (ECM Fitting)...")
    if stop_event.is_set(): return "ECM Fitting Cancelled at Start"
    N_replicates = detect_num_replicates(input_dir, N_replicates_from_gui)
    if N_replicates == 0: N_replicates = 1
    if not os.path.isdir(input_dir): return "Error (A3): Input directory not found."
    if not os.path.isdir(output_dir): os.makedirs(output_dir, exist_ok=True)

    replicate_folders = sorted([d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))])
    if not replicate_folders: replicate_folders = [None]

    for ecm_idx, current_ecm_str in enumerate(ecm_strings_list):
        if stop_event.is_set(): return f"ECM Fitting Cancelled before ECM {current_ecm_str}"
        logging.info(f"Processing ECM ({ecm_idx+1}/{len(ecm_strings_list)}): {current_ecm_str}")
        
        all_files_ecm_data = [] 
        parameter_names_ordered = [] 
        first_successful_fit_for_ecm = False
        
        try:
            fit_settings = deareis.FitSettings(current_ecm_str, method=fit_method_str, weight=fit_weight_str, max_nfev=fit_max_nfev_int)
        except Exception as e_fset:
            logging.error(f"A3 ECM: Error creating FitSettings for '{current_ecm_str}': {e_fset}. Skipping this ECM."); continue

        for rep_idx, replicate_folder_name in enumerate(replicate_folders):
            if stop_event.is_set(): break
            current_path = os.path.join(input_dir, replicate_folder_name) if replicate_folder_name else input_dir
            current_replicate_id = replicate_folder_name if replicate_folder_name else f"rep{rep_idx}"
            try:
                dta_files_unsorted = [f for f in os.listdir(current_path) if f.lower().endswith('.dta') and os.path.isfile(os.path.join(current_path, f))]
                dta_files = sorted(dta_files_unsorted, key=lambda f_name: extract_day_from_filename(f_name))
            except FileNotFoundError: logging.warning(f"A3 ECM: Path not found: {current_path}"); continue

            for filename in dta_files:
                if stop_event.is_set(): break
                filepath = os.path.join(current_path, filename)
                day_val = extract_day_from_filename(filename)
                if day_val == -1: logging.warning(f"A3 ECM: Skipping {filename}, invalid name."); continue
                
                logging.debug(f"A3 ECM: Fitting {current_ecm_str} to {filepath} for day/seq: {day_val}")
                try:
                    eis_data_list = deareis.parse_data(filepath)
                    if not eis_data_list: logging.warning(f"A3 ECM: Could not parse {filepath}"); continue
                    
                    fit_results = deareis.fit_circuit(eis_data_list[0], fit_settings)
                    current_params_dict_this_file = {}
                    temp_param_names_this_fit = []

                    if fit_results and fit_results.parameters:
                        for comp_name_fit in sorted(fit_results.parameters.keys()): # Ensure consistent order
                            for param_name_fit in sorted(fit_results.parameters[comp_name_fit].keys()):
                                full_param_key = f"{comp_name_fit}_{param_name_fit}"
                                current_params_dict_this_file[full_param_key] = fit_results.parameters[comp_name_fit][param_name_fit].value
                                if not first_successful_fit_for_ecm: 
                                    temp_param_names_this_fit.append(full_param_key)
                        
                        if not first_successful_fit_for_ecm and temp_param_names_this_fit: 
                            parameter_names_ordered = temp_param_names_this_fit
                            first_successful_fit_for_ecm = True
                        
                        params_for_df_row = {p_name: current_params_dict_this_file.get(p_name, np.nan) for p_name in parameter_names_ordered} if first_successful_fit_for_ecm else current_params_dict_this_file
                        all_files_ecm_data.append({'day': day_val, 'replicate_id': current_replicate_id, 'params': params_for_df_row})
                    
                    elif first_successful_fit_for_ecm : # Fit failed but we know param names
                        all_files_ecm_data.append({'day': day_val, 'replicate_id': current_replicate_id, 'params': {p_name: np.nan for p_name in parameter_names_ordered}})
                    else: 
                         logging.warning(f"A3 ECM: Fit failed for {filepath} (ECM {current_ecm_str}) and no parameters established yet for this ECM.")
                except Exception as e_fit_single:
                    logging.error(f"A3 ECM: Error fitting {current_ecm_str} to {filepath}: {e_fit_single}", exc_info=False)
                    if first_successful_fit_for_ecm:
                        all_files_ecm_data.append({'day': day_val, 'replicate_id': current_replicate_id if 'current_replicate_id' in locals() else f"unknown_rep", 'params': {p_name: np.nan for p_name in parameter_names_ordered}})
            if stop_event.is_set(): break 
        if stop_event.is_set(): break
        
        if stop_event.is_set() or not all_files_ecm_data or not first_successful_fit_for_ecm:
            logging.warning(f"A3 ECM: No data or parameters determined for ECM {current_ecm_str}. Skipping this ECM."); continue
        
        records_for_df = []
        for record in all_files_ecm_data:
            row_data = {'day':record['day'], 'replicate_id':record['replicate_id']}
            for p_name in parameter_names_ordered: row_data[p_name] = record['params'].get(p_name, np.nan)
            records_for_df.append(row_data)

        if not records_for_df: logging.warning(f"A3 ECM: No records for DataFrame (ECM {current_ecm_str})"); continue
            
        raw_params_df_ecm = pd.DataFrame(records_for_df)
        cols_to_select_ecm = ['day', 'replicate_id'] + parameter_names_ordered
        raw_params_df_ecm = raw_params_df_ecm[[col for col in cols_to_select_ecm if col in raw_params_df_ecm.columns]]

        processed_df_ecm = raw_params_df_ecm.copy()
        if apply_day0_norm_global:
            processed_df_ecm = apply_global_day0_normalization(processed_df_ecm, N_replicates, 'day', 'replicate_id', stop_event)
            if stop_event.is_set() or processed_df_ecm.empty : logging.warning(f"A3 ECM: Data empty/cancelled after Day0 for {current_ecm_str}."); continue
        
        data_cols_ecm = [col for col in processed_df_ecm.columns if col not in ['day', 'replicate_id']]
        dfs_for_ecm_heatmaps = {} 

        if average_replicates_global:
            if N_replicates > 0 and not processed_df_ecm.empty and data_cols_ecm:
                processed_df_ecm['day'] = pd.to_numeric(processed_df_ecm['day'], errors='coerce')
                avg_df_ecm = processed_df_ecm.groupby('day')[data_cols_ecm].mean().sort_index()
                dfs_for_ecm_heatmaps["averaged"] = avg_df_ecm
            else: 
                dfs_for_ecm_heatmaps["averaged"] = processed_df_ecm.set_index('day')[data_cols_ecm] if 'day' in processed_df_ecm and data_cols_ecm else pd.DataFrame()
        else: 
            if 'replicate_id' in processed_df_ecm.columns:
                for rep_id_ecm, group_df_ecm in processed_df_ecm.groupby('replicate_id'):
                    if stop_event.is_set(): break
                    if not group_df_ecm.empty and 'day' in group_df_ecm.columns and data_cols_ecm:
                        group_df_ecm['day'] = pd.to_numeric(group_df_ecm['day'], errors='coerce')
                        unique_day_group_ecm = group_df_ecm.groupby('day')[data_cols_ecm].mean().sort_index()
                        dfs_for_ecm_heatmaps[rep_id_ecm] = unique_day_group_ecm
        if stop_event.is_set(): continue # To next ECM if cancelled here

        if not dfs_for_ecm_heatmaps: logging.warning(f"A3 ECM: No data for heatmaps for ECM {current_ecm_str}."); continue

        for replicate_key_ecm, data_for_heatmap_ecm in dfs_for_ecm_heatmaps.items():
            if stop_event.is_set() or data_for_heatmap_ecm.empty: continue
            
            title_prefix_for_file = sanitize_filename(custom_title_prefix)
            csv_sfx_ecm = f"_ECM_{sanitize_filename(current_ecm_str)}"
            csv_sfx_ecm += f"_{title_prefix_for_file}" if title_prefix_for_file else ""
            csv_sfx_ecm += f"_rep_{sanitize_filename(replicate_key_ecm)}" if replicate_key_ecm != "averaged" else "_averaged"
            
            structure_and_save_transposed_csv(data_for_heatmap_ecm,
                                              os.path.join(output_dir, f"A3_ECM_params_data{csv_sfx_ecm}_transposed.csv"),
                                              analysis_type_hint="A3_ECM")

            processed_ecm_plot, norm_obj_ecm = normalize_heatmap_data(
                data_for_heatmap_ecm, heatmap_norm_strategy, 
                frequency_values=None, # ECM params don't have a frequency axis in the same way
                param_name_for_title=f"ECM {current_ecm_str} ({replicate_key_ecm})", 
                stop_event=stop_event
            )
            if stop_event.is_set() or processed_ecm_plot.empty: continue
            
            plt.figure(figsize=(12, max(6, len(parameter_names_ordered)*0.3)))
            auto_title_suffix_ecm = f'ECM: {current_ecm_str} ({replicate_key_ecm}, Norm: {heatmap_norm_strategy}) - A3'
            plot_title_ecm = format_plot_title(custom_title_prefix, auto_title_suffix_ecm, include_auto_titles)
            xlabel_ecm_plot = custom_x_axis_title if custom_x_axis_title else "Sequence Point"

            x_ticks_indices_ecm, x_tick_labels_ecm = get_spaced_ticks_and_labels(processed_ecm_plot.index.values, custom_x_labels_str)
            
            # Y-labels are the parameter names, ensure they are from the processed_ecm_plot columns after transpose
            y_labels_ecm_plot = processed_ecm_plot.columns 

            ax = sns.heatmap(processed_ecm_plot.T, cmap=selected_palette, fmt=".2E", annot=False, 
                             xticklabels=False, yticklabels=y_labels_ecm_plot, norm=norm_obj_ecm)
            if x_ticks_indices_ecm.size > 0:
                ax.set_xticks(x_ticks_indices_ecm + 0.5)
                ax.set_xticklabels(x_tick_labels_ecm, rotation=45, ha="right")

            plt.title(plot_title_ecm); plt.xlabel(xlabel_ecm_plot); plt.ylabel('Circuit Parameter'); plt.yticks(rotation=0); plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"A3_ECM_heatmap{csv_sfx_ecm}.png")); plt.close()
            logging.info(f"Saved ECM heatmap for {current_ecm_str} ({replicate_key_ecm}).")
            if stop_event.is_set(): break # from replicate_key loop
        if stop_event.is_set(): break # from ecm_idx loop

    if stop_event.is_set(): return "ECM Fitting Cancelled."
    return "Analysis Type 3 (ECM Fitting) completed."


# ---------------------------------------------------------
# ANALYSIS FUNCTION 4: Peak Tracking 
# ---------------------------------------------------------
def run_analysis_peak_tracking(input_dir, output_dir, N_replicates_from_gui,
                               apply_day0_norm_global, average_replicates_global, heatmap_norm_strategy,
                               custom_title_prefix, include_auto_titles, custom_x_labels_str, custom_x_axis_title,
                               selected_palette, 
                               min_freq_peak, max_freq_peak, stop_event):
    logging.info("Starting Analysis Type 4 (Peak Tracking)...")
    if stop_event.is_set(): return "Peak Tracking Cancelled at Start"
    N_replicates = detect_num_replicates(input_dir, N_replicates_from_gui)
    if N_replicates == 0: N_replicates = 1
    if not os.path.isdir(input_dir): return "Error (A4): Input directory not found."
    if not os.path.isdir(output_dir): os.makedirs(output_dir, exist_ok=True)

    all_files_peak_data = []
    replicate_folders = sorted([d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))])
    if not replicate_folders: replicate_folders = [None]

    for rep_idx, replicate_folder_name in enumerate(replicate_folders):
        if stop_event.is_set(): break
        current_path = os.path.join(input_dir, replicate_folder_name) if replicate_folder_name else input_dir
        current_replicate_id = replicate_folder_name if replicate_folder_name else f"rep{rep_idx}"
        try:
            dta_files_unsorted = [f for f in os.listdir(current_path) if f.lower().endswith('.dta') and os.path.isfile(os.path.join(current_path, f))]
            dta_files = sorted(dta_files_unsorted, key=lambda f_name: extract_day_from_filename(f_name))
        except FileNotFoundError: logging.warning(f"A4 Peak: Path not found: {current_path}"); continue

        for filename in dta_files:
            if stop_event.is_set(): break
            filepath = os.path.join(current_path, filename)
            day_val = extract_day_from_filename(filename)
            if day_val == -1: logging.warning(f"A4 Peak: Skipping {filename}, invalid name."); continue
            
            logging.debug(f"A4 Peak: Processing {filepath} for day/seq: {day_val}")
            try:
                data_eis_list = deareis.parse_data(filepath)
                if not data_eis_list: logging.warning(f"A4 Peak: Could not parse {filepath}"); continue
                eis_obj = data_eis_list[0]; bode = eis_obj.get_bode_data(); nyq = eis_obj.get_nyquist_data()
                if bode is None or nyq is None or not bode[0].size: logging.warning(f"A4 Peak: No Bode/Nyquist in {filepath}"); continue

                f_orig, Zm_orig, negZphz_orig = bode[0], bode[1], bode[2]
                Zr_orig, Zimag_orig_nyq = nyq[0], nyq[1] # nyq[1] is -Zimag
                Zimag_orig = -Zimag_orig_nyq # Convert to Zimag
                
                valid_idx = (f_orig >= min_freq_peak) & (f_orig <= max_freq_peak)
                if not np.any(valid_idx): 
                    logging.warning(f"A4 Peak: No data in freq range ({min_freq_peak}-{max_freq_peak} Hz) for {filepath}.")
                    all_files_peak_data.append({'day':day_val, 'replicate_id':current_replicate_id, 'Freq_Zphz':np.nan, 'Val_Zphz':np.nan, 'Freq_Cimg':np.nan, 'Val_Cimg':np.nan})
                    continue

                f_filt, Zm_filt, negZphz_filt, Zr_filt = f_orig[valid_idx], Zm_orig[valid_idx], negZphz_orig[valid_idx], Zr_orig[valid_idx]
                # Cimg = Zreal / (2*pi*f*Zmod^2)
                Cimg_filt = np.array([(zr/(2*np.pi*f*zm**2)) if (f!=0 and zm!=0) else np.nan for zr,f,zm in zip(Zr_filt,f_filt,Zm_filt)])
                
                file_data = {'day':day_val, 'replicate_id':current_replicate_id, 'Freq_Zphz':np.nan, 'Val_Zphz':np.nan, 'Freq_Cimg':np.nan, 'Val_Cimg':np.nan}
                if len(negZphz_filt)>0 and not pd.Series(negZphz_filt).isna().all(): 
                    idx_phz=pd.Series(negZphz_filt).idxmax()
                    file_data['Freq_Zphz']=f_filt[idx_phz]
                    file_data['Val_Zphz']=negZphz_filt[idx_phz]
                
                # Find peak of |Cimg| but store actual Cimg value (can be negative)
                if len(Cimg_filt)>0 and not pd.Series(np.abs(Cimg_filt)).isna().all(): 
                    idx_cimg=pd.Series(np.abs(Cimg_filt)).idxmax()
                    file_data['Freq_Cimg']=f_filt[idx_cimg]
                    file_data['Val_Cimg']=Cimg_filt[idx_cimg]
                all_files_peak_data.append(file_data)
            except Exception as e: logging.error(f"A4 Peak: file processing error for {filepath}: {e}", exc_info=True)
        if stop_event.is_set(): break
    if stop_event.is_set() or not all_files_peak_data: return "Peak Tracking Cancelled or No data."

    raw_peak_df = pd.DataFrame(all_files_peak_data)
    processed_peak_df = raw_peak_df.copy()

    if apply_day0_norm_global:
        # For peak tracking, Day0 norm applies to 'Val_Zphz' and 'Val_Cimg'. Frequencies are not normalized.
        cols_to_norm_peak = ['Val_Zphz', 'Val_Cimg']
        temp_df_for_norm = processed_peak_df[['day','replicate_id'] + cols_to_norm_peak].copy()
        normed_values_df = apply_global_day0_normalization(temp_df_for_norm, N_replicates, 'day','replicate_id', stop_event)
        
        if stop_event.is_set() or normed_values_df.empty: return "Cancelled/Empty after Day0 (A4)"
        
        # Merge normed values back, keeping original frequencies
        # Ensure 'day' and 'replicate_id' are suitable for merge keys
        processed_peak_df.update(normed_values_df) # Update will align on index and overwrite matching columns
        # If apply_global_day0 drops rows, need a more careful merge:
        if len(normed_values_df) < len(processed_peak_df[processed_peak_df['day'] > 1]): # Day 1 is baseline
            processed_peak_df = pd.merge(processed_peak_df[['day','replicate_id','Freq_Zphz','Freq_Cimg']], 
                                          normed_values_df, on=['day','replicate_id'], how='inner')


    data_cols_peak = [col for col in processed_peak_df.columns if col not in ['day', 'replicate_id']]
    dfs_for_peak_heatmaps = {}

    if average_replicates_global:
        if N_replicates > 0 and not processed_peak_df.empty and data_cols_peak:
            processed_peak_df['day'] = pd.to_numeric(processed_peak_df['day'], errors='coerce')
            avg_df_peak = processed_peak_df.groupby('day')[data_cols_peak].mean().sort_index()
            dfs_for_peak_heatmaps["averaged"] = avg_df_peak
        else: 
            dfs_for_peak_heatmaps["averaged"] = processed_peak_df.set_index('day')[data_cols_peak] if 'day' in processed_peak_df and data_cols_peak else pd.DataFrame()
    else:
        if 'replicate_id' in processed_peak_df.columns:
            for rep_id_peak, group_df_peak in processed_peak_df.groupby('replicate_id'):
                if stop_event.is_set(): break
                if not group_df_peak.empty and 'day' in group_df_peak.columns and data_cols_peak:
                    group_df_peak['day'] = pd.to_numeric(group_df_peak['day'], errors='coerce')
                    unique_day_group_peak = group_df_peak.groupby('day')[data_cols_peak].mean().sort_index()
                    dfs_for_peak_heatmaps[rep_id_peak] = unique_day_group_peak
    if stop_event.is_set(): return "Cancelled during A4 per-replicate split."
    if not dfs_for_peak_heatmaps : return "A4: No peak data for heatmaps."

    for replicate_key_peak, data_for_rep_or_avg_peak in dfs_for_peak_heatmaps.items():
        if stop_event.is_set() or data_for_rep_or_avg_peak.empty: continue
        
        title_pfx_file_peak = sanitize_filename(custom_title_prefix)
        csv_sfx_peak_base = f"_{title_pfx_file_peak}" if title_pfx_file_peak else ""
        csv_sfx_peak_base += f"_rep_{sanitize_filename(replicate_key_peak)}" if replicate_key_peak != "averaged" else "_averaged"

        peak_metric_map = {
            "ZphzPeak": ['Freq_Zphz', 'Val_Zphz'], 
            "CimgPeak": ['Freq_Cimg', 'Val_Cimg']
        }

        for peak_type_name, data_cols_for_type in peak_metric_map.items():
            if stop_event.is_set(): break
            
            # Select relevant columns for this peak type's heatmap
            heatmap_data_current_peak_type = data_for_rep_or_avg_peak[[col for col in data_cols_for_type if col in data_for_rep_or_avg_peak.columns]].copy()
            if heatmap_data_current_peak_type.empty or heatmap_data_current_peak_type.shape[1] != 2:
                logging.warning(f"A4: Missing columns for {peak_type_name} heatmap ({replicate_key_peak}). Expected {data_cols_for_type}. Got {heatmap_data_current_peak_type.columns}. Skipping.")
                continue
            
            heatmap_data_current_peak_type.columns = ['Freq Max', 'Value'] # Standardize column names for heatmap y-axis
            
            structure_and_save_transposed_csv(heatmap_data_current_peak_type,
                                              os.path.join(output_dir, f"A4_{peak_type_name}_data{csv_sfx_peak_base}_transposed.csv"),
                                              analysis_type_hint="A4_Peak") # Hint will use index directly

            processed_peak_plot, norm_obj_peak = normalize_heatmap_data(
                heatmap_data_current_peak_type, heatmap_norm_strategy, 
                frequency_values=None, # Not frequency-based in the same way as A1/A2
                param_name_for_title=f"{peak_type_name} ({replicate_key_peak})", stop_event=stop_event
            )
            if stop_event.is_set() or processed_peak_plot.empty: continue
            
            auto_sfx_peak = f"{peak_type_name} ({replicate_key_peak}, Norm: {heatmap_norm_strategy})"
            plot_title_peak = format_plot_title(custom_title_prefix, auto_sfx_peak, include_auto_titles)
            xlabel_peak_plot = custom_x_axis_title if custom_x_axis_title else "Sequence Point"
            
            x_ticks_indices_peak, x_tick_labels_peak = get_spaced_ticks_and_labels(processed_peak_plot.index.values, custom_x_labels_str)
            
            # Y-labels are 'Freq Max', 'Value'
            y_labels_peak_plot = processed_peak_plot.columns 

            plt.figure(figsize=(8,5))
            ax_peak = sns.heatmap(processed_peak_plot.T, cmap=selected_palette, fmt=".2E", # Use .2E for potential large/small values
                                  yticklabels=y_labels_peak_plot, xticklabels=False, norm=norm_obj_peak)
            if x_ticks_indices_peak.size > 0:
                ax_peak.set_xticks(x_ticks_indices_peak + 0.5)
                ax_peak.set_xticklabels(x_tick_labels_peak, rotation=45, ha="right")
            
            plt.title(plot_title_peak); plt.xlabel(xlabel_peak_plot); plt.ylabel('Peak Metric'); plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"A4_{peak_type_name}_heatmap{csv_sfx_peak_base}.png")); plt.close()
            logging.info(f"Saved A4 {peak_type_name} heatmap ({replicate_key_peak}).")
        if stop_event.is_set(): break # from replicate_key loop

    if stop_event.is_set(): return "Peak Tracking Cancelled."
    return "Analysis Type 4 (Peak Tracking) completed."

# ---------------------------------------------------------
# Tkinter GUI Application
# ---------------------------------------------------------
class TextHandler(logging.Handler): b
    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue
        self.formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        self.setFormatter(self.formatter)
    def emit(self, record):
        msg = self.format(record)
        self.log_queue.put(msg)

class EISAnalysisApp:
    def __init__(self, master):
        self.master = master
        master.title("HeatPiV5") 
        master.geometry("900x980") 

        style = ttk.Style(); style.theme_use('clam') 
        style.configure("TLabel", padding=3, font=('Helvetica', 10))
        style.configure("TButton", padding=3, font=('Helvetica', 10))
        style.configure("TEntry", padding=3, font=('Helvetica', 10), width=45) 
        style.configure("TRadiobutton", padding=3, font=('Helvetica', 10))
        style.configure("TCombobox", padding=3, font=('Helvetica', 10), width=28) 
        style.configure("TSpinbox", padding=3, font=('Helvetica', 10), width=7)
        style.configure("TLabelframe.Label", font=('Helvetica', 11, 'bold'))
        
        main_frame = ttk.Frame(master, padding="10 10 10 10"); main_frame.pack(fill=tk.BOTH, expand=True)
        self.log_queue = queue.Queue(); self.analysis_thread = None; self.analysis_stop_event = threading.Event()
        current_row = 0 # To keep track of grid rows

        # --- General Settings ---
        general_frame = ttk.LabelFrame(main_frame, text="General Settings", padding=10)
        general_frame.grid(row=current_row, column=0, columnspan=3, sticky="ew", pady=5, padx=5)
        current_row_gf = 0 # Row counter for within general_frame

        ttk.Label(general_frame, text="Input Dir:").grid(row=current_row_gf, column=0, sticky="w", padx=5, pady=2)
        self.input_dir_var = tk.StringVar(); ttk.Entry(general_frame, textvariable=self.input_dir_var).grid(row=current_row_gf, column=1, sticky="ew", pady=2)
        ttk.Button(general_frame, text="Browse...", command=self.browse_input_dir).grid(row=current_row_gf, column=2, padx=2, pady=2); current_row_gf += 1
        
        ttk.Label(general_frame, text="Output Dir:").grid(row=current_row_gf, column=0, sticky="w", padx=5, pady=2)
        self.output_dir_var = tk.StringVar(); ttk.Entry(general_frame, textvariable=self.output_dir_var).grid(row=current_row_gf, column=1, sticky="ew", pady=2)
        ttk.Button(general_frame, text="Browse...", command=self.browse_output_dir).grid(row=current_row_gf, column=2, padx=2, pady=2); current_row_gf += 1
        
        ttk.Label(general_frame, text="Replicates (N Fallback):").grid(row=current_row_gf, column=0, sticky="w", padx=5, pady=2)
        self.replicates_gui_var = tk.IntVar(value=3); ttk.Spinbox(general_frame, from_=1, to=100, textvariable=self.replicates_gui_var, width=10).grid(row=current_row_gf, column=1, sticky="w", pady=2); current_row_gf += 1
        
        self.apply_day0_global_var = tk.BooleanVar(value=False); ttk.Checkbutton(general_frame, text="Difference From First File (Day0)", variable=self.apply_day0_global_var).grid(row=current_row_gf, column=0, sticky="w", padx=5, pady=2) 
        self.average_replicates_global_var = tk.BooleanVar(value=True); ttk.Checkbutton(general_frame, text="Average Replicates", variable=self.average_replicates_global_var).grid(row=current_row_gf, column=1, sticky="w", padx=5, pady=2); current_row_gf += 1
        
        ttk.Label(general_frame, text="Heatmap Norm Strategy:").grid(row=current_row_gf, column=0, sticky="w", padx=5, pady=2)
        self.heatmap_norm_strategy_var = tk.StringVar()
        self.heatmap_norm_options = ["Raw Values", "Per Parameter/Timeline (Column-wise)", 
                                     "Global Max Scaling", "Frequency Sections (L/M/H)",
                                     "Logarithmic Color Scale"] # Added new option
        self.heatmap_norm_combo = ttk.Combobox(general_frame, textvariable=self.heatmap_norm_strategy_var, values=self.heatmap_norm_options, state="readonly")
        self.heatmap_norm_combo.grid(row=current_row_gf, column=1, columnspan=2, sticky="ew", pady=2); self.heatmap_norm_combo.set(self.heatmap_norm_options[1]); current_row_gf += 1
        
        ttk.Label(general_frame, text="Custom Plot Title Prefix:").grid(row=current_row_gf, column=0, sticky="w", padx=5, pady=2)
        self.custom_title_prefix_var = tk.StringVar(); ttk.Entry(general_frame, textvariable=self.custom_title_prefix_var).grid(row=current_row_gf, column=1, columnspan=2, sticky="ew", pady=2); current_row_gf += 1
        
        self.include_auto_titles_var = tk.BooleanVar(value=True); ttk.Checkbutton(general_frame, text="Include Auto Suffix in Titles", variable=self.include_auto_titles_var).grid(row=current_row_gf, column=0, columnspan=2, sticky="w", padx=5, pady=2); current_row_gf += 1
        
        ttk.Label(general_frame, text="Custom X-axis Labels (CSV):").grid(row=current_row_gf, column=0, sticky="w", padx=5, pady=2)
        self.custom_x_labels_var = tk.StringVar(); ttk.Entry(general_frame, textvariable=self.custom_x_labels_var).grid(row=current_row_gf, column=1, columnspan=2, sticky="ew", pady=2); current_row_gf += 1
        
        ttk.Label(general_frame, text="Custom X-axis Title:").grid(row=current_row_gf, column=0, sticky="w", padx=5, pady=2)
        self.custom_x_axis_title_var = tk.StringVar(); ttk.Entry(general_frame, textvariable=self.custom_x_axis_title_var).grid(row=current_row_gf, column=1, columnspan=2, sticky="ew", pady=2); current_row_gf += 1

        ttk.Label(general_frame, text="Heatmap Palette:").grid(row=current_row_gf, column=0, sticky="w", padx=5, pady=2)
        self.heatmap_palette_var = tk.StringVar()
        self.heatmap_palette_options = ['viridis', 'plasma', 'cividis', 'rocket', 'mako', 'Spectral'] # Added Spectral back as it was commonly used
        self.heatmap_palette_combo = ttk.Combobox(general_frame, textvariable=self.heatmap_palette_var, 
                                                  values=self.heatmap_palette_options, state="readonly")
        self.heatmap_palette_combo.grid(row=current_row_gf, column=1, columnspan=2, sticky="ew", pady=2); self.heatmap_palette_combo.set(self.heatmap_palette_options[0]); current_row_gf += 1
        
        general_frame.columnconfigure(1, weight=1)
        current_row +=1 # Increment main_frame row counter

        # --- Analysis Type Selection Frame ---
        analysis_select_frame = ttk.LabelFrame(main_frame, text="Select Analysis Type", padding=10)
        analysis_select_frame.grid(row=current_row, column=0, columnspan=3, sticky="ew", pady=5, padx=5); current_row += 1
        self.analysis_type_var = tk.IntVar(value=1)
        analysis_options = [("Raw Vars Heatmaps (A1)", 1), ("DRT Analysis (A2)", 2), ("ECM Fitting (A3)", 3), ("Peak Tracking (A4)", 4)]
        for i, (text, val) in enumerate(analysis_options): ttk.Radiobutton(analysis_select_frame, text=text, variable=self.analysis_type_var, value=val, command=self.toggle_parameter_frames).grid(row=i//2, column=i%2, sticky="w", padx=5, pady=2)

        self.param_frames_container_row = current_row; current_row +=1 
        self.param_frames = {}

        # --- Analysis 1 Parameter Frame ---
        self.param_frames[1] = ttk.LabelFrame(main_frame, text="Analysis 1: Raw Variables Settings", padding=10)
        self.a1_norm_stddev_var = tk.BooleanVar(value=False) 
        ttk.Checkbutton(self.param_frames[1], text="Apply Chosen Norm to StdDev Heatmap", variable=self.a1_norm_stddev_var).pack(padx=5, pady=5, fill=tk.X)

        # --- Analysis 2 Parameter Frame (DRT) --- (Layout from GitHub)
        self.param_frames[2] = ttk.LabelFrame(main_frame, text="Analysis 2: DRT Parameters", padding=10)
        drt_mappings = get_drt_enum_mapping() # Ensure this helper is correct
        ttk.Label(self.param_frames[2], text="DRT Method:").grid(row=0, column=0, sticky="w"); self.drt_method_var = tk.StringVar(); self.drt_method_options = list(drt_mappings["DRTMethod"].keys())
        self.drt_method_combo = ttk.Combobox(self.param_frames[2], textvariable=self.drt_method_var, values=self.drt_method_options, state="readonly"); self.drt_method_combo.grid(row=0, column=1, sticky="w"); self.drt_method_combo.set(self.drt_method_options[0] if self.drt_method_options else "")
        ttk.Label(self.param_frames[2], text="DRT Mode:").grid(row=0, column=2, sticky="w", padx=5); self.drt_mode_var = tk.StringVar(); self.drt_mode_options = list(drt_mappings["DRTMode"].keys())
        self.drt_mode_combo = ttk.Combobox(self.param_frames[2], textvariable=self.drt_mode_var, values=self.drt_mode_options, state="readonly"); self.drt_mode_combo.grid(row=0, column=3, sticky="w"); self.drt_mode_combo.set(self.drt_mode_options[0] if self.drt_mode_options else "")
        ttk.Label(self.param_frames[2], text="Lambda Values (CSV):").grid(row=1, column=0, sticky="w"); self.lambda_values_var = tk.StringVar(value="0.1,0.01,0.001")
        ttk.Entry(self.param_frames[2], textvariable=self.lambda_values_var, width=20).grid(row=1, column=1, columnspan=3, sticky="ew")
        ttk.Label(self.param_frames[2], text="RBF Type:").grid(row=2, column=0, sticky="w"); self.rbf_type_var = tk.StringVar(); self.rbf_type_options = list(drt_mappings["RBFType"].keys())
        self.rbf_type_combo = ttk.Combobox(self.param_frames[2], textvariable=self.rbf_type_var, values=self.rbf_type_options, state="readonly"); self.rbf_type_combo.grid(row=2, column=1, sticky="w"); self.rbf_type_combo.set("Gaussian")
        ttk.Label(self.param_frames[2], text="RBF Shape:").grid(row=2, column=2, sticky="w", padx=5); self.rbf_shape_var = tk.StringVar(); self.rbf_shape_options = list(drt_mappings["RBFShape"].keys())
        self.rbf_shape_combo = ttk.Combobox(self.param_frames[2], textvariable=self.rbf_shape_var, values=self.rbf_shape_options, state="readonly"); self.rbf_shape_combo.grid(row=2, column=3, sticky="w"); self.rbf_shape_combo.set("FWHM")
        ttk.Label(self.param_frames[2], text="RBF Size:").grid(row=3, column=0, sticky="w"); self.rbf_size_var = tk.DoubleVar(value=0.5)
        ttk.Entry(self.param_frames[2], textvariable=self.rbf_size_var, width=10).grid(row=3, column=1, sticky="w")
        ttk.Label(self.param_frames[2], text="Fit Penalty (Deriv.):").grid(row=3, column=2, sticky="w", padx=5); self.fit_penalty_drt_var = tk.IntVar(value=1)
        ttk.Spinbox(self.param_frames[2], from_=0, to=5, textvariable=self.fit_penalty_drt_var, width=10).grid(row=3, column=3, sticky="w")
        ttk.Label(self.param_frames[2], text="Num. DRT Attempts:").grid(row=4, column=0, sticky="w"); self.num_attempts_drt_var = tk.IntVar(value=100)
        ttk.Entry(self.param_frames[2], textvariable=self.num_attempts_drt_var, width=10).grid(row=4, column=1, sticky="w")
        ttk.Label(self.param_frames[2], text="Prelim Fit CDC (for MRQ):").grid(row=4, column=2, sticky="w", padx=5); self.mrq_cdc_drt_var = tk.StringVar(value="R")
        ttk.Entry(self.param_frames[2], textvariable=self.mrq_cdc_drt_var, width=20).grid(row=4, column=3, sticky="ew")
        self.include_inductance_drt_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.param_frames[2], text="Include Inductance", variable=self.include_inductance_drt_var).grid(row=5, column=0, columnspan=2, sticky="w")
        self.param_frames[2].columnconfigure(1, weight=1); self.param_frames[2].columnconfigure(3, weight=1)

        # --- Analysis 3 Parameter Frame (ECM Fitting) --- 
        self.param_frames[3] = ttk.LabelFrame(main_frame, text="Analysis 3: ECM Fitting Parameters", padding=10)
        ttk.Label(self.param_frames[3], text="ECMs (CSV):").grid(row=0, column=0, sticky="w"); self.ecm_list_a3_var = tk.StringVar(value='R(Q[RW]),(RC)')
        ttk.Entry(self.param_frames[3], textvariable=self.ecm_list_a3_var, width=35).grid(row=0, column=1, columnspan=3, sticky="ew")
        ttk.Label(self.param_frames[3], text="Fit Method:").grid(row=1, column=0, sticky="w"); self.ecm_fit_method_a3_var = tk.StringVar(); self.ecm_fit_method_options_a3 = ["AUTO", "Nelder-Mead", "L-BFGS-B", "SLSQP", "Powell"]
        self.ecm_fit_method_combo_a3 = ttk.Combobox(self.param_frames[3], textvariable=self.ecm_fit_method_a3_var, values=self.ecm_fit_method_options_a3, state="readonly"); self.ecm_fit_method_combo_a3.grid(row=1, column=1, sticky="w"); self.ecm_fit_method_combo_a3.set("AUTO")
        ttk.Label(self.param_frames[3], text="Fit Weight:").grid(row=1, column=2, sticky="w", padx=5); self.ecm_fit_weight_a3_var = tk.StringVar(); self.ecm_fit_weight_options_a3 = ["AUTO", "MODULUS", "UNIT", "PROPIMAG", "PROPREAL"]
        self.ecm_fit_weight_combo_a3 = ttk.Combobox(self.param_frames[3], textvariable=self.ecm_fit_weight_a3_var, values=self.ecm_fit_weight_options_a3, state="readonly"); self.ecm_fit_weight_combo_a3.grid(row=1, column=3, sticky="w"); self.ecm_fit_weight_combo_a3.set("AUTO")
        ttk.Label(self.param_frames[3], text="Max Evals (Fit):").grid(row=2, column=0, sticky="w"); self.ecm_max_nfev_a3_var = tk.IntVar(value=30000)
        ttk.Entry(self.param_frames[3], textvariable=self.ecm_max_nfev_a3_var, width=10).grid(row=2, column=1, sticky="w")
        self.param_frames[3].columnconfigure(1, weight=1); self.param_frames[3].columnconfigure(3, weight=1)

        # --- Analysis 4 Parameter Frame (Peak Tracking) --- 
        self.param_frames[4] = ttk.LabelFrame(main_frame, text="Analysis 4: Peak Tracking Parameters", padding=10)
        ttk.Label(self.param_frames[4], text="Min Freq for Peak Search (Hz):").grid(row=0, column=0, sticky="w", pady=2); self.min_freq_a4_var = tk.DoubleVar(value=0.1) 
        ttk.Entry(self.param_frames[4], textvariable=self.min_freq_a4_var, width=10).grid(row=0, column=1, sticky="w", pady=2)
        ttk.Label(self.param_frames[4], text="Max Freq for Peak Search (Hz):").grid(row=1, column=0, sticky="w", pady=2); self.max_freq_a4_var = tk.DoubleVar(value=100000.0) 
        ttk.Entry(self.param_frames[4], textvariable=self.max_freq_a4_var, width=10).grid(row=1, column=1, sticky="w", pady=2)
        self.param_frames[4].columnconfigure(1, weight=1)

        for i, frame in self.param_frames.items():
            frame.grid(row=self.param_frames_container_row, column=0, columnspan=3, sticky="nsew", pady=5, padx=5)
            if i != self.analysis_type_var.get(): frame.grid_remove()
        
        # --- Run Button and Log --- (Layout from GitHub)
        self.run_button = ttk.Button(main_frame, text="Run Selected Analysis", command=self.run_analysis_thread, style="Accent.TButton")
        style.configure("Accent.TButton", font=('Helvetica', 11, 'bold'), foreground="white", background="#0078D7") # Example accent
        self.run_button.grid(row=current_row, column=0, columnspan=3, pady=20, ipady=5); current_row += 1
        
        log_label = ttk.Label(main_frame, text="Log / Status:", font=('Helvetica', 10, 'italic'))
        log_label.grid(row=current_row, column=0, columnspan=3, sticky="w", pady=(5,0), padx=5); current_row += 1
        
        self.log_text = scrolledtext.ScrolledText(main_frame, height=18, width=90, wrap=tk.WORD, state=tk.DISABLED, relief=tk.SUNKEN, borderwidth=1)
        self.log_text.grid(row=current_row, column=0, columnspan=3, sticky="nsew", padx=5, pady=5)
        
        main_frame.rowconfigure(current_row, weight=1) # Make log area expand
        main_frame.columnconfigure(1, weight=1) # Make middle column of general settings expand
        
        # --- Logging Setup ---
        self.log_handler = TextHandler(self.log_queue)
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        if logger.hasHandlers(): logger.handlers.clear() # Clear existing to avoid duplicates if script is re-run in some envs
        logger.addHandler(self.log_handler)
        # Optional: Add console handler for debugging if needed
        # console_handler = logging.StreamHandler()
        # console_handler.setFormatter(self.log_handler.formatter)
        # logger.addHandler(console_handler)
        
        self.master.protocol("WM_DELETE_WINDOW", self.on_app_close)
        self.process_log_queue() # Start queue processing
        self.toggle_parameter_frames() # Show initial correct param frame

    def toggle_parameter_frames(self): 
        selected_analysis = self.analysis_type_var.get()
        for analysis_num, frame in self.param_frames.items():
            if analysis_num == selected_analysis:
                frame.grid()
            else:
                frame.grid_remove()

    def browse_input_dir(self): 
        dirname = filedialog.askdirectory(title="Select Input Data Directory")
        if dirname:
            self.input_dir_var.set(dirname)

    def browse_output_dir(self): 
        dirname = filedialog.askdirectory(title="Select Output Directory")
        if dirname:
            self.output_dir_var.set(dirname)
    
    def process_log_queue(self): 
        try:
            while True: 
                item = self.log_queue.get_nowait()
                if isinstance(item, str): message = item
                elif isinstance(item, tuple) and len(item) >= 2:
                    command_type, title_or_msg, msg_body = item[0], item[1], item[2] if len(item)==3 else ""
                    if self.master.winfo_exists():
                        if command_type == "SHOW_ERROR": messagebox.showerror(title_or_msg, msg_body)
                        elif command_type == "SHOW_INFO": messagebox.showinfo(title_or_msg, msg_body)
                        elif command_type == "SHOW_WARNING": messagebox.showwarning(title_or_msg, msg_body)
                    continue 
                else: message = str(item) 
                if self.log_text.winfo_exists(): # Check if widget exists
                    self.log_text.configure(state=tk.NORMAL)
                    self.log_text.insert(tk.END, message + '\n')
                    self.log_text.see(tk.END)
                    self.log_text.configure(state=tk.DISABLED)
        except queue.Empty: pass 
        except tk.TclError: pass # Handle cases where GUI elements might be destroyed
        except Exception as e: print(f"Error processing log queue item: {e}") # Fallback print
        
        if self.master.winfo_exists(): # Schedule next check only if master window exists
            self.master.after(100, self.process_log_queue)

    def on_app_close(self): 
        if messagebox.askokcancel("Quit", "Are you sure you want to quit? Any running analysis will be stopped."):
            logging.info("Application shutdown initiated by user.")
            self.analysis_stop_event.set() # Signal any running analysis to stop
            if self.analysis_thread and self.analysis_thread.is_alive():
                logging.info("Waiting for analysis thread to complete (max 2 seconds)...")
                self.analysis_thread.join(timeout=2.0) 
                if self.analysis_thread.is_alive():
                    logging.warning("Analysis thread did not terminate gracefully within timeout.")
            logging.info("Destroying main window.")
            self.master.destroy() 

    def run_analysis_thread(self): 
        if self.analysis_thread and self.analysis_thread.is_alive():
            messagebox.showwarning("Busy", "An analysis is already running. Please wait.")
            return
        self.run_button.config(state=tk.DISABLED)
        self.analysis_stop_event.clear() 
        self.analysis_thread = threading.Thread(target=self.run_analysis, daemon=True)
        self.analysis_thread.start()

    def run_analysis(self): 
        input_dir = self.input_dir_var.get()
        output_dir = self.output_dir_var.get()
        n_replicates_fallback = self.replicates_gui_var.get()
        apply_day0_global = self.apply_day0_global_var.get()
        average_replicates_global = self.average_replicates_global_var.get()
        heatmap_norm_global = self.heatmap_norm_strategy_var.get() 
        custom_title_prefix = self.custom_title_prefix_var.get()
        include_auto_titles = self.include_auto_titles_var.get()
        custom_x_labels_str_gui = self.custom_x_labels_var.get() 
        custom_x_axis_title_gui = self.custom_x_axis_title_var.get()
        selected_palette_gui = self.heatmap_palette_var.get() 
        analysis_type = self.analysis_type_var.get()

        if not input_dir or not output_dir:
            self.log_queue.put(("SHOW_ERROR", "Validation Error", "Input and Output directories must be specified."))
            logging.error("Input or Output directory not specified.")
            if not self.analysis_stop_event.is_set() and self.master.winfo_exists(): 
                self.master.after(0, lambda: {self.run_button.config(state=tk.NORMAL) if self.run_button.winfo_exists() else None})
            return

        logging.info("="*40 + f"\nRUNNING ANALYSIS TYPE: {analysis_type}\n" + "="*40)
        result_message = "Analysis type not found or did not run."
        if self.analysis_stop_event.is_set(): 
            logging.info("Analysis aborted pre-start due to stop event."); 
            if self.master.winfo_exists(): self.master.after(0, lambda: {self.run_button.config(state=tk.NORMAL) if self.run_button.winfo_exists() else None})
            return "Analysis Aborted Pre-Start"

        try:
            common_args = [input_dir, output_dir, n_replicates_fallback, 
                           apply_day0_global, average_replicates_global, heatmap_norm_global,
                           custom_title_prefix, include_auto_titles, custom_x_labels_str_gui, custom_x_axis_title_gui,
                           selected_palette_gui] # Add selected_palette_gui

            if analysis_type == 1:
                a1_norm_stddev = self.a1_norm_stddev_var.get()
                result_message = run_analysis_1(*common_args, a1_norm_stddev, self.analysis_stop_event) 
            elif analysis_type == 2:
                mappings = get_drt_enum_mapping()
                drt_method = mappings["DRTMethod"].get(self.drt_method_var.get())
                drt_mode = mappings["DRTMode"].get(self.drt_mode_var.get())
                rbf_type = mappings["RBFType"].get(self.rbf_type_var.get())
                rbf_shape = mappings["RBFShape"].get(self.rbf_shape_var.get())
                if not all([drt_method, drt_mode, rbf_type, rbf_shape]): raise ValueError("Invalid DRT enum selection.")
                lambda_vals_str = self.lambda_values_var.get()
                if not lambda_vals_str.strip(): raise ValueError("Lambda list is empty.")
                lambda_vals = [float(L.strip()) for L in lambda_vals_str.split(',') if L.strip()]
                if not lambda_vals: raise ValueError("Parsed Lambda list is empty.")
                
                drt_specific_args = [drt_method, drt_mode, lambda_vals, rbf_type, rbf_shape, 
                                     self.rbf_size_var.get(), self.fit_penalty_drt_var.get(), 
                                     self.include_inductance_drt_var.get(), self.num_attempts_drt_var.get(), 
                                     self.mrq_cdc_drt_var.get(), 0] # num_drt_procs=0 (serial)
                result_message = run_analysis_drt(*common_args, *drt_specific_args, self.analysis_stop_event)
            elif analysis_type == 3:
                ecms_str = self.ecm_list_a3_var.get()
                if not ecms_str.strip(): raise ValueError("ECM list is empty.")
                ecms = [ecm.strip() for ecm in ecms_str.split(',') if ecm.strip()]
                if not ecms: raise ValueError("Parsed ECM list is empty.")
                ecm_specific_args = [ecms, self.ecm_fit_method_a3_var.get(), 
                                     self.ecm_fit_weight_a3_var.get(), self.ecm_max_nfev_a3_var.get()]
                result_message = run_analysis_ecm_fitting(*common_args, *ecm_specific_args, self.analysis_stop_event)
            elif analysis_type == 4:
                min_freq = self.min_freq_a4_var.get() 
                max_freq = self.max_freq_a4_var.get() 
                if min_freq >= max_freq : raise ValueError("Min frequency must be less than Max frequency for peak tracking.")
                peak_specific_args = [min_freq, max_freq]
                result_message = run_analysis_peak_tracking(*common_args, *peak_specific_args, self.analysis_stop_event)
            
            if not self.analysis_stop_event.is_set(): # Only show messages if not cancelled
                logging.info(f"Analysis Type {analysis_type} final message: {result_message}")
                if "error" in result_message.lower() or "Error" in result_message : 
                    self.log_queue.put(("SHOW_ERROR", "Analysis Error", result_message))
                elif not any(x in result_message for x in ["Cancelled", "Aborted"]): 
                    self.log_queue.put(("SHOW_INFO", "Analysis Complete", result_message))
        
        except InterruptedError: # Not a standard Python error, but could be a custom one for stop_event
             inter_msg = result_message if 'result_message' in locals() and any(x in result_message for x in ["Cancelled", "Aborted"]) else 'Operation Cancelled by User.'
             logging.info(f"Analysis explicitly interrupted: {inter_msg}")
             if self.master.winfo_exists() and not self.analysis_stop_event.is_set():
                 self.log_queue.put(("SHOW_WARNING", "Cancelled", inter_msg))
        except ValueError as ve: 
            if not self.analysis_stop_event.is_set(): logging.error(f"Input Error (A{analysis_type}): {ve}"); self.log_queue.put(("SHOW_ERROR", "Input Error", str(ve)))
        except Exception as e:
            if not self.analysis_stop_event.is_set(): logging.error(f"Unhandled Error (A{analysis_type}): {e}", exc_info=True); self.log_queue.put(("SHOW_ERROR", "Critical Error", f"Error in A{analysis_type}:\n{e}"))
        finally:
            if not self.analysis_stop_event.is_set(): # Re-enable button if not stopped externally
                try:
                    if self.master.winfo_exists(): self.master.after(0, lambda: {self.run_button.config(state=tk.NORMAL) if self.run_button.winfo_exists() else None})
                except tk.TclError: pass # Widget might already be destroyed
            self.analysis_thread = None # Clear thread reference

# ---------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------
if __name__ == '__main__':
    root = tk.Tk()
    app = EISAnalysisApp(root)
    root.mainloop()
