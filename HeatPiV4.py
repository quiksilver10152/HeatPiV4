# -*- coding: utf-8 -*-
"""
Created on Wed May 14 18:47:04 2025

@author: quiks and Gemini 2.5 Pro

HeatPiV4
"""

import os
import logging
import threading
import tkinter as tk
from tkinter import filedialog, scrolledtext, ttk, messagebox
import queue 
import re # For filename sanitization
import deareis 
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use('Agg') 

# ---------------------------------------------------------
# --- HELPER FUNCTIONS ---
# ---------------------------------------------------------

def sanitize_filename(name_str, max_len=40):
    """Sanitizes a string to be used as part of a filename."""
    if not name_str:
        return ""
    # Remove characters that are problematic in filenames
    sane_name = re.sub(r'[\\/*?:"<>|]', "", name_str)
    # Replace spaces and multiple underscores with a single underscore
    sane_name = re.sub(r'\s+', "_", sane_name)
    sane_name = re.sub(r'_+', "_", sane_name)
    # Truncate if too long (ensure it doesn't break extension if added later)
    return sane_name[:max_len].strip('_')

def detect_num_replicates(input_dir, gui_replicates_fallback):
    """Counts subdirectories in input_dir to determine N. Falls back to GUI value."""
    try:
        subdirectories = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
        n_detected = len(subdirectories)
        if n_detected > 0:
            logging.info(f"Detected N={n_detected} replicate folders.")
            return n_detected
        else:
            logging.info(f"No replicate subfolders detected in {input_dir}. Using N={gui_replicates_fallback} from GUI settings.")
            return gui_replicates_fallback
    except Exception as e:
        logging.warning(f"Error detecting replicate folders: {e}. Using N={gui_replicates_fallback} from GUI settings.")
        return gui_replicates_fallback

def apply_global_day0_normalization(df_input, N_replicates, day_col_name='day', replicate_col_name='replicate_id', stop_event=None):
    """
    Applies Day0 normalization: Day_K_Rep_X - Day_0_Rep_X for that replicate.
    The first timepoint's data (which becomes ~0) is then dropped.
    """
    if df_input.empty:
        logging.warning("Day0 normalization skipped: Input DataFrame is empty.")
        return df_input
    
    logging.info(f"Applying Day0 normalization with N_replicates = {N_replicates} (if applicable)...")
    if stop_event and stop_event.is_set(): return df_input
    
    df_with_rep_id = df_input.copy()
    if replicate_col_name not in df_with_rep_id.columns:
        if N_replicates == 1 or len(df_with_rep_id[day_col_name].unique()) == len(df_with_rep_id):
            df_with_rep_id[replicate_col_name] = 'rep_0' # Single series
        else: # Try to assign based on sort order and N - this assumes balanced data
            df_with_rep_id = df_with_rep_id.sort_values(by=[day_col_name]) # Must sort by day first for this to work
            df_with_rep_id[replicate_col_name] = [f'rep{i%N_replicates}' for i in range(len(df_with_rep_id))]
            logging.warning(f"'{replicate_col_name}' not found. Created dummy IDs based on N={N_replicates} and data order. Ensure data is sorted by day, then replicate for accuracy.")

    all_norm_dfs = []
    unique_replicates = df_with_rep_id[replicate_col_name].unique()
    data_cols = [col for col in df_with_rep_id.columns if col not in [day_col_name, replicate_col_name]]

    for rep_id in unique_replicates:
        if stop_event and stop_event.is_set(): break
        rep_df = df_with_rep_id[df_with_rep_id[replicate_col_name] == rep_id].sort_values(by=day_col_name)
        if rep_df.empty: continue
        
        baseline_day_val = rep_df[day_col_name].min()
        baseline_row_s = rep_df[rep_df[day_col_name] == baseline_day_val]
        if baseline_row_s.empty: 
            logging.warning(f"No baseline data for replicate '{rep_id}' at day {baseline_day_val}. Skipping its Day0 norm.")
            # Add non-baseline days as is, or filter them out if strict day0 norm required
            all_norm_dfs.append(rep_df[rep_df[day_col_name] != baseline_day_val]) 
            continue
        baseline_row = baseline_row_s.iloc[0]
        
        norm_rep_df_rows = []
        for _, row in rep_df.iterrows():
            if stop_event and stop_event.is_set(): break
            if row[day_col_name] == baseline_day_val: continue 

            new_row = row.copy()
            for d_col in data_cols:
                try: new_row[d_col] = float(row[d_col]) - float(baseline_row[d_col])
                except (ValueError, TypeError): new_row[d_col] = np.nan 
            norm_rep_df_rows.append(new_row)
        
        if stop_event and stop_event.is_set(): break
        if norm_rep_df_rows: all_norm_dfs.append(pd.DataFrame(norm_rep_df_rows))
            
    if stop_event and stop_event.is_set(): return df_input
    if not all_norm_dfs: 
        logging.warning("Day0 normalization resulted in no data.")
        return pd.DataFrame(columns=df_input.columns) 
    
    result_df = pd.concat(all_norm_dfs, ignore_index=True)
    logging.info(f"Day0 normalization applied. Original rows: {len(df_input)}, Result rows: {len(result_df)}.")
    return result_df

def normalize_heatmap_data(data_for_heatmap, strategy, frequency_values=None, param_name_for_title="", stop_event=None):
    if stop_event and stop_event.is_set(): logging.info(f"Heatmap norm for {param_name_for_title} cancelled."); return data_for_heatmap
    if data_for_heatmap.empty: return data_for_heatmap
    logging.info(f"Applying heatmap norm: '{strategy}' for {param_name_for_title or 'data'}")
    
    try: df_float = data_for_heatmap.astype(float)
    except Exception as e:
        logging.warning(f"Could not convert heatmap data to float for norm ({param_name_for_title}): {e}. Using Raw.")
        return data_for_heatmap.fillna(0.0) 

    if strategy == "Raw Values": return df_float.fillna(0.0)
    
    scaled_df = df_float.copy() 

    if strategy == "Per Parameter/Timeline (Column-wise)":
        for col in scaled_df.columns:
            if stop_event and stop_event.is_set(): return data_for_heatmap
            series = scaled_df[col]; min_v, max_v = series.min(), series.max()
            if pd.isna(min_v) or pd.isna(max_v) or (max_v == min_v): scaled_df[col] = 0.5 if not pd.isna(min_v) else np.nan
            elif (max_v - (0.99 * min_v)) == 0 : scaled_df[col] = 0.5 
            else: scaled_df[col] = (series - (0.99 * min_v)) / (max_v - (0.99 * min_v))
        return scaled_df.fillna(0.0)
    
    elif strategy == "Global Max Scaling":
        if scaled_df.empty or scaled_df.size == 0: return scaled_df.fillna(0.0)
        global_min_s = scaled_df.min(); global_max_s = scaled_df.max()
        if global_min_s.empty or global_max_s.empty : return scaled_df.fillna(0.0)
        global_min, global_max = global_min_s.min(), global_max_s.max()
        if pd.isna(global_min) or pd.isna(global_max) or (global_max == global_min): return scaled_df.fillna(0.0) 
        denominator = global_max - (0.99 * global_min)
        if denominator == 0: return (scaled_df / global_max if global_max !=0 else scaled_df*0).fillna(0.0)
        return ((scaled_df - (0.99 * global_min)) / denominator).fillna(0.0)

    elif strategy == "Frequency Sections (L/M/H)":
        if frequency_values is None or not isinstance(frequency_values, (list, np.ndarray)) or len(frequency_values) != len(scaled_df.columns):
            logging.warning("Freq values issue for 'Freq Sections'. Falling back to 'Per Timeline'.")
            return normalize_heatmap_data(scaled_df, "Per Parameter/Timeline (Column-wise)", None, param_name_for_title, stop_event)
        
        num_freqs = len(frequency_values)
        if num_freqs < 3: logging.warning("Too few freqs for 'Freq Sections'. Falling back to 'Per Timeline'."); return normalize_heatmap_data(scaled_df, "Per Parameter/Timeline (Column-wise)", None, param_name_for_title, stop_event)

        n_high = num_freqs // 3; n_mid = num_freqs // 3; n_low = num_freqs - n_high - n_mid
        sections_indices = []
        if n_high > 0: sections_indices.append((0, n_high)) 
        if n_mid > 0: sections_indices.append((n_high, n_high + n_mid)) 
        if n_low > 0: sections_indices.append((n_high + n_mid, num_freqs)) 
        
        temp_scaled_df = scaled_df.copy() 
        for start_idx, end_idx in sections_indices:
            if stop_event and stop_event.is_set(): return data_for_heatmap
            section_cols = temp_scaled_df.columns[start_idx:end_idx]
            if section_cols.empty: continue
            section_data = temp_scaled_df[section_cols]
            if section_data.empty or section_data.size == 0: continue
            sec_min_s = section_data.min(); sec_max_s = section_data.max()
            if sec_min_s.empty or sec_max_s.empty: continue
            sec_min, sec_max = sec_min_s.min(), sec_max_s.max()
            denominator = sec_max - (0.99 * sec_min)
            if pd.isna(sec_min) or pd.isna(sec_max) or (sec_max == sec_min) or denominator == 0:
                temp_scaled_df[section_cols] = 0.5 
            else: temp_scaled_df[section_cols] = (section_data - (0.99 * sec_min)) / denominator
        return temp_scaled_df.fillna(0.0)
    
    logging.warning(f"Unknown heatmap norm strategy: '{strategy}'. Ret unnormalized for {param_name_for_title}.")
    return df_float.fillna(0.0)

def get_drt_enum_mapping():
    return {
        "DRTMethod": {"MRQ Fit": deareis.DRTMethod.MRQ_FIT, "Tikhonov NNLS": deareis.DRTMethod.TR_NNLS, 
                      "Tikhonov RBF": deareis.DRTMethod.TR_RBF, "Bayesian Hilbert": deareis.DRTMethod.BHT},
        "DRTMode": {"Imaginary": deareis.DRTMode.IMAGINARY, "Real": deareis.DRTMode.REAL, 
                    "Complex": deareis.DRTMode.COMPLEX},
        "RBFType": {"Gaussian": deareis.RBFType.GAUSSIAN, "C0 Matern": deareis.RBFType.C0_MATERN, 
                    "C2 Matern": deareis.RBFType.C2_MATERN, "Cauchy": deareis.RBFType.CAUCHY,
                    "Inverse Quadratic": deareis.RBFType.INVERSE_QUADRATIC},
        "RBFShape": {"FWHM": deareis.RBFShape.FWHM, "Factor": deareis.RBFShape.FACTOR}
    }

def format_plot_title(custom_prefix, auto_suffix, include_auto_suffix_bool):
    """Formats plot title based on custom prefix and auto suffix options."""
    if custom_prefix and include_auto_suffix_bool: return f"{custom_prefix} - {auto_suffix}"
    elif custom_prefix: return custom_prefix
    elif include_auto_suffix_bool: return auto_suffix
    return "EIS Analysis Plot" # Default if nothing useful

def get_spaced_ticks_and_labels(data_axis_len, custom_labels_str, max_ticks=15, default_label_prefix="Day"):
    """Helper for spacing out x or y tick labels for heatmaps. data_axis_len refers to number of rows in heatmap (transposed data)."""
    custom_labels = [label.strip() for label in custom_labels_str.split(',') if label.strip()] if custom_labels_str else []
    
    if not custom_labels: # Use default numeric labels if no custom ones
        if data_axis_len == 0: return [], []
        if data_axis_len <= max_ticks:
            ticks = np.arange(data_axis_len)
            labels = [f"{default_label_prefix} {i}" for i in ticks] if default_label_prefix else [str(i) for i in ticks]
            return ticks, labels
        else:
            ticks = np.linspace(0, data_axis_len - 1, max_ticks, dtype=int)
            labels = [f"{default_label_prefix} {i}" for i in ticks] if default_label_prefix else [str(i) for i in ticks]
            return ticks, labels

    num_custom_labels = len(custom_labels)
    if num_custom_labels == 0: return np.array([]), []

    if num_custom_labels >= data_axis_len: # More or equal labels than data points - use subset of labels for data points
        ticks = np.arange(data_axis_len)
        labels = custom_labels[:data_axis_len]
        return ticks, labels
    else: # Fewer labels than data points - space them out
        ticks = np.linspace(0, data_axis_len -1 , num_custom_labels, dtype=int)
        return ticks, custom_labels

# ---------------------------------------------------------
# --- ANALYSIS FUNCTION 1: Raw Variables & Standard Deviations ---
# ---------------------------------------------------------
def run_analysis_1(input_dir, output_dir, N_replicates_from_gui, 
                   apply_day0_norm_global, average_replicates_global, heatmap_norm_strategy,
                   custom_title_prefix, include_auto_titles, custom_x_labels, custom_x_axis_title, # New global GUI options
                   a1_norm_stddev_heatmap_global, # A1 Specific
                   stop_event):
    logging.info("Starting Analysis Type 1: Raw Variables Heatmaps...")
    if stop_event.is_set(): return "Analysis Cancelled at Start (A1)"
    
    N_replicates = detect_num_replicates(input_dir, N_replicates_from_gui)

    all_files_data_list = []
    processed_one_file_for_cols = False
    dynamic_param_column_names = [] 
    common_freqs_hz_a1 = np.array([]) 

    replicate_folders = sorted([d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))])
    if not replicate_folders: replicate_folders = [None]

    for rep_idx, replicate_folder_name in enumerate(replicate_folders):
        if stop_event.is_set(): return "Analysis Cancelled During Replicate Loop (A1)"
        current_path = os.path.join(input_dir, replicate_folder_name) if replicate_folder_name else input_dir
        current_replicate_id = replicate_folder_name if replicate_folder_name else f"rep{rep_idx}"
        try:
            dta_files = sorted([f for f in os.listdir(current_path) if f.lower().endswith('.dta') and os.path.isfile(os.path.join(current_path, f))])
        except FileNotFoundError: logging.warning(f"Path not found: {current_path}. Skipping."); continue

        for filename in dta_files:
            if stop_event.is_set(): return "Analysis Cancelled During File Loop (A1)"
            filepath = os.path.join(current_path, filename)
            logging.debug(f"A1: Processing {filepath}")
            try:
                day_val = int(os.path.basename(filename).split('.')[0])
                data_eis_list = deareis.parse_data(filepath)
                if not data_eis_list: logging.warning(f"A1: Could not parse {filepath}"); continue
                
                eis_obj = data_eis_list[0]
                bode = eis_obj.get_bode_data(); nyq = eis_obj.get_nyquist_data()
                if bode is None or nyq is None or not bode[0].size: logging.warning(f"A1: No Bode/Nyquist in {filepath}"); continue

                file_data_row = {'day': day_val, 'replicate_id': current_replicate_id}
                freqs_hz, Zmod_vals, negZphz_vals = bode[0], bode[1], bode[2]
                Zreal_vals, negZimg_vals = nyq[0], nyq[1]

                if not processed_one_file_for_cols or common_freqs_hz_a1.size == 0 : 
                    common_freqs_hz_a1 = freqs_hz.copy() # Capture common frequencies

                Creal_calc, Cimg_calc, Cmod_calc = [], [], []
                for i, f_hz in enumerate(freqs_hz):
                    Cr_val, Ci_val, Cm_val = np.nan, np.nan, np.nan
                    if f_hz != 0:
                        if i < len(negZimg_vals) and negZimg_vals[i] != 0: Cr_val = -1 / (2 * np.pi * f_hz * negZimg_vals[i]) 
                        if i < len(Zreal_vals) and i < len(Zmod_vals) and Zmod_vals[i] != 0: Ci_val = Zreal_vals[i] / (2 * np.pi * f_hz * Zmod_vals[i]**2)
                        if not (np.isnan(Cr_val) or np.isnan(Ci_val)): Cm_val = abs(complex(Cr_val, Ci_val))
                    Creal_calc.append(Cr_val); Cimg_calc.append(Ci_val); Cmod_calc.append(Cm_val)
                
                all_param_series = {'Zmod': Zmod_vals, '-Zphz': negZphz_vals, 'Zreal': Zreal_vals, 
                                    'Zimg': negZimg_vals, 'Creal': np.array(Creal_calc), 
                                    'Cimg': np.array(Cimg_calc), 'Cmod': np.array(Cmod_calc)}

                if not processed_one_file_for_cols and common_freqs_hz_a1.size > 0:
                    temp_dynamic_cols = []
                    for param_name in ['Zmod', '-Zphz', 'Zreal', 'Zimg', 'Creal', 'Cimg', 'Cmod']:
                        for freq_val in common_freqs_hz_a1: # Use common freqs
                            freq_str = f"{freq_val:.2E}" if freq_val > 1000 or freq_val < 0.01 else f"{freq_val:.3f}"
                            temp_dynamic_cols.append(f"{param_name}_{freq_str}Hz")
                    dynamic_param_column_names = temp_dynamic_cols
                    processed_one_file_for_cols = True
                
                # Populate file_data_row using dynamic_param_column_names and common_freqs_hz_a1
                for param_key_ordered in ['Zmod', '-Zphz', 'Zreal', 'Zimg', 'Creal', 'Cimg', 'Cmod']:
                    param_values_for_key = all_param_series[param_key_ordered]
                    for freq_idx_common, freq_val_common in enumerate(common_freqs_hz_a1):
                        freq_str_common = f"{freq_val_common:.2E}" if freq_val_common > 1000 or freq_val_common < 0.01 else f"{freq_val_common:.3f}"
                        expected_col_name = f"{param_key_ordered}_{freq_str_common}Hz"
                        # Find corresponding value in current file's data by matching frequency
                        current_file_freq_idx = np.where(np.isclose(freqs_hz, freq_val_common))[0]
                        if current_file_freq_idx.size > 0:
                            actual_idx_in_file = current_file_freq_idx[0]
                            if actual_idx_in_file < len(param_values_for_key):
                                 file_data_row[expected_col_name] = param_values_for_key[actual_idx_in_file]
                            else: file_data_row[expected_col_name] = np.nan # Should not happen if freqs_hz matches param_values_for_key length
                        else: file_data_row[expected_col_name] = np.nan 
                all_files_data_list.append(file_data_row)
            except Exception as e: logging.error(f"A1 file processing error for {filepath}: {e}", exc_info=True)

    if not all_files_data_list: return "Error (A1): No data files could be processed."
    
    master_cols = ['day', 'replicate_id'] + (dynamic_param_column_names if processed_one_file_for_cols else [])
    raw_data_df = pd.DataFrame(all_files_data_list)
    for col in master_cols: 
        if col not in raw_data_df.columns: raw_data_df[col] = np.nan
    raw_data_df = raw_data_df[master_cols]

    processed_df_for_means = raw_data_df.copy() 
    processed_df_for_std = raw_data_df.copy()   

    if apply_day0_norm_global:
        logging.info("A1: Applying global Day0 normalization...")
        processed_df_for_means = apply_global_day0_normalization(processed_df_for_means, N_replicates, 'day', 'replicate_id', stop_event)
        processed_df_for_std = apply_global_day0_normalization(processed_df_for_std, N_replicates, 'day', 'replicate_id', stop_event)
        if stop_event.is_set() or processed_df_for_means.empty: return "Cancelled or Empty after Day0 (A1)"

    data_cols = [col for col in raw_data_df.columns if col not in ['day', 'replicate_id']]
    dfs_for_main_heatmaps = {} 
    
    if average_replicates_global:
        if N_replicates > 0 and not processed_df_for_means.empty and data_cols:
            df_for_avg = processed_df_for_means.sort_values(by=['day', 'replicate_id']).reset_index(drop=True)
            if len(df_for_avg) >= N_replicates: 
                avg_num = df_for_avg[data_cols].groupby(np.arange(len(df_for_avg)) // N_replicates).mean()
                avg_day = df_for_avg[['day']].groupby(np.arange(len(df_for_avg)) // N_replicates).first()
                dfs_for_main_heatmaps["averaged"] = pd.concat([avg_day, avg_num], axis=1).set_index('day')
            else: dfs_for_main_heatmaps["averaged"] = processed_df_for_means.groupby('day')[data_cols].mean()
            logging.info("Applied global replicate averaging for A1.")
        else: # Fallback even if averaging is true but conditions not met
            dfs_for_main_heatmaps["averaged"] = processed_df_for_means.groupby('day')[data_cols].mean() if 'day' in processed_df_for_means and data_cols else pd.DataFrame()
    else: # Per-replicate output for main heatmaps
        if 'replicate_id' in processed_df_for_means.columns:
            for rep_id, group_df in processed_df_for_means.groupby('replicate_id'):
                if stop_event.is_set(): return "Cancelled during A1 per-replicate split"
                if not group_df.empty and 'day' in group_df.columns:
                    unique_day_group = group_df.groupby('day')[data_cols].mean() if group_df.duplicated(subset=['day']).any() else group_df.set_index('day')[data_cols]
                    dfs_for_main_heatmaps[rep_id] = unique_day_group
    
    if stop_event.is_set(): return "Analysis Cancelled (A1)"
    
    # --- Standard Deviation Heatmap ---
    if N_replicates > 1 and 'replicate_id' in processed_df_for_std.columns and not processed_df_for_std.empty and data_cols:
        df_for_std_calc = processed_df_for_std.sort_values(by=['day', 'replicate_id']).reset_index(drop=True)
        if len(df_for_std_calc) >= N_replicates:
            std_num = df_for_std_calc[data_cols].groupby(np.arange(len(df_for_std_calc))//N_replicates).std()
            std_day = df_for_std_calc[['day']].groupby(np.arange(len(df_for_std_calc))//N_replicates).first()
            replicate_std_devs_df = pd.concat([std_day, std_num], axis=1).set_index('day')
            if not replicate_std_devs_df.empty:
                std_dev_records = []
                for col_name_std in data_cols:
                    try:
                        param_t_std, freq_s_hz_std = col_name_std.rsplit('_',1)
                        mean_std_for_param_freq = replicate_std_devs_df[col_name_std].mean() 
                        std_dev_records.append({'param_type': param_t_std, 'freq_str': freq_s_hz_std.replace('Hz',''), 'mean_std_dev': mean_std_for_param_freq})
                    except: continue
                if std_dev_records:
                    std_pivot_df = pd.DataFrame(std_dev_records).groupby(['freq_str', 'param_type'])['mean_std_dev'].first().unstack()
                    try: std_pivot_df.index = sorted(std_pivot_df.index, key=lambda x: float(str(x).split('E')[0]) * (10**int(str(x).split('E')[1])) if 'E' in str(x) else float(str(x)), reverse=True)
                    except: logging.warning("A1: Could not sort StDev heatmap freqs numerically.")
                    
                    std_dev_heatmap_to_plot = std_pivot_df
                    if a1_norm_stddev_heatmap_global:
                        std_dev_heatmap_to_plot = normalize_heatmap_data(std_pivot_df, heatmap_norm_strategy, 
                                                                       param_name_for_title="A1_StdDevs", stop_event=stop_event)
                    if not std_dev_heatmap_to_plot.empty:
                        plt.figure(figsize=(10, max(6, len(std_dev_heatmap_to_plot.index)*0.3)))
                        title_std_auto = f"Avg Replicate StdDev (Norm: {heatmap_norm_strategy if a1_norm_stddev_heatmap_global else 'Raw'}) - A1"
                        title_std = format_plot_title(custom_title_prefix, title_std_auto, include_auto_titles)
                        xlabel_std = custom_x_axis_title if custom_x_axis_title else "Parameter Type"
                        sns.heatmap(std_dev_heatmap_to_plot.fillna(0), cmap="viridis", fmt=".2E", annot=False, norm=LogNorm() if std_dev_heatmap_to_plot.min().min() > 0 else None)
                        plt.title(title_std); plt.xlabel(xlabel_std); plt.ylabel("Frequency (Hz)"); plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0); plt.tight_layout()
                        fname_std_suffix = sanitize_filename(custom_title_prefix) if custom_title_prefix else ""
                        plt.savefig(os.path.join(output_dir, f"A1_avg_replicate_std_dev_heatmap{'_' if fname_std_suffix else ''}{fname_std_suffix}.png")); plt.close()
                        logging.info("Saved A1 Avg Replicate StdDev heatmap.")
    if stop_event.is_set(): return "Analysis Cancelled (A1)"

    # --- Generate Main Heatmaps ---
    if not dfs_for_main_heatmaps: logging.warning("A1: No data for main heatmaps."); return "A1 Completed (No main heatmap data)."

    unique_param_types_a1 = ['Zmod', '-Zphz', 'Zreal', 'Zimg', 'Creal', 'Cimg', 'Cmod']
    freq_labels_for_heatmap_y_a1 = [f"{f:.2E}" if f > 1000 or f < 0.01 else f"{f:.3f}" for f in common_freqs_hz_a1] if common_freqs_hz_a1.size > 0 else []

    for replicate_key, data_for_heatmap_current_rep_or_avg in dfs_for_main_heatmaps.items():
        if stop_event.is_set(): return f"Cancelled before heatmaps for {replicate_key} (A1)"
        if data_for_heatmap_current_rep_or_avg.empty: continue
        
        title_prefix_for_file = sanitize_filename(custom_title_prefix)
        csv_filename_suffix = f"_{title_prefix_for_file}" if title_prefix_for_file else ""
        csv_filename_suffix += f"_rep_{replicate_key}" if replicate_key != "averaged" else "_averaged"
        csv_path = os.path.join(output_dir, f"A1_main_heatmap_data{csv_filename_suffix}.csv")
        data_for_heatmap_current_rep_or_avg.to_csv(csv_path)
        logging.info(f"Saved A1 data for heatmap to {csv_path}")

        for param_type in unique_param_types_a1:
            if stop_event.is_set(): return f"Cancelled plotting {param_type} for {replicate_key} (A1)"
            cols_for_this_param = [col for col in data_for_heatmap_current_rep_or_avg.columns if col.startswith(param_type + "_")]
            if not cols_for_this_param: continue
            heatmap_actual_data = data_for_heatmap_current_rep_or_avg[cols_for_this_param]
            
            sorted_cols_for_param, current_param_freq_floats_for_norm = [], []
            if common_freqs_hz_a1.size > 0:
                for f_val in common_freqs_hz_a1:
                    f_str = f"{f_val:.2E}" if f_val > 1000 or f_val < 0.01 else f"{f_val:.3f}"
                    expected_col = f"{param_type}_{f_str}Hz"
                    if expected_col in heatmap_actual_data.columns:
                        sorted_cols_for_param.append(expected_col); current_param_freq_floats_for_norm.append(f_val)
                if sorted_cols_for_param: heatmap_actual_data = heatmap_actual_data[sorted_cols_for_param]
            
            normalized_data = normalize_heatmap_data(heatmap_actual_data, heatmap_norm_strategy, 
                                                     frequency_values=current_param_freq_floats_for_norm if current_param_freq_floats_for_norm else common_freqs_hz_a1, 
                                                     param_name_for_title=f"{param_type} ({replicate_key})", stop_event=stop_event)
            if stop_event.is_set() or normalized_data.empty: continue

            plt.figure(figsize=(10, max(6, len(common_freqs_hz_a1)*0.25 if common_freqs_hz_a1.size > 0 else 6)))
            auto_title_suffix_main = f'{param_type} ({replicate_key}, Norm: {heatmap_norm_strategy}) - A1'
            title_main = format_plot_title(custom_title_prefix, auto_title_suffix_main, include_auto_titles)
            xlabel_main = custom_x_axis_title if custom_x_axis_title else "Day"
            
            actual_x_tick_labels_data = normalized_data.index # These are the 'day' values
            x_ticks_positions, x_tick_labels_to_display = get_spaced_ticks_and_labels(len(actual_x_tick_labels_data), custom_x_labels, default_label_prefix="")
            if x_tick_labels_to_display is None: # Use actual day values if no custom ones
                 x_tick_labels_to_display = [str(int(d)) for d in actual_x_tick_labels_data[x_ticks_positions]]


            y_labels_for_plot = freq_labels_for_heatmap_y_a1 if common_freqs_hz_a1.size == len(normalized_data.columns) else [col.split('_')[-1] for col in normalized_data.columns] # Fallback if mismatch

            ax = sns.heatmap(normalized_data.T, cmap="Spectral", fmt=".2f", yticklabels=y_labels_for_plot, xticklabels=False) # xticklabels set below
            if x_ticks_positions.size > 0 :
                ax.set_xticks(x_ticks_positions + 0.5) 
                ax.set_xticklabels(x_tick_labels_to_display, rotation=45, ha="right")
            
            plt.title(title_main); plt.xlabel(xlabel_main); plt.ylabel('Frequency (Hz)'); plt.yticks(rotation=0); plt.tight_layout()
            filename_suffix_hm = f"_{title_prefix_for_file}" if title_prefix_for_file else ""
            filename_suffix_hm += f"_rep_{replicate_key}" if replicate_key != "averaged" else "_averaged"
            plt.savefig(os.path.join(output_dir, f"A1_heatmap_{param_type.replace('/','_')}{filename_suffix_hm}.png")); plt.close()
            logging.info(f"Saved A1 heatmap for {param_type} ({replicate_key}).")

    return "Analysis 1 (Raw Vars Heatmaps) completed."

# ---------------------------------------------------------
# --- ANALYSIS FUNCTION 2: DRT ---
# ---------------------------------------------------------
def run_analysis_drt(input_dir, output_dir, N_replicates_from_gui,
                     apply_day0_norm_global, average_replicates_global, heatmap_norm_strategy,
                     custom_title_prefix, include_auto_titles, custom_x_labels, custom_x_axis_title, # New global
                     drt_method_enum, drt_mode_enum, lambda_values_list,
                     rbf_type_enum, rbf_shape_enum, rbf_size_float,
                     fit_penalty_int, include_inductance_bool, num_attempts_int,
                     mrq_fit_cdc_string, num_drt_procs, stop_event): # mrq_fit_cdc_string is for the prelim fit
    logging.info("Starting Analysis Type 2 (DRT Analysis)...")
    if stop_event.is_set(): return "DRT Cancelled at Start"
    N_replicates = detect_num_replicates(input_dir, N_replicates_from_gui)
    # ... (Initial checks for input_dir, output_dir)
    if not os.path.isdir(input_dir): return "Error (A2): Input directory not found."
    if not os.path.isdir(output_dir): os.makedirs(output_dir, exist_ok=True)

    first_file_processed_for_taus_overall = False; common_tau_values_overall = np.array([]) 
    replicate_folders = sorted([d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]);
    if not replicate_folders: replicate_folders = [None]

    for lambda_val_idx, lambda_val in enumerate(lambda_values_list):
        # ... (DRT settings setup, prelim_ecm_fit calculation, all_files_drt_data_current_lambda population)
        # ... (This inner part is complex and needs the full logic from previous working version)
        if stop_event.is_set(): return f"DRT Cancelled before lambda {lambda_val}"
        logging.info(f"Processing DRT for Lambda ({lambda_val_idx+1}/{len(lambda_values_list)}): {lambda_val}")
        
        all_files_drt_data_current_lambda = [] 
        processed_a_file_this_lambda = False
        for rep_idx, replicate_folder_name in enumerate(replicate_folders):
            # ... (file iteration logic for this replicate folder, as in previous complete DRT function)
            if stop_event.is_set(): break # Break from replicate loop
            current_path = os.path.join(input_dir, replicate_folder_name) if replicate_folder_name else input_dir
            current_replicate_id = replicate_folder_name if replicate_folder_name else f"rep{rep_idx}"
            try:
                files_in_path = sorted([f for f in os.listdir(current_path) if f.lower().endswith('.dta') and os.path.isfile(os.path.join(current_path, f))])
            except FileNotFoundError: logging.warning(f"Path not found for DRT: {current_path}"); continue

            for filename in files_in_path:
                if stop_event.is_set(): break # Break from file loop
                filepath = os.path.join(current_path, filename)
                try:
                    day_val = int(os.path.basename(filename).split('.')[0])
                    eis_data_list = deareis.parse_data(filepath)
                    if not eis_data_list: continue
                    
                    # Preliminary ECM fit (user's fix for DRT fit requirement)
                    prelim_fit_settings = deareis.FitSettings(mrq_fit_cdc_string or "R", method='AUTO', weight='AUTO', max_nfev=10000)
                    prelim_ecm_fit_obj = deareis.fit_circuit(eis_data_list[0], prelim_fit_settings)
                    
                    drt_settings_dict = {
                        'method': drt_method_enum, 'mode': drt_mode_enum, 'lambda_value': float(lambda_val),
                        'rbf_type': rbf_type_enum, 'derivative_order': fit_penalty_int, 'rbf_shape': rbf_shape_enum,
                        'shape_coeff': rbf_size_float, 'inductance': include_inductance_bool, 'num_attempts': num_attempts_int,
                        'credible_intervals': False, 'timeout': 120, 'num_samples': 100, 'maximum_symmetry': 0.5,
                        'fit': prelim_ecm_fit_obj, # Pass the calculated fit
                        'gaussian_width': 0.5, 'num_per_decade': 10 # Numeric
                    }
                    drt_current_settings = deareis.DRTSettings(**drt_settings_dict)
                    drt_results = deareis.calculate_drt(eis_data_list[0], drt_current_settings, num_procs=num_drt_procs)
                    if not drt_results: continue

                    gammas = drt_results.get_gammas()[0]; taus = drt_results.get_time_constants()
                    if not first_file_processed_for_taus_overall: common_tau_values_overall = np.round(taus, 6); first_file_processed_for_taus_overall = True
                    if len(gammas) != len(common_tau_values_overall) and first_file_processed_for_taus_overall: continue
                    all_files_drt_data_current_lambda.append({'day': day_val, 'replicate_id': current_replicate_id, 'gammas': gammas})
                    processed_a_file_this_lambda = True
                except Exception as e_calc: logging.error(f"A2: DRT calc error {filepath} (L {lambda_val}): {e_calc}", exc_info=True)
            if stop_event.is_set(): break # from rep_idx loop
        if stop_event.is_set() or not processed_a_file_this_lambda or not first_file_processed_for_taus_overall: continue # to next lambda

        # --- DataFrame creation, Day0, Averaging/Per-Rep, CSV, Heatmap Norm, Plotting ---
        # (This part follows the pattern established in run_analysis_1 for dfs_for_main_heatmaps)
        tau_col_names = [f"tau_{tau_val:.3E}" for tau_val in common_tau_values_overall]
        gamma_records = [{'day': r['day'], 'replicate_id': r['replicate_id'], **{tau_col_names[i]: r['gammas'][i] for i in range(len(common_tau_values_overall))}} for r in all_files_drt_data_current_lambda]
        if not gamma_records: continue
        raw_drt_df_lambda = pd.DataFrame(gamma_records)
        processed_df_lambda = raw_drt_df_lambda.copy()
        if apply_day0_norm_global:
            processed_df_lambda = apply_global_day0_normalization(processed_df_lambda, N_replicates, 'day', 'replicate_id', stop_event)
            if stop_event.is_set() or processed_df_lambda.empty: continue
        
        data_cols_drt = [col for col in processed_df_lambda.columns if col not in ['day', 'replicate_id']]
        dfs_for_drt_heatmaps_this_lambda = {}
        if average_replicates_global: # Generate one "averaged" dataset
            if N_replicates > 0 and not processed_df_lambda.empty and data_cols_drt:
                # ... (averaging logic as in A1) ...
                df_for_avg = processed_df_lambda.sort_values(by=['day', 'replicate_id']).reset_index(drop=True)
                if len(df_for_avg) >= N_replicates:
                    avg_num = df_for_avg[data_cols_drt].groupby(np.arange(len(df_for_avg)) // N_replicates).mean()
                    avg_day = df_for_avg[['day']].groupby(np.arange(len(df_for_avg)) // N_replicates).first()
                    dfs_for_drt_heatmaps_this_lambda["averaged"] = pd.concat([avg_day, avg_num], axis=1).set_index('day')
                else: dfs_for_drt_heatmaps_this_lambda["averaged"] = processed_df_lambda.groupby('day')[data_cols_drt].mean()
            else: dfs_for_drt_heatmaps_this_lambda["averaged"] = processed_df_lambda.groupby('day')[data_cols_drt].mean() if 'day' in processed_df_lambda else pd.DataFrame()
        else: # Per-replicate output
             if 'replicate_id' in processed_df_lambda.columns:
                for rep_id, group_df in processed_df_lambda.groupby('replicate_id'):
                    if not group_df.empty and 'day' in group_df.columns:
                         dfs_for_drt_heatmaps_this_lambda[rep_id] = group_df.groupby('day')[data_cols_drt].mean() # Ensure unique days per rep

        for replicate_key, data_for_heatmap in dfs_for_drt_heatmaps_this_lambda.items():
            if stop_event.is_set() or data_for_heatmap.empty: continue
            title_prefix_for_file = sanitize_filename(custom_title_prefix)
            csv_suffix = f"_L{str(lambda_val).replace('.','_')}"
            csv_suffix += f"_{title_prefix_for_file}" if title_prefix_for_file else ""
            csv_suffix += f"_rep_{replicate_key}" if replicate_key != "averaged" else "_averaged"
            data_for_heatmap.to_csv(os.path.join(output_dir, f"A2_DRT_data{csv_suffix}.csv"))
            logging.info(f"Saved A2 DRT data ({replicate_key}, L {lambda_val})")

            normalized_data = normalize_heatmap_data(data_for_heatmap, heatmap_norm_strategy, common_tau_values_overall, f"DRT (L{lambda_val},{replicate_key})", stop_event)
            if stop_event.is_set() or normalized_data.empty: continue
            
            num_taus_total = len(common_tau_values_overall); max_y_ticks = 20 
            y_tick_indices = np.linspace(0, num_taus_total - 1, min(num_taus_total,max_y_ticks), dtype=int) if num_taus_total > 0 else []
            ytick_labels_drt_subset = [f"{common_tau_values_overall[i]:.1E}" for i in y_tick_indices] if y_tick_indices.size > 0 else False

            plt.figure(figsize=(10, 8))
            auto_title_sfx_drt = f'DRT (L {lambda_val}, {replicate_key}, M {drt_mode_enum.name if hasattr(drt_mode_enum,"name") else drt_mode_enum}, N {heatmap_norm_strategy})'
            title_drt = format_plot_title(custom_title_prefix, auto_title_sfx_drt, include_auto_titles)
            xlabel_drt = custom_x_axis_title if custom_x_axis_title else "Day"
            
            x_ticks_pos, x_labels_cust = get_spaced_ticks_and_labels(len(normalized_data.index), custom_x_labels, default_label_prefix="")
            if x_labels_cust is None and x_ticks_pos.size > 0 : x_labels_cust = [str(int(d)) for d in normalized_data.index.values[x_ticks_pos]]


            ax = sns.heatmap(normalized_data.T, cmap="Spectral", fmt=".2f", xticklabels=False, yticklabels=False, norm=LogNorm() if heatmap_norm_strategy!="Raw Values" and normalized_data.values.min()>0 else None)
            if ytick_labels_drt_subset: ax.set_yticks(y_tick_indices + 0.5); ax.set_yticklabels(ytick_labels_drt_subset, rotation=0)
            if x_ticks_pos.size > 0 : ax.set_xticks(x_ticks_pos + 0.5); ax.set_xticklabels(x_labels_cust, rotation=45, ha="right")
            
            plt.title(title_drt); plt.xlabel(xlabel_drt); plt.ylabel('τ (s)'); plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"A2_drt_heatmap{csv_suffix}.png")); plt.close()
            logging.info(f"Saved DRT heatmap for L {lambda_val}, {replicate_key}")
            if stop_event.is_set(): break # from replicate_key loop
        if stop_event.is_set(): break # from lambda_val loop

    if stop_event.is_set(): return "DRT Analysis Cancelled."
    return "Analysis Type 2 (DRT Analysis) completed."


# ---------------------------------------------------------
# --- ANALYSIS FUNCTION 3: ECM Fitting --- 
# ---------------------------------------------------------
def run_analysis_ecm_fitting(input_dir, output_dir, N_replicates_from_gui, 
                             apply_day0_norm_global, average_replicates_global, heatmap_norm_strategy,
                             custom_title_prefix, include_auto_titles, custom_x_labels, custom_x_axis_title, # New global
                             ecm_strings_list, fit_method_str, fit_weight_str, fit_max_nfev_int, stop_event):
    logging.info("Starting Analysis Type 3 (ECM Fitting)...")
    if stop_event.is_set(): return "ECM Fitting Cancelled at Start"
    
    N_replicates = detect_num_replicates(input_dir, N_replicates_from_gui)
    if N_replicates == 0: N_replicates = 1 # Ensure N is at least 1 for logic below

    if not os.path.isdir(input_dir): logging.error("A3: Input directory not found."); return "Error (A3): Input directory not found."
    if not os.path.isdir(output_dir): os.makedirs(output_dir, exist_ok=True)

    replicate_folders = sorted([d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))])
    if not replicate_folders: replicate_folders = [None] # Process input_dir directly if no subfolders

    for ecm_idx, current_ecm_str in enumerate(ecm_strings_list):
        if stop_event.is_set(): return f"ECM Fitting Cancelled before ECM {current_ecm_str}"
        logging.info(f"Processing ECM ({ecm_idx+1}/{len(ecm_strings_list)}): {current_ecm_str}")
        
        all_files_ecm_data = [] # List of dicts: {'day':day, 'replicate_id':rep, 'params':{p1:v1,...}}
        parameter_names_ordered = [] 
        first_successful_fit_for_ecm = False
        
        try:
            fit_settings = deareis.FitSettings(current_ecm_str, method=fit_method_str, weight=fit_weight_str, max_nfev=fit_max_nfev_int)
        except Exception as e_fset:
            logging.error(f"A3: Error creating FitSettings for ECM '{current_ecm_str}': {e_fset}. Skipping this ECM.")
            continue


        for rep_idx, replicate_folder_name in enumerate(replicate_folders):
            if stop_event.is_set(): return f"ECM Fitting Cancelled processing replicates for {current_ecm_str}"
            current_path = os.path.join(input_dir, replicate_folder_name) if replicate_folder_name else input_dir
            current_replicate_id = replicate_folder_name if replicate_folder_name else f"rep{rep_idx}"
            try:
                files_in_path = sorted([f for f in os.listdir(current_path) if f.lower().endswith('.dta') and os.path.isfile(os.path.join(current_path, f))])
            except FileNotFoundError: logging.warning(f"Path not found for ECM: {current_path}"); continue

            for filename in files_in_path:
                if stop_event.is_set(): return f"ECM Fitting Cancelled processing files for {current_ecm_str}"
                filepath = os.path.join(current_path, filename)
                logging.debug(f"A3: Fitting {current_ecm_str} to {filepath}")
                try:
                    day_val = int(os.path.basename(filename).split('.')[0])
                    eis_data_list = deareis.parse_data(filepath)
                    if not eis_data_list: logging.warning(f"A3: Could not parse {filepath}"); continue
                    
                    fit_results = deareis.fit_circuit(eis_data_list[0], fit_settings)
                    current_params_dict_this_file = {}
                    temp_param_names_this_fit = []

                    if fit_results and fit_results.parameters:
                        for comp_name in sorted(fit_results.parameters.keys()):
                            for param_name_fit in sorted(fit_results.parameters[comp_name].keys()):
                                full_param_key = f"{comp_name}_{param_name_fit}"
                                current_params_dict_this_file[full_param_key] = fit_results.parameters[comp_name][param_name_fit].value
                                if not first_successful_fit_for_ecm: # Capture param names from first success
                                    temp_param_names_this_fit.append(full_param_key)
                        
                        if not first_successful_fit_for_ecm and temp_param_names_this_fit: 
                            parameter_names_ordered = temp_param_names_this_fit
                            first_successful_fit_for_ecm = True
                        
                        # Ensure current file's data matches the established parameter order
                        params_for_df_row = {p_name: current_params_dict_this_file.get(p_name, np.nan) for p_name in parameter_names_ordered} if first_successful_fit_for_ecm else current_params_dict_this_file
                        all_files_ecm_data.append({'day': day_val, 'replicate_id': current_replicate_id, 'params': params_for_df_row})
                    
                    elif first_successful_fit_for_ecm : # Fit failed but we know param names
                        all_files_ecm_data.append({'day': day_val, 'replicate_id': current_replicate_id, 'params': {p_name: np.nan for p_name in parameter_names_ordered}})
                    else: # Fit failed and no param names known yet for this ECM
                         logging.warning(f"A3: Fit failed for {filepath} (ECM {current_ecm_str}) and no parameters known yet.")
                except Exception as e_fit_single:
                    logging.error(f"A3: Error fitting ECM {current_ecm_str} to {filepath}: {e_fit_single}", exc_info=False) # Set exc_info=True for more detail
                    if first_successful_fit_for_ecm: # If param names are known, add a row of NaNs
                        all_files_ecm_data.append({'day': day_val, 'replicate_id': current_replicate_id if 'current_replicate_id' in locals() else f"unknown_rep_at_error", 'params': {p_name: np.nan for p_name in parameter_names_ordered}})
        
        if stop_event.is_set(): return f"ECM Fitting Cancelled after file processing for {current_ecm_str}"
        if not all_files_ecm_data or not first_successful_fit_for_ecm:
            logging.warning(f"A3: No ECM data or parameters determined for ECM {current_ecm_str}. Skipping this ECM.")
            continue
        
        # Create DataFrame from the list of dictionaries
        records_for_df = []
        for record in all_files_ecm_data:
            row_data = {'day':record['day'], 'replicate_id':record['replicate_id']}
            # Ensure all parameters from parameter_names_ordered are present, even if NaN
            for p_name in parameter_names_ordered:
                row_data[p_name] = record['params'].get(p_name, np.nan)
            records_for_df.append(row_data)

        if not records_for_df: logging.warning(f"A3: No records to form DataFrame for ECM {current_ecm_str}"); continue
            
        raw_params_df_ecm = pd.DataFrame(records_for_df)
        # Ensure correct column order based on parameter_names_ordered
        cols_to_select_ecm = ['day', 'replicate_id'] + parameter_names_ordered
        # Filter out any unexpected columns that might have crept in, and ensure order
        raw_params_df_ecm = raw_params_df_ecm[[col for col in cols_to_select_ecm if col in raw_params_df_ecm.columns]]


        processed_df_ecm = raw_params_df_ecm.copy()
        if apply_day0_norm_global:
            processed_df_ecm = apply_global_day0_normalization(processed_df_ecm, N_replicates, 'day', 'replicate_id', stop_event)
            if stop_event.is_set() or processed_df_ecm.empty : logging.warning(f"A3: ECM data empty/cancelled after Day0 for {current_ecm_str}."); continue
        
        data_cols_ecm = [col for col in processed_df_ecm.columns if col not in ['day', 'replicate_id']]
        dfs_for_ecm_heatmaps = {} # Key: replicate_id or "averaged"

        if average_replicates_global:
            if N_replicates > 0 and not processed_df_ecm.empty and data_cols_ecm:
                df_for_avg = processed_df_ecm.sort_values(by=['day', 'replicate_id']).reset_index(drop=True)
                if len(df_for_avg) >= N_replicates: # Check N_replicates actually makes sense
                    avg_data_num = df_for_avg[data_cols_ecm].groupby(np.arange(len(df_for_avg)) // N_replicates).mean()
                    avg_data_day = df_for_avg[['day']].groupby(np.arange(len(df_for_avg)) // N_replicates).first()
                    dfs_for_ecm_heatmaps["averaged"] = pd.concat([avg_data_day, avg_data_num], axis=1).set_index('day')
                else: # Fallback if N_replicates is 1 or data is insufficient for block averaging
                    dfs_for_ecm_heatmaps["averaged"] = processed_df_ecm.groupby('day')[data_cols_ecm].mean()
                logging.info(f"A3: Applied global replicate averaging for ECM {current_ecm_str}.")
            else: # Not enough data or N_reps=0 (should be caught by N_replicates=1 default), or no data columns
                dfs_for_ecm_heatmaps["averaged"] = processed_df_ecm.groupby('day')[data_cols_ecm].mean() if 'day' in processed_df_ecm and data_cols_ecm else pd.DataFrame()
        else: # Per-replicate output
            if 'replicate_id' in processed_df_ecm.columns:
                for rep_id, group_df in processed_df_ecm.groupby('replicate_id'):
                    if stop_event.is_set(): break
                    if not group_df.empty and 'day' in group_df.columns and data_cols_ecm:
                         # Average if duplicate days for a rep (should not happen if data is clean)
                        unique_day_group = group_df.groupby('day')[data_cols_ecm].mean() if group_df.duplicated(subset=['day']).any() else group_df.set_index('day')[data_cols_ecm]
                        dfs_for_ecm_heatmaps[rep_id] = unique_day_group
            if stop_event.is_set(): continue # To next ECM if cancelled here

        if stop_event.is_set() or not dfs_for_ecm_heatmaps: logging.warning(f"A3: No data for heatmaps for ECM {current_ecm_str} after processing."); continue

        for replicate_key, data_for_heatmap in dfs_for_ecm_heatmaps.items():
            if stop_event.is_set() or data_for_heatmap.empty: continue
            
            title_prefix_for_file = sanitize_filename(custom_title_prefix)
            csv_filename_suffix = f"_ECM_{sanitize_filename(current_ecm_str)}"
            csv_filename_suffix += f"_{title_prefix_for_file}" if title_prefix_for_file else ""
            csv_filename_suffix += f"_rep_{sanitize_filename(replicate_key)}" if replicate_key != "averaged" else "_averaged"
            
            csv_path = os.path.join(output_dir, f"A3_ECM_params_data{csv_filename_suffix}.csv")
            data_for_heatmap.to_csv(csv_path)
            logging.info(f"Saved A3 ECM data to {csv_path}")

            # "Frequency Sections" norm not applicable here directly, pass None for frequency_values
            normalized_data = normalize_heatmap_data(data_for_heatmap, heatmap_norm_strategy, 
                                                     frequency_values=None, 
                                                     param_name_for_title=f"ECM {current_ecm_str} ({replicate_key})", 
                                                     stop_event=stop_event)
            if stop_event.is_set() or normalized_data.empty: continue
            
            plt.figure(figsize=(12, max(6, len(parameter_names_ordered)*0.3))) # Dynamic height
            auto_title_suffix_ecm = f'ECM: {current_ecm_str} ({replicate_key}, Norm: {heatmap_norm_strategy}) - A3'
            plot_title = format_plot_title(custom_title_prefix, auto_title_suffix_ecm, include_auto_titles)
            xlabel_ecm = custom_x_axis_title if custom_x_axis_title else "Day"

            actual_x_tick_labels_data_ecm = normalized_data.index 
            x_ticks_pos_ecm, x_labels_cust_ecm = get_spaced_ticks_and_labels(len(actual_x_tick_labels_data_ecm), custom_x_labels, default_label_prefix="")
            if x_labels_cust_ecm is None and x_ticks_pos_ecm.size > 0: x_labels_cust_ecm = [str(int(d)) for d in actual_x_tick_labels_data_ecm.values[x_ticks_pos_ecm]]


            ax = sns.heatmap(normalized_data.T, cmap="Spectral", fmt=".2E", annot=False, 
                             xticklabels=False, # Set below
                             yticklabels=True) # Use DataFrame columns (parameter_names_ordered)
            if x_ticks_pos_ecm.size > 0:
                ax.set_xticks(x_ticks_pos_ecm + 0.5); ax.set_xticklabels(x_labels_cust_ecm, rotation=45, ha="right")

            plt.title(plot_title); plt.xlabel(xlabel_ecm); plt.ylabel('Circuit Parameter'); plt.yticks(rotation=0); plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"A3_ECM_heatmap{csv_filename_suffix}.png")); plt.close()
            logging.info(f"Saved ECM heatmap for {current_ecm_str} ({replicate_key}).")
            if stop_event.is_set(): break # from replicate_key loop
        if stop_event.is_set(): break # from ecm_idx loop

    if stop_event.is_set(): return "ECM Fitting Cancelled."
    return "Analysis Type 3 (ECM Fitting) completed."

# ---------------------------------------------------------
# --- ANALYSIS FUNCTION 4: Peak Tracking ---
# ---------------------------------------------------------
def run_analysis_peak_tracking(input_dir, output_dir, N_replicates_from_gui,
                               apply_day0_norm_global, average_replicates_global, heatmap_norm_strategy,
                               custom_title_prefix, include_auto_titles, custom_x_labels, custom_x_axis_title, # New global
                               min_freq_peak, max_freq_peak, # A4 specific
                               stop_event):
    logging.info("Starting Analysis Type 4 (Peak Tracking)...")
    if stop_event.is_set(): return "Peak Tracking Cancelled at Start"
    N_replicates = detect_num_replicates(input_dir, N_replicates_from_gui)
    if not os.path.isdir(input_dir): return "Error (A4): Input directory not found."
    if not os.path.isdir(output_dir): os.makedirs(output_dir, exist_ok=True)

    all_files_peak_data = [] # List of dicts for all files
    replicate_folders = sorted([d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))])
    if not replicate_folders: replicate_folders = [None]

    for rep_idx, replicate_folder_name in enumerate(replicate_folders):
        if stop_event.is_set(): return "Peak Tracking Cancelled processing replicates"
        current_path = os.path.join(input_dir, replicate_folder_name) if replicate_folder_name else input_dir
        current_replicate_id = replicate_folder_name if replicate_folder_name else f"rep{rep_idx}"
        try:
            dta_files = sorted([f for f in os.listdir(current_path) if f.lower().endswith('.dta') and os.path.isfile(os.path.join(current_path, f))])
        except FileNotFoundError: logging.warning(f"Path not found for Peak Tracking: {current_path}"); continue

        for filename in dta_files:
            if stop_event.is_set(): return "Peak Tracking Cancelled processing files"
            filepath = os.path.join(current_path, filename)
            logging.debug(f"A4: Processing {filepath}")
            try:
                day_val = int(os.path.basename(filename).split('.')[0])
                data_eis_list = deareis.parse_data(filepath)
                if not data_eis_list: logging.warning(f"A4: Could not parse {filepath}"); continue
                
                eis_obj = data_eis_list[0]
                bode = eis_obj.get_bode_data(); nyq = eis_obj.get_nyquist_data()
                if bode is None or nyq is None or not bode[0].size: logging.warning(f"A4: No Bode/Nyquist in {filepath}"); continue

                freqs_hz_orig, Zmod_vals_orig, negZphz_vals_orig = bode[0], bode[1], bode[2]
                Zreal_vals_orig = nyq[0]
                
                # Filter by frequency range for peak finding
                valid_indices = (freqs_hz_orig >= min_freq_peak) & (freqs_hz_orig <= max_freq_peak)
                if not np.any(valid_indices): 
                    logging.warning(f"A4: No data in specified frequency range ({min_freq_peak}-{max_freq_peak} Hz) for {filepath}. Skipping peak find.")
                    all_files_peak_data.append({'day': day_val, 'replicate_id': current_replicate_id, 'Freq_Zphz': np.nan, 'Val_Zphz': np.nan, 'Freq_Cimg': np.nan, 'Val_Cimg': np.nan})
                    continue

                freqs_hz, Zmod_vals, negZphz_vals = freqs_hz_orig[valid_indices], Zmod_vals_orig[valid_indices], negZphz_vals_orig[valid_indices]
                Zreal_vals = Zreal_vals_orig[valid_indices]

                c_img_calc = np.array([ (zr / (2*np.pi*f*zm**2)) if (f!=0 and zm!=0) else np.nan for zr,f,zm in zip(Zreal_vals, freqs_hz, Zmod_vals) ])
                # No Cimg cutoff factor now, using full (frequency-filtered) c_img_calc for peak finding

                file_peak_data = {'day': day_val, 'replicate_id': current_replicate_id, 'Freq_Zphz': np.nan, 'Val_Zphz': np.nan, 'Freq_Cimg': np.nan, 'Val_Cimg': np.nan}
                if len(negZphz_vals)>0 and not pd.Series(negZphz_vals).isna().all(): 
                    idx_max_phz = pd.Series(negZphz_vals).idxmax()
                    file_peak_data['Freq_Zphz'] = freqs_hz[idx_max_phz]
                    file_peak_data['Val_Zphz'] = negZphz_vals[idx_max_phz]
                
                if len(c_img_calc)>0 and not pd.Series(np.abs(c_img_calc)).isna().all():
                    idx_max_abs_cimg = pd.Series(np.abs(c_img_calc)).idxmax()
                    file_peak_data['Freq_Cimg'] = freqs_hz[idx_max_abs_cimg] # Use freqs_hz corresponding to c_img_calc
                    file_peak_data['Val_Cimg'] = c_img_calc[idx_max_abs_cimg]
                all_files_peak_data.append(file_peak_data)
            except Exception as e: logging.error(f"A4 file processing error for {filepath}: {e}", exc_info=True)

    if stop_event.is_set() or not all_files_peak_data: return "Peak Tracking Cancelled or No data."

    raw_peak_df = pd.DataFrame(all_files_peak_data)
    processed_peak_df = raw_peak_df.copy()
    if apply_day0_norm_global:
        processed_peak_df = apply_global_day0_normalization(processed_peak_df, N_replicates, 'day', 'replicate_id', stop_event)
        if stop_event.is_set() or processed_peak_df.empty : return "Analysis Cancelled or Data Empty after Day0 Norm (A4)"

    data_cols_peak = [col for col in processed_peak_df.columns if col not in ['day', 'replicate_id']]
    dfs_for_peak_heatmaps = {}

    if average_replicates_global:
        if N_replicates > 0 and not processed_peak_df.empty and data_cols_peak:
            df_for_avg = processed_peak_df.sort_values(by=['day', 'replicate_id']).reset_index(drop=True)
            if len(df_for_avg) >= N_replicates:
                avg_num = df_for_avg[data_cols_peak].groupby(np.arange(len(df_for_avg)) // N_replicates).mean()
                avg_day = df_for_avg[['day']].groupby(np.arange(len(df_for_avg)) // N_replicates).first()
                dfs_for_peak_heatmaps["averaged"] = pd.concat([avg_day, avg_num], axis=1).set_index('day')
            else: dfs_for_peak_heatmaps["averaged"] = processed_peak_df.groupby('day')[data_cols_peak].mean()
        else: dfs_for_peak_heatmaps["averaged"] = processed_peak_df.groupby('day')[data_cols_peak].mean() if 'day' in processed_peak_df and data_cols_peak else pd.DataFrame()
    else: # Per-replicate
        if 'replicate_id' in processed_peak_df.columns:
            for rep_id, group_df in processed_peak_df.groupby('replicate_id'):
                if stop_event.is_set(): break
                if not group_df.empty and 'day' in group_df.columns and data_cols_peak:
                     dfs_for_peak_heatmaps[rep_id] = group_df.groupby('day')[data_cols_peak].mean() # Ensure unique days
        if stop_event.is_set(): return "Cancelled during A4 per-replicate split."
    
    if stop_event.is_set() or not dfs_for_peak_heatmaps : return "Analysis Cancelled or No peak data for heatmaps."

    for replicate_key, data_for_rep_or_avg in dfs_for_peak_heatmaps.items():
        if stop_event.is_set() or data_for_rep_or_avg.empty: continue
        
        phz_peak_cols_df = ['Freq_Zphz', 'Val_Zphz']
        cimg_peak_cols_df = ['Freq_Cimg', 'Val_Cimg']

        # Ensure columns exist before trying to slice
        data_phz_hm_current = data_for_rep_or_avg[[col for col in phz_peak_cols_df if col in data_for_rep_or_avg.columns]].copy()
        data_cimg_hm_current = data_for_rep_or_avg[[col for col in cimg_peak_cols_df if col in data_for_rep_or_avg.columns]].copy()
        
        if not data_phz_hm_current.empty: data_phz_hm_current.columns = ['Freq Max', 'Value']
        if not data_cimg_hm_current.empty: data_cimg_hm_current.columns = ['Freq Max', 'Value']
        
        title_prefix_for_file = sanitize_filename(custom_title_prefix)
        csv_filename_suffix = f"_{title_prefix_for_file}" if title_prefix_for_file else ""
        csv_filename_suffix += f"_rep_{sanitize_filename(replicate_key)}" if replicate_key != "averaged" else "_averaged"

        if not data_phz_hm_current.empty:
            data_phz_hm_current.to_csv(os.path.join(output_dir, f"A4_ZphzPeak_data{csv_filename_suffix}.csv"))
            logging.info(f"Saved A4 Zphz Peak data ({replicate_key})")
            norm_phz = normalize_heatmap_data(data_phz_hm_current, heatmap_norm_strategy, param_name_for_title=f"-Zphz Peaks ({replicate_key})", stop_event=stop_event)
            if not (stop_event.is_set() or norm_phz.empty):
                auto_sfx_phz = f"-Zphz Peak ({replicate_key}, Norm: {heatmap_norm_strategy})"
                plot_title_phz = format_plot_title(custom_title_prefix,auto_sfx_phz,include_auto_titles)
                xlabel_phz = custom_x_axis_title if custom_x_axis_title else "Day"
                x_ticks_pos_phz, x_labels_cust_phz = get_spaced_ticks_and_labels(len(norm_phz.index), custom_x_labels, default_label_prefix="")
                if x_labels_cust_phz is None and x_ticks_pos_phz.size > 0: x_labels_cust_phz = [str(int(d)) for d in norm_phz.index.values[x_ticks_pos_phz]]

                plt.figure(figsize=(8,5)); ax_phz = sns.heatmap(norm_phz.T, cmap="Spectral", fmt=".2f", yticklabels=True, xticklabels=False)
                if x_ticks_pos_phz.size > 0: ax_phz.set_xticks(x_ticks_pos_phz + 0.5); ax_phz.set_xticklabels(x_labels_cust_phz, rotation=45, ha="right")
                plt.title(plot_title_phz); plt.xlabel(xlabel_phz); plt.ylabel('Parameter'); plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"A4_peak_tracking_Zphz{csv_filename_suffix}.png")); plt.close()
                logging.info(f"Saved A4 -Zphz peak heatmap ({replicate_key}).")
        
        if stop_event.is_set(): break # from replicate_key loop

        if not data_cimg_hm_current.empty:
            data_cimg_hm_current.to_csv(os.path.join(output_dir, f"A4_CimgPeak_data{csv_filename_suffix}.csv"))
            logging.info(f"Saved A4 Cimg Peak data ({replicate_key})")
            norm_cimg = normalize_heatmap_data(data_cimg_hm_current, heatmap_norm_strategy, param_name_for_title=f"Cimg Peaks ({replicate_key})", stop_event=stop_event)
            if not (stop_event.is_set() or norm_cimg.empty):
                auto_sfx_cimg = f"Cimg Peak ({replicate_key}, Norm: {heatmap_norm_strategy})"
                plot_title_cimg = format_plot_title(custom_title_prefix,auto_sfx_cimg,include_auto_titles)
                xlabel_cimg = custom_x_axis_title if custom_x_axis_title else "Day"
                x_ticks_pos_cimg, x_labels_cust_cimg = get_spaced_ticks_and_labels(len(norm_cimg.index), custom_x_labels, default_label_prefix="")
                if x_labels_cust_cimg is None and x_ticks_pos_cimg.size > 0: x_labels_cust_cimg = [str(int(d)) for d in norm_cimg.index.values[x_ticks_pos_cimg]]
                
                plt.figure(figsize=(8,5)); ax_cimg = sns.heatmap(norm_cimg.T, cmap="Spectral", fmt=".2f", yticklabels=True, xticklabels=False)
                if x_ticks_pos_cimg.size > 0: ax_cimg.set_xticks(x_ticks_pos_cimg + 0.5); ax_cimg.set_xticklabels(x_labels_cust_cimg, rotation=45, ha="right")
                plt.title(plot_title_cimg); plt.xlabel(xlabel_cimg); plt.ylabel('Parameter'); plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"A4_peak_tracking_Cimg{csv_filename_suffix}.png")); plt.close()
                logging.info(f"Saved A4 Cimg peak heatmap ({replicate_key}).")
        if stop_event.is_set(): break # from replicate_key loop

    if stop_event.is_set(): return "Peak Tracking Cancelled."
    return "Analysis Type 4 (Peak Tracking) completed."


# ---------------------------------------------------------
# --- Tkinter GUI Application ---
# ---------------------------------------------------------
class TextHandler(logging.Handler):
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
        master.title("EIS Data Analyzer Suite v2.3")
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
        current_row = 0

        general_frame = ttk.LabelFrame(main_frame, text="General Settings", padding=10)
        general_frame.grid(row=current_row, column=0, columnspan=3, sticky=tk.EW, pady=5, padx=5); current_row += 1
        ttk.Label(general_frame, text="Input Dir:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2); self.input_dir_var = tk.StringVar(); ttk.Entry(general_frame, textvariable=self.input_dir_var).grid(row=0, column=1, sticky=tk.EW, pady=2); ttk.Button(general_frame, text="Browse...", command=self.browse_input_dir).grid(row=0, column=2, padx=2, pady=2)
        ttk.Label(general_frame, text="Output Dir:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2); self.output_dir_var = tk.StringVar(); ttk.Entry(general_frame, textvariable=self.output_dir_var).grid(row=1, column=1, sticky=tk.EW, pady=2); ttk.Button(general_frame, text="Browse...", command=self.browse_output_dir).grid(row=1, column=2, padx=2, pady=2)
        ttk.Label(general_frame, text="Replicates:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2); self.replicates_gui_var = tk.IntVar(value=3); ttk.Spinbox(general_frame, from_=1, to=100, textvariable=self.replicates_gui_var, width=10).grid(row=2, column=1, sticky=tk.W, pady=2)
        self.apply_day0_global_var = tk.BooleanVar(value=False); ttk.Checkbutton(general_frame, text="Apply Day 0 Normalization", variable=self.apply_day0_global_var).grid(row=3, column=0, sticky=tk.W, padx=5, pady=2) # columnspan removed
        self.average_replicates_global_var = tk.BooleanVar(value=True); ttk.Checkbutton(general_frame, text="Average Replicates", variable=self.average_replicates_global_var).grid(row=3, column=1, sticky=tk.W, padx=5, pady=2) # Next to Day0
        ttk.Label(general_frame, text="Heatmap Norm Strategy:").grid(row=4, column=0, sticky=tk.W, padx=5, pady=2); self.heatmap_norm_strategy_var = tk.StringVar(); self.heatmap_norm_options = ["Raw Values", "Per Parameter/Timeline (Column-wise)", "Global Max Scaling", "Frequency Domain (Low/Mid/High)"]
        self.heatmap_norm_combo = ttk.Combobox(general_frame, textvariable=self.heatmap_norm_strategy_var, values=self.heatmap_norm_options, state="readonly"); self.heatmap_norm_combo.grid(row=4, column=1, columnspan=2, sticky=tk.EW, pady=2); self.heatmap_norm_combo.set(self.heatmap_norm_options[1])
        ttk.Label(general_frame, text="Custom Plot Title Prefix:").grid(row=5, column=0, sticky=tk.W, padx=5, pady=2); self.custom_title_prefix_var = tk.StringVar(); ttk.Entry(general_frame, textvariable=self.custom_title_prefix_var).grid(row=5, column=1, columnspan=2, sticky=tk.EW, pady=2)
        self.include_auto_titles_var = tk.BooleanVar(value=True); ttk.Checkbutton(general_frame, text="Include Auto-gen Suffix in Titles", variable=self.include_auto_titles_var).grid(row=6, column=0, columnspan=2, sticky=tk.W, padx=5, pady=2)
        ttk.Label(general_frame, text="Custom X-axis Labels (CSV):").grid(row=7, column=0, sticky=tk.W, padx=5, pady=2); self.custom_x_labels_var = tk.StringVar(); ttk.Entry(general_frame, textvariable=self.custom_x_labels_var).grid(row=7, column=1, columnspan=2, sticky=tk.EW, pady=2)
        ttk.Label(general_frame, text="Custom X-axis Title:").grid(row=8, column=0, sticky=tk.W, padx=5, pady=2); self.custom_x_axis_title_var = tk.StringVar(); ttk.Entry(general_frame, textvariable=self.custom_x_axis_title_var).grid(row=8, column=1, columnspan=2, sticky=tk.EW, pady=2)
        general_frame.columnconfigure(1, weight=1)

        analysis_select_frame = ttk.LabelFrame(main_frame, text="Select Analysis Type", padding=10)
        analysis_select_frame.grid(row=current_row, column=0, columnspan=3, sticky=tk.EW, pady=5, padx=5); current_row += 1
        self.analysis_type_var = tk.IntVar(value=1)
        analysis_options = [("Raw Vars Heatmaps (A1)", 1), ("DRT Analysis (A2)", 2), ("ECM Fitting (A3)", 3), ("Peak Tracking (A4)", 4)]
        for i, (text, val) in enumerate(analysis_options): ttk.Radiobutton(analysis_select_frame, text=text, variable=self.analysis_type_var, value=val, command=self.toggle_parameter_frames).grid(row=i//2, column=i%2, sticky=tk.W, padx=5, pady=2)

        self.param_frames_container_row = current_row; current_row +=1 
        self.param_frames = {}
        self.param_frames[1] = ttk.LabelFrame(main_frame, text="Analysis 1: Raw Variables Settings", padding=10)
        self.a1_norm_stddev_var = tk.BooleanVar(value=False) 
        ttk.Checkbutton(self.param_frames[1], text="Apply Normalization to Standard Deviation", variable=self.a1_norm_stddev_var).pack(padx=5, pady=5, fill=tk.X)

        self.param_frames[2] = ttk.LabelFrame(main_frame, text="Analysis 2: DRT Parameters", padding=10)
        drt_mappings = get_drt_enum_mapping()
        ttk.Label(self.param_frames[2], text="DRT Method:").grid(row=0, column=0, sticky=tk.W); self.drt_method_var = tk.StringVar(); self.drt_method_options = list(drt_mappings["DRTMethod"].keys())
        self.drt_method_combo = ttk.Combobox(self.param_frames[2], textvariable=self.drt_method_var, values=self.drt_method_options, state="readonly"); self.drt_method_combo.grid(row=0, column=1, sticky=tk.W); self.drt_method_combo.set(self.drt_method_options[0] if self.drt_method_options else "")
        ttk.Label(self.param_frames[2], text="DRT Mode:").grid(row=0, column=2, sticky=tk.W, padx=5); self.drt_mode_var = tk.StringVar(); self.drt_mode_options = list(drt_mappings["DRTMode"].keys())
        self.drt_mode_combo = ttk.Combobox(self.param_frames[2], textvariable=self.drt_mode_var, values=self.drt_mode_options, state="readonly"); self.drt_mode_combo.grid(row=0, column=3, sticky=tk.W); self.drt_mode_combo.set(self.drt_mode_options[0] if self.drt_mode_options else "")
        ttk.Label(self.param_frames[2], text="Lambda Values (CSV):").grid(row=1, column=0, sticky=tk.W); self.lambda_values_var = tk.StringVar(value="0.1,0.01,0.001")
        ttk.Entry(self.param_frames[2], textvariable=self.lambda_values_var, width=20).grid(row=1, column=1, columnspan=3, sticky=tk.EW)
        ttk.Label(self.param_frames[2], text="RBF Type:").grid(row=2, column=0, sticky=tk.W); self.rbf_type_var = tk.StringVar(); self.rbf_type_options = list(drt_mappings["RBFType"].keys())
        self.rbf_type_combo = ttk.Combobox(self.param_frames[2], textvariable=self.rbf_type_var, values=self.rbf_type_options, state="readonly"); self.rbf_type_combo.grid(row=2, column=1, sticky=tk.W); self.rbf_type_combo.set("Gaussian")
        ttk.Label(self.param_frames[2], text="RBF Shape:").grid(row=2, column=2, sticky=tk.W, padx=5); self.rbf_shape_var = tk.StringVar(); self.rbf_shape_options = list(drt_mappings["RBFShape"].keys())
        self.rbf_shape_combo = ttk.Combobox(self.param_frames[2], textvariable=self.rbf_shape_var, values=self.rbf_shape_options, state="readonly"); self.rbf_shape_combo.grid(row=2, column=3, sticky=tk.W); self.rbf_shape_combo.set("FWHM")
        ttk.Label(self.param_frames[2], text="RBF Size:").grid(row=3, column=0, sticky=tk.W); self.rbf_size_var = tk.DoubleVar(value=0.5)
        ttk.Entry(self.param_frames[2], textvariable=self.rbf_size_var, width=10).grid(row=3, column=1, sticky=tk.W)
        ttk.Label(self.param_frames[2], text="Fit Penalty (Derivative):").grid(row=3, column=2, sticky=tk.W, padx=5); self.fit_penalty_drt_var = tk.IntVar(value=1)
        ttk.Spinbox(self.param_frames[2], from_=0, to=5, textvariable=self.fit_penalty_drt_var, width=10).grid(row=3, column=3, sticky=tk.W)
        ttk.Label(self.param_frames[2], text="Num. DRT Attempts:").grid(row=4, column=0, sticky=tk.W); self.num_attempts_drt_var = tk.IntVar(value=100)
        ttk.Entry(self.param_frames[2], textvariable=self.num_attempts_drt_var, width=10).grid(row=4, column=1, sticky=tk.W)
        ttk.Label(self.param_frames[2], text="ECM Str for MRQ:").grid(row=4, column=2, sticky=tk.W, padx=5); self.mrq_cdc_drt_var = tk.StringVar(value="R")
        ttk.Entry(self.param_frames[2], textvariable=self.mrq_cdc_drt_var, width=20).grid(row=4, column=3, sticky=tk.EW)
        self.include_inductance_drt_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.param_frames[2], text="Include Inductance", variable=self.include_inductance_drt_var).grid(row=5, column=0, columnspan=2, sticky=tk.W)
        self.param_frames[2].columnconfigure(1, weight=1); self.param_frames[2].columnconfigure(3, weight=1)

        self.param_frames[3] = ttk.LabelFrame(main_frame, text="Analysis 3: ECM Fitting Parameters", padding=10)
        ttk.Label(self.param_frames[3], text="ECMs (CSV):").grid(row=0, column=0, sticky=tk.W); self.ecm_list_a3_var = tk.StringVar(value='R(Q[RW]),(RC)')
        ttk.Entry(self.param_frames[3], textvariable=self.ecm_list_a3_var, width=35).grid(row=0, column=1, columnspan=3, sticky=tk.EW)
        ttk.Label(self.param_frames[3], text="Fit Method:").grid(row=1, column=0, sticky=tk.W); self.ecm_fit_method_a3_var = tk.StringVar(); self.ecm_fit_method_options_a3 = ["AUTO", "Nelder-Mead", "L-BFGS-B", "SLSQP", "Powell"]
        self.ecm_fit_method_combo_a3 = ttk.Combobox(self.param_frames[3], textvariable=self.ecm_fit_method_a3_var, values=self.ecm_fit_method_options_a3, state="readonly"); self.ecm_fit_method_combo_a3.grid(row=1, column=1, sticky=tk.W); self.ecm_fit_method_combo_a3.set("AUTO")
        ttk.Label(self.param_frames[3], text="Fit Weight:").grid(row=1, column=2, sticky=tk.W, padx=5); self.ecm_fit_weight_a3_var = tk.StringVar(); self.ecm_fit_weight_options_a3 = ["AUTO", "MODULUS", "UNIT", "PROPIMAG", "PROPREAL"]
        self.ecm_fit_weight_combo_a3 = ttk.Combobox(self.param_frames[3], textvariable=self.ecm_fit_weight_a3_var, values=self.ecm_fit_weight_options_a3, state="readonly"); self.ecm_fit_weight_combo_a3.grid(row=1, column=3, sticky=tk.W); self.ecm_fit_weight_combo_a3.set("AUTO")
        ttk.Label(self.param_frames[3], text="Max Evaluations:").grid(row=2, column=0, sticky=tk.W); self.ecm_max_nfev_a3_var = tk.IntVar(value=30000)
        ttk.Entry(self.param_frames[3], textvariable=self.ecm_max_nfev_a3_var, width=10).grid(row=2, column=1, sticky=tk.W)
        self.param_frames[3].columnconfigure(1, weight=1); self.param_frames[3].columnconfigure(3, weight=1)

        self.param_frames[4] = ttk.LabelFrame(main_frame, text="Analysis 4: Peak Tracking Parameters", padding=10)
        ttk.Label(self.param_frames[4], text="Min Freq for Peak Search (Hz):").grid(row=0, column=0, sticky=tk.W, pady=2); self.min_freq_a4_var = tk.DoubleVar(value=0.1) 
        ttk.Entry(self.param_frames[4], textvariable=self.min_freq_a4_var, width=10).grid(row=0, column=1, sticky=tk.W, pady=2)
        ttk.Label(self.param_frames[4], text="Max Freq for Peak Search (Hz):").grid(row=1, column=0, sticky=tk.W, pady=2); self.max_freq_a4_var = tk.DoubleVar(value=100000.0) 
        ttk.Entry(self.param_frames[4], textvariable=self.max_freq_a4_var, width=10).grid(row=1, column=1, sticky=tk.W, pady=2)
        self.param_frames[4].columnconfigure(1, weight=1)

        for i, frame in self.param_frames.items():
            frame.grid(row=self.param_frames_container_row, column=0, columnspan=3, sticky=tk.NSEW, pady=5, padx=5)
            if i != self.analysis_type_var.get(): frame.grid_remove()
        
        self.run_button = ttk.Button(main_frame, text="Run Selected Analysis", command=self.run_analysis_thread, style="Accent.TButton")
        style.configure("Accent.TButton", font=('Helvetica', 11, 'bold'), foreground="white", background="#0078D7")
        self.run_button.grid(row=current_row, column=0, columnspan=3, pady=20, ipady=5); current_row += 1
        log_label = ttk.Label(main_frame, text="Log / Status:", font=('Helvetica', 10, 'italic'))
        log_label.grid(row=current_row, column=0, columnspan=3, sticky=tk.W, pady=(5,0), padx=5); current_row += 1
        self.log_text = scrolledtext.ScrolledText(main_frame, height=18, width=90, wrap=tk.WORD, state=tk.DISABLED, relief=tk.SUNKEN, borderwidth=1)
        self.log_text.grid(row=current_row, column=0, columnspan=3, sticky=tk.NSEW, padx=5, pady=5)
        main_frame.rowconfigure(current_row, weight=1); main_frame.columnconfigure(1, weight=1) 
        
        self.log_handler = TextHandler(self.log_queue)
        logger = logging.getLogger(); logger.setLevel(logging.INFO)
        if logger.hasHandlers(): logger.handlers.clear()
        logger.addHandler(self.log_handler)
        console_handler = logging.StreamHandler(); console_handler.setFormatter(self.log_handler.formatter); logger.addHandler(console_handler)
        self.master.protocol("WM_DELETE_WINDOW", self.on_app_close)
        self.process_log_queue(); self.toggle_parameter_frames()

    def toggle_parameter_frames(self): 
        selected_analysis = self.analysis_type_var.get()
        for analysis_num, frame in self.param_frames.items():
            if analysis_num == selected_analysis: frame.grid()
            else: frame.grid_remove()
    def browse_input_dir(self): 
        dirname = filedialog.askdirectory(title="Select Input Data Directory");
        if dirname: self.input_dir_var.set(dirname)
    def browse_output_dir(self):
        dirname = filedialog.askdirectory(title="Select Output Directory");
        if dirname: self.output_dir_var.set(dirname)
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
                if self.log_text.winfo_exists():
                    self.log_text.configure(state=tk.NORMAL); self.log_text.insert(tk.END, message + '\n')
                    self.log_text.see(tk.END); self.log_text.configure(state=tk.DISABLED)
        except queue.Empty: pass 
        except tk.TclError: pass 
        except Exception as e: print(f"Error processing log queue item: {e}")
        if self.master.winfo_exists(): self.master.after(100, self.process_log_queue)
    def on_app_close(self): 
        if messagebox.askokcancel("Quit", "Are you sure you want to quit?"):
            logging.info("App shutdown initiated."); self.analysis_stop_event.set()
            if self.analysis_thread and self.analysis_thread.is_alive():
                logging.info("Waiting for analysis thread (max 2s)..."); self.analysis_thread.join(timeout=2.0)
                if self.analysis_thread.is_alive(): logging.warning("Analysis thread did not terminate gracefully.")
            logging.info("Destroying main window."); self.master.destroy()
    def run_analysis_thread(self): 
        if self.analysis_thread and self.analysis_thread.is_alive(): messagebox.showwarning("Busy", "Analysis already running."); return
        self.run_button.config(state=tk.DISABLED); self.analysis_stop_event.clear()
        self.analysis_thread = threading.Thread(target=self.run_analysis, daemon=True); self.analysis_thread.start()

    def run_analysis(self): 
        input_dir = self.input_dir_var.get(); output_dir = self.output_dir_var.get()
        n_replicates_fallback = self.replicates_gui_var.get()
        apply_day0_global = self.apply_day0_global_var.get()
        average_replicates_global = self.average_replicates_global_var.get()
        heatmap_norm_global = self.heatmap_norm_strategy_var.get()
        custom_title_prefix = self.custom_title_prefix_var.get()
        include_auto_titles = self.include_auto_titles_var.get()
        custom_x_labels = self.custom_x_labels_var.get()
        custom_x_axis_title = self.custom_x_axis_title_var.get()
        analysis_type = self.analysis_type_var.get()

        if not input_dir or not output_dir:
            self.log_queue.put(("SHOW_ERROR", "Validation Error", "Input and Output directories must be specified."))
            logging.error("Input or Output directory not specified.")
            if not self.analysis_stop_event.is_set() and self.master.winfo_exists(): self.master.after(0, lambda: self.run_button.config(state=tk.NORMAL))
            return

        logging.info("="*40 + f"\nRUNNING ANALYSIS TYPE: {analysis_type}\n" + "="*40)
        result_message = "Analysis type not found or did not run."
        if self.analysis_stop_event.is_set(): logging.info("Analysis aborted pre-start."); return "Analysis Aborted Pre-Start"

        try:
            if analysis_type == 1:
                a1_norm_stddev = self.a1_norm_stddev_var.get()
                result_message = run_analysis_1(input_dir, output_dir, n_replicates_fallback, 
                                                apply_day0_global, average_replicates_global, heatmap_norm_global,
                                                custom_title_prefix, include_auto_titles, custom_x_labels, custom_x_axis_title,
                                                a1_norm_stddev, self.analysis_stop_event) 
            elif analysis_type == 2:
                mappings = get_drt_enum_mapping(); drt_method = mappings["DRTMethod"].get(self.drt_method_var.get()); drt_mode = mappings["DRTMode"].get(self.drt_mode_var.get())
                rbf_type = mappings["RBFType"].get(self.rbf_type_var.get()); rbf_shape = mappings["RBFShape"].get(self.rbf_shape_var.get())
                if not all([drt_method, drt_mode, rbf_type, rbf_shape]): raise ValueError("Invalid DRT enum selection.")
                lambda_vals = [float(L.strip()) for L in self.lambda_values_var.get().split(',') if L.strip()]
                if not lambda_vals: raise ValueError("Lambda list is empty.")
                result_message = run_analysis_drt(input_dir, output_dir, n_replicates_fallback, apply_day0_global, average_replicates_global, heatmap_norm_global, 
                    custom_title_prefix, include_auto_titles, custom_x_labels, custom_x_axis_title,
                    drt_method, drt_mode, lambda_vals, rbf_type, rbf_shape, self.rbf_size_var.get(), self.fit_penalty_drt_var.get(), 
                    self.include_inductance_drt_var.get(), self.num_attempts_drt_var.get(), self.mrq_cdc_drt_var.get(), num_drt_procs=0, stop_event=self.analysis_stop_event)
            elif analysis_type == 3:
                ecms = [ecm.strip() for ecm in self.ecm_list_a3_var.get().split(',') if ecm.strip()]
                if not ecms: raise ValueError("ECM list is empty.")
                result_message = run_analysis_ecm_fitting(input_dir, output_dir, n_replicates_fallback, apply_day0_global, average_replicates_global, heatmap_norm_global, 
                    custom_title_prefix, include_auto_titles, custom_x_labels, custom_x_axis_title,
                    ecms, self.ecm_fit_method_a3_var.get(), self.ecm_fit_weight_a3_var.get(), self.ecm_max_nfev_a3_var.get(), stop_event=self.analysis_stop_event)
            elif analysis_type == 4:
                min_freq = self.min_freq_a4_var.get() 
                max_freq = self.max_freq_a4_var.get() 
                if min_freq >= max_freq : raise ValueError("Min frequency must be less than Max frequency for peak tracking.")
                result_message = run_analysis_peak_tracking(input_dir, output_dir, n_replicates_fallback, apply_day0_global, average_replicates_global, heatmap_norm_global, 
                    custom_title_prefix, include_auto_titles, custom_x_labels, custom_x_axis_title,
                    min_freq, max_freq, 
                    stop_event=self.analysis_stop_event)
            
            if not self.analysis_stop_event.is_set():
                logging.info(f"A{analysis_type} final message: {result_message}")
                if "error" in result_message.lower() or "Error" in result_message : self.log_queue.put(("SHOW_ERROR", "Analysis Error", result_message))
                elif not any(x in result_message for x in ["Cancelled", "Aborted"]): self.log_queue.put(("SHOW_INFO", "Analysis Complete", result_message))
        except InterruptedError: # Custom exception for stop_event
             inter_msg = result_message if 'result_message' in locals() and any(x in result_message for x in ["Cancelled", "Aborted"]) else 'Operation Cancelled by User.'
             logging.info(f"Analysis explicitly interrupted: {inter_msg}")
             if self.master.winfo_exists() and not self.analysis_stop_event.is_set(): 
                 self.log_queue.put(("SHOW_WARNING", "Cancelled", inter_msg))
        except ValueError as ve: 
            if not self.analysis_stop_event.is_set(): logging.error(f"Input Error A{analysis_type}: {ve}"); self.log_queue.put(("SHOW_ERROR", "Input Error", str(ve)))
        except Exception as e:
            if not self.analysis_stop_event.is_set(): logging.error(f"Error A{analysis_type}: {e}", exc_info=True); self.log_queue.put(("SHOW_ERROR", "Critical Error", f"Error in A{analysis_type}:\n{e}"))
        finally:
            if not self.analysis_stop_event.is_set():
                try:
                    if self.master.winfo_exists(): self.master.after(0, lambda: {self.run_button.config(state=tk.NORMAL) if self.run_button.winfo_exists() else None})
                except tk.TclError: pass 
            self.analysis_thread = None

# ---------------------------------------------------------
# --- MAIN EXECUTION ---
# ---------------------------------------------------------
if __name__ == '__main__':
    root = tk.Tk()
    app = EISAnalysisApp(root)
    root.mainloop()
