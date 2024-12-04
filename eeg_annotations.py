# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 17:08:36 2024

@author: Francesco
"""


import numpy as np
import pandas as pd
import mne
import os
import re

#%%

projpath = "C:\\Users\\Francesco\\Desktop\\bh" #Set your path here, create the following folders into this folder: eegdata, pupildata, restdata
outpath = os.path.join(projpath, "outpath")
eegdatapath = os.path.join(projpath, "eegdata")
pupildata_path = os.path.join(eegdatapath, "pupildata")
restdata_path = os.path.join(eegdatapath, "restdata")

filename = os.listdir(outpath)[0]
filepath = os.path.join(outpath, filename)

#%%

# Loop through the files in restdata_path
for eeg_file in os.listdir(restdata_path):
    if eeg_file.endswith(".set"):
        # Extract subject and session info from the filename
        eeg_match = re.match(r"sub-(\d+)_ses-(\d+)_task-rest_eeg\.set", eeg_file)
        if not eeg_match:
            print(f"Skipping file {eeg_file} (filename doesn't match expected pattern).")
            continue
        
        sub_id, ses_id = eeg_match.groups()
        
        # Match the corresponding pupil data file
        pupil_pattern = rf"pupil_hl_{int(sub_id):02d}_{int(ses_id):02d}.*\.csv"
        pupil_file = next((f for f in os.listdir(pupildata_path) if re.match(pupil_pattern, f)), None)
        
        if not pupil_file:
            print(f"No matching pupil file found for {eeg_file}. Skipping.")
            continue
        
        # Load EEG data
        eeg_filepath = os.path.join(restdata_path, eeg_file)
        raw = mne.io.read_raw_eeglab(eeg_filepath, preload=True)
        raw.set_annotations(None)  # Clear any existing annotations
        
        # Sampling frequency
        sf = raw.info["sfreq"]
        
        # Load pupil data
        pupil_filepath = os.path.join(pupildata_path, pupil_file)
        pupil_low_high_raw = pd.read_csv(pupil_filepath)
        columns = ['Unnamed: 0', 'time', 'zerocol', 'pupil_high_low']
        pupil_low_high = np.array(pupil_low_high_raw[columns])
        
        # Select rows based on step size
        step = 250
        selected_rows = []
        for i in range(0, len(pupil_low_high), step):
            interval = pupil_low_high[i:i + step]
            if len(np.unique(interval[:, 3])) == 1:
                selected_rows.append(pupil_low_high[i])  # Append the first row of the interval
        
        # Prepare the pupil_size_steps array
        pupil_size_steps = np.array(selected_rows)
        
        # Create annotations (low = 0, high = 1) in seconds
        low_annotations = []
        high_annotations = []
        for i in range(len(pupil_size_steps)):
            timepoint = pupil_size_steps[i, 0] / sf  # Convert samples to seconds
            label = 'low' if int(pupil_size_steps[i, 3]) == 0 else 'high'
            if label == 'low':
                low_annotations.append([timepoint, 1 / sf, label])  
            elif label == 'high':
                high_annotations.append([timepoint, 1 / sf, label])  
        
        # Convert annotations to MNE format
        def create_annotations(annotations):
            return mne.annotations.Annotations(
                onset=[ann[0] for ann in annotations],
                duration=[ann[1] for ann in annotations],
                description=[ann[2] for ann in annotations]
            )
        
        low_annotations_mne = create_annotations(low_annotations)
        high_annotations_mne = create_annotations(high_annotations)
        
        # Save the data with "low" annotations
        raw_low = raw.copy()
        raw_low.set_annotations(low_annotations_mne)
        low_output_filename = f"{os.path.splitext(eeg_file)[0]}_low_annot.fif"
        low_output_filepath = os.path.join(outpath, low_output_filename)
        raw_low.save(low_output_filepath, overwrite=True)
        
        # Save the data with "high" annotations
        raw_high = raw.copy()
        raw_high.set_annotations(high_annotations_mne)
        high_output_filename = f"{os.path.splitext(eeg_file)[0]}_high_annot.fif"
        high_output_filepath = os.path.join(outpath, high_output_filename)
        raw_high.save(high_output_filepath, overwrite=True)
        
        print(f"Saved 'low' annotated data for {eeg_file} to {low_output_filepath}.")
        print(f"Saved 'high' annotated data for {eeg_file} to {high_output_filepath}.")

#%%


