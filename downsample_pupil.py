import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

# Define params
filepath = 'data/sub-{}_ses-{}_task-rest_recording-eyetracking_physio.tsv'
sub_list = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
ses_list = ['01', '02']
folder_output = 'derivatives/pupil/'
columns = ['time', 'gazeH', 'gazeV', 'pupilsize', 'resX', 'resY', 'fixation', 'saccade', 'blink', 'tasktrigger', 'timetrigger', 'fmritrigger', 'interpolsamples']

for s, sub in enumerate(sub_list):
    for ses, session in  enumerate(ses_list):
        
        # Load data for each sub and session
        try:
            data = pd.read_csv(filepath.format(sub, session),  sep='\t', header=None)

        except FileNotFoundError:
            continue
        
        # Rename columns
        data = data.rename({c: col for c, col in enumerate(columns)}, axis=1)

        # Downsampling
        data_ds = data.loc[data['fmritrigger'] == 1]
        pupil_ds = data_ds.pupilsize.to_numpy()

        # Downsampling moving average
        TR = 2.1 #seconds
        sf = 250
        si = 1/sf
        window = int((TR/si)/2)
    
        indices = data.loc[data['fmritrigger'] == 1].index.to_numpy()
        start = indices - window
        
        pupil_ds_ma = np.full((len(indices)), np.nan)
        for c, s in enumerate(start):
            epoch = data[s: s+window*2].pupilsize.to_numpy()
            pupil_ds_ma[c] = np.mean(epoch)

        # Save
        np.savetxt(folder_output+'pupil_ds_{}_{}.1D'.format(sub, session), pupil_ds)
        np.savetxt(folder_output+'pupil_ds_ma_{}_{}.1D'.format(sub, session), pupil_ds_ma)

        # Get pupilsize 
        data_pos = data.loc[data['time']>=0]

        # Normalization
        pupil_mean = np.mean(data_pos.pupilsize)
        pupil_std = np.std(data_pos.pupilsize)
        data_pos['pupil_zscore'] = (data_pos.pupilsize-pupil_mean)/pupil_std
        
        data_pos['pupil_high_low'] = (data_pos.pupil_zscore > 0)*1
        data_pos['zerocol'] = 0

        data_pos[['time', 'zerocol', 'pupil_high_low']].to_csv(folder_output+'pupil_hl_{}_{}.csv'.format(sub, session))