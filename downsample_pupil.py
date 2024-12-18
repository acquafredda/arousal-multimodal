import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import os

def get_pupilsize_categ(pupilsize):

    # Normalization
    pupil_mean = np.mean(pupilsize)
    pupil_std = np.std(pupilsize)
    pupil_zscore = (pupilsize-pupil_mean)/pupil_std
    
    # Create categorical variable
    pupil_high_low = (pupil_zscore > 0)*1
       
    return pupil_high_low

if __name__ == '__main__':
        
    # Define params
    save = True
    filepath = 'data/sub-{}/sub-{}_ses-{}_task-rest_recording-eyetracking_physio.tsv'
    sub_list = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '17', '18', '19', '21', '22']
    ses_list = ['01']
    folder_output = 'derivatives/pupil/sub-{}/'
    columns = ['time', 'gazeH', 'gazeV', 'pupilsize', 'resX', 'resY', 'fixation', 'saccade', 'blink', 'tasktrigger', 'timetrigger', 'fmritrigger', 'interpolsamples']

    for s, sub in enumerate(sub_list):
        for ses, session in  enumerate(ses_list):
            
            # Load data for each sub and session
            try:
                data = pd.read_csv(filepath.format(sub, sub, session),  sep='\t', header=None)

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

            # Get pupilsize 
            data_pos = data.loc[data['time']>=0]

            # Get categorical variable
            pupil_highlow = get_pupilsize_categ(data_pos.pupilsize.to_numpy())
            pupil_highlow_dsma = get_pupilsize_categ(pupil_ds_ma)
            
            # Create df for EEG
            data_pos['pupil_high_low'] = pupil_highlow
            data_pos['zerocol'] = 0

            # Save
            if save:
                if not os.path.exists(folder_output.format(sub)):
                    os.makedirs(folder_output.format(sub))
                
                data_pos[['time', 'zerocol', 'pupil_high_low']].to_csv(folder_output.format(sub)+'sub-{}_pupil_hl.csv'.format(sub, session))
                np.savetxt(folder_output.format(sub)+'sub-{}_pupil_ds.1D'.format(sub), pupil_ds)
                np.savetxt(folder_output.format(sub)+'sub-{}_pupil_ds_ma.1D'.format(sub), pupil_ds_ma)
                np.savetxt(folder_output.format(sub)+'sub-{}_pupil_cat_ds_ma.txt'.format(sub), pupil_highlow_dsma)