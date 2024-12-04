import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr, ttest_1samp, false_discovery_control

def load_sub(sub, session):
    
    # Load data
    path = 'data/sub-{}/ses-{}/func/sub-{}_ses-{}_task-rest_bold/'.format(sub, session, sub, session)
    data = pd.read_csv(path + 'func_atlas/sub-{}_ses-{}_task-rest_space-MNI152Lin_res-3mm_atlas-Schaefer2018_dens-400parcels7networks_desc-sm0_bold.tsv'.format(sub, session), sep='\t', header=None).to_numpy()

    # Load categ variable
    pupil = np.loadtxt('derivatives/pupil/pupil_cat_ds_ma_{}_{}.txt'.format(sub, session)).astype(int)
    missingtp = data.shape[1] - pupil.shape[0]
    pupil = np.hstack((pupil, np.full((missingtp), np.nan))) # Add missing timepoints 

    return data, pupil

def functional_conn_ROIs(data, pupil, measure='spearman'):

    # Divide for epochs of high and low
    pupil_high = np.where(pupil==1)[0]
    pupil_low = np.where(pupil==0)[0]
    data_high = data[:, pupil_high]
    data_low = data[:, pupil_low]

    # Get correlation matrix
    if measure == 'spearman':
        correlation_high, _ = spearmanr(data_high, axis=1)
        correlation_low, _ = spearmanr(data_low, axis=1)

    elif measure == 'pearson':
        distances_h = squareform(pdist(data_high, metric='correlation'))
        correlation_high = 1 - distances_h

        distances_l = squareform(pdist(data_low, metric='correlation'))
        correlation_low = 1 - distances_l
        
    return correlation_high, correlation_low

def create_super_subject(sub_list, session, n_rois):
    
    # Initialize super subject matrix
    supersub_data = np.full((n_rois, 1), np.nan)
    supersub_pupil = np.nan

    # Concatenate each sub data
    for sub in sub_list:
                
        # Load data
        data, pupil = load_sub(sub, session)
        
        # Concatenate
        supersub_data = np.concatenate((supersub_data, data), axis=1)
        supersub_pupil = np.hstack((supersub_pupil, pupil))
        
    # Remove first row/element
    supersub_data = supersub_data[:,1:]
    supersub_pupil = supersub_pupil[1:]

    return supersub_data, supersub_pupil

def plot_fc(fc, fc_diff, picname):
    
    # Create Figure
    fig, axs = plt.subplots(1, 3, figsize=(8.3, 3), dpi=300)
    
    # Plot Heatmaps
    titles = ['high', 'low']
    for a, ax in enumerate(axs[:2]):
        sns.heatmap(fc[:,:,a], cmap='rocket_r', square=True, ax=ax, cbar_kws={'shrink':0.6, 'pad':0.01})
        ax.set_title('FC during {} arousal'.format(titles[a]))
    
    sns.heatmap(fc_diff, vmin=np.min(fc_diff), vmax=np.max(fc_diff), cmap='crest', square=True, ax=axs[2], cbar_kws={'shrink':0.6, 'pad':0.01})
    axs[2].set_title('Difference in FC between\nhigh and low arousal')

    # Remove ticks and adjust layout
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()

    # Save  
    plt.savefig('{}.png'.format(picname))

def t_tests(func_conn_diff):
    
    n_rois = func_conn_diff.shape[-1]

    # Initialize matrices to store t-stats and p-values
    t_stats = np.full((n_rois, n_rois), np.nan)
    p_values = np.full((n_rois, n_rois), np.nan)

    # Loop through each pair of ROIs
    for i in range(n_rois):
        for j in range(n_rois):
            # Extract differences across subjects for the ROI pair (i, j)
            roi_diff = func_conn_diff[:, i, j]
            
            # Perform a one-sample t-test against zero
            t_stat, p_val = ttest_1samp(roi_diff, popmean=0, nan_policy='omit')
            
            # Store the results
            t_stats[i, j] = t_stat
            p_values[i, j] = p_val

    # Correct for multiple comparisons using False Discovery Rate (FDR)
    q_values = false_discovery_control(p_values)

    # Save t-stats and p-values
    np.savez('derivatives/functional_connectivity/FC_differences_stats.npz', t_stats=t_stats, p_values=p_values, q_values=q_values)

    return t_stats, p_values, q_values

if __name__ == '__main__':
    
    sub_list = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
    ses_list = ['01']
    session = '01'
    measure = 'spearman' # 'pearson'
    n_rois = 400
    plot = False

    # Get super subject (concatenate subjects along time)
    supersub_data, supersub_pupil = create_super_subject(sub_list, session, n_rois)

    # Init res matrix
    func_conn_subs = np.full((len(sub_list), n_rois, n_rois, 2), np.nan)
    func_conn_super = np.full((n_rois, n_rois, 2), np.nan)

    # Get functional connectivity matrix
    for s, sub in enumerate(sub_list):
        data, pupil = load_sub(sub, session)
        func_conn_subs[s, :, :, 0], func_conn_subs[s, :, :, 1] = functional_conn_ROIs(data, pupil, measure)
        func_conn_super[:, :, 0], func_conn_super[:, :, 1] = functional_conn_ROIs(supersub_data, supersub_pupil, measure)
    
    # Save
    np.save('derivatives/functional_connectivity/FC_allsubs_highlowpupil', func_conn_subs)
    np.save('derivatives/functional_connectivity/FC_supersub_highlowpupil', func_conn_super)

    # Get difference
    func_conn_diff = func_conn_subs[:,:,:,0] - func_conn_subs[:,:,:,1]
    func_conn_super_diff = func_conn_super[:,:,0] - func_conn_super[:,:,1]

    # Group mean
    func_conn_group = np.mean(func_conn_subs, axis=0)
    func_conn_group_diff = np.mean(func_conn_diff, axis=0)

    # Plot FC for high, low and difference
    if plot:
        
        picname = 'func_conn_group_{}'.format(measure)
        plot_fc(func_conn_group, func_conn_group_diff, picname)

        picname = 'func_conn_supersub_{}'.format(measure)
        plot_fc(func_conn_super, func_conn_super_diff, picname)

    # T-tests
    t_stats = np.load('derivatives/functional_connectivity/FC_differences_stats.npz')['t_stats']
    p_values = np.load('derivatives/functional_connectivity/FC_differences_stats.npz')['p_values']
    q_values = np.load('derivatives/functional_connectivity/FC_differences_stats.npz')['q_values']