from scipy.signal import find_peaks
from scipy.stats.mstats import mquantiles
import numpy as np
import matplotlib
import pandas as pd

import matplotlib.pyplot as plt

#defaultdict to use nested dictionaries
from collections import defaultdict

#decide color series
import itertools
#dates
import matplotlib.dates as mdates
#closest distance
from scipy.spatial import distance

from sklearn.cluster import KMeans

import seaborn as sns


def peak_box_multipeaks(ens, obs, sim_start):
    """
    The following script refers to the development of the peak-box algorithm for the detection of multiple peak flows,
    altogether with the older classic algorithm (Zappa et al. 2013). The function needs as input a pandas DataFrame of the 
    ensemble runoff forecasts, the observed runoff time series in the basin, and the model initialization time, 
    and produces as output a plot comprehensive of the classic peak-box, the newly peak-box for forecasting multiple 
    peak flows, and the tables of sharpness and peak verification values. 
    An example of the input needed and the output produced can be found in the jupyter-notebook within this repository.
    
    The function is divided into:
        - Peak detection of the ensemble runoff forecasts (both for classic peak-box, for the multiple peak-box) and of
          the observation time series
        - Boxes construction and sharpness and verification calculation
        - Plotting 
    
    Function's description: plot the peak-box approach for the group of runoff realizations considered together 
    with observation: find the peak for every realization in the entire temporal domain, find out the first and 
    the last one taking place and the ones with highest and lowest magnitudes, plot the peak-box, find the IQR box 
    from all the peaks and timings and plot it, find the peak and timing medians and plot it.
    Calculate the full and IQR sharpness of the forecasts and the deviations of observation peak from the peak 
    represented by peak and timing median.    
    """
    
    #some preliminary settings:
    all_dates_hours = pd.DataFrame(index=range(120), columns=['date','hour'])
    all_dates_hours['date'] = ens['date']
    all_dates_hours['hour'] = ens.index.values
    
    obs = obs.reset_index()
    
    #area of the basin in km2:
    A = 186
    
    #use latex with matplotlib
    pgf_with_latex = {
        "pgf.texsystem": "xelatex",
        "text.usetex": True,            # use LaTeX to write all text
        "font.family": "serif",         # use serif rather than sans-serif
         "pgf.rcfonts": False,
        "axes.labelsize": 10,
        "font.size": 12,
        "legend.fontsize": 10,
        "axes.titlesize": 14,           # Title size when one figure
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.titlesize": 12,         # Overall figure title
        "text.latex.unicode": True,
        "pgf.preamble": [
            r'\usepackage{xcolor}', # xcolor for colours
            r'\usepackage{color}',
            r'\usepackage{fontspec}',
            r'\setmainfont{Ubuntu}',
            r'\setmonofont{Ubuntu Mono}',
            r'\usepackage{unicode-math}',
            r'\setmathfont{Ubuntu}'
            r'\usepackage{fontspec}'
        ]
    }
    
    matplotlib.rcParams.update(pgf_with_latex)
    
    #function to estimate distance between two points, to calculate D_peak and D_time (verification of peak-boxes)
    def closest_node(node, nodes):
        closest_index = distance.cdist([node], nodes).argmin()
        return nodes[closest_index]
    
    """
    CLASSIC PEAK-BOX APPROACH: look at peak maximum for every realization
    """
    
    #initialize dataframe containing the values of peak discharges for every realization
    c_df_max_runoff = pd.DataFrame(index=(ens.columns[~ens.columns.isin(['date'])]),
                                 columns=['max', 'date', 'hour'])
    
    for member in ens.columns[~ens.columns.isin(['date'])]:
    
        # Find all the local peak maximums for every realization, excluding borders of x domain (hour 0 and hour 120)
        c_peaks = find_peaks(ens[member][1:-1], height=0)
        
        # Select the maximum value of the peaks found and find its date and hour associated
        c_df_max_runoff['max'][member] = max(c_peaks[1]['peak_heights'])
        c_df_max_runoff['date'][member] = ens['date'][1:-1].loc[ens[member][1:-1] == 
                        c_df_max_runoff['max'][member]].iloc[0]
        c_df_max_runoff['hour'][member] = int(ens.loc[ens['date'] == c_df_max_runoff['date'][member]].index.values) 
        
    #report all peak and timing(hour) and correspondent dates quantiles in a dataframe
    c_peaks_timings = pd.DataFrame(index=range(5), columns=['peak', 'timing', 'date'])
    c_peaks_timings['peak'] = mquantiles(c_df_max_runoff['max'], prob=[0.0,0.25,0.5,0.75,1.0])
    c_peaks_timings['timing'] = mquantiles(c_df_max_runoff.hour, prob=[0.0,0.25,0.5,0.75,1.0]).astype(int)
    for i in range(5):
        c_peaks_timings['date'][i] = str(all_dates_hours['date'].loc[all_dates_hours['hour'] == 
                                                                   c_peaks_timings['timing'][i]].iloc[0])
    
    """
    MULTIPLE PEAKS APPROACH: 
        
    1 - Know all the peaks presented by a realization
    2 - Decide a criteria to consider just "relevant" peaks i.e. peaks that can be associated to different events
    3 - Based on the remained peaks regroup them considering all realizations
    """
    
    # Implement the division of peaks for all realizations considered:
    
    # dictionary to contain all the event peaks for different ens members
    peaks_dict = lambda: defaultdict(peaks_dict)
    event_peaks_dict = peaks_dict()
    
    #for loop on the forecast ensemble members
    for member in list(ens.columns[~ens.columns.isin(['date'])]):
       
        #Find all the local peak maximums, excluding borders of x domain (hour 0 and hour 120), with a certain prominence, that must 
        #depend on the range of discharge covered by the specific forecast:
        
        #CRITERIA TO KEEP JUST THE IMPORTANT PEAKS:
        #Set the prominence based on the maximum value reached within the forecast range:
        if max(ens[member][1:-1]) < 100:
            prom = 2
        else:
            prom=8
            
        peaks = find_peaks(ens[member][1:-1], prominence=np.concatenate((np.ones(5),np.zeros(108)+prom, np.ones(5))), height=0) 
    
        #condition to avoid low flow: keep the peaks with a value >= 23.1 m3s-1
        peaks_index = list(peaks[0][peaks[1]['peak_heights'] >=23.1])
        peaks[1]['peak_heights'] = peaks[1]['peak_heights'][peaks[1]['peak_heights'] >23.1]
        
        peak_date = pd.DataFrame(index=range(len(peaks[1]['peak_heights'])), columns=['date'])
        
        for p in range(len(peaks[1]['peak_heights'])):
            peak_date['date'][p] = ens['date'][1:-1].loc[ens[member][1:-1] == peaks[1]['peak_heights'][p]].iloc[0]
        
        # empty dataframe to contain so-called event peaks i.e. the relatively important peaks associated to events
        event_peaks = pd.DataFrame(index=range(120),columns=['hour','date', 'peak'])
        
        n_event = 0
            
        for p in range(len(peaks[1]['peak_heights'])):
            
            # if condition: must not go beyond the 120 hours limit and before the beginning at 0 hours
            if (peaks_index[p] > 0) and (peaks_index[p] < 120):
                
                event_peaks['hour'][n_event] = peaks_index[p]+1
                event_peaks['date'][n_event] = ens['date'][1:-1].loc[ens[1:-1].index == event_peaks['hour'][n_event]].iloc[0]
                event_peaks['peak'][n_event] = ens[member][peaks_index[p]+1]
                n_event = n_event+1
            
        #keep just the rows with peaks
        event_peaks = event_peaks[pd.notnull(event_peaks['peak'])]
                                
        #loop to keep just one peak if other peaks are very near (within temporal window of +- 10 hours):           
        while True:
            
            #"save" old index to compare it with the new one at the end when some peak are withdrawn
            old_event_peaks_index = event_peaks.index
            
            for i,j in zip(event_peaks.index, event_peaks.index+1):
                
                #conditions to avoid problems when considering the last peak of the domain
                if (i == event_peaks.index[-1] + 1) or (j == event_peaks.index[-1] + 1):
                    break
                
                #condition to discard very near in time peaks with very similar values:
                if (event_peaks.hour[i] >= event_peaks.hour[j] - 10): 
                    
                    #condition to keep the highest peak between the two near peaks considered:
                    if event_peaks['peak'][j] > event_peaks['peak'][i]:
                        event_peaks = event_peaks.drop(event_peaks.index[i])
                    
                    elif event_peaks['peak'][j] < event_peaks['peak'][i]:
                        event_peaks = event_peaks.drop(event_peaks.index[j])
                        
                    event_peaks.index = range(len(event_peaks))
            
            #condition to keep the length of the index correct: if old index and new index lengths are equal exit the while loop
            if len(old_event_peaks_index) == len(event_peaks.index):
                break
        
        #write all the event peaks obtained in a dictionary for different members:
        event_peaks_dict[member] = event_peaks            
            
    #PEAKS SPLITTING INTO GROUPS RELATED TO DIFFERENT EVENTS:
    #Count how many peaks are present in every realization: needed to automatically select the correct number of clusters when splitting into groups
    N_peaks_for_realization = pd.DataFrame(index=range(len(ens.columns[~ens.columns.isin(['date'])])), columns=['n_peaks'])                            
    
    index_i = 0
    for member in ens.columns[~ens.columns.isin(['date'])]:
        N_peaks_for_realization.loc[N_peaks_for_realization.index == index_i] = len(event_peaks_dict[member])
        index_i = index_i + 1
        
    mean_peaks_per_realiz = int(round(np.mean(N_peaks_for_realization['n_peaks'])))
    
    #count all the imporant peaks found and write them in one dataframe sorting all of them
    n_important_peaks = 0
    
    for member in event_peaks_dict.keys():
        n_important_peaks = n_important_peaks + len(event_peaks_dict[member])
                                
    important_peaks = pd.DataFrame(index=range(n_important_peaks),columns=['hour','date', 'peak', 'memb'])
    
    w=0
    for member in event_peaks_dict.keys():
        important_peaks.loc[w:w+len(event_peaks_dict[member])-1][['hour', 'date', 'peak']] = event_peaks_dict[member].values
        important_peaks['memb'][w:w+len(event_peaks_dict[member])] = member
        w=w+len(event_peaks_dict[member])
    important_peaks = important_peaks.sort_values(by=['hour']).reset_index(drop=True)
    
    
    #divide the important peaks into group with a k-means clustering algorithm:
    nclusters = mean_peaks_per_realiz
    
    scaled_important_peaks = important_peaks.copy(deep=True)
    
    #scale the peak variable to give more importance to the hour in the clustering:
    scaled_important_peaks['peak'] = important_peaks['peak']*0.1
    
    kmeans = KMeans(n_clusters = nclusters, random_state=0).fit(scaled_important_peaks[['hour', 'peak']])
    
    clust_labels = kmeans.labels_
    clust_labels = pd.DataFrame({"clust": clust_labels})
    
    #rewrite the cluster labels in ascending order:
    c_index = []
    c_index.append(0)
    
    clust_labels_new = pd.DataFrame()
    
    if nclusters > 1:
        for cluster in range(1, nclusters):
            
            c_index.append(clust_labels[c_index[cluster-1]:][clust_labels['clust'] != clust_labels['clust'][c_index[cluster-1]]].index[0])
            group = 0*clust_labels['clust'][c_index[cluster-1] : c_index[cluster]] + cluster-1
            clust_labels_new = pd.concat([clust_labels_new, group]).astype(int)
    
    else:
        cluster=0
    
    c_index.append(clust_labels.index[-1])
    group = 0*clust_labels['clust'][c_index[cluster] : c_index[cluster+1]+1] + cluster
    clust_labels_new = pd.concat([clust_labels_new, group]).astype(int)
        
    clust_labels_new.columns = ['clust']
    
    
    clust_important_peaks = pd.concat([important_peaks, clust_labels_new], axis=1)
    
    #divide all the peaks found in the correct groups based on cluster label
    peak_groups_dict = lambda: defaultdict(peak_groups_dict)         
    peak_groups = peak_groups_dict()               
        
    for group in np.unique(clust_labels_new):
            
        peak_groups[group] = clust_important_peaks.loc[clust_important_peaks.clust == group]
     
        
    #add condition of uniqueness: in one group there must be JUST 1 peak for every different realization! 
    #If one realization have more than 1 peak in one group -> delete the lowest peak in magnitude, keep the largest
    
    for group in np.unique(clust_labels_new):
        #find where the member is repeated
        dupl_subset = peak_groups[group][peak_groups[group].duplicated(subset='memb',keep=False)]
        
        for dupl_memb in np.unique(dupl_subset.memb):
            
            #find the indexes of the rows where the peak is not the maximum for that group to discard those rows
            dupl_subset = dupl_subset.drop(dupl_subset.loc[dupl_subset['peak'] == 
                                             max(dupl_subset['peak'].loc[dupl_subset['memb'] == dupl_memb])].index, axis=0)
        
        discard_indexes = dupl_subset.index
        for d_index in discard_indexes:
            peak_groups[group] = peak_groups[group].drop(peak_groups[group].loc[peak_groups[group].index == d_index].index, axis=0)
        
        
    """   
    OBSERVED PEAKS: 
    """
    
    # apply the same procedure as before to distinguish peaks related to different events:
    
    #reset obs index    
    obs = obs.reset_index()
    
    #CRITERIA TO KEEP JUST THE IMPORTANT PEAKS:
    #Set the prominence based on the maximum value reached within the forecast range:
    if max(obs.runoff[1:-1]) < 100:
        prom = 2
    else:
        prom=8
        
    #Find all the local peak maximums for obs, excluding borders of x domain (hour 0 and hour 120)
    OBSpeaks = find_peaks(obs.runoff[1:-1], prominence=np.concatenate((np.ones(5),np.zeros(108)+prom, np.ones(5))), height=0)
    
    #condition to avoid low flow: keep the peaks with a value >= 23.1 m3s-1
    OBSpeaks_index = list(OBSpeaks[0][OBSpeaks[1]['peak_heights'] >=23.1])
    OBSpeaks[1]['peak_heights'] = OBSpeaks[1]['peak_heights'][OBSpeaks[1]['peak_heights'] >23.1]
    
    OBSpeak_date = pd.DataFrame(index=range(len(OBSpeaks[1]['peak_heights'])), columns=['date'])
    
    for p in range(len(OBSpeaks[1]['peak_heights'])):
        OBSpeak_date['date'][p] = obs['date'][1:-1].loc[obs['runoff'][1:-1] == OBSpeaks[1]['peak_heights'][p]].iloc[0]
    
    # empty dataframe to contain so-called event peaks i.e. the important peaks associated to events
    OBSevent_peaks = pd.DataFrame(index=range(120),columns=['hour','date', 'peak'])
    
    n_event = 0
    
    for p in range(len(OBSpeaks[1]['peak_heights'])):
            
        # if condition: must not go beyond the 120 hours limit and before the beginning at 0 hours, 
            
        if (OBSpeaks_index[p] > 0) and (OBSpeaks_index[p] < 120):
            
            OBSevent_peaks['hour'][n_event] = OBSpeaks_index[p]+1
            OBSevent_peaks['date'][n_event] = obs['date'][1:-1].loc[ens[1:-1].index == OBSevent_peaks['hour'][n_event]].iloc[0]
            OBSevent_peaks['peak'][n_event] = obs.runoff[OBSpeaks_index[p]+1]
            n_event = n_event+1
        
    #keep just the rows with peaks
    OBSevent_peaks = OBSevent_peaks[pd.notnull(OBSevent_peaks['peak'])]
        
    #loop to keep just one peak if other peaks are very near (+- 10 hours):
    while True:
        
        #"save" old index to compare it with the new one at the end when some peak are withdrawn
        OBSold_event_peaks_index = OBSevent_peaks.index
        
        for i,j in zip(OBSevent_peaks.index, OBSevent_peaks.index+1):
            
            #conditions to avoid problems when considering the last peak of the domain
            
            if (i == OBSevent_peaks.index[-1] + 1) or (j == OBSevent_peaks.index[-1] + 1):
                break
            
            #condition to discard very near in time peaks with very similar values:
            
            if (OBSevent_peaks.hour[i] >= OBSevent_peaks.hour[j] - 10): #or (event_peaks.hour[i] <= event_peaks.hour[j] + 4):
                
                #condition to keep the highest peak between the two near peaks considered:
                
                if OBSevent_peaks['peak'][j] > OBSevent_peaks['peak'][i]:
                    OBSevent_peaks = OBSevent_peaks.drop(OBSevent_peaks.index[i])
                
                elif OBSevent_peaks['peak'][j] < OBSevent_peaks['peak'][i]:
                    OBSevent_peaks = OBSevent_peaks.drop(OBSevent_peaks.index[j])
                    
                OBSevent_peaks.index = range(len(OBSevent_peaks))
        
        #condition to keep the length of the index correct: if old index and new index lengths are equal exit the while loop
        if len(OBSold_event_peaks_index) == len(OBSevent_peaks.index):
            break               
          
    """
    PLOT:
    plot both the classic and the multiple peaks peakbox, in two different panels
    """        
            
    # plot all peaks in different groups as coloured solid dots:
    colors = itertools.cycle(["#e60000", "#0000e6", "#e6e600", "#bf00ff", "#009933", "#b35900"])
    
    #pal = sns.hls_palette( 8, l=.5, s=.8)
    #colors = itertools.cycle(['#e619d9', '#8B6200', '#e62619', '#1940e6', '#8c19e6', '#08A539']) 
                
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,8), dpi=100)
        
    ax1 = plt.subplot2grid((2,1), (0,0), rowspan=1, colspan=1)
    ax2 = plt.subplot2grid((2,1), (1,0), rowspan=1, colspan=1, sharex=ax1)
    
    
    ax1.spines["bottom"].set_visible(False)
    ax1.tick_params(axis='x', length=0)
    ax1.tick_params(axis='both', which='both', bottom=True, top=False, labelbottom=False, labelsize=13)
    ax2.tick_params(axis='both', which='both', labelsize=13)
    ax2.set_xlabel('Time [yyyy-mm-dd]', fontsize=18)
    fig.subplots_adjust(hspace=0)
    
    for member in ens.columns[~ens.columns.isin(['date'])]:
        #classic peak-box peaks dots:
        ax1.plot(c_df_max_runoff.date[member], c_df_max_runoff['max'][member], 'o',markersize=2, color='#636363', alpha=0.45,
               zorder=2)
        #ensemble realizations
        runoff_member = ax1.plot(ens.date, ens[member], color='#32AAB5', linewidth=0.75, alpha=0.65, zorder=1)
        ax2.plot(ens.date, ens[member], color='#32AAB5', linewidth=0.75, alpha=0.65, zorder=1)
    for group in peak_groups.keys():
        color = next(colors)
        #multipeak-box peaks dots divided in groups:
        peak_member = ax2.plot(peak_groups[group]['date'], peak_groups[group]['peak'],'o',markersize=2, color=color, 
                              alpha=0.5, zorder=3)
    #observation series plot
    l2 = ax1.plot(obs.date, obs.runoff, linewidth=2, label='Runoff obs', color='orange', zorder = 15)
    ax2.plot(obs.date, obs.runoff, linewidth=2, label='Runoff obs', color='orange', zorder = 15)
    #observation peaks plot
    for OBSpeak in OBSevent_peaks.index:
        peak_obs = ax1.plot(OBSevent_peaks['date'][OBSpeak], OBSevent_peaks['peak'][OBSpeak],'*',markersize=20, color='orange', 
                           markeredgecolor='black', markeredgewidth=1.5, alpha=1, zorder=100)
        ax2.plot(OBSevent_peaks['date'][OBSpeak], OBSevent_peaks['peak'][OBSpeak],'*',markersize=20, color='orange', 
                           markeredgecolor='black', markeredgewidth=1.5, alpha=1, zorder=100)
            
    
    """
    DEVELOP PEAK BOXES FOR THE CLASSIC APPROACH AND THEN FOR EVERY DIFFERENT GROUP OF MULTIPEAKS APPROACH:
    
    Peak-Box (outer rectangle)
    IQR-box (inner rectangle)
    Median of the peak discharge
    Median of the peak timing
    """
    
    """
    - classic approach (for detailed description and comments go to multi-peaks approach peakbox construction)
    """
    
    #Peak-Box (outer rectangle):
    c_lower_left_pb = [c_peaks_timings['date'][0], c_peaks_timings['peak'][0]]
    c_upper_right_pb =  [c_peaks_timings['date'][4], c_peaks_timings['peak'][4]]
    
    alpha=0.75
    color='#636363'
    lw=2
    zorder = 20
    
    ax1.plot([pd.to_datetime(c_lower_left_pb[0]), pd.to_datetime(c_lower_left_pb[0])],
                [c_lower_left_pb[1], c_upper_right_pb[1]], color=color, alpha=alpha, lw=lw, zorder=zorder)
    
    ax1.plot([pd.to_datetime(c_lower_left_pb[0]), pd.to_datetime(c_upper_right_pb[0])],
                [c_lower_left_pb[1], c_lower_left_pb[1]], color=color, alpha=alpha, lw=lw, zorder=zorder)
    
    ax1.plot([pd.to_datetime(c_upper_right_pb[0]), pd.to_datetime(c_upper_right_pb[0])],
                [c_lower_left_pb[1], c_upper_right_pb[1]], color=color, alpha=alpha, lw=lw, zorder=zorder)
    
    ax1.plot([pd.to_datetime(c_lower_left_pb[0]), pd.to_datetime(c_upper_right_pb[0])],
                [c_upper_right_pb[1], c_upper_right_pb[1]], color=color, alpha=alpha, lw=lw, zorder=zorder)
    
    #IQR-box (inner rectangle):
    c_lower_left_IQRbox = [c_peaks_timings['date'][1], c_peaks_timings['peak'][1]]
    c_upper_right_IQRbox = [c_peaks_timings['date'][3], c_peaks_timings['peak'][3]]
    
    
    ax1.plot([pd.to_datetime(c_lower_left_IQRbox[0]), pd.to_datetime(c_lower_left_IQRbox[0])],
                [c_lower_left_IQRbox[1], c_upper_right_IQRbox[1]], color=color, alpha=alpha, lw=lw, zorder=zorder)
    
    ax1.plot([pd.to_datetime(c_lower_left_IQRbox[0]), pd.to_datetime(c_upper_right_IQRbox[0])],
                [c_lower_left_IQRbox[1], c_lower_left_IQRbox[1]], color=color, alpha=alpha, lw=lw, zorder=zorder)
    
    ax1.plot([pd.to_datetime(c_upper_right_IQRbox[0]), pd.to_datetime(c_upper_right_IQRbox[0])],
                [c_lower_left_IQRbox[1], c_upper_right_IQRbox[1]], color=color, alpha=alpha, lw=lw, zorder=zorder)
    
    ax1.plot([pd.to_datetime(c_lower_left_IQRbox[0]), pd.to_datetime(c_upper_right_IQRbox[0])],
                [c_upper_right_IQRbox[1], c_upper_right_IQRbox[1]], color=color, alpha=alpha, lw=lw, zorder=zorder)
    
    #Median of the peak discharge:
    ax1.plot([pd.to_datetime(c_lower_left_pb[0]), pd.to_datetime(c_upper_right_pb[0])],
                [c_peaks_timings['peak'][2], c_peaks_timings['peak'][2]], color=color, alpha=alpha, lw=lw, zorder=zorder)
    
    #Median of the peak timing:
    ax1.plot([pd.to_datetime(c_peaks_timings['date'][2]), pd.to_datetime(c_peaks_timings['date'][2])],
                [c_lower_left_pb[1], c_upper_right_pb[1]], color=color, alpha=alpha, lw=lw, zorder=zorder)
    
    #Median value:
    c_median_value = ax1.plot(pd.to_datetime(c_peaks_timings['date'][2]), c_peaks_timings['peak'][2], '*', markersize=20, 
                             color=color, alpha=0.85, lw=lw, zorder=zorder+1, markeredgecolor='black', markeredgewidth=1.5, 
                             label='Classic ($t_{50}$, $p_{50}$) (Zappa et al. 2013)')
    
    #Peak-box and IQR-box volumes:
    c_volume_pb = (c_peaks_timings['timing'][4] - c_peaks_timings['timing'][0])*(c_peaks_timings['peak'][4] - c_peaks_timings['peak'][0])
    c_volume_iqr = (c_peaks_timings['timing'][3] - c_peaks_timings['timing'][1])*(c_peaks_timings['peak'][3] - c_peaks_timings['peak'][1])
    
    c_sharpness_pb = c_volume_pb*3.6/A
    c_sharpness_iqr = c_volume_iqr*3.6/A
    
    #Verification of peak median vs obs: we have to associate correctly the peak median to the correct obs peak
    #Dpeak = |p50-pobs|
    #Dtime = |t50-tobs|
    
    c_obs_peaks_assoc = pd.DataFrame(columns=['hour', 'date', 'peak'])
    
    #check if there is one obs peak inside the IQR box firstly, and then, if not, inside the peak box 
    for index, observed_peak in OBSevent_peaks.iterrows():
        
        #look if the observed peak is inside the IQR box, and in that case dont even look in the peak-box
        if (str(observed_peak['date']) >= c_peaks_timings['date'][1]) and (str(observed_peak['date']) <= c_peaks_timings['date'][3]) and (observed_peak['peak'] >= c_peaks_timings['peak'][1]) and (observed_peak['peak'] <= c_peaks_timings['peak'][3]) :
            c_obs_peaks_assoc.loc[index] = observed_peak
            break
        
        #look if the observed peak is inside the peak box
        if (str(observed_peak['date']) >= c_peaks_timings['date'][0]) and (str(observed_peak['date']) <= c_peaks_timings['date'][4]) and (observed_peak['peak'] >= c_peaks_timings['peak'][0]) and (observed_peak['peak'] <= c_peaks_timings['peak'][4]) :
            c_obs_peaks_assoc.loc[index] = observed_peak
        
    c_obs_peaks_assoc = c_obs_peaks_assoc.reset_index(drop=True)    
    
    # write hour and peak from observation peaks just found in a list to perform the distance calculation:
    c_obs_peaks_points = [(c_obs_peaks_assoc.hour[i], c_obs_peaks_assoc.peak[i]) for i in range(len(c_obs_peaks_assoc))]
        
    #if we have at least one obs peak found inside the boxes:
    if len(c_obs_peaks_assoc) != 0:
        
        #condition if more than one obs_peak was found inside the boxes: take the nearest one
        if (len(c_obs_peaks_assoc) > 1):
            c_nearest_obs_peak = closest_node((c_peaks_timings['timing'][2], c_peaks_timings['peak'][2]), c_obs_peaks_points)
                        
        elif len(c_obs_peaks_assoc) == 1:
            c_nearest_obs_peak = (c_obs_peaks_assoc.hour[0], c_obs_peaks_assoc.peak[0])
    
        #calculate difference in peak magnitude and timing between the near observed peak and the median value of the peak-box:
        c_verification_d_peak = abs(c_peaks_timings['peak'][2] - c_nearest_obs_peak[1])
        c_verification_d_timing = abs(c_peaks_timings['timing'][2] - c_nearest_obs_peak[0])
    
    else:
        c_verification_d_peak = np.nan
        c_verification_d_timing = np.nan
    
    
    """
    - multipeaks approach
    """
    
    #dataframe to contain the values of peak-box and iqr-box volumes for every different group
    box_volumes = pd.DataFrame(index=range(len(peak_groups.keys())), columns=['pb','iqr'])
    
    #dataframe to contain the values of peak-box and iqr-box sharpness for every different group
    sharpness = pd.DataFrame(index=range(len(peak_groups.keys())), columns=['pb','iqr'])
    
    #dataframe to contain the verification values for peak and timing, related to observation, for every different group
    verification = pd.DataFrame(index=range(len(peak_groups.keys())), columns=['d_peak','d_timing'])
    
    #index to write sharpness/verification in the dataframe for every cycle on the groups
    i_group = 0
    
    colors = itertools.cycle(["#e60000", "#0000e6", "#e6e600", "#bf00ff", "#009933", "#b35900"])
    #colors = itertools.cycle(['#e619d9', '#8B6200', '#e62619', '#1940e6', '#8c19e6', '#08A539']) 
                
    for group in peak_groups.keys():
        
        color = next(colors)
      
        #empty arrays to contain all the dates/peaks for every different realization of one specific group
        all_dates_of_group = []
        all_hours_of_group = []
        all_peaks_of_group = []
        
        #write all dates, hours and peaks for every possible realizations for every group in peak_groups
        for date in peak_groups[group]['date']:
            all_dates_of_group.append(str(date))
        for peak in peak_groups[group]['peak']:
            all_peaks_of_group.append(peak)
        for hour in peak_groups[group]['hour']:
            all_hours_of_group.append(hour)
    
        
        # PEAK-BOX:
        
        #the lower left coordinate set to the earliest time when a peak flow occurred in the available ensemble members (t0) 
        #and the lowest peak discharge of all members during the whole forecast period (p0)
        lower_left_pb = [min(all_dates_of_group), min(all_peaks_of_group)]
            
        #upper right coordinate set to the latest time when a peak flow occurred in the available ensemble members (t100) 
        #and the highest peak discharge of all members during the whole forecast period (p100)
        upper_right_pb =  [max(all_dates_of_group), max(all_peaks_of_group)]
    
        #plot the peak-boxes
        alpha=0.75
        lw=2
        zorder = 20
        
        ax2.plot([pd.to_datetime(lower_left_pb[0]), pd.to_datetime(lower_left_pb[0])],
                    [lower_left_pb[1], upper_right_pb[1]], color=color, alpha=alpha, lw=lw, zorder=zorder)
        
        ax2.plot([pd.to_datetime(lower_left_pb[0]), pd.to_datetime(upper_right_pb[0])],
                    [lower_left_pb[1], lower_left_pb[1]], color=color, alpha=alpha, lw=lw, zorder=zorder)
        
        ax2.plot([pd.to_datetime(upper_right_pb[0]), pd.to_datetime(upper_right_pb[0])],
                    [lower_left_pb[1], upper_right_pb[1]], color=color, alpha=alpha, lw=lw, zorder=zorder)
        
        ax2.plot([pd.to_datetime(lower_left_pb[0]), pd.to_datetime(upper_right_pb[0])],
                    [upper_right_pb[1], upper_right_pb[1]], color=color, alpha=alpha, lw=lw, zorder=zorder)
    
    
        # IQR-BOX:
        
        #calculate the quantiles of peaks and timings and convert timings in dates
        peaks_quantiles = mquantiles(all_peaks_of_group, prob=[0.0,0.25,0.5,0.75,1.0])
        hours_quantiles = mquantiles(sorted(all_hours_of_group), prob=[0.0,0.25,0.5,0.75,1.0]).astype(int)
        dates_quantiles = ['']*5
        for i in range(5):
            dates_quantiles[i] = str(all_dates_hours['date'].loc[all_dates_hours['hour'] == 
                                                                       hours_quantiles[i]].iloc[0])
           
        #lower left coordinate set to the 25% quartile of the peak timing (t25) 
        #and the 25% quartile of the peak discharges of all members during the whole forecast period (p25)
        lower_left_IQRbox = [dates_quantiles[1], peaks_quantiles[1]]
    
        #upper right coordinate of the IQR-Box is defined as the 75% quartile of the peak timing (t75) 
        #and the 75% quartile of the peak discharges of all members (p75)
        upper_right_IQRbox = [dates_quantiles[3], peaks_quantiles[3]]
        
        ax2.plot([pd.to_datetime(lower_left_IQRbox[0]), pd.to_datetime(lower_left_IQRbox[0])],
                    [lower_left_IQRbox[1], upper_right_IQRbox[1]], color=color, alpha=alpha, lw=lw, zorder=zorder)
        
        ax2.plot([pd.to_datetime(lower_left_IQRbox[0]), pd.to_datetime(upper_right_IQRbox[0])],
                    [lower_left_IQRbox[1], lower_left_IQRbox[1]], color=color, alpha=alpha, lw=lw, zorder=zorder)
        
        ax2.plot([pd.to_datetime(upper_right_IQRbox[0]), pd.to_datetime(upper_right_IQRbox[0])],
                    [lower_left_IQRbox[1], upper_right_IQRbox[1]], color=color, alpha=alpha, lw=lw, zorder=zorder)
        
        ax2.plot([pd.to_datetime(lower_left_IQRbox[0]), pd.to_datetime(upper_right_IQRbox[0])],
                    [upper_right_IQRbox[1], upper_right_IQRbox[1]], color=color, alpha=alpha, lw=lw, zorder=zorder)
    
        # MEDIAN OF THE PEAK DISCHARGE:
        
        #horizontal line going from t0 to t100 representing the median of the peak discharge (p50) 
        #of all members of the ensemble forecast
        ax2.plot([pd.to_datetime(lower_left_pb[0]), pd.to_datetime(upper_right_pb[0])],
                    [peaks_quantiles[2], peaks_quantiles[2]], color=color, alpha=alpha, lw=lw, zorder=zorder)
    
        # MEDIAN OF THE PEAK TIMING:
        
        #vertical line going from p0 to p100 representing the median of the peak timing (t50)
        ax2.plot([pd.to_datetime(dates_quantiles[2]), pd.to_datetime(dates_quantiles[2])],
                    [lower_left_pb[1], upper_right_pb[1]], color=color, alpha=alpha, lw=lw)
    
        # MEDIAN VALUE: CROSS OF THE TWO MEDIANS
        
        median_value = ax2.plot(pd.to_datetime(dates_quantiles[2]), peaks_quantiles[2], '*', markersize=20, color=color, alpha=1.0, lw=lw, zorder=zorder+1,
                               markeredgecolor='black', markeredgewidth=1.5, label='($t_{50}$, $p_{50}$)')
        
        #PEAK-BOX AND IQR-BOX VOLUMES:
        box_volumes['pb'][i_group] = (hours_quantiles[4] - hours_quantiles[0])*(peaks_quantiles[4] - peaks_quantiles[0])
        box_volumes['iqr'][i_group] = (hours_quantiles[3] - hours_quantiles[1])*(peaks_quantiles[3] - peaks_quantiles[1])
        
        
        #Sharpness of the forecast: basically the box volumes multiplied by a coefficient 3.6/A
        #PB_full = (p100-p0)(t100-t0)*3.6/A  with A the area of the basin in km2
        #PB_IQR = (p75-p25)(t75-t25)*3.6/A
        
        sharpness['pb'][i_group] = ((peaks_quantiles[4] - peaks_quantiles[0]) * (hours_quantiles[4] - hours_quantiles[0])*3.6/A)
        sharpness['iqr'][i_group] = ((peaks_quantiles[3] - peaks_quantiles[1]) * (hours_quantiles[3] - hours_quantiles[1])*3.6/A)
    
    
        #Verification of peak median vs obs: we have to associate correctly the peak median to the correct obs peak
        #Dpeak = |p50-pobs|
        #Dtime = |t50-tobs|
        
        obs_peaks_assoc_to_group = pd.DataFrame(columns=['hour', 'date', 'peak'])
        
        #check if there is one obs peak inside the IQR box firstly, and then, if not, inside the peak box 
        for index, observed_peak in OBSevent_peaks.iterrows():
            
            #look if the observed peak is inside the IQR box, and in that case dont even look in the peak-box
            if (str(observed_peak['date']) >= dates_quantiles[1]) and (str(observed_peak['date']) <= dates_quantiles[3]) and (observed_peak['peak'] >= peaks_quantiles[1]) and (observed_peak['peak'] <= peaks_quantiles[3]) :
                obs_peaks_assoc_to_group.loc[index] = observed_peak
                break
            
            #look if the observed peak is inside the peak box
            if (str(observed_peak['date']) >= dates_quantiles[0]) and (str(observed_peak['date']) <= dates_quantiles[4]) and (observed_peak['peak'] >= peaks_quantiles[0]) and (observed_peak['peak'] <= peaks_quantiles[4]) :
                obs_peaks_assoc_to_group.loc[index] = observed_peak
            
        obs_peaks_assoc_to_group = obs_peaks_assoc_to_group.reset_index(drop=True)    
        
        # write hour and peak from observation peaks just found in a list to perform the distance calculation:
        obs_peaks_points = [(obs_peaks_assoc_to_group.hour[i], obs_peaks_assoc_to_group.peak[i]) for i in range(len(obs_peaks_assoc_to_group))]
            
        #if we have at least one obs peak found inside the boxes:
        if len(obs_peaks_assoc_to_group) != 0:
        
            #condition if more than one obs_peak was found inside the boxes: take the nearest one
            if (len(obs_peaks_assoc_to_group) > 1):
                nearest_obs_peak = closest_node((hours_quantiles[2], peaks_quantiles[2]), obs_peaks_points)
                                
            elif len(obs_peaks_assoc_to_group) == 1:
                nearest_obs_peak = (obs_peaks_assoc_to_group.hour[0], obs_peaks_assoc_to_group.peak[0])
        
            #calculate difference in peak magnitude and timing between the near observed peak and the median value of the peak-box:
            verification['d_peak'][i_group] = abs(peaks_quantiles[2] - nearest_obs_peak[1])
            verification['d_timing'][i_group] = abs(hours_quantiles[2] - nearest_obs_peak[0])
        
        else:
            verification['d_peak'][i_group] = float('nan')
            verification['d_timing'][i_group] = float('nan')
        
    
        i_group = i_group + 1
        
       
    ax1.grid(True)
    ax2.grid(True)
    
    
    fig.text(0.05, 0.5, 'Discharge [m$^3$ s$^{-1}$]', va='center', rotation='vertical', fontsize=18)

    #x axis ticks and limits
    days = mdates.DayLocator()
    hours = mdates.HourLocator()
    yearsFmt = mdates.DateFormatter('%Y-%m-%d') # %H:%M')
    
    ax1.xaxis.set_major_locator(days)
    ax1.xaxis.set_major_formatter(yearsFmt)
    ax1.xaxis.set_minor_locator(hours)
    
    ax2.xaxis.set_major_locator(days)
    ax2.xaxis.set_major_formatter(yearsFmt)
    ax2.xaxis.set_minor_locator(hours)
    
    # min and max on x axis
    datemin = np.datetime64(ens.date[0], 'm') - np.timedelta64(60, 'm')
    datemax = np.datetime64(ens.date[119], 'm') + np.timedelta64(25, 'm')
    ax2.set_xlim(datemin, datemax)
    ax1.set_xlim(datemin, datemax)
    #ax1.set_ylim(0, 99)
    #ax2.set_ylim(0, 99)
    
    # Shrink current axis by 20%
    #box1 = ax1.get_position()
    #box2 = ax2.get_position()
    #ax1.set_position([box1.x0, box1.y0, box1.width * 0.8, box1.height])
    #ax2.set_position([box2.x0, box2.y0, box2.width * 0.8, box2.height])
    
    fig.suptitle(f'Initialization: {sim_start}', y=0.925, fontsize=22)
    
    #Add warning levels lines: WL1 (), WL2, WL3 ???
    """
    ...TO DO...
    
    """
    
    #Legend:
    #fig.legend(handles=[runoff_member[0], l2[0], peak_member[0], median_value[0], c_median_value[0], peak_obs[0]], ncol=1, 
    #           loc=(0.0985,0.728), numpoints = 1,
    #       labels=['Runoff member', 'Runoff obs', '($t_i$, $p_i$) peaks of 1 PBM group', '($t_{50}$, $p_{50}$) of 1 PBM group', 
    #               '($t_{50}$, $p_{50}$) PBC', '($t_{obs}$, $p_{obs}$)'], fontsize=12); #loc=(0.66,0.66)
        
    #plt.rcParams.update({'font.size': 11});
    """
    #Tables for sharpness and verification for every group:
    if len(box_volumes) == 1:
        sharpness_table=r"\begin{tabular}{ c | c | c } \multicolumn{3}{c}{\textbf{Sharpness [mm]}} \\ & PB$_{FULL}$ [mm] & PB$_{IQR}$ [mm] \\\hline \textbf{Classic} & {%.2f} & {%.2f} \\\hline \textbf{Group 1} & {%.2f} & {%.2f} \end{tabular}" % (c_sharpness_pb, c_sharpness_iqr, sharpness['pb'][0], sharpness['iqr'][0])
        verification_table = r"\begin{tabular}{ c | c | c } \multicolumn{3}{c}{\textbf{Peak-box verification}} \\ & D$_{PEAK}$ [m$^3$ s$^{-1}$] & D$_{TIME}$ [h] \\\hline \textbf{Classic} & {%.2f} & {%.2f} \\\hline \textbf{Group 1} & {%.2f} & {%.2f} \end{tabular}" % (c_verification_d_peak, c_verification_d_timing, verification['d_peak'][0], verification['d_timing'][0])
    if len(box_volumes) == 2:
        sharpness_table=r"\begin{tabular}{ c | c | c } \multicolumn{3}{c}{\textbf{Sharpness [mm]}} \\ & PB$_{FULL}$ [mm] & PB$_{IQR}$ [mm] \\\hline \textbf{Classic} & {%.2f} & {%.2f} \\\hline \textbf{Group 1} & {%.2f} & {%.2f} \\\hline \textbf{Group 2} & {%.2f} & {%.2f} \end{tabular}" % (c_sharpness_pb, c_sharpness_iqr, sharpness['pb'][0], sharpness['iqr'][0], sharpness['pb'][1], sharpness['iqr'][1])
        verification_table = r"\begin{tabular}{ c | c | c } \multicolumn{3}{c}{\textbf{Peak-box verification}} \\ & D$_{PEAK}$ [m$^3$ s$^{-1}$] & D$_{TIME}$ [h] \\\hline \textbf{Classic} & {%.2f} & {%.2f} \\\hline \textbf{Group 1} & {%.2f} & {%.2f} \\\hline \textbf{Group 2} & {%.2f} & {%.2f} \end{tabular}" % (c_verification_d_peak, c_verification_d_timing, verification['d_peak'][0], verification['d_timing'][0], verification['d_peak'][1], verification['d_timing'][1])
    if len(box_volumes) == 3:
        sharpness_table=r"\begin{tabular}{ c | c | c } \multicolumn{3}{c}{\textbf{Sharpness [mm]}} \\ & PB$_{FULL}$ [mm] & PB$_{IQR}$ [mm] \\\hline \textbf{Classic} & {%.2f} & {%.2f} \\\hline \textbf{Group 1} & {%.2f} & {%.2f} \\\hline \textbf{Group 2} & {%.2f} & {%.2f} \\\hline \textbf{Group 3} & {%.2f} & {%.2f} \end{tabular}" % (c_sharpness_pb, c_sharpness_iqr, sharpness['pb'][0], sharpness['iqr'][0], sharpness['pb'][1], sharpness['iqr'][1], sharpness['pb'][2], sharpness['iqr'][2])
        verification_table = r"\begin{tabular}{ c | c | c } \multicolumn{3}{c}{\textbf{Peak-box verification}} \\ & D$_{PEAK}$ [m$^3$ s$^{-1}$] & D$_{TIME}$ [h] \\\hline \textbf{Classic} & {%.2f} & {%.2f} \\\hline \textbf{Group 1} & {%.2f} & {%.2f} \\\hline \textbf{Group 2} & {%.2f} & {%.2f} \\\hline \textbf{Group 3} & {%.2f} & {%.2f} \end{tabular}" % (c_verification_d_peak, c_verification_d_timing, verification['d_peak'][0], verification['d_timing'][0], verification['d_peak'][1], verification['d_timing'][1], verification['d_peak'][2], verification['d_timing'][2])
    if len(box_volumes) == 4:
        sharpness_table=r"\begin{tabular}{ c | c | c } \multicolumn{3}{c}{\textbf{Sharpness [mm]}} \\ & PB$_{FULL}$ [mm] & PB$_{IQR}$ [mm] \\\hline \textbf{Classic} & {%.2f} & {%.2f} \\\hline \textbf{Group 1} & {%.2f} & {%.2f} \\\hline \textbf{Group 2} & {%.2f} & {%.2f} \\\hline \textbf{Group 3} & {%.2f} & {%.2f} \\\hline \textbf{Group 4} & {%.2f} & {%.2f} \end{tabular}" % (c_sharpness_pb, c_sharpness_iqr, sharpness['pb'][0], sharpness['iqr'][0], sharpness['pb'][1], sharpness['iqr'][1], sharpness['pb'][2], sharpness['iqr'][2], sharpness['pb'][3], sharpness['iqr'][3])
        verification_table = r"\begin{tabular}{ c | c | c } \multicolumn{3}{c}{\textbf{Peak-box verification}} \\ & D$_{PEAK}$ [m$^3$ s$^{-1}$] & D$_{TIME}$ [h] \\\hline \textbf{Classic} & {%.2f} & {%.2f} \\\hline \textbf{Group 1} & {%.2f} & {%.2f} \\\hline \textbf{Group 2} & {%.2f} & {%.2f} \\\hline \textbf{Group 3} & {%.2f} & {%.2f} \\\hline \textbf{Group 4} & {%.2f} & {%.2f} \end{tabular}" % (c_verification_d_peak, c_verification_d_timing, verification['d_peak'][0], verification['d_timing'][0], verification['d_peak'][1], verification['d_timing'][1], verification['d_peak'][2], verification['d_timing'][2], verification['d_peak'][3], verification['d_timing'][3])
    if len(box_volumes) == 5:
        sharpness_table=r"\begin{tabular}{ c | c | c } \multicolumn{3}{c}{\textbf{Sharpness [mm]}} \\ & PB$_{FULL}$ [mm] & PB$_{IQR}$ [mm] \\\hline \textbf{Classic} & {%.2f} & {%.2f} \\\hline \textbf{Group 1} & {%.2f} & {%.2f} \\\hline \textbf{Group 2} & {%.2f} & {%.2f} \\\hline \textbf{Group 3} & {%.2f} & {%.2f} \\\hline \textbf{Group 4} & {%.2f} & {%.2f} \\\hline \textbf{Group 5} & {%.2f} & {%.2f} \end{tabular}" % (c_sharpness_pb, c_sharpness_iqr, sharpness['pb'][0], sharpness['iqr'][0], sharpness['pb'][1], sharpness['iqr'][1], sharpness['pb'][2], sharpness['iqr'][2], sharpness['pb'][3], sharpness['iqr'][3], sharpness['pb'][4], sharpness['iqr'][4])
        verification_table = r"\begin{tabular}{ c | c | c } \multicolumn{3}{c}{\textbf{Peak-box verification}} \\ & D$_{PEAK}$ [m$^3$ s$^{-1}$] & D$_{TIME}$ [h] \\\hline \textbf{Classic} & {%.2f} & {%.2f} \\\hline \textbf{Group 1} & {%.2f} & {%.2f} \\\hline \textbf{Group 2} & {%.2f} & {%.2f} \\\hline \textbf{Group 3} & {%.2f} & {%.2f} \\\hline \textbf{Group 4} & {%.2f} & {%.2f} \\\hline \textbf{Group 5} & {%.2f} & {%.2f} \end{tabular}" % (c_verification_d_peak, c_verification_d_timing, verification['d_peak'][0], verification['d_timing'][0], verification['d_peak'][1], verification['d_timing'][1], verification['d_peak'][2], verification['d_timing'][2], verification['d_peak'][3], verification['d_timing'][3], verification['d_peak'][4], verification['d_timing'][4])
    
    plt.text(1.0, 0.5, sharpness_table, ha="left", va="bottom", transform=ax2.transAxes, size=10)
    plt.text(1.0, 0.0, verification_table, ha="left", va="bottom", transform=ax2.transAxes, size=10)
    """
    ax1.text(0.992, 0.98, 'PBC', ha="right", va="top", transform=ax1.transAxes, size=15, 
             bbox=dict(facecolor='#32AAB5',edgecolor='none', alpha=0.3))
    
    ax2.text(0.992, 0.98, 'PBM', ha="right", va="top", transform=ax2.transAxes, size=15, 
             bbox=dict(facecolor='#32AAB5',edgecolor='none', alpha=0.3))

    return plt.show()