# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 21:17:08 2022

@author: super
"""

import os
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


K_constant_suffix = 'K_constant'
K_dirac_left_l_stiff_suffix = 'K_dirac_left_l_stiff'
K_error_function_suffix = 'K_error_function'

lp_100_average_dir = 'C:/Users/super/OneDrive - University of California, Davis/School/UCD_Files/Work/Scripts/Brownian_Stability_Analysis/Analysis_Results/Brownian_Extensional_lp_L_100/'
lp_10_average_dir = 'C:/Users/super/OneDrive - University of California, Davis/School/UCD_Files/Work/Scripts/Brownian_Stability_Analysis/Analysis_Results/Brownian_Extensional_lp_L_10/'

### Lp = 100 directories ###
K_constant_lp_100_dir = os.path.join(lp_100_average_dir,K_constant_suffix)
K_dirac_left_l_stiff_lp_100_dir = os.path.join(lp_100_average_dir,K_dirac_left_l_stiff_suffix)
K_error_function_lp_100_dir = os.path.join(lp_100_average_dir,K_error_function_suffix)

### lp = 10 directories ###
K_constant_lp_10_dir = os.path.join(lp_10_average_dir,K_constant_suffix)
K_dirac_left_l_stiff_lp_10_dir = os.path.join(lp_10_average_dir,K_dirac_left_l_stiff_suffix)
K_error_function_lp_10_dir = os.path.join(lp_10_average_dir,K_error_function_suffix)


output_dir = 'C:/Users/super/OneDrive - University of California, Davis/School/UCD_Files/Work/Scripts/Manuscript_Figures/01_Non_Constant_K/07_ln_rms_amplitude_data/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

#%% Read in Data
mu_bar = 50000

### Lp = 100 data ###
K_constant_lp_100_amplt_data = pd.read_csv(os.path.join(K_constant_lp_100_dir,'N_100_ensemble_average_amplitude_unfiltered_data.csv'),
                                           index_col = 0,header = 0)
K_constant_lp_100_amplt_data = K_constant_lp_100_amplt_data[K_constant_lp_100_amplt_data['Mu_bar'] == mu_bar].reset_index(drop = True)
K_dirac_left_l_stiff_lp_100_amplt_data = pd.read_csv(os.path.join(K_dirac_left_l_stiff_lp_100_dir,'N_100_ensemble_average_amplitude_unfiltered_data.csv'),
                                           index_col = 0,header = 0)
K_dirac_left_l_stiff_lp_100_amplt_data = K_dirac_left_l_stiff_lp_100_amplt_data[K_dirac_left_l_stiff_lp_100_amplt_data['Mu_bar'] == mu_bar].reset_index(drop = True)
K_error_function_lp_100_amplt_data = pd.read_csv(os.path.join(K_error_function_lp_100_dir,'N_100_ensemble_average_amplitude_unfiltered_data.csv'),
                                           index_col = 0,header = 0)
K_error_function_lp_100_amplt_data = K_error_function_lp_100_amplt_data[K_error_function_lp_100_amplt_data['Mu_bar'] == mu_bar].reset_index(drop = True)

### Lp = 10 data ###
K_constant_lp_10_amplt_data = pd.read_csv(os.path.join(K_constant_lp_10_dir,'N_100_ensemble_average_amplitude_unfiltered_data.csv'),
                                           index_col = 0,header = 0)
K_constant_lp_10_amplt_data = K_constant_lp_10_amplt_data[K_constant_lp_10_amplt_data['Mu_bar'] == mu_bar].reset_index(drop = True)
K_dirac_left_l_stiff_lp_10_amplt_data = pd.read_csv(os.path.join(K_dirac_left_l_stiff_lp_10_dir,'N_100_ensemble_average_amplitude_unfiltered_data.csv'),
                                           index_col = 0,header = 0)
K_dirac_left_l_stiff_lp_10_amplt_data = K_dirac_left_l_stiff_lp_10_amplt_data[K_dirac_left_l_stiff_lp_10_amplt_data['Mu_bar'] == mu_bar].reset_index(drop = True)
K_error_function_lp_10_amplt_data = pd.read_csv(os.path.join(K_error_function_lp_10_dir,'N_100_ensemble_average_amplitude_unfiltered_data.csv'),
                                           index_col = 0,header = 0)
K_error_function_lp_10_amplt_data = K_error_function_lp_10_amplt_data[K_error_function_lp_10_amplt_data['Mu_bar'] == mu_bar].reset_index(drop = True)


#%% Functions
def write_mode_num_ranking(eigenval,all_sorted_eigenvals):
    ranking  = int(np.where(all_sorted_eigenvals == eigenval)[0][0]) + 1
    return ranking


def calculate_mode_ranking(amplt_data_df):
    mode_group_ranking = amplt_data_df.groupby(by = ['Mode Number'])
    for group in mode_group_ranking.groups.keys():
            group_df = mode_group_ranking.get_group(group)
            if group_df['Linear_Eigenvals_real'].unique()[0] > 1:
                eigvals_sorted = np.sort(group_df['Linear_Eigenvals_real'].unique())
                group_df['Linear_Eigenvals_Ranking'] = group_df['Linear_Eigenvals_real'].apply(lambda x: write_mode_num_ranking(x,eigvals_sorted))
                group_df['Mode Number|Ranking'] = group_df[['Mode Number','Linear_Eigenvals_Ranking']].apply(lambda row: '|'.join(row.values.astype(str)), axis=1)
                amplt_data_df.loc[group_df.index,'Linear_Eigenvals_Ranking'] = group_df['Linear_Eigenvals_Ranking'].astype(int)
                amplt_data_df.loc[group_df.index,'Mode Number|Ranking'] = group_df['Mode Number|Ranking']
            else:
                group_df['Linear_Eigenvals_Ranking'] = 1
                group_df['Mode Number|Ranking'] = group_df[['Mode Number','Linear_Eigenvals_Ranking']].apply(lambda row: '|'.join(row.values.astype(str)), axis=1)
                amplt_data_df.loc[group_df.index,'Linear_Eigenvals_Ranking'] = group_df['Linear_Eigenvals_Ranking'].astype(int)
                amplt_data_df.loc[group_df.index,'Mode Number|Ranking'] = group_df['Mode Number|Ranking']
    return amplt_data_df

#%% Rank each eigenvalue in terms of unstable for each mode number 

K_constant_lp_100_amplt_data = calculate_mode_ranking(K_constant_lp_100_amplt_data)
K_dirac_left_l_stiff_lp_100_amplt_data = calculate_mode_ranking(K_dirac_left_l_stiff_lp_100_amplt_data)
K_error_function_lp_100_amplt_data = calculate_mode_ranking(K_error_function_lp_100_amplt_data)

K_constant_lp_10_amplt_data = calculate_mode_ranking(K_constant_lp_10_amplt_data)
K_dirac_left_l_stiff_lp_10_amplt_data = calculate_mode_ranking(K_dirac_left_l_stiff_lp_10_amplt_data)
K_error_function_lp_10_amplt_data = calculate_mode_ranking(K_error_function_lp_10_amplt_data)


time_vals_10 = np.array([v for i,v in enumerate(K_constant_lp_100_amplt_data['Time'].unique()[1:]) if i % 1 == 0])     
time_vals_first = np.array([v for i,v in enumerate(K_constant_lp_100_amplt_data['Time'].unique()[1:]) if i % 1 == 0])     
### Filter for only every other 10 points ###
K_constant_lp_100_amplt_data = K_constant_lp_100_amplt_data[K_constant_lp_100_amplt_data['Time'].isin(time_vals_10)]
K_dirac_left_l_stiff_lp_100_amplt_data = K_dirac_left_l_stiff_lp_100_amplt_data[K_dirac_left_l_stiff_lp_100_amplt_data['Time'].isin(time_vals_10)]
K_error_function_lp_100_amplt_data = K_error_function_lp_100_amplt_data[K_error_function_lp_100_amplt_data['Time'].isin(time_vals_10)]

K_constant_lp_10_amplt_data = K_constant_lp_10_amplt_data[K_constant_lp_10_amplt_data['Time'].isin(time_vals_first)]
K_dirac_left_l_stiff_lp_10_amplt_data = K_dirac_left_l_stiff_lp_10_amplt_data[K_dirac_left_l_stiff_lp_10_amplt_data['Time'].isin(time_vals_first)]
K_error_function_lp_10_amplt_data = K_error_function_lp_10_amplt_data[K_error_function_lp_10_amplt_data['Time'].isin(time_vals_first)]
#%% Plot subplots-normal time scale

### Draw Brownian noise floor lines ###
lp_100_floor = np.log(np.sqrt((1**2)*((1 + 0.5)**4*np.pi**4)**-1*(1/100)))
lp_10_floor = np.log(np.sqrt((1**2)*((1 + 0.5)**4*np.pi**4)**-1*(1/10)))

#Denote which plot is which
lp_vals = [100,10]
col_titles = [r'$\ell_{{p}}/L={}$'.format(lp_vals[i]) for i in range(0,2)]      
row_headers = [r'$B_{1}(s)$',r'$B_{2}(s)$',r'$B_{3}(s)$']

#Hue order
h_order = ['2','3','Other']
omit_palette = sns.color_palette("bright")[1:4]
fig,axes = plt.subplots(nrows = 3,ncols = 2,figsize = (9,16),sharey = 'row',sharex = 'col')
fig.subplots_adjust(wspace = 0.05,hspace = -0.20)   

### lp = 100 ###
sns.lineplot(y = 'Amplitude_Coefficient_RMS_LN',x = 'Adjusted Time',hue = 'Mode Number',size = 'Linear_Eigenvals_Ranking',ci = None,legend = False,
                                  palette = omit_palette,hue_order = h_order,data = K_constant_lp_100_amplt_data,sizes = {1:2.5,2:5},ax = axes[0,0])
sns.lineplot(y = 'Amplitude_Coefficient_RMS_LN',x = 'Adjusted Time',hue = 'Mode Number',size = 'Linear_Eigenvals_Ranking',ci = None,legend = False,
                                  palette = omit_palette,hue_order = h_order,data = K_dirac_left_l_stiff_lp_100_amplt_data,sizes = {1:2.5,2:5},ax = axes[1,0])
sns.lineplot(y = 'Amplitude_Coefficient_RMS_LN',x = 'Adjusted Time',hue = 'Mode Number',size = 'Linear_Eigenvals_Ranking',ci = None,legend = False,
                                  palette = omit_palette,hue_order = h_order,data = K_error_function_lp_100_amplt_data,sizes = {1:2.5,2:5},ax = axes[2,0])

sns.lineplot(y = 'Amplitude_Coefficient_RMS_LN',x = 'Adjusted Time',hue = 'Mode Number',size = 'Linear_Eigenvals_Ranking',ci = None,legend = False,
                                  palette = omit_palette,hue_order = h_order,data = K_constant_lp_10_amplt_data,sizes = {1:2.5,2:5},ax = axes[0,1])
sns.lineplot(y = 'Amplitude_Coefficient_RMS_LN',x = 'Adjusted Time',hue = 'Mode Number',size = 'Linear_Eigenvals_Ranking',ci = None,legend = False,
                                  palette = omit_palette,hue_order = h_order,data = K_dirac_left_l_stiff_lp_10_amplt_data,sizes = {1:2.5,2:5},ax = axes[1,1])
sns.lineplot(y = 'Amplitude_Coefficient_RMS_LN',x = 'Adjusted Time',hue = 'Mode Number',size = 'Linear_Eigenvals_Ranking',ci = None,legend = False,
                                  palette = omit_palette,hue_order = h_order,data = K_error_function_lp_10_amplt_data,sizes = {1:2.5,2:5},ax = axes[2,1])

#Hide every other label
n = 2

for n_row,ax_row in enumerate(axes):
    for n_col,ax_col in enumerate(ax_row):
        ax_col.set_xlim(-0.5,5.5)
        ax_col.set_ylim(-10,3)
        ax_col.set_xticks(np.linspace(0,5,6))
        ax_col.set_yticks(np.linspace(-10,2,7))
        ax_col.tick_params(axis='x', which='both',labelsize = 30, 
                           size = 10,width = 3,direction = 'in')
        ax_col.tick_params(axis='y', which='both',labelsize = 30, 
                           size = 10,width = 3,direction = 'in')
        ax_col.set_aspect((5.5--0.5)/(3--10))
        [l.set_visible(False) for (i,l) in enumerate(ax_col.xaxis.get_ticklabels()) if i % n != 0]
        [l.set_visible(False) for (i,l) in enumerate(ax_col.yaxis.get_ticklabels()) if i % n != 0]
        ax_col.set(xlabel=None,ylabel=None)
        if n_col == 0:
            ax_col.axhline(y = lp_100_floor,color = 'gray',linestyle = 'dashed',linewidth = 3)
            ax_col.axvline(x = 0.75,color = 'magenta',linestyle = 'dotted',linewidth = 3)
        elif n_col == 1:
            ax_col.axhline(y = lp_10_floor,color = 'gray',linestyle = 'dashed',linewidth = 3)
            ax_col.axvline(x = 0.25,color = 'magenta',linestyle = 'dotted',linewidth = 3)
        

### Label each column based on persistence length ###
for ax, col in zip(axes[0], col_titles):
    ax.set_title(col,fontsize = 30,pad = 20)

### Label each plot based on rigidity profile ###
for i in range(0,2):
    for ax, row in zip(axes[:,i], row_headers):
        ax.text(x = 0.65,y = 0.85,s = row,transform=ax.transAxes,
                        size=30, weight='bold')
        
legend_elements = []
for i,v in enumerate(['2','3','Other']):
    legend_elements.append(Line2D([0], [0],color = omit_palette[i], label=v,
                              linewidth = 2.5))
    
fig.supxlabel(r"$\bar{\mu}t^{Br}$",fontsize = 35,y = 0.08,x = 0.5125)
fig.supylabel(r"$\ln\left(\sqrt{\langle a^{2}\rangle}\right)$",fontsize = 35,y = 0.5,x = -0.0625)

axes[1,1].legend(handles = legend_elements,loc='center left', 
                bbox_to_anchor=(1, 0.5),prop={'size': 20},title= "Mode Number").get_title().set_fontsize("25")
# fig.savefig(os.path.join(output_dir,'simulation_amplitude_lp_comparison.png'),bbox_inches = 'tight',dpi = 200)
# fig.savefig(os.path.join(output_dir,'simulation_amplitude_lp_comparison.pdf'),bbox_inches = 'tight',dpi = 200)
# fig.savefig(os.path.join(output_dir,'simulation_amplitude_lp_comparison.eps'),
#             bbox_inches = 'tight',format = 'eps',dpi = 200)
plt.show()

#%% Plot subplots-Broownian time scale

### Draw Brownian noise floor lines ###
lp_100_floor = np.log(np.sqrt((1**2)*((1 + 0.5)**4*np.pi**4)**-1*(1/100)))
lp_10_floor = np.log(np.sqrt((1**2)*((1 + 0.5)**4*np.pi**4)**-1*(1/10)))

#Denote which plot is which
lp_vals = [100,10]
col_titles = [r'$\ell_{{p}}/L={}$'.format(lp_vals[i]) for i in range(0,2)]      
row_headers = [r'$B_{1}(s)$',r'$B_{2}(s)$',r'$B_{3}(s)$']

#Hue order
h_order = ['2','3','Other']
omit_palette = sns.color_palette("bright")[1:4]
fig,axes = plt.subplots(nrows = 3,ncols = 2,figsize = (9,16),sharey = 'row',sharex = 'col')
fig.subplots_adjust(wspace = 0.05,hspace = -0.20)   

### lp = 100 ###
sns.lineplot(y = 'Amplitude_Coefficient_RMS_LN',x = 'Time',hue = 'Mode Number',size = 'Linear_Eigenvals_Ranking',ci = None,legend = False,
                                  palette = omit_palette,hue_order = h_order,data = K_constant_lp_100_amplt_data,sizes = {1:2.5,2:5},ax = axes[0,0])
sns.lineplot(y = 'Amplitude_Coefficient_RMS_LN',x = 'Time',hue = 'Mode Number',size = 'Linear_Eigenvals_Ranking',ci = None,legend = False,
                                  palette = omit_palette,hue_order = h_order,data = K_dirac_left_l_stiff_lp_100_amplt_data,sizes = {1:2.5,2:5},ax = axes[1,0])
sns.lineplot(y = 'Amplitude_Coefficient_RMS_LN',x = 'Time',hue = 'Mode Number',size = 'Linear_Eigenvals_Ranking',ci = None,legend = False,
                                  palette = omit_palette,hue_order = h_order,data = K_error_function_lp_100_amplt_data,sizes = {1:2.5,2:5},ax = axes[2,0])

sns.lineplot(y = 'Amplitude_Coefficient_RMS_LN',x = 'Time',hue = 'Mode Number',size = 'Linear_Eigenvals_Ranking',ci = None,legend = False,
                                  palette = omit_palette,hue_order = h_order,data = K_constant_lp_10_amplt_data,sizes = {1:2.5,2:5},ax = axes[0,1])
sns.lineplot(y = 'Amplitude_Coefficient_RMS_LN',x = 'Time',hue = 'Mode Number',size = 'Linear_Eigenvals_Ranking',ci = None,legend = False,
                                  palette = omit_palette,hue_order = h_order,data = K_dirac_left_l_stiff_lp_10_amplt_data,sizes = {1:2.5,2:5},ax = axes[1,1])
sns.lineplot(y = 'Amplitude_Coefficient_RMS_LN',x = 'Time',hue = 'Mode Number',size = 'Linear_Eigenvals_Ranking',ci = None,legend = False,
                                  palette = omit_palette,hue_order = h_order,data = K_error_function_lp_10_amplt_data,sizes = {1:2.5,2:5},ax = axes[2,1])

#Hide every other label
n = 2

for n_row,ax_row in enumerate(axes):
    for n_col,ax_col in enumerate(ax_row):
        ax_col.ticklabel_format(axis="x", style="sci", scilimits=(-5,-5))
        ax_col.xaxis.offsetText.set_fontsize(0)
        ax_col.set_xlim(-1e-5,1e-4)
        ax_col.set_ylim(-10,3)
        ax_col.set_xticks(np.linspace(0,1e-4,5))
        
        ax_col.set_yticks(np.linspace(-10,2,7))
        ax_col.tick_params(axis='x', which='both',labelsize = 30, 
                           size = 10,width = 3,direction = 'in')
        ax_col.tick_params(axis='y', which='both',labelsize = 30, 
                           size = 10,width = 3,direction = 'in')
        ax_col.set_aspect((1e-4--1e-5)/(3--10))
        [l.set_visible(False) for (i,l) in enumerate(ax_col.xaxis.get_ticklabels()) if i % n != 0]
        [l.set_visible(False) for (i,l) in enumerate(ax_col.yaxis.get_ticklabels()) if i % n != 0]
        # ax_col.set_xticklabels((np.array([0,5e-5,10e-5])/1e-5).astype(int))
        ax_col.set(xlabel=None,ylabel=None)
        if n_col == 0:
            ax_col.axhline(y = lp_100_floor,color = 'gray',linestyle = 'dashed',linewidth = 3)
            ax_col.axvline(x = 0.75/50000,color = 'magenta',linestyle = 'dotted',linewidth = 3)
        elif n_col == 1:
            ax_col.axhline(y = lp_10_floor,color = 'gray',linestyle = 'dashed',linewidth = 3)
            ax_col.axvline(x = 0.25/50000,color = 'magenta',linestyle = 'dotted',linewidth = 3)
        

### Label each column based on persistence length ###
for ax, col in zip(axes[0], col_titles):
    ax.set_title(col,fontsize = 30,pad = 20)

### Label each plot based on rigidity profile ###
for i in range(0,2):
    for ax, row in zip(axes[:,i], row_headers):
        ax.text(x = 0.65,y = 0.85,s = row,transform=ax.transAxes,
                        size=30, weight='bold')
        
legend_elements = []
for i,v in enumerate(['2','3','Other']):
    legend_elements.append(Line2D([0], [0],color = omit_palette[i], label=v,
                              linewidth = 2.5))
    
fig.supxlabel(r"$t^{Br} \times 10^{-5}$",fontsize = 35,y = 0.04,x = 0.5125)
fig.supylabel(r"$\ln\left(\sqrt{\langle a^{2}\rangle}\right)$",fontsize = 35,y = 0.5,x = -0.0625)

axes[1,1].legend(handles = legend_elements,loc='center left', 
                bbox_to_anchor=(1, 0.5),prop={'size': 20},title= "Mode Number").get_title().set_fontsize("25")
plt.show() 