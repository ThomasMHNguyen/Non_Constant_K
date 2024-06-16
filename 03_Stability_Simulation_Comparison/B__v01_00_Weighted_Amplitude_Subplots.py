# -*- coding: utf-8 -*-
"""
FILE NAME:      B__v01_00_Weighted_Amplitude_Subplots.py


COMPLEMENTARY
SCRIPT(S)/
FILE(S):        N/A


DESCRIPTION:    Based on the distribution of excited modes across all Brownian ensembles,
                this script will create a stacked area plot that shows this distribution for all
                rigidity profiles.

INPUT
FILES(S):       

1)              .CSV files that count the percentage of excited modes across all ensembles for
                a particular rigidity profile.
2)              .CSV file that lists the onset of instability for each mode number of a particular
                rigidity profile.
                

OUTPUT
FILES(S):       

1)              .PNG file that plots the distribution of excited modes for the 3 different rigidity profiles.
2)              .PDF file that plots the distribution of excited modes for the 3 different rigidity profiles.
3)              .EPS file that plots the distribution of excited modes for the 3 different rigidity profiles.



INPUT
ARGUMENT(S):    

None
CREATED:        21Oct21

MODIFICATIONS
LOG:
    
            
LAST MODIFIED
BY:             Thomas Nguyen

PYTHON
VERSION USED
TO WRITE
SCRIPT:         3.8.11

VERSION:        1.0

AUTHOR(S):      Thomas Nguyen

STATUS:         Working

TO DO LIST:     None

NOTE(S):        N/A

"""

import sys, os, string
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

### Import amplitude data ###
K_constant_emsb_data_dir = 'C:/Users/super/OneDrive - University of California, Davis/Research/00_Projects/01_Non_Constant_K/00_Scripts/03_Stability_Simulation_Comparison/03_Run_Results/Brownian_Extensional_lp_100_adj/K_constant/'
K_dirac_left_l_stiff_emsb_data_dir = 'C:/Users/super/OneDrive - University of California, Davis/Research/00_Projects/01_Non_Constant_K/00_Scripts/03_Stability_Simulation_Comparison/03_Run_Results/Brownian_Extensional_lp_100_adj/K_dirac_left_l_stiff/'
K_error_function_amp_emsb_dir = 'C:/Users/super/OneDrive - University of California, Davis/Research/00_Projects/01_Non_Constant_K/00_Scripts/03_Stability_Simulation_Comparison/03_Run_Results/Brownian_Extensional_lp_100_adj/K_error_function/'

K_constant_ensmb_df = pd.read_csv(os.path.join(K_constant_emsb_data_dir,'N_200_ensemble_mode_count.csv'),index_col = 0,header = 0)
K_dirac_left_l_stiff_ensmb_df = pd.read_csv(os.path.join(K_dirac_left_l_stiff_emsb_data_dir,'N_200_ensemble_mode_count.csv'),index_col = 0,header = 0)
K_error_function_ensmb_df = pd.read_csv(os.path.join(K_error_function_amp_emsb_dir,'N_200_ensemble_mode_count.csv'),index_col = 0,header = 0)


### Import unstable buckling threshold data ###
K_constant_thres_dir = 'C:/Users/super/OneDrive - University of California, Davis/Research/00_Projects/01_Non_Constant_K/00_Scripts/02_Stability_Analysis/03_Run_Results/Mode_Number_Threshold/K_constant/'
K_dirac_left_l_stiff_thres_dir = 'C:/Users/super/OneDrive - University of California, Davis/Research/00_Projects/01_Non_Constant_K/00_Scripts/02_Stability_Analysis/03_Run_Results/Mode_Number_Threshold/K_dirac_left_l_stiff/'
K_error_function_thres_dir = 'C:/Users/super/OneDrive - University of California, Davis/Research/00_Projects/01_Non_Constant_K/00_Scripts/02_Stability_Analysis/03_Run_Results/Mode_Number_Threshold/K_error_function/'

K_constant_thres_df = pd.read_csv(os.path.join(K_constant_thres_dir,'mode_number_threshold_K_constant.csv'),index_col = 0,header = 0)
K_dirac_left_l_stiff_thres_df = pd.read_csv(os.path.join(K_dirac_left_l_stiff_thres_dir,'mode_number_threshold_K_dirac_left_l_stiff.csv'),index_col = 0,header = 0)
K_error_function_thres_df = pd.read_csv(os.path.join(K_error_function_thres_dir,'mode_number_threshold_K_error_function.csv'),index_col = 0,header = 0)

output_dir = 'C:/Users/super/OneDrive - University of California, Davis/Research/00_Projects/01_Non_Constant_K/00_Scripts/03_Stability_Simulation_Comparison/04_Dissertation/Exited_Modes/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
#%% Format for Stacked Area Format 


def format_stacked_area_data(ensmb_df):
    mu_bar_vals = sorted(ensmb_df['Mu_bar'].unique())
    diff_buckling_modes = sorted(ensmb_df['Mode Number'].unique())
    all_coeff_vals = []
    for mode_num in diff_buckling_modes:
        lst_coeff_vals = []
        for mu_bar in mu_bar_vals:
            fil_ensmb_df = ensmb_df[(ensmb_df['Mu_bar'] == mu_bar) & \
                                                          (ensmb_df['Mode Number'] == mode_num)]
            coeff_per = fil_ensmb_df['Single Percentage'].values[0]
            lst_coeff_vals.append(coeff_per)
        all_coeff_vals.append(lst_coeff_vals)
    return all_coeff_vals
    
#%% Plot figures on subplots

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": "Times",
    'text.latex.preamble': r'\usepackage{amsmath}'})

from matplotlib.lines import Line2D
import matplotlib.patches  as mpatches
### Sort Data for all profiles ###

K_constant_sorted_data = format_stacked_area_data(K_constant_ensmb_df)
K_dirac_left_l_stiff_sorted_data = format_stacked_area_data(K_dirac_left_l_stiff_ensmb_df)
K_error_function_sorted_data = format_stacked_area_data(K_error_function_ensmb_df)

all_sorted_data = [K_constant_sorted_data,K_dirac_left_l_stiff_sorted_data,K_error_function_sorted_data]

### Plot Data ###
fig,axes = plt.subplots(ncols = 3,figsize = (10,7),layout = 'constrained',sharey = True)
# bright_palette = sns.color_palette("bright")
color_palette = ["#574D68","#FA9F42","#0B6E4F","#DB7F8E"]


axes[0].stackplot(sorted(K_constant_ensmb_df['Mu_bar'].unique()),
                  all_sorted_data[0], 
                  labels=sorted(K_constant_ensmb_df['Mode Number'].unique()),
                  colors = color_palette)

axes[1].stackplot(sorted(K_dirac_left_l_stiff_ensmb_df['Mu_bar'].unique()),
                  all_sorted_data[1], 
                  labels=sorted(K_dirac_left_l_stiff_ensmb_df['Mode Number'].unique()),
                  colors = color_palette)

axes[2].stackplot(sorted(K_error_function_ensmb_df['Mu_bar'].unique()),
                  all_sorted_data[2], 
                  labels=sorted(K_error_function_ensmb_df['Mode Number'].unique()),
                  colors = color_palette)
axes[0].legend().remove()
axes[2].legend().remove()
### Format each plot ###
for ax in axes:    
    ax.ticklabel_format(axis="x", style="sci", scilimits=(4,4))
    ax.set_xticks(np.arange(0.5e4,5.1e4,0.5e4))
    ax.set_yticks(np.linspace(0,100,6))
    ax.set_xlim(0.45e4,5.0e4)
    ax.set_ylim(0,100)
    [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_ticklabels()) if (i-1) % 2 != 0] #Hide every other label
    ax.tick_params(axis='both', which='major', labelsize=13,size = 5,length = 5,
                   direction = 'out',pad = 5)
    ax.xaxis.offsetText.set_fontsize(0)
    ax.set_xticklabels((np.arange(0.5e4,5.1e4,0.5e4)/1e4).astype(int))
    ax.set_xlabel(r"$\bar{\mu} \times 10^{4}$",fontsize = 15,labelpad = 5)
    ax.set_aspect(np.diff(ax.get_xlim())/np.diff(ax.get_ylim()))


### Plot vertical line on each subplot to indicate theoretical value of instability for second and third mode ###
for i in range(2,5):
    fil_K_constant_thres_df = K_constant_thres_df[K_constant_thres_df['Mode Number'] == i]
    fil_K_dirac_left_l_stiff_thres_df = K_dirac_left_l_stiff_thres_df[K_dirac_left_l_stiff_thres_df['Mode Number'] == i]
    fil_K_error_function_thres_df = K_error_function_thres_df[K_error_function_thres_df['Mode Number'] == i]
    
    axes[0].axvline(x=fil_K_constant_thres_df['Eigenvalues_real'].unique()[0], color='k', linestyle='dashdot',linewidth = 2)
    axes[1].axvline(x=fil_K_dirac_left_l_stiff_thres_df['Eigenvalues_real'].unique()[0], color='k', linestyle='dashdot',linewidth = 2)
    if i != 4:
        axes[2].axvline(x=fil_K_error_function_thres_df['Eigenvalues_real'].unique()[0], color='k', linestyle='dashdot',linewidth = 2)


### Label unstable thresholds ###
axes[0].text(x = -2.41,y = 1.03,s = "(2)",transform=ax.transAxes,
                    size=15)
axes[0].text(x = -2.19,y = 1.03,s = "(3)",transform=ax.transAxes,
                    size=15)
axes[0].text(x = -1.89,y = 1.03,s = "(4)",transform=ax.transAxes,
                    size=15)

axes[1].text(x = -1.21,y = 1.03,s = "(2)",transform=ax.transAxes,
                    size=15)
axes[1].text(x = -1.03,y = 1.03,s = "(3)",transform=ax.transAxes,
                    size=15)
axes[1].text(x = -0.76,y = 1.03,s = "(4)",transform=ax.transAxes,
                    size=15)

axes[2].text(x = 0.10,y = 1.03,s = "(2)",transform=ax.transAxes,
                    size=15)
axes[2].text(x = 0.45,y = 1.03,s = "(3)",transform=ax.transAxes,
                    size=15)

### Label subplot figures ###

axes[0].text(-9000,95,r"$\textbf{(a)}$",
                    size=15)
axes[1].text(-2000,95,r"$\textbf{(b)}$",
                    size=15)
axes[2].text(-2000,95,r"$\textbf{(c)}$",
                    size=15)
    
### Distinguish each plot ###
# axes[0].set_title(r"$\kappa(s) = 1$",fontsize = 25,pad = 15)
# axes[1].set_title(r"$\kappa(s) = 1- \frac{1}{2} e^{-100\left(s + \frac{1}{4}\right)^{2}} $",fontsize = 25,pad = 15 )
# axes[2].set_title("$\kappa(s) = 2 + $erf$(10s) $",fontsize = 25,pad = 15)

### Add legend for plot ###
handles = [mpatches.Patch([0], [0], color=color_palette[i], label = v,
                linewidth = 2,linestyle = '-') for i,v in enumerate(sorted(K_constant_ensmb_df['Mode Number'].unique()))]
fig.legend(handles = handles,loc='lower center', 
                bbox_to_anchor=(0.5,0.07),prop={'size': 13},title= "Mode Number",
                ncol = 4,title_fontsize = 15)
# axes[0].legend().set_visible(False)
# axes[2].legend().set_visible(False)
# fig.supxlabel(r"$\bar{\mu} \: \times 10^{4}$",fontsize = 15,y = -0.01,x = 0.52)
fig.supylabel(r"Percentage",fontsize = 15,x=0.015)

fig.savefig(os.path.join(output_dir,'most_excited_mode_distributions.png'),
            bbox_inches = 'tight',dpi = 600)
fig.savefig(os.path.join(output_dir,'most_excited_mode_distributions.pdf'),
            format = 'pdf',bbox_inches = 'tight',dpi = 600)
fig.savefig(os.path.join(output_dir,'most_excited_mode_distributions.eps'),
            bbox_inches = 'tight',format = 'eps',dpi = 600)
plt.show()
