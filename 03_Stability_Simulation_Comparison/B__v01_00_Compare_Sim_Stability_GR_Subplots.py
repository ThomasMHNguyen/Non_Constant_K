# -*- coding: utf-8 -*-
"""
FILE NAME:      B__v01_00_Weighted_Amplitude_Subplots.py


COMPLEMENTARY
SCRIPT(S)/
FILE(S):        N/A


DESCRIPTION:    This script will plot the modes with the highest growth rate predicted from 
                linear stability analysis and plot the mode with the highest extracted growth
                rate from the Brownian simulations for all rigidity profiles.

INPUT
FILES(S):       

1)              .CSV files that list the growth rate predicted from stability analysis and their
                corresponding mode classification.
2)              .CSV files that list the growth rates extracted from the non-linear simulations and
                their corresponding mode classification
                

OUTPUT
FILES(S):       

1)              .PNG file that compares the predicted and actual growth rates for the 3 different rigidity profiles.
2)              .PDF file that compares the predicted and actual growth rates for the 3 different rigidity profiles.
3)              .EPS file that compares the predicted and actual growth rates for the 3 different rigidity profiles.



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
import os, string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
import matplotlib.patches  as mpatches
output_dir = 'C:/Users/super/OneDrive - University of California, Davis/Research/00_Projects/01_Non_Constant_K/00_Scripts/03_Stability_Simulation_Comparison/04_Dissertation/Sim_Stability_Predict/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
    
#%% Functions

def get_most_unstable_mode_stability_data(mode_df):
    max_gr = mode_df.groupby('Mu_bar').Eigenvalues_real.transform(max)
    return mode_df[mode_df.Eigenvalues_real == max_gr]

def get_most_unstable_mode_simulations(mode_df):
    max_gr = mode_df.groupby('Mu_bar').Actual_Growth_Rate.transform(max)
    return mode_df[mode_df.Actual_Growth_Rate == max_gr]
#%% Read in data
K_constant_suffix = 'K_constant'
K_dirac_left_l_stiff_suffix = 'K_dirac_left_l_stiff'
K_error_function_suffix = 'K_error_function'

### Read in stability map data ###
stability_mode_type_dir = 'C:/Users/super/OneDrive - University of California, Davis/Research/00_Projects/01_Non_Constant_K/00_Scripts/02_Stability_Analysis/03_Run_Results/Mode_Type_Data/'
K_constant_mode_type_df = pd.read_csv(os.path.join(stability_mode_type_dir,K_constant_suffix,'{}_mode_types.csv'.format(K_constant_suffix)),
                                      index_col = 0,header = 0)
K_dirac_left_l_stiff_type_df = pd.read_csv(os.path.join(stability_mode_type_dir,K_dirac_left_l_stiff_suffix,'{}_mode_types.csv'.format(K_dirac_left_l_stiff_suffix)),
                                           index_col = 0,header = 0)
K_error_function_mode_type_df = pd.read_csv(os.path.join(stability_mode_type_dir,K_error_function_suffix,'{}_mode_types.csv'.format(K_error_function_suffix)),
                                            index_col = 0,header = 0)

K_constant_mode_type_df = K_constant_mode_type_df[K_constant_mode_type_df['Mu_bar']>= 5000]
K_dirac_left_l_stiff_type_df = K_dirac_left_l_stiff_type_df[K_dirac_left_l_stiff_type_df['Mu_bar']>= 5000]
K_error_function_mode_type_df = K_error_function_mode_type_df[K_error_function_mode_type_df['Mu_bar']>= 5000]

### Chose only most unstable mode at each mu_bar value ###
K_constant_max_mode_type_df = get_most_unstable_mode_stability_data(K_constant_mode_type_df)
K_dirac_left_l_stiff_max_mode_type_df = get_most_unstable_mode_stability_data(K_dirac_left_l_stiff_type_df)
K_error_function_max_mode_type_df = get_most_unstable_mode_stability_data(K_error_function_mode_type_df)


### Read in ensemble averaged data for lp/L=100 ###
lp_100_ensmb_sim_results_dir = 'C:/Users/super/OneDrive - University of California, Davis/Research/00_Projects/01_Non_Constant_K/00_Scripts/03_Stability_Simulation_Comparison/03_Run_Results/Brownian_Extensional_lp_100_adj/'
K_constant_lp_100_ensb_df= pd.read_csv(os.path.join(lp_100_ensmb_sim_results_dir,K_constant_suffix,'N_200_ensemble_avg_regression.csv'),
                                               index_col = 0,header = 0)
K_dirac_left_l_stiff_lp_100_ensb_df = pd.read_csv(os.path.join(lp_100_ensmb_sim_results_dir,K_dirac_left_l_stiff_suffix,'N_200_ensemble_avg_regression.csv'),
                                               index_col = 0,header = 0)
K_error_function_lp_100_ensb_df = pd.read_csv(os.path.join(lp_100_ensmb_sim_results_dir,K_error_function_suffix,'N_200_ensemble_avg_regression.csv'),
                                               index_col = 0,header = 0)

### Read in ensemble averaged data for lp/L=10 ###
# lp_10_ensmb_sim_results_dir = 'C:/Users/super/OneDrive - University of California, Davis/Research/Scripts/Brownian_Stability_Analysis/Analysis_Results/Brownian_Extensional_lp_L_10/'
# K_constant_lp_10_ensb_df= pd.read_csv(os.path.join(lp_100_ensmb_sim_results_dir,K_constant_suffix,'N_100_ensemble_average_regression_data.csv'),
#                                                index_col = 0,header = 0)
# K_constant_lp_10_ensb_df = K_constant_lp_10_ensb_df[K_constant_lp_10_ensb_df['Mu_bar'] == 50000]
# K_dirac_left_l_stiff_lp_10_ensb_df = pd.read_csv(os.path.join(lp_10_ensmb_sim_results_dir,K_dirac_left_l_stiff_suffix,'N_100_ensemble_average_regression_data.csv'),
#                                                index_col = 0,header = 0)
# K_dirac_left_l_stiff_lp_10_ensb_df = K_dirac_left_l_stiff_lp_10_ensb_df[K_dirac_left_l_stiff_lp_10_ensb_df['Mu_bar'] == 50000]
# K_error_function_lp_10_ensb_df = pd.read_csv(os.path.join(lp_10_ensmb_sim_results_dir,K_error_function_suffix,'N_100_ensemble_average_regression_data.csv'),
#                                                index_col = 0,header = 0)
# K_error_function_lp_10_ensb_df = K_error_function_lp_10_ensb_df[K_error_function_lp_10_ensb_df['Mu_bar'] == 50000]

### Read in ensemble averaged data for lp/L=100 ###
K_constant_lp_100_sim_mode_type_df = get_most_unstable_mode_simulations(K_constant_lp_100_ensb_df)
K_dirac_left_l_stiff_lp_100_sim_mode_type_df = get_most_unstable_mode_simulations(K_dirac_left_l_stiff_lp_100_ensb_df)
K_error_function_lp_100_sim_mode_type_df = get_most_unstable_mode_simulations(K_error_function_lp_100_ensb_df)

### Read in ensemble averaged data for lp/L=10 ###
# K_constant_lp_10_sim_mode_type_df = get_most_unstable_mode_simulations(K_constant_lp_10_ensb_df)
# K_dirac_left_l_stiff_lp_10_sim_mode_type_df = get_most_unstable_mode_simulations(K_dirac_left_l_stiff_lp_10_ensb_df)
# K_error_function_lp_10_sim_mode_type_df = get_most_unstable_mode_simulations(K_error_function_lp_10_ensb_df)

#%% Plot most unstable modes on each 

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": "Times",
    'text.latex.preamble': r'\usepackage{amsmath}'})

color_palette = ["#574D68","#FA9F42","#0B6E4F","#DB7F8E"]

fig,axes = plt.subplots(ncols = 3,figsize = (10,7),layout = 'constrained',sharey = True)
# fig.subplots_adjust(wspace = 0.05)
# bright_palette = sns.color_palette("bright")
### Plot Stability Predictions ###
sns.lineplot(x = 'Mu_bar',y = 'Eigenvalues_real',data = K_constant_max_mode_type_df,
                ax = axes[0],hue = 'Mode Number',palette = color_palette,linewidth = 4,legend = False)
sns.lineplot(x = 'Mu_bar',y = 'Eigenvalues_real',data = K_dirac_left_l_stiff_max_mode_type_df,
                ax = axes[1],hue = 'Mode Number',palette = color_palette,linewidth = 4,legend = False)
sns.lineplot(x = 'Mu_bar',y = 'Eigenvalues_real',data = K_error_function_max_mode_type_df,
                ax = axes[2],hue = 'Mode Number',palette = color_palette,linewidth = 4,legend = False)

### Plot Ensemble Average Simulation Results for lp_L=100 ###
sns.scatterplot(x = 'Mu_bar',y = 'Actual_Growth_Rate',data = K_constant_lp_100_sim_mode_type_df ,
                ax = axes[0],hue = 'Mode Number',palette = color_palette,s = 100,marker='d',legend = False)
sns.scatterplot(x = 'Mu_bar',y = 'Actual_Growth_Rate',data = K_dirac_left_l_stiff_lp_100_sim_mode_type_df ,
                ax = axes[1],hue = 'Mode Number',palette = color_palette,s = 100,marker='d',legend = False)
sns.scatterplot(x = 'Mu_bar',y = 'Actual_Growth_Rate',data = K_error_function_lp_100_sim_mode_type_df ,
                ax = axes[2],hue = 'Mode Number',palette = color_palette,s = 100,marker='d',legend = False)

subfig_labels = [r"$\textbf{(a)}$",r"$\textbf{(b)}$",r"$\textbf{(c)}$"]
for n,ax in enumerate(axes):
    ax.ticklabel_format(axis="x", style="sci", scilimits=(4,4))
    ax.set_xlim(3000,51000)
    ax.set_ylim(1.5,5.1)
    ax.set_xticks(np.arange(0.5e4,5.1e4,0.5e4))
    # ax.set_yticks(np.linspace(2,14,7))
    ax.set_yticks(np.linspace(2,5,4))
    [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_ticklabels()) if (i-1) % 2 != 0] #Hide every other label
    # [l.set_visible(False) for (i,l) in enumerate(ax.yaxis.get_ticklabels()) if (i-1) % 2 != 0] #Hide every other label
    ax.tick_params(axis='x', which='both',labelsize = 13, size = 5,length = 3,
                   direction = 'in',pad = 5)
    ax.tick_params(axis='y', which='both',labelsize = 13, size = 5,length = 3,
                   direction = 'in',pad = 5)
    ax.text(2500,5.2,'{}'.format(subfig_labels[n]),
            size=17)
    ax.xaxis.offsetText.set_fontsize(0)
    ax.set_xlabel(r"$\bar{\mu} \times 10^{4}$",fontsize = 15,labelpad = 5)
    ax.set_aspect(np.diff(ax.get_xlim())/np.diff(ax.get_ylim()))
    ax.set_ylabel(None)
    ax.set_xticklabels((np.arange(0.5e4,5.1e4,0.5e4)/1e4).astype(int))

legend_elements = []
for i,v in enumerate(['1','2','3','Other']):
    legend_elements.append(mpatches.Patch([0], [0],color = color_palette[i], label=v))
    
# legend_elements.append(Line2D([0], [0],linestyle = '-',linewidth = 2,
#                               color = 'black', label="Stability Analysis"))
# legend_elements.append(Line2D([0], [0],linestyle = '',marker = 'd',
#                               color = 'black', label="Simulations"))

                     
fig.legend(handles = legend_elements,loc='lower center', bbox_to_anchor=(0.5, 0.07),
                prop={'size': 13},title= "Mode Number ",title_fontsize = 15,ncol = 6)
# fig.supxlabel(r"$\bar{\mu} \: \times 10^{4}$",fontsize = 15,y = 0.08,x = 0.52)
fig.supylabel(r"$\sigma_{i}$ [Growth Rate]",fontsize = 15,x=-0.045)

# fig.savefig(os.path.join(output_dir,'stability_simulation_comparison_labeled.png'),
#             bbox_inches = 'tight',dpi = 600)
# fig.savefig(os.path.join(output_dir,'stability_simulation_comparison_labeled.pdf'),
#             format = 'pdf',bbox_inches = 'tight',dpi = 600)
# fig.savefig(os.path.join(output_dir,'stability_simulation_comparison_labeled.eps'),
#             bbox_inches = 'tight',format = 'eps',dpi = 600)
fig.savefig(os.path.join(output_dir,'stability_simulation_comparison.png'),
            bbox_inches = 'tight',dpi = 600)
fig.savefig(os.path.join(output_dir,'stability_simulation_comparison.pdf'),
            format = 'pdf',bbox_inches = 'tight',dpi = 600)
fig.savefig(os.path.join(output_dir,'stability_simulation_comparison.eps'),
            bbox_inches = 'tight',format = 'eps',dpi = 600)
plt.show()
