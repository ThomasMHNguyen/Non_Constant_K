# -*- coding: utf-8 -*-
"""
FILE NAME:      B__v04_00_Plot_Filament_Position_Tension_Subplots.py

COMPLEMENTARY
SCRIPT(S)/
FILE(S):        N/A

DESCRIPTION:    This script will plot filament position and tension configurations
                at specific timepoints on multiple subplots on a single canvas. 

INPUT
FILES(S):       None

OUTPUT
FILES(S):       1) .PNG file that plots the filament positional configuration at 
                various timepoints.
                2) .PDF file that plots the filament positional configuration at 
                various timepoints. 
                3) .EPS file that plots the filament positional configuration at 
                various timepoints.
                4) .PNG file that plots the filament tensional configuration at 
                various timepoints.
                5) .PDF file that plots the filament tensional configuration at 
                various timepoints. 
                6) .EPS file that plots the filament tensional configuration at 
                various timepoints.          


INPUT
ARGUMENT(S):    None

CREATED:        02Jan21

MODIFICATIONS
LOG:

04Mar22:        Added functionality to include generating .PDF and .EPS files 
                of filament position and tension.

    
LAST MODIFIED
BY:             Thomas Nguyen

PYTHON
VERSION USED
TO WRITE
SCRIPT:         3.8.11

VERSION:        1.1

AUTHOR(S):      Thomas Nguyen

STATUS:         Working

TO DO LIST:     None

NOTE(S):        N/A

"""

import os, string
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import special
import pandas as pd

output_dir = 'C:/Users/super/OneDrive - University of California, Davis/Research/00_Projects/01_Non_Constant_K/00_Scripts/01_Filament_Movement/04_Dissertation/Position_Tension/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
        
        
#directories of position and tension data
K_constant_dir = 'C:/Users/super/OneDrive - University of California, Davis/Research/00_Projects/01_Non_Constant_K/00_Scripts/01_Filament_Movement/03_Run_Results/Shear_Flow_Manuscript/K_constant/Mu_bar_500000/perturb_O4/'
K_error_function_dir = 'C:/Users/super/OneDrive - University of California, Davis/Research//00_Projects/01_Non_Constant_K/00_Scripts/01_Filament_Movement/03_Run_Results/Shear_Flow_Manuscript/K_error_function/Mu_bar_500000/perturb_O4/'
K_dirac_left_l_stiff_dir = 'C:/Users/super/OneDrive - University of California, Davis/Research//00_Projects/01_Non_Constant_K/00_Scripts/01_Filament_Movement/03_Run_Results/Shear_Flow_Manuscript/K_dirac_left_l_stiff/Mu_bar_500000/perturb_O4/'


#Read in position files
K_constant_pos = np.load(os.path.join(K_constant_dir,'filament_allstate.npy'))
K_dirac_left_l_stiff_pos = np.load(os.path.join(K_dirac_left_l_stiff_dir,'filament_allstate.npy'))
K_error_function_pos = np.load(os.path.join(K_error_function_dir,'filament_allstate.npy'))

#Read in tension files
K_constant_ten = np.load(os.path.join(K_constant_dir,'filament_tension.npy'))
K_dirac_left_l_stiff_ten = np.load(os.path.join(K_dirac_left_l_stiff_dir,'filament_tension.npy'))
K_error_function_ten = np.load(os.path.join(K_error_function_dir,'filament_tension.npy'))

#Read in parameter files
K_constant_params = pd.read_csv(os.path.join(K_constant_dir,'parameter_values.csv'),index_col = 0,header = 0)
K_error_function_params = pd.read_csv(os.path.join(K_error_function_dir,'parameter_values.csv'),index_col = 0,header = 0)
K_dirac_left_l_stiff_params = pd.read_csv(os.path.join(K_dirac_left_l_stiff_dir,'parameter_values.csv'),index_col = 0,header = 0)


#Read in angle files
K_constant_angle = np.load(os.path.join(K_constant_dir,'filament_angle.npy'))
K_dirac_left_l_stiff_angle = np.load(os.path.join(K_dirac_left_l_stiff_dir,'filament_angle.npy'))
K_error_function_angle = np.load(os.path.join(K_error_function_dir,'filament_angle.npy'))

snapshot_times = [0.00,2.25,3.25,5.464]
n = 2  # Keeps every 2nd label

K_constant_col_code = '#32CD32'
K_dirac_left_l_stiff_col_code = '#1E90FF'
K_error_function_col_code = '#FF7F50'


#%% Plot tile of filament positions

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": "Times",
    'text.latex.preamble': r'\usepackage{amsmath}'})



fig,axes = plt.subplots(ncols = 4,nrows = 3,figsize = (10,7),layout = 'constrained',sharey = 'row',sharex = 'col')
# fig.subplots_adjust(wspace = -0.60,hspace = 0.050)


col_titles = ['t  = {}'.format(snapshot_times[i]) for i in range(0,len(snapshot_times))]
row_headers = [r'$B_{1}(s)$',r'$B_{2}(s)$',r'$B_{3}(s)$']


offset = -0.45
channel_height = 0.75
y = np.linspace(-1.5*channel_height,1.5*channel_height,200)
ux = 0.4*y.copy() + offset


for n_row,ax_row in enumerate(axes):
    for n_col,ax_col in enumerate(ax_row):
        adj_time_K_constant = int(snapshot_times[n_col]/(float(
            K_constant_params.loc['dt','Value'])*int(K_constant_params.loc['Adjusted Scaling','Value'])))
        
        adj_time_K_error_function = int(snapshot_times[n_col]/(float(
            K_error_function_params.loc['dt','Value'])*int(K_error_function_params.loc['Adjusted Scaling','Value'])))
        
        adj_time_K_dirac_left_l_stiff = int(snapshot_times[n_col]/(float(
            K_dirac_left_l_stiff_params.loc['dt','Value'])*int(K_dirac_left_l_stiff_params.loc['Adjusted Scaling','Value'])))
        
        if n_row == 0:
            ax_col.plot(K_constant_pos[:,0,adj_time_K_constant],
                    K_constant_pos[:,1,adj_time_K_constant],color = K_constant_col_code,linewidth = 2)
        elif n_row == 1:
            ax_col.plot(K_dirac_left_l_stiff_pos[:,0,adj_time_K_dirac_left_l_stiff],
                    K_dirac_left_l_stiff_pos[:,1,adj_time_K_dirac_left_l_stiff],color = K_dirac_left_l_stiff_col_code,linewidth = 2)
        elif n_row == 2:
            ax_col.plot(K_error_function_pos[:,0,adj_time_K_error_function],
                    K_error_function_pos[:,1,adj_time_K_error_function],color = K_error_function_col_code,linewidth = 2)
                   
        
        ax_col.set_xlim(-0.6,0.6)
        ax_col.set_ylim(-0.6,0.6)
        ax_col.set_xticks(np.linspace(-0.5,0.5,5))
        ax_col.set_yticks(np.linspace(-0.5,0.5,5))
        # ax_col.set_xlabel(r"$x$",fontsize = 13,labelpad = 5)
        ax_col.tick_params(axis='both', which='both',labelsize = 13, 
                           length = 5,size = 3,direction = 'in')
        ax_col.set_aspect(np.diff(ax_col.get_xlim())/np.diff(ax_col.get_ylim()))
        [l.set_visible(False) for (i,l) in enumerate(ax_col.xaxis.get_ticklabels()) if i % n != 0]
        [l.set_visible(False) for (i,l) in enumerate(ax_col.yaxis.get_ticklabels()) if i % n != 0]

        ax_col.set_aspect(np.diff(ax_col.get_xlim())/np.diff(ax_col.get_ylim()))
### Quiver plots to show flow direction ###
for i in range(3):
    axes[i,0].plot(ux,y,color = 'blue',linewidth = 1.2,
                 linestyle = 'solid',alpha = 0.2)
    axes[i,0].axvline(x = offset,ymin = 0,ymax = 1,
                   color = 'blue',linewidth = 1.2,
                   linestyle = 'solid',alpha = 0.2)
    slicing_factor = 20
    y_subset = y[::slicing_factor].copy()
    x_subset = ux[::slicing_factor].copy()
    quiv_x = np.zeros_like(y_subset) +offset
    quiv_y = y_subset.copy()
    quiv_ar_x = x_subset.copy() - offset
    quiv_ar_y = np.zeros_like(y_subset)
    axes[i,0].quiver(quiv_x,quiv_y,quiv_ar_x,quiv_ar_y,angles='xy', 
                   scale_units='xy', scale=1,color = 'blue',alpha = 0.2)


#Denote differences in column and rows
for ax, col in zip(axes[0], col_titles):
    ax.set_title(col,fontsize = 19,pad = 5)
    
for ax, row in zip(axes[:,0], row_headers):
    ax.text(x = 0.25,y = 0.45,s = row,
                    size=17, weight='bold')
    
fig.supxlabel(r"$x$",fontsize = 17,y = -0.04,x = 0.5125)
fig.supylabel(r"$y$",fontsize = 17,y = 0.5,x = 0.005)
# axes[0,0].set_ylabel(r"$y$",fontsize = 13,labelpad = 5)
# axes[1,0].set_ylabel(r"$y$",fontsize = 13,labelpad = 5)
# axes[2,0].set_ylabel(r"$y$",fontsize = 13,labelpad = 5)
axes[0,0].text(-1.05,0.755,r'$\textbf{(a)}$',size=19)
    
fig.savefig(os.path.join(output_dir,'filament_position_shear_simulations.png'),bbox_inches = 'tight',dpi = 600)
fig.savefig(os.path.join(output_dir,'filament_position_shear_simulations.pdf'),format = 'pdf',bbox_inches = 'tight',dpi = 600)
fig.savefig(os.path.join(output_dir,'filament_position_shear_simulations.eps'),
            bbox_inches = 'tight',format = 'eps',dpi = 600)
plt.show()


#%% Plot tile of filament tension

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": "Times",
    'text.latex.preamble': r'\usepackage{amsmath}'})



fig,axes = plt.subplots(ncols = 4,nrows = 3,figsize = (10,7),layout = 'constrained',sharey = 'row',sharex = 'col')


s_K_constant = np.linspace(float(K_constant_params.loc['Filament s start','Value']),
                           float(K_constant_params.loc['Filament s end','Value']),
                           int(K_constant_params.loc['N','Value']))

s_K_dirac_left_l_stiff = np.linspace(float(K_constant_params.loc['Filament s start','Value']),
                           float(K_constant_params.loc['Filament s end','Value']),
                           int(K_constant_params.loc['N','Value']))

s_error_function = np.linspace(float(K_constant_params.loc['Filament s start','Value']),
                           float(K_constant_params.loc['Filament s end','Value']),
                           int(K_constant_params.loc['N','Value']))



col_titles = ['t  = {}'.format(snapshot_times[i]) for i in range(0,len(snapshot_times))]
row_headers = [r'$B_{1}(s)$',r'$B_{2}(s)$',r'$B_{3}(s)$']


for n_row,ax_row in enumerate(axes):
    for n_col,ax_col in enumerate(ax_row):
        adj_time_K_constant = int(snapshot_times[n_col]/(float(
            K_constant_params.loc['dt','Value'])*int(K_constant_params.loc['Adjusted Scaling','Value'])))
        
        adj_time_K_error_function = int(snapshot_times[n_col]/(float(
            K_error_function_params.loc['dt','Value'])*int(K_error_function_params.loc['Adjusted Scaling','Value'])))
        
        adj_time_K_dirac_left_l_stiff = int(snapshot_times[n_col]/(float(
            K_dirac_left_l_stiff_params.loc['dt','Value'])*int(K_dirac_left_l_stiff_params.loc['Adjusted Scaling','Value'])))
        
        if n_row == 0:
            ax_col.plot(s_K_constant,K_constant_ten[:,adj_time_K_constant],color = K_constant_col_code,linewidth = 2)
        elif n_row == 1:
            ax_col.plot(s_K_dirac_left_l_stiff,K_dirac_left_l_stiff_ten[:,adj_time_K_dirac_left_l_stiff],color = K_dirac_left_l_stiff_col_code,linewidth = 2)
        elif n_row == 2:
            ax_col.plot(s_error_function,K_error_function_ten[:,adj_time_K_error_function],color = K_error_function_col_code,linewidth =2)
        ax_col.ticklabel_format(axis="y", style="sci", scilimits=(3,3))
        ax_col.yaxis.offsetText.set_fontsize(0)
        ax_col.set_xlim(-0.6,0.6)
        ax_col.set_ylim(-2400,2400)
        ax_col.set_xticks(np.linspace(-0.5,0.5,5))
        ax_col.set_yticks(np.linspace(-2200,2200,5))
        ax_col.tick_params(axis='both', which='both',labelsize = 13, 
                           length = 5,size = 3,direction = 'in')
        ax_col.set_aspect(np.diff(ax_col.get_xlim())/np.diff(ax_col.get_ylim()))
        [l.set_visible(False) for (i,l) in enumerate(ax_col.xaxis.get_ticklabels()) if i % n != 0]
        [l.set_visible(False) for (i,l) in enumerate(ax_col.yaxis.get_ticklabels()) if i % n != 0]

        ax_col.set_aspect(np.diff(ax_col.get_xlim())/np.diff(ax_col.get_ylim()))


#Denote differences in column and rows
for ax, col in zip(axes[0], col_titles):
    ax.set_title(col,fontsize = 19,pad = 5)
    
for ax, row in zip(axes[:,0], row_headers):
    ax.text(x = 0.25,y = 1.8e3,s = row,
                    size=17, weight='bold')
    
fig.supxlabel(r"$s$",fontsize = 17,y = -0.04,x = 0.5125)
fig.supylabel(r"$T(s) \times 10^{3}$",fontsize = 17,y = 0.5,x = 0.005)
axes[0,0].text(-1.05,3e3,r'$\textbf{(b)}$',size=19)
    
fig.savefig(os.path.join(output_dir,'filament_tension_shear_simulations.png'),bbox_inches = 'tight',dpi = 600)
fig.savefig(os.path.join(output_dir,'filament_tension_shear_simulations.pdf'),format = 'pdf',bbox_inches = 'tight',dpi = 600)
fig.savefig(os.path.join(output_dir,'filament_tension_shear_simulations.eps'),
            bbox_inches = 'tight',format = 'eps',dpi = 600)
plt.show()

        