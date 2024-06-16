# -*- coding: utf-8 -*-
"""
FILE NAME:      B__v05_00_Plot_Energy_Stresses_subplots.py

COMPLEMENTARY
SCRIPT(S)/
FILE(S):        N/A

DESCRIPTION:    This script will plot the end-to-end length deficit, elastic energy,
                and stress tensor components at various timepoints on multiple subplots
                on a single canvas to compare the three different rigidity profiles.

INPUT
FILES(S):       None

OUTPUT
FILES(S):       

1)              .PNG file that plots the filament end-to-end length deficit at 
                various timepoints for 3 different rigidity profiles.
2)              .PDF file that plots the filament end-to-end length deficit at 
                various timepoints for 3 different rigidity profiles.
3)              .EPS file that plots the filament end-to-end length deficit at 
                various timepoints for 3 different rigidity profiles.
4)              .PNG file that plots the filament elastic energy at 
                various timepoints for 3 different rigidity profiles.
5)              .PDF file that plots the filament elastic energy at 
                various timepoints for 3 different rigidity profiles.
6)              .EPS file that plots the filament elastic energy at 
                various timepoints for 3 different rigidity profiles.
7)              .PNG file that plots the filament stress tensor components at 
                various timepoints for 3 different rigidity profiles.
8)              .PDF file that plots the filament stress tensor components at 
                various timepoints for 3 different rigidity profiles.
9)              .EPS file that plots the filament stress tensor components at 
                various timepoints for 3 different rigidity profiles.           


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

import os,string
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib as mpl
# rc_fonts = {
#     "text.usetex": True,

#     'text.latex.preamble': [r"\usepackage{bm}",  r"\usepackage{siunitx}",r"\sisetup{detect-all}",
#                             r"\usepackage{DejaVuSans}",r"\usepackage{sansmath}",r"\sansmath"],
# }
# mpl.rcParams.update(rc_fonts)
import matplotlib.pyplot as plt

### Import data ###
K_constant_dir = 'C:/Users/super/OneDrive - University of California, Davis/Research/00_Projects/01_Non_Constant_K/00_Scripts/01_Filament_Movement/03_Run_Results/Shear_Flow_Manuscript/K_constant/Mu_bar_500000/perturb_O4/'
K_dirac_left_l_stiff_dir = 'C:/Users/super/OneDrive - University of California, Davis/Research/00_Projects/01_Non_Constant_K/00_Scripts/01_Filament_Movement/03_Run_Results/Shear_Flow_Manuscript/K_dirac_left_l_stiff/Mu_bar_500000/perturb_O4/'
K_error_function_dir = 'C:/Users/super/OneDrive - University of California, Davis/Research/00_Projects/01_Non_Constant_K/00_Scripts/01_Filament_Movement/03_Run_Results/Shear_Flow_Manuscript/K_error_function/Mu_bar_500000/perturb_O4/'
K_constant_rigid_dir = 'C:/Users/super/OneDrive - University of California, Davis/Research/00_Projects/01_Non_Constant_K/00_Scripts/01_Filament_Movement/03_Run_Results/Shear_Flow_Manuscript/K_constant/Mu_bar_500000/no_perturb/'

### Load in position data ###
K_constant_loc_data = np.load(os.path.join(K_constant_dir,'filament_allstate.npy'))
K_dirac_left_l_stiff_loc_data  = np.load(os.path.join(K_dirac_left_l_stiff_dir,'filament_allstate.npy'))
K_error_function_loc_data  = np.load(os.path.join(K_error_function_dir,'filament_allstate.npy'))
K_constant_rigid_loc_data = np.load(os.path.join(K_constant_rigid_dir,'filament_allstate.npy'))

### Load in Elastic Energy Data ###
K_constant_elastic_data = np.load(os.path.join(K_constant_dir,'filament_elastic_energy.npy'))
K_dirac_left_l_stiff_elastic_data  = np.load(os.path.join(K_dirac_left_l_stiff_dir,'filament_elastic_energy.npy'))
K_error_function_elastic_data  = np.load(os.path.join(K_error_function_dir,'filament_elastic_energy.npy'))
K_constant_rigid_elastic_data = np.load(os.path.join(K_constant_rigid_dir,'filament_elastic_energy.npy'))


### Load in Stress Data ###
K_constant_stress_data = np.load(os.path.join(K_constant_dir,'filament_stress_all.npy'))
K_dirac_left_l_stiff_stress_data  = np.load(os.path.join(K_dirac_left_l_stiff_dir,'filament_stress_all.npy'))
K_error_function_stress_data  = np.load(os.path.join(K_error_function_dir,'filament_stress_all.npy'))
K_constant_rigid_stress_data = np.load(os.path.join(K_constant_rigid_dir,'filament_stress_all.npy'))

### Load in parameter data ###
K_constant_params = pd.read_csv(os.path.join(K_constant_dir,'parameter_values.csv'),index_col = 0,header = 0)
K_dirac_left_l_stiff_params = pd.read_csv(os.path.join(K_dirac_left_l_stiff_dir,'parameter_values.csv'),index_col = 0,header = 0)
K_error_function_params = pd.read_csv(os.path.join(K_error_function_dir,'parameter_values.csv'),index_col = 0,header = 0)
K_constant_rigid_params = pd.read_csv(os.path.join(K_constant_rigid_dir,'parameter_values.csv'),index_col = 0,header = 0)

output_dir = 'C:/Users/super/OneDrive - University of California, Davis/Research/00_Projects/01_Non_Constant_K/00_Scripts/01_Filament_Movement/04_Dissertation/Energy_Stresses/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

### Specify colors for each item ###
rigid_col_code = '#FF00FF'
K_constant_col_code = '#32CD32'
K_dirac_left_l_stiff_col_code = '#1E90FF'
K_error_function_col_code = '#FF7F50'
vert_col_code = '#696969'
#%% Compare Filament Compression for end-to-end length, elastic energy, first normal stress, and shear stress
#magenta, dodger blue, lime green, coral
#


def calculate_norm_ee_length(position_data):
    ee_length = 1-((np.sqrt(np.sum((position_data[-1,:,:] - position_data[0,:,:])**2,axis = 0)))/1)
    return ee_length

K_constant_ee_length = calculate_norm_ee_length(K_constant_loc_data)
K_dirac_left_l_stiff_ee_length = calculate_norm_ee_length(K_dirac_left_l_stiff_loc_data)
K_error_function_ee_length = calculate_norm_ee_length(K_error_function_loc_data)
K_constant_rigid_ee_length = calculate_norm_ee_length(K_constant_rigid_loc_data)

K_constant_time_val = np.round(np.arange(0,float(K_constant_params.loc['Total Run Time','Value'])+float(K_constant_params.loc['dt','Value']),
                                float(K_constant_params.loc['dt','Value'])*float(K_constant_params.loc['Adjusted Scaling','Value'])),3)
K_dirac_left_l_stiff_time_val = np.round(np.arange(0,float(K_dirac_left_l_stiff_params.loc['Total Run Time','Value'])+float(K_dirac_left_l_stiff_params.loc['dt','Value']),
                                float(K_error_function_params.loc['dt','Value'])*float(K_error_function_params.loc['Adjusted Scaling','Value'])),3)
K_error_function_time_val = np.round(np.arange(0,float(K_error_function_params.loc['Total Run Time','Value'])+float(K_error_function_params.loc['dt','Value']),
                                float(K_error_function_params.loc['dt','Value'])*float(K_error_function_params.loc['Adjusted Scaling','Value'])),3)
K_constant_rigid_time_val = np.round(np.arange(0,float(K_constant_rigid_params.loc['Total Run Time','Value'])+float(K_constant_rigid_params.loc['dt','Value']),
                                float(K_constant_rigid_params.loc['dt','Value'])*float(K_constant_rigid_params.loc['Adjusted Scaling','Value'])),3)

### Define how much data to show ###
vert_time = 2.732
full_rot_time = 5.464
cut_off_time = full_rot_time
cot_idx = np.where(K_constant_time_val == cut_off_time)[0][0]+1

#%% Plot end-to-end-length and elastic energy

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": "Times",
    'text.latex.preamble': r'\usepackage{amsmath}'})


n = 2  # Keeps every 2nd label
fig,axes = plt.subplots(ncols = 2,figsize = (10,7),layout = 'constrained',sharex = True)
# fig.subplots_adjust(wspace = 0.35)


### Filament Compression ###
axes[0].axvline(x = vert_time,color = vert_col_code,linestyle = 'dashdot',
                linewidth = 3)
axes[0].plot(K_constant_time_val[:cot_idx],K_constant_ee_length[:cot_idx],color = K_constant_col_code,
             linewidth = 2,label = r"$B_{1}(s)$")
axes[0].plot(K_dirac_left_l_stiff_time_val[:cot_idx],K_dirac_left_l_stiff_ee_length[:cot_idx],color = K_dirac_left_l_stiff_col_code,
             linewidth = 2,label = r"$B_{2}(s)$")
axes[0].plot(K_error_function_time_val[:cot_idx],K_error_function_ee_length[:cot_idx],color = K_error_function_col_code,linewidth = 2,
             label = r"$B_{3}(s)$")
axes[0].plot(K_constant_rigid_time_val[:cot_idx],K_constant_rigid_ee_length[:cot_idx],color = rigid_col_code,
             linewidth = 3,linestyle = 'dotted')

axes[0].set_ylim(-0.02,0.2)
axes[0].set_xlim(-0.25,5.5)
axes[0].set_yticks(np.linspace(0,0.2,5))
axes[0].set_xticks(np.linspace(0,5,3))
axes[0].ticklabel_format(axis="x", style="sci", scilimits=(0,0))
axes[0].tick_params(axis='both', which='both', labelsize = 13, length = 5,size = 5,
                    pad = 5,direction = 'in')
axes[0].set_ylabel(r"$L_{ee}^{*}$",fontsize=15,labelpad = 5)
axes[0].set_aspect(np.diff(axes[0].get_xlim())/np.diff(axes[0].get_ylim()))
[l.set_visible(False) for (i,l) in enumerate(axes[0].yaxis.get_ticklabels()) if i % n != 0]




### Filament Elastic Energy ###
axes[1].axvline(x = vert_time,color = vert_col_code,
                linestyle = 'dashdot',linewidth = 3)
axes[1].plot(K_constant_time_val[:cot_idx],K_constant_elastic_data[:cot_idx],color = K_constant_col_code,
             linewidth = 2,label = r"$B_{1}(s)$")
axes[1].plot(K_dirac_left_l_stiff_time_val[:cot_idx],K_dirac_left_l_stiff_elastic_data[:cot_idx],
             color = K_dirac_left_l_stiff_col_code,linewidth = 2,label = r"$B_{2}(s)$")
axes[1].plot(K_error_function_time_val[:cot_idx],K_error_function_elastic_data[:cot_idx],
             color = K_error_function_col_code,linewidth = 2,label = r"$B_{3}(s)$")
axes[1].plot(K_constant_rigid_time_val[:cot_idx],K_constant_rigid_elastic_data[:cot_idx],
             color = rigid_col_code,linewidth = 3,linestyle = 'dotted')
axes[1].set_ylim(-10,100)
axes[1].set_xlim(-0.25,5.5)
axes[1].set_yticks(np.linspace(0,100,5))
axes[1].set_xticks(np.linspace(0,5,3))
axes[1].ticklabel_format(axis="x", style="sci", scilimits=(0,0))
axes[1].tick_params(axis='both', which='both', labelsize = 13, length = 5,size = 5,
                    pad = 5,direction = 'in')
axes[1].set_ylabel(r"$E_{elastic}$",fontsize=15,labelpad = 5)
axes[1].set_aspect(np.diff(axes[1].get_xlim())/np.diff(axes[1].get_ylim()))
[l.set_visible(False) for (i,l) in enumerate(axes[1].yaxis.get_ticklabels()) if i % n != 0]

axes[0].set_xlabel(r"$t$",size = 15,labelpad = 5)
axes[1].set_xlabel(r"$t$",size = 15,labelpad = 5)
### common x-axis labe & legend ###
# fig.supxlabel(r"$t$",fontsize = 15,y = 0.110,x = 0.5125)
axes[1].legend(loc='upper right', 
                prop={'size': 13},title= r"$B(s)$",title_fontsize = 15)

### Use if bold symbol for sigma below ###
# axes[0].text(-2.1,0.18,r"$\textbf{{({0})}}$".format(string.ascii_uppercase[0]),
#                     size=35, weight='bold')
# axes[0].text(4.7,0.18,r"$\textbf{{({0})}}$".format(string.ascii_uppercase[1]),
#                     size=35, weight='bold')

axes[0].text(-1.50,0.20,r"$\textbf{(a)}$",
                    size=17)
axes[1].text(-1.5,99,r"$\textbf{(b)}$",
                    size=17)

plt.savefig(os.path.join(output_dir,'ee_length_comp.png'),bbox_inches = 'tight',dpi = 600)
plt.savefig(os.path.join(output_dir,'ee_length_comp.pdf'),format = 'pdf',bbox_inches = 'tight',dpi = 600)
plt.savefig(os.path.join(output_dir,'ee_length_comp.eps'),
            bbox_inches = 'tight',format = 'eps',dpi = 600)
plt.show()
#%% Plot Stresses
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": "Times",
    'text.latex.preamble': r'\usepackage{amsmath}'})

fig,axes = plt.subplots(nrows = 3,figsize = (7,10),layout = 'constrained',sharex = True)
# fig.subplots_adjust(hspace = 0.11)


### First Normal Stress ###
axes[0].axvline(x = vert_time,color = vert_col_code,linestyle = 'dashdot',linewidth = 2)
axes[0].plot(K_constant_time_val[:cot_idx],
               (K_constant_stress_data[0,0,:] - K_constant_stress_data[1,1,:])[:cot_idx],
               color = K_constant_col_code,linewidth = 2,label = r"$B_{1}(s)$")
axes[0].plot(K_dirac_left_l_stiff_time_val[:cot_idx],
               (K_dirac_left_l_stiff_stress_data[0,0,:] - K_dirac_left_l_stiff_stress_data[1,1,:])[:cot_idx],
               color = K_dirac_left_l_stiff_col_code,linewidth = 2,label = r"$B_{2}(s)$")
axes[0].plot(K_error_function_time_val[:cot_idx],
               (K_error_function_stress_data[0,0,:] - K_error_function_stress_data[1,1,:])[:cot_idx],
               color = K_error_function_col_code,linewidth = 2,label = r"$B_{3}(s)$")
axes[0].plot(K_constant_rigid_time_val[:cot_idx],
               (K_constant_rigid_stress_data[0,0,:] - K_constant_rigid_stress_data[1,1,:])[:cot_idx],
               color = rigid_col_code,linewidth = 2,linestyle = 'dotted')

### Second Normal Stress ###
axes[1].axvline(x = vert_time,color = vert_col_code,linestyle = 'dashdot',linewidth = 2)
axes[1].plot(K_constant_time_val[:cot_idx],(K_constant_stress_data[1,1,:] - K_constant_stress_data[2,2,:])[:cot_idx],
               color = K_constant_col_code,linewidth = 2,label = r"$B_{1}(s)$")
axes[1].plot(K_dirac_left_l_stiff_time_val[:cot_idx],(K_dirac_left_l_stiff_stress_data[1,1,:] - K_dirac_left_l_stiff_stress_data[2,2,:])[:cot_idx],
               color = K_dirac_left_l_stiff_col_code,linewidth = 2,label = r"$B_{2}(s)$")
axes[1].plot(K_error_function_time_val[:cot_idx],(K_error_function_stress_data[1,1,:] - K_error_function_stress_data[2,2,:])[:cot_idx],
               color = K_error_function_col_code,linewidth = 2,label = r"$B_{3}(s)$")
axes[1].plot(K_constant_rigid_time_val[:cot_idx],
               (K_constant_rigid_stress_data[1,1,:] - K_constant_rigid_stress_data[2,2,:])[:cot_idx],
               color = rigid_col_code,linewidth = 2,linestyle = 'dotted')

### Shear Stress ###
axes[2].axvline(x = vert_time,color = vert_col_code,linestyle = 'dashdot',linewidth = 2)
axes[2].plot(K_constant_time_val[:cot_idx],K_constant_stress_data[0,1,:][:cot_idx],
               color = K_constant_col_code,linewidth = 2,label = r"$B_{1}(s)$")
axes[2].plot(K_dirac_left_l_stiff_time_val[:cot_idx],K_dirac_left_l_stiff_stress_data[0,1,:][:cot_idx],
               color = K_dirac_left_l_stiff_col_code,linewidth = 2,label = r"$B_{2}(s)$")
axes[2].plot(K_error_function_time_val[:cot_idx],K_error_function_stress_data[0,1,:][:cot_idx],
               color = K_error_function_col_code,linewidth = 2,label = r"$B_{3}(s)$")
axes[2].plot(K_constant_rigid_time_val[:cot_idx],
               K_constant_rigid_stress_data[0,1,:][:cot_idx],
               color = rigid_col_code,linewidth = 2,linestyle = 'dotted')

### common x-axis label & legend ###
fig.supxlabel(r"$t$",fontsize = 15,y = -0.007,x = 0.525)
axes[0].legend(loc='lower right', 
                prop={'size': 13},title= r"$B(s)$",title_fontsize = 15)

## Format all plots with uniform commands
n = 2 # Hide every other label
n_counter = -1
# y_labels = [r"$N_{1}$",r"$N_{2}$",r"$\bm{\sigma}_{xy}$"]
y_labels = [r"$N_{1}$",r"$N_{2}$",r"$\sigma_{xy}$"]
subfig_labels = [r"$\textbf{(a)}$",r"$\textbf{(b)}$",r"$\textbf{(c)}$"]
for i,ax in enumerate(axes):
    n_counter += 1
    ### Use if bold symbol for sigma below ###
    # ax.text(-0.35,950,r"$\textbf{{({0})}}$".format(string.ascii_uppercase[n_counter]),
    #             size=35, weight='bold',ha='left', va='top')
    
    ax.text(-1.5,950,"{}".format(subfig_labels[n_counter]),
                size=17)
    # [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_ticklabels()) if i % n != 0]
    [l.set_visible(False) for (i,l) in enumerate(ax.yaxis.get_ticklabels()) if i % n != 0]
    ax.set_ylabel(y_labels[i],fontsize=15,labelpad = 5)
    ax.set_ylim(-1000,1000)
    ax.set_xlim(-0.5,6)
    ax.set_yticks(np.linspace(-1000,1000,5))
    ax.set_xticks(np.linspace(0,6,4))
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    ax.tick_params(axis='both', which='both', labelsize = 13, 
                   pad = 5,length = 5,size = 5,direction = 'in')
    ax.set_aspect(np.diff(ax.get_xlim())/(2.5*np.diff(ax.get_ylim())))

plt.savefig(os.path.join(output_dir,'simulation_behavior_stress.png'),
            bbox_inches = 'tight',dpi = 600)
plt.savefig(os.path.join(output_dir,'simulation_behavior_stress.pdf'),format = 'pdf',
            bbox_inches = 'tight',dpi = 600)
plt.savefig(os.path.join(output_dir,'simulation_behavior_stress.eps'),
            bbox_inches = 'tight',format = 'eps',dpi = 600)
plt.show()

#%% Calculate Stress differences for each profile

low_time = 0
high_time = 5.464
# low_time = np.round(vert_time - 1.5,3)
# high_time = np.round(vert_time + 1.5,3)
low_time_idx = np.where(K_constant_time_val == low_time)[0][0]
high_time_idx = np.where(K_constant_time_val == high_time)[0][0]+1

### N1 intergrated over time ###
K_constant_N1 = np.trapz(y = (K_constant_stress_data[0,0,:] - K_constant_stress_data[1,1,:])[low_time_idx:high_time_idx],
                         x = K_constant_time_val[low_time_idx:high_time_idx])
K_dirac_left_l_stiff_N1 = np.trapz(y = (K_dirac_left_l_stiff_stress_data[0,0,:] - K_dirac_left_l_stiff_stress_data[1,1,:])[low_time_idx:high_time_idx],
                                   x = K_dirac_left_l_stiff_time_val[low_time_idx:high_time_idx])
K_error_function_N1 = np.trapz(y = (K_error_function_stress_data[0,0,:] - K_error_function_stress_data[1,1,:])[low_time_idx:high_time_idx],
                               x = K_error_function_time_val[low_time_idx:high_time_idx])
K_constant_rigid_N1 = np.trapz(y = (K_constant_rigid_stress_data[0,0,:] - K_constant_rigid_stress_data[1,1,:])[low_time_idx:high_time_idx],
                         x = K_constant_rigid_time_val[low_time_idx:high_time_idx])

### N2 integrated over time ###

K_constant_N2 = np.trapz(y = (K_constant_stress_data[1,1,:] - K_constant_stress_data[2,2,:])[low_time_idx:high_time_idx],
                         x = K_constant_time_val[low_time_idx:high_time_idx])
K_dirac_left_l_stiff_N2 = np.trapz(y = (K_dirac_left_l_stiff_stress_data[1,1,:] - K_dirac_left_l_stiff_stress_data[2,2,:])[low_time_idx:high_time_idx],
                                   x = K_dirac_left_l_stiff_time_val[low_time_idx:high_time_idx])
K_error_function_N2 = np.trapz(y = (K_error_function_stress_data[1,1,:] - K_error_function_stress_data[2,2,:])[low_time_idx:high_time_idx],
                               x = K_error_function_time_val[low_time_idx:high_time_idx])
K_constant_rigid_N2 = np.trapz(y = (K_constant_rigid_stress_data[1,1,:] - K_constant_rigid_stress_data[2,2,:])[low_time_idx:high_time_idx],
                         x = K_constant_rigid_time_val[low_time_idx:high_time_idx])

### Sigma_xy integrated over time ###

K_constant_sxy = np.trapz(y = (K_constant_stress_data[0,1,:])[low_time_idx:high_time_idx],
                         x = K_constant_time_val[low_time_idx:high_time_idx])
K_dirac_left_l_stiff_sxy = np.trapz(y = (K_dirac_left_l_stiff_stress_data[0,1,:])[low_time_idx:high_time_idx],
                                   x = K_dirac_left_l_stiff_time_val[low_time_idx:high_time_idx])
K_error_function_sxy = np.trapz(y = (K_error_function_stress_data[0,1,:])[low_time_idx:high_time_idx],
                               x = K_error_function_time_val[low_time_idx:high_time_idx])
K_constant_rigid_sxy = np.trapz(y = (K_constant_rigid_stress_data[0,1,:])[low_time_idx:high_time_idx],
                         x = K_constant_rigid_time_val[low_time_idx:high_time_idx])
stress_results = pd.DataFrame(index = ['K_constant','K_dirac_left_l_stiff','K_error_function','K_constant_rigid'],
                              columns  = ['N1','N2','Sigma_xy','Start Time','End Time','Mu_bar'])

stress_results.loc['K_constant','N1'] = K_constant_N1
stress_results.loc['K_constant','N2'] = K_constant_N2
stress_results.loc['K_constant','Sigma_xy'] = K_constant_sxy

stress_results.loc['K_dirac_left_l_stiff','N1'] = K_dirac_left_l_stiff_N1
stress_results.loc['K_dirac_left_l_stiff','N2'] = K_dirac_left_l_stiff_N2
stress_results.loc['K_dirac_left_l_stiff','Sigma_xy'] = K_dirac_left_l_stiff_sxy

stress_results.loc['K_error_function','N1'] = K_error_function_N1
stress_results.loc['K_error_function','N2'] = K_error_function_N2
stress_results.loc['K_error_function','Sigma_xy'] = K_error_function_sxy

stress_results.loc['K_constant_rigid','N1'] = K_constant_rigid_N1
stress_results.loc['K_constant_rigid','N2'] = K_constant_rigid_N2
stress_results.loc['K_constant_rigid','Sigma_xy'] = K_constant_rigid_sxy

stress_results.loc[:,'Start Time'] = low_time
stress_results.loc[:,'End Time'] = high_time
stress_results.loc[:,'Mu_bar'] = 5e5
stress_results = stress_results.transpose()
stress_results.to_csv(os.path.join(output_dir,'Simulation_Stress_integration_results.csv'))
