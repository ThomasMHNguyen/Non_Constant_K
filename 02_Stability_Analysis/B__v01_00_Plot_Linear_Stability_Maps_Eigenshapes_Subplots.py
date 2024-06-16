# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 10:38:54 2022

@author: super
"""

import os, string
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy import special
from matplotlib.lines import Line2D
import matplotlib.patches  as mpatches

eigendata_main_dir = 'C:/Users/super/OneDrive - University of California, Davis/Research/00_Projects/01_Non_Constant_K/00_Scripts/02_Stability_Analysis/03_Run_Results/Mode_Type_Data/'
K_constant_eigendata = pd.read_csv(os.path.join(
    eigendata_main_dir,'K_constant/K_constant_mode_types.csv'),index_col = 0,header = 0)
K_dirac_left_l_stiff_eigendata = pd.read_csv(
    os.path.join(eigendata_main_dir,'K_dirac_left_l_stiff/K_dirac_left_l_stiff_mode_types.csv'),index_col = 0,header = 0)
K_error_function_eigendata = pd.read_csv(
    os.path.join(eigendata_main_dir,'K_error_function/K_error_function_mode_types.csv'),index_col = 0,header = 0)

eigenshape_main_dir = 'C:/Users/super/OneDrive - University of California, Davis/Research/00_Projects/01_Non_Constant_K/00_Scripts/02_Stability_Analysis/03_Run_Results/Mode_Number_Threshold/'
K_constant_eigenshape = pd.read_csv(os.path.join(
    eigenshape_main_dir,'K_constant/mode_number_threshold_K_constant.csv'),index_col = 0,header = 0)
K_dirac_left_l_stiff_eigenshape = pd.read_csv(
    os.path.join(eigenshape_main_dir,'K_dirac_left_l_stiff/mode_number_threshold_K_dirac_left_l_stiff.csv'),index_col = 0,header = 0)
K_error_function_eigenshape = pd.read_csv(
    os.path.join(eigenshape_main_dir,'K_error_function/mode_number_threshold_K_error_function.csv'),index_col = 0,header = 0)


### Draw Mode Shapes at onset of instability ###
K_constant_eigenshape = K_constant_eigenshape[K_constant_eigenshape['Mode Number'] <= 3]
K_dirac_left_l_stiff_eigenshape = K_dirac_left_l_stiff_eigenshape[K_dirac_left_l_stiff_eigenshape['Mode Number'] <= 3]
K_error_function_eigenshape = K_error_function_eigenshape[K_error_function_eigenshape['Mode Number'] <= 3]


#Adjust direction of first mode of profiles at unstable instance
K_constant_first_mode_idx = K_constant_eigenshape.index[K_constant_eigenshape['Mode Number'] == 1].tolist()
K_dirac_left_l_stiff_first_mode_idx = K_dirac_left_l_stiff_eigenshape.index[K_dirac_left_l_stiff_eigenshape['Mode Number'] == 1].tolist()
K_error_function_first_mode_idx = K_error_function_eigenshape.index[K_error_function_eigenshape['Mode Number'] == 1].tolist()

K_constant_eigenshape.loc[K_constant_first_mode_idx,'Eigenfunctions_real'] = K_constant_eigenshape.loc[K_constant_first_mode_idx,'Eigenfunctions_real'] * -1
K_dirac_left_l_stiff_eigenshape.loc[K_dirac_left_l_stiff_first_mode_idx,'Eigenfunctions_real'] = K_dirac_left_l_stiff_eigenshape.loc[K_dirac_left_l_stiff_first_mode_idx,'Eigenfunctions_real'] * -1
K_error_function_eigenshape.loc[K_error_function_first_mode_idx,'Eigenfunctions_real'] = K_error_function_eigenshape.loc[K_error_function_first_mode_idx,'Eigenfunctions_real'] * -1

#Adjust direction of third mode of profiles at unstable instance
K_constant_third_mode_idx = K_constant_eigenshape.index[K_constant_eigenshape['Mode Number'] == 3].tolist()
K_error_function_third_mode_idx = K_error_function_eigenshape.index[K_error_function_eigenshape['Mode Number'] == 3].tolist()
K_dirac_left_l_stiff_third_mode_idx = K_dirac_left_l_stiff_eigenshape.index[K_dirac_left_l_stiff_eigenshape['Mode Number'] == 3].tolist()

K_constant_eigenshape.loc[K_constant_third_mode_idx,'Eigenfunctions_real'] = K_constant_eigenshape.loc[K_constant_third_mode_idx,'Eigenfunctions_real'] * -1
# K_dirac_left_l_stiff_eigenshape.loc[K_dirac_left_l_stiff_third_mode_idx,'Eigenfunctions_real'] = K_dirac_left_l_stiff_eigenshape.loc[K_dirac_left_l_stiff_third_mode_idx,'Eigenfunctions_real'] * -1
# K_error_function_eigenshape.loc[K_dirac_left_l_stiff_third_mode_idx,'Eigenfunctions_real'] = K_error_function_eigenshape.loc[K_dirac_left_l_stiff_third_mode_idx,'Eigenfunctions_real'] * -1


### Plot Mode Shapes at specific mu_bar values ###

stability_data_dir = 'C:/Users/super/OneDrive - University of California, Davis/Research/00_Projects/01_Non_Constant_K/00_Scripts/02_Stability_Analysis/03_Run_Results/Stability_Data/'

K_constant_stability_df = pd.read_csv(os.path.join(
    stability_data_dir,'K_constant/N_1001_stability_plot_data.csv'),index_col = 0,header = 0,dtype = {"Mode Number": str})
K_dirac_left_l_stiff_stability_df = pd.read_csv(os.path.join(
    stability_data_dir,'K_dirac_left_l_stiff/N_1001_stability_plot_data.csv'),index_col = 0,header = 0,dtype = {"Mode Number": str})
K_error_function_stability_df = pd.read_csv(os.path.join(
    stability_data_dir,'K_error_function/N_1001_stability_plot_data.csv'),index_col = 0,header = 0,dtype = {"Mode Number": str})


#Adjust direction of first mode at specified mu_bar value
mu_bar_val_m1 = 18000
K_constant_m1_idx = K_constant_stability_df[
        (K_constant_stability_df['Mode Number'] == "1") & 
        (K_constant_stability_df['Mu_bar'] == mu_bar_val_m1)].index
K_dirac_left_l_stiff_m1_idx = K_dirac_left_l_stiff_stability_df[
        (K_dirac_left_l_stiff_stability_df['Mode Number'] == "1") & 
        (K_dirac_left_l_stiff_stability_df['Mu_bar'] == mu_bar_val_m1)].index
K_error_function_m1_idx = K_error_function_stability_df[
        (K_error_function_stability_df['Mode Number'] == "1") & 
        (K_error_function_stability_df['Mu_bar'] == mu_bar_val_m1)].index

# K_constant_stability_df.loc[K_constant_m1_idx,
#                                       'Eigenfunctions_real'] =\
#     K_constant_stability_df.loc[K_constant_m1_idx,'Eigenfunctions_real']*-1 
# K_dirac_left_l_stiff_stability_df.loc[K_dirac_left_l_stiff_m1_idx,
#                                       'Eigenfunctions_real'] =\
#     K_dirac_left_l_stiff_stability_df.loc[K_dirac_left_l_stiff_m1_idx,'Eigenfunctions_real']*-1
K_error_function_stability_df.loc[K_error_function_m1_idx,
                                      'Eigenfunctions_real'] =\
    K_error_function_stability_df.loc[K_error_function_m1_idx,'Eigenfunctions_real']*-1


    

#Adjust direction of third mode at specified mu_bar value
mu_bar_val_m3 = 30000
K_constant_m3_idx = K_constant_stability_df[
        (K_constant_stability_df['Mode Number'] == "3") & 
        (K_constant_stability_df['Mu_bar'] == mu_bar_val_m3)].index
K_dirac_left_l_stiff_m3_idx = K_dirac_left_l_stiff_stability_df[
        (K_dirac_left_l_stiff_stability_df['Mode Number'] == "3") & 
        (K_dirac_left_l_stiff_stability_df['Mu_bar'] == mu_bar_val_m3)].index
K_error_function_m3_idx = K_error_function_stability_df[
        (K_error_function_stability_df['Mode Number'] == "3") & 
        (K_error_function_stability_df['Mu_bar'] == mu_bar_val_m3)].index

K_constant_stability_df.loc[K_constant_m3_idx,
                                      'Eigenfunctions_real'] =\
    K_constant_stability_df.loc[K_constant_m3_idx,'Eigenfunctions_real']*-1 
K_dirac_left_l_stiff_stability_df.loc[K_dirac_left_l_stiff_m3_idx,
                                      'Eigenfunctions_real'] =\
    K_dirac_left_l_stiff_stability_df.loc[K_dirac_left_l_stiff_m3_idx,'Eigenfunctions_real']*-1
    
K_error_function_stability_df.loc[K_error_function_m3_idx,
                                      'Eigenfunctions_real'] =\
    K_error_function_stability_df.loc[K_error_function_m3_idx,'Eigenfunctions_real']*-1

output_dir = 'C:/Users/super/OneDrive - University of California, Davis/Research/00_Projects/01_Non_Constant_K/00_Scripts/02_Stability_Analysis/04_Dissertation/'
if not os.path.exists(output_dir):
        os.makedirs(output_dir)


### Colors ###
K_constant_col_code = '#32CD32'
K_dirac_left_l_stiff_col_code = '#1E90FF'
K_error_function_col_code = '#FF7F50'
#%% For Inset Images

class Constants:
    """
    This class will bundle up all constants and parameters needed in the simulations to be 
    easily accessed in the functions due to multi-processing implementation.
    """
    
    ########    
    def __init__(self,rigid_func,N):
        
        ##### Traditional Constants #####
        
        self.rigidity_suffix = rigid_func
        self.N = N
        self.L = 1
        self.s = np.linspace(-(self.L/2),(self.L/2),N)
        self.centerline_idx = np.where(self.s == 0)[0][0]
        self.ds = 1/(self.N-1)
        
        ##### Determine Form of filament rigidity #####
        
        if self.rigidity_suffix == 'K_constant':
            self.K = np.ones(self.s.shape[0],dtype = float)
            self.Ks = np.zeros(self.s.shape[0],dtype = float)
            self.Kss = np.zeros(self.s.shape[0],dtype = float)
        elif self.rigidity_suffix == 'K_parabola_center_l_stiff':
            self.K = 1/2 + 2*self.s**2
            self.Ks = 4*self.s
            self.Kss = 4*np.ones(self.s.shape[0],dtype = float)
        elif self.rigidity_suffix == 'K_parabola_center_m_stiff':
            self.K = 1.5 - 2*(self.s**2)
            self.Ks = -4*self.s
            self.Kss = -4*np.ones(self.s.shape[0],dtype = float)
        elif self.rigidity_suffix == 'K_linear':
            self.K = self.s+1
            self.Ks = 1*np.ones(self.s.shape[0],dtype = float)
            self.Kss = 0*np.ones(self.s.shape[0],dtype = float)
        elif self.rigidity_suffix == 'K_dirac_center_l_stiff':
            self.K = 1-0.5*np.exp(-100*self.s**2)
            self.Ks = 100*self.s*np.exp(-100*self.s**2)
            self.Kss = np.exp(-100*self.s**2)*(100-2e4*self.s**2)
        elif self.rigidity_suffix == 'K_dirac_center_l_stiff2':
            self.K = 1-0.5*np.exp(-500*self.s**2)
            self.Ks = 500*self.s*np.exp(-500*self.s**2)
            self.Kss = np.exp(-500*self.s**2)*(500-5e5*self.s**2)
        elif self.rigidity_suffix == 'K_dirac_center_m_stiff':
            self.K = 1+np.exp(-100*self.s**2)
            self.Ks = -200*self.s*np.exp(-100*self.s**2)
            self.Kss = 200*np.exp(-100*self.s**2)*(200*self.s**2-1)
        elif self.rigidity_suffix == 'K_parabola_shifted':
            self.K = 1.5-0.5*(self.s-0.5)**2
            self.Ks = -1*self.s-0.5
            self.Kss = -1*np.ones(self.s.shape[0],dtype = float)
        elif self.rigidity_suffix == 'K_error_function':
            self.K = special.erf(10*self.s)+2
            self.Ks = (20/np.sqrt(np.pi))*np.exp(-100*self.s**2)
            self.Kss = (-4000*self.s/np.sqrt(np.pi))*np.exp(-100*self.s**2)  
        elif self.rigidity_suffix == 'K_dirac_left_l_stiff':
            self.K = 1-0.5*np.exp(-100*(self.s+0.25)**2)
            self.Ks = 100*(self.s+0.25)*np.exp(-100*(self.s+0.25)**2)
            self.Kss = np.exp(-100*(self.s+0.25)**2)*-2e4*(self.s**2+0.5*self.s+0.0575)
        elif self.rigidity_suffix == 'K_dirac_left_l_stiff2':
            self.K = 1-0.5*np.exp(-500*(self.s+0.25)**2)
            self.Ks = 500*(self.s+0.25)*np.exp(-500*(self.s+0.25)**2)
            self.Kss = np.exp(-500*(self.s+0.25)**2)*-5e5*(self.s**2+0.5*self.s+0.0615)
        elif self.rigidity_suffix == 'K_dirac_left_l_stiff3':
            self.K = 1-0.5*np.exp(-1000*(self.s+0.25)**2)
            self.Ks = 1000*(self.s+0.25)*np.exp(-1000*(self.s+0.25)**2)
            self.Kss = np.exp(-1000*(self.s+0.25)**2)*-2e6*(self.s**2+0.5*self.s+0.062)
#%% Eigenspectrum 

K_constant_prof = Constants('K_constant', 101)
K_gaussian_left_l_stiff_prof = Constants('K_dirac_left_l_stiff', 101)
K_error_prof = Constants('K_error_function', 101)

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": "Times",
    'text.latex.preamble': r'\usepackage{amsmath} \usepackage{amssymb}'})

color_palette = ["#574D68","#FA9F42","#0B6E4F","#DB7F8E"]
marker_shapes = ["o","v","d","s"]
mu_bar_vals_1000 = np.array([i for i in K_constant_eigendata['Mu_bar'].unique() if i % 1000 == 0])
fig,axes = plt.subplots(ncols = 3,figsize = (10,7),layout = 'constrained',sharey = True)
# fig.subplots_adjust(wspace = 0.05)
# sns.scatterplot(x = 'Mu_bar',y = 'Eigenvalues_real',data = K_constant_eigendata,
#                 ax = axes[0],hue = 'Mode Number',palette = 'bright')
# sns.scatterplot(x = 'Mu_bar',y = 'Eigenvalues_real',data = K_dirac_left_l_stiff_eigendata,
#                 ax = axes[1],hue = 'Mode Number',palette = 'bright')
# sns.scatterplot(x = 'Mu_bar',y = 'Eigenvalues_real',data = K_error_function_eigendata,
#                 ax = axes[2],hue = 'Mode Number',palette = 'bright')

sns.scatterplot(x = 'Mu_bar',y = 'Eigenvalues_real',data = K_constant_eigendata[K_constant_eigendata['Mu_bar'].isin(mu_bar_vals_1000)],
                ax = axes[0],hue = 'Mode Number',style = "Mode Number",
                markers = marker_shapes,palette = color_palette,s = 50,legend = False)
sns.scatterplot(x = 'Mu_bar',y = 'Eigenvalues_real',data = K_dirac_left_l_stiff_eigendata[K_dirac_left_l_stiff_eigendata['Mu_bar'].isin(mu_bar_vals_1000)],
                ax = axes[1],hue = 'Mode Number',style = "Mode Number",
                markers = marker_shapes,palette = color_palette,s = 50)
sns.scatterplot(x = 'Mu_bar',y = 'Eigenvalues_real',data = K_error_function_eigendata[K_error_function_eigendata['Mu_bar'].isin(mu_bar_vals_1000)],
                ax = axes[2],hue = 'Mode Number',style = "Mode Number",
                markers = marker_shapes,palette = color_palette,s = 50,legend = False)

plt.axis()

subfig_labels = [r"$\textbf{(a)}$",r"$\textbf{(b)}$",r"$\textbf{(c)}$"]
for n,ax in enumerate(axes):
    ax.ticklabel_format(axis="x", style="sci", scilimits=(4,4))
    ax.set_xlim(0,50100)
    ax.set_ylim(-4,8.6)
    ax.set_xticks(np.linspace(0,50000,6))
    ax.set_yticks(np.linspace(-4,8,7))
    [l.set_visible(False) for (i,l) in enumerate(ax.yaxis.get_ticklabels()) if i % 2 != 0] #Hide every other label
    ax.tick_params(axis='both', which='both',labelsize = 13, length = 5,size = 5,direction = 'in',pad = 3)
    # ax.tick_params(axis='y', which='both',labelsize = 35, size = 10,width = 3,direction = 'in',pad = 10)
    ax.text(0.03,9,'{}'.format(subfig_labels[n]),
            size=17)
    ax.xaxis.offsetText.set_fontsize(0)
    ax.set_aspect(np.diff(ax.get_xlim())/np.diff(ax.get_ylim()))
    

axes[0].set(xlabel=None)
axes[2].set(xlabel=None)
axes[0].legend([],[],frameon = False)
axes[1].legend([],[],frameon = False)
fig.legend(loc='lower center', 
                bbox_to_anchor=(0.5,0.1),prop={'size': 13},
                title= "Mode Number",ncol=4).get_title().set_fontsize("15")
axes[1].set_xlabel(r"$\bar{\mu}\times 10^{4}$",fontsize = 15,labelpad = 5)
axes[0].set_ylabel(r"$\mathbb{R}(\sigma)$",fontsize = 15,labelpad = 5)

### Inset images
axins_1 = inset_axes(axes[0], width="20%", height="20%", loc='upper right')
axins_2 = inset_axes(axes[1], width="20%", height="20%", loc='upper right')
axins_3 = inset_axes(axes[2], width="20%", height="20%", loc='upper right')
axins_1.plot(K_constant_prof.s,K_constant_prof.K,color = 'black',lw = 2)
axins_2.plot(K_gaussian_left_l_stiff_prof.s,K_gaussian_left_l_stiff_prof.K,color = 'black',lw = 2)
axins_3.plot(K_error_prof.s,K_error_prof.K,color = 'black',lw =2)

for n,ax in enumerate([axins_1,axins_2,axins_3]):
    ax.set_xlim(-0.6,0.6)
    ax.set_ylim(0.3,3.2)
    # ax.set_xticks(np.linspace(-0.5,0.5,5))
    # ax.set_yticks(np.linspace(0.5,3,6))
    # ax.tick_params(axis='x', which='major',labelsize = 9, size = 4,width = 2,rotation = 45)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    # ax.tick_params(axis='y', which='major',labelsize = 9, size = 4,width = 2)
    # ax.set_xlabel(r"Arclength $[s]$",fontsize = 9,labelpad = 5)
    # ax.set_ylabel(r"$\kappa (s)$",fontsize = 9,labelpad = 5)
    ax.set_aspect(np.diff(ax.get_xlim())/np.diff(ax.get_ylim()))
    
# axins_1.text(x = -0.58,y = 2.54,s = r"$B_{1}(s)$",
#                     size=15)
# axins_2.text(x = -0.58,y = 2.54,s = r"$B_{2}(s)$",
#                     size=15)
# axins_3.text(x = -0.58,y = 2.54,s = r"$B_{3}(s)$",
#                     size=15)

fig.savefig(os.path.join(output_dir,'eig_map_colored.png'),
            bbox_inches = 'tight',dpi = 600)
fig.savefig(os.path.join(output_dir,'eig_map_colored.pdf'),
            format = 'pdf',bbox_inches = 'tight',dpi = 600)
fig.savefig(os.path.join(output_dir,'eig_map_colored.eps'),
            bbox_inches = 'tight',format = 'eps',dpi = 600)

plt.show()
#%% Plot Mode shapes at onset of instability and various mu_bar_values

### Target mu_bar and mode vals: 3rd mode- 30,000; 1st mode- 18,000

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": "Times",
    'text.latex.preamble': r'\usepackage{amsmath} \usepackage{amssymb}'})



fig,axes = plt.subplots(ncols = 2,nrows = 2,figsize = (7,7),layout = 'constrained',
                        sharey = True,sharex = True)
# fig.subplots_adjust(wspace = 0.05,hspace = 0.050)

### Plot second mode shapes at onset of instability ###
axes[0,0].plot(K_constant_eigenshape[K_constant_eigenshape['Mode Number'] == 1]['s'].to_numpy(),
               K_constant_eigenshape[K_constant_eigenshape['Mode Number'] == 1]['Eigenfunctions_real'].to_numpy(),
               linewidth = 2,color = K_constant_col_code,label = r"$B_{1}(s)$")
axes[0,0].plot(K_dirac_left_l_stiff_eigenshape[K_dirac_left_l_stiff_eigenshape['Mode Number'] == 1]['s'].to_numpy(),
               K_dirac_left_l_stiff_eigenshape[K_dirac_left_l_stiff_eigenshape['Mode Number'] == 1]['Eigenfunctions_real'].to_numpy(),
               linewidth = 2,color = K_dirac_left_l_stiff_col_code,label = r"$B_{2}(s)$")
axes[0,0].plot(K_error_function_eigenshape[K_error_function_eigenshape['Mode Number'] == 1]['s'].to_numpy(),
               K_error_function_eigenshape[K_error_function_eigenshape['Mode Number'] == 1]['Eigenfunctions_real'].to_numpy(),
               linewidth = 2,color = K_error_function_col_code,label = r"$B_{3}(s)$")



### Plot second mode shapes at mu_bar_value of interest ###
axes[0,1].plot(K_constant_stability_df[
    (K_constant_stability_df['Mode Number'] == "1") & 
    (K_constant_stability_df['Mu_bar'] == mu_bar_val_m1)]['s'].to_numpy(),
    K_constant_stability_df[
    (K_constant_stability_df['Mode Number'] == "1") & 
    (K_constant_stability_df['Mu_bar'] == mu_bar_val_m1)]['Eigenfunctions_real'].to_numpy(),
               linewidth = 2,color = K_constant_col_code,label = r"$B_{1}(s)$")
axes[0,1].plot(K_dirac_left_l_stiff_stability_df[
    (K_dirac_left_l_stiff_stability_df['Mode Number'] == "1") & 
    (K_dirac_left_l_stiff_stability_df['Mu_bar'] == mu_bar_val_m1)]['s'].to_numpy(),
    K_dirac_left_l_stiff_stability_df[
    (K_dirac_left_l_stiff_stability_df['Mode Number'] == "1") & 
    (K_dirac_left_l_stiff_stability_df['Mu_bar'] == mu_bar_val_m1)]['Eigenfunctions_real'].to_numpy(),
               linewidth = 2,color = K_dirac_left_l_stiff_col_code,label = r"$B_{2}(s)$")
axes[0,1].plot(K_error_function_stability_df[
    (K_error_function_stability_df['Mode Number'] == "1") & 
    (K_error_function_stability_df['Mu_bar'] == mu_bar_val_m1)]['s'].to_numpy(),
    K_error_function_stability_df[
    (K_error_function_stability_df['Mode Number'] == "1") & 
    (K_error_function_stability_df['Mu_bar'] == mu_bar_val_m1)]['Eigenfunctions_real'].to_numpy(),
               linewidth = 2,color = K_error_function_col_code,label = r"$B_{3}(s)$")


### Plot third mode shapes at onset of instability ###
axes[1,0].plot(K_constant_eigenshape[K_constant_eigenshape['Mode Number'] == 3]['s'].to_numpy(),
               K_constant_eigenshape[K_constant_eigenshape['Mode Number'] == 3]['Eigenfunctions_real'].to_numpy(),
               linewidth = 2,color = K_constant_col_code,label = r"$B_{1}(s)$")
axes[1,0].plot(K_dirac_left_l_stiff_eigenshape[K_dirac_left_l_stiff_eigenshape['Mode Number'] == 3]['s'].to_numpy(),
               K_dirac_left_l_stiff_eigenshape[K_dirac_left_l_stiff_eigenshape['Mode Number'] == 3]['Eigenfunctions_real'].to_numpy(),
               linewidth = 2,color = K_dirac_left_l_stiff_col_code,label = r"$B_{2}(s)$")
axes[1,0].plot(K_error_function_eigenshape[K_error_function_eigenshape['Mode Number'] == 3]['s'].to_numpy(),
               K_error_function_eigenshape[K_error_function_eigenshape['Mode Number'] == 3]['Eigenfunctions_real'].to_numpy(),
               linewidth = 2,color = K_error_function_col_code,label = r"$B_{3}(s)$")

### Plot third mode shapes at mu_bar_value of interest ###
axes[1,1].plot(K_constant_stability_df[
    (K_constant_stability_df['Mode Number'] == "3") & 
    (K_constant_stability_df['Mu_bar'] == mu_bar_val_m3)]['s'].to_numpy(),
    K_constant_stability_df[
    (K_constant_stability_df['Mode Number'] == "3") & 
    (K_constant_stability_df['Mu_bar'] == mu_bar_val_m3)]['Eigenfunctions_real'].to_numpy(),
               linewidth = 2,color = K_constant_col_code,label = r"$B_{1}(s)$")
axes[1,1].plot(K_dirac_left_l_stiff_stability_df[
    (K_dirac_left_l_stiff_stability_df['Mode Number'] == "3") & 
    (K_dirac_left_l_stiff_stability_df['Mu_bar'] == mu_bar_val_m3)]['s'].to_numpy(),
    K_dirac_left_l_stiff_stability_df[
    (K_dirac_left_l_stiff_stability_df['Mode Number'] == "3") & 
    (K_dirac_left_l_stiff_stability_df['Mu_bar'] == mu_bar_val_m3)]['Eigenfunctions_real'].to_numpy(),
               linewidth = 2,color = K_dirac_left_l_stiff_col_code,label = r"$B_{2}(s)$")
axes[1,1].plot(K_error_function_stability_df[
    (K_error_function_stability_df['Mode Number'] == "3") & 
    (K_error_function_stability_df['Mu_bar'] == mu_bar_val_m3)]['s'].to_numpy(),
    K_error_function_stability_df[
    (K_error_function_stability_df['Mode Number'] == "3") & 
    (K_error_function_stability_df['Mu_bar'] == mu_bar_val_m3)]['Eigenfunctions_real'].to_numpy(),
               linewidth = 2,color = K_error_function_col_code,label = r"$B_{3}(s)$")

subfig_labels = [r"$\textbf{(a)}$",r"$\textbf{(b)}$",r"$\textbf{(c)}$",r"$\textbf{(d)}$"]
i = -1
for n_row,ax_row in enumerate(axes):
    for n_col,ax_col in enumerate(ax_row):
        i +=1 
        ax_col.set_xlim(-0.6,0.6)
        ax_col.set_ylim(-1.1,1.1)
        ax_col.set_xticks(np.linspace(-0.5,0.5,5))
        ax_col.set_yticks(np.linspace(-1,1,5))
        ax_col.tick_params(axis='both', which='both',labelsize = 13, size = 5,
                           length = 5,direction = 'in',pad = 5)
        # ax_col.tick_params(axis='y', which='both',labelsize = 35, size = 10,width = 3,direction = 'in',pad = 5)
        [l.set_visible(False) for (i,l) in enumerate(ax_col.xaxis.get_ticklabels()) if i % 2 != 0] #Hide every other label
        [l.set_visible(False) for (i,l) in enumerate(ax_col.yaxis.get_ticklabels()) if i % 2 != 0] #Hide every other label
        ax_col.set_aspect(np.diff(ax_col.get_xlim())/np.diff(ax_col.get_ylim()))
        ax_col.set(xlabel = None,ylabel = None)
        ax_col.text(-0.54,0.92,'{}'.format(subfig_labels[i]),size=17)
        
axes[1,0].set_xlabel(r"$s$",size = 15,labelpad = 5)
axes[1,1].set_xlabel(r"$s$",size = 15,labelpad = 5)
axes[0,0].set_ylabel(r"$\hat{h}(s)$",size = 15,labelpad = 5)
axes[1,0].set_ylabel(r"$\hat{h}(s)$",size = 15,labelpad = 5)


axes[1,1].legend(loc='upper right', 
                prop={'size': 13},title= r"$B(s)$").get_title().set_fontsize(15)
fig.savefig(os.path.join(output_dir,'Mode_shape_discrepancies.png'),
            bbox_inches = 'tight',dpi = 600)
fig.savefig(os.path.join(output_dir,'Mode_shape_discrepancies.pdf'),
            format = 'pdf',bbox_inches = 'tight',dpi = 600)
fig.savefig(os.path.join(output_dir,'Mode_shape_discrepancies.eps'),
            bbox_inches = 'tight',format = 'eps',dpi = 600)
plt.show()



 #%% Subplots- Mode shapes
# fig,axes = plt.subplots(ncols = 3,figsize = (22,11),sharey = True)
# fig.subplots_adjust(wspace = 0.05)
# sns.lineplot(x = 's',y = 'Eigenfunctions_real',data = K_constant_eigenshape,
#                 ax = axes[0],hue = 'Mode Number',palette = 'bright',linewidth = 4)
# sns.lineplot(x = 's',y = 'Eigenfunctions_real',data = K_dirac_left_l_stiff_eigenshape,
#                 ax = axes[1],hue = 'Mode Number',palette = 'bright',linewidth = 4)
# sns.lineplot(x = 's',y = 'Eigenfunctions_real',data = K_error_function_eigenshape,
#                 ax = axes[2],hue = 'Mode Number',palette = 'bright',linewidth = 4)

# plt.axis()


# for n,ax in enumerate(axes):
#     ax.set_xlim(-0.6,0.6)
#     ax.set_ylim(-1.1,1.1)
#     ax.set_xticks(np.linspace(-0.5,0.5,5))
#     ax.set_yticks(np.linspace(-1,1,5))
#     ax.tick_params(axis='x', which='both',labelsize = 30, size = 10,width = 5,direction = 'in',pad = 10)
#     ax.tick_params(axis='y', which='both',labelsize = 35, size = 10,width = 5,direction = 'in',pad = 10)
#     [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_ticklabels()) if i % 2 != 0] #Hide every other label
#     [l.set_visible(False) for (i,l) in enumerate(ax.yaxis.get_ticklabels()) if i % 2 != 0] #Hide every other label
#     ax.text(0.03,0.92,'({})'.format(string.ascii_uppercase[n+3]),transform=ax.transAxes,
#             size=35, weight='bold')
#     ax.xaxis.offsetText.set_fontsize(0)
#     ax.set_aspect(1.2/2.2)
    

# axes[0].set(xlabel=None)
# axes[2].set(xlabel=None)
# axes[0].legend([],[],frameon = False)
# axes[1].legend([],[],frameon = False)
# axes[2].legend(loc='center left', 
#                 bbox_to_anchor=(1, 0.5),prop={'size': 20},title= "Mode Number").get_title().set_fontsize("25")
# axes[1].set_xlabel(r"arclength $[s]$",fontsize = 35,labelpad = 20)
# axes[0].set_ylabel(r"$\hat{h}(s)$",fontsize = 35,labelpad = 20)
# fig.savefig(os.path.join(output_dir,'02_three_eigenspectrum_shapes.png'),bbox_inches = 'tight',dpi = 200)
# fig.savefig(os.path.join(output_dir,'02_three_eigenspectrum_shapes.pdf'),bbox_inches = 'tight',dpi = 200)
# fig.savefig(os.path.join(output_dir,'02_three_eigenspectrum_shapes.eps'),
#             bbox_inches = 'tight',format = 'eps',dpi = 200)
# plt.show()

#%% Regular subplots

# K_constant_prof = Constants('K_constant', 101)
# K_gaussian_left_l_stiff_prof = Constants('K_dirac_left_l_stiff', 101)
# K_error_prof = Constants('K_error_function', 101)


# fig,axes = plt.subplots(ncols = 3,nrows = 2,figsize = (28,22),sharey = 'row')
# fig.subplots_adjust(wspace = 0.05)

# sns.scatterplot(x = 'Mubar.1',y = 'Eigenvalues_real',data = K_constant_eigendata,
#                 ax = axes[0,0],hue = 'Mode Number',palette = 'bright',s = 75)
# sns.scatterplot(x = 'Mubar.1',y = 'Eigenvalues_real',data = K_dirac_left_l_stiff_eigendata,
#                 ax = axes[0,1],hue = 'Mode Number',palette = 'bright',s = 75)
# sns.scatterplot(x = 'Mubar.1',y = 'Eigenvalues_real',data = K_error_function_eigendata,
#                 ax = axes[0,2],hue = 'Mode Number',palette = 'bright',s = 75)
# sns.lineplot(x = 's',y = 'Eigenfunctions_real',data = K_constant_eigenshape,
#                 ax = axes[1,0],hue = 'Mode Number',palette = 'bright',linewidth = 4)
# sns.lineplot(x = 's',y = 'Eigenfunctions_real',data = K_dirac_left_l_stiff_eigenshape,
#                 ax = axes[1,1],hue = 'Mode Number',palette = 'bright',linewidth = 4)
# sns.lineplot(x = 's',y = 'Eigenfunctions_real',data = K_error_function_eigenshape,
#                 ax = axes[1,2],hue = 'Mode Number',palette = 'bright',linewidth = 4)

# plt.axis()
# ax = plt.gca()
# counter = -1
# for n_row,ax_row in enumerate(axes):
#     for n_col,ax_col in enumerate(ax_row):
#         counter += 1
#         if n_row == 0:
#             ax_col.ticklabel_format(axis="x", style="sci", scilimits=(4,4))
#             ax_col.set_xlim(0,50050)
#             ax_col.set_ylim(-4,8.2)
#             ax_col.set_xticks(np.linspace(0,50000,6))
#             ax_col.set_yticks(np.linspace(-4,8,7))
#             ax_col.tick_params(axis='x', which='both',labelsize = 30, size = 10,width = 5,direction = 'in',pad = 10)
#             ax_col.tick_params(axis='y', which='both',labelsize = 35, size = 10,width = 5,direction = 'in',pad = 10)
#             [l.set_visible(False) for (i,l) in enumerate(ax_col.yaxis.get_ticklabels()) if i % 2 != 0] #Hide every other label
#             # ax_col.text(-1,0.1,'({})'.format(string.ascii_uppercase[counter]),transform=ax.transAxes,
#             #         size=35, weight='bold')
#             ax_col.xaxis.offsetText.set_fontsize(0)
#             ax_col.set_aspect(50050/12)
#         elif n_row == 1:
#             ax_col.set_xlim(-0.6,0.6)
#             ax_col.set_ylim(-1.1,1.1)
#             ax_col.set_xticks(np.linspace(-0.5,0.5,5))
#             [l.set_visible(False) for (i,l) in enumerate(ax_col.xaxis.get_ticklabels()) if i % 2 != 0] #Hide every other label
#             [l.set_visible(False) for (i,l) in enumerate(ax_col.yaxis.get_ticklabels()) if i % 2 != 0] #Hide every other label
#             ax_col.set_yticks(np.linspace(-1,1,5))
#             ax_col.tick_params(axis='x', which='both',labelsize = 30, size = 10,width = 5,direction = 'in',pad = 10)
#             ax_col.tick_params(axis='y', which='both',labelsize = 35, size = 10,width = 5,direction = 'in',pad = 10)
#             # ax_col.text(0.03,0.92,'({})'.format(string.ascii_uppercase[counter]),transform=ax.transAxes,
#             #         size=35, weight='bold')
#             ax_col.xaxis.offsetText.set_fontsize(0)
#             ax_col.set_aspect(1.2/2.2)
            
# ## Subfigure denotion texts
# axes[0,0].text(-2.07,2.22,'({})'.format(string.ascii_uppercase[0]),transform=ax.transAxes,
#                     size=35, weight='bold')
# axes[0,1].text(-1.03,2.22,'({})'.format(string.ascii_uppercase[1]),transform=ax.transAxes,
#                     size=35, weight='bold')
# axes[0,2].text(0.02,2.22,'({})'.format(string.ascii_uppercase[2]),transform=ax.transAxes,
#                     size=35, weight='bold')
# axes[1,0].text(-2.07,0.92,'({})'.format(string.ascii_uppercase[3]),transform=ax.transAxes,
#                     size=35, weight='bold')
# axes[1,1].text(-1.03,0.92,'({})'.format(string.ascii_uppercase[4]),transform=ax.transAxes,
#                     size=35, weight='bold')
# axes[1,2].text(0.02,0.92,'({})'.format(string.ascii_uppercase[5]),transform=ax.transAxes,
#                     size=35, weight='bold')

    
# #Top row of plots formatting
# axes[0,0].set(xlabel=None)
# axes[0,2].set(xlabel=None)
# axes[0,0].legend([],[],frameon = False)
# axes[0,1].legend([],[],frameon = False)
# axes[0,2].legend(loc='center left', 
#                 bbox_to_anchor=(1, 0.5),prop={'size': 20},title= "Mode Number").get_title().set_fontsize("25")
# axes[0,1].set_xlabel(r"$\bar{\mu}\times 10^{4}$",fontsize = 35,labelpad = 20)
# axes[0,0].set_ylabel(r"$\mathbb{R}(\sigma)$",fontsize = 35,labelpad = 40)


# #Bottom row of plots formatting
# axes[1,0].set(xlabel=None)
# axes[1,2].set(xlabel=None)
# axes[1,0].legend([],[],frameon = False)
# axes[1,1].legend([],[],frameon = False)
# axes[1,2].legend(loc='center left', 
#                 bbox_to_anchor=(1, 0.5),prop={'size': 20},title= "Mode Number").get_title().set_fontsize("25")
# axes[1,1].set_xlabel(r"$s$",fontsize = 35,labelpad = 20)
# axes[1,0].set_ylabel(r"$\hat{h}(s)$",fontsize = 35,labelpad = 2)
# fig.savefig(os.path.join(output_dir,'03_all_eigenspectrum_maps_shapes.png'),bbox_inches = 'tight',dpi = 200)
# fig.savefig(os.path.join(output_dir,'03_all_eigenspectrum_maps_shapes.pdf'),bbox_inches = 'tight',dpi = 200)
# fig.savefig(os.path.join(output_dir,'03_all_eigenspectrum_maps_shapes.eps'),
#             bbox_inches = 'tight',format = 'eps',dpi = 200)
# plt.show() 

#%% Subplots with inset images
# K_constant_prof = Constants('K_constant', 101)
# K_gaussian_left_l_stiff_prof = Constants('K_dirac_left_l_stiff', 101)
# K_error_prof = Constants('K_error_function', 101)


# fig,axes = plt.subplots(ncols = 3,nrows = 2,figsize = (28,22),sharey = 'row')
# fig.subplots_adjust(wspace = 0.05)

# sns.scatterplot(x = 'Mubar.1',y = 'Eigenvalues_real',data = K_constant_eigendata,
#                 ax = axes[0,0],hue = 'Mode Number',palette = 'bright',s = 75)
# sns.scatterplot(x = 'Mubar.1',y = 'Eigenvalues_real',data = K_dirac_left_l_stiff_eigendata,
#                 ax = axes[0,1],hue = 'Mode Number',palette = 'bright',s = 75)
# sns.scatterplot(x = 'Mubar.1',y = 'Eigenvalues_real',data = K_error_function_eigendata,
#                 ax = axes[0,2],hue = 'Mode Number',palette = 'bright',s = 75)
# sns.lineplot(x = 's',y = 'Eigenfunctions_real',data = K_constant_eigenshape,
#                 ax = axes[1,0],hue = 'Mode Number',palette = 'bright',linewidth = 4)
# sns.lineplot(x = 's',y = 'Eigenfunctions_real',data = K_dirac_left_l_stiff_eigenshape,
#                 ax = axes[1,1],hue = 'Mode Number',palette = 'bright',linewidth = 4)
# sns.lineplot(x = 's',y = 'Eigenfunctions_real',data = K_error_function_eigenshape,
#                 ax = axes[1,2],hue = 'Mode Number',palette = 'bright',linewidth = 4)

# plt.axis()
# ax = plt.gca()
# counter = -1
# for n_row,ax_row in enumerate(axes):
#     for n_col,ax_col in enumerate(ax_row):
#         counter += 1
#         if n_row == 0:
#             ax_col.ticklabel_format(axis="x", style="sci", scilimits=(4,4))
#             ax_col.set_xlim(0,50050)
#             ax_col.set_ylim(-4,8.2)
#             ax_col.set_xticks(np.linspace(0,50000,6))
#             ax_col.set_yticks(np.linspace(-4,8,7))
#             ax_col.tick_params(axis='x', which='both',labelsize = 30, size = 10,width = 5,direction = 'in',pad = 10)
#             ax_col.tick_params(axis='y', which='both',labelsize = 35, size = 10,width = 5,direction = 'in',pad = 10)
#             [l.set_visible(False) for (i,l) in enumerate(ax_col.yaxis.get_ticklabels()) if i % 2 != 0] #Hide every other label
#             # ax_col.text(-1,0.1,'({})'.format(string.ascii_uppercase[counter]),transform=ax.transAxes,
#             #         size=35, weight='bold')
#             ax_col.xaxis.offsetText.set_fontsize(0)
#             ax_col.set_aspect(50050/12)
#         elif n_row == 1:
#             ax_col.set_xlim(-0.6,0.6)
#             ax_col.set_ylim(-1.1,1.1)
#             ax_col.set_xticks(np.linspace(-0.5,0.5,5))
#             [l.set_visible(False) for (i,l) in enumerate(ax_col.xaxis.get_ticklabels()) if i % 2 != 0] #Hide every other label
#             [l.set_visible(False) for (i,l) in enumerate(ax_col.yaxis.get_ticklabels()) if i % 2 != 0] #Hide every other label
#             ax_col.set_yticks(np.linspace(-1,1,5))
#             ax_col.tick_params(axis='x', which='both',labelsize = 30, size = 10,width = 5,direction = 'in',pad = 10)
#             ax_col.tick_params(axis='y', which='both',labelsize = 35, size = 10,width = 5,direction = 'in',pad = 10)
#             # ax_col.text(0.03,0.92,'({})'.format(string.ascii_uppercase[counter]),transform=ax.transAxes,
#             #         size=35, weight='bold')
#             ax_col.xaxis.offsetText.set_fontsize(0)
#             ax_col.set_aspect(1.2/2.2)
            
# ## Subfigure denotion texts
# axes[0,0].text(-2.07,2.22,'({})'.format(string.ascii_uppercase[0]),transform=ax.transAxes,
#                     size=35, weight='bold')
# axes[0,1].text(-1.03,2.22,'({})'.format(string.ascii_uppercase[1]),transform=ax.transAxes,
#                     size=35, weight='bold')
# axes[0,2].text(0.02,2.22,'({})'.format(string.ascii_uppercase[2]),transform=ax.transAxes,
#                     size=35, weight='bold')
# axes[1,0].text(-2.07,0.92,'({})'.format(string.ascii_uppercase[3]),transform=ax.transAxes,
#                     size=35, weight='bold')
# axes[1,1].text(-1.03,0.92,'({})'.format(string.ascii_uppercase[4]),transform=ax.transAxes,
#                     size=35, weight='bold')
# axes[1,2].text(0.02,0.92,'({})'.format(string.ascii_uppercase[5]),transform=ax.transAxes,
#                     size=35, weight='bold')

# ## Inset images
# axins_1 = inset_axes(axes[0,0], width="20%", height="20%", loc='upper right')
# axins_2 = inset_axes(axes[0,1], width="20%", height="20%", loc='upper right')
# axins_3 = inset_axes(axes[0,2], width="20%", height="20%", loc='upper right')
# axins_1.plot(K_constant_prof.s,K_constant_prof.K,color = 'black',lw = 4)
# axins_2.plot(K_gaussian_left_l_stiff_prof.s,K_gaussian_left_l_stiff_prof.K,color = 'black',lw = 4)
# axins_3.plot(K_error_prof.s,K_error_prof.K,color = 'black',lw = 4)

# for n,ax in enumerate([axins_1,axins_2,axins_3]):
#     ax.set_xlim(-0.6,0.6)
#     ax.set_ylim(0.3,3.2)
#     # ax.set_xticks(np.linspace(-0.5,0.5,5))
#     # ax.set_yticks(np.linspace(0.5,3,6))
#     # ax.tick_params(axis='x', which='major',labelsize = 9, size = 4,width = 2,rotation = 45)
#     ax.xaxis.set_visible(False)
#     ax.yaxis.set_visible(False)
#     # ax.tick_params(axis='y', which='major',labelsize = 9, size = 4,width = 2)
#     # ax.set_xlabel(r"Arclength $[s]$",fontsize = 9,labelpad = 5)
#     # ax.set_ylabel(r"$\kappa (s)$",fontsize = 9,labelpad = 5)
#     ax.set_aspect(1.2/2.9)
    
    
# #Top row of plots formatting
# axes[0,0].set(xlabel=None)
# axes[0,2].set(xlabel=None)
# axes[0,0].legend([],[],frameon = False)
# axes[0,1].legend([],[],frameon = False)
# axes[0,2].legend(loc='center left', 
#                 bbox_to_anchor=(1, 0.5),prop={'size': 20},title= "Mode Number").get_title().set_fontsize("25")
# axes[0,1].set_xlabel(r"$\bar{\mu}\times 10^{4}$",fontsize = 35,labelpad = 20)
# axes[0,0].set_ylabel(r"$\mathbb{R}(\sigma)$",fontsize = 35,labelpad = 40)


# #Bottom row of plots formatting
# axes[1,0].set(xlabel=None)
# axes[1,2].set(xlabel=None)
# axes[1,0].legend([],[],frameon = False)
# axes[1,1].legend([],[],frameon = False)
# axes[1,2].legend(loc='center left', 
#                 bbox_to_anchor=(1, 0.5),prop={'size': 20},title= "Mode Number").get_title().set_fontsize("25")
# axes[1,1].set_xlabel(r"$s$",fontsize = 35,labelpad = 20)
# axes[1,0].set_ylabel(r"$\hat{h}(s)$",fontsize = 35,labelpad = 2)
# fig.savefig(os.path.join(output_dir,'04_all_eigenspectrum_maps_shapes_insets.png'),bbox_inches = 'tight',dpi = 200)
# fig.savefig(os.path.join(output_dir,'04_all_eigenspectrum_maps_shapes_insets.pdf'),bbox_inches = 'tight',dpi = 200)
# fig.savefig(os.path.join(output_dir,'04_all_eigenspectrum_maps_shapes_insets.eps'),
#             bbox_inches = 'tight',format = 'eps',dpi = 200)
# plt.show() 