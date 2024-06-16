# -*- coding: utf-8 -*-
"""
FILE NAME:      B__v03_00_Plot_Rigidity_Profiles_subplots.py

COMPLEMENTARY
SCRIPT(S)/
FILE(S):        N/A

DESCRIPTION:    This script will plot the analytical form for the rigidity of 
                the filament on multiple subplots on a single canvas. 

INPUT
FILES(S):       None

OUTPUT
FILES(S):       1) .PNG file that plots the chosen rigidity profiles as a 
                function of s on subplots.
                2) .PDF file that plots the chosen rigidity profiles as a 
                function of s on subplots.  
                3) .EPS file that plots the chosen rigidity profiles as a 
                function of s on subplots.       


INPUT
ARGUMENT(S):    None

CREATED:        02Jan21

MODIFICATIONS
LOG:

04Mar22:        Added functionality to include generating .PDF and .EPS files 
                of rigidity profile.

    
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

class Constants:
    """
    This class will bundle up all constants and parameters needed in the simulations to be 
    easily accessed in the functions due to multi-processing implementation.
    """
    
    ########
    
    output_dir = 'C:/Users/super/OneDrive - University of California, Davis/Research/00_Projects/01_Non_Constant_K/00_Scripts/01_Filament_Movement/04_Dissertation/rigidity_profile/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
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
            
#%% Generate Profiles

#Initialize profiles for each rigidity profile
K_constant_prof = Constants('K_constant', 101)
K_gaussian_left_l_stiff_prof = Constants('K_dirac_left_l_stiff', 101)
K_error_prof = Constants('K_error_function', 101)

K_constant_col_code = '#32CD32'
K_dirac_left_l_stiff_col_code = '#1E90FF'
K_error_function_col_code = '#FF7F50'
#%% Plot data on subplots
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": "Times",
    'text.latex.preamble': r'\usepackage{amsmath}'})


fig,axes = plt.subplots(ncols = 3,figsize = (10,7),layout = 'constrained',sharey = True)
subfigure_text = [r"$\textbf{(a)}$",r"$\textbf{(b)}$",r"$\textbf{(c)}$"]

axes[0].plot(K_constant_prof.s,K_constant_prof.K,color = K_constant_col_code,lw = 2)
axes[1].plot(K_gaussian_left_l_stiff_prof.s,K_gaussian_left_l_stiff_prof.K,color = K_dirac_left_l_stiff_col_code,lw = 2)
axes[2].plot(K_error_prof.s,K_error_prof.K,color = K_error_function_col_code,lw = 2)
plt.axis()
for n,ax in enumerate(axes):
    ax.set_xlim(-0.6,0.6)
    ax.set_ylim(0.3,3.2)
    ax.set_xticks(np.linspace(-0.5,0.5,5))
    ax.set_yticks(np.linspace(0.5,3,6))
    ax.tick_params(axis='both', which='both',labelsize = 13, length = 5,size = 2,pad = 5,direction = 'in')
    [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_ticklabels()) if i % 2 != 0] #Hide every other label
    ax.text(-0.60,3.27,r'{}'.format(subfigure_text[n]),size=17)
    ax.set_aspect(np.diff(ax.get_xlim())/np.diff(ax.get_ylim()))
    ax.set_xlabel(r"$s$",size= 15,labelpad = 5)
    
    
# fig.supxlabel(r"$s$",fontsize = 15,y = 0.05)
fig.supylabel(r"$B (s)$",fontsize = 15,x=-0.030)
fig.savefig(os.path.join(Constants.output_dir,'filament_rigidity_profiles_all.png'),bbox_inches = 'tight',dpi = 600)
fig.savefig(os.path.join(Constants.output_dir,'filament_rigidity_profiles_all.pdf'),format = 'pdf',bbox_inches = 'tight',dpi = 600)
fig.savefig(os.path.join(Constants.output_dir,'filament_rigidity_profiles_all.eps'),
            bbox_inches = 'tight',format = 'eps',dpi = 600)
plt.show()

