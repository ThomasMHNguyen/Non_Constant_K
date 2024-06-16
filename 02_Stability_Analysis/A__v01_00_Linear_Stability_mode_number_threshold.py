# -*- coding: utf-8 -*-
"""
FILE NAME:      A__v01_00_Linear_Stability_mode_number_threshold.py


COMPLEMENTARY
SCRIPT(S)/
FILE(S):        N/A


DESCRIPTION:    For a given rigidity profile, this script will calculate when a particular mode
                number becomes unstable (sigma = 0). 

INPUT
FILES(S):       None
                

OUTPUT
FILES(S):       

1)              .CSV file that lists the approximate mu_bar value for a a mode number when sigma is 
                equal to 0.

INPUT
ARGUMENT(S):    None

None
CREATED:        01Feb22

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

TO DO LIST:     
1)              Clean up code.
2)              Create classes for better readibility.

NOTE(S):        N/A

"""

import os, re, math
import numpy as np
import seaborn as sns
from scipy.linalg import eig
from scipy import special, sparse
import matplotlib.pyplot as plt
import pandas as pd
# https://web.ics.purdue.edu/~nowack/geos657/lecture6-dir/lecture6.htm
N = 1001
L = 1
ds = L/(N-1)
s  = np.linspace(-L/2,L/2,N)
c =np.log(0.01**2*np.exp(1))

rigidity_suffix = 'K_dirac_left_l_stiff'
output_dir = 'C:/Users/super/OneDrive - University of California, Davis/School/UCD_Files/Work/Scripts/Perturbation Analysis/Mode_Number_Threshold/'
output_dir = os.path.join(output_dir,rigidity_suffix)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)



if rigidity_suffix == 'K_constant':
    K = np.ones(s.shape[0],dtype = float)
    Ks = np.zeros(s.shape[0],dtype = float)
    Kss = np.zeros(s.shape[0],dtype = float)
    rigidity_title = r"$\kappa(s) = 1$"
    
elif rigidity_suffix == 'K_parabola_center_l_stiff':
    K = 1/2 + 2*s**2
    Ks = 4*s
    Kss = 4*np.ones(s.shape[0],dtype = float)
    rigidity_title = r"$\kappa(s) = \frac{1}{2} + 2s^{2} $"
    
elif rigidity_suffix == 'K_parabola_center_m_stiff':
    K = 1.5 - 2*(s**2)
    Ks = -4*s
    Kss = -4*np.ones(s.shape[0],dtype = float)
    rigidity_title = r"$\kappa(s) = \frac{3}{2} - 2s^{2} $"
    
elif rigidity_suffix == 'K_linear':
    K = s+1
    Ks = 1*np.ones(s.shape[0],dtype = float)
    Kss = 0*np.ones(s.shape[0],dtype = float)
    rigidity_title = r"$\kappa(s) = s+1 $"
    
elif rigidity_suffix == 'K_dirac_center_l_stiff':
    K = 1-0.5*np.exp(-100*s**2)
    Ks = 100*s*np.exp(-100*s**2)
    Kss = np.exp(-100*s**2)*(100-2e4*s**2)
    rigidity_title = r"$\kappa(s) = 1- \frac{1}{2} e^{-100s^{2}} $"
    
elif rigidity_suffix == 'K_dirac_center_l_stiff2':
    K = 1-0.5*np.exp(-500*s**2)
    Ks = 500*s*np.exp(-500*s**2)
    Kss = np.exp(-500*s**2)*(500-5e5*s**2)
    rigidity_title = r"$\kappa(s) = 1- \frac{1}{2} e^{-500s^{2}} $"
    
elif rigidity_suffix == 'K_dirac_center_m_stiff':
    K = 1+np.exp(-100*s**2)
    Ks = -200*s*np.exp(-100*s**2)
    Kss = 200*np.exp(-100*s**2)*(200*s**2-1)
    rigidity_title = r"$\kappa(s) = 1 + e^{-100s^{2}} $"
    
elif rigidity_suffix == 'K_parabola_shifted':
    K = 1.5-0.5*(s-0.5)**2
    Ks = -1*s-0.5
    Kss = -1*np.ones(s.shape[0],dtype = float)
    rigidity_title = r"$\kappa(s) = \frac{3}{2}-\frac{1}{2}\left(s-\frac{1}{2}\right)^{2} $"
    
elif rigidity_suffix == 'K_error_function':
    K = special.erf(10*s)+2
    Ks = (20/np.sqrt(np.pi))*np.exp(-100*s**2)
    Kss = (-4000*s/np.sqrt(np.pi))*np.exp(-100*s**2)
    rigidity_title = r"$\kappa(s) = 2 + erf(10s) $"
    
elif rigidity_suffix == 'K_dirac_left_l_stiff':
    K = 1-0.5*np.exp(-100*(s+0.25)**2)
    Ks = 100*(s+0.25)*np.exp(-100*(s+0.25)**2)
    Kss = np.exp(-100*(s+0.25)**2)*-2e4*(s**2+0.5*s+0.0575)
    rigidity_title = r"$\kappa (s) = 1 - \frac{1}{2} e^{-100\left(s + \frac{1}{4}\right)^{2}}$"
    
elif rigidity_suffix == 'K_dirac_left_l_stiff2':
    K = 1-0.5*np.exp(-500*(s+0.25)**2)
    Ks = 500*(s+0.25)*np.exp(-500*(s+0.25)**2)
    Kss = np.exp(-500*(s+0.25)**2)*-5e5*(s**2+0.5*s+0.0615)
    rigidity_title = r"$\kappa(s) = 1- \frac{1}{2} e^{-500\left(s + \frac{1}{4}\right)^{2}} $"
    
elif rigidity_suffix == 'K_dirac_left_l_stiff3':
    K = 1-0.5*np.exp(-1000*(s+0.25)**2)
    Ks = 1000*(s+0.25)*np.exp(-1000*(s+0.25)**2)
    Kss = np.exp(-1000*(s+0.25)**2)*-2e6*(s**2+0.5*s+0.062)
    rigidity_title = r"$\kappa(s) = 1- \frac{1}{2} e^{-1000\left(s + \frac{1}{4}\right)^{2}} $"
#%%
def calculate_lhs_matrix_comp():
    """
    This function will calculate all of the components in the sparse 
    coefficient matrix needed to solve the eigenvalue-eigenvector problem.
    
    Input Variables:
    mu_bar: non-dimensionalized ratio of viscous drag forces to elastic forces.
    """ 
    global c, ds, s, N, K, Ks, Kss
    
    zeta = (K/ds**4) + ((2*Ks)/(-2*ds**3))
    epsilon = (-4*K/ds**4) + (2*Ks/ds**3) + (Kss/ds**2)
    alpha = (6*K/ds**4) + (-2*Kss/ds**2)
    beta = (-4*K/ds**4) + (-2*Ks/ds**3) + (Kss/ds**2)
    gamma = (K/ds**4) + ((2*Ks)/(2*ds**3))
    
    lhs_diags = [zeta[2:],epsilon[1:],alpha,beta[:-1],gamma[:-2]]
    return lhs_diags

def calculate_rhs_matrix_comp():
    global c, ds, s, N
    
    tension_adj = 1/(4*c)*(0.25-s**2)
    dTds = -s/(2*c)
    dT2ds2 = -1/(2*c)
    
    beta = (tension_adj/ds**2) + (2*dTds/(-2*ds))
    alpha = (-2*tension_adj/ds**2)+(2*dT2ds2)
    gamma = (tension_adj/ds**2) + (2*dTds/(2*ds))
    rhs_diags = [beta[1:],alpha,gamma[:-1]]
    return rhs_diags

def apply_boundary_conditions(matrix):
    
    #Coefficients
    A = float(48/11)
    B = float(-52/11)
    C = float(15/11)
    D = float(28/11)
    E = float(-23/11)
    F = float(6/11)
    
    #Foward differences
    matrix[2,2] += A*matrix[2,0] + D*matrix[2,1]
    matrix[2,3] += B*matrix[2,0] + E*matrix[2,1]
    matrix[2,4] += C*matrix[2,0] + F*matrix[2,1]
    matrix[3,2] += D*matrix[3,1]
    matrix[3,3] += E*matrix[3,1]
    matrix[3,4] += F*matrix[3,1]

    #Backward differences
    matrix[-3,-3] += A*matrix[-3,-1] + D*matrix[-3,-2]
    matrix[-3,-4] += B*matrix[-3,-1] + E*matrix[-3,-2]
    matrix[-3,-5] += C*matrix[-3,-1] + F*matrix[-3,-2]
    matrix[-4,-3] += D*matrix[-4,-2] 
    matrix[-4,-4] += E*matrix[-4,-2] 
    matrix[-4,-5] += F*matrix[-4,-2]
    
    return matrix
    
    
def solve_buckling_threshold(lhs_matrix,rhs_matrix):
    
    #Coefficients
    A = float(48/11)
    B = float(-52/11)
    C = float(15/11)
    D = float(28/11)
    E = float(-23/11)
    F = float(6/11)
    
    #Adjust matrix for boundary conditions
    lhs_matrix,rhs_matrix = apply_boundary_conditions(lhs_matrix),\
        apply_boundary_conditions(rhs_matrix)
    lhs_matrix = lhs_matrix[2:-2,2:-2]
    rhs_matrix = rhs_matrix[2:-2,2:-2]
    
    #Solve for right-hand side eigenvalues & eigenvectors
    eigenvalues, eigenvectors = eig(a = lhs_matrix,b = rhs_matrix)
#     eigenvalues, eigenvectors = eigenvalues.real, eigenvectors.real
    
    #Sort eigenvalues and eigenvectors in increasing order
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigvecs_s= eigenvectors[:,idx]

    #Solve for h1, h2, h_n, h_n-1
    eigvecs_t = eigvecs_s.T
#     eigvecs_t = eigvecs_s.copy()
    h3 = eigvecs_t[:,0]
    h4 = eigvecs_t[:,1]
    h5 = eigvecs_t[:,2]
    n_2 = eigvecs_t[:,-1]
    n_3 = eigvecs_t[:,-2]
    n_4 = eigvecs_t[:,-3]
    h1 = (A*h3) + (B*h4) + (C*h5)
    h2 = (D*h3) + (E*h4) + (F*h5)
    n_0 = (A*n_2) + (B*n_3) + (C*n_4)
    n_1 = (D*n_2) + (E*n_3) + (F*n_4)
    
    eigvecs_all = np.column_stack((h1,h2,eigvecs_t,n_1,n_0))
    for row_n in range(0,eigvecs_all.shape[0]):
       eigvecs_all[row_n,:] = eigvecs_all[row_n,:]/np.absolute(eigvecs_all[row_n,:]).max()
    eigvecs_all = eigvecs_all.T
    return np.round(eigenvalues,4), eigvecs_all

def find_useful_eigen(eigenvals,eigenfuncs):
    
    #Ignore trivial solutions
    true_eigenvals = np.round(eigenvals[2:],4)
    true_eigenfuncs = np.round(eigenfuncs[:,2:],4)
    
    return true_eigenvals,true_eigenfuncs

def append_data(eigenvals,eigenfuncs):
    global s
    mode_num_thres_df = pd.DataFrame()
    
    for i,v in enumerate(eigenvals):
        indiv_thres_datapoint = pd.DataFrame(index = range(s.shape[0]),
                                             columns = ['s','Mode Number','Eigenvalues_real',
                                                        'Eigenvalues_im','Eigenfunctions_real','Eigenfunctions_im'])
        indiv_thres_datapoint['s'] = s
        indiv_thres_datapoint['Mode Number'] = i+1
        indiv_thres_datapoint['Eigenvalues_real'] = v.real
        indiv_thres_datapoint['Eigenvalues_im'] = v.imag
        indiv_thres_datapoint['Eigenfunctions_real'] = eigenfuncs[:,i].real
        indiv_thres_datapoint['Eigenfunctions_im'] = eigenfuncs[:,i].imag
        mode_num_thres_df = pd.concat([mode_num_thres_df,indiv_thres_datapoint],ignore_index = True)
    
    return mode_num_thres_df
        
        
        
    
    
    
#%% Determine instability modes

#1) Create Diagonal terms based on finite differences
lhs_diags = calculate_lhs_matrix_comp()
rhs_diags = calculate_rhs_matrix_comp()

#1a) Create Matrix, Solve & Append to DataFrame
lhs_big_m = sparse.diags(lhs_diags, offsets = [-2,-1,0,1,2],
                 shape = (N,N)).toarray()
rhs_big_m = sparse.diags(rhs_diags, offsets = [-1,0,1],
                 shape = (N,N)).toarray()
eigvals,eigvecs = solve_buckling_threshold(lhs_big_m,rhs_big_m)
eigvals,eigvecs = find_useful_eigen(eigvals,eigvecs)
mode_num_inst_df = append_data(eigvals,eigvecs)

#%% Plot Results

fil_mode_num_inst_df = mode_num_inst_df[(mode_num_inst_df['Mode Number'] <= 3)]
fil_mode_num_inst_df.sort_values(by = ['Mode Number'],ascending = True,inplace = True)
plt.figure(figsize = (11,11))
sns.lineplot(x = 's',y = 'Eigenfunctions_real',hue = 'Mode Number',ci = None,
                          palette = 'bright',data = fil_mode_num_inst_df,linewidth = 4)
ax = plt.gca()
ax.set_title(r"First Three Modes Representative Shapes:" 
                          "\n" r"{0}".format(rigidity_title),
                      fontsize = 35,pad = 20)
ax.set_xlabel("s",fontsize = 35,labelpad = 20)
ax.set_ylabel(r"Normalized $\hat{{h}}(s)$",fontsize = 35,labelpad = 20)
ax.set_ylim(-1.1,1.1)
ax.set_xlim(-0.55,0.55)
ax.set_xticks(np.linspace(-0.5,0.5,5))
ax.set_yticks(np.linspace(-1,1,5))
ax.grid(b=True, which='major',color='black', linewidth=0.3)
ax.tick_params(axis='both', which='major', labelsize=35)
plt.legend(loc='center left', 
                bbox_to_anchor=(1, 0.5),prop={'size': 25},title= "Mode Number")
filename = os.path.join(output_dir,'first_three_mode_shapes_{}.png'.format(rigidity_suffix))
plt.savefig(filename,
                dpi = 600,bbox_inches = 'tight')
plt.show()


mode_num_inst_df[mode_num_inst_df['Mode Number'] <= 6].to_csv(os.path.join(output_dir,'mode_number_threshold_{}.csv'.format(rigidity_suffix)))
