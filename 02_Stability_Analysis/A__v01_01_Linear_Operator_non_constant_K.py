# -*- coding: utf-8 -*-
"""
FILE NAME:      A__v01_01_Linear_Operator_non_constant_K.py


COMPLEMENTARY
SCRIPT(S)/
FILE(S):        N/A


DESCRIPTION:    This script will calculate the eigenvalues and eigenfunctions that correspond to 
                both the linear and adjoint operator for a given rigidity profile. For these values,
                a corresponding normalization coefficient for amplitude calculations and their respective
                mode number will also be labeled as well.

INPUT
FILES(S):       

1)              .CSV files that list the mode number corresponding to a particular eigenvalue.
                

OUTPUT
FILES(S):       

1)              .CSV file that lists the linear operator's eigenfunctions and eigenvalues, the respective 
                adjoint operator's eigenfunctions and eigenvalues, their respective normalization coefficents,
                and mode numbers.
2)              .CSV file that is interpolated from above for the number of points used for linear simulations.


INPUT
ARGUMENT(S):    

1) output_directory:                Path to directory where the resulting .CSV files will be created.
2) mode_data_directory:             Path to directory where the .CSV files that classify the mode number for each
                                    eigenvalue is located.
3) rigidity_type:                   Type of rigidity profile used for analysis.


CREATED:        05Oct21

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

import os, re, math, argparse
import numpy as np
import seaborn as sns
from numpy.linalg import eig
from scipy import special, sparse
import matplotlib.pyplot as plt
import pandas as pd
# https://web.ics.purdue.edu/~nowack/geos657/lecture6-dir/lecture6.htm
class filament_parameters():
    """
    This class will bundle up all constants and parameters needed for analysis.
    """
    def __init__(self,rigid_func, N, L, epsilon,mu_bar_low,mu_bar_high,mu_bar_range):
        
        #Constant parameters
        self.N = N
        self.L = L
        self.rigidity_suffix = rigid_func
        self.ds = L/(N-1)
        self.s = np.linspace(-self.L/2,self.L/2,N)
        self.c = np.log(epsilon**2*np.exp(1))
        self.mu_bar_vals = range(int(mu_bar_low),int(mu_bar_high),mu_bar_range)
        
        #Rigidity functions
        if self.rigidity_suffix == 'K_constant':
            self.K = np.ones(self.s.shape[0],dtype = float)
            self.Ks = np.zeros(self.s.shape[0],dtype = float)
            self.Kss = np.zeros(self.s.shape[0],dtype = float)
            self.rigidity_title = r"$\kappa(s) = 1 $"
            
        elif self.rigidity_suffix == 'K_parabola_center_l_stiff':
            self.K = 1/2 + 2*self.s**2
            self.Ks = 4*self.s
            self.Kss = 4*np.ones(self.s.shape[0],dtype = float)
            self.rigidity_title = r"$\kappa(s) = \frac{1}{2} + 2s^{2} $"
            
        elif self.rigidity_suffix == 'K_parabola_center_m_stiff':
            self.K = 1.5 - 2*(self.s**2)
            self.Ks = -4*self.s
            self.Kss = -4*np.ones(self.s.shape[0],dtype = float)
            self.rigidity_title = r"$\kappa(s) = \frac{3}{2} - 2s^{2} $"
            
        elif self.rigidity_suffix == 'K_linear':
            self.K = self.s+1
            self.Ks = 1*np.ones(self.s.shape[0],dtype = float)
            self.Kss = 0*np.ones(self.s.shape[0],dtype = float)
            self.rigidity_title = r"$\kappa(s) = s+1 $"
            
        elif self.rigidity_suffix == 'K_dirac_center_l_stiff':
            self.K = 1-0.5*np.exp(-100*self.s**2)
            self.Ks = 100*self.s*np.exp(-100*self.s**2)
            self.Kss = np.exp(-100*self.s**2)*(100-2e4*self.s**2)
            self.rigidity_title = r"$\kappa(s) = 1- \frac{1}{2} e^{-100s^{2}} $"
            
        elif self.rigidity_suffix == 'K_dirac_center_l_stiff2':
            self.K = 1-0.5*np.exp(-500*self.s**2)
            self.Ks = 500*self.s*np.exp(-500*self.s**2)
            self.Kss = np.exp(-500*self.s**2)*(500-5e5*self.s**2)
            self.rigidity_title = r"$\kappa(s) = 1- \frac{1}{2} e^{-500s^{2}} $"
            
        elif self.rigidity_suffix == 'K_dirac_center_m_stiff':
            self.K = 1+np.exp(-100*self.s**2)
            self.Ks = -200*self.s*np.exp(-100*self.s**2)
            self.Kss = 200*np.exp(-100*self.s**2)*(200*self.s**2-1)
            self.rigidity_title = r"$\kappa(s) = 1 + e^{-100s^{2}} $"
            
        elif self.rigidity_suffix == 'K_parabola_shifted':
            self.K = 1.5-0.5*(self.s-0.5)**2
            self.Ks = -1*self.s-0.5
            self.Kss = -1*np.ones(self.s.shape[0],dtype = float)
            self.rigidity_title = r"$\kappa(s) = \frac{3}{2}-\frac{1}{2}\left(s-\frac{1}{2}\right)^{2} $"
            
        elif self.rigidity_suffix == 'K_error_function':
            self.K = special.erf(10*self.s)+2
            self.Ks = (20/np.sqrt(np.pi))*np.exp(-100*self.s**2)
            self.Kss = (-4000*self.s/np.sqrt(np.pi))*np.exp(-100*self.s**2)
            self.rigidity_title = r"$\kappa(s) = 2 + erf(10s) $"
            
        elif self.rigidity_suffix == 'K_dirac_left_l_stiff':
            self.K = 1-0.5*np.exp(-100*(self.s+0.25)**2)
            self.Ks = 100*(self.s+0.25)*np.exp(-100*(self.s+0.25)**2)
            self.Kss = np.exp(-100*(self.s+0.25)**2)*-2e4*(self.s**2+0.5*self.s+0.0575)
            self.rigidity_title = r"$\kappa (s) = 1 - \frac{1}{2} e^{-100\left(s + \frac{1}{4}\right)^{2}}$"
            
        elif self.rigidity_suffix == 'K_dirac_left_l_stiff2':
            self.K = 1-0.5*np.exp(-500*(self.s+0.25)**2)
            self.Ks = 500*(self.s+0.25)*np.exp(-500*(self.s+0.25)**2)
            self.Kss = np.exp(-500*(self.s+0.25)**2)*-5e5*(self.s**2+0.5*self.s+0.0615)
            self.rigidity_title = r"$\kappa(s) = 1- \frac{1}{2} e^{-500\left(s + \frac{1}{4}\right)^{2}} $"
            
        elif self.rigidity_suffix == 'K_dirac_left_l_stiff3':
            self.K = 1-0.5*np.exp(-1000*(self.s+0.25)**2)
            self.Ks = 1000*(self.s+0.25)*np.exp(-1000*(self.s+0.25)**2)
            self.Kss = np.exp(-1000*(self.s+0.25)**2)*-2e6*(self.s**2+0.5*self.s+0.062)
            self.rigidity_title = r"$\kappa(s) = 1- \frac{1}{2} e^{-1000\left(s + \frac{1}{4}\right)^{2}} $"


#%% Functions

def matr_diags(filament_parameters,mu_bar,calc_type):
    """
    This function calculates the diagonal components of the matrix used in the
    eigenvalue-eigenfunction problem. Terminology listed here:
    zeta- diagonal component 2 units below from main diagona
    gamma- diagonal component 2 units above from main diagonal
    pi- diagonal component 1 unit lower than main diagonal
    alpha- main diagonal component
    beta- diagonal component 1 unit higher than main diagonal
    
    Inputs:
    filament_parameters: Class that contains all of the parameters for filament analysis. 
    mu_bar: mu_bar value relating viscous forces to elastic forces
    calc_type: string to denot whether to calculate for linear operator or 
    adjoint operator
    """
    N, c, ds, s = filament_parameters.N, filament_parameters.c, \
        filament_parameters.ds,filament_parameters.s
    K,Ks,Kss = filament_parameters.K, filament_parameters.Ks, filament_parameters.Kss
    tension = (float(mu_bar)/4/c)*(0.25-s**2)
    dTds = -float(mu_bar)*s/2/c
    d2Tds2 = -float(mu_bar)*np.ones(N)/2/c
    
    zeta = (K/ds**4) + (2*Ks/(-2*ds**3))
    gamma = (K/ds**4) + (2*Ks/(2*ds**3))
    
    if calc_type == 'linear':
        epsilon = (-4*K/ds**4) + (2*Ks/ds**3) + (Kss/ds**2) + (-tension/ds**2) +\
        (-2*-dTds/(2*ds)) 
        alpha = (6*K/ds**4) + (-2*Kss/ds**2) + (-2*-tension/ds**2) + (mu_bar/c)
        # alpha = (6*K/ds**4) + (-2*Kss/ds**2) + (-2*-tension/ds**2)
        beta = (-4*K/ds**4) + (-2*Ks/ds**3) + (Kss/ds**2) + (-tension/ds**2) +\
        (-2*dTds/(2*ds))
    elif calc_type == 'adjoint':
        epsilon = (-4*K/ds**4) + (2*Ks/ds**3) + (Kss/ds**2) + (-tension/ds**2)
        alpha = (6*K/ds**4) + (-2*Kss/ds**2) + (-2*-tension/ds**2) + d2Tds2 + (mu_bar/c)
        # alpha = (6*K/ds**4) + (-2*Kss/ds**2) + (-2*-tension/ds**2) + d2Tds2
        beta = (-4*K/ds**4) + (-2*Ks/ds**3) + (Kss/ds**2) + (-tension/ds**2)
        
    diags = [zeta[2:],epsilon[1:],alpha,beta[:-1],gamma[:-2]]
    return diags

def calc_linear_eigen(filament_parameters,matrix,mu_bar):
    """
    This function implements the boundary conditions for the problem into the 
    matrix and solves the eigenvalue-eigenfunction problem of the linear operator.
    
    Inputs:
    filament_parameters:    Class that contains all of the parameters for filament analysis. 
    matrix:                 Matrix that contains the derivatives approximated as finite differences.
    mu_bar:                 mu_bar value relating viscous forces to elastic forces.
    """
    c = filament_parameters.c
    #Coefficients for Boundary Conditiosn
    A = float(48/11)
    B = float(-52/11)
    C = float(15/11)
    D = float(28/11)
    E = float(-23/11)
    F = float(6/11)
    
    #Forward differences
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
    
    #Adjust matrix for boundary conditions
    matrix = (c/mu_bar)*matrix
    matrix = matrix[2:-2,2:-2]
    
    #Solve for right-hand side eigenvalues & eigenvectors
    eigenvalues, eigenvectors = eig(matrix)
    
    #Sort eigenvalues and eigenvectors in increasing order
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigvecs_s= eigenvectors[:,idx]

    #Solve for phi_1, phi_2, phi__n, phi_n-1
    eigvecs_t = eigvecs_s.T #eigenvalues[i] = eigvecs_s[:,i] --> put discretized phi component as column value
    phi_2 = eigvecs_t[:,0]
    phi_3 = eigvecs_t[:,1]
    phi_4 = eigvecs_t[:,2]
    phi_n_3 = eigvecs_t[:,-1]
    phi_n_4 = eigvecs_t[:,-2]
    phi_n_5 = eigvecs_t[:,-3]
    
    phi_0 = (A*phi_2) + (B*phi_3) + (C*phi_4)
    phi_1 = (D*phi_2) + (E*phi_3) + (F*phi_4)
    phi_n_1 = (A*phi_n_3) + (B*phi_n_4) + (C*phi_n_5)
    phi_n_2 = (D*phi_n_3) + (E*phi_n_4) + (F*phi_n_5)

    #Append corrected end terms
    eigvecs_all = np.column_stack((phi_0,phi_1,eigvecs_t,phi_n_2,phi_n_1))
    eigvecs_all = eigvecs_all.T   
    
    return eigenvalues, eigvecs_all

def calc_adjoint_eigen(filament_parameters,matrix,mu_bar):
    """
    This function implements the boundary conditions for the problem into the 
    matrix and solves the eigenvalue-eigenfunction problem of the adjoint operator.
    
    Inputs:
    filament_parameters:    Class that contains all of the parameters for filament analysis. 
    matrix:                 Matrix that contains the derivatives approximated as finite differences.
    mu_bar:                 mu_bar value relating viscous forces to elastic forces.
    """
    c, ds = filament_parameters.c, filament_parameters.ds
    K = filament_parameters.K
    
    #Coefficients for boundary conditions
    prefactor1 = (float(22)*K[0]*c)+(5*mu_bar*ds**3)
    prefactor2 = (float(22)*K[-1]*c)+(5*mu_bar*ds**3)
    #Forward coefficients
    A = float(96*K[0]*c/prefactor1)
    B = float(-104*K[0]*c/prefactor1)
    C = float(30*K[0]*c/prefactor1)
    D = float(((56*K[0]*c)+(4*mu_bar*ds**3))/prefactor1)
    E = float(((-46*K[0]*c)-(mu_bar*ds**3))/prefactor1)
    F = float(12*K[0]*c/prefactor1)
    #Backward coefficients
    G = float(96*K[-1]*c/prefactor2)
    H = float(-104*K[-1]*c/prefactor2)
    I = float(30*K[-1]*c/prefactor2)
    J = float(((56*K[-1]*c)+(4*mu_bar*ds**3))/prefactor2)
    L = float(((-46*K[-1]*c)-(mu_bar*ds**3))/prefactor2)
    M = float(12*K[-1]*c/prefactor2) 
    
    #Forward differences
    matrix[2,2] += A*matrix[2,0] + D*matrix[2,1]
    matrix[2,3] += B*matrix[2,0] + E*matrix[2,1]
    matrix[2,4] += C*matrix[2,0] + F*matrix[2,1]
    matrix[3,2] += D*matrix[3,1]
    matrix[3,3] += E*matrix[3,1]
    matrix[3,4] += F*matrix[3,1]
    
    #Backward differences
    matrix[-3,-3] += G*matrix[-3,-1] + J*matrix[-3,-2]
    matrix[-3,-4] += H*matrix[-3,-1] + L*matrix[-3,-2]
    matrix[-3,-5] += I*matrix[-3,-1] + M*matrix[-3,-2]
    matrix[-4,-3] += J*matrix[-4,-2] 
    matrix[-4,-4] += L*matrix[-4,-2] 
    matrix[-4,-5] += M*matrix[-4,-2]
    
    #Adjust matrix for boundary conditions
    matrix = (c/mu_bar)*matrix
    matrix = matrix[2:-2,2:-2]
    
    #Solve for right-hand side eigenvalues & eigenvectors
    eigenvalues, eigenvectors = eig(matrix)
    
    #Sort eigenvalues and eigenvectors in increasing order
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigvecs_s= eigenvectors[:,idx]
    
    #Solve for phi_1, phi_2, phi__n, phi_n-1
    eigvecs_t = eigvecs_s.T #eigenvalues[i] = eigvecs_s[:,i] --> put discretized phi component as column value
    phi_2 = eigvecs_t[:,0]
    phi_3 = eigvecs_t[:,1]
    phi_4 = eigvecs_t[:,2]
    phi_n_3 = eigvecs_t[:,-1]
    phi_n_4 = eigvecs_t[:,-2]
    phi_n_5 = eigvecs_t[:,-3]
    
    phi_0 = (A*phi_2) + (B*phi_3) + (C*phi_4)
    phi_1 = (D*phi_2) + (E*phi_3) + (F*phi_4)
    phi_n_1 = (G*phi_n_3) + (H*phi_n_4) + (I*phi_n_5)
    phi_n_2 = (J*phi_n_3) + (L*phi_n_4) + (M*phi_n_5)

    #Append corrected end terms
    eigvecs_all = np.column_stack((phi_0,phi_1,eigvecs_t,phi_n_2,phi_n_1))
    eigvecs_all = eigvecs_all.T    
    return eigenvalues, eigvecs_all


def find_useful_eigenvals(eigenvalues):
    """
    This function filters out for eigenvalues above a certain threshold and aren't 
    trivial solutions. Additionally, it obtains the index values for these eigenvalues of interest.
    
    
    Input variables:
    
    eigenvalues:        array of eigenvalues corresponding to a specific mu_bar value.
    """
    eigenvalues = np.round(eigenvalues,4)
    useful_eigenvals_idx = np.nonzero((eigenvalues.real > -6) & (eigenvalues != 1) & (eigenvalues.real != 2))
    return useful_eigenvals_idx


def remove_conjugate_eigenvalues(eigenvals_all,eigenvals_idx):
    """
    This function checks to see if there's eigenvalues that are the complex conjugates
    of each other and keeps the first one that appears based on ranking.
    
    Inputs:
        
    eigenvalues:        array of eigenvalues corresponding to a specific mu_bar value.
    eigenvals_idx:      array of the eigenvalue indices corresponding to their ranking of ascending order. 
        
    """
    eigenval_dup_df = pd.DataFrame(index = range(eigenvals_idx[0].shape[0]),
                                   columns = ['Eigenvalue_real','Eigenvalue_all',
                                              'Rank'])
    eigenval_dup_df['Eigenvalue_real'] = eigenvals_all[eigenvals_idx[0]].real
    eigenval_dup_df['Eigenvalue_all'] = eigenvals_all[eigenvals_idx[0]]
    eigenval_dup_df['Rank'] = eigenvals_idx[0]
    eigenval_dup_df = eigenval_dup_df.drop_duplicates(subset = ['Eigenvalue_real'],keep = 'first')
    return np.array(eigenval_dup_df['Eigenvalue_real']),np.array(eigenval_dup_df['Rank'])

def append_data(filament_parameters, eigenvalues_idx,lin_eigenvalues,lin_eigenfunc,adjoint_eigenvals,adjoint_eigenfunc,mu_bar,mode_data,all_data):
    """
    This function filters out for eigenvalues above a certain threshold and aren't 
    trivial solutions. Additionally, it obtains the index values for these eigenvalues.
    
    
    Input variables:
    
    eigenvalues: array of eigenvalues corresponding to a specific mu_bar value.
    eigenvalues_idx: array of indices corresponding to eigenvalues to be used.
    eigenvectors: 2-D array of eigenvector values.
    mu_bar: dimensionless ratio of viscous forces to elastic forces
    s: parameterized points on filament.
    """
    s, c = filament_parameters.s, filament_parameters.c
    for i,v in enumerate(eigenvalues_idx):
        mu_bar_df = pd.DataFrame(index = range(0,len(s)),columns = ['Mu_bar','s','c','Linear_Eigenvals_real','Linear_Eigenvals_im',
                                                                    'Linear_Eigenfuncs_real','Linear_Eigenfuncs_im',
                                                                    'Adjoint_Eigenvals_real','Adjoint_Eigenvals_im',
                                                                    'Adjoint_Eigenfuncs_real','Adjoint_Eigenfuncs_im',
                                                                    'Coefficient_real','Coefficient_im','Norm_Coefficient_real','Mode Number'])
        
        mu_bar_df['Mu_bar'] = mu_bar
        mu_bar_df['s'] = s
        mu_bar_df['c'] = c
        mu_bar_df['Linear_Eigenvals_real'] = lin_eigenvalues[i].real
        mu_bar_df['Linear_Eigenvals_im'] = lin_eigenvalues[i].imag
        mu_bar_df['Linear_Eigenfuncs_real'] = lin_eigenfunc[:,v].real
        mu_bar_df['Linear_Eigenfuncs_im'] = lin_eigenfunc[:,v].imag
        mu_bar_df['Adjoint_Eigenvals_real'] = adjoint_eigenvals[v].real
        mu_bar_df['Adjoint_Eigenvals_im'] = adjoint_eigenvals[v].imag
        mu_bar_df['Adjoint_Eigenfuncs_real'] = adjoint_eigenfunc[:,v].real
        mu_bar_df['Adjoint_Eigenfuncs_im'] = adjoint_eigenfunc[:,v].imag
        mu_bar_df['Coefficient_real'] = np.trapz(
            (lin_eigenfunc[:,v].real*adjoint_eigenfunc[:,v].real) + \
                (lin_eigenfunc[:,v].imag*adjoint_eigenfunc[:,v].imag),x = s) 
        mu_bar_df['Coefficient_im'] = np.trapz(
            (lin_eigenfunc[:,v].real*adjoint_eigenfunc[:,v].imag) + \
                (lin_eigenfunc[:,v].imag*adjoint_eigenfunc[:,v].real),x = s)
        mu_bar_df['Norm_Coefficient_real'] = np.sqrt(np.abs(np.trapz(
            (lin_eigenfunc[:,v].real*adjoint_eigenfunc[:,v].real) + \
                (lin_eigenfunc[:,v].imag*adjoint_eigenfunc[:,v].imag),x = s)))
        fil_mode_data = mode_data[(mode_data['Mu_bar'] == mu_bar) & (mode_data['Eigenvalues_real'] == np.round(lin_eigenvalues[i].real,4))]
        
        mu_bar_df['Mode Number'] = fil_mode_data.loc[mu_bar,'Mode Number']
        all_data = pd.concat([all_data,mu_bar_df],ignore_index = True)
    return all_data

#%% Main Script

### Set-up for input arguements in the script
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("output_directory", 
                    help="Specify the parent directory where the resulting CSV files (will be formatted by rigidity profile) will be saved to",
                type = str)
parser.add_argument("mode_data_directory", 
                    help="Specify the location of the parent directory where the CSV files (formatted by rigidity profile) that list the mode number corresponding to each eigenvalue is",
                type = str)
parser.add_argument("rigidity_type",
                    help = "Specify what kind of rigidity profile this simulation will run on",
                    type = str,
                    choices = {"K_constant","K_parabola_center_l_stiff",'K_parabola_center_l_stiff',
                                'K_parabola_center_m_stiff','K_linear','K_dirac_center_l_stiff',
                                'K_dirac_center_l_stiff2','K_dirac_center_m_stiff','K_parabola_shifted',
                                'K_error_function','K_dirac_left_l_stiff','K_dirac_left_l_stiff2',
                                'K_dirac_left_l_stiff3'})

args = parser.parse_args()

### Modify the output directories to account for rigidity type
new_output_directory = os.path.join(args.output_directory,args.rigidity_type)
new_mode_data_directory = os.path.join(args.mode_data_directory,args.rigidity_type)
if not os.path.exists(new_output_directory):
    os.makedirs(new_output_directory)

## #Initialize the filament parameter class
filament_data = filament_parameters(rigid_func = args.rigidity_type, 
                                    N = 1001, L = 1, epsilon= 0.01, 
                                    mu_bar_low = 500,
                                    mu_bar_high=50001,mu_bar_range=500)

#Read in the CSV file that lists the mode number for each eigenvalue/eigenfunction
mode_data = pd.read_csv(os.path.join(new_mode_data_directory,'{}_mode_types.csv'.format(filament_data.rigidity_suffix)),index_col = 0,header = 0)
mode_data.set_index("Mu_bar",inplace = True,drop = False)

#Initialize dataframe that will store all data
all_eigen_data = pd.DataFrame()

for mu_bar in filament_data.mu_bar_vals:
    
    #Solve for eigenvalues & eigenfunctions of Linear Operator
    linop_diags = matr_diags(filament_data,mu_bar,'linear')
    linop_matr = sparse.diags(linop_diags, offsets = [-2,-1,0,1,2],
                          shape = (filament_data.N,filament_data.N)).toarray()
    sorted_linop_eigenval,sorted_linop_eigenf = calc_linear_eigen(filament_data,linop_matr,mu_bar)
    linop_eigenval_idx = find_useful_eigenvals(sorted_linop_eigenval)
    sorted_linop_eigenval,linop_eigenval_idx = remove_conjugate_eigenvalues(sorted_linop_eigenval,linop_eigenval_idx)
    
    #Solve for eigenvalues & eigenfunctions of  Adjoint Operator
    adjoint_diags = matr_diags(filament_data,mu_bar,'adjoint')
    adjoint_matr = sparse.diags(adjoint_diags, offsets = [-2,-1,0,1,2],
                          shape = (filament_data.N,filament_data.N)).toarray()
    sorted_adjoint_eigenval,sorted_adjoint_eigenf = calc_adjoint_eigen(filament_data,adjoint_matr,mu_bar)
    
    #Append the linear and adjoint data to the CSV files
    all_eigen_data = append_data(filament_data,linop_eigenval_idx,sorted_linop_eigenval,
                                  sorted_linop_eigenf,sorted_adjoint_eigenval,
                                  sorted_adjoint_eigenf,mu_bar,mode_data,all_eigen_data)
    
all_eigen_data = all_eigen_data.assign(Rigidity_Type = args.rigidity_type)
all_eigen_data.to_csv(os.path.join(new_output_directory,'N_{}_mu_bar_50_50k_linear_adjoint_eig_data_all.csv'.format(filament_data.N)))


#Interpolate for 101 points
adj_s = np.linspace(-filament_data.L/2,filament_data.L/2,101)
inter_eigen_data = all_eigen_data[all_eigen_data['s'].isin(adj_s)]
inter_eigen_data.reset_index(inplace = True,drop = True)
inter_eigen_data.to_csv(os.path.join(new_output_directory,'N_101_mu_bar_50_50k_linear_adjoint_eig_data_interp.csv'))
    
#%% Calculate Error between Linear and Adjoint
# =============================================================================
# 
# 
# #Calculate Eigenvalue differenes
# eignval_diff_real = np.abs(sorted_linop_eigenval.real) - np.abs(sorted_adjoint_eigenval.real)
# eignval_diff_real_per = (np.abs(sorted_linop_eigenval.real) - np.abs(sorted_adjoint_eigenval.real))/np.abs(sorted_linop_eigenval.real)*100
# eignval_diff_im = np.abs(sorted_linop_eigenval.imag) - np.abs(sorted_adjoint_eigenval.imag)
# eignval_diff_im_per = np.divide(np.abs(sorted_linop_eigenval.imag) - np.abs(sorted_adjoint_eigenval.imag),np.abs(sorted_linop_eigenval.imag),
#                                 out = np.zeros_like(np.abs(sorted_adjoint_eigenval.imag)),where=np.abs(sorted_linop_eigenval.imag)!= 0)
# 
# #Calculate Eigenfunction Orthogonality
# orth_matrix = np.zeros((N-4,N-4),dtype = float)
# 
# for i in range(0,N-4):
#     for j in range(0,N-4):
#         orth_matrix[i,j] = np.trapz(y = sorted_linop_eigenf[:,i].real*sorted_adjoint_eigenf[:,j].real + \
#                                     sorted_linop_eigenf[:,i].imag*sorted_adjoint_eigenf[:,j].imag,x = s)
# orth_matrix = pd.DataFrame(orth_matrix)
# =============================================================================
       

#%% Plot Adjoint Eigenfunctions-Individual

# plt.rc('text', usetex=True)
# for i in range(1,6):
#     plt.figure(figsize = (7,7))
#     plt.plot(s,sorted_adjoint_eigenf[:,-i],linewidth= 3,color = 'red')
#     ax = plt.gca()
#     ax.set_title(r"Filament Shapes: $K(s) = 1$" 
#                              "\n" r"$\bar{{\mu}}={0} \: | \: \sigma = {1}$".format(mu_bar,np.round(sorted_adjoint_eigenval[-i],3)),
#                          fontsize = 20,pad = 15 )
#     ax.set_xlabel("s",fontsize = 16)
#     ax.set_ylabel(r"$\hat{{h}}(s)$",fontsize = 16)
#     ax.set_ylim(-1.1,1.1)
#     ax.set_xlim(-0.55,0.55)
#     ax.set_xticks(np.linspace(-0.5,0.5,5))
#     ax.set_yticks(np.linspace(-1,1,5))
#     ax.tick_params(axis='both', which='major', labelsize=15)
#     filename3 = os.path.join(output_dir,'filament_shapes_K_constant_mubar_{}_eigval_{}.png'.format(mu_bar,np.round(sorted_adjoint_eigenval[-i],3)))
#     plt.savefig(filename3,dpi = 600,bbox_inches = 'tight')
#     plt.show()
#%% Plot Results
# =============================================================================
# 
# ### Differences between linear eigenvalues and adjoint eigenvalues- Raw Values
# 
# 
# # plt.rc('text', usetex=True)
# plt.figure(figsize = (11,11))
# ax = plt.gca()
# ax.bar(np.arange(0,N-4),eignval_diff_real,color = 'r',label = 'Real')
# ax.bar(np.arange(0,N-4) + 0.75,eignval_diff_im,color = 'b',label = 'Imaginary')
# ax.set_title("Raw Differences" "\n" r"$(\bar{{\mu}} = {0} \: | \: N = {1})$".format(mu_bar,N),fontsize = 30,pad = 15)
# plt.xlabel("Eigenvalue Index",labelpad = 15,fontsize = 25)
# plt.ylabel("Eigenvalue difference",labelpad = 15,fontsize = 25)
# plt.xticks(fontsize = 30)
# plt.yticks(fontsize = 30)
# # plt.ylim(-1.5,1.5)
# # ax.set_yticks(np.linspace(-1.5,1.5,5))
# # ax.set_yticks(np.linspace(-0.005,0.005,5))
# plt.legend(loc='upper right', 
#           prop={'size': 20})
# filename1 = os.path.join(output_dir,'Error between Eigenvalues_mubar_{}_N_{}_raw_values.png'.format(mu_bar,N))
# plt.savefig(filename1,dpi = 600,bbox_inches = 'tight')
# plt.show()
# 
# ### Differences between linear eigenvalues and adjoint eigenvalues- Pcercentage Values
# 
# 
# # plt.rc('text', usetex=True)
# plt.figure(figsize = (11,11))
# ax = plt.gca()
# ax.bar(np.arange(0,N-4),eignval_diff_real_per,color = 'r',label = 'Real')
# ax.bar(np.arange(0,N-4) + 0.75,eignval_diff_im_per,color = 'b',label = 'Imaginary')
# ax.set_title("Eigenvalue Differences" "\n" r"$(\bar{{\mu}} = {0} \: | \: N = {1})$".format(mu_bar,N),fontsize = 30,pad = 15)
# plt.xlabel("Eigenvalue Index",labelpad = 15,fontsize = 25)
# plt.ylabel("Percentage difference",labelpad = 15,fontsize = 25)
# plt.xticks(fontsize = 30)
# plt.yticks(fontsize = 30)
# # plt.ylim(-1.5,1.5)
# # ax.set_yticks(np.linspace(-1.5,1.5,5))
# # ax.set_yticks(np.linspace(-0.005,0.005,5))
# plt.legend(loc='upper right', 
#           prop={'size': 20})
# filename1 = os.path.join(output_dir,'Error between Eigenvalues_mubar_{}_N_{}_percent_values.png'.format(mu_bar,N))
# plt.savefig(filename1,dpi = 600,bbox_inches = 'tight')
# plt.show()
# 
# ### Heatmap of differences between calculated eigenvalue Orthogonality
# 
# orth_matrix_to_array = orth_matrix.to_numpy()
# max_value  = np.abs(orth_matrix_to_array).max()
# orth_matrix_to_array = orth_matrix_to_array/max_value
# norm_orth_matrix = pd.DataFrame(orth_matrix_to_array)
# 
# 
# plt.figure(figsize = (11,11))
# # htmap = sns.heatmap(norm_orth_matrix,cmap = 'coolwarm',
# #                     xticklabels = N-5 if ((N-4) % 2) == 1 else (N-4),
# #                     yticklabels = N-5 if ((N-4) % 2) == 1 else (N-4))
# 
# #Scaled based on max value of orthogonality rule
# htmap = sns.heatmap(norm_orth_matrix,cmap = 'coolwarm',
#                     xticklabels = False,
#                     yticklabels = False,
#                     vmin = -1,vmax = 1)
# 
# #Scaled to raw values of orthogonality rule
# # htmap = sns.heatmap(norm_orth_matrix,cmap = 'coolwarm',
# #                     xticklabels = False,
# #                     yticklabels = False)
# ax = plt.gca()
# ax.tick_params(left = False,bottom = False)
# ax.set_title("Linear and Adjoint Operator Eigenfunction Orthogonality Values",fontsize = 30,pad = 15)
# plt.xlabel("Adjoint Operator Eigenfunction Index Value",fontsize = 30,labelpad = 30)
# plt.ylabel("Linear Operator Eigenfunction Index Value",fontsize = 30,labelpad = 30)
# ax.set_title("Eigenfunction orthogonality" "\n" r"$(\bar{{\mu}} = {0}\: | \: N = {1})$".format(mu_bar,N),fontsize = 30,pad = 25)
# plt.xticks(fontsize = 30)
# plt.yticks(fontsize = 30)
# cbar = ax.collections[0].colorbar
# cbar.ax.tick_params(labelsize=30)
# filename2 = os.path.join(output_dir,'Orthogonality between Eigenvalues_mubar_{}_N_{}.png'.format(mu_bar,N))
# plt.savefig(filename2,dpi = 600,bbox_inches = 'tight')
# plt.show()
# 
# =============================================================================
