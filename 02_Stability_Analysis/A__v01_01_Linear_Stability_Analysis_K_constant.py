# -*- coding: utf-8 -*-

"""
FILE NAME:      06b1_Non_Brownian_class_movement_reverse.py

COMPLEMENTARY
SCRIPT(S)/
FILE(S):        N/A

DESCRIPTION:    Given an analytical form for the rigidity of the filament, this
                script will solve the stability analysis problem and will yield
                all eigenfunctions and corresponding eigenvalues. This mode will
                also classify each eigenfunction baseed on their mode number.

INPUT
FILES(S):       N/A

OUTPUT
FILES(S):       1) .CSV file that lists the mode number for each eigenvalue.
                2) .CSV file that lists 
                3) .NPY file that contains the numerical values of the rigidity
                of the function. 
                4) .NPY file that contains the length of the filament at each
                timestep. 

INPUT
ARGUMENT(S):    

1) Rigidity Type: The specific rigidity profile that will be 
used for the simulation:
    K_constant:                     K(s) = 1
    K_parabola_center_l_stiff:      K(s) = /frac{1}{2} + 2s^{2}
    K_parabola_center_m_stiff:      K(s) = /frac{3}{2} - 2s^{2}
    K_linear:                       K(s) = s + 1
    K_dirac_center_l_stiff:         K(s) = 1-\frac{1}{2}\exp^{-100s^{2}}
    K_dirac_center_l_stiff2:        K(s) = 1-\frac{1}{2}\exp^{-500s^{2}}
    K_dirac_center_m_stiff:         K(s) = 1+\exp^{-100s^{2}}
    K_parabola_shifted:             K(s) = \frac{3}{2} - \frac{1}{2}\left(s-\frac{1}{2}\right)^{2}
    K_error_function:               K(s) = 2+\erf(10s)
    K_dirac_left_l_stiff:           K(s) = 1-\frac{1}{2}\exp^{-100\left(s+frac{1}{4}\right)^{2}}
    K_dirac_left_l_stiff2:          K(s) = 1-\frac{1}{2}\exp^{-500\left(s+frac{1}{4}\right)^{2}}
    K_dirac_left_l_stiff3:          K(s) = 1-\frac{1}{2}\exp^{-1000\left(s+frac{1}{4}\right)^{2}}
2) Boundary Number: Number of points should be solved in this boundary system.
3) mu_bar_low: The lowest mu_bar value that will be solved in 
this code.
4) mu_bar_highest: The highest mu_bar value that will be solved in 
this code.
5) mu_bar_iteration: The mu_bar resolution value that will be solved in 
this code.
6) save_mode_class_data: Specify if you want to save the pandas 
dataframe file that lists the mode number for each eigenvalue.
7) save_mode_class_data: Specify if you want to read the CSV 
file that lists the mode number for each eigenvalue.
8) save_eig_data: Specify if you want to save the pandas dataframe
that contains all of the information regarding the stability problem.
9) class_data_dir: If you specified "Y" on argument #6 or #7, specify
the directory where the mode data is located.
10) eig_data_dir: If you specified "Y" on argument #8, specify
the directory where the mode data is located.

CREATED:        18Jun20

MODIFICATIONS
LOG:
    
18Jun20:        1) Implemented solving ODE via eigenvector/eigenvalue problem.
25Jun20:        2) Implemented boundary conditions into sparse matrix.
30Jun20:        3) Fixed boundary conditions in sparse matrix
11Jul20:        4) Automate obtaining eigenvalues & eigenvectors for range of values of mu_bar. 
12Oct21:        5) Created function to specify which eigenfunctions are which
                mode number.
24Feb22:        6) Allowed mode number to be manually edited via CSV file.
                Created class functionality and argument-based inputs. 
                Code specific for only K_constant.
    
            
LAST MODIFIED
BY:             Thomas Nguyen

PYTHON
VERSION USED
TO WRITE
SCRIPT:         3.8.11

VERSION:        1.1

AUTHOR(S):      Thomas Nguyen

STATUS:         Working

TO DO LIST:    

NOTE(S):        N/A


CHANGE LOG: 
"""

import numpy as np
import pandas as pd
import seaborn as sns
from scipy import sparse, special
from scipy.linalg import eig
import matplotlib.pyplot as plt
import os, sys, argparse, logging
# import matplotlib.animation as animation

 ### Parameters for Mode Number classification-Old Method ###
first_mode_check = False
second_mode_check = False
third_mode_check = False
other_mode_check = False


class stability_parameters():
    """
    This class will bundle up all constants and parameters needed stability 
    analysis for easier readability.
    """
    ### Coefficients for boundary conditions ###
    A = float(48/11)
    B = float(-52/11)
    C = float(15/11)
    D = float(28/11)
    E = float(-23/11)
    F = float(6/11)
    
    def __init__(self,rigid_func,slender_f,N_points):
        
        ### Traditional parameters ###
        self.rigidity_suffix = rigid_func
        self.N = N_points
        self.L = 1
        self.ds = self.L/(self.N-1)
        self.s = np.linspace(-1/2,1/2,self.N)
        self.c = np.log(slender_f**2*np.exp(1))
                
        ##### Determine Form of filament rigidity #####
        
        if self.rigidity_suffix == 'K_constant':
            self.K = np.ones(self.s.shape[0],dtype = float)
            self.Ks = np.zeros(self.s.shape[0],dtype = float)
            self.Kss = np.zeros(self.s.shape[0],dtype = float)
            self.rigidity_title = r"$\kappa(s) = 1$"
            
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
            self.rigidity_title = r"$\kappa(s) = s+1$"
            
        elif self.rigidity_suffix == 'K_dirac_center_l_stiff':
            self.K = 1-0.5*np.exp(-100*self.s**2)
            self.Ks = 100*self.s*np.exp(-100*self.s**2)
            self.Kss = np.exp(-100*self.s**2)*(100-2e4*self.s**2)
            self.rigidity_title = r"$\kappa(s) = 1- \frac{1}{2} e^{-100s^{2}}$"
            
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

def create_output_dir(output_directory):
    """
    This function will create a directory based on a path if it doesn't exist.
    Inputs:
        
    output_directory:       Directory path that needs to be created.
    """
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        
        
def calculate_growth_matrix_comp(stability_params,mu_bar):
    """
    This function will calculate all of the components in the sparse 
    coefficient matrix needed to solve the eigenvalue-eigenvector problem.
    
    Inputs:
        
    stability_params:       Class that contains all of the general parameters 
                        needed to solve for the problem.
    mu_bar:             Non-dimensionalized ratio of viscous drag forces to 
                        elastic forces.
    """
    c,ds,s = stability_params.c,stability_params.ds,stability_params.s
    K,Ks,Kss = stability_params.K, stability_params.Ks, stability_params.Kss

    zeta = ((K/(ds**4)) - (Ks/(ds**3)))[2:]
    
    epsilon = ((-4*K/(ds**4)) + ((2*Ks)/(ds**3)) + (
        Kss/(ds**2)) - (mu_bar*(0.25-(s**2))/(4*c*(ds**2))) - (mu_bar*s/(2*c*ds)))[1:]
    
    alpha = (6*K/(ds**4)) - (2*Kss/(ds**2))+(mu_bar*(0.25-(s**2))/(2*c*(ds**2))) + (mu_bar/c)  
    
    beta = ((-4*K/(ds**4)) - ((2*Ks)/(ds**3)) + (
        Kss/(ds**2)) - (mu_bar*(0.25-(s**2))/(4*c*(ds**2))) + (mu_bar*s/(2*c*ds)))[:-1]
    
    gamma = ((K/(ds**4)) + (Ks/(ds**3)))[:-2]
    return [zeta, epsilon, alpha, beta,gamma]


def solve_growth_eigen(stability_params,mu_bar,matrix):
    """
    This function will perform the necessary eigenvalue/eigenvector processing 
    such as sort the eigenvalues in increasing order, solve for the boundary 
    conditions, and normalize the amplitude to a magnitude of 1. 
    
    Inputs:
        
    stability_params:       Class that contains all of the general parameters 
                            needed to solve for the problem.
    mu_bar:                 Non-dimensionalized ratio of viscous drag forces to 
                            elastic forces.
    matrix:                 The NxN sparse matrix containing the coefficients 
                            for the eigenvalue problem.
    """
    c = stability_params.c
    A, B, C = stability_params.A, stability_params.B, stability_params.C
    D, E, F =stability_params.D, stability_params.E, stability_params.F
    
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

    #Adjust matrix for boundary conditions
    matrix = matrix*(c/mu_bar)
    matrix = matrix[2:-2,2:-2]
    
    #Solve for right-hand side eigenvalues & eigenvectors
    eigenvalues, eigenvectors = eig(matrix)
    
    #Sort eigenvalues and eigenvectors in increasing order
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigvecs_s= eigenvectors[:,idx]

    #Solve for h1, h2, h_n, h_n-1
    eigvecs_t = eigvecs_s.T
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

    #Normalize amplitude to (max amplitude is either 1 or -1)
    eigvecs_all = np.column_stack((h1,h2,eigvecs_t,n_1,n_0))
    for row_n in range(0,eigvecs_all.shape[0]):
        eigvecs_all[row_n,:] = eigvecs_all[row_n,:]/np.absolute(eigvecs_all[row_n,:]).max()
    eigvecs_all = eigvecs_all.T
    return eigenvalues, eigvecs_all


def find_useful_eigenvals(eigenvalues):
    """
    This function filters out for eigenvalues above a certain threshold and 
    aren't trivial solutions. Additionally, it obtains the index values for 
    these eigenvalues of interest.
    
    
    Inputs:
    
    eigenvalues:            array of eigenvalues corresponding to a 
                            specific mu_bar value.
    """
    eigenvalues = np.round(eigenvalues,4)
    
    useful_eigenvals = eigenvalues[
        (eigenvalues.real > -6) & (eigenvalues.real != 1) & \
            (eigenvalues.real != 2)]
        
    useful_eigenvals_idx = np.nonzero(
        (eigenvalues.real > -6) & (eigenvalues != 1) & (eigenvalues.real != 2))
    return useful_eigenvals, useful_eigenvals_idx


def remove_conjugate_eigenvalues(eigenvals_all,eigenvals_idx):
    """
    This function looks for conjugate eigenvalues based on the real and 
    imaginary components and removes the last one when sorted. 
    
    Inputs:
    
    eigenvals_all:          Array of all eigenvalues corresponding to a 
                            specific mu_bar value.
    eigenvals_idx:          Array of all eigenvalue rankings in ascending order
                            corresponding to a specific mu_bar value.
    """
    eigenval_dup_df = pd.DataFrame(index = range(eigenvals_all.shape[0]),
                                   columns = ['Eigenvalue_real','Eigenvalue_all',
                                              'Rank'])
    eigenval_dup_df['Eigenvalue_real'] = eigenvals_all.real
    eigenval_dup_df['Eigenvalue_all'] = eigenvals_all
    eigenval_dup_df['Rank'] = eigenvals_idx[0]
    eigenval_dup_df = eigenval_dup_df.drop_duplicates(
        subset = ['Eigenvalue_real'],keep = 'first')
    return np.array(
        eigenval_dup_df['Eigenvalue_real']),np.array(eigenval_dup_df['Rank'])


def check_odd_even_function(eigenvector):
    """
    This function checks to see whether the eigenfunction is an even or odd
    function.
    
    Inputs:
    
    eigenvector:            Eigenfunction of interest.
    """
    if np.round(eigenvector[0],4) == np.round(eigenvector[-1],4):
        func_type = "Even"
    else:
        func_type = "Odd"
    return func_type


def check_mode_number(mu_bar,all_funcs,eigenval_all,idx):
    """
    This function classifies each eigenfunction based on the ranking of the 
    relative ranking of eigenvalues.
    
    Inputs:
    
    mu_bar:                 Non-dimensionalized ratio of viscous drag forces to 
                            elastic forces.
    all_funcs:              List that contains whether each eigenfunction
                            corresponding to each eigenvalue is an odd or even
                            function.
    eigenvals_idx:          Array of all eigenvalue rankings in ascending order
                            corresponding to a specific mu_bar value.
    eigenval_all:           Array of all eigenvalues corresponding to a 
                            specific mu_bar value.
    """
    global first_mode_check,second_mode_check,third_mode_check,other_mode_check
    
    
    even_ids = np.array([i for i in range(len(all_funcs)) if all_funcs[i] == 'Even'])
    odd_ids = np.array([i for i in range(len(all_funcs)) if all_funcs[i] == 'Odd'])
    
    mode_num = None
    if not first_mode_check and not second_mode_check and not third_mode_check and not other_mode_check:
        if eigenval_all.shape[0] == 1 and all_funcs[idx] == 'Even':
            mode_num = "1"
            first_mode_check = True
    elif first_mode_check and not second_mode_check and not third_mode_check and not other_mode_check:
        if eigenval_all.shape[0] == 1 and all_funcs[idx] == 'Even':
            mode_num = "1"
        elif eigenval_all.shape[0] == 2 and all_funcs[idx] == 'Odd':
            mode_num = "2"
            second_mode_check = True
    elif first_mode_check and second_mode_check and not third_mode_check and not other_mode_check:
        if eigenval_all.shape[0] == 2:
            if all_funcs[idx] == 'Even':
                mode_num = "1"
            elif all_funcs[idx] == 'Odd':
                mode_num = "2"
        elif eigenval_all.shape[0] == 3:
            if all_funcs[idx] == 'Even' and idx == 0:
                mode_num = "3"
                third_mode_check = True
            elif all_funcs[idx] == 'Even' and idx == 2:
                mode_num = "1"
    elif first_mode_check and second_mode_check and third_mode_check and not other_mode_check:
        if eigenval_all.shape[0] == 3:
            if all_funcs[idx] == 'Even' and idx == 0:
                mode_num = "3"
            elif all_funcs[idx] == 'Odd':
                mode_num = "2"
            elif all_funcs[idx] == 'Even' and idx != 0:
                mode_num = "1"
        elif eigenval_all.shape[0] == 4:
            if all_funcs[idx] == 'Odd' and idx == 0:
                mode_num = "Other"
                other_mode_check = True
            elif all_funcs[idx] == 'Odd' and idx != 0:
                mode_num = "2"
            elif all_funcs[idx] == 'Even' and even_ids.shape[0] == 2:
                if idx > even_ids[even_ids != idx]:
                    mode_num = "1"
                elif idx < even_ids[even_ids != idx]:
                    mode_num = "3"
               
    elif first_mode_check and second_mode_check and third_mode_check and other_mode_check:
        if eigenval_all.shape[0] == 4 and mu_bar <= 25000:
            if all_funcs[idx] == 'Odd' and idx == 0:
                mode_num = "Other"
            elif all_funcs[idx] == 'Odd' and idx != 0:
                mode_num = "2"
            elif all_funcs[idx] == 'Even' and even_ids.shape[0] == 2:
                if idx > even_ids[even_ids != idx]:
                    mode_num = "1"
                elif idx < even_ids[even_ids != idx]:
                    mode_num = "3"
        elif eigenval_all.shape[0] == 3 and mu_bar <= 37000:
            if idx == 0:
                mode_num = "Other"
            elif all_funcs[idx] == 'Odd' and idx != 0:
                mode_num = "2"
            elif all_funcs[idx] == 'Even':
                mode_num = "3"
        elif eigenval_all.shape[0] == 3 and mu_bar > 37000:
            if idx == 0:
                mode_num = "Other"
            elif all_funcs[idx] == 'Odd' and idx != 0:
                mode_num = "Other"
            elif all_funcs[idx] == 'Even' and idx != 0:
                mode_num = "3"
            else:
                mode_num = "Other"
        elif eigenval_all.shape[0] == 4 and mu_bar > 37000 and mu_bar < 42500:
            if idx == 0:
                mode_num = 'Other'
            elif all_funcs[idx] == 'Odd' and idx != 0 and idx > odd_ids[odd_ids != idx].any():
                mode_num = 'Other'
            elif all_funcs[idx] == 'Even' and idx != 0 and idx > even_ids[even_ids != idx].any():
                mode_num = '3'
            else:
                mode_num = 'Other'
        elif eigenval_all.shape[0] == 4 and mu_bar >= 35000 and mu_bar <= 37000:
            if idx == 0:
                mode_num = 'Other'
            elif all_funcs[idx] == 'Odd' and idx != 0 and idx > odd_ids[odd_ids != idx].any():
                mode_num = '2'
            elif all_funcs[idx] == 'Even' and idx != 0 and idx > even_ids[even_ids != idx].any():
                mode_num = '3'
            else:
                mode_num = 'Other'
        elif eigenval_all.shape[0] == 4 and mu_bar >= 42500:
            if idx == 0:
                mode_num = 'Other'
            elif all_funcs[idx] == 'Odd' and idx != 0 and idx > odd_ids[odd_ids != idx].any():
                mode_num = 'Other'
            elif all_funcs[idx] == 'Even' and idx != 0 and idx > even_ids[even_ids != idx].any():
                mode_num = '3'
            elif all_funcs[idx] == 'Even' and idx != 0 and (idx == 1 or idx == 2):
                mode_num = "3"
            else:
                mode_num = 'Other'
        
                 
    if not mode_num:
        print("Error for Mu_bar = {}, eigenvalue index = {}".format(mu_bar,idx))
        sys.exit(1)
    return mode_num


def append_data(stability_params,mu_bar,save_command,eigenvalues, eigenvalues_idx,eigenvectors,all_data):
    """
    This function filters out for eigenvalues above a certain threshold and aren't 
    trivial solutions. Additionally, it obtains the index values for these eigenvalues.
    
    
    Inputs:
    stability_params:       Class that contains all of the general parameters 
                            needed to solve for the problem.  
    mu_bar:                 Non-dimensionalized ratio of viscous drag forces to 
                            elastic forces.
    save_command:           "Y" or "N" command to use the old method to list
                            mode number for each eigenvalue. 
    eigenvalues:            Array of eigenvalues corresponding to a specific 
                            mu_bar value.
    eigenvals_idx:          Array of all eigenvalue rankings in ascending order
                            corresponding to a specific mu_bar value.
    eigenvectors:           2-D array of eigenvector values corresponding
                            to a specific mu_bar value.
    all_data:               Pandas dataframe that contains all information 
                            corresponding to the eigenvalues and eigenfunctions
                            of the stability problem.
    """
    s = stability_params.s
    all_func_types = [check_odd_even_function(eigenvectors[:,v]) for i,v in enumerate(eigenvalues_idx)]
    for true_idx,eig_idx in enumerate(eigenvalues_idx):
        mu_bar_df = pd.DataFrame(index = range(0,len(s)),
                                 columns = ['Mu_bar','s','Eigenvalues_real',
                                            'Eigenfunctions_real',
                                            'Eigenvalues_im',
                                            'Eigenfunctions_im','Mode Number'])
        mu_bar_df['Mu_bar'] = mu_bar
        mu_bar_df['s'] = s
        mu_bar_df['Eigenvalues_real'] = eigenvalues[true_idx].real
        mu_bar_df['Eigenfunctions_real'] = eigenvectors[:,eig_idx].real
        mu_bar_df['Eigenvalues_im'] = eigenvalues[true_idx].imag
        mu_bar_df['Eigenfunctions_im'] = eigenvectors[:,eig_idx].imag
        if save_command == "Y":
            mode_num = check_mode_number(mu_bar,all_func_types,eigenvalues.real,true_idx) #Old Method of classifying mode number
        else:
            mode_num = 0
        mu_bar_df['Mode Number'] = mode_num
        all_data = pd.concat([all_data,mu_bar_df],ignore_index = True)
    return all_data

#%% Plotting routines 

def plot_gen_eigenspectrum_map(all_eigen_data,rigidity_type,output_dir):
    """
    This function plots the general eigenspectrum map that shows the growth rates
    for each mu_bar value. 
    
    Inputs:
    
    all_eigen_data:         Pandas dataframe that contains all information 
                            corresponding to the eigenvalues and eigenfunctions
                            of the stability problem.
    rigidity_type:          String parameter to specify what rigidity profile
                            is used for this analysis.
    output_dir:             Output directory where the resulting image will be 
                            saved to.
    """
    eigval_data = all_eigen_data.copy()
    eigval_data = eigval_data.filter(items = ['Mu_bar','Eigenvalues_real'])
    eigval_data = eigval_data.drop_duplicates(subset = ['Mu_bar','Eigenvalues_real'],keep = 'last')
    plt.figure(figsize = (12,12))
    plt.ticklabel_format(axis="x", style="sci", scilimits=(4,4))
    sns.scatterplot(x = "Mu_bar",y = "Eigenvalues_real",data = eigval_data,color = 'darkblue')
    ax = plt.gca()
    ax.grid(b=True, which='major',color='black', linewidth=0.3)
    ax.set_title(r"Filament Stability Growth Rate Analysis: {0}".format(rigidity_type),fontsize = 30,pad = 20)
    ax.set_xlabel(r"$\bar{\mu}$",fontsize = 25,labelpad = 15)
    ax.set_ylabel(r"Real component of $\sigma$",fontsize = 25,labelpad = 15)
    ax.set_xlim(0,50001)
    ax.set_ylim(-6,10)
    ax.set_xticks(np.linspace(0,50000,6))
    ax.set_yticks(np.linspace(-6,10,9))
    ax.tick_params(axis='both', which='major', labelsize=35,size = 5,width = 5)
    ax.xaxis.offsetText.set_fontsize(25)
    # filename1 = os.path.join(output_dir,'{}_eigenvalue_gen_behavior_50_100k.png'.format(rigidity_type))
    # plt.savefig(filename1,dpi = 200,bbox_inches = 'tight')
    plt.show()


def plot_mode_num_eigenspectrum_map(all_eigen_data,rigidity_type,output_dir):
    """
    This function plots the eigenspectrum map color-coded by mode number 
    that shows the growth rates for each mu_bar value. 
    
    Inputs:
    
    all_eigen_data:         Pandas dataframe that contains all information 
                            corresponding to the eigenvalues and eigenfunctions
                            of the stability problem.
    rigidity_type:          String parameter to specify what rigidity profile
                            is used for this analysis.
    output_dir:             Output directory where the resulting image will be 
                            saved to.
    """
    eigval_data = all_eigen_data.copy()
    eigval_data = eigval_data.filter(items = ['Mu_bar','Eigenvalues_real','Mode Number'])
    eigval_data = eigval_data.drop_duplicates(subset = ['Mu_bar','Eigenvalues_real'],keep = 'first')
    plt.figure(figsize = (12,12))
    plt.ticklabel_format(axis="x", style="sci", scilimits=(4,4))
    sns.scatterplot(x = "Mu_bar",y = "Eigenvalues_real",data = eigval_data,hue = 'Mode Number',s = 100,palette = 'bright')
    ax = plt.gca()
    ax.grid(b=True, which='major',color='black', linewidth=0.3)
    ax.set_title(r"Filament Stability Growth Rate Analysis: {0}".format(rigidity_type),fontsize = 30,pad = 30)
    ax.set_xlabel(r"$\bar{\mu}$",fontsize = 25,labelpad = 15)
    ax.set_ylabel(r"Real component of $\sigma$",fontsize = 25,labelpad = 15)
    ax.set_xlim(0,50001)
    ax.set_ylim(-6,10)
    ax.set_xticks(np.linspace(0,50000,6))
    ax.set_yticks(np.linspace(-6,10,9))
    ax.tick_params(axis='both', which='major', labelsize=35,size = 5,width = 5)
    ax.xaxis.offsetText.set_fontsize(25)
    ax.set_aspect(50001/16)
    plt.legend(loc='center left', 
                    bbox_to_anchor=(1, 0.5),prop={'size': 20},title= "Mode Number")
    # filename1 = os.path.join(output_dir,'{}_eigenvalue_mode_num_behavior_50_50k.png'.format(rigidity_type))
    # plt.savefig(filename1,dpi = 200,bbox_inches = 'tight')
    plt.show()
    
def plot_eigenfunctions_superimposed(stability_params,all_eigen_data,mu_bar,rigidity_type,output_dir):
    """
    This function plots all of the eigenfunctions for specified mu_bar on the 
    same plot.
    
    Inputs:
    
    stability_params:       Class that contains all of the general parameters 
                            needed to solve for the problem.
    all_eigen_data:         Pandas dataframe that contains all information 
                            corresponding to the eigenvalues and eigenfunctions
                            of the stability problem.
    mu_bar:                 Non-dimensionalized ratio of viscous drag forces to 
                            elastic forces.
    rigidity_type:          String parameter to specify what rigidity profile
                            is used for this analysis.
    output_dir:             Output directory where the resulting image will be 
                            saved to.
    """
    fil_eigen_data = all_eigen_data[(all_eigen_data['Mu_bar'] == mu_bar) & (all_eigen_data['Eigenvalues_real'] > -10)]
    # fil_eigen_data.sort_values(by = ['Mode Number'],ascending = True,inplace = True)
    plt.figure(figsize = (7,7))
    sns.lineplot(x = 's',y = 'Eigenfunctions_real',hue = 'Mode_Number_Eigenvalue',ci = None,
                              palette = 'bright',data = fil_eigen_data,linewidth = 3)
    ax = plt.gca()
    ax.set_title(r"Filament Shapes: {0}" 
                              "\n" r"$\bar{{\mu}}={1}$".format(stability_params.rigidity_title,mu_bar),
                          fontsize = 20)
    ax.set_xlabel("s",fontsize = 16)
    ax.set_ylabel(r"Normalized $\hat{{h}}(s)$",fontsize = 16)
    ax.set_ylim(-1.1,1.1)
    ax.set_xlim(-0.55,0.55)
    ax.set_xticks(np.linspace(-0.5,0.5,5))
    ax.set_yticks(np.linspace(-1,1,5))
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),prop={'size': 12})
    # filename2 = os.path.join(output_dir,'filament_shapes_{}_mubar_{}_all_eigval.png'.format(rigidity_type,mu_bar))
    # plt.savefig(filename2,dpi = 200,bbox_inches = 'tight')
    plt.show()
    
def plot_eigenfunctions_indiv(all_eigen_data,mu_bar,rigidity_type,output_dir):
    """
    This function plots all of the eigenfunctions for specified mu_bar on different
    plots.
    
    Inputs:
    
    stability_params:       Class that contains all of the general parameters 
                            needed to solve for the problem.
    all_eigen_data:         Pandas dataframe that contains all information 
                            corresponding to the eigenvalues and eigenfunctions
                            of the stability problem.
    mu_bar:                 Non-dimensionalized ratio of viscous drag forces to 
                            elastic forces.
    rigidity_type:          String parameter to specify what rigidity profile
                            is used for this analysis.
    output_dir:             Output directory where the resulting image will be 
                            saved to.
    """
    fil_eigen_data = all_eigen_data[(all_eigen_data['Mu_bar'] == mu_bar) & (all_eigen_data['Eigenvalues_real'] > 0)]
    fil_eigen_data['Eigenfunctions_real'] = -1*fil_eigen_data['Eigenfunctions_real']
    for i in fil_eigen_data['Eigenvalues_real'].unique():
        plt.figure(figsize = (7,7))
        fil_2_eigen_data = fil_eigen_data[fil_eigen_data['Eigenvalues_real'] == i]
        sns.lineplot(x = 's',y = 'Eigenfunctions_real',ci = None,
                                  color = 'red',data = fil_2_eigen_data,linewidth = 3)
        ax = plt.gca()
        ax.set_title(r"Filament Shapes: $K(s) = 1$" 
                                  "\n" r"$\bar{{\mu}}={0} \: | \: \sigma = ${1}".format(mu_bar,i),
                              fontsize = 20,pad = 15)
        ax.set_xlabel("s",fontsize = 25,labelpad = 15)
        ax.set_ylabel(r"$\hat{{h}}(s)$",fontsize = 25,labelpad  = 15)
        ax.set_ylim(-1.1,1.1)
        ax.set_xlim(-0.55,0.55)
        ax.set_xticks(np.linspace(-0.5,0.5,5))
        ax.set_yticks(np.linspace(-1,1,5))
        ax.tick_params(axis='both', which='major', labelsize=25)
        # filename3 = os.path.join(output_dir,'filament_shapes_{}_mubar_{}_eigval_{}.png'.format(rigidity_type,mu_bar,i))
        # plt.savefig(filename3,dpi = 600,bbox_inches = 'tight')
        plt.show()
#%% Argparse to set up input arguments 

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("rigidity_type",
                    help = "Specify what kind of rigidity profile this simulation will run on",
                    type = str,
                    choices = {"K_constant","K_parabola_center_l_stiff",'K_parabola_center_l_stiff',
                               'K_parabola_center_m_stiff','K_linear','K_dirac_center_l_stiff',
                               'K_dirac_center_l_stiff2','K_dirac_center_m_stiff','K_parabola_shifted',
                               'K_error_function','K_dirac_left_l_stiff','K_dirac_left_l_stiff2',
                               'K_dirac_left_l_stiff3'})
parser.add_argument("boundary_number",
                    help = "Specify how many points you want the boundary system to contain",
                    type = int)
parser.add_argument("mu_bar_low",
                    help = "Specify the lowest mu_bar value to solve the eigenfunction problem at",
                    type = int)
parser.add_argument("mu_bar_high",
                    help = "Specify the highest mu_bar value to solve the eigenfunction problem at",
                    type = int)
parser.add_argument("mu_bar_iteration",
                    help = "Specify the mu_bar resolution value",
                    type = int)
parser.add_argument("save_mode_class_data",
                    help = "Specify if you want to save the CSV file that lists the mode number for each eigenvalue",
                    type = str,choices = {"Y","N"})
parser.add_argument("read_mode_class_data",
                    help = "Specify if you want to read the CSV file that lists the mode number for each eigenvalue",
                    type = str,choices = {"Y","N"})
parser.add_argument("save_eig_data",
                    help = "Specify if you want to read the CSV file that lists the mode number for each eigenvalue",
                    type = str,choices = {"Y","N"})
parser.add_argument("--class_data_dir",'-cdd',
                    help = "Specify the directory where the CSV file that lists the mode number for each eigenvalue will reside in",
                    type = str,
                    default = None)
parser.add_argument("--eig_data_dir",'-edd',
                    help = "Specify the directory where the CSV file that lists contains all eigenfunction and eigenvalue information will reside in",
                    type = str,
                    default = None)
args = parser.parse_args()

#%% Calculate all Growth rates and corresponding eigenfunctions

logging.basicConfig(level=logging.INFO,
            format='%(asctime)s|%(filename)s|%(levelname)s|%(message)s',
            datefmt="%A, %B %d at %I:%M:%S %p")
logging.info(
    "Started stability analysis for a rigidity profile of {}".format(
        args.rigidity_type))
    
    
stability_params = stability_parameters(args.rigidity_type,0.01,args.boundary_number)
all_eigen_data = pd.DataFrame()
for mu_bar in range(args.mu_bar_low,int(args.mu_bar_high + 1e2),args.mu_bar_iteration):
    
    #1) Create Diagonal terms based on finite differences
    diags = calculate_growth_matrix_comp(stability_params,mu_bar)
    
    #1a) Create Matrix, Solve & Append to DataFrame
    big_m = sparse.diags(diags, offsets = [-2,-1,0,1,2],
                     shape = (stability_params.N,stability_params.N)).toarray()
    eigvals,eigvecs = solve_growth_eigen(stability_params,mu_bar,big_m)
    use_eigvals,use_eigvals_idx = find_useful_eigenvals(eigvals)
    use_eigvals,use_eigvals_idx = remove_conjugate_eigenvalues(use_eigvals,use_eigvals_idx)
    all_eigen_data = append_data(stability_params,mu_bar,args.save_mode_class_data,use_eigvals,use_eigvals_idx,eigvecs,all_eigen_data)

logging.info(
    "Code has finished calculating all eigenfunctions and eigenvalues starting at mu_bar = {}, ending at mu_bar = {} in increments of {} with {} discretized points".format(
        args.mu_bar_low,args.mu_bar_high,args.mu_bar_iteration,args.boundary_number))

### Keep data with mu_bar and eigenvalue numbers ###
if args.save_mode_class_data == 'Y':
    mu_bar_eigenvals_df = all_eigen_data.filter(
        items = ['Mu_bar','Eigenvalues_real','Mode Number']).drop_duplicates(
            subset = ['Mu_bar','Eigenvalues_real','Mode Number'],keep = 'first').reset_index(drop = True)
    mode_class_dir = os.path.join(args.class_data_dir,args.rigidity_type)
    create_output_dir(mode_class_dir)
    mu_bar_eigenvals_df.to_csv(os.path.join(mode_class_dir,'{}_mode_types.csv'.format(args.rigidity_type)))
    logging.info(
    "Code has finished saving the CSV file that lists the mode number corresponding to each eigenvalue.")

### Pre-read the CSV file that lists the mode number for each eigenvalue and 
### modify the mode number classification 
if args.save_mode_class_data == 'N' and args.read_mode_class_data == 'Y':
    mode_class_dir = os.path.join(args.class_data_dir,args.rigidity_type)
    mu_bar_eigenvals_df = pd.read_csv(os.path.join(mode_class_dir,'{}_mode_types.csv'.format(args.rigidity_type)),index_col = 0,header = 0,
                                      dtype={'Mu_bar': int,'Eigenvalues_real': float, 'Mode Number': str})
    logging.info(
    "Code has finished reading the CSV file that lists the mode number corresponding to each eigenvalue.")
    
    ### Classify each eigenvalue as mode number based on CSV classification ###
    for mu_bar in sorted(mu_bar_eigenvals_df['Mu_bar'].unique()):
        fil_mu_bar_eigenvals_df = mu_bar_eigenvals_df[mu_bar_eigenvals_df['Mu_bar'] == mu_bar]
        for eigval in fil_mu_bar_eigenvals_df['Eigenvalues_real'].unique():
            fil2_mu_bar_eigenvals_df = fil_mu_bar_eigenvals_df[fil_mu_bar_eigenvals_df['Eigenvalues_real'] == eigval]
            idx_key = all_eigen_data[(all_eigen_data['Mu_bar'] == mu_bar) & (all_eigen_data['Eigenvalues_real'] == eigval)].index.values
            all_eigen_data.loc[idx_key,'Mode Number'] = fil2_mu_bar_eigenvals_df['Mode Number'].unique()[0]
    
all_eigen_data['Mode_Number_Eigenvalue'] = all_eigen_data[['Mode Number','Eigenvalues_real']].apply(lambda row: '|'.join(row.values.astype(str)), axis=1)
all_eigen_data['Rigidity Type'] = stability_params.rigidity_suffix

### Save DataFrame that lists all relevant eigenvalues with corresponding eigenfunctions and mode numbers ###
if args.save_eig_data == 'Y':
    eig_data_dir = os.path.join(args.eig_data_dir,args.rigidity_type)
    create_output_dir(eig_data_dir)
    all_eigen_data.to_csv(os.path.join(eig_data_dir,'N_{}_stability_plot_data.csv'.format(stability_params.N)))
    logging.info(
    "Code has finished saving the CSV file that lists has all of the information needed to plot the eigenspectrum maps.")



### Plotting Routines ###
#%% General eigenspectrum map
# gen_map_location = os.path.join('./Stability_Maps/',args.rigidity_type)
# create_output_dir(gen_map_location)
# plot_gen_eigenspectrum_map(all_eigen_data,args.rigidity_type,gen_map_location)


#%% Eigenspectrum map color coded by mode number
mode_map_location = os.path.join('./Stability_Maps_v2/',args.rigidity_type)
# create_output_dir(mode_map_location)
plot_mode_num_eigenspectrum_map(all_eigen_data,args.rigidity_type,mode_map_location)

#%% Plot eigenfunctions at a specific mu_bar-superimposed

# eigenfunc_imp_dir = os.path.join('./eigenfunc_all/',args.rigidity_type)
# create_output_dir(eigenfunc_imp_dir)
# mu_bar_of_int = 18000
# for mu_bar_of_int in np.arange(15000,50500,1000):
#     plot_eigenfunctions_superimposed(stability_params,all_eigen_data,mu_bar_of_int,args.rigidity_type,eigenfunc_imp_dir)
#%% Plot eigenfunctions at a specific mu_bar-individual

# eigenfunc_imp_dir = os.path.join('./eigenfunc_indiv/',args.rigidity_type)
# # create_output_dir(eigenfunc_imp_dir)
# mu_bar_of_int = 12000
# plot_eigenfunctions_indiv(all_eigen_data,mu_bar_of_int,args.rigidity_type,eigenfunc_imp_dir)
    