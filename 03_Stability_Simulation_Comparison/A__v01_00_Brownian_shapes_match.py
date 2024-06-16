# -*- coding: utf-8 -*-
"""
FILE NAME:      A__v01_00_Brownian_shapes_match.py


COMPLEMENTARY
SCRIPT(S)/
FILE(S):        N/A


DESCRIPTION:    This script will read in the filament positional data from the non-linear simulations
                and calculate the amplitude of each mode number-eigenvalue based on the linear stability
                analysis results for each ensemble. This script will also average out the amplitude data 
                across all ensembles. 

INPUT
FILES(S):       

1)              .NPY files correponding to the filament positional data. 
2)              .CSV files corresponding to the parameters used for the non-linear simulations. 
3)              .CSV file that contains the eigenfunctions and normalized coefficients of the linear and adjoint
                operators.


OUTPUT
FILES(S):       



INPUT
ARGUMENT(S):    

1) --rigidity_type/-rt:                         The type of rigidity profile the simulation data used:

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

2) --process_ensemble_data/-ped:                Keyword argument to specify whether to read in all of the ensemble data 
                                                to calculate the amplitude data. 
3) --process_deflect_data/-pdd':                Keyword argument to specify whether to read in the filament deflection data 
                                                to calculate the amplitude data. 
4) --ensemble_count/-N:                         The number of ensembles used for each mu_bar value.
5) --lp_val/-lp:                                The persistence length value used for Brownian extensional simulations.
6) --input_simulation_directory/-isd:           The path to the parent directory where the simulation data (formatted by 
                                                rigidity profile, Mu_bar values, and potentially replicate values) that 
                                                contains the .npy and .csv files.
7) --linear_adjoint_data_directory/-ladd:       The path to the parent directory where the linear and adjoint data 
                                                (formatted by rigidity profile) that contains the .CSV files.
8) --output_directory/-od:                      The path to the parent directory where the resulting .CSV files 
                                                (formatted by rigidity profile) will be saved to.


CREATED:        21Oct21

MODIFICATIONS
LOG:

18Jun22:        Logging functionality.
19Jun22:        Fixed filtering for individual ensembles to include negative amplitude
                values as well.
            
LAST MODIFIED
BY:             Thomas Nguyen

PYTHON
VERSION USED
TO WRITE
SCRIPT:         3.8.11

VERSION:        1.0

AUTHOR(S):      Thomas Nguyen

STATUS:         Working

TO DO LIST:    More streamlined  workflow (less classes).

NOTE(S):        N/A

"""
import re, os, math, argparse, sys, time, logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from itertools import product


#%% Classes 

class sim_params():
    """
    This class will initialize the parameters for each ensemble by reading in the 
    parameter CSV file.
    
    Inputs:
    
    path_to_param_df:           Absolute path to the parameter CSV file to be read.    
    """
    
    def __init__(self,path_to_param_df):
        """
        This class funciton initializes the class and reads in the parameters from the 
        CSV file. 
        """
        self.params_df = pd.read_csv(path_to_param_df,index_col = 0,header = 0)
        self.iteration_count = int(self.params_df.loc['Number of iterations','Value'])
        self.adj_iteration_count = int(self.params_df.loc['Adjusted Number of iterations','Value'])
        self.dt_time = float(self.params_df.loc['dt','Value'])
        self.tot_run_time = self.dt_time*self.iteration_count
        self.adj_scaling = float(self.params_df.loc['Adjusted Scaling','Value'])
        self.time_scale = np.arange(0,self.tot_run_time,self.dt_time*self.adj_scaling)
        self.mu_bar = int(self.params_df.loc['Mu_bar','Value'])
        self.s = np.linspace(float(self.params_df.loc['Filament s start','Value']),float(self.params_df.loc['Filament s end','Value']),int(self.params_df.loc['N','Value']))
        
        
class amplt_coeff_data():
    """
    This class will initialize the a large Pandas dataframe that will have all of the 
    amplitude cofficients for a particular ensemble.
    
    Inputs:
    
    ensemble_amplt_coeff_df:        Pandas DataFrame that lists the amplitude cofficients for each mode number-eigenvalue pair.
    mode_num_eigenvals:             Each unique modenumber-eigenvalue pair. 
    mode_num_all:                   A sorted list of mode numbers calculated for this particular ensemble. 
    lin_eigvals:                    A sorted list of eigenvalues calculated for this particular ensemble. 
    mu_bar:                         The specific mu_bar value that this ensemble simulation was run at. 
    ensemble_num:                   The specific ensemble number for nomenclature purposes. 
    """
    def __init__(self,ensemble_amplt_coeff_df,mode_num_eigenvals,mode_num_all,lin_eigvals,mu_bar,ensemble_num):
        """
        This class funciton initializes the class and reads in the parameters from the 
        CSV file. 
        """
        self.amplt_coeff_df = ensemble_amplt_coeff_df
        self.amplt_coeff_df['Mode_Number_Eigenvalue'] = mode_num_eigenvals
        self.amplt_coeff_df['Mode Number'] = mode_num_all
        self.amplt_coeff_df['Linear_Eigenvals_real'] = lin_eigvals
        self.amplt_coeff_df['Mu_bar'] = mu_bar
        self.amplt_coeff_df['Ensemble Number'] = int(ensemble_num)
    
    def long_form_df(self,time_points):
        """
        This class function converts the dataframes to long form style where
        each ensemble amplitude cofficient for each mode number-eigenvalue pair
        is listed with its specific point in time. 
        
        Inputs:
        
        time_points:                Numpy array of all the timepoints in the ensemble simulation.
        """
        self.amplt_coeff_long_df = pd.melt(self.amplt_coeff_df,
                                           id_vars = ['Mode_Number_Eigenvalue',
                                                            'Mode Number',
                                                            'Linear_Eigenvals_real',
                                                            'Mu_bar','Ensemble Number'],value_vars = time_points,
                                                        var_name = 'Time',value_name = 'Amplitude_Coefficient')
        self.amplt_coeff_long_df.reset_index(drop = True,inplace = True)
        self.amplt_coeff_long_df.sort_values(by = ['Linear_Eigenvals_real','Time'],ascending = True,inplace = True)
        self.amplt_coeff_long_df['Adjusted Time'] = self.amplt_coeff_long_df['Mu_bar'].astype(int)*self.amplt_coeff_long_df['Time'].astype(float)
        self.amplt_coeff_long_df['Adjusted Time'] = self.amplt_coeff_long_df['Adjusted Time'].astype(float)
        self.amplt_coeff_long_df['Mode Number'] = self.amplt_coeff_long_df['Mode Number'].astype(str)
        self.amplt_coeff_long_df['Linear_Eigenvals_real'] = self.amplt_coeff_long_df['Linear_Eigenvals_real'].astype(float)


class ensemble_regression_data():
    """
    This class will iterate through every mu_bar value, mode number-eigenvalue 
    pair, and every ensemble and fit the filtered data to a linear regression model
    and extract the slope and intercept values from it.
    
    Inputs:
    
    regression_df:        Pandas DataFrame that lists the amplitude cofficients for each mode number-eigenvalue pair for each ensemble.
    """
    def __init__(self,regression_df):
        """
        This class function initializes the class and sets up the CSV file to contain
        the regression data statistics. 
        """
        self.reg_df = regression_df
        # self.reg_df.reset_index(drop = True,inplace = True)
        self.reg_df = self.reg_df.assign(
            MAGNTDE_AMPL_LN=np.log(self.reg_df['Amplitude_Coefficient_ABS'].to_numpy()),
            ln_time_indep_Amplitude = 0,Time_indep_Amplitude = 0,Actual_Growth_Rate = 0,
            Difference_between_Expected_and_Actual_Growth_Rate = 0,
            R_Squared_Value = 0,Predicted_Values = 0,Residuals = 0)

    def calc_regression(self):
        """
        This class function calculates the regression statistics for each ensemble, 
        mode number-eigenvalue, and mu_bar. 
        """
        
        ### Group by mu_bar value, mode number-eigenvalue, and ensemble number ###
        
        self.ensmbl_groups = self.reg_df.groupby(by = ['Mode_Number_Eigenvalue', 'Mode Number', 'Linear_Eigenvals_real',
       'Mu_bar','Ensemble Number'])
        for group in self.ensmbl_groups.groups.keys():
            self.group_df = self.ensmbl_groups.get_group(group)
            ### Check if enough data points after filtering ###
            if self.group_df.shape[0] > 10:
                self.linear_model = smf.ols('MAGNTDE_AMPL_LN ~ Q("Adjusted Time")',data = self.group_df).fit()
                if self.linear_model.params.shape[0]  == 2:
                    self.intercept,self.slope,self.rsquared = self.linear_model.params[0],self.linear_model.params[1],self.linear_model.rsquared
                    self.residuals = self.linear_model.resid
                    self.reg_df.loc[self.group_df.index,'ln_time_indep_Amplitude'] = self.intercept
                    self.reg_df.loc[self.group_df.index,'Time_indep_Amplitude'] = np.exp(self.intercept)
                    self.reg_df.loc[self.group_df.index,'Actual_Growth_Rate'] = self.slope
                    self.reg_df.loc[self.group_df.index,'Difference_between_Expected_and_Actual_Growth_Rate'] = float(self.group_df['Linear_Eigenvals_real'].unique()[0]) - self.slope 
                    self.reg_df.loc[self.group_df.index,'R_Squared_Value'] = self.rsquared
                    self.reg_df.loc[self.group_df.index,'Predicted_Values'] = self.linear_model.fittedvalues
                    self.reg_df.loc[self.group_df.index,'Residuals'] = self.residuals
                else:
                    print("Error! The Linear regression fitting did not work on mu_bar = {}, mode number|eigenvalue = {},ensemble # = {}".format(
                        self.group_df['Mu_bar'].unique()[0],
                        self.group_df['Mode_Number_Eigenvalue'].unique()[0],
                        self.group_df['Ensemble Number'].unique()[0]))
            else:
                self.reg_df.loc[self.group_df.index,'ln_time_indep_Amplitude'] = np.nan
                self.reg_df.loc[self.group_df.index,'Time_indep_Amplitude'] = np.nan
                self.reg_df.loc[self.group_df.index,'Actual_Growth_Rate'] = np.nan
                self.reg_df.loc[self.group_df.index,'Difference_between_Expected_and_Actual_Growth_Rate'] = np.nan
                self.reg_df.loc[self.group_df.index,'R_Squared_Value'] = np.nan
                self.reg_df.loc[self.group_df.index,'Predicted_Values'] = np.nan
                self.reg_df.loc[self.group_df.index,'Residuals'] = np.nan

        self.cut_reg_df = self.reg_df.copy().dropna(how = 'any')
        self.cut_reg_df = self.cut_reg_df[self.cut_reg_df['R_Squared_Value'] >= 0.60]

                    
class most_dom_mode():
    """
    This class will look through each ensemble for each mu_bar value and determine which ensemble
    has the highest slope value which corresponds to the actual growth rate of the filament shape.
    It then calculates the percentage of mode numbers that are excited for each mode number. 
    
    Inputs:
    
    reg_data:           Pandas DataFrame that contains the linear regression model statistics
                        for each ensemble.
    """
    def __init__(self,reg_data):
        """
        This function initializes the class and calculates which mode number has 
        the most excited by examining each ensemble's amplitude slope from linear regression.
        """
        self.reg_data_df = reg_data
        self.mu_bar_mode_vals = ['{}|{}'.format(int(float(i)),j) for i in sorted(self.reg_data_df['Mu_bar'].unique()) for j in ['1','2','3','Other']]
        self.most_dom_mode_df = pd.DataFrame(index = self.mu_bar_mode_vals,columns = ['Mu_bar','Mode Number','Frequency','Double Frequency','Single Percentage'])
        self.most_dom_mode_df.fillna(0,inplace = True)
        self.most_dom_mode_df['Mu_bar'] = [int(float(i.split('|')[0])) for i in self.mu_bar_mode_vals]
        self.most_dom_mode_df['Mode Number'] = [i.split('|')[1] for i in self.mu_bar_mode_vals]
        
        
        self.ensmbl_groups = self.reg_data_df.groupby(by = ['Mu_bar','Ensemble Number'])
        for group in self.ensmbl_groups.groups.keys():
            self.group_df = self.ensmbl_groups.get_group(group)
            
            #Select Data that has the maximum growth rate
            self.reg_data_mx_gr_df = self.group_df[self.group_df['Actual_Growth_Rate'] == self.group_df['Actual_Growth_Rate'].unique().max()]
            
            if self.reg_data_mx_gr_df['Mode Number'].unique().shape[0] == 1: #See if there's only one mode number with the expected growth rate
                self.most_dom_mode_gr = self.reg_data_mx_gr_df['Mode Number'].unique()[0]
                self.df_idx = '{}|{}'.format(int(float(self.group_df['Mu_bar'].unique()[0])),self.most_dom_mode_gr)
                self.most_dom_mode_df.loc[self.df_idx,'Frequency'] += 1
            else:
                print("Error! 2 modes have the same growth rate at mu_bar = {} and ensemble # = {}.".format(
                    self.group_df['Mu_bar'].unique()[0],
                    self.group_df['Ensemble Number'].unique()[0]))
                for mode_num in self.reg_data_mx_gr_df['Mode Number'].unique():
                    self.df_idx = '{}|{}'.format(
                        int(float( self.group_df['Mu_bar'].unique()[0])),
                        mode_num)
                    self.most_dom_mode_df.loc[self.df_idx,'Double Frequency'] += 1
                sys.exit(1)
                    
            ### Calculate percentage of ensembles that buckled based on most dominant mode ###
            self.most_dom_mode_df.loc[self.most_dom_mode_df[self.most_dom_mode_df['Mu_bar'] == \
                                                            self.group_df['Mu_bar'].unique()[0]].index,
                                      'Single Percentage'] = 100*\
                self.most_dom_mode_df.loc[self.most_dom_mode_df[
                    self.most_dom_mode_df['Mu_bar'] ==  self.group_df['Mu_bar'].unique()[0]].index,'Frequency']/\
                    self.most_dom_mode_df[self.most_dom_mode_df['Mu_bar'] ==  self.group_df['Mu_bar'].unique()[0]]['Frequency'].sum()
#%% Calculation routines 

def create_output_dir(output_directory):
    """
    This function creates an directory if it does not yet exist.
    
    Inputs:
        
    output_directory:      Path to directory to be created.
    """


    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

def calculate_deflection(filament_position):
    """
    This function calculates the max deflection (y-component) of the filament.
    
    Inputs:
        
    filament_position:      Nx3xT vectorized array that contains the filament 
                            position at each iteration.
    """
    fil_deflection = np.abs(filament_position[:,1,:]).max(axis = 0)
    return fil_deflection

def determine_end_to_end_distance(start_position,end_position):
    """
    This function calculates the end-to-end distance of the filament. 
    
    Inputs:
        
    start_position:         Nx3 array that contains the filament 
                            position at s = -0.5.
    end_position:           Nx3 array that contains the filament 
                            position at s = +0.5.
    """
    ee_length = np.sqrt(np.sum((end_position-start_position)**2))
    return ee_length
    
def calc_coefficients(s,filament_position,iteration_count,eigen_df):
    """
    This function will calculate the amplitude coefficients corresponding to the deflection
    of the filament based on the normalized coefficients and the adjoint eigenfunctions.
    
    Inputs:
        
    s:                      N array that parameterizes the filament position.
    filament_position:      Nx3xT vectorized array that contains the filament 
                            position at each iteration.
    iteraction_count:       Number of timepoints to calcualte the amplitude coefficients at. 
    eigen_df:               Pandas dataframe that contains the information regarding
                            linear eigenvalues-eigenfunctions, adjoint eigenvalues-eigenfunctions,
                            and the normalized coefficients corresponding to 
                            each pair. 
    """
    coeff_all = np.zeros((eigen_df['Linear_Eigenvals_real'].unique().shape[0],iteration_count))
    mode_num_eigval_key_all = []
    
    #Iterate through each eigenvalue
    for i,u_eigenval in enumerate(sorted(eigen_df['Linear_Eigenvals_real'].unique())):
        fil_eigen_df = eigen_df[eigen_df['Linear_Eigenvals_real'] == u_eigenval]
        if fil_eigen_df['Mode_Number_Eigenvalue'].unique().shape[0] == 1:
            mode_num_eigval_key_all.append(fil_eigen_df['Mode_Number_Eigenvalue'].unique()[0]) #Record which eigenvalue was calculated
        else:
            print("Error! There are more than 1 Mode Number-Eigenvalue key associated with with this Eigenvalue of {}. Please check the CSV file again!".format(u_eigenval))
                    
        ### Vectorized method to calculate amplitude coefficients  ###
        integrand = (filament_position[:,1,:].real*fil_eigen_df['Adjoint_Eigenfuncs_real'].to_numpy()[:,None]) +\
            (filament_position[:,1,:].imag*fil_eigen_df['Adjoint_Eigenfuncs_im'].to_numpy()[:,None])
        coeff = np.trapz(y = integrand,x = s,axis = 0)
        coeff_all[i,:] = coeff
        ### Check to make sure there's only one normalization cofficient associated with an eigenvalue ###
        if fil_eigen_df['Norm_Coefficient_real'].unique().shape[0] == 1:
            coeff_all[i,:] = coeff_all[i,:]/fil_eigen_df['Norm_Coefficient_real'].unique()[0]
        else:
            print("Error! There are more than 1 normalization coefficients associated with with this Eigenvalue of {}. Please check the CSV file again!".format(u_eigenval))
            sys.exit(1)

    return coeff_all,mode_num_eigval_key_all


def calculate_brownian_floor(L,lp,rms_data):
    """
    This function calculates the theoretical amplitude of the Brownian noise 
    floor.
    
    Inputs: 
    
    L:                      Length of filament.
    lp:                     Persistence length of filament.
    rms_data:               Pandas dataframe that contains the natural log RMS data
                            for each mu_bar value at each point during the simulation. 
    """
    mode_num_all = range(1,4)
    noise_floor_df = pd.DataFrame(index = range(0,len(mode_num_all)*len(rms_data['Mu_bar'].unique())),
                                  columns = ['Mode Number','Mu_bar','Noise_Floor_amplitude','Noise_Floor_amplitude_LN','Cutoff Adjusted Time'])
    rms_data['Mode Number'] = rms_data['Mode Number'].astype(str)

    counter = -1
    for mode_num in mode_num_all:
        for mu_bar_val in sorted(rms_data['Mu_bar'].unique()):
            counter += 1
            theor_amplt = np.sqrt((L**2)*((mode_num + 0.5)**4*np.pi**4)**-1*(L/lp))
            fil_rms_data = rms_data[(rms_data['Mu_bar'] == mu_bar_val) & 
                (rms_data['Amplitude_Coefficient_RMS_LN'] >= np.log(theor_amplt))]
            if str(mode_num) in fil_rms_data['Mode Number'].unique():
                fil_rms_data = fil_rms_data[fil_rms_data['Mode Number'] == str(mode_num)]
                ### Choose the latest time if more than 1 eigenvalue exists for a mode number
                if len(fil_rms_data['Linear_Eigenvals_real'].unique()) > 1:
                    all_adj_times = []
                    for lin_eigval_real in sorted(fil_rms_data['Linear_Eigenvals_real'].unique()):
                        fil2_rms_data = fil_rms_data[fil_rms_data['Linear_Eigenvals_real'] == lin_eigval_real]
                        all_adj_times.append(fil2_rms_data['Adjusted Time'].values[0])
                    ct_time = max(all_adj_times)
                else:
                    ct_time = fil_rms_data['Adjusted Time'].values[0]
            else:
                ct_time = np.nan
            noise_floor_df.loc[counter,'Mode Number'] = mode_num
            noise_floor_df.loc[counter,'Mu_bar'] = mu_bar_val
            noise_floor_df.loc[counter,'Noise_Floor_amplitude'] = theor_amplt
            noise_floor_df.loc[counter,'Noise_Floor_amplitude_LN'] = np.log(theor_amplt)
            noise_floor_df.loc[counter,'Cutoff Adjusted Time'] = ct_time
    noise_floor_df.dropna(subset = ['Cutoff Adjusted Time'],inplace = True)
    return noise_floor_df

def calculate_ensemble_average(all_ensemble_data_df):
    """
    This function will calculate the root mean squared (RMS) amplitude value of each
    ensemble. 
    
    Inputs:
        
    all_ensemble_data_df:   Pandas dataframe that contains the information regarding
                            amplitudes of each eigenvalue-eigenfunction for each ensemble.
    """
    all_coeff_sq_df = all_ensemble_data_df.copy()
    all_coeff_sq_df['Amplitude_Coefficient_SQ'] = all_coeff_sq_df['Amplitude_Coefficient']**2
    
    all_coeff_sq_mn_df = pd.DataFrame(all_coeff_sq_df.groupby(by = ['Mode_Number_Eigenvalue',
                                                    'Mode Number',
                                                    'Linear_Eigenvals_real',
                                                    'Mu_bar','Time','Adjusted Time'])['Amplitude_Coefficient_SQ'].mean())
    all_coeff_sq_mn_df['Amplitude_Coefficient_RMS'] = np.sqrt(all_coeff_sq_mn_df['Amplitude_Coefficient_SQ'])
    all_coeff_sq_mn_df['Amplitude_Coefficient_RMS_LN'] = np.log(all_coeff_sq_mn_df['Amplitude_Coefficient_RMS'])
    all_coeff_sq_mn_df = all_coeff_sq_mn_df.reset_index()
    all_coeff_sq_mn_df.sort_values(by = ['Mu_bar','Linear_Eigenvals_real','Time'],ascending = True,inplace = True)
    return all_coeff_sq_mn_df
    
def calc_ensemble_avg_regression(ensemble_df):
    """
    This function will fit the ensembled averaged amplitude data to a linear model and extract the statistics. 
    
    Inputs:
        
    ensemble_df:            Pandas dataframe that contains the information regarding
                            RMS amplitudes of each eigenvalue-eigenfunction for the ensemble averages. 
    """
    ensmbl_groups = ensemble_df.groupby(by = ['Mode_Number_Eigenvalue', 'Mode Number', 'Linear_Eigenvals_real',
       'Mu_bar'])
    all_unique_mod_eigvals_mubar_df_lst = []
    for group in ensmbl_groups.groups.keys():
        group_df = ensmbl_groups.get_group(group).copy()
        linear_model = smf.ols('Amplitude_Coefficient_RMS_LN ~ Q("Adjusted Time")',data = group_df).fit()
        intercept,slope,rsquared = linear_model.params[0],linear_model.params[1],linear_model.rsquared
        residuals = linear_model.resid
        group_df.loc[:,'ln_time_indep_Amplitude'] = intercept
        group_df.loc[:,'Time_indep_Amplitude'] = np.exp(intercept)
        group_df.loc[:,'Actual_Growth_Rate'] = slope
        group_df.loc[:,'Difference_between_Expected_and_Actual_Growth_Rate'] = slope - float(group_df['Linear_Eigenvals_real'].unique()[0])
        group_df.loc[:,'R_Squared_Value'] = rsquared
        group_df.loc[:,'Predicted_Values'] = linear_model.fittedvalues
        group_df.loc[:,'Residuals'] = residuals
        
        avg_ensmbl_dict_data = {col: group_df[col].values for col in group_df.columns.values}
        all_unique_mod_eigvals_mubar_df_lst.append(avg_ensmbl_dict_data)
    all_ensmbl_reg_data = pd.concat([pd.DataFrame.from_dict(i) for i in all_unique_mod_eigvals_mubar_df_lst],ignore_index = True)
    return all_ensmbl_reg_data


#%% ####### Plotting routines #######
        
def average_deflection(deflection_df,output_dir,rigidity_suffix):
    """
    This function plots the ensemble average deflection of the filament over the 
    entire course of the simulation.
    
    Inputs:
        
    deflection_df:          Pandas dataframe that lists the average deflection
                            value for each mu_bar value across the entire duraction
                            of the simulation.
    output_dir:             Output directory where the resulting plots will reside in.
    rigidity_suffix:        Type of rigidity profile used to plot the data.   
    """
    for mu_bar_val in sorted(deflection_df['Mu_bar'].unique()):
        fil_deflection_df = deflection_df[deflection_df['Mu_bar'] == mu_bar_val]
        # fil_deflection_df.dropna(subset = ['Second_Derivative_Deflection'],inplace = True)
        plt.figure(figsize = (14,14))
        # plt.ticklabel_format(axis="x", style="sci", scilimits=(4,4))
        sns.lineplot(data = fil_deflection_df,x = 'Adjusted Time',y = 'Deflection',
                              ci = None,palette = 'bright',
                              linewidth = 4,markersize = 25)
        
        
        plt.axis()
        ax = plt.gca()
        # ax.ticklabel_format(axis="x", style="sci", scilimits=(-4,-4))
        # ax.ticklabel_format(axis="y", style="sci", scilimits=(10,10))
        # ax.set_xticks(np.arange(1e4,5.1e4,1e4))
        # ax.set_yticks(np.linspace(0,1,11))
        ax.set_xlabel(r"Time",fontsize = 25,labelpad = 25)
        ax.set_ylabel(r"Deflection",fontsize = 25,labelpad = 25)
        ax.set_title(r"Deflection of Filament over Time" "\n" r"$(\bar{{\mu}} = {0}$)".format(int(mu_bar_val)),size = 25,pad = 25)
        ax.tick_params(axis='both', which='major', labelsize=25)
        ax.xaxis.offsetText.set_fontsize(25)
        plt.savefig(os.path.join(output_dir,'Average Deflection Values-{}_mu_bar_{}.png'.format(rigidity_suffix,mu_bar_val)),bbox_inches = 'tight',
                    dpi = 200)
        plt.show()
    
def plot_ln_RMS_coefficients_all(rms_data,output_dir,rigidity_suffix,filename_prefix,plot_type):
    """
    This function plots the natural log of the root mean squared (RMS) data of 
    the ensemble amplitude values over the course of the simulation based on the 
    mu_bar value.
    
    Inputs:
        
    rms_data:               Pandas dataframe that contains the natural log RMS data
                            for each mu_bar value at each point during the simulation. 
    output_dir:             Output directory where the resulting plots will reside in.
    rigidity_suffix:        Type of rigidity profile used to plot the data.  
    filename_prefix:        Specify the prefix for the filenames.
    plot_type:              Specify whether to plot the global data or individual data.
    """
    plt.figure(figsize = (10,6))
    for mu_bar in sorted(rms_data['Mu_bar'].unique()):
        if plot_type == 'global':
            fil_rms_data = rms_data[(rms_data['Mu_bar'] == mu_bar) & (rms_data['Time'] > sorted(rms_data['Time'].unique())[20])]
            sns.scatterplot(y = 'Amplitude_Coefficient_RMS_LN',x = 'Adjusted Time',hue = 'Mode_Number_Eigenvalue',
                                  palette = 'bright',data = fil_rms_data,s = 15,marker = '+')
        elif plot_type == 'linear':
            fil_rms_data = rms_data[(rms_data['Mu_bar'] == mu_bar)]
            sns.scatterplot(y = 'Amplitude_Coefficient_RMS_LN',x = 'Adjusted Time',hue = 'Mode_Number_Eigenvalue',
                                  palette = 'bright',data = fil_rms_data,s = 15,marker = '+')
        elif plot_type == 'regression':
            fil_rms_data = rms_data[(rms_data['Mu_bar'] == mu_bar) & (rms_data['Time'].isin(np.array(sorted(rms_data['Time'].unique()))[::5]))]
            sns.scatterplot(y = 'Amplitude_Coefficient_RMS_LN',x = 'Adjusted Time',hue = 'Mode_Number_Eigenvalue',
                                  palette = 'bright',data = fil_rms_data,s = 75,marker = 'd')
            sns.lineplot(y = 'Predicted_Values',x = 'Adjusted Time',hue = 'Mode_Number_Eigenvalue',
                                  palette = 'bright',data = fil_rms_data,linewidth = 2,legend = False)
            
            
            
        ax = plt.gca()
        plt.axis()
        ax.set_title(
            r"LN of RMS of Amplitude Coefficients-{0}" "\n" r"$(\bar{{\mu}} = {1}$)".format(
                plot_type,int(mu_bar)),fontsize = 30,pad = 25)
        ax.set_xlabel(r"Normalized Time [$\bar{\mu}t^{Br}$]",fontsize = 25,labelpad = 15)
        ax.set_ylabel(r"$\ln\left(\sqrt{\langle a^{2}\rangle}\right)$",fontsize = 25,labelpad = 15)
        if plot_type == 'global':
            # ax.set_ylim(-8,2)
            ax.set_xlim(0,5)
            ax.set_xticks(np.linspace(0,5,6))
            # ax.set_yticks(np.linspace(-8,2,6))
            ax.tick_params(axis='both', which='major', labelsize=35)
            ax.tick_params(axis = 'x')
            plt.xticks(fontsize = 35)
            plt.yticks(fontsize = 35)
            ax.xaxis.offsetText.set_fontsize(25)
            # ax.set_aspect(5/10)
        elif plot_type == 'linear' or plot_type == 'regression':
            ax.set_xlim(0.2,0.8)
            # ax.set_ylim(-8,0)
            ax.set_xticks(np.linspace(0.25,0.75,3))
            # ax.set_yticks(np.linspace(-8,0,5))
            ax.tick_params(axis='both', which='major', labelsize=35)
            ax.tick_params(axis = 'x')
            plt.xticks(fontsize = 35)
            plt.yticks(fontsize = 35)
            ax.xaxis.offsetText.set_fontsize(25)
            # ax.set_aspect(0.6/8)
        ax.legend(loc='center left', 
                bbox_to_anchor=(1, 0.5),prop={'size': 20},
                title= "Mode Number|Eigenvalue").get_title().set_fontsize("25")
        if plot_type == 'global':
            new_output_dir = os.path.join(output_dir,'global_data')
            create_output_dir(new_output_dir)
        if plot_type == 'linear':
            new_output_dir = os.path.join(output_dir,'linear_data')
            create_output_dir(new_output_dir)
        if plot_type == 'regression':
            new_output_dir = os.path.join(output_dir,'regression_data')
            create_output_dir(new_output_dir)
        filename1 = os.path.join(new_output_dir,'{}-mu_bar_{}.png'.format(filename_prefix,int(mu_bar)))
        plt.savefig(filename1,dpi = 200,bbox_inches = 'tight')
        plt.show()    
        

def plot_residual_data(rms_data,output_dir,rigidity_suffix,filename_prefix):
    """
    This function plots the residuals between the predicted growth rate from linear
    regression fits and the growth rate extracted from the simulations.
    
    Inputs:
        
    rms_data:               Pandas dataframe that contains the natural log RMS data
                            for each mu_bar value at each point during the simulation. 
    output_dir:             Output directory where the resulting plots will reside in.
    rigidity_suffix:        Type of rigidity profile used to plot the data.  
    filename_prefix:        Specify the prefix for the filenames.
    """

    plt.figure(figsize = (10,6))
    for mu_bar_of_int in sorted(rms_data['Mu_bar'].unique()):
        fil_rms_data = rms_data[(rms_data['Mu_bar'] == mu_bar_of_int)]
    
    
        sns.scatterplot(y = 'Residuals',x = 'Adjusted Time',hue = 'Mode_Number_Eigenvalue',
                              palette = 'bright',data = fil_rms_data,s = 15,marker = '+')
                
        ax = plt.gca()
        plt.axis()
        ax.set_title(
            r"Residuals of Linear Regression Model" "\n" r"$(\bar{{\mu}} = {0}$)".format(
                int(mu_bar_of_int)),fontsize = 30,pad = 25)
        ax.set_xlabel(r"Normalized Time [$\bar{\mu}t^{Br}$]",fontsize = 25,labelpad = 15)
        ax.set_ylabel(r"Residuals [$\ln\left(\sqrt{\langle a^{2}\rangle}\right)$]",fontsize = 25,labelpad = 15)

        ax.set_xlim(0.2,0.8)
        ax.set_xticks(np.linspace(0.25,0.75,3))
        # ax.set_ylim(-7.5,-2)
        # ax.set_yticks(np.linspace(-7,-2,6))
        ax.tick_params(axis='both', which='major', labelsize=35)
        ax.tick_params(axis = 'x')
        plt.xticks(fontsize = 35)
        plt.yticks(fontsize = 35)
        ax.xaxis.offsetText.set_fontsize(25)
        # ax.set_aspect(2.1/5.5)
        
        ax.legend(loc='center left', 
                bbox_to_anchor=(1, 0.5),prop={'size': 20},
                title= "Mode Number|Eigenvalue").get_title().set_fontsize("25")
        new_output_dir = os.path.join(output_dir,'residual_data')
        create_output_dir(new_output_dir)
        filename1 = os.path.join(new_output_dir,'{}-{}_mu_bar_{}.png'.format(filename_prefix,rigidity_suffix,int(mu_bar_of_int)))
        plt.savefig(filename1,dpi = 200,bbox_inches = 'tight')
        plt.show() 
        
def plot_amplitude_data(rms_data,output_dir,rigidity_suffix,filename_prefix):
    """
    This function plots the slope of the linear regression fits (time independent amplitude) as a function
    of the eigenvalue.
    
    Inputs:
        
    rms_data:               Pandas dataframe that contains the natural log RMS data
                            for each mu_bar value at each point during the simulation. 
    output_dir:             Output directory where the resulting plots will reside in.
    rigidity_suffix:        Type of rigidity profile used to plot the data.  
    filename_prefix:        Specify the prefix for the filenames.
    """
    plt.figure(figsize = (10,6))
    for mu_bar_of_int in sorted(rms_data['Mu_bar'].unique()):
        fil_rms_data = rms_data[(rms_data['Mu_bar'] == mu_bar_of_int)]
    
    
        sns.scatterplot(y = 'ln_time_indep_Amplitude',x = 'Linear_Eigenvals_real',hue = 'Mode Number',
                              palette = 'bright',data = fil_rms_data,s = 75,marker = 'o',hue_order = ['1','2','3','Other'])
                
        ax = plt.gca()
        plt.axis()
        ax.set_title(
            r"$\ln(A)$ vs. Eigenvalues" "\n" r"$(\bar{{\mu}} = {0}$)".format(
                int(mu_bar_of_int)),fontsize = 30,pad = 25)
        ax.set_xlabel(r"Eigenvalue",fontsize = 25,labelpad = 15)
        ax.set_ylabel(r"$\ln(A)$",fontsize = 25,labelpad = 15)

        ax.set_xlim(0,5)
        ax.set_xticks(np.linspace(0,5,6))
        ax.set_ylim(-8,-2)
        ax.set_yticks(np.linspace(-8,-2,4))
        ax.tick_params(axis='both', which='major', labelsize=35)
        ax.tick_params(axis = 'x')
        plt.xticks(fontsize = 35)
        plt.yticks(fontsize = 35)
        ax.xaxis.offsetText.set_fontsize(25)
        ax.set_aspect(5/6)
        
        ax.legend(loc='center left', 
                bbox_to_anchor=(1, 0.5),prop={'size': 20},
                title= "Mode Number").get_title().set_fontsize("25")
        new_output_dir = os.path.join(output_dir,'individual_amplitude_data')
        create_output_dir(new_output_dir)
        filename1 = os.path.join(new_output_dir,'{}-{}_mu_bar_{}.png'.format(filename_prefix,rigidity_suffix,int(mu_bar_of_int)))
        plt.savefig(filename1,dpi = 200,bbox_inches = 'tight')
        plt.show() 


def plot_eigenvalue_differences(rms_data,output_dir,rigidity_suffix,filename_prefix):
    """
    This function plots the differences between the growth rate predicted by stability analysis
    and the growth rate extracted from the simulations.
    
    Inputs:
        
    rms_data:               Pandas dataframe that contains the natural log RMS data
                            for each mu_bar value at each point during the simulation. 
    output_dir:             Output directory where the resulting plots will reside in.
    rigidity_suffix:        Type of rigidity profile used to plot the data.  
    filename_prefix:        Specify the prefix for the filenames.
    """
    ### Plot differences on same plot ###
    plt.figure(figsize = (15,9))
    sns.scatterplot(y = 'Difference_between_Expected_and_Actual_Growth_Rate',x = 'Linear_Eigenvals_real',hue = 'Mu_bar',
                          style = 'Mode Number',palette = 'bright',data = rms_data,s = 250)
         
    ax = plt.gca()
    plt.axis()
    ax.set_title(
        r"Eigenvalue Errors-{}".format(rigidity_suffix),fontsize = 30,pad = 25)
    ax.set_xlabel(r"$\sigma_{stability}$",fontsize = 25,labelpad = 15)
    ax.set_ylabel(r"$\sigma_{simul}$ - $\sigma_{stability}$",fontsize = 25,labelpad = 15)
    ax.hlines(y=0, xmin = -0.5,xmax = 5.5,linewidth=2, color='r')
    ax.set_xlim(-0.5,5.5)
    ax.set_xticks(np.linspace(0,5,6))
    ax.set_ylim(-4.1,3.1)
    ax.set_yticks(np.linspace(-3,2,6))
    ax.tick_params(axis='both', which='major', labelsize=35)
    ax.tick_params(axis = 'x')
    plt.xticks(fontsize = 35)
    plt.yticks(fontsize = 35)
    ax.xaxis.offsetText.set_fontsize(25)
    ax.set_aspect(5/6.2)
    
    ax.legend(loc='center left', 
            bbox_to_anchor=(1, 0.5),prop={'size': 20},
            title= "Mode Number").get_title().set_fontsize("25")
    new_output_dir = os.path.join(output_dir,'Differences')
    create_output_dir(new_output_dir)
    filename1 = os.path.join(new_output_dir,'{}_mu_bar_all.png'.format(filename_prefix))
    plt.savefig(filename1,dpi = 200,bbox_inches = 'tight')
    plt.show()
    
    ### Plot actual vs. theoretical eigenvalue
    plt.figure(figsize = (15,9))
    sns.scatterplot(y = 'Actual_Growth_Rate',x = 'Linear_Eigenvals_real',hue = 'Mu_bar',
                          style = 'Mode Number',palette = 'bright',data = rms_data,s = 250)
    xvals = np.linspace(-4,6,20)
    plt.plot(xvals,1*xvals,color = 'red',linewidth = 2,label = '0 error difference',linestyle = 'dashed')
            
    ax = plt.gca()
    plt.axis()
    ax.set_title(
        r"Eigenvalue Errors-{}".format(rigidity_suffix),fontsize = 30,pad = 25)
    ax.set_xlabel(r"$\sigma_{stability}$",fontsize = 25,labelpad = 15)
    ax.set_ylabel(r"$\sigma_{simulations}$",fontsize = 25,labelpad = 15)
    ax.set_xlim(-0.5,5.5)
    ax.set_xticks(np.linspace(0,5,6))
    ax.set_ylim(-0.5,5.5)
    ax.set_yticks(np.linspace(0,5,6))
    ax.tick_params(axis='both', which='major', labelsize=35)
    ax.tick_params(axis = 'x')
    plt.xticks(fontsize = 35)
    plt.yticks(fontsize = 35)
    ax.xaxis.offsetText.set_fontsize(25)
    ax.set_aspect(1)
    
    ax.legend(loc='center left', 
            bbox_to_anchor=(1, 0.5),prop={'size': 20},
            title= "Mode Number").get_title().set_fontsize("25")
    new_output_dir = os.path.join(output_dir,'differences')
    create_output_dir(new_output_dir)
    filename1 = os.path.join(new_output_dir,'{}_mu_bar_all_v2.png'.format(filename_prefix))
    plt.savefig(filename1,dpi = 200,bbox_inches = 'tight')
    plt.show()
    

    ### Plot differences based on mu_bar value ###
    # for mu_bar_of_int in sorted(rms_data['Mu_bar'].unique()):
        # plt.figure(figsize = (15,9))
        # fil_rms_data = rms_data[(rms_data['Mu_bar'] == mu_bar_of_int)]
    
    
        # sns.scatterplot(y = 'Difference_between_Expected_and_Actual_Growth_Rate',x = 'Linear_Eigenvals_real',hue = 'Mode Number',
        #                       palette = 'bright',data = fil_rms_data,s = 150)
        # # xvals = np.linspace(-4,6,20)
        # # plt.plot(xvals,1*xvals,color = 'black',linewidth = 2,label = '0 error difference',linestyle = 'dashed')
        
                
        # ax = plt.gca()
        # plt.axis()
        # ax.set_title(
        #     r"Error between Expected and Actual Growth Rate" "\n" r"$(\bar{{\mu}} = {0}$)".format(
        #         int(mu_bar_of_int)),fontsize = 30,pad = 25)
        # ax.set_xlabel(r"Expected Eigenvalue",fontsize = 25,labelpad = 15)
        # ax.set_ylabel(r"Eigenvalue Error",fontsize = 25,labelpad = 15)
        # ax.hlines(y=0, xmin = -0.5,xmax = 5.5,linewidth=2, color='r')
        # ax.set_xlim(-0.5,5.5)
        # ax.set_xticks(np.linspace(0,5,6))
        # ax.set_ylim(-3.1,3.1)
        # ax.set_yticks(np.linspace(-2,2,5))
        # ax.tick_params(axis='both', which='major', labelsize=35)
        # ax.tick_params(axis = 'x')
        # plt.xticks(fontsize = 35)
        # plt.yticks(fontsize = 35)
        # ax.xaxis.offsetText.set_fontsize(25)
        # ax.set_aspect(5/6.2)
        
        # ax.legend(loc='center left', 
        #         bbox_to_anchor=(1, 0.5),prop={'size': 20},
        #         title= "Mode Number").get_title().set_fontsize("25")
        # new_output_dir = os.path.join(output_dir,'eigenvalue_differences')
        # create_output_dir(new_output_dir)
        # filename1 = os.path.join(new_output_dir,'{}-{}_mu_bar_{}.png'.format(filename_prefix,rigidity_suffix,int(mu_bar_of_int)))
        # # plt.savefig(filename1,dpi = 200,bbox_inches = 'tight')
        # plt.show() 


def plot_dominant_modes(amplitude_data,output_dir,rigidity_suffix):
    """
    This function plots the percentage distribution of modes that are the most excited
    across all ensembles based on the slope of the linear fits (calculated growth rate).
    
    Inputs:
        
    amplitude_data:         Pandas dataframe that has the percentage of mode excitation
                            across all ensembles. 
    output_dir:             Output directory where the resulting plots will reside in.
    rigidity_suffix:        Type of rigidity profile used to plot the data.  
    filename_prefix:        Specify the prefix for the filenames.
    """
    mu_bar_vals = sorted(amplitude_data['Mu_bar'].unique())
    diff_buckling_modes = sorted(amplitude_data['Mode Number'].unique())
    all_coeff_vals = []
    for mode_num in diff_buckling_modes:
        lst_coeff_vals = []
        for mu_bar in mu_bar_vals:
            fil_amp_modes_df = amplitude_data[(amplitude_data['Mu_bar'] == mu_bar) & \
                                                          (amplitude_data['Mode Number'] == mode_num)]
            coeff_per = fil_amp_modes_df['Single Percentage'].values[0]
            lst_coeff_vals.append(coeff_per)
        all_coeff_vals.append(lst_coeff_vals)
        
    bright_palette = sns.color_palette("bright")
    # sns.set_theme()
    plt.figure(figsize = (10,10))
    plt.ticklabel_format(axis="x", style="sci", scilimits=(4,4))
    plt.stackplot(mu_bar_vals,all_coeff_vals, labels=diff_buckling_modes,colors = bright_palette)
    plt.axis()
    ax = plt.gca()
    ax.set_xticks(np.arange(0.5e4,5.1e4,0.5e4))
    ax.set_yticks(np.linspace(0,100,6))
    ax.set_xlim(0.5e4,5.0e4)
    ax.set_ylim(0,100)
    [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_ticklabels()) if (i-1) % 2 != 0] #Hide every other label
    ax.set_xlabel(r"$\bar{\mu} \: \left(\times 10^{4}\right)$",fontsize = 35,labelpad = 25)
    ax.set_ylabel(r"Percentage",fontsize = 35,labelpad = 25)
    ax.set_title("Percentage of Excited Ensembles by Mode Number",size = 35,pad = 25)
    ax.tick_params(axis='both', which='major', labelsize=35,size = 10,width = 5)
    ax.xaxis.offsetText.set_fontsize(0)
    plt.legend(loc='center left', 
                    bbox_to_anchor=(1, 0.5),prop={'size': 25},title= "Mode Number",title_fontsize = 25)
    # new_output_dir = os.path.join(output_dir,'amplitude_weight')
    # create_output_dir(new_output_dir)
    # plt.savefig(os.path.join(new_output_dir,'Most_Dominant_Mode Count_by_Amplitude-{}.png'.format(rigidity_suffix)),bbox_inches = 'tight',
    #             dpi = 200)
    plt.show()


#%% Argparse for arguments

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--rigidity_type","-rt",
                    help = "Specify what kind of rigidity profile this simulation will run on",
                    type = str,
                    choices = {"K_constant","K_parabola_center_l_stiff",'K_parabola_center_l_stiff',
                                'K_parabola_center_m_stiff','K_linear','K_dirac_center_l_stiff',
                                'K_dirac_center_l_stiff2','K_dirac_center_m_stiff','K_parabola_shifted',
                                'K_error_function','K_dirac_left_l_stiff','K_dirac_left_l_stiff2',
                                'K_dirac_left_l_stiff3'})
parser.add_argument("--process_ensemble_data","-ped",
                    help = 'Specify whether or not you want to process each ensemble data or not ("Yes" means read in each ensemble data; "No" means use the average data',
                    choices = {"Yes","No","Both"},default = "Y",required = True)
parser.add_argument("--process_deflect_data", '-pdd',
                    help = 'Specify whether or not you want to process the deflection data or not ("Yes" means read in each ensemble data and save the deflection data; "No" means skip this step',
                    choices = {"Yes","No","Both"},type = str,default = "N",required = True)
parser.add_argument("--lp_val",'-lp',
                    help = "Specify the persistence length of the Brownian extensional simulations",
                    type = int,
                    default = 100,required = False)
parser.add_argument("--input_simulation_directory","-isd",
                    help="Specify the location of the parent directory where the simulation data (formatted by rigidity profile, Mu_bar values, and potentially replicate values) that contains the .npy files",
                type = str,required = True)
parser.add_argument("--linear_adjoint_data_directory","-ladd", 
                    help="Specify the location of the parent directory where the linear and adjoint data (formatted by rigidity profile) that contains the CSV files",
                type = str,required = True)
parser.add_argument("--output_directory","-od",
                    help="Specify the parent directory where the resulting CSV files (will be formatted by rigidity profile) will be saved to",
                type = str,required = True)
args = parser.parse_args()

#%% Data Processing

logging.basicConfig(level=logging.INFO,
            format='%(asctime)s|%(filename)s|%(levelname)s|%(message)s',
            datefmt="%A, %B %d at %I:%M:%S %p")
logging.info(
    "Started stability analysis for a rigidity profile of {}".format(
        args.rigidity_type))


start_time = time.perf_counter()



input_simulation_dir = os.path.join(args.input_simulation_directory,args.rigidity_type) #Path too long for current file structure
input_simulation_dir = os.path.join(os.path.dirname(os.getcwd()),'01_Filament_Movement/03_Run_Results/Brownian_Extensional_lp_100_adj/',args.rigidity_type)
# linear_adjoint_data_dir = os.path.join(args.linear_adjoint_data_directory,args.rigidity_type) #Path name too long for current file structure
linear_adjoint_data_dir = os.path.join(os.path.dirname(os.getcwd()),'02_Stability_Analysis/03_Run_Results/Linear_Adjoint_Data_New/',args.rigidity_type)
output_directory = os.path.join(os.path.dirname(os.getcwd()),'03_Stability_Simulation_Comparison/03_Run_Results/Brownian_Extensional_lp_100_adj/',
                                args.rigidity_type)
# output_directory = os.path.join(args.output_directory,args.rigidity_type) #Path name too long for current file structure
create_output_dir(output_directory)

### Load Linear and Adjoint Eigenfunction Data & Process it ###
linear_adjoint_data_df = pd.read_csv(os.path.join(
    linear_adjoint_data_dir,'N_101_mu_bar_50_50k_linear_adjoint_eig_data_interp.csv'),index_col = 0,header = 0)
linear_adjoint_data_df['Linear_Eigenvals_real'] = np.round(linear_adjoint_data_df['Linear_Eigenvals_real'],4)
linear_adjoint_data_df['Mode_Number_Eigenvalue'] = linear_adjoint_data_df[['Mode Number','Linear_Eigenvals_real']].apply(lambda row: '|'.join(row.values.astype(str)), axis=1)
linear_adjoint_data_df.sort_values(by = ['Mu_bar','Linear_Eigenvals_real','s'],ascending = True,inplace = True)

logging.info("Code has finished reading in the Linear and Adjoint Eigenvalue-Eigenfunction data.")

### Create new Dataframe for each unique positive growth rate for each mu_bar ###
linear_adjoint_data_df = linear_adjoint_data_df[linear_adjoint_data_df['Linear_Eigenvals_real'] > -4]

### Read through all directories and process data ###
if args.process_ensemble_data == "Yes" or args.process_ensemble_data == "Both":
    deflect_list = []
    amplt_coeff_list = []
    for root,dirs,files in os.walk(input_simulation_dir):
        for dir_ in dirs:
            check_file = os.path.join(root,dir_,'filament_allstate.npy')
            if os.path.exists(check_file):
                if "rep" in dir_:
                    match = re.search(r"rep_(\d{1,})",dir_)
                    if match:
                        replicate_number = int(match.group(1))
                
                        #Load & Pre-process each ensemble data
                        esim_params = sim_params(os.path.join(root,dir_,'parameter_values.csv'))
                        fil_loc_data = np.load(os.path.join(root,dir_,'filament_allstate.npy'))
                        fil_linear_adjoint_data_df = linear_adjoint_data_df[linear_adjoint_data_df['Mu_bar'] == esim_params.mu_bar]
                        fil_linear_adjoint_data_df.index = fil_linear_adjoint_data_df['Linear_Eigenvals_real'].values
                        fil_deflect_values = calculate_deflection(fil_loc_data)
                        data_length = range(0,esim_params.time_scale.shape[0])
                        
                        ### Save the deflection data for all ensembles if necessary ###
                        if args.process_deflect_data == "Yes":
                            deflect_dict_data = {'Time': esim_params.time_scale,
                                'Mu_bar': [esim_params.mu_bar for i in data_length],
                                               'Replicate Number': [replicate_number for i in data_length],
                                               'Deflection': fil_deflect_values}
                            deflect_list.append(deflect_dict_data)
                                    
                        ###Calculate normalized amplitudes corresponding to each eigenvalue ###
                        ortho_coeff_to_plot,mode_num_eignvals_pair = calc_coefficients(esim_params.s, fil_loc_data, 
                                                                                       esim_params.adj_iteration_count, fil_linear_adjoint_data_df)
                        mode_num_key = [i.split('|')[0] for i in mode_num_eignvals_pair]
                        eigvals_key = [i.split('|')[1] for i in mode_num_eignvals_pair]
                        ensemble_coeff_array = pd.DataFrame(ortho_coeff_to_plot,index = mode_num_eignvals_pair,columns=esim_params.time_scale)
                        
                        # Convert coefficients to pandas dataframe #
                        ensemble_coeff_data = amplt_coeff_data(ensemble_coeff_array,
                                             mode_num_eignvals_pair,
                                             mode_num_key,eigvals_key,
                                             esim_params.mu_bar,replicate_number)
                        ensemble_coeff_data.long_form_df(esim_params.time_scale)
                        ensemble_ampl_coeff_dict = {"Mode_Number_Eigenvalue": ensemble_coeff_data.amplt_coeff_long_df['Mode_Number_Eigenvalue'],
                                                    "Mode Number": ensemble_coeff_data.amplt_coeff_long_df['Mode Number'],
                                                    "Linear_Eigenvals_real": ensemble_coeff_data.amplt_coeff_long_df['Linear_Eigenvals_real'],
                                                    "Mu_bar": ensemble_coeff_data.amplt_coeff_long_df['Mu_bar'],
                                                    "Ensemble Number": ensemble_coeff_data.amplt_coeff_long_df['Ensemble Number'],
                                                    "Time": ensemble_coeff_data.amplt_coeff_long_df['Time'],
                                                    "Adjusted Time": ensemble_coeff_data.amplt_coeff_long_df['Adjusted Time'],
                                                    "Amplitude_Coefficient": ensemble_coeff_data.amplt_coeff_long_df['Amplitude_Coefficient']}
                        amplt_coeff_list.append(ensemble_ampl_coeff_dict)
    
    # Append all amplitude coefficients from each ensemble #
    coeff_df_all = pd.concat([pd.DataFrame.from_dict(i) for i in amplt_coeff_list],ignore_index = True)
    ensemble_count = coeff_df_all['Ensemble Number'].unique().size
    coeff_df_all['Amplitude_Coefficient_ABS'] = np.abs(coeff_df_all['Amplitude_Coefficient'])     
    logging.info("Code has finished reading in all ensemble data. Number of unique ensembles dectected: {}".format(ensemble_count))
    
    ##### Calculate averages from all ensembles #####
    avg_ensemble_df = calculate_ensemble_average(coeff_df_all)
    logging.info("Code has finished calculating the ensemble average data.")

    ### Calculate noise floor of ensemble average data & determine limits on linear data ###
    noise_floor_amplitude_df = calculate_brownian_floor(1,args.lp_val,avg_ensemble_df)
    low_limit_adj_time = noise_floor_amplitude_df[noise_floor_amplitude_df['Mode Number'] == 1]['Cutoff Adjusted Time'].max()
    # low_limit_adj_time = 0
    upper_limit_adj_time = 0.75 #Upper limit on adjusted time (shear time) based on eyeballing linear data
    logging.info("Code has finished calculating the Brownian noise floor.")
    
    #####################################################################################################
    ###### Filter out individual ensemble data based on ensemble average data #####
    ### Calculate Slope and Intercept for each individual Ensemble ###
    fil_coeff_time_df_all = coeff_df_all[(coeff_df_all['Adjusted Time'] <= upper_limit_adj_time) & 
                                            (coeff_df_all['Linear_Eigenvals_real'] >= 0) & 
                                            (coeff_df_all['Amplitude_Coefficient_ABS'] >= \
                                              noise_floor_amplitude_df[noise_floor_amplitude_df['Mode Number'] == 1]['Noise_Floor_amplitude'].unique()[0])
                                                ].reset_index(drop = True)
    
    ensem_reg_df = ensemble_regression_data(fil_coeff_time_df_all)
    ensem_reg_df.calc_regression()
    logging.info("Code has finished fitting each ensemble to linear fits. The slope and intercept for these fits has been extracted.")
    ### Perform linear regression for each individual ensemble ### 
    ### Calculate number of ensembles with largest slope (growth rate) value ###
    ensemble_mode_count = most_dom_mode(ensem_reg_df.cut_reg_df)
    plot_dominant_modes(ensemble_mode_count.most_dom_mode_df,args.output_directory,args.rigidity_type)
    logging.info("Code has finished determining which mode is the most dominant for each ensemble. The stacked area plots have been generated.")
    #################################################################################################
    
    #################################################################################################
    ###### Filter out total ensemble averaged data based on ensemble average data #####
    ### Filter out ensemble averaged data based on Brownian noise floor, upper limit, and unstable growth rates ###
    fil_avg_ensemble_df = avg_ensemble_df[(avg_ensemble_df['Adjusted Time'] >= low_limit_adj_time) & (avg_ensemble_df['Adjusted Time'] <= upper_limit_adj_time) & 
                                            (avg_ensemble_df['Linear_Eigenvals_real'] >= 0)]
    
    ### Calculate regression data on ensemble average data ###
    ensmbl_avg_reg_data = calc_ensemble_avg_regression(fil_avg_ensemble_df)
    logging.info("Code has finished fitting the average ensemble data to linear regression fits. Now plotting the theoretical and actual growth rates.")
    
    ### Plot theoretical and actual eigenvalue differences ###
    plot_eigenvalue_differences(ensmbl_avg_reg_data,output_directory,args.rigidity_type,'Comp')
    
    ### Plot Global Data ###
    plot_ln_RMS_coefficients_all(avg_ensemble_df,output_directory,args.rigidity_type,"Global","global")
    ####################################################################################################
    
    
    ### Save Data Routine ###
    avg_ensemble_df.to_csv(os.path.join(output_directory,'N_{}_ensemble_avg_amplt_unfilt.csv'.format(ensemble_count)))
    ensmbl_avg_reg_data.to_csv(os.path.join(output_directory,'N_{}_ensemble_avg_regression.csv'.format(ensemble_count)))
    fil_avg_ensemble_df.to_csv(os.path.join(output_directory,'N_{}_ensemble_avg_amplt_filt.csv'.format(ensemble_count)))
    ensemble_mode_count.most_dom_mode_df.to_csv(os.path.join(output_directory,'N_{}_ensemble_mode_count.csv'.format(ensemble_count)))
    
    if args.process_deflect_data == "Yes":
        deflect_df_all = pd.concat([pd.DataFrame.from_dict(i) for i in deflect_list],ignore_index = True)
        deflect_df_all.to_csv(os.path.join(output_directory,'N_{}_ensemble_deflection_avg.csv'.format(ensemble_count)))
    
    end_time = time.perf_counter()
    print("Took {} min to read and calculate the ensemble data".format(float((end_time-start_time)/60)))

    


