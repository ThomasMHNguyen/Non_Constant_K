# -*- coding: utf-8 -*-
"""
FILE NAME:      06c1_Non_Brownian_class_movement_ext.py

COMPLEMENTARY
SCRIPT(S)/
FILE(S):        N/A

DESCRIPTION:    Given an analytical form for the rigidity of the filament and 
                a certain fluid velocity profile, this script will calculate 
                and predict the movmeent and tension of the filament for a 
                specified duration. This script is to evalulate the filament in 
                extensional flow.

INPUT
FILES(S):       N/A

OUTPUT
FILES(S):       1) .NPY file that contains all positions of each discretized
                position of the filament for the duration of the simulation.
                2) .NPY file that contains all tension values of each discretized
                position of the filament for the duration of the simulation.
                3) .NPY file that contains the numerical values of the rigidity
                of the function. 
                4) .NPY file that contains the length of the filament at each
                timestep. 
                5) .NPY file that contains the stresses of the filament at each 
                timestep. 
                6) .NPY file that contains the elastic energy of the filament 
                at each timestep.
                7) .NPY file that contains the average angle of the filament 
                at each timestep.
                7) .CSV file that lists all parameters used for run. 

INPUT
ARGUMENT(S):    1) Parent Output directory: Parent directory where main directory 
                containing output files will reside in. 
                2) Main Output directory: main directory name that will 
                house all of the output files.

CREATED:        1Jun20

MODIFICATIONS
LOG:
    
15Jun20         1) Added compatibility for non-uniform rigidity of the filament.
15Jun20         2) Vectorized derivative calculations for faster computation
                time.
01Aug20         3) Changed calculation of tension from TDMA solver function
                to np.linalg.solve.
10Nov20         4) Added functionality for semi-implicit method (4th order 
                spatial derivative term only). Re-arranged calculations of 
                parameters to account for this. 
12Nov20         5) Removed functionality to plot .MP4 files for filament 
                movmeent and tension. Functionality has been moved to 
                Plot_Results.py.
03Apr21         6) Cleaned up code. Added global variables to each function.
03Apr21         7) Added function to calculate tensile and rigidity forces.
20Jul21         8) Added functions to calculate various spatial derivatives to 
                shorten code.
20Jul21         9) Added functions to calculate stresses and elastic energy. 
16Aug21         10) Adjusted time scaling for numpy array outputs to conserve memory.
01Sep21         11) Added boundary conditions for torque-free and force-free filament
                at end of Euler step. Adjusted saving steps with small Euler-step.
26Sep21         12) Created a class to house all parameters and data. Simulation
                is now initiated if "if __name__ == __main__".
25Feb22         13) Code now contains argparse and logging implementation.
20May22         14) Fixed force-free boundary conditions in the semi-implicit steps.
    
            
LAST MODIFIED
BY:             Thomas Nguyen

PYTHON
VERSION USED
TO WRITE
SCRIPT:         3.8.11

VERSION:        6.05

AUTHOR(S):      Thomas Nguyen

STATUS:         Working

TO DO LIST:    1) Brownian motion.


NOTE(S):        N/A

"""


import sys, os, math, time, random,argparse, logging
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=6
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from fractions import Fraction
from scipy import special, sparse

#%% ####################### Functions #######################


def first_deriv(base_array,N,ds,dim):
    """
    This function calculates the first derivative of the scalar or matrix of 
    interest and applies the appropriate end correction terms accurate to O(s^2).
    
    Inputs:
        
    base_array:     Nx3 array who needs its first derivative to be calculated.
    N:              Number of points used to discretize the length of the filament.
    ds:             spacing between each point on the filament.
    dim:            The number of dimensions the array has.
    """
    if dim == 1:
        dds_base = np.zeros((N),dtype = float)
        dds_base[0] = (0.5/ds)*(-3*base_array[0] + 4*base_array[1] - 1*base_array[2])
        dds_base[N-1] = (0.5/ds)*(3*base_array[N-1] - 4*base_array[N-2] + 1*base_array[N-3])
        dds_base[1:N-1] = (0.5/ds)*(-base_array[0:N-2] + base_array[2:N])
    elif dim == 2:
        dds_base = np.zeros((N,3),dtype = float)
        dds_base[0,:] = (0.5/ds)*(-3*base_array[0,:] + 4*base_array[1,:] - 1*base_array[2,:])
        dds_base[N-1,:] = (0.5/ds)*(3*base_array[N-1,:] - 4*base_array[N-2,:] + 1*base_array[N-3,:])
        dds_base[1:N-1,:] = (0.5/ds)*(-base_array[0:N-2,:] + base_array[2:N,:])
    return dds_base


def second_deriv(base_array,N,ds):
    """
    This function calculates the 2nd derivative of a vector and applies the 
    appropirate end correction terms accurate to 0(s^2).
    
    Inputs:
        
    base_array:     Nx3 array who needs its first derivative to be calculated.
    N:              Number of points used to discretize the length of the filament.
    ds:             spacing between each point on the filament.
    """
    d2ds2_base = np.zeros((N,3),dtype = float)
    d2ds2_base[0,:] = (1/ds**2)*(2*base_array[0,:]-5*base_array[1,:]+4*base_array[2,:]-base_array[3,:])
    d2ds2_base[N-1,:] = (1/ds**2)*(2*base_array[N-1,:]-5*base_array[N-2,:]+4*base_array[N-3,:]-base_array[N-4,:])
    d2ds2_base[1:N-1,:] = (1/ds**2)*(base_array[2:N,:]-2*base_array[1:N-1,:]+base_array[0:N-2,:])
    return d2ds2_base


def third_deriv(base_array,N,ds):
    """
    This function calculates the 3rd derivative of a vector and applies the 
    appropirate end correction terms accurate to 0(s^2). Additional end corrections
    are applied via https://www.geometrictools.com/Documentation/FiniteDifferences.pdf. 
    
    Inputs:
        
    base_array:     Nx3 array who needs its first derivative to be calculated.
    N:              Number of points used to discretize the length of the filament.
    ds:             spacing between each point on the filament.
    """
    d3ds3_base = np.zeros((N,3),dtype = float)
    d3ds3_base[0,:] = 1/ds**3*(-2.5*base_array[0,:]+9*base_array[1,:]-12*base_array[2,:]+7*base_array[3,:]-1.5*base_array[4,:])
    d3ds3_base[1,:] = 1/ds**3*(-1.5*base_array[0,:]+5*base_array[1,:]-6*base_array[2,:]+3*base_array[3,:]-0.5*base_array[4,:])             
    d3ds3_base[N-1,:] = 1/ds**3*(2.5*base_array[N-1,:]-9*base_array[N-2,:]+12*base_array[N-3,:]-7*base_array[N-4,:]+1.5*base_array[N-5,:])
    d3ds3_base[N-2,:] = 1/ds**3*(1.5*base_array[N-1,:]-5*base_array[N-2,:]+6*base_array[N-3,:]-3*base_array[N-4,:]+0.5*base_array[N-5,:])     
    d3ds3_base[2:N-2,:] = 1/ds**3*(0.5*base_array[4:N,:]-base_array[3:N-1,:]+base_array[1:N-3,:]-0.5*base_array[0:N-4,:])
    return d3ds3_base
 
       
def fourth_deriv(base_array,N,ds):
    """
    This function calculates the 4th derivative of a vector and applies the 
    appropirate end correction terms accurate to 0(s^2). Additional end corrections
    are applied via https://www.geometrictools.com/Documentation/FiniteDifferences.pdf. 
    
    Inputs:
        
    base_array:     Nx3 array who needs its first derivative to be calculated.
    N:              Number of points used to discretize the length of the filament.
    ds:             spacing between each point on the filament.
    """
    d4ds4_base = np.zeros((N,3),dtype = float)
    d4ds4_base[0,:] = 1/ds**4*(3*base_array[0,:]-14*base_array[1,:]+26*base_array[2,:]-24*base_array[3,:]+11*base_array[4,:]-2*base_array[5,:])
    d4ds4_base[1,:] = 1/ds**4*(2*base_array[0,:]-9*base_array[1,:]+16*base_array[2,:]-14*base_array[3,:]+6*base_array[4,:]-1*base_array[5,:])        
    d4ds4_base[N-1,:] = 1/ds**4*(3*base_array[N-1,:]-14*base_array[N-2,:]+26*base_array[N-3,:]-24*base_array[N-4,:]+11*base_array[N-5,:]-2*base_array[N-6,:])
    d4ds4_base[N-2,:] = 1/ds**4*(2*base_array[N-1,:]-9*base_array[N-2,:]+16*base_array[N-3,:]-14*base_array[N-4,:]+6*base_array[N-5,:]-1*base_array[N-6,:])    
    d4ds4_base[2:N-2,:] = 1/ds**4*(base_array[4:N,:]-4*base_array[3:N-1,:]+6*base_array[2:N-2,:]-4*base_array[1:N-3,:]+base_array[0:N-4,:])
    return d4ds4_base


def spatialderiv(sim_c,x_vec):
    """
    This function evaluates the spatial derivatives of the filament.
    
    Inputs:
    sim_c:       Class that contains all parameters needed for the simulation 
                 and arrays to store data.
    x_vec:       Nx3 vectorized array that contains the location of each point
    """
    N,ds = sim_c.N,sim_c.ds
    dxds = first_deriv(x_vec,N,ds,2)
    d2xds2 = second_deriv(x_vec,N,ds)
    d3xds3 = third_deriv(x_vec,N,ds)
    d4xds4 = fourth_deriv(x_vec,N,ds) 
    return dxds, d2xds2, d3xds3, d4xds4

def rdot(ar1,ar2):
    """
    This function computes the dot product of 2 vectors (Check LaTeX markup for
     actual markup) but accounts for the vectors as numpy arrays.
    
    Inputs:
    ar1:        Numpy 2D array representing Vector #1
    ar2:        Numpy 2D array representing Vector #2
    """
    return np.sum(np.multiply(ar1,ar2),axis=1)


def fluid_velo(sim_c,x_vec,t):
    """
    This function calculates the fluid velocity based on the position of the 
    filament. 
    
    Inputs:
    sim_c:      Class that contains all parameters needed for the simulation 
                and arrays to store data.
    x_vec:      Nx3  array that contains the coordinates of each point.
    t:          current iteration of simulation.
    """
    N = sim_c.N
    s = sim_c.s
    ds = sim_c.ds
    # dt = sim_c.dt
    
    u0 = np.zeros([N,3],dtype = float)
    
    # omega = 1/(7.5*2) #switch shear flow direction every t = 6.5
    #Shear flow
    u0[:,0] = x_vec[:,1]
        
    #Oscillatory shear flow-v2
    # u0[:,0] = np.sin(2*np.pi*omega*t*dt)*x_vec[:,1]
    
    #Extensional Flow
    # Find center of mass position (average of x, y, z)
   
    # u0[:,0] = -x_vec[:,0] - x_vec[np.where(s == 0)[0][0],0]   # Change to Relative to center of x-component
    # u0[:,1] = x_vec[:,1]  - x_vec[np.where(s == 0)[0][0],1]   # Change to relative to center of y-component

    #Velocity derivative
    du0ds = first_deriv(u0,N,ds,2)
    
    return u0, du0ds
   
def rod_length(N,x_vec):
    """
    This function calculates the length of the filament.
    
    Inputs:
    N:          Number of points used to discretize filament.
    x_vec:      Nx3 array that contains the coordinates of each point along the
                filament.
    """
    r_length = np.sqrt(((x_vec[1:N,:]-x_vec[0:N-1,:])**2).sum(axis = 1)).sum()
    return r_length
    

def calculate_N_ex(sim_c,params):
    """
    This function calculates the non-semi implicit terms.
    
    Inputs:
    sim_c:      Class that contains all parameters needed for the simulation 
                and arrays to store data.
    params:     list of arrays that contain components needed to solve explicit terms
                in the semi-implicit time-stepping. 
    """
    N,c, mu_bar = sim_c.N,sim_c.c, sim_c.mu_bar
    
    #Unpack params
    u0, du0ds, dxds, d2xds2, d3xds3,\
    Txss,Tsxs,Kxs,Kxssss,Ksxs,Ksxsss,Kssxss = params
    
    N_ex = (mu_bar*u0) - ((c+1)*(-Tsxs + Kssxss - Txss + 2*Ksxsss) + 
                          (c-3)*(-Tsxs + 2*Ksxs*np.repeat(rdot(dxds,d3xds3),3).reshape(N,3)))                                    
    return N_ex


def solve_tension(sim_c,du0ds,dxds,d2xds2,d3xds3,d4xds4):
    """
    This function solves for the tension equation using np.linalg.solve.
    
    Inputs:

    sim_c:      Class that contains all parameters needed for the simulation 
                and arrays to store data.
    du0ds:      Nx3 vectorized array that represents 1st spatial derivative of 
                fluid velocity.
    dxds:       Nx3 vectorized array that represents 1st spatial derivative of 
                filament position. 
    d2xds2:     Nx3 vectorized array that represents 2nd spatial derivative of 
                filament position. 
    d3xds3:     Nx3 vectorized array that represents 3rd spatial derivative of 
                filament position. 
    d4xds4:     Nx3 vectorized array that represents 4th spatial derivative of 
                filament position. 
    """
    mu_bar,N,c,ds = sim_c.mu_bar,sim_c.N,sim_c.c,sim_c.ds
    K,Ks,Kss = sim_c.K,sim_c.Ks,sim_c.Kss
    zeta = sim_c.zeta

    
    # Evaluating Tension Equation with BC: Tension = 0 at ends of Filament 
    a = np.ones(N-3)*(-2*(c-1)/ds**2) # Lower and Upper Diag
    b = np.ones(N-2)*(4*(c-1)/ds**2)+ (c+1)*rdot(d2xds2,d2xds2)[1:N-1] # Center Diag    
    d = ((mu_bar*rdot(du0ds,dxds))+ ((5*c-3)*(Kss*rdot(d2xds2,d2xds2))) + \
         (4*(4*c-3)*(Ks*rdot(d2xds2,d3xds3))) + \
            ((7*c-5)*K*rdot(d2xds2,d4xds4))+(6*(c-1)*K*rdot(d3xds3,d3xds3)) -\
         zeta*(1-rdot(dxds,dxds)))[1:N-1] # RHS-non constant K
        
    ### Evluate tension ###
    A = sparse.diags([a,b,a],offsets = [-1,0,1],shape = (N-2,N-2)).toarray()
    tension = np.insert(np.linalg.solve(A,d),(0,N-2),0)
    return tension


def calc_force(Txss,Tsxs,Kssxss,Ksxsss,Kxssss,f_type):
    """
    This function calculates the force experienced by the filament due to tensile
    and rigidity forces.
    
    Inputs:

    Txss:       Nx3 vectorized array of tension multiplied by 2nd derivative 
                of filament position. 
    Tsxs:       Nx3 vectorized array of tension derivative multplied by 1st 
                derivative of filament position. 
    Kssxss:     Nx3 vectorized array of 2nd derivative of rigidity multiplied 
                by 2nd derivative of filament position. 
    Ksxsss:     Nx3 vectorized array of 1st derivative of rigidity multplied 
                by 3rd derivative of filament position. 
    Kxssss:     Nx3 vectorized array of rigidity multiplied by 4th derivative 
                of filament position. 
    f_type:     string argument to determine whether to use non-tensile forces
                (for sake of future non-local operator calculations) or all 
                forces (including tensile forces).
    """
    
    if f_type == 'rigid':
        force = Kssxss + 2*Ksxsss + Kxssss
    elif f_type == 'all':
        force = -Tsxs - Txss + Kssxss + 2*Ksxsss + Kxssss
    return force


def calc_vals(sim_c,tension, dTds, dxds, d2xds2, d3xds3, d4xds4):
    """
    This function creates the arrays for the terms coupled to the tension, its
    derivative, rigidity, and spatial derivatives.
    
    Inputs:

    sim_c:      Class that contains all parameters needed for the simulation 
                and arrays to store data.
    tension:    Numpy 1D array that represents tension at each point on the 
                filament. 
    dTds:       Numpy 1D array that represents 1st spatial derivative of 
                tension at each point on the filament. 
    dxds:       Nx3 vectorized array that represents 1st spatial derivative of
                filament position. 
    d2xds2:     Nx3 vectorized array that represents 2nd spatial derivative of 
                filament position. 
    d3xds3:     Nx3 vectorized array that represents 3rd spatial derivative of
                filament position. 
    d4xds4:     Nx3 vectorized array that represents 4th spatial derivative of 
                filament position. 
    """
    K,Ks,Kss = sim_c.K,sim_c.Ks,sim_c.Kss
    
    Txss = d2xds2*(np.column_stack((tension,tension,tension)))
    Tsxs = dxds*(np.column_stack((dTds,dTds,dTds)))
    Kxs = dxds*(np.column_stack((K,K,K)))
    Kxssss = d4xds4*(np.column_stack((K,K,K)))
    Ksxs = dxds*(np.column_stack((Ks,Ks,Ks)))
    Ksxsss = d3xds3*(np.column_stack((Ks,Ks,Ks)))
    Kssxss = d2xds2*(np.column_stack((Kss,Kss,Kss)))
    
        
    calc_params = [Txss,Tsxs,Kxs,Kxssss,Ksxs,Ksxsss,Kssxss]
    return calc_params

def construct_lhs_matrix(sim_c,adj_xs):
    """
    This function creates the LHS matrix needed for np.linalg.solve. First, the 
    diagonals of the matrix are constructed using the finite difference coefficients.
    Next, the non-end terms (terms used for boundary conditions) are calculated by first 
    calculating the dyadic term, and then computing the dot product. After this 
    computation, the values are re-substituted back into each submatrix. Finally,
    the end terms are adjusted for the force-free and torque-free boundary conditions.
    
    Inputs:
    sim_c:      Class that contains all parameters needed for the simulation 
                and arrays to store data.
    adj_xs:     The 2D vectorized array that represents 2N(x^n_s) - N(x^n-1_s)
    """
    N, r1, c, K,Ks = sim_c.N, sim_c.r1, sim_c.c, sim_c.K,sim_c.Ks
    #Construct diagonals based on finite differences
    alpha = 1.5*np.ones(3*N,dtype = float)
    lhs_matrix = sparse.diags(alpha,0).toarray()
    prefactors = [r1,-4*r1,6*r1,-4*r1,r1]
    block_pos = [-6,-3,0,3,6]
    
    #### Calculate non-end terms of dyadic ###########
    identity = np.identity(3,dtype = float)
    for i in range(2,N-2):
        dyad_p = K[i]*((c+1)*identity + (c-3)*np.outer(adj_xs[i,:],adj_xs[i,:]))
        for j in block_pos: #Iterate through each index corresponding to the sub-block from the center sub-block
            lhs_matrix[3*i:3*i+3,3*i+j:3*i+j+3] += prefactors[block_pos.index(j)]*dyad_p
    
    #### Adjust end-terms for BC's ########
    lhs_matrix[0,0],lhs_matrix[1,1],lhs_matrix[2,2] = 2*K[0]*np.ones(3)
    lhs_matrix[0,3],lhs_matrix[1,4],lhs_matrix[2,5] = -5*K[0]*np.ones(3)
    lhs_matrix[0,6],lhs_matrix[1,7],lhs_matrix[2,8] = 4*K[0]*np.ones(3)
    lhs_matrix[0,9],lhs_matrix[1,10],lhs_matrix[2,11] = -1*K[0]*np.ones(3)
    
    ### 2nd row FD
    lhs_matrix[3,0],lhs_matrix[4,1],lhs_matrix[5,2] = -float(5)/2*K[0]*np.ones(3) + 2*Ks[0]*np.ones(3)
    lhs_matrix[3,3],lhs_matrix[4,4],lhs_matrix[5,5] = 9*K[0]*np.ones(3) + -5*Ks[0]*np.ones(3)
    lhs_matrix[3,6],lhs_matrix[4,7],lhs_matrix[5,8] = -12*K[0]*np.ones(3) + 4*Ks[0]*np.ones(3)
    lhs_matrix[3,9],lhs_matrix[4,10],lhs_matrix[5,11] = 7*K[0]*np.ones(3) + -1*Ks[0]*np.ones(3)
    lhs_matrix[3,12],lhs_matrix[4,13],lhs_matrix[5,14] = -float(3)/2*K[0]*np.ones(3)
    
    ### (N-1)th row BD
    lhs_matrix[-1,-1],lhs_matrix[-2,-2],lhs_matrix[-3,-3] = 2*K[-1]*np.ones(3)
    lhs_matrix[-1,-4],lhs_matrix[-2,-5],lhs_matrix[-3,-6] = -5*K[-1]*np.ones(3)
    lhs_matrix[-1,-7],lhs_matrix[-2,-8],lhs_matrix[-3,-9] = 4*K[-1]*np.ones(3)
    lhs_matrix[-1,-10],lhs_matrix[-2,-11],lhs_matrix[-3,-12] = -1*K[-1]*np.ones(3)
    
    ### (N-2)th row BD
    lhs_matrix[-4,-1],lhs_matrix[-5,-2],lhs_matrix[-6,-3] = float(5)/2*K[-1]*np.ones(3) + 2*Ks[-1]*np.ones(3)
    lhs_matrix[-4,-4],lhs_matrix[-5,-5],lhs_matrix[-6,-6] = -9*K[-1]*np.ones(3) + -5*Ks[-1]*np.ones(3)
    lhs_matrix[-4,-7],lhs_matrix[-5,-8],lhs_matrix[-6,-9] = 12*K[-1]*np.ones(3) + 4*Ks[-1]*np.ones(3)
    lhs_matrix[-4,-10],lhs_matrix[-5,-11],lhs_matrix[-6,-12] = -7*K[-1]*np.ones(3) + -1*Ks[-1]*np.ones(3)
    lhs_matrix[-4,-13],lhs_matrix[-5,-14],lhs_matrix[-6,-15] = float(3)/2*K[-1]*np.ones(3)

    return lhs_matrix

def calc_stress(sim_c,force,x_loc):
    """
    This function calculates the stress (sigma) using the following equation:
    sigma = \int^{1}_{0}{\textbf{f(s)}\textbf{x(s)} ds} with the integrand being 
    a dyadic product.
    
    Inputs: 

    sim_c:      Class that contains all parameters needed for the simulation 
                and arrays to store data.
    force:      Nx3 vectorized array that contains the force acting on each
                point of the filament. 
    x_loc:      Nx3 vectorized array that containss the location of each filament
                point.
    """
    diag_index,N,s = sim_c.diag_index,sim_c.N,sim_c.s
    ### Method 1 ### (more efficient)
    dyad_prod_all = np.outer(force,x_loc)
    dyad_prod_of_int = dyad_prod_all.flatten()[diag_index].reshape(3*N,3).reshape(N,3,3)
    stress = np.trapz(y = dyad_prod_of_int,x = s,axis = 0)
    true_stress = 0.5*(stress+stress.transpose())
    return true_stress


def calc_E_elastic(sim_c,x_ss):
    """
    This function calculates the elastic energy of the filament at a given 
    point in time using the following equation: 
        E_{elastic} = \frac{1}{2}\int^{1}_{0}{|\textbf{x}_{ss}|^{2} ds}. 
    
    Inputs:
    
    sim_c:      Class that contains all parameters needed for the simulation 
                and arrays to store data.
    x_ss:       Nx3 vectorized array that contains the 2nd derivative of the filament
                position at each point along the filament. 
    """
    s = sim_c.s
    K = sim_c.K
    integrand = K*np.linalg.norm(x_ss,axis = 1)**2
    elastic_en = 0.5*np.trapz(y = integrand,x = s)
    return elastic_en


def calc_angle(sim_c,x_vec):
    """
    This function calculates the orientation of the filament by calculating 
    the average angle at each point across the filament and averaging them.
    
    Inputs:
    sim_c:      Class that contains all parameters needed for the simulation 
                and arrays to store data.
    x_vec:      Nx3 array that contains the coordinates at each point 
                along the filament. 
    """
    s,centerline_idx = sim_c.s,sim_c.centerline_idx
    #Center filament at origin
    adj_fil_loc = x_vec[:,:] - x_vec[centerline_idx,:]
    #Adjust for arclength
    adj_loc = np.divide(adj_fil_loc[:,:],np.column_stack((s,s,s)),
                                out = np.zeros_like(x_vec[:,:]),
                                where=np.column_stack((s,s,s))!=0)
    #Calculate angle
    angle_adj_loc = np.arctan(np.divide(adj_loc[:,1],adj_loc[:,0],out = np.zeros_like(adj_loc[:,1]),where=adj_loc[:,0]!=0))
    angle_adj_loc[angle_adj_loc<0] = angle_adj_loc[angle_adj_loc<0] + np.pi #Adjust for negative angles
    fil_angle = np.average(angle_adj_loc)
    return fil_angle


def calc_deflect(sim_c,x_vec,angle):
    """
    This function calculates the deflection of the filament by measuring the 
    different between the y-coordinates of the filament to a "Base state" of the same
    angle.
    
    Inputs:
    sim_c:      Class that contains all parameters needed for the simulation 
                and arrays to store data.
    x_vec:      Nx3 array that contains the coordinates at each point 
                along the filament. 
    angle:      average angle of the filament.
    """
    
    s,centerline_idx = sim_c.s,sim_c.centerline_idx
    
    base_state_fil = np.column_stack((s*np.cos(angle),s*np.sin(angle),s*0))
    fil_deflect = x_vec[:,:] - x_vec[centerline_idx,:] #Second term adjusts for any translation of filament
    fil_deflect_all = (fil_deflect[:,1] - base_state_fil[:,1])**2 
    fil_deflect = np.sqrt(np.sum(fil_deflect_all))
    return fil_deflect


def det_rotation(sim_c,fil_loc,cur_time):
    """
    This function calculates the angle of each position along the filament and
    calculates the average angle. If the average angle is roughly 1*pi/9 while
    the shear flow moves in the postive x-direction or 8*pi/9 while the shear flow
    moves in the negative x-direction, it will reverse the direction of the shear
    flow.
    
    Inputs:
    sim_c:      Class that contains all parameters needed for the simulation 
                and arrays to store data.
    fil_loc:    Nx3 vectorized array that contains the coordinates of 
                each filament position at the curren time.
    cur_time:   current iteration in the simulation.
    
    """
    
    filament_angle = calc_angle(sim_c,fil_loc)
    filament_deflect = calc_deflect(sim_c,fil_loc, filament_angle)
        
    return filament_angle, filament_deflect
    
    
class Constants:
    """
    This class will bundle up all constants and parameters needed in the simulations to be 
    easily accessed in the functions due to multi-processing implementation.
    """
    
    ########
    
    dt_scale_count = 0
    
    def __init__(self,end_dir,rigid_func,slender_f,mubar,zeta_f,top, bottom,perturb_order,N,dt,tot_run_time,dt_save_iteration):
        
        ##### Traditional Constants #####
        self.output_dir = end_dir
        self.rigidity_suffix = rigid_func
        self.c = np.log(1/slender_f**2)
        self.mu_bar = mubar
        self.zeta = (zeta_f/self.mu_bar)*self.mu_bar
        self.N = N
        self.numerator = top
        self.denominator = bottom
        self.theta = top*np.pi/bottom
        self.theta_reflection = np.pi-self.theta
        self.perturb_order = perturb_order
        self.L = 1
        self.s = np.linspace(-(self.L/2),(self.L/2),N)
        self.centerline_idx = np.where(self.s == 0)[0][0]
        self.dt = dt
        self.ds = 1/(self.N-1)
        self.r1 = self.dt/(self.mu_bar*self.ds**4)
        self.r2 = self.dt/self.mu_bar
        self.tot_time = tot_run_time
        self.iterations = int(math.ceil(self.tot_time/self.dt/100))*100+1
        self.adj_dt_scale = math.ceil(dt_save_iteration/self.dt)
        self.adj_iterations = int(((self.iterations-1)/self.adj_dt_scale) + 1)
        self.rand_seed = random.randrange(0,1e6) + os.getpid() + int(time.time()/1e6)
        self.rng = np.random.default_rng(self.rand_seed)
        
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
        elif self.rigidity_suffix == 'K_dirac_right_l_stiff':
            self.K = 1-0.5*np.exp(-100*(self.s-0.25)**2)
            self.Ks = 100*(self.s-0.25)*np.exp(-100*(self.s-0.25)**2)
            self.Kss = np.exp(-100*(self.s-0.25)**2)*-2e4*(self.s**2-0.5*self.s+0.0575)
        elif self.rigidity_suffix == 'K_dirac_left_l_stiff2':
            self.K = 1-0.5*np.exp(-500*(self.s+0.25)**2)
            self.Ks = 500*(self.s+0.25)*np.exp(-500*(self.s+0.25)**2)
            self.Kss = np.exp(-500*(self.s+0.25)**2)*-5e5*(self.s**2+0.5*self.s+0.0615)
        elif self.rigidity_suffix == 'K_dirac_left_l_stiff3':
            self.K = 1-0.5*np.exp(-1000*(self.s+0.25)**2)
            self.Ks = 1000*(self.s+0.25)*np.exp(-1000*(self.s+0.25)**2)
            self.Kss = np.exp(-1000*(self.s+0.25)**2)*-2e6*(self.s**2+0.5*self.s+0.062)
            
        ##### Initialize parameters for fast dyadic calculations #####
        
        self.matrix_number = np.arange(0,(3*self.N)**2, dtype=np.int64()).reshape(3*self.N,3*self.N)
        self.indices = np.repeat(np.arange(0,3*self.N).reshape((self.N,3)), 3, axis=0)
        self.diag_index = np.take_along_axis(self.matrix_number, self.indices, axis=1).flatten()
        
        #### Initialize staring filament location #####
        
        self.start_x = self.s*np.cos(self.theta)
        self.start_y = self.s*np.sin(self.theta)
        # self.start_y = self.s*np.sin(self.theta) + (self.s**self.perturb_order)*(1e-1)**self.perturb_order
        # self.start_y = self.s*np.sin(self.theta) + np.ones(self.s.shape[0],dtype = float)*(1e-1)**self.perturb_order
        self.start_z = np.zeros(self.N)
        self.initial_loc = np.column_stack((self.start_x,self.start_y,self.start_z))
        
        
        ##### Initilaize Dataframe to save rotation data #####
        
        self.rotate_df = pd.DataFrame(index = range(0,200),
                         columns = ['Rotation Count','Iteration of Rotation','Time of Rotation',
                                    'Rotation Type','Radian Angle Rotation',
                                    'Radian Fraction Numerator Approximation of pi',
                                    'Radian Fraction Denominator Approximation of pi',
                                    'Filament Deflection at rotation',
                                    'Filament centerpoint x coordinate at rotation',
                                    'Filament centerpoint y coordinate at rotation',
                                    'Filament centerpoint z coordinate at rotation',
                                    'Filament center of mass x coordinate at rotation',
                                    'Filament center of mass y coordinate at rotation',
                                    'Filament center of mass z coordinate at rotation'])
        
        ##### Initialize Arrays for storing numerical data #####
        
        self.allstate = np.zeros((self.N,3,self.adj_iterations),dtype = float)
        self.all_tension_states = np.zeros((self.N,self.adj_iterations),dtype = float) 
        self.all_dTds_states = np.zeros((self.N,self.adj_iterations),dtype = float) 
        self.all_u0_states = np.zeros((self.N,3,self.adj_iterations),dtype = float) 
        self.all_du0ds_states = np.zeros((self.N,3,self.adj_iterations),dtype = float) 
        self.all_forces_states = np.zeros((self.N,3,self.adj_iterations),dtype = float)
        self.all_stress_states = np.zeros((3,3,self.adj_iterations),dtype = float)
        self.all_elastic_states = np.zeros(self.adj_iterations,dtype = float)
        
        ##### Keep track of all spatial derivatives #####
        self.all_dxds_states = np.zeros((self.N,3,self.adj_iterations),dtype = float)
        self.all_d2xds2_states = np.zeros((self.N,3,self.adj_iterations),dtype = float)
        self.all_d3xds3_states = np.zeros((self.N,3,self.adj_iterations),dtype = float)
        self.all_d4xds4_states = np.zeros((self.N,3,self.adj_iterations),dtype = float)
        
        ##### Keep track of all terms coupled to T & K #####
        self.all_Tsxs_states = np.zeros((self.N,3,self.adj_iterations),dtype = float)
        self.all_Txss_states = np.zeros((self.N,3,self.adj_iterations),dtype = float)
        self.all_Kssxss_states = np.zeros((self.N,3,self.adj_iterations),dtype = float)
        self.all_Ksxsss_states = np.zeros((self.N,3,self.adj_iterations),dtype = float)
        self.all_Kxssss_states = np.zeros((self.N,3,self.adj_iterations),dtype = float)
        self.all_Kxs_states = np.zeros((self.N,3,self.adj_iterations),dtype = float)
        self.all_Ksxs_states = np.zeros((self.N,3,self.adj_iterations),dtype = float)
        
        self.length = np.zeros(self.adj_iterations,dtype = float) #Track filament length
        self.angle_calculations = np.zeros(self.adj_iterations,dtype = float)
        self.deflection_calculations = np.zeros(self.adj_iterations,dtype = float)
        
        self.all_data = [self.allstate,self.all_u0_states,self.all_du0ds_states,self.all_dxds_states,self.all_d2xds2_states, #0-4
                    self.all_d3xds3_states, self.all_d4xds4_states, self.all_Txss_states, self.all_Tsxs_states, self.all_Kxs_states, #5-9
                    self.all_Kxssss_states, self.all_Ksxs_states, self.all_Ksxsss_states, self.all_Kssxss_states, #10-13
                    self.all_forces_states, self.all_stress_states, #14-15
                    self.all_tension_states, self.all_dTds_states, #16-17
                    self.all_elastic_states, self.length,self.angle_calculations,self.deflection_calculations] #18-21

   
    ##### Instances #####
    
    def add_scale_count(self):
        self.dt_scale_count += 1
        
    def reset_scale_count(self):
        self.dt_scale_count = 0



def eval_time_semi(sim_c,prev_params,curr_params,t):
    """
    This function solves for position of the filament at the future time step using
    the semi-implicit method.
    
    Inputs:
    sim_c:          Class that contains all parameters needed for the simulation 
                    and arrays to store data.    
    prev_params:    Parameters of the previous time step needed to solve for 
                    the future filament position.
    curr_params:    Parameters of the current time step needed to solve for
                    the future filament position.
    t:              Current iteration of the simulation.
    """ 
    
    ### Calculate explicit terms ###
    curr_N = calculate_N_ex(sim_c,curr_params[1:])
    prev_N = calculate_N_ex(sim_c,prev_params[1:])
    
    ### Calculate component needed for dyadic ###
    curr_xs = curr_params[2]
    prev_xs = prev_params[2]
    adj_xs = 2*curr_xs - prev_xs
    
    ### Matrix construction ###
    lhs_matrix = construct_lhs_matrix(sim_c,adj_xs)
    rhs_matrix = ((2*curr_params[0]) + (-0.5 * prev_params[0]) + 
                  ((2*curr_N - prev_N)*sim_c.r2)).flatten()
    
    ### Force and torque-free boundary conditions ###
    rhs_matrix[0:6] = np.zeros(6,dtype = float)
    rhs_matrix[-6:] = np.zeros(6,dtype = float)
    
    ### Solve for future iteration ###
    try:
        future_xbar = np.linalg.solve(lhs_matrix,rhs_matrix).reshape(sim_c.N,3)
    except np.linalg.LinAlgError:
        print("\n Error, Instability & Singular Matrix detected, t = ",t)
        sys.exit(1)
    fil_length = rod_length(sim_c.N,future_xbar)
    sim_c.add_scale_count()
    
    ##### Calculate parameters at time = t+1 #####
    
    ### Determine fluid velocity and derivative
    u0,du0ds = fluid_velo(sim_c,future_xbar,t+1)
    
    ### Calculate spatial derivative ###
    dxds, d2xds2, d3xds3, d4xds4 = spatialderiv(sim_c,future_xbar)
    
    ### Caculate Tension and its derivative ###
    tension = solve_tension(sim_c,du0ds, dxds, d2xds2, d3xds3, d4xds4)
    dTds = first_deriv(tension,sim_c.N,sim_c.ds,1)
    
    #Obtain spatial-derivative coupled terms
    fut_params = calc_vals(sim_c,tension, dTds, dxds, d2xds2, d3xds3, d4xds4)
    Txss,Tsxs,Kxs,Kxssss,Ksxs,Ksxsss,Kssxss = fut_params  
    force = calc_force(Txss,Tsxs,Kssxss,Ksxsss,Kxssss,'all')
    stress1 = calc_stress(sim_c,force, future_xbar)
    elastic_energy = calc_E_elastic(sim_c,d2xds2)
    fil_angle,fil_deflect = det_rotation(sim_c,future_xbar,t+1)
    
    var_to_pack = []
    var_to_pack = [future_xbar,u0,du0ds,dxds, d2xds2, #0-4
                   d3xds3, d4xds4,Txss,Tsxs,Kxs, #5-9
                   Kxssss,Ksxs,Ksxsss,Kssxss,force, #10-14
                   stress1, #15
                   tension,dTds, #16-17
                   elastic_energy,fil_length,fil_angle,fil_deflect] #18-21
    if sim_c.dt_scale_count == sim_c.adj_dt_scale:
        
        #Record variables into arrays
        scaling_position = int(t/sim_c.adj_dt_scale)
        for i in range(0,16):
                sim_c.all_data[i][:,:,scaling_position] = var_to_pack[i]
        for i in range(16,18):
            sim_c.all_data[i][:,scaling_position] = var_to_pack[i]
        for i in range(18,22):
            sim_c.all_data[i][scaling_position] = var_to_pack[i]
        sim_c.reset_scale_count()
    prev_params = curr_params
    curr_params = [v for i,v in enumerate(var_to_pack[:14]) if i != 6]
    
    return sim_c,prev_params,curr_params


def eval_time_euler(sim_c,t):
    """
    This function solves for position of the filament at the future time step using a
    general Euler method.
    
    Inputs:
    sim_c:      Class that contains all parameters needed for the simulation 
                and arrays to store data.
    t:          Current iteration of the simulation.
    """  
    
    ### Get filament length
    fil_length = rod_length(sim_c.N,sim_c.initial_loc)
    ### Determine fluid velocity and derivative
    u0,du0ds = fluid_velo(sim_c,sim_c.initial_loc,t)
    
    ### Calculate spatial derivative
    dxds, d2xds2, d3xds3, d4xds4 = spatialderiv(sim_c,sim_c.initial_loc)

    ### Caculate Tension and its derivative ###
    tension = solve_tension(sim_c,du0ds, dxds, d2xds2, d3xds3, d4xds4)
    dTds = first_deriv(tension,sim_c.N,sim_c.ds,1)
    
    #Obtain spatial-derivative coupled terms
    curr_params = calc_vals(sim_c,tension, dTds, dxds, d2xds2, d3xds3, d4xds4)
    Txss,Tsxs,Kxs,Kxssss,Ksxs,Ksxsss,Kssxss = curr_params    
    force = calc_force(Txss,Tsxs,Kssxss,Ksxsss,Kxssss,'all')
    stress = calc_stress(sim_c,force, sim_c.initial_loc)
    elastic_energy = calc_E_elastic(sim_c,d2xds2)
    fil_angle,fil_deflect = det_rotation(sim_c,sim_c.initial_loc,t)
    #Record variables into arrays
    var_to_pack = []
    var_to_pack = [sim_c.initial_loc,u0,du0ds,dxds, d2xds2, #0-4
                   d3xds3, d4xds4,Txss,Tsxs,Kxs, #5-9
                   Kxssss,Ksxs,Ksxsss,Kssxss,force, #10-14
                   stress, #15
                   tension,dTds, #16-17
                   elastic_energy,fil_length,fil_angle,fil_deflect] #18-21
    

    for i in range(0,16):
            sim_c.all_data[i][:,:,t] = var_to_pack[i]
    for i in range(16,18):
            sim_c.all_data[i][:,t] = var_to_pack[i]
    for i in range(18,22):
            sim_c.all_data[i][t] = var_to_pack[i]
    
    mu_bar,c,N,dt = sim_c.mu_bar,sim_c.c,sim_c.N,sim_c.dt
    ##### Solve for next iteration #####
    xt = (mu_bar*u0 -(c+1)*force-(c-3)*(Kxs*np.repeat(rdot(
        dxds,d4xds4),3).reshape(N,3) + \
            2*Ksxs*np.repeat(rdot(dxds,d3xds3),3).reshape(N,3) - Tsxs))/mu_bar
                
    xt[0,:] = (1/11)*(48*xt[2,:] - 52*xt[3,:] + 15*xt[4,:])
    xt[1,:] = (1/11)*(28*xt[2,:] - 23*xt[3,:] + 6*xt[4,:])
    xt[N-1,:] = (1/11)*(48*xt[N-3,:] - 52*xt[N-4,:] + 15*xt[N-5,:])
    xt[N-2,:] = (1/11)*(28*xt[N-3,:] - 23*xt[N-4,:] + 6*xt[N-5,:])

    future_xbar = sim_c.initial_loc + 1e-5*dt*xt #filament position at t=1
        
    fil_length_1 = rod_length(sim_c.N,future_xbar)
    
    ### Determine fluid velocity and derivative
    u0_1,du0ds_1 = fluid_velo(sim_c,future_xbar,t+1)
    
    ### Calculate spatial derivative
    dxds_1, d2xds2_1, d3xds3_1, d4xds4_1 = spatialderiv(sim_c,future_xbar)

    ### Caculate Tension and its derivative ###
    tension_1 = solve_tension(sim_c,du0ds_1, dxds_1, d2xds2_1, d3xds3_1, d4xds4_1)
    dTds_1 = first_deriv(tension_1,sim_c.N,sim_c.ds,1)
    
    #Obtain spatial-derivative coupled terms
    fut_params = calc_vals(sim_c,tension_1, dTds_1, dxds_1, d2xds2_1, d3xds3_1, d4xds4_1)
    Txss_1,Tsxs_1,Kxs_1,Kxssss_1,Ksxs_1,Ksxsss_1,Kssxss_1 = fut_params    
    force_1 = calc_force(Txss_1,Tsxs_1,Kssxss_1,Ksxsss_1,Kxssss_1,'all')
    stress_1 = calc_stress(sim_c,force_1, future_xbar)
    elastic_energy_1 = calc_E_elastic(sim_c,d2xds2_1)
    fil_angle_1,fil_deflect_1 = det_rotation(sim_c,future_xbar,t+1)
    #Save data for next iteration
    fut_var_to_pack = []
    fut_var_to_pack = [future_xbar,u0_1,du0ds_1,dxds_1, d2xds2_1, #0-4
                   d3xds3_1, d4xds4_1,Txss_1,Tsxs_1,Kxs_1, #5-9
                   Kxssss_1,Ksxs_1,Ksxsss_1,Kssxss_1,force_1, #10-14
                   stress_1, #15
                   tension_1,dTds_1, #16-17
                   elastic_energy_1,fil_length_1,fil_angle_1,fil_deflect_1] #18-21
    
    return sim_c,fut_var_to_pack
            


def run_simulation(target_dir,rigid_type,mu_bar,suffix):
    """
    This function runs the simulation for the filament at a specified mu_bar value.
    
    Inputs:
    target_dir:     Main output directory where the simulations files & folders
                    will reside in.
    rigid_type:     Type of filament rigidity that the simulations will run on.
    mu_bar:         Mu_bar value to run the simulation at.
    suffix:         Directory suffix where files will reside in.
    """
    
    ####################### Initialize all parameters and output directory #######################    
    sim_c = Constants(end_dir = target_dir,rigid_func = rigid_type,
                                slender_f = 0.01,mubar = mu_bar,zeta_f = 100*mu_bar,
                                top = 8, bottom = 9,perturb_order = 4,
                                N = 101,dt = 0.001,tot_run_time = 6,
                                dt_save_iteration = 0.001)
    
    dir_name = '{}/Mu_bar_{}/{}'.format(sim_c.rigidity_suffix,int(sim_c.mu_bar),suffix)
    output_dir = os.path.join(sim_c.output_dir,dir_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
     ####################### Time iterations-Initial Euler Step #######################
        
    
    start_time = time.perf_counter()
    t = 0
    sim_c,small_step_vars = eval_time_euler(sim_c,t)
    prev_info = [sim_c.all_data[i][:,:,0] for i in range(0,14) if i != 6]#  unpack parameters at t = 0 #
    curr_info = [small_step_vars[i] for i in range(0,14)  if i != 6] # unpack parameters at next iteration
    
    ####################### Time iterations-Subsequent Steps #######################
    # for t in set(range(1,sim_c.iterations)):
    for t in tqdm(set(range(1,sim_c.iterations))):
        sim_c,prev_info,curr_info = eval_time_semi(sim_c,prev_info,curr_info,t)
        if sim_c.dt_scale_count == 0:
            if np.isnan(sim_c.all_data[0][:,:,int(t/sim_c.adj_dt_scale)]).any():
                print('t=',t)
                sim_c.iterations = t
                sys.exit(1)
            
    end_time = time.perf_counter()
    print("\n Total computing time took {} seconds".format(end_time - start_time))
    
    ### ####################### Save numpy arrays & parameter values #######################
    # np.save(os.path.join(output_dir,'filament_allstate.npy'),sim_c.all_data[0])
    # np.save(os.path.join(output_dir,'filament_stress_all.npy'),sim_c.all_data[15])
    # np.save(os.path.join(output_dir,'filament_tension.npy'),sim_c.all_data[16])
    # np.save(os.path.join(output_dir,'filament_elastic_energy.npy'),sim_c.all_data[18])
    # np.save(os.path.join(output_dir,'filament_length.npy'),sim_c.all_data[19])
    # np.save(os.path.join(output_dir,'filament_angle.npy'),sim_c.all_data[20])
    
    parameter_df = pd.DataFrame(index = ['Random Seed','Filament Length','Filament s start','Filament s end',
                                          'c','Mu_bar','zeta','N','theta_num','theta_den','dt',
                                          'Number of iterations','Total Run Time','Adjusted Scaling',
                                          'Adjusted Number of iterations',
                                          'Perturb Order','Starting x_pos',
                                          'Starting y_pos','Starting z_pos',
                                          'Calculation time','Rigidity Function Type'],
                                columns = ['Value'])
    parameter_df.loc['Random Seed','Value'] =sim_c.rand_seed
    parameter_df.loc['Filament Length','Value'] = sim_c.L
    parameter_df.loc['Filament s start','Value'] = -sim_c.L/2
    parameter_df.loc['Filament s end','Value'] = sim_c.L/2
    parameter_df.loc['c','Value'] = sim_c.c
    parameter_df.loc['Mu_bar','Value'] = sim_c.mu_bar
    parameter_df.loc['zeta','Value'] = sim_c.zeta
    parameter_df.loc['N','Value'] = sim_c.N
    parameter_df.loc['theta_num','Value'] = sim_c.numerator
    parameter_df.loc['theta_den','Value'] = sim_c.denominator
    parameter_df.loc['dt','Value'] = sim_c.dt
    parameter_df.loc['Number of iterations','Value'] = sim_c.iterations
    parameter_df.loc['Total Run Time'] = sim_c.tot_time
    parameter_df.loc['Adjusted Scaling','Value'] = sim_c.adj_dt_scale
    parameter_df.loc['Adjusted Number of iterations','Value'] = sim_c.adj_iterations
    parameter_df.loc['Calculation time','Value']  = end_time - start_time
    parameter_df.loc['Rigidity Function Type','Value']  = sim_c.rigidity_suffix
    parameter_df.loc['Perturb Order','Value'] = 4
    # parameter_df.to_csv(os.path.join(output_dir,'parameter_values.csv'))
    print("Finished all computations.")
    
    return sim_c
    
def Main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output_directory", 
                        help="The main directory where the simulation files will reside in",
                    type = str)
    parser.add_argument("rigidity_type",
                        help = "Specify what kind of rigidity profile this simulation will run on",
                        type = str,
                        choices = {"K_constant","K_parabola_center_l_stiff",'K_parabola_center_l_stiff',
                                   'K_parabola_center_m_stiff','K_linear','K_dirac_center_l_stiff',
                                   'K_dirac_center_l_stiff2','K_dirac_center_m_stiff','K_parabola_shifted',
                                   'K_error_function','K_dirac_left_l_stiff','K_dirac_left_l_stiff2',
                                   'K_dirac_left_l_stiff3'})
    parser.add_argument("mu_bar",
                        help = "Specify what the mu_bar value will be to run the simulations on",
                        type = int)
    parser.add_argument("dir_suffix",
                        help = "Specify what the directory suffix name will be",
                        type = str)
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO,
            format='%(asctime)s|%(filename)s|%(levelname)s|%(message)s',
            datefmt="%A, %B %d at %I:%M:%S %p")
    logging.info(
        "Started Simulation script for a rigidity profile of {},Mu_bar = {}, Directory suffix name = {}".format(
            args.rigidity_type,args.mu_bar,args.dir_suffix))
                 
                 
    start_time = time.perf_counter()
    sim_c = run_simulation(args.output_directory,args.rigidity_type,args.mu_bar,args.dir_suffix)
    end_time = time.perf_counter()
    logging.info("Finished all computations. Time to complete all tasks is {} seconds".format(
            end_time - start_time))
    return sim_c
    
#%% ####################### Initialization of Main Script #######################
if __name__ == '__main__':
    __spec__ = None
    sim_data = Main()
    
time_val_of_int = 2.732
npy_time_val_of_int = int(time_val_of_int/(sim_data.dt*sim_data.adj_dt_scale))
#%% ####################### Post-Visualization: Filament Length #######################

plt.figure(figsize = (8,8))
plt.plot(np.linspace(0,sim_data.tot_time,sim_data.adj_iterations),sim_data.all_data[19])

plt.axis()
ax = plt.gca()
plt.xlabel('Time',fontsize=16)
plt.ylabel('Filament length',fontsize=16)
ax.set_title('Filament Length over Simulation')
# plt.ylim(0.999,1.001)
plt.show()



#%% ####################### Post-Visualization: Filament Tension #######################


plt.figure(figsize = (8,8))
plt.plot(sim_data.all_data[16][:,0],'b',label = 'Initial Tension')
plt.plot(sim_data.all_data[16][:,npy_time_val_of_int],'green',label = 't = {}'.format(time_val_of_int))
plt.plot(sim_data.all_data[16][:,-1],'r',label = 'Final Tension')

plt.axis()
ax = plt.gca()
plt.ylabel('Tension',fontsize=16)
plt.legend(fontsize=20,loc='upper right')
# plt.ylim(0.999,1.001)

plt.show()

#%% ####################### Post-Visualization: Filament Elastic Energy #######################


plt.figure(figsize = (8,8))
plt.plot(np.arange(0,sim_data.tot_time+sim_data.dt,sim_data.dt*sim_data.adj_dt_scale),
          sim_data.all_data[18],'b',label = 'Elastic Energy')


plt.axis()
ax = plt.gca()
plt.ylabel('Elastic Energy',fontsize=16)
plt.legend(fontsize=20,loc='upper right')


plt.show()
#%% ####################### Post-Visualization: Filament Positions #######################

## Initial and Final Filament Position ##
plt.figure(figsize = (8,8))
plt.plot(sim_data.all_data[0][:,0,0],sim_data.all_data[0][:,1,0],'b',label = 'Initial Position')
plt.plot(sim_data.all_data[0][:,0,npy_time_val_of_int],sim_data.all_data[0][:,1,npy_time_val_of_int],'green',label = 't = {}'.format(time_val_of_int))
plt.plot(sim_data.all_data[0][:,0,-1],sim_data.all_data[0][:,1,-1],'r',label = 'Final Position')
plt.axis('square')
plt.xlabel('x',fontsize=20)
plt.ylabel('y',fontsize=20)
# plt.legend(['Initial Pos','Buckling','Final Pos'],fontsize=20,loc='upper right')
plt.legend(fontsize=20,loc='upper right')
plt.xlim(-0.6,0.6)
plt.ylim(-0.6,0.6)

plt.show()

