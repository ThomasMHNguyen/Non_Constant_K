# About

This repository contains the scripts needed for analysis associated with this [publication](https://doi.org/10.3389/frsfm.2022.977729). 

# Summary

Given a mechanical stiffness profile across the backbone of a fiber-like particle, the deformation (bending, buckling, etc.) of the fiber particle will be different compared to a uniform stiffness profile. This framework allows you to predict the deformation properties of the fiber-like particle as a result of different stiffness profiles in fluid flow. This prediction is done through theory and corroborated through computational fluid dynamics simulations. There are 3 main components to this framework:

1. Quantification of fiber bending in non-stochastic simulations in simple shear flow

2. Linear stability analysis to predict most dominant fiber bending shapes in extensional fluid flow

3. Image processing methodology (i.e.: Linear Operator Theory (Fourier-like transforms)) to corroborate linear stability analysis with actual fiber bending shapes from stochastic ensemble simulations of fiber bending.

# Main Findings

1. Fibers with localized areas of low stiffness are more susceptible to bending and buckling. Fibers with localized areas of high stiffness are less susceptible to bending and buckling. 

2. Linear operator theory is able to predict shapes from linear stability analysis with actual fiber bending shapes with at least 75\% accuracy in terms of actual mode growth rates vs growth rates from the stochastic simulations.

# Relevant Python Packages

**Scientific Computing**: NumPy, SciPy, Pandas, statsmodels

**Scientific Plotting**: Matplotlib, Seaborn 