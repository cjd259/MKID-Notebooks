# -*- coding: utf-8 -*-
'''
Created on Thu Feb 21 10:10:49 2019

@author: Heather McCarrick

Given a complex s21 sweep, returns the data fit
to the resonator model and the resonator parameters 

Based on equation 11 from Kahlil et al. and adapted from
Columbia KIDs open source analysis code

'''

import numpy as np

'''
first the parts of the fitting equation
'''
def linear_resonator(f, f_0, Q, Q_e_real, Q_e_imag):
    Q_e = Q_e_real + 1j*Q_e_imag
    return (1 - (Q * Q_e**(-1) /(1 + 2j * Q * (f - f_0) / f_0) ) )

def cable_delay(f, delay, phi, f_min):
    return np.exp(1j * (-2 * np.pi * (f - f_min) * delay + phi))

def general_cable(f, delay, phi, f_min, A_mag, A_slope):
    phase_term =  cable_delay(f,delay,phi,f_min)
    magnitude_term = ((f-f_min)*A_slope + 1)* A_mag
    return magnitude_term*phase_term
    
def resonator_cable(f, f_0, Q, Q_e_real, Q_e_imag, delay, phi, f_min, A_mag, A_slope):
    #combine above functions into our full fitting functions
    resonator_term = linear_resonator(f, f_0, Q, Q_e_real, Q_e_imag) 
    cable_term = general_cable(f, delay, phi, f_min, A_mag, A_slope)
    return cable_term*resonator_term


'''
some other functions you probably want that the fitting does not directly return
'''
def get_qi(Q, Q_e_real):
    return (Q**-1 - Q_e_real**-1)**-1

def get_br(Q, f_0):
    return f_0*(2 * Q)**-1

def reduced_chi_squared(ydata, ymod, n_param=9, sd=None):
    #red chi squared in lmfit does not return something reasonable 
    #so here is a handwritten function
    #you want sd to be the complex error

    chisq = np.sum((np.real(ydata) - np.real(ymod))**2/((np.real(sd))**2)) + np.sum((np.imag(ydata) - np.imag(ymod))**2/((np.imag(sd))**2))
    nu=2*ydata.size-n_param     #multiply  the usual by 2 since complex
    red_chisq = chisq/nu 
    return chisq, red_chisq

def residuals(ydata, ymod):
    return ydata-ymod
