# -*- coding: utf-8 -*-
"""

Fit any arbitrary function to a Fourier series. The function sets a tolerance for
the fitting of the series. If at a given n the series error is not within this tolerance
higher values of n are then fit. This continues until the unknown function is
estimated, or fails after a certain amount of iterations.

@author: Michael Grant
"""


import numpy as np
from scipy.optimize import minimize
 
def solve_fourier_series(x, a0, an, bn):
    """ 
    Given lists of known fourier coefficents, estimates what the fouries series
    would return at each value of x.
    """
    L = np.max(x) - np.min(x)
    fx = sum([a0/2 + (a * np.cos(2 * np.pi * n * x / L)) + (b * np.sin(2 * np.pi * n * x / L)) for n, (a, b) in enumerate(zip(an, bn), start=1)])
    
    return fx

def error_function_minimization(coeffs, x, y):
    
    """ 
    Created for the scipy.minimization function. Allows this function call to
    optimize the fourier coefficents for a given function.
    """
    
    a0, an, bn = list_to_coeffs(coeffs)
    fourier = solve_fourier_series(x, a0, an, bn)
    
    return np.mean((fourier - y)**2)

def minimize_fourier(x, y, threshold=1e-7, max_n=50):
    
    """ 
    Conducts fitting of the function to a fourier series. Starts with fewest n
    terms possible. Will continue minimization until either the error threshold
    is obtained or the max number of n specified is reached.
    """
    
    error_difference = np.inf
    current_error = np.inf
    n_terms = 1
    
    while error_difference > threshold:
        initial_guess = [0.0] + [0.0] * n_terms + [0.0] * n_terms
        
        #Chose Powell because it can minimize constant functions
        result = minimize(error_function_minimization, initial_guess, args=(x,y), method="Powell")
        
        a0, an, bn = list_to_coeffs(result.x)
    
        function_approx = solve_fourier_series(x, a0, an, bn)
        
        iteration_error = np.mean((function_approx - y)**2)
        
        error_difference = current_error - iteration_error

        current_error = iteration_error

        n_terms += 1

    return a0, an, bn        
        
def list_to_coeffs(coefficent_list):
    
    """
    The minimization function from scipy requires a list to be passed, not a list
    of lists. Therefore, this function was created to parse the coefficents list.
    the first index is always a0, then the rest are split in half, where the first
    half are the an coefficents and the second half are the bn coefficents.
    """
    
    total_an_bn = len(coefficent_list)
    an_index_end = int((total_an_bn / 2) + 1)
    a0 = coefficent_list[0]
    an = coefficent_list[1:an_index_end]
    bn = coefficent_list[an_index_end:]
    
    return a0, an, bn



#Uncomment for example

# import matplotlib.pyplot as plt

# x = np.linspace(0, 2*np.pi, 100)
# y = np.sin(3*x) + np.cos(4*x)**3 - np.cos(np.sin(2.5*x)**2)**3

# best_values = minimize_fourier(x, y)

# a0, an, bn = best_values

# function_approx = solve_fourier_series(x, a0, an, bn)

# # Create the plots
# plt.figure(figsize=(12, 6))

# plt.subplot(2, 1, 1)
# plt.plot(x, y, label='Actual Function', color='blue')
# plt.title('Actual Function vs. Fourier Series Approximation')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend()

# plt.subplot(2, 1, 2)
# plt.plot(x, function_approx, label='Fourier Series Approximation', color='red')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend()

# plt.tight_layout()
# plt.show()
