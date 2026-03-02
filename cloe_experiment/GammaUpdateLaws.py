# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 10:18:02 2025

@author: rebecca.hart
"""

import numpy as np
from numpy.linalg import inv, eig

def get_gamma_dot(law_name, nn, law_params, current_step):
    """
    Dispatcher for the learning rate (Gamma) update laws.
    """
    if law_name == 'CLOE':
        return _CLOE_Gamma(nn, current_step)
    else:
        # Default to no update if the law is unknown or not specified
        return np.zeros_like(nn.learning_rate)

def _CLOE_Gamma(nn, current_step):
    """
    d/dt(Gamma^-1) = -beta*Gamma^-1 + gamma_1 * sum(Phi'T * Phi')
    """
    # Get parameters from the NN config
    params = nn.config['gamma_update_law_params']
    beta_g = params['beta_g']
    gamma_3_g = nn.config['update_law_params']['gamma3']
    gamma_5_g = nn.config['update_law_params']['gamma5']
    lambda_min_g = params['lambda_min_g']
    lambda_max_g = params['lambda_max_g']

    # Get the current learning rate matrix (Gamma) and its inverse
    gamma = nn.learning_rate
    
    dt = nn.time_step_delta
    update_interval = nn.config['history_update_interval']
    
    is_update_time = (current_step % update_interval == 0)
    
    if not is_update_time:
        # If it's not time to update, return the last calculated value.
        return np.zeros_like(gamma)
    
    try:
        gamma_inv = inv(gamma)
    except np.linalg.LinAlgError:
        # If Gamma is singular, we cannot proceed. Return zero update.
        raise ValueError(
            "Gamma is singular"
        )
        return np.zeros_like(gamma)

    # Check the eigenvalue conditions
    eigenvalues = eig(gamma)[0] # We only need the eigenvalues, not eigenvectors
    lambda_min_current = np.min(eigenvalues)
    lambda_max_current = np.max(eigenvalues)

    # If conditions are NOT met, do not update (return zeros)
    if not (lambda_min_current > lambda_min_g and lambda_max_current < lambda_max_g):
        return np.zeros_like(gamma)

    # --- If conditions are met, proceed with the update ---
    
    # Get the pre-calculated sum of squared gradients from the NN state
    gradients_squared_sum = nn.last_grad_hist_sum
    
    # Calculate the derivative of the inverse of Gamma
    gamma_inv_dot = -beta_g * gamma_inv + gamma_3_g *gamma_5_g * gradients_squared_sum
    
    # Convert the derivative of the inverse to the derivative of Gamma itself
    # using the formula: dG/dt = -G * d(G^-1)/dt * G
    gamma_dot = -gamma @ gamma_inv_dot @ gamma
    
    return gamma_dot
