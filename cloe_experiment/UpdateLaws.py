# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 12:59:14 2025

@author: rebecca.hart
"""

import numpy as np
from joblib import Parallel, delayed
import os 

def get_weights_dot(law_name, nn, loss_signal, params, entity, step):
        return _CLOE(nn, loss_signal, params, entity, step)

def _CLOE(nn, loss_signal, params, entity, step):
    # ARTIFICAL DISTURBANCE
    t = entity.config.time_steps_array[step - 1]
    # art_d_value = (np.cos(0.2*t)**2 + 
    #                 np.sin(2.0*t)**2 * np.cos(0.1*t) + 
    #                 np.sin(-1.2*t)**2 * np.cos(0.5*t) + 
    #                 np.sin(t)**5)
    art_d_value = (np.cos(0.2*t)**2 + 
               np.sin(2.0*t)**2 * (1.0 + np.cos(0.1*t)) +
               np.sin(-1.2*t)**2 * (1.0 + np.cos(0.5*t)) +
               (np.sin(t) + 1.0)**2)
    art_d = art_d_value * np.ones_like(nn.weights)
    
    gradient_sum_f_tilde, gradient_sum_f_tilde_dot = CLOE_history_stack(nn, params, entity, step)
    #print(f"Time Step: {step}")
    # print("gradient_sum_f_tilde:\n", gradient_sum_f_tilde)
   # print("gradient_sum_f_tilde_dot:\n", gradient_sum_f_tilde_dot)
    weights_dot = nn.learning_rate @ (
        nn.neural_network_gradient_wrt_weights.T @ loss_signal
        + gradient_sum_f_tilde
        - nn.gamma2 * nn.weights
        + nn.gamma3*gradient_sum_f_tilde_dot
        + nn.gamma3*nn.gamma1*nn.gamma5*gradient_sum_f_tilde
        + nn.gamma4*art_d
        )
   # print("Theta Hat Dot:\n", weights_dot)    
    return weights_dot

def CLOE_history_stack(nn, params, entity, current_step):

    update_interval = nn.config['history_update_interval']

    # --- Time-based update check ---
    is_update_time = (current_step % update_interval == 0)

    if not is_update_time:
                return nn.last_grad_f_tilde_sum, nn.last_grad_f_tilde_dot_sum
    
    gradient_sum_f_tilde = np.zeros_like(nn.weights) # This will accumulate the integral term
    gradient_sum_f_tilde_dot = np.zeros_like(nn.weights) # This will accumulate the instantaneous error term
    grad_hist_sum = np.zeros_like(nn.learning_rate)

    dt = nn.time_step_delta
    window_size = nn.config['history_window_size'] # Access as dictionary key

    offline_data = nn.config['offline_training_data'] # Access as dictionary key
    state_size = nn.config['state_size'] # Access as dictionary key


    if offline_data.shape[0] < window_size:
        raise ValueError(f"Offline training data has {offline_data.shape[0]} points, "
                          f"but window_size is {window_size}. Not enough data.")
    
    start_idx_offline = offline_data.shape[0] - window_size
    end_idx_offline = offline_data.shape[0]



    # Initialize a variable to track the cumulative integral of the instantaneous error
    cumulative_f_tilde_integral_at_point = np.zeros(state_size).reshape(-1, 1)

    for i in range(start_idx_offline, end_idx_offline):
        # Extract q, q_dot, and f_true for the current historical point from the offline data
        q_hist = offline_data[i, :state_size]
        q_dot_hist = offline_data[i, state_size : 2 * state_size]
        f_true_hist = offline_data[i, 2 * state_size : 3 * state_size].reshape(-1, 1)
        f_tilde_previous_val = nn.last_cumulative_f_tilde_integral_at_point
        
        
        x_input_hist = np.hstack([q_hist, q_dot_hist])
               
        transposed_weights = nn.construct_transposed_weight_matrices()
        activated, unactivated = nn.perform_forward_propagation(transposed_weights, nn.get_input_with_bias(x_input_hist))
        grad_hist = nn.perform_backward_propagation(activated, unactivated, transposed_weights)
        f_hat_hist = unactivated[-1] + nn.gamma1*f_tilde_previous_val
        

        # instantaneous_f_mismatch is f_true_hist - f_hat_hist, which is what f_tilde_dot represents
        instantaneous_f_mismatch = nn.gamma5 * (f_true_hist - f_hat_hist)

        # Accumulate for the f_tilde_dot term in the update law
        gradient_sum_f_tilde_dot += grad_hist.T @ instantaneous_f_mismatch

        # Integrate instantaneous_f_mismatch to get the value that f_tilde represents
        cumulative_f_tilde_integral_at_point += instantaneous_f_mismatch * dt
        
        # Accumulate for the f_tilde term in the update law
        gradient_sum_f_tilde += grad_hist.T @ cumulative_f_tilde_integral_at_point
        
        # for Gamma
        grad_hist_sum += grad_hist.T @ grad_hist

    # Store the calculated sums for subsequent simulation steps
    nn.last_grad_f_tilde_sum = gradient_sum_f_tilde
    nn.last_grad_f_tilde_dot_sum = gradient_sum_f_tilde_dot
    nn.last_grad_hist_sum = grad_hist_sum
    nn.last_cumulative_f_tilde_integral_at_point = cumulative_f_tilde_integral_at_point

    return gradient_sum_f_tilde, gradient_sum_f_tilde_dot
