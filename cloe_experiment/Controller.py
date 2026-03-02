# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 11:39:22 2025

@author: rebecca.hart
"""

import numpy as np
from cloe_experiment.DesiredTrajectories import generate_trajectory

def get_control_tau(controller_name, entity, i, params):
    """
    Main dispatcher to select and run a specific controller.
    """
    if controller_name == 'pd_control':
        return _pd_controller(entity, i, params)
    
    elif controller_name == 'nn_controller':
        return _nn_adaptive_controller(entity, i, params)
    elif controller_name == 'nn_sgn_controller':
        return _nn_discontinuous_controller(entity, i, params)
    
    else:
        raise ValueError(f"Unknown controller name: '{controller_name}'")

# --- Specific Controller Implementations ---

def _pd_controller(entity, i, params):
    """
    Calculates tau using a Proportional-Derivative (PD) control law.
    """
    # Get gains from the controller-specific parameter dictionary
    kp = params['k1']
    kv = params['alpha1']

    # Get the current simulation time `t`
    t = entity.config.time_steps_array[i-1]

    # Get trajectory info from the main config
    traj_name = entity.config.trajectory_name
    traj_params = entity.config.trajectory_params[traj_name]
    
    # Generate the desired state for the current time
    qd, qd_dot, qd_ddot = generate_trajectory(traj_name, t, traj_params)

    # Get the current measured state
    q_current = entity.positions[:, i-1]
    q_dot_current = entity.velocities[:, i-1]

    # Calculate errors
    error_pos = qd - q_current
    error_vel = qd_dot - q_dot_current

    # Calculate control input `tau`
    tau = kp * error_pos + kv * error_vel
    return tau


def _nn_adaptive_controller(entity, i, params):
    """
    Calculates tau using a neural network-based adaptive control law.
    This implements: tau = -f_hat -k1*r -e - alpha_1*e_dot + qd_ddot
    """
    k1 = params['k1']
    alpha1 = params['alpha1']

    # 2. Get the current time and generate the desired trajectory
    t = entity.config.time_steps_array[i-1]
    traj_name = entity.config.trajectory_name
    traj_params = entity.config.trajectory_params[traj_name]
    
    # Get the trajectory points and immediately reshape them to be column vectors
    qd, qd_dot, qd_ddot = generate_trajectory(traj_name, t, traj_params)
    qd_col = qd.reshape(-1, 1)
    qd_dot_col = qd_dot.reshape(-1, 1)
    qd_ddot_col = qd_ddot.reshape(-1, 1)

    # 3. Get the system's current measured state
    # Slices are 1D, so reshape them immediately to be column vectors
    q_current_col = entity.positions[:, i-1].reshape(-1, 1)
    q_dot_current_col = entity.velocities[:, i-1].reshape(-1, 1)

    # 4. Calculate position and velocity errors (e, e_dot)
    # The result of subtracting two (2, 1) vectors is a (2, 1) vector.
    e_col = q_current_col - qd_col
    e_dot_col = q_dot_current_col - qd_dot_col
    
    # 5. Calculate the sliding surface variable (r)
    # The result is also a (2, 1) vector.
    r_col = e_dot_col + alpha1 * e_col

    # 6. Get the neural network's estimate (f_hat)
    # Create the flat input vector for the NN
    x_input = np.hstack([q_current_col.flatten(), q_dot_current_col.flatten()])
    entity.nn.nn_input = lambda step: x_input
    
    # The loss signal must also be a column vector for the update law
    loss_signal = r_col
    f_hat = entity.nn.compute_neural_network_output(step=i, loss_signal=loss_signal, entity=entity)
    
    # Ensure f_hat is also a column vector just to be safe
    f_hat_col = f_hat.reshape(-1, 1)
    if i > 0:
        entity.f_hat_history[:, i-1] = f_hat_col.flatten()

    # 7. Calculate tau. All components are now (2, 1), so the result is (2, 1).
    tau = -f_hat_col - k1 * r_col - e_col - alpha1 * e_dot_col + qd_ddot_col
   # print(f"DEBUG: Shape of tau is {tau.shape}")
    
    return tau, f_hat_col

def _nn_discontinuous_controller(entity, i, params):
    """
    Calculates tau using a neural network-based adaptive control law.
    This implements: tau = -f_hat -k1*r -e - alpha_1*e_dot + qd_ddot
    """
    k1 = params['k1']
    k2 = params['k2']
    alpha1 = params['alpha1']

    # 2. Get the current time and generate the desired trajectory
    t = entity.config.time_steps_array[i-1]
    traj_name = entity.config.trajectory_name
    traj_params = entity.config.trajectory_params[traj_name]
    
    # Get the trajectory points and immediately reshape them to be column vectors
    qd, qd_dot, qd_ddot = generate_trajectory(traj_name, t, traj_params)
    qd_col = qd.reshape(-1, 1)
    qd_dot_col = qd_dot.reshape(-1, 1)
    qd_ddot_col = qd_ddot.reshape(-1, 1)

    # 3. Get the system's current measured state
    # Slices are 1D, so reshape them immediately to be column vectors
    q_current_col = entity.positions[:, i-1].reshape(-1, 1)
    q_dot_current_col = entity.velocities[:, i-1].reshape(-1, 1)

    # 4. Calculate position and velocity errors (e, e_dot)
    # The result of subtracting two (2, 1) vectors is a (2, 1) vector.
    e_col = q_current_col - qd_col
    e_dot_col = q_dot_current_col - qd_dot_col
    
    # 5. Calculate the sliding surface variable (r)
    # The result is also a (2, 1) vector.
    r_col = e_dot_col + alpha1 * e_col

    # 6. Get the neural network's estimate (f_hat)
    # Create the flat input vector for the NN
    x_input = np.hstack([q_current_col.flatten(), q_dot_current_col.flatten()])
    entity.nn.nn_input = lambda step: x_input
    
    # The loss signal must also be a column vector for the update law
    loss_signal = r_col
    f_hat = entity.nn.compute_neural_network_output(step=i, loss_signal=loss_signal, entity=entity)
    
    # Ensure f_hat is also a column vector just to be safe
    f_hat_col = f_hat.reshape(-1, 1)
    if i > 0:
        entity.f_hat_history[:, i-1] = f_hat_col.flatten()

    # 7. Calculate tau. All components are now (2, 1), so the result is (2, 1).
    tau = -f_hat_col - k1 * r_col - (k2*np.sign(r_col)) - e_col - alpha1 * e_dot_col + qd_ddot_col
   # print(f"DEBUG: Shape of tau is {tau.shape}")
    
    return tau, f_hat_col
