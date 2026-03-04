# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 08:52:59 2025

@author: rebecca.hart
"""

import numpy as np
from numpy.linalg import eigvals
from cloe_experiment.DesiredTrajectories import generate_trajectory
from cloe_experiment.Controller import get_control_tau
from cloe_experiment.UpdateLaws import get_weights_dot

class Entity:
    def __init__(self, config, nn_instance):
        self.positions = np.zeros((config.state_size, config.total_steps))
        self.velocities = np.zeros((config.state_size, config.total_steps))
        self.acceleration = np.zeros((config.state_size, config.total_steps))
        self.tau = np.zeros((config.state_size, config.total_steps))
        self.r = np.zeros((config.state_size, config.total_steps))
        self.r_hat = np.zeros((config.state_size, config.total_steps))
        self.r_tilde = np.zeros((config.state_size, config.total_steps))
        self.delta_hat = np.zeros((config.state_size, config.total_steps))
        self.delta_hat_int = np.zeros((config.state_size, config.total_steps))
        
        self.fx_history = np.zeros((config.state_size, config.total_steps))
        self.f_hat_history = np.zeros((config.state_size, config.total_steps))
        self.f_tilde_int_history = np.zeros((config.state_size, config.total_steps))

        # Initialize the first time step (t=0) for all arrays
        self.positions[:, 0] = config.q_init
        self.velocities[:, 0] = config.q_dot_init

        self.r_hat[:, 0] = config.r_hat0
        self.delta_hat[:, 0] = config.delta_hat0
        self.delta_hat_int[:, 0] = config.delta_hat_int0
        
        self.nn = nn_instance
        self.weights_history = np.zeros((self.nn.weights.size, config.total_steps))
        self.weights_history[:,0] = self.nn.weights.flatten()
        
        self.gamma_history = np.zeros((self.nn.learning_rate.shape[0],  config.total_steps)) # Stores the full matrix
        self.gamma_history[:,0] =np.diag( self.nn.learning_rate)
        
        #grad_sum_matrix_shape = self.nn.last_grad_hist_sum.shape
        #self.grad_hist_sum_history = np.zeros((grad_sum_matrix_shape[0],grad_sum_matrix_shape[1], config.total_steps))
        self.min_eig_grad_hist_sum_history = np.zeros(config.total_steps)
        initial_grad_sum = self.nn.last_grad_hist_sum
        try:
            # Ensure it's not empty and calculate eigenvalues
            if initial_grad_sum.size > 0 and not np.isnan(initial_grad_sum).any() and not np.isinf(initial_grad_sum).any():
                self.min_eig_grad_hist_sum_history[0] = np.min(np.real(eigvals(initial_grad_sum)))
            else:
                self.min_eig_grad_hist_sum_history[0] = 0.0 # Or np.nan if you prefer
        except np.linalg.LinAlgError:
            self.min_eig_grad_hist_sum_history[0] = 0.0 # Handle singular matrix at start
        except Exception as e:
            print(f"Warning: Error calculating initial min eigenvalue: {e}")
            self.min_eig_grad_hist_sum_history[0] = 0.0
            
        # --- Other Setup ---
        self.dynamics = config.dynamics_func

        self.trajectory_name = config.trajectory_name
        self.trajectory_params = config.trajectory_params[config.trajectory_name]
        self.dt = config.dt
        self.config = config
        self.nn = nn_instance
        
        self.disturbance_signal = None
        if self.config.disturbance["enabled"]: # Corrected: Access as attribute, then dictionary
            disturbance_type = self.config.disturbance["type"]
            disturbance_params = self.config.disturbance.get("params", {})
            self.disturbance_signal = self.generate_disturbance(
                total_steps=config.total_steps, num_dof=config.state_size, disturbance_type=disturbance_type, params=disturbance_params # Pass config.state_size as the num_dof argument
            )
        else:
            # If disturbance is disabled, create a zero disturbance signal
            self.disturbance_signal = np.zeros((config.state_size, config.total_steps)) # Use config.state_size directly


    def generate_disturbance(self, total_steps, num_dof, disturbance_type, params=None): # Changed 'config.state_size' to 'num_dof'
        if params is None:
            params = {}

        disturbance = np.zeros((num_dof, total_steps)) # Use 'num_dof' here
        time_vector = np.arange(0, total_steps * self.dt, self.dt)

        if disturbance_type == 'white_noise':
            mean = params.get('mean', 0.0)
            std_dev = params.get('std_dev', 0.1) # Default standard deviation
            #print(f"DEBUG: generate_disturbance using std_dev = {std_dev}") # Add this line
            for dof in range(num_dof): # Use 'num_dof' here
                disturbance[dof, :] = np.random.normal(mean, std_dev, total_steps)
                
        
        elif disturbance_type == 'sinusoidal':
            amplitude = params.get('amplitude', 0.2)
            frequency = params.get('frequency', 0.5) # Hz
            for dof in range(num_dof): # Use 'num_dof' here
                disturbance[dof, :] = amplitude * np.sin(2 * np.pi * frequency * time_vector)
        # Add more disturbance types as needed (e.g., impulse, colored noise)
        else:
            raise ValueError(f"Unknown disturbance type: {disturbance_type}")
        return disturbance

    def update_state(self, i, x, dx):
        # 1. Get state from the PREVIOUS time step (i-1)
        q_prev = self.positions[:, i-1]
        q_dot_prev = self.velocities[:, i-1]
    
        # 2. Calculate the control input `tau` for the current step
        tau, f_hat_history = get_control_tau(self.config.controller_name, self, i, self.config.controller_params)
        
        self.f_hat_history[:,i]=f_hat_history.flatten()
        self.f_tilde_int_history[:,i] = self.nn.last_cumulative_f_tilde_integral_at_point.flatten()
        
        self.tau[:,i] = tau.flatten()
        
        self.weights_history[:, i] = self.nn.weights.flatten()
        self.gamma_history[:, i] = np.diag(self.nn.learning_rate)
        #self.grad_hist_sum_history[:,:,i] = self.nn.last_grad_hist_sum
        # --- NEW: Store Minimum Eigenvalue ---
        current_grad_sum = self.nn.last_grad_hist_sum
        try:
            if current_grad_sum.size > 0 and not np.isnan(current_grad_sum).any() and not np.isinf(current_grad_sum).any():
                self.min_eig_grad_hist_sum_history[i] = np.min(np.real(eigvals(current_grad_sum)))
            else:
                self.min_eig_grad_hist_sum_history[i] = 0.0 # Or np.nan
        except np.linalg.LinAlgError:
            self.min_eig_grad_hist_sum_history[i] = 0.0 # Handle singular matrix
        except Exception as e:
            print(f"Warning: Error calculating min eigenvalue at step {i}: {e}")
            self.min_eig_grad_hist_sum_history[i] = 0.0
        # --- END NEW ---
        
        # 3. Calculate acceleration by passing the correct inputs to your dynamics function
        q_dotdot, fx = self.dynamics(q_prev, q_dot_prev, tau)
        
        self.fx_history[:, i] = fx # Store true dynamics
        
        if self.disturbance_signal is None or i >= self.disturbance_signal.shape[1]:
            # Handle cases where disturbance isn't set or is too short
            # For simplicity, if not set, assume zero disturbance
            d_t = np.zeros(self.config.state_size)
            print(f"Warning: Disturbance signal not available for step {i}. Using zero disturbance.")
        else:
            d_t = self.disturbance_signal[:, i] # Get the disturbance for the current time step
        
        self.acceleration[:, i] = q_dotdot + d_t

        
        
    
        # 4. Update velocity and position for the CURRENT step (i)
        #    - Use the NEW acceleration (q_dotdot)
        #self.velocities[:, i] = q_dot_prev + self.acceleration[:, i] * self.dt
        self.velocities[:, i] = dx
        #    - Use the NEW velocity to get a more accurate position
        #self.positions[:, i] = q_prev + self.velocities[:, i] * self.dt 
        self.positions[:, i] = x
        return self.acceleration[:,i], self.positions[:,i], self.velocities[:,i]
        
    
    def update_observer(self, i, qd, qd_dot, qd_ddot):
        # Get current state from the main history arrays at column `i`
        q = self.positions[:, i]
        q_dot = self.velocities[:, i]
        tau = self.tau[:, i]
        
        # Get previous observer states from their history arrays at column `i-1`
        r_hat_prev = self.r_hat[:, i-1]
        delta_hat_prev = self.delta_hat[:, i-1]
        delta_hat_int_prev = self.delta_hat_int[:, i-1]

        # Get gains from config
        alpha_1 = self.config.alpha1
        alpha_2 = self.config.alpha2
        k_delta = self.config.k_delta

        # --- r observer ---
        r = (q_dot - qd_dot) + alpha_1 * (q - qd)
        r_hat_dot = delta_hat_prev - qd_ddot + alpha_1 * (q_dot - qd_dot) + alpha_2 * (r - r_hat_prev)
        r_hat_new = r_hat_prev + self.dt * r_hat_dot

        r_tilde = r - r_hat_new

        # --- delta observer ---
        delta_hat_int_dot = ((k_delta * alpha_2) + 1) * (r_tilde - self.config.r_tilde0)
        delta_hat_int_new = delta_hat_int_prev + self.dt * delta_hat_int_dot
        delta_hat_new = self.config.delta_hat0 + k_delta * (r_tilde - self.config.r_tilde0) + (tau - self.config.tau0) + delta_hat_int_new

        # --- Store all results directly into their respective history arrays for step `i` ---
        self.r[:, i] = r
        self.r_hat[:, i] = r_hat_new
        self.r_tilde[:, i] = r_tilde
        self.delta_hat[:, i] = delta_hat_new
        self.delta_hat_int[:, i] = delta_hat_int_new
        
    def update_neural_network_weights(self, loss_signal):
       
        # Get the name and parameters for the chosen update law from config
        law_name = self.config['update_law_name']
        law_params = self.config['update_law_params']
        
        # Call the dispatcher to get the rate of change of the weights
        weights_dot = get_weights_dot(law_name, self, loss_signal, law_params)
        projected_weights = self.proj(weights_dot, self.weights, self.weight_bounds)
        self.current_weights = projected_weights
        return projected_weights
        
        
    def proj(self, Theta, thetaHat, thetaBar):
        max_term = max(0.0, np.dot(thetaHat.T, thetaHat) - thetaBar**2)
        dot_term = np.dot(thetaHat.T, Theta)
        numerator = max_term**2 * (dot_term + np.sqrt(dot_term**2 + 1.0)) * thetaHat
        denominator = 2.0 * (1.0 + 2.0 * thetaBar)**2 * thetaBar**2
        return Theta - (numerator / denominator)
        


