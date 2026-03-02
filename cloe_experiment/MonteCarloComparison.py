# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 15:00:51 2025

@author: rebecca.hart
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
from joblib import Parallel, delayed # For parallelizing Monte Carlo trials
import copy
import csv # For robust CSV export

# Import Dynamics
from cloe_experiment.GeneralDynamics import _complex_trig_dynamics
from cloe_experiment.DesiredTrajectories import generate_trajectory

# Simulation setup
from cloe_experiment.Config import Config
from cloe_experiment.Entity import Entity
from cloe_experiment.DNN_Try1 import NeuralNetwork

# --- Helper Function for a Single Monte Carlo Trial ---
def run_single_simulation_trial(trial_params, offline_data_full, max_error_threshold):
    """
    Runs a single simulation trial with the given parameters and returns key metrics.
    Terminates early and returns None if tracking error exceeds max_error_threshold.

    Args:
        trial_params (dict): A dictionary containing all simulation parameters for this trial.
        offline_data_full (np.array): The full offline training data loaded once.
        max_error_threshold (float): The maximum allowed tracking error norm before discarding.

    Returns:
        dict: A dictionary of key performance metrics for this trial, or None if discarded.
    """
    # Create a deep copy of trial_params to avoid modifying the original dictionary
    # especially important if running in parallel processes.
    current_sim_params = copy.deepcopy(trial_params)

    # --- Derived Parameters and Initial Conditions (as in original Main.py) ---
    current_sim_params["delta_hat0"] = np.zeros(current_sim_params["state_size"])
    current_sim_params["delta_hat_int0"] = np.zeros(current_sim_params["state_size"])
    current_sim_params["tau0"] = np.zeros(current_sim_params["state_size"])

    traj_name = current_sim_params["trajectory_name"]
    specific_params = current_sim_params["trajectory_params"][traj_name]

    qd0, qd_dot0, _ = generate_trajectory(traj_name, 0.0, specific_params)
    r_hat0 = current_sim_params['q_dot_init'] - qd_dot0 + current_sim_params["controller_params"]['alpha1'] * (current_sim_params['q_init'] - qd0)
    current_sim_params["r_hat0"] = r_hat0
    current_sim_params["r_tilde0"] = np.zeros(current_sim_params["state_size"])

    # Attach the offline data to the current simulation parameters
    current_sim_params["offline_training_data"] = offline_data_full
    # Ensure history_window_size is set correctly after loading offline data
    current_sim_params["history_window_size"] = offline_data_full.shape[0] if offline_data_full is not None else 200 # Fallback

    # --- Simulation Setup ---
    config = Config(**current_sim_params)
    initial_nn_input = np.hstack((current_sim_params['q_init'], current_sim_params['q_dot_init']))
    nn_instance = NeuralNetwork(initial_nn_input, config=current_sim_params) # Pass full params to NN if needed
    Sys = Entity(config, nn_instance)

    # --- Simulation Loop with Error Check ---
    qd_history_local = np.zeros_like(Sys.positions) # Store desired trajectory locally for early error check
    for t_step in range(1,config.total_steps):
        Sys.update_state(t_step)

        # Generate desired trajectory for current time step
        qd_current, _, _ = generate_trajectory(config.trajectory_name, config.time_steps_array[t_step], config.trajectory_params[config.trajectory_name])
        qd_history_local[:, t_step] = qd_current

        # Calculate current tracking error
        current_error_at_step = Sys.positions[:, t_step] - qd_current
        current_error_norm = np.linalg.norm(current_error_at_step)

        # Check if error exceeds threshold
        if current_error_norm > max_error_threshold:
            print(f"Trial terminated early due to large error ({current_error_norm:.4f} > {max_error_threshold}) for {current_sim_params['update_law_name']}.")
            return None # Discard this trial

    # If simulation completes without exceeding error threshold, calculate full metrics
    tracking_error = Sys.positions - qd_history_local # Use the full history now
    f_tilde = Sys.fx_history - Sys.f_hat_history

    # Calculate norms
    norm_e = np.linalg.norm(tracking_error, axis=0)
    norm_f_tilde = np.linalg.norm(f_tilde, axis=0)

    avg_tracking_error_norm = np.mean(norm_e)
    max_tracking_error_norm = np.max(norm_e)
    avg_f_tilde_error_norm = np.mean(norm_f_tilde)
    final_weight_norm = np.linalg.norm(Sys.nn.weights) # Norm of the final flattened weights

    return {
        'params': {
            'update_law_name': current_sim_params['update_law_name'],
            'alpha1': current_sim_params['controller_params']['alpha1'],
            'alpha2': current_sim_params['controller_params']['alpha2'],
            'k1': current_sim_params['controller_params']['k1'],
            'kDelta': current_sim_params['controller_params']['kDelta'],
            'gamma': current_sim_params['update_law_params']['gamma'],
            'gamma1': current_sim_params['update_law_params']['gamma1'],
            'gamma2': current_sim_params['update_law_params']['gamma2'],
            'gamma3': current_sim_params['update_law_params']['gamma3'],
            'gamma4': current_sim_params['update_law_params']['gamma4'],
            'beta_g': current_sim_params['gamma_update_law_params']['beta_g'],
        },
        'metrics': {
            'avg_tracking_error_norm': avg_tracking_error_norm,
            'max_tracking_error_norm': max_tracking_error_norm,
            'avg_f_tilde_error_norm': avg_f_tilde_error_norm,

        }
    }


# ==============================================================================
# --- USER CONFIGURATION FOR MONTE CARLO SIMULATION ---
# ==============================================================================

NUM_TRIALS_PER_UPDATE_LAW = 500 # Number of trials to run for EACH update law
SAVE_RESULTS_TO_CSV = True
OUTPUT_MC_DIR_PREFIX = "monte_carlo_results"
MAX_ERROR_THRESHOLD = 15.0 # Max allowed tracking error norm (in radians) before discarding a trial


# Define the ranges for parameters you want to vary
PARAM_RANGES = {
    'general_sim_params': {
        'update_law_name': ['CLOE', 'CLDNN1', 'CLDNN2', 'OG_DNN'] # The update laws to test
    },
    'controller_params': {
        'k1': [1,2,3,4,5, 10, 20, 30, 40, 50, 60],
        'alpha1': [1, 5, 10, 15, 20]
        # If you want to vary alpha2 or kDelta, add them here:
        # 'alpha2': [40, 50, 60],
        # 'kDelta': [15, 20, 25],
    },
    # Define specific parameter ranges for each update law
    'update_law_specific_ranges': {
        'CLOE': {
            'gamma': np.arange(0.1, 1.1, 0.1).tolist(),
            'gamma1': np.arange(1.0, 5, 0.1).tolist(),
            'gamma2': [0.001, 0.1, 0.001],
            'gamma3': np.arange(0.001, 0.1, 0.001).tolist(),
            'gamma4': np.arange(0.000, 0.005, 0.001).tolist()
        },
        'CLDNN1': {
            'gamma': np.arange(0.5, 1.1, 0.1).tolist(),
            'gamma1': np.arange(0.1, 0.5, 0.1).tolist(),
            'gamma2': [0.005, 0.01, 0.001],
            'gamma3': np.arange(0.000, 0.005, 0.001).tolist(),
            'gamma4': [0.0], # Explicitly 0.0 
        },
        'CLDNN2': {
            'gamma': np.arange(0.5, 1.1, 0.1).tolist(),
            'gamma1': np.arange(0.1, 0.5, 0.1).tolist(),
            'gamma2': [0.005, 0.01, 0.001],
            'gamma3': np.arange(0.000, 0.005, 0.001).tolist(),
            'gamma4': [0.0], # Explicitly 0.0 
        },
        'OG_DNN': {
            'gamma': np.arange(1, 10.5, 0.5).tolist(), # Changed end value to include 10
            'gamma1': [0.0], # Explicitly 0.0 for OGDNN as per example
            'gamma2': [0.0], # Explicitly 0.0 for OGDNN as per example
            'gamma3': [0.0], # Explicitly 0.0 for OGDNN as per example
            'gamma4': [0.0], # Explicitly 0.0 
        }
    }
}


# --- Initial fixed sim_params from your original script ---
# This dictionary will be deep-copied and modified for each trial
base_sim_params = {
    "q_init": np.array([7.0472, -0.5236]),  #np.array([1.0472, -0.5236]),
    "q_dot_init": np.array([0.0,0.0]),
    "state_size": 2,
    "num_inputs": 4,
    "num_outputs": 2,
    "num_layers": 4,
    "num_neurons": 2,
    "activation_functions": ["swish", "tanh"],
    "T_sim": 5.0, # Shorter for Monte Carlo to save time
    "dt": 0.001,
    "settling_time": 2.0,
    "history_window_size": 200, # Will be set to full offline data size later
    "history_update_interval": 5,
    "dynamics_func": _complex_trig_dynamics,
    "trajectory_name": 'circular',
    "trajectory_params": { # Full traj_params definition here
        'circular': { 'A': 10, 'f1': np.pi / 4, },
        'figure_eight': { 'A': 0.7, 'B': 0.7, 'f1': np.pi / 4, 'f2': np.pi / 4, },
        'multi_sinusoid': { 'A': 0.7, 'f1': np.pi / 4, 'f2': np.pi / 4, },
        'spiral': { 'A': 0.7, 'f1': np.pi / 4, },
        'growing_sinusoid': { 'A': 0.7, 'B': 0.7, 'f1': np.pi / 4, 'f2': np.pi / 4, }
    },
    "controller_params": {
        "alpha1": 15, "alpha2": 50, "k1": 40, "kDelta": 20, # Base values
    },
    "controller_name": 'nn_controller',
    "update_law_name": 'CLOE', # This will be overwritten by Monte Carlo
    "update_law_params": {
        "gamma": 0.5, "gamma1": 1.5, "gamma2": 0.01, "gamma3": 0.005, "gamma4": 0.0, # Base values
        "weight_bounds": 5,
    },
    "gamma_update_law_params": {
        "beta_g": 0.01, "lambda_min_g": 0.01, "lambda_max_g": 10.0
    },
}

# --- Load Offline Training Data ONCE ---
offline_data_file_path = r'C:\Users\rebecca.hart\OneDrive\Documents\NCR Research\[20XX_XXX] - Using Offline Learning\Sims\SimsV1\offline_data_output\offline_data_output_20250620_140817\offline_predicted_training_data.csv' # <--- CHANGE THIS PATH
offline_training_data_full = None
try:
    offline_training_data_full = np.loadtxt(offline_data_file_path, delimiter=',', skiprows=1)
    print(f"Loaded offline data with {offline_training_data_full.shape[0]} points.")
except FileNotFoundError:
    print(f"Error: Offline data file not found at {offline_data_file_path}")
    num_dummy_points = 100
    state_size = base_sim_params["state_size"]
    dummy_offline_q = np.random.rand(num_dummy_points, state_size) * 2 - 1
    dummy_offline_q_dot = np.random.rand(num_dummy_points, state_size) * 0.5 - 0.25
    dummy_offline_f_true = np.random.rand(num_dummy_points, state_size) * 10 - 5
    offline_training_data_full = np.hstack((dummy_offline_q, dummy_offline_q_dot, dummy_offline_f_true))
    print("Using dummy offline data instead for Monte Carlo.")


# ==============================================================================
# --- MONTE CARLO SIMULATION EXECUTION ---
# ==============================================================================
print("\nStarting Monte Carlo Simulation...")
monte_carlo_start_time = datetime.datetime.now()

# Generate parameter combinations
param_combinations = []
all_update_laws = PARAM_RANGES['general_sim_params']['update_law_name']

# Calculate the TOTAL_TRIALS based on the new logic
TOTAL_TRIALS = NUM_TRIALS_PER_UPDATE_LAW * len(all_update_laws)

for selected_update_law_name in all_update_laws:
    # Get the specific update_law_params ranges for the current update law
    specific_update_law_ranges = PARAM_RANGES['update_law_specific_ranges'][selected_update_law_name]

    for _ in range(NUM_TRIALS_PER_UPDATE_LAW): # Run specified number of trials for this update law
        trial_params = copy.deepcopy(base_sim_params)
        trial_params['update_law_name'] = selected_update_law_name # Explicitly set update law for this trial

        # Randomly select parameters from controller_params
        for param_name, values_list in PARAM_RANGES['controller_params'].items():
            trial_params['controller_params'][param_name] = np.random.choice(values_list)

        # Randomly select parameters from the specific update_law_params ranges
        for param_name, values_list in specific_update_law_ranges.items():
            # Handle empty lists (e.g., if a gamma parameter is fixed at 0.0 as a list [0.0])
            if not values_list:
                # If values_list is empty, it means the parameter is effectively not varied,
                # so we can skip selecting it. Its base_sim_params value will persist.
                # Or, if you intend for it to be a specific single value, ensure values_list
                # contains that single value, e.g., [0.0].
                continue
            trial_params['update_law_params'][param_name] = np.random.choice(values_list)

        param_combinations.append(trial_params)

# Update NUM_MONTE_CARLO_TRIALS to reflect the actual total
NUM_MONTE_CARLO_TRIALS = TOTAL_TRIALS

# Run simulations in parallel, passing the max_error_threshold
num_cores = os.cpu_count() or 1
print(f"Running {NUM_MONTE_CARLO_TRIALS} Monte Carlo trials using {num_cores} CPU cores.")

results = Parallel(n_jobs=num_cores, verbose=10)( # Added verbose=10 for progress indicator
    delayed(run_single_simulation_trial)(params, offline_training_data_full, MAX_ERROR_THRESHOLD)
    for params in param_combinations
)

monte_carlo_end_time = datetime.datetime.now()
mc_duration = (monte_carlo_end_time - monte_carlo_start_time).total_seconds()
print(f"Monte Carlo Simulation Finished in {mc_duration:.2f} seconds.")


# ==============================================================================
# --- POST-PROCESSING AND ANALYSIS OF MONTE CARLO RESULTS ---
# ==============================================================================

# Filter out discarded trials (those that returned None)
initial_num_results = len(results)
results = [r for r in results if r is not None]
num_discarded_trials = initial_num_results - len(results)

if num_discarded_trials > 0:
    print(f"\nNote: {num_discarded_trials} trials were discarded due to exceeding the maximum error threshold.")

if SAVE_RESULTS_TO_CSV:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    mc_output_dir = os.path.join(OUTPUT_MC_DIR_PREFIX, f"{OUTPUT_MC_DIR_PREFIX}_{timestamp}")
    os.makedirs(mc_output_dir, exist_ok=True)
    csv_filename = os.path.join(mc_output_dir, "monte_carlo_results.csv")

    # Prepare data for CSV
    header_list = []
    data_rows = []

    # Dynamically build header from the first *valid* result's params and metrics
    if results: # Only proceed if there are valid results
        sample_result = results[0]
        # Ensure order matches the params dict for correct CSV columns
        # Sort keys for consistent header order across runs and for easier debugging
        for key in sorted(sample_result['params'].keys()):
            header_list.append(key)
        for key in sorted(sample_result['metrics'].keys()):
            header_list.append(key)
        # header_str = ",".join(header_list) # No longer needed directly for csv.writer

        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header_list) # Write header row
            for res in results:
                row_data = []
                # Ensure order matches the header for correct CSV rows
                for key in sorted(sample_result['params'].keys()): # Use sorted keys here too
                    row_data.append(res['params'][key])
                for key in sorted(sample_result['metrics'].keys()): # Use sorted keys here too
                    row_data.append(res['metrics'][key])
                writer.writerow(row_data)
        print(f"Monte Carlo results saved to: {csv_filename}")
    else:
        print("No valid Monte Carlo results to save.")


# --- Example of basic analysis (you can expand this) ---
print("\n--- Monte Carlo Results Summary ---")
all_avg_tracking_errors = [res['metrics']['avg_tracking_error_norm'] for res in results]
all_k1 = [res['params']['k1'] for res in results]
all_gamma = [res['params']['gamma'] for res in results]
all_update_law_names = [res['params']['update_law_name'] for res in results] # Capture update law names

# Find best performing trial based on avg_tracking_error_norm
if all_avg_tracking_errors:
    best_trial_idx = np.argmin(all_avg_tracking_errors)
    best_trial_result = results[best_trial_idx]
    print(f"Best Trial (min Avg Tracking Error Norm: {best_trial_result['metrics']['avg_tracking_error_norm']:.6f}):")
    for param, value in best_trial_result['params'].items():
        print(f"  {param}: {value}")
else:
    print("No trials completed.")

# You can add more sophisticated plotting/analysis here, e.g.:
# - Scatter plots: avg_tracking_error vs. k1, avg_tracking_error vs. gamma
# - Histograms of metric distributions
# - Heatmaps if you're varying only two parameters
# - Etc.

# Example: Scatter plot of Avg Tracking Error vs. K1, colored by Gamma and Update Law
if all_avg_tracking_errors:
    plt.figure(figsize=(12, 7))
    # Using a categorical color map for update_law_names
    unique_update_laws = sorted(list(set(all_update_law_names))) # Sort for consistent coloring
    colors = plt.cm.get_cmap('tab10', len(unique_update_laws))
    color_map = {law: colors(i) for i, law in enumerate(unique_update_laws)}

    for i, law in enumerate(unique_update_laws):
        law_indices = [j for j, ul_name in enumerate(all_update_law_names) if ul_name == law]
        plt.scatter(
            np.array(all_k1)[law_indices],
            np.array(all_avg_tracking_errors)[law_indices],
            c=[color_map[law]] * len(law_indices),
            label=f'Update Law: {law}',
            alpha=0.7,
            s=50 # size of markers
        )

    plt.xlabel('K1 Gain')
    plt.ylabel('Average Tracking Error Norm')
    plt.title('Monte Carlo: Avg Tracking Error vs. K1, colored by Update Law')
    plt.legend(title='Update Law')
    plt.grid(True)
    plt.show()

# If you also want to visualize the effect of Gamma within each update law:
if all_avg_tracking_errors:
    fig, axes = plt.subplots(1, len(unique_update_laws), figsize=(6 * len(unique_update_laws), 6), sharey=True)
    if len(unique_update_laws) == 1: # Handle case of single subplot
        axes = [axes] # Make it iterable

    for i, law in enumerate(unique_update_laws):
        ax = axes[i]
        law_indices = [j for j, ul_name in enumerate(all_update_law_names) if ul_name == law]
        
        # Extract data for this update law
        k1_for_law = np.array(all_k1)[law_indices]
        avg_err_for_law = np.array(all_avg_tracking_errors)[law_indices]
        gamma_for_law = np.array(all_gamma)[law_indices]

        scatter = ax.scatter(k1_for_law, avg_err_for_law, c=gamma_for_law, cmap='viridis', s=50, alpha=0.7)
        fig.colorbar(scatter, ax=ax, label='Gamma Value')
        
        ax.set_xlabel('K1 Gain')
        if i == 0: # Only set ylabel for the first subplot
            ax.set_ylabel('Average Tracking Error Norm')
        ax.set_title(f'Avg Error for {law}')
        ax.grid(True)
    
    plt.tight_layout()
    plt.suptitle('Monte Carlo: Avg Tracking Error vs. K1 per Update Law (colored by Gamma)', y=1.02)
    plt.show()
