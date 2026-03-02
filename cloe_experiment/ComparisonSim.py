# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 08:11:23 2025

@author: rebecca.hart
"""
import numpy as np
import matplotlib # Import the base matplotlib module first

import matplotlib.pyplot as plt # Now import pyplot
import matplotlib.cm as cm
import copy # To deep copy sim_params for each run
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

import os # Import os for path manipulation
import datetime # Import datetime for timestamping

# Import necessary modules from your project structure
from GeneralDynamics import _duffing_squared_dynamics
from DesiredTrajectories import generate_trajectory
from Config import Config
from Entity import Entity
from DNN_Try1 import NeuralNetwork

# --- 1. Define all simulation parameters at the top ---

# Define the robots physical parameters in a dictionary
dynam_params = {
    'TwoLink.p1': 3.473, 'TwoLink.p2': 0.196, 'TwoLink.p3': 0.242,
    'TwoLink.f1': 5.3, 'TwoLink.f2': 1.1
    }

# Define Trajectory parameters in a dictionary
traj_params = {
    'circular': {
        'A': 8,
        'f1': (np.pi/16), #(np.pi / 4)*(0.7/10),
    },
    'figure_eight': {
        'A': 8, #10, 0.7
        'B': 8, #10, 0.7
        'f1': np.pi / 16, #np.pi/4
        'f2': np.pi / (16*np.sqrt(2)), #np.pi/4
    },
    'multi_sinusoid': {
        'A': 0.7,
        'f1':  np.pi / 4,
        'f2':  np.pi / 4,
    },
    'spiral': {
        'A': 0.7,
        'f1': np.pi / 4,
    },
    'growing_sinusoid': {
        'A': 0.7,
        'B': 0.7,
        'f1':  np.pi / 4,
        'f2':  np.pi / 4,
    }
}

# Base simulation parameters - these will be copied and modified for each run
base_sim_params = {
    "q_init": np.array([7.0472, -0.5236]),
    "q_dot_init": np.array([0.0,0.0]),
    "state_size": 2,
    "num_inputs": 4,
    "num_outputs": 2,
    "num_layers": 4, #4
    "num_neurons": 2, #2
    "activation_functions": ["tanh", "identity"], # previously "tanh", "identity"
    "T_sim": 100,
    "dt": 0.01,
    "settling_time": 3.0,
    "history_window_size": 200,
    "history_update_interval": 5,
    "dynamics_func":_duffing_squared_dynamics,
    "trajectory_name": 'circular',
    "trajectory_params": traj_params,
    "disturbance": {
        "enabled": False, # Set to False to run without disturbance
        "type": "sinusoidal", # Options: "white_noise", "sinusoidal"
        "params": {
            "mean": 5,
            "std_dev": 3, # Slightly higher standard deviation
            "amplitude": 1,
            "freqency": 10, #0.5
        }
    },
    "delta_hat0": np.zeros(2), # Initial delta_hat value
    "delta_hat_int0": np.zeros(2), # Initial integral of delta_hat
    "tau0": np.zeros(2), # Initial control input

}

# Define the path to your offline data file
offline_data_file_path = r'C:\Users\rebecca.hart\OneDrive\Documents\NCR Research\[20XX_XXX] - Using Offline Learning\Sims\Sims for Dixon Draft V2\SimFiles\offline_data_output\VaryingPositive_trainingRegion_duffingSqd_delta03\offline_data_output_50pts_8_00\offline_predicted_training_data.csv'

offline_training_data_full = None
try:
    offline_training_data_full = np.loadtxt(offline_data_file_path, delimiter=',', skiprows=1)
    print(f"Loaded offline data with {offline_training_data_full.shape[0]} points.")
except FileNotFoundError:
    print(f"Error: Offline data file not found at {offline_data_file_path}")
    # Generate dummy data if the file is not found, for demonstration purposes
    num_dummy_points = 100
    state_size = base_sim_params["state_size"]
    dummy_offline_q = np.random.rand(num_dummy_points, state_size) * 2 - 1
    dummy_offline_q_dot = np.random.rand(num_dummy_points, state_size) * 0.5 - 0.25
    dummy_offline_f_true = np.random.rand(num_dummy_points, state_size) * 10 - 5
    offline_training_data_full = np.hstack((dummy_offline_q, dummy_offline_q_dot, dummy_offline_f_true))
    print("Using dummy offline data instead.")

offline_training_data_combined = offline_training_data_full
base_sim_params["offline_training_data"] = offline_training_data_combined
# Set history_window_size to the total number of loaded points for offline data
base_sim_params["history_window_size"] = offline_training_data_combined.shape[0]

# --- Define the Update Law Configurations for Comparison (moved to top) ---
# You can modify this list to change update laws and their parameters
# Add, remove, or adjust the parameters for each update law as needed for your comparison.
update_law_configs = [
    {
        "name": "CLOE", # MAKE SURE GAMMA 5 Is noted in the paper!
        "update_law_name": 'CLOE',
        "controller_name": 'nn_sgn_controller',
        "controller_params": {
            "alpha1": 1, #5 #15
            "alpha2": 0, #Not USED
            "k1": 10, #10 #40
            "k2": 0,
            "kDelta": 0,
            },
        "update_law_params": {
            "gamma": 1, # Overall Learning Gain
            "gamma1": 1, # 1000, #\kappa 2 in paper -- convergence of \tilde Y
            "gamma2": 0.005, # Sigma Mod
            "gamma3": 0.01, # 5 # \gamma_1 in paper -- related to \theta convergence
            "gamma4": 0.000, # Disturbance Gain #0.0005
            "gamma5": .0001, # \kappa 1 in paper -- Scaling factor for \tilde Y
            "weight_bounds": 10,
            
            #k_1 > 1
            #\alpha_1 > 0
            # \gamma2 > 2*gamma_3*gamma5
            # Gamma will stop updating at 1/gamma * beta / gamma 3* gamma_5
        },
        "gamma_update_law_params": {
            "beta_g": 0.1,
            "lambda_min_g": 0.0001,
            "lambda_max_g": 1000
        },
    }
 ]


# --- 2. Function definitions below parameters ---

def run_simulation(sim_params_current):
    """
    Runs a single simulation given the parameters and returns the collected results.

    Args:
        sim_params_current (dict): A dictionary containing all simulation parameters
                                   for the current run.

    Returns:
        dict: A dictionary containing the simulation results needed for plotting.
    """
    print(f"\nStarting Simulation for: {sim_params_current['update_law_name']} with params: {sim_params_current['update_law_params']}")

    # Get trajectory parameters for initial conditions calculation
    traj_name = sim_params_current["trajectory_name"]
    specific_params = sim_params_current["trajectory_params"][traj_name]

    # Generate initial desired trajectory values
    qd0, qd_dot0, _ = generate_trajectory(traj_name, 0.0, specific_params)

    # Calculate initial r_hat based on the formula
    r_hat0 = sim_params_current['q_dot_init'] - qd_dot0 + sim_params_current["controller_params"]['alpha1'] * (sim_params_current['q_init'] - qd0)
    sim_params_current["r_hat0"] = r_hat0
    # r_tilde0 is always zero at the start
    sim_params_current["r_tilde0"] = np.zeros(sim_params_current["state_size"])

    # Create an instance of the Config class with the current simulation parameters
    config = Config(**sim_params_current)
    # Prepare initial input for the Neural Network
    initial_nn_input = np.hstack((sim_params_current['q_init'], sim_params_current['q_dot_init'])) 
    # Create NN instance (assuming DNN_Try1.py defines NeuralNetwork)
    nn_instance = NeuralNetwork(initial_nn_input, config=sim_params_current)
    # Create the Entity (system) instance
    Sys = Entity(config, nn_instance)

    # --- Simulation Loop ---
    for t_step in range(1,config.total_steps):
        # Update system states using the Entity's method
        Sys.update_state(t_step)

    print("Simulation Finished.")

   # --- Prepare results for plotting ---
    # Generate the full desired trajectory history for error calculation
    qd_history = np.zeros_like(Sys.positions)
    for i, t in enumerate(config.time_steps_array):
        qd, _, _ = generate_trajectory(config.trajectory_name, t, config.trajectory_params[config.trajectory_name])
        qd_history[:, i] = qd

    # Calculate errors
    tracking_error = Sys.positions - qd_history
    f_tilde = Sys.fx_history - Sys.f_hat_history
    # Ensure delta_tilde uses delta_hat_history
    delta_tilde = Sys.acceleration - Sys.delta_hat

    # Store all necessary data from Sys and config for later plotting
    results = {
        'time_steps_array': config.time_steps_array,
        'tracking_error': tracking_error,
        'tau': Sys.tau,
        'qd': qd_history,
        'q': Sys.positions,
        'weights_history': np.array(Sys.weights_history),
        'r_tilde': Sys.r_tilde,
        'f_hat': Sys.f_hat_history,
        'f_actual': Sys.fx_history,
        'f_tilde': f_tilde,
        'delta_tilde': delta_tilde,
        'f_tilde_integral_history': Sys.f_tilde_int_history,
        'gamma_history': Sys.gamma_history, # Correct transpose for 3D array
        #'grad_hist_sum_history': Sys.grad_hist_sum_history,
        'min_eig_grad_hist_sum_history': Sys.min_eig_grad_hist_sum_history,
        
    }
    return results




# --- 3. Main execution logic ---

# Dictionary to store results from all simulation runs
all_simulation_results = {}


# --- Run Simulations for Each Configuration ---
for config_data in update_law_configs:
    sim_params_for_run = copy.deepcopy(base_sim_params)
    # Override specific parameters for the current update law configuration
    sim_params_for_run["update_law_name"] = config_data["update_law_name"]
    sim_params_for_run["update_law_params"] = config_data["update_law_params"]
    sim_params_for_run["gamma_update_law_params"] = config_data["gamma_update_law_params"]
    sim_params_for_run["controller_name"] = config_data["controller_name"]
    sim_params_for_run["controller_params"] = config_data["controller_params"]

    run_name = config_data["name"]
    # Run the simulation and store its results
    all_simulation_results[run_name] = run_simulation(sim_params_for_run)



print("\n--- Plotting Comparative Results ---")

# --- SAVE RESULTS ---


# Define the base directory for saving simulation results this will create a folder like 'C:\Users\rebecca.hart\Desktop\CLOEResults'
# or '/home/rebecca.hart/Desktop/CLOEResults' on Linux/macOS.

base_save_dir = os.path.join(os.path.expanduser("~"), "Desktop", "CLOEResults")
# Create a timestamped subdirectory within the base save directory
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# The timestamped folder name will be like 'CLOEResults_20250707_153000'
sim_results_output_dir = os.path.join(base_save_dir, f"CLOEResults_{timestamp}")
# Ensure the directory exists. This will create all necessary parent directories.
os.makedirs(sim_results_output_dir, exist_ok=True)
# Define the full path for the saved results file
pickle_filepath = os.path.join(sim_results_output_dir, "all_simulation_data.pkl")
# Create a dictionary to hold all relevant data for saving
data_to_save = {
    'all_simulation_results': all_simulation_results,
    'update_law_configs': update_law_configs,
    'base_sim_params': base_sim_params,
    'dynam_params': dynam_params,
    'traj_params': traj_params,
    'offline_data_file_path': offline_data_file_path,
    'offline_training_data_full': offline_training_data_full
}
    # Save the combined dictionary using pickle
import pickle # Ensure pickle is imported at the top of your script
with open(pickle_filepath, 'wb') as f:
    pickle.dump(data_to_save, f)
print(f"\nAll simulation results and configurations saved to: {pickle_filepath}") 


# --- Define a consistent set of colors and line styles for plotting ---
colors = ['blue', 'red', 'green', 'purple', 'orange', 'cyan', 'magenta', 'yellow']
line_styles = ['-', '--', ':', '-.']

# Helper function to get line style for a given joint (for 2-joint systems)
def get_joint_linestyle(joint_idx):
    return line_styles[joint_idx % len(line_styles)]

# =============================================================
# Comparative Plot: Tracking Error Only
# =============================================================
plt.figure(figsize=(12, 8)) # Adjusted figure size for a single plot
plt.suptitle('Comparative Position Tracking Error Over Time', fontsize=16)

# Plot Tracking Error for all runs
for i, (run_name, results) in enumerate(all_simulation_results.items()):
    time_steps = results['time_steps_array']
    tracking_error = results['tracking_error']
    plt.plot(time_steps, np.rad2deg(tracking_error[0, :]),
             label=f'Joint 1 Error - {run_name}',
             color=colors[i % len(colors)], linestyle=get_joint_linestyle(0))
    plt.plot(time_steps, np.rad2deg(tracking_error[1, :]),
             label=f'Joint 2 Error - {run_name}',
             color=colors[i % len(colors)], linestyle=get_joint_linestyle(1))

plt.xlabel('Time [s]')
plt.ylabel('Tracking Error [deg]')
plt.title('Position Tracking Error Comparison for Different Update Laws')
plt.legend(loc='best', fontsize='small', ncol=2) # Adjust legend to avoid clutter
plt.grid(True)
plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.show()


# =============================================================
# Comparative Plot: Actual vs. Desired X-Y Trajectories
# =============================================================
print("\n--- Plotting Comparative Actual vs. Desired X-Y Trajectories ---")

# Create a single figure for the X-Y plot
plt.figure(figsize=(10, 10)) # Adjusted figure size for a square X-Y plot
plt.suptitle('Actual vs. Desired X-Y Trajectories Over Time', fontsize=16)
ax = plt.gca() # Get current axes for the single subplot

# Plot desired trajectory (only need to do this once)
first_run_name = list(all_simulation_results.keys())[0]
qd_history_for_plot = all_simulation_results[first_run_name]['qd'] # Shape (2, total_steps)

# Plot q1 vs q2 for the desired trajectory
ax.plot(qd_history_for_plot[0, :], # q1 (X-axis)
         qd_history_for_plot[1, :], # q2 (Y-axis)
         label='Desired Trajectory',
         color='black', linestyle='--', linewidth=2)

# Plot actual trajectories for each update law
for i, (run_name, results) in enumerate(all_simulation_results.items()):
    q_actual = results['q'] # Shape (2, total_steps)

    # Plot q1 vs q2 for the actual trajectory
    ax.plot(q_actual[0, :], # q1 (X-axis)
            q_actual[1, :], # q2 (Y-axis)
             label=f'Actual - {run_name}',
             color=colors[i % len(colors)], linestyle='-') # Using solid line for actuals for clarity

ax.set_xlabel('Joint 1 Position') # Assuming q1 maps to X
ax.set_ylabel('Joint 2 Position') # Assuming q2 maps to Y
ax.set_title('X-Y Trajectory Comparison for Different Update Laws', fontsize=12)
ax.grid(True)
ax.legend(loc='best', fontsize='small') # Legend for all lines
ax.set_aspect('equal', adjustable='box') # Ensure X and Y axes have equal scaling

plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Adjust layout
plt.show()

# =============================================================
# Comparative Plot: Actual vs. Desired X-Y Trajectories (3D with Time)
# =============================================================
print("\n--- Plotting Comparative Actual vs. Desired X-Y Trajectories in 3D (X, Y, Time) ---")

# Create a figure and add a 3D subplot
fig = plt.figure(figsize=(12, 10)) # Adjusted figure size for 3D plot
ax = fig.add_subplot(111, projection='3d') # ADD THIS LINE: Create a 3D axes

plt.suptitle('Actual vs. Desired X-Y Trajectories Over Time (3D)', fontsize=16)

# Plot desired trajectory (only need to do this once)
first_run_name = list(all_simulation_results.keys())[0]
qd_history_for_plot = all_simulation_results[first_run_name]['qd'] # Shape (2, total_steps)
time_steps_array = all_simulation_results[first_run_name]['time_steps_array'] # Get time array

# Plot q1 vs q2 with Time on Z-axis for the desired trajectory
ax.plot(qd_history_for_plot[0, :], # q1 (X-axis)
         qd_history_for_plot[1, :], # q2 (Y-axis)
         time_steps_array,           # Time (Z-axis)
         label='Desired Trajectory',
         color='black', linestyle='--', linewidth=2)

# Plot actual trajectories for each update law
for i, (run_name, results) in enumerate(all_simulation_results.items()):
    q_actual = results['q'] # Shape (2, total_steps)
    # time_steps_array is the same for all runs, so we can reuse it

    # Plot q1 vs q2 with Time on Z-axis for the actual trajectory
    ax.plot(q_actual[0, :], # q1 (X-axis)
            q_actual[1, :], # q2 (Y-axis)
            time_steps_array,           # Time (Z-axis)
            label=f'Actual - {run_name}',
            color=colors[i % len(colors)], linestyle='-') # Using solid line for actuals for clarity

# Set labels for X, Y, and Z axes
ax.set_xlabel('Joint 1 Position') # Assuming q1 maps to X
ax.set_ylabel('Joint 2 Position') # Assuming q2 maps to Y
ax.set_zlabel('Time [s]')       # ADD THIS LINE for Z-axis label

ax.set_title('X-Y Trajectory Comparison for Different Update Laws (3D View)', fontsize=12)
ax.legend(loc='best', fontsize='small') # Legend for all lines
ax.grid(True) #

# Note: set_aspect('equal') behaves differently in 3D plots.
# If you want truly equal scaling, you might need to manually set limits and calculate aspect.
# ax.set_aspect('equal', adjustable='box') # This might not work as expected in 3D

plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Adjust layout
plt.show()

# =============================================================
# Comparative Plot: f_tilde Error
# =============================================================
plt.figure(figsize=(12, 8))
plt.suptitle('Comparative F_tilde Error Over Time', fontsize=16)

# Plot f_tilde for all runs
for i, (run_name, results) in enumerate(all_simulation_results.items()):
    time_steps = results['time_steps_array']
    f_tilde = results['f_tilde'] # f_tilde is (state_size, num_steps)

    plt.plot(time_steps, f_tilde[0, :],
             label=f'$f_{{tilde_1}}$ - {run_name}',
             color=colors[i % len(colors)], linestyle=get_joint_linestyle(0))
    plt.plot(time_steps, f_tilde[1, :],
             label=f'$f_{{tilde_2}}$ - {run_name}',
             color=colors[i % len(colors)], linestyle=get_joint_linestyle(1))

plt.xlabel('Time [s]')
plt.ylabel('$f_{tilde}$ Error')
plt.title('F_tilde Error Comparison for Different Update Laws')
plt.legend(loc='best', fontsize='small', ncol=2)
plt.grid(True)
plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.show()

# =============================================================
# Comparative Plot: f_tilde Error (Norm)
# =============================================================
plt.figure(figsize=(12, 8))
plt.suptitle('Comparative $f_{tilde}$ Error Norm Over Time', fontsize=16)

# Plot f_tilde norm for all runs
for i, (run_name, results) in enumerate(all_simulation_results.items()):
    time_steps = results['time_steps_array']
    f_tilde = results['f_tilde'] # f_tilde is (state_size, num_steps)
    
    # Calculate the L2 norm (magnitude) of f_tilde at each time step
    f_tilde_norm = np.linalg.norm(f_tilde, axis=0) # axis=0 computes norm across rows (joints) for each column (time step)

    plt.plot(time_steps, f_tilde_norm,
             label=f'$||f_{{tilde}}||_2$ - {run_name}',
             color=colors[i % len(colors)], linestyle='-') # Use a single linestyle for the norm

plt.xlabel('Time [s]')
plt.ylabel('$||f_{tilde}||_2$ Error Norm')
plt.title('F_tilde Error Norm Comparison for Different Update Laws')
plt.legend(loc='best', fontsize='small', ncol=1) # Reduced ncol as there's only one line per run
plt.grid(True)
plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.show()


# =============================================================
# Consolidated Plot for Neural Network Weights across Adaptation Laws
# =============================================================
print("\n--- Plotting All Neural Network Weights per Adaptation Law (Consolidated) ---")

num_adaptation_laws = len(all_simulation_results)

# Create a figure with one subplot row per adaptation law
# Adjust figsize as needed based on the number of adaptation laws and desired height.
fig, axes = plt.subplots(num_adaptation_laws, 1, figsize=(12, 6 * num_adaptation_laws), sharex=True)
fig.suptitle('Neural Network Weight Convergence for Each Adaptation Law', fontsize=16)

# Ensure 'axes' is an iterable, even if there's only one subplot
if num_adaptation_laws == 1:
    axes = [axes] # Wrap single axis object in a list for consistent iteration

# Define a separate set of colors/linestyles for individual weights within a subplot
# This helps distinguish between weights for a single adaptation law.
weight_colors = plt.cm.get_cmap('tab10', 10).colors # Use a colormap for more distinct colors
weight_line_styles = ['-', '--', ':', '-.'] # Cycle through these if more weights than styles

for i, (run_name, results) in enumerate(all_simulation_results.items()):
    time_steps = results['time_steps_array']
    weights = results['weights_history'] # Shape (num_weights, num_steps)
    num_weights = weights.shape[0]

    ax = axes[i] # Get the current subplot for this adaptation law

    for w_idx in range(num_weights):
        ax.plot(time_steps, weights[w_idx, :],
                label=f'Weight {w_idx+1}',
                color=weight_colors[w_idx % len(weight_colors)],
                linestyle=weight_line_styles[w_idx % len(weight_line_styles)])

    ax.set_title(f'Weights for: {run_name}', fontsize=12)
    ax.set_ylabel('Weight Value')
    ax.grid(True)
    
    #plt.xlim(0, 5) 
    

# Set common x-label for the entire figure at the bottom
axes[-1].set_xlabel('Time [s]')

plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Adjust layout to prevent overlap
plt.show()

# =============================================================
# Comparative Plot: Minimum Eigenvalue of grad_hist_sum_history
# =============================================================
plt.figure(figsize=(12, 8)) # Adjusted figure size for comparison
plt.suptitle(r'Comparative Evolution of $\lambda_{min}(\sum \Phi^{\prime T} \Phi^{\prime})$ Over Time', fontsize=16)

# Iterate through all simulation results to plot for each run
for i, (run_name, results) in enumerate(all_simulation_results.items()):
    # NO LONGER NEED TO CALCULATE EIGENVALUES HERE - THEY ARE ALREADY STORED
    min_eig_values = results['min_eig_grad_hist_sum_history'] # DIRECTLY ACCESS STORED HISTORY
    time_steps_array = results['time_steps_array']

    # --- Plot the results for the current run ---
    # The error handling for NaN/Inf is now handled in Entity.py when values are stored
    plt.plot(time_steps_array, min_eig_values,
             label=fr'$\lambda_{{min}}(\sum \Phi^{{\prime T}} \Phi^{{\prime}})$ - {run_name}',
             color=colors[i % len(colors)],
             linestyle=line_styles[i % len(line_styles)]) # Use different line styles for distinctness

plt.xlabel('Time [s]')
plt.ylabel(r'Minimum Eigenvalue of $\sum \Phi^{\prime T} \Phi^{\prime}$')
plt.title(r'Comparative Evolution of Minimum Eigenvalue Over Time')
plt.legend(loc='best', fontsize='small', ncol=1) # Adjust legend position and number of columns
plt.grid(True)
plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Adjust layout to prevent overlap with title
plt.show()

    

# =============================================================
# Consolidated Plot for Gamma Matrix Diagonals across Adaptation Laws
# (Adjusted for gamma_history storing only diagonal elements)
# =============================================================
print("\n--- Plotting Gamma Matrix Diagonals per Adaptation Law (Consolidated) ---")


num_adaptation_laws = len(all_simulation_results)

# Create a figure with one subplot row per adaptation law
# Adjust figsize as needed based on the number of adaptation laws and desired height.
fig, axes = plt.subplots(num_adaptation_laws, 1, figsize=(12, 5 * num_adaptation_laws), sharex=True)
fig.suptitle(r'Diagonal Elements of $\Gamma$ Matrix Over Time for Each Adaptation Law', fontsize=16)

# Ensure 'axes' is an iterable, even if there's only one subplot
if num_adaptation_laws == 1:
    axes = [axes] # Wrap single axis object in a list for consistent iteration

# Define a separate set of colors/linestyles for individual diagonal elements within a subplot
# Using a colormap that provides distinct colors for many lines
diagonal_colors = cm.get_cmap('tab20', 170).colors # Use 'tab20' for up to 20 distinct colors, or adjust as needed
diagonal_line_styles = ['-', '--', ':', '-.'] # Cycle through these if more diagonals than styles

for i, (run_name, results) in enumerate(all_simulation_results.items()):
    time_steps = results['time_steps_array']
    gamma_history = results['gamma_history'] # Expected shape: (num_parameters, num_stored_time_steps)

    # Determine the number of diagonal elements to plot
    # Now, num_diagonals is simply the first dimension of gamma_history
    num_diagonals = gamma_history.shape[0] # Should be 170 in your case

    ax = axes[i] # Get the current subplot for this adaptation law

    for d_idx in range(num_diagonals):
        # Extract the d_idx-th diagonal element across all time steps
        # This now directly accesses the row corresponding to the diagonal element.
        diagonal_values = gamma_history[d_idx, :]
        
        ax.plot(time_steps, diagonal_values,
                label=fr'$\Gamma_{{{d_idx+1},{d_idx+1}}}$', # Label as Gamma_11, Gamma_22, etc.
                color=diagonal_colors[d_idx % len(diagonal_colors)],
                linestyle=diagonal_line_styles[d_idx % len(diagonal_line_styles)])

    ax.set_title(fr'Diagonal Elements of $\Gamma$ for: {run_name}', fontsize=12)
    ax.set_ylabel(r'Diagonal Value of $\Gamma$')
    ax.grid(True)

# Set common x-label for the entire figure at the bottom
axes[-1].set_xlabel('Time [s]')

plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Adjust layout to prevent overlap with title
plt.savefig('gamma_diagonals_plot.png') # Save the figure
plt.show() # Display the plot


# =============================================================
# Consolidated Plot: f_hat and (Tau 0 f_hat) with Subplots per Update Law
# =============================================================
print("\n--- Plotting Estimated Dynamics and Control Component in Subplots ---")

num_adaptation_laws = len(all_simulation_results)

# Create a figure with one subplot row per adaptation law, and two columns
# sharex=True ensures all subplots share the same x-axis range for easy comparison
fig, axes = plt.subplots(num_adaptation_laws, 2, figsize=(16, 6 * num_adaptation_laws), sharex=True)
fig.suptitle(r'Comparison of $\hat{f}(x)$ and $(\tau - \hat{f}(x))$ Over Time', fontsize=16)

# Ensure 'axes' is always a 2D array, even if there's only one adaptation law
if num_adaptation_laws == 1:
    axes = np.expand_dims(axes, axis=0) # Makes a (1, 2) array if only one row

for i, (run_name, results) in enumerate(all_simulation_results.items()):
    time_steps = results['time_steps_array']
    f_hat = results['f_hat'] # Shape (state_size, num_steps)
    f_acutal = results['f_actual']
    tau = results['tau'] # Shape (state_size, num_steps)
    
    # Calculate the difference
    tau_minus_f_hat = tau + f_hat # Element-wise subtraction

    # Use the same color for both subplots of a given update law
    current_color = colors[i % len(colors)] 

    # --- Plot f_hat in the first column's subplot for the current update law ---
    ax_f_hat = axes[i, 0]
    ax_f_hat.plot(time_steps, f_hat[0, :],
                   label=fr'$\hat{{f}}_1$', # Label just the component for within-subplot legend
                   color=current_color, linestyle=get_joint_linestyle(0))
    ax_f_hat.plot(time_steps, f_hat[1, :],
                   label=fr'$\hat{{f}}_2$',
                   color=current_color, linestyle=get_joint_linestyle(1))
    
    ax_f_hat.set_title(fr'$\hat{{f}}(x)$ for: {run_name}', fontsize=12)
    ax_f_hat.set_ylabel(r'Value of $\hat{f}(x)$')
    ax_f_hat.grid(True)
    ax_f_hat.legend(loc='best', fontsize='small', ncol=1) # Adjust ncol if needed

    # --- Plot (tau - f_hat) in the second column's subplot for the current update law ---
    ax_tau_minus_f_hat = axes[i, 1]
    ax_tau_minus_f_hat.plot(time_steps, tau_minus_f_hat[0, :],
                             label=fr'$(\tau_1 + \hat{{f}}_1)$',
                             color=current_color, linestyle=get_joint_linestyle(0))
    ax_tau_minus_f_hat.plot(time_steps, tau_minus_f_hat[1, :],
                             label=fr'$(\tau_2 + \hat{{f}}_2)$',
                             color=current_color, linestyle=get_joint_linestyle(1))
    
    ax_tau_minus_f_hat.set_title(fr'$(\tau + \hat{{f}}(x))$ for: {run_name}', fontsize=12)
    ax_tau_minus_f_hat.set_ylabel(r'Value of $(\tau + \hat{f}(x))$')
    ax_tau_minus_f_hat.grid(True)
    ax_tau_minus_f_hat.legend(loc='best', fontsize='small', ncol=1) # Adjust ncol if needed

# Set common x-labels for the bottom row of subplots
for j in range(2): # For both columns (0 and 1)
    axes[-1, j].set_xlabel('Time [s]')

plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Adjust layout to prevent overlap with title
#plt.savefig('f_hat_and_tau_minus_f_hat_subplots.png') # Save with a new filename
plt.show()

# =============================================================
# Consolidated Plot: f_actual vs f_hat for Each Update Law and Joint
# =============================================================
print("\n--- Plotting Actual vs. Estimated Unknown Dynamics Comparison ---")

num_adaptation_laws = len(all_simulation_results)
state_size = all_simulation_results[list(all_simulation_results.keys())[0]]['f_actual'].shape[0] # Get 2 (for 2 joints)

# Create a figure with one subplot row per adaptation law, and 'state_size' columns
fig, axes = plt.subplots(num_adaptation_laws, state_size, figsize=(16, 5 * num_adaptation_laws), sharex=True)
fig.suptitle(r'Actual $f(x)$ vs. Estimated $\hat{f}(x)$ Over Time', fontsize=16)

# Ensure 'axes' is always a 2D array, even if there's only one adaptation law or one state_size
if num_adaptation_laws == 1 and state_size == 1: # Single subplot case
    axes = np.array([[axes]])
elif num_adaptation_laws == 1: # Single row, multiple columns
    axes = np.expand_dims(axes, axis=0)
elif state_size == 1: # Multiple rows, single column
    axes = np.expand_dims(axes, axis=1)


for i, (run_name, results) in enumerate(all_simulation_results.items()):
    time_steps = results['time_steps_array']
    f_actual = results['f_actual'] # Shape (state_size, num_steps)
    f_hat = results['f_hat']     # Shape (state_size, num_steps)

    # Choose a base color for the current update law's plots
    current_law_color = colors[i % len(colors)]

    for joint_idx in range(state_size):
        ax = axes[i, joint_idx] # Get the current subplot (row 'i', column 'joint_idx')

        # Plot f_actual
        ax.plot(time_steps, f_actual[joint_idx, :],
                 label=fr'$f_{{{joint_idx+1}}}(x)$ Actual',
                 color=current_law_color, linestyle='-', linewidth=2) # Solid, slightly thicker

        # Plot f_hat
        ax.plot(time_steps, f_hat[joint_idx, :],
                 label=fr'$\hat{{f}}_{{{joint_idx+1}}}(x)$ Estimated',
                 color=current_law_color, linestyle='--', linewidth=1.5, alpha=0.8) # Dashed, slightly transparent

        ax.set_title(fr'{run_name}: Joint {joint_idx+1} Dynamics', fontsize=11)
        ax.set_ylabel('Value')
        ax.grid(True)
        ax.legend(loc='best', fontsize='small')

# Set common x-labels for the bottom row of subplots
for j in range(state_size):
    axes[-1, j].set_xlabel('Time [s]')

plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Adjust layout to prevent overlap with title
#plt.savefig('f_actual_vs_f_hat_plot.png')
plt.show()

# =============================================================
# NEW Comparative Plot: f_tilde_integral_history
# =============================================================
plt.figure(figsize=(12, 8))
plt.suptitle(r'Comparative $\int \tilde{f} dt$ Over Time', fontsize=16)

# Plot f_tilde_integral_history for all runs
for i, (run_name, results) in enumerate(all_simulation_results.items()):
    time_steps = results['time_steps_array']
    f_tilde_integral = results['f_tilde_integral_history'] # f_tilde_integral_history is (state_size, num_steps)

    plt.plot(time_steps, f_tilde_integral[0, :],
             label=fr'$\int \tilde{{f}}_1 dt$ - {run_name}',
             color=colors[i % len(colors)], linestyle=get_joint_linestyle(0))
    plt.plot(time_steps, f_tilde_integral[1, :],
             label=fr'$\int \tilde{{f}}_2 dt$ - {run_name}',
             color=colors[i % len(colors)], linestyle=get_joint_linestyle(1))

plt.xlabel('Time [s]')
plt.ylabel(r'$\int \tilde{f} dt$')
plt.title(r'Integral of $\tilde{f}$ Comparison for Different Update Laws')
plt.legend(loc='best', fontsize='small', ncol=2)
plt.grid(True)
plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.show()

# =============================================================
# RMS Values
# =============================================================

print("\n--- Comparative RMS Error Results ---")
rms_results_table = []
# Initialize header with common fields
header = ["Update Law", "RMS Tracking Error", "RMS f_tilde Error", "RMS f_tilde_integral_history", "RMS tau"]

# Function to calculate RMS error for a given array
def calculate_rms(data_array):
    """
    Calculates the Root Mean Square (RMS) error for a given numpy array.
    Handles NaN values by ignoring them.
    """
    flattened_data = data_array.flatten()
    cleaned_data = flattened_data[~np.isnan(flattened_data)]
    if cleaned_data.size == 0:
        return np.nan # Return NaN if no valid data points
    return np.sqrt(np.mean(cleaned_data**2))


for run_name, results in all_simulation_results.items():
    # Calculate RMS for each metric
    rms_tracking_error = calculate_rms(results['tracking_error'])
    rms_f_tilde = calculate_rms(results['f_tilde'])
    rms_f_tilde_integral_history = calculate_rms(results['f_tilde_integral_history'])
    rms_tau = calculate_rms(results['tau'])

    # Create a dictionary for the current run's results
    row_data = {
        "Update Law": run_name,
        "RMS Tracking Error": f"{rms_tracking_error:.4f}", # Convert to degrees
        "RMS f_tilde Error": f"{rms_f_tilde:.4f}",
        "RMS f_tilde_integral_history": f"{rms_f_tilde_integral_history:.4f}",
        "RMS tau": f"{rms_tau:.4f}",
    }
    rms_results_table.append(row_data)

# Print table header
print("{:<20} {:<25} {:<20} {:<30} {:<15}".format(*header))
print("-" * 110) # Separator line

# Print table rows
for row in rms_results_table:
    print("{:<20} {:<25} {:<20} {:<30} {:<15}".format(
        row["Update Law"], row["RMS Tracking Error"], row["RMS f_tilde Error"],
        row["RMS f_tilde_integral_history"], row["RMS tau"]))
    
    
