# -*- coding: utf-8 -*-
"""
Created on Wed Jul  9 09:28:41 2025

@author: rebecca.hart
"""

import pickle
import os
import numpy as np
import matplotlib
# Explicitly set the backend before importing pyplot
# 'Agg' is a non-interactive backend suitable for saving figures
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.animation as animation
from tqdm import tqdm

# --- Configuration for Loading ---

# **IMPORTANT: MODIFY THIS LINE** to the exact path of your .pkl file
saved_file_path = r'C:\Users\rebecca.hart\Desktop\CLOEResults\CLOEResults_20250710_102157_WeirdFig8_SinNoise_piOffline\all_simulation_data.pkl'

# --- Attempt to load the data ---
loaded_data = None
if os.path.exists(saved_file_path):
    try:
        with open(saved_file_path, 'rb') as f:
            loaded_data = pickle.load(f)

        # Unpack the loaded data
        all_simulation_results = loaded_data['all_simulation_results']
        update_law_configs = loaded_data['update_law_configs']
        base_sim_params = loaded_data['base_sim_params']
        dynam_params = loaded_data['dynam_params']
        traj_params = loaded_data['traj_params']
        offline_data_file_path = loaded_data['offline_data_file_path']
        offline_training_data_full = loaded_data['offline_training_data_full']

        print(f"Successfully loaded simulation data from: {saved_file_path}")
        print(f"Number of simulation runs loaded: {len(all_simulation_results)}")
        print(f"Update law configs loaded: {len(update_law_configs)}")

        if "disturbance" in base_sim_params and base_sim_params["disturbance"]["enabled"]:
            disturbance_type = base_sim_params["disturbance"].get("type", "N/A")
            disturbance_params = base_sim_params["disturbance"].get("params", {})
            print(f"\nNoise Status: Enabled (Type: {disturbance_type})")
            print(f"Noise Parameters: {disturbance_params}")
        else:
            print("\nNoise Status: Disabled")

    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
        # Reset loaded_data to None if loading fails to prevent plotting
        loaded_data = None
else:
    print(f"Error: Specified data file not found at {saved_file_path}")
    print("Please ensure the path is correct and the file exists.")


# def animate_trajectory_with_training_region(all_simulation_results, trajectory_type, color_map, default_colors, train_range_lower_bound, train_range_upper_bound, apply_clean_ax_style):
#     """
#     Creates and saves an animation of actual vs. desired X-Y trajectories
#     with a training region, using frame rate sampling for efficiency.

#     Args:
#         all_simulation_results (dict): Dictionary containing simulation results for each run.
#                                         Expected keys: 'time_steps_array', 'qd', 'q'.
#         trajectory_type (str): Name of the trajectory for the plot title.
#         color_map (dict): Dictionary mapping run names to specific colors.
#         default_colors (list): List of default colors to use if a run name is not in color_map.
#         train_range_lower_bound (float): Lower bound for the training region on both X and Y axes.
#         train_range_upper_bound (float): Upper bound for the training region on both X and Y axes.
#         apply_clean_ax_style (function): A helper function to apply consistent
#                                           plotting styles to an axes object.
#     """
#     print("\nSaving Trajectory Animation... This may take some time.")

#     # Increased figure size to potentially give more room for the legend
#     fig, ax = plt.subplots(figsize=(10, 11)) # Increased height from 10 to 11 (or even 12)
#     ax.set_title(f'Actual vs. Desired Trajectories for {trajectory_type} Trajectory', fontsize=plt.rcParams['axes.titlesize'])
#     ax.set_xlabel('Joint 1 Position', fontsize=plt.rcParams['axes.labelsize'])
#     ax.set_ylabel('Joint 2 Position', fontsize=plt.rcParams['axes.labelsize'])
#     ax.set_aspect('equal', adjustable='box')
#     apply_clean_ax_style(ax) # Apply consistent style

#     # Shaded region for training data sampling range (static elements)
#     ax.fill_between(
#         [train_range_lower_bound, train_range_upper_bound],
#         train_range_lower_bound,
#         train_range_upper_bound,
#         color='lightgray', alpha=0.3, label='Offline Training Region'
#     )
#     ax.plot([train_range_lower_bound, train_range_upper_bound, train_range_upper_bound, train_range_lower_bound, train_range_lower_bound],
#             [train_range_lower_bound, train_range_lower_bound, train_range_upper_bound, train_range_upper_bound, train_range_lower_bound],
#             color='gray', linestyle='--', linewidth=1.0, label='_nolegend_')

#     first_run_name = list(all_simulation_results.keys())[0]
#     qd_history_for_plot = all_simulation_results[first_run_name]['qd']
#     time_steps_array = all_simulation_results[first_run_name]['time_steps_array']
#     full_num_frames = len(time_steps_array)

#     animation_fps = 30
#     target_animation_frames = 500
#     frame_interval = max(1, full_num_frames // target_animation_frames)
#     animation_indices = np.arange(0, full_num_frames, frame_interval)
#     num_animation_frames = len(animation_indices)

#     print(f"Original simulation data points: {full_num_frames}")
#     print(f"Sampling for animation with interval: {frame_interval}")
#     print(f"Number of frames in final animation: {num_animation_frames}")

#     desired_line, = ax.plot([], [], label='Desired Trajectory', color='black', linestyle='--', linewidth=2.0)
#     desired_point, = ax.plot([], [], 'o', color='black', markersize=5, label='_nolegend_')

#     actual_lines = {}
#     actual_points = {}
#     current_color_idx = 0

#     display_names = {
#         "CLDNN1": "Baseline [21, Eqn. 44] $k_{2}=0$",
#         "CLOE": "Developed method $k_{2}=0$",
#         "CLOE-Robust": "Developed method $k_{2}=1$",
#         "OG_DNN": "Baseline Method [6] $k_{2}=0$",
#         "OG_DNN-Robust": "Baseline Method [6] $k_{2}=1$",
#     }

#     for run_name_config in all_simulation_results:
#         color_for_run = color_map.get(run_name_config, default_colors[current_color_idx % len(default_colors)])
#         if run_name_config not in color_map:
#             current_color_idx += 1

#         plot_label_name = display_names.get(run_name_config, run_name_config)
        
#         alpha_value = 1.0
#         if run_name_config in ['OG_DNN', 'OG_DNN-Robust']:
#             alpha_value = 0.25

#         actual_lines[run_name_config], = ax.plot([], [], label=plot_label_name, color=color_for_run, linestyle='-', linewidth=1.5, alpha=alpha_value)
#         actual_points[run_name_config], = ax.plot([], [], 'o', color=color_for_run, markersize=5, label='_nolegend_')

#     time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=12, verticalalignment='top')

#     # --- Custom Legend Application ---
#     desired_order_labels = [
#         'Offline Training Region',
#         'Desired Trajectory',
#         'Developed method $k_{2}=0$',
#         'Baseline [21, Eqn. 44] $k_{2}=0$',
#         'Baseline Method [6] $k_{2}=0$',
#         'Developed method $k_{2}=1$',
#         'Baseline Method [6] $k_{2}=1$',
#     ]

#     ordered_handles = []
#     ordered_labels = []
    
#     current_handles, current_labels = ax.get_legend_handles_labels()
#     label_to_handle_map = dict(zip(current_labels, current_handles))

#     for label_text in desired_order_labels:
#         if label_text in label_to_handle_map:
#             ordered_handles.append(label_to_handle_map[label_text])
#             ordered_labels.append(label_text)
#         else:
#             print(f"Warning: Legend label '{label_text}' not found in plot. Skipping.")

#     # KEY CHANGES FOR LEGEND VISIBILITY:
#     # 1. Use transform=ax.transAxes (or fig.transFigure if you want it truly outside the axes)
#     #    For consistency with your static plot, ax.transAxes is often fine.
#     # 2. Adjust bbox_to_anchor more aggressively downwards.
#     # 3. Use fig.subplots_adjust to create explicit space at the bottom.

#     # Option A: Legend relative to Axes (within figure)
#     # If the legend is still not showing, try increasing the figure height (figsize=(10, 12) or more)
#     # or adjust bbox_to_anchor
#     legend = ax.legend(handles=ordered_handles, labels=ordered_labels,
#                         loc='upper center', # Place the legend's *upper* center point
#                         bbox_to_anchor=(0.5, -0.15), # At X=0.5 (center), Y=-0.15 (below axes). Adjust Y as needed.
#                         fontsize=plt.rcParams['legend.fontsize'], ncol=2,
#                         facecolor='white', framealpha=0.8, edgecolor='none') # Added some styling for better visibility

#     # Option B: Legend relative to Figure (more robust for outside placement)
#     # If the above doesn't work, try this instead:
#     # legend = fig.legend(handles=ordered_handles, labels=ordered_labels,
#     #                     loc='lower center', # Place the legend's *lower* center point
#     #                     bbox_to_anchor=(0.5, 0.01), # At X=0.5 (center), Y=0.01 (just above bottom edge of figure)
#     #                     fontsize=plt.rcParams['legend.fontsize'], ncol=2,
#     #                     facecolor='white', framealpha=0.8, edgecolor='none')


#     # --- Create space for the legend ---
#     # Adjust the subplot parameters for a tight layout, especially the bottom margin
#     # This leaves more room at the bottom of the figure for the legend.
#     # The 'bottom' value should be large enough to accommodate your legend.
#     # You might need to fine-tune this (0.1, 0.15, 0.2 etc.)
#     plt.subplots_adjust(bottom=0.25) # Adjust this value (e.g., 0.15 to 0.3)

#     # Set initial axis limits based on the full data range
#     all_x = np.concatenate([qd_history_for_plot[0, :]] + [results['q'][0, :] for results in all_simulation_results.values()])
#     all_y = np.concatenate([qd_history_for_plot[1, :]] + [results['q'][1, :] for results in all_simulation_results.values()])

#     ax.set_xlim([all_x.min() - 0.5, all_x.max() + 0.5])
#     ax.set_ylim([all_y.min() - 0.5, all_y.max() + 0.5])

#     def update(frame_idx_in_animation):
#         actual_data_frame_index = animation_indices[frame_idx_in_animation]

#         desired_line.set_data(qd_history_for_plot[0, :actual_data_frame_index+1], qd_history_for_plot[1, :actual_data_frame_index+1])
#         desired_point.set_data([qd_history_for_plot[0, actual_data_frame_index]], [qd_history_for_plot[1, actual_data_frame_index]])

#         for run_name_config, results in all_simulation_results.items():
#             q_actual = results['q']
#             actual_lines[run_name_config].set_data(q_actual[0, :actual_data_frame_index+1], q_actual[1, :actual_data_frame_index+1])
#             actual_points[run_name_config].set_data([q_actual[0, actual_data_frame_index]], [q_actual[1, actual_data_frame_index]])

#         time_text.set_text(f'Time: {time_steps_array[actual_data_frame_index]:.2f} s')

#         all_artists = [desired_line, desired_point, time_text]
#         for run_name_config in all_simulation_results:
#             all_artists.append(actual_lines[run_name_config])
#             all_artists.append(actual_points[run_name_config])
#         return all_artists

#     progress_bar = tqdm(total=num_animation_frames, unit="frames", desc="Saving Progress")

#     def update_progress_callback(current_frame, total_frames_for_callback):
#         progress = int((current_frame / num_animation_frames) * 100)
#         progress_bar.n = progress
#         progress_bar.refresh()

#     ani = animation.FuncAnimation(fig, update, frames=num_animation_frames, interval=30, blit=False)

#     save_dir = os.path.join(os.getcwd(), "simulation_outputs")
#     os.makedirs(save_dir, exist_ok=True)
#     save_path = os.path.join(save_dir, "trajectory_animation.gif")

#     try:
#         ani.save(save_path, writer='pillow', fps=animation_fps, progress_callback=lambda i, n: update_progress_callback(i, n))
#         progress_bar.close()
#         print(f"\nAnimation saved successfully to: {save_path}")
#     except Exception as e:
#         progress_bar.close()
#         print(f"\nError saving animation: {str(e)}")
#         import traceback
#         traceback.print_exc()

#     plt.close(fig)


# --- Plotting Section (Only proceeds if data was loaded successfully) ---
if loaded_data:
    print("\n--- Proceeding to Plotting ---")

    # Determine the trajectory name from the loaded base_sim_params
    trajectory_type = base_sim_params.get("trajectory_name", "Unknown Trajectory").replace('_', ' ').title()

    # --- Global Plotting Style Configuration ---
    # Apply a clean, professional style globally
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        "text.usetex": True, # DISABLED LaTeX rendering for math text
        "font.family": "serif", # Use serif font for a more formal look
        "font.size": 18, # Base font size
        "axes.titlesize": 20, # Size for subplot titles
        "axes.labelsize": 18, # Size for axis labels
        "legend.fontsize": 15, # Size for legend text
        "xtick.labelsize": 12, # Size for x-axis tick labels
        "ytick.labelsize": 12, # Size for y-axis tick labels
        "lines.linewidth": 1.5, # Default line width
        "axes.grid": True, # Ensure grid is on
        "grid.linestyle": '--', # Dashed grid lines
        "grid.color": 'lightgray', # Light gray grid color
    })

    # --- Define a consistent set of colors and line styles for plotting ---
    colors = ['blue', 'red', 'orange', 'green', 'purple', 'cyan', 'magenta', 'yellow']
    color_map = {
        "CLDNN1": 'blue',
        "CLOE": 'red',
        "CLOE-Robust": 'orange',
        "OG_DNN": 'green',
        "OG_DNN-Robust": 'purple',
    }
    default_colors = ['blue', 'red', 'orange', 'green', 'purple', 'cyan', 'magenta', 'yellow']
    current_color_idx = 0

    line_styles = ['-', '--', ':', '-.']

    # Helper function to get line style for a given joint (for 2-joint systems)
    def get_joint_linestyle(joint_idx):
        return line_styles[joint_idx % len(line_styles)]

    # Helper function to apply common clean style elements to an axes object
    def apply_clean_ax_style(ax):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='x', direction='inout', length=5, width=1)
        ax.tick_params(axis='y', direction='inout', length=5, width=1)
        ax.grid(True, linestyle='--', alpha=0.6)

    # =============================================================
    # Comparative Plot: Tracking Error Only
    # =============================================================
    plt.figure(figsize=(10, 6))
    plt.title(f'Comparative Position Tracking Error for {trajectory_type} Trajectory', fontsize=plt.rcParams['axes.titlesize'])

    for run_name_config, results in all_simulation_results.items():
        time_steps = results['time_steps_array']
        tracking_error = results['tracking_error']

        color_for_run = color_map.get(run_name_config, default_colors[current_color_idx % len(default_colors)])
        if run_name_config not in color_map:
            current_color_idx += 1

        plt.plot(time_steps, np.rad2deg(tracking_error[0, :]),
                 label=f'Joint 1 Error - {run_name_config}',
                 color=color_for_run, linestyle=get_joint_linestyle(0), linewidth=1.5)
        plt.plot(time_steps, np.rad2deg(tracking_error[1, :]),
                 label=f'Joint 2 Error - {run_name_config}',
                 color=color_for_run, linestyle=get_joint_linestyle(1), linewidth=1.5)

    plt.xlabel('Time [s]', fontsize=plt.rcParams['axes.labelsize'])
    plt.ylabel('Tracking Error [deg]', fontsize=plt.rcParams['axes.labelsize'])
    plt.legend(loc='upper right', fontsize=plt.rcParams['legend.fontsize'], ncol=1)
    plt.grid(True, linestyle='--', alpha=0.6)
    apply_clean_ax_style(plt.gca())
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()
    
    # =============================================================
    # Comparative Plot: Actual vs. Desired X-Y Trajectories w/ Training Region
    # =============================================================
    plt.figure(figsize=(8, 8))
    plt.title(f'Actual vs. Desired Trajectories for {trajectory_type} Trajectory', fontsize=plt.rcParams['axes.titlesize'])
    ax = plt.gca()
    
    # --- Add Shaded Region for Training Data Sampling Range ---
    train_range_lower_bound = -1 * np.pi
    train_range_upper_bound = 1 * np.pi
 
    # Shade the region
    ax.fill_between(
        [train_range_lower_bound, train_range_upper_bound], # X-coords for fill
        train_range_lower_bound,                           # Y-start for fill
        train_range_upper_bound,                           # Y-end for fill
        color='lightgray', alpha=0.3, label='Offline Training Region' # Lighter color, more transparent
    )
    
    # Add a border to the shaded region
    ax.plot([train_range_lower_bound, train_range_upper_bound, train_range_upper_bound, train_range_lower_bound, train_range_lower_bound],
            [train_range_lower_bound, train_range_lower_bound, train_range_upper_bound, train_range_upper_bound, train_range_lower_bound],
            color='gray', linestyle='--', linewidth=1.0, label='_nolegend_') # _nolegend_ prevents duplicate legend entry
    
    
    first_run_name = list(all_simulation_results.keys())[0]
    qd_history_for_plot = all_simulation_results[first_run_name]['qd']
    
    ax.plot(qd_history_for_plot[0, :],
             qd_history_for_plot[1, :],
             label='Desired Trajectory',
             color='black', linestyle='--', linewidth=2.0)
    
    current_color_idx = 0
    for run_name_config, results in all_simulation_results.items():
        q_actual = results['q']
        color_for_run = color_map.get(run_name_config, default_colors[current_color_idx % len(default_colors)])
        if run_name_config not in color_map:
            current_color_idx += 1
    
        ax.plot(q_actual[0, :],
                q_actual[1, :],
                 label=f'Actual - {run_name_config}',
                 color=color_for_run, linestyle='-', linewidth=1.5)
    
    ax.set_xlabel('Joint 1 Position', fontsize=plt.rcParams['axes.labelsize'])
    ax.set_ylabel('Joint 2 Position', fontsize=plt.rcParams['axes.labelsize'])
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.3), # Position below the plot
              fontsize=plt.rcParams['legend.fontsize'], ncol=2) # Adjust ncol for horizontal layout
    ax.set_aspect('equal', adjustable='box')
    apply_clean_ax_style(ax)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()
 
   #  # Call the animation function here
   # # Call the animation function here
   #  animate_trajectory_with_training_region(
   #      all_simulation_results,
   #      trajectory_type,
   #      color_map,
   #      default_colors,
   #      train_range_lower_bound,
   #      train_range_upper_bound,
   #      apply_clean_ax_style # <--- Make sure this line is present
   #  )

    
    # =============================================================
    # Comparative Plot: Actual vs. Desired X-Y Trajectories w/ Training Region
    # =============================================================
    plt.figure(figsize=(8, 8))
    plt.title(f'Actual vs. Desired Trajectories for {trajectory_type} Trajectory', fontsize=plt.rcParams['axes.titlesize'])
    ax = plt.gca()
    
    # --- Add Shaded Region for Training Data Sampling Range ---
    train_range_lower_bound = -1*  np.pi
    train_range_upper_bound = 1 * np.pi

    # Shade the region
    ax.fill_between(
        [train_range_lower_bound, train_range_upper_bound], # X-coords for fill
        train_range_lower_bound,                            # Y-start for fill
        train_range_upper_bound,                            # Y-end for fill
        color='lightgray', alpha=0.3, label='Offline Training Region' # Lighter color, more transparent
    )
    
    # Add a border to the shaded region
    ax.plot([train_range_lower_bound, train_range_upper_bound, train_range_upper_bound, train_range_lower_bound, train_range_lower_bound],
            [train_range_lower_bound, train_range_lower_bound, train_range_upper_bound, train_range_upper_bound, train_range_lower_bound],
            color='gray', linestyle='--', linewidth=1.0, label='_nolegend_') # _nolegend_ prevents duplicate legend entry
    
    
    first_run_name = list(all_simulation_results.keys())[0]
    qd_history_for_plot = all_simulation_results[first_run_name]['qd']
    
    ax.plot(qd_history_for_plot[0, :],
             qd_history_for_plot[1, :],
             label='Desired Trajectory',
             color='black', linestyle='--', linewidth=2.0)
    
    current_color_idx = 0
    for run_name_config, results in all_simulation_results.items():
        q_actual = results['q']
        color_for_run = color_map.get(run_name_config, default_colors[current_color_idx % len(default_colors)])
        if run_name_config not in color_map:
            current_color_idx += 1
        
        ax.plot(q_actual[0, :],
                q_actual[1, :],
                 label=f'Actual - {run_name_config}',
                 color=color_for_run, linestyle='-', linewidth=1.5)
    
    ax.set_xlabel('Joint 1 Position', fontsize=plt.rcParams['axes.labelsize'])
    ax.set_ylabel('Joint 2 Position', fontsize=plt.rcParams['axes.labelsize'])
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.3), # Position below the plot
              fontsize=plt.rcParams['legend.fontsize'], ncol=2) # Adjust ncol for horizontal layout
    ax.set_aspect('equal', adjustable='box')
    apply_clean_ax_style(ax)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()
    

    # =============================================================
    # Comparative Plot: Actual vs. Desired X-Y Trajectories (3D with Time)
    # =============================================================
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    plt.title(f'Actual vs. Desired Trajectories over Time (3D) for {trajectory_type} Trajectory', fontsize=plt.rcParams['axes.titlesize'])

    first_run_name = list(all_simulation_results.keys())[0]
    qd_history_for_plot = all_simulation_results[first_run_name]['qd']
    time_steps_array = all_simulation_results[first_run_name]['time_steps_array']

    ax.plot(qd_history_for_plot[0, :],
             qd_history_for_plot[1, :],
             time_steps_array,
             label='Desired Trajectory',
             color='black', linestyle='--', linewidth=2.0)

    current_color_idx = 0
    for run_name_config, results in all_simulation_results.items():
        q_actual = results['q']
        color_for_run = color_map.get(run_name_config, default_colors[current_color_idx % len(default_colors)])
        if run_name_config not in color_map:
            current_color_idx += 1

        ax.plot(q_actual[0, :],
                q_actual[1, :],
                time_steps_array,
                label=f'Actual - {run_name_config}',
                color=color_for_run, linestyle='-', linewidth=1.5)

    ax.set_xlabel('Joint 1 Position', fontsize=plt.rcParams['axes.labelsize'])
    ax.set_ylabel('Joint 2 Position', fontsize=plt.rcParams['axes.labelsize'])
    ax.set_zlabel('Time [s]', fontsize=plt.rcParams['axes.labelsize'])
    ax.legend(loc='best', fontsize=plt.rcParams['legend.fontsize'])
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()

    # =============================================================
    # Comparative Plot: f_tilde Error
    # =============================================================
    plt.figure(figsize=(10, 6))
    plt.title('Comparative $\|\\tilde{f}\\|$ Error for ' + f'{trajectory_type} Trajectory', fontsize=plt.rcParams['axes.titlesize'])

    current_color_idx = 0
    for run_name_config, results in enumerate(all_simulation_results.items()):
        run_name_config, results = results
        time_steps = results['time_steps_array']
        f_tilde = results['f_tilde']

        color_for_run = color_map.get(run_name_config, default_colors[current_color_idx % len(default_colors)])
        if run_name_config not in color_map:
            current_color_idx += 1

        plt.plot(time_steps, f_tilde[0, :],
                 label='$\\tilde{f}_1$ - ' + f'{run_name_config}',
                 color=color_for_run, linestyle=get_joint_linestyle(0), linewidth=1.5)
        plt.plot(time_steps, f_tilde[1, :],
                 label='$\\tilde{f}_2$ - ' + f'{run_name_config}',
                 color=color_for_run, linestyle=get_joint_linestyle(1), linewidth=1.5)

    plt.xlabel('Time [s]', fontsize=plt.rcParams['axes.labelsize'])
    plt.ylabel('$\\tilde{f}$ Error', fontsize=plt.rcParams['axes.labelsize'])
    plt.legend(loc='upper right', fontsize=plt.rcParams['legend.fontsize'], ncol=1)
    apply_clean_ax_style(plt.gca())
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()
    
    # =============================================================
    # NEW: Individual Plots for f_tilde Error per Adaptation Law
    # =============================================================
    print("\n--- Plotting Individual f_tilde Error Plots ---")
    for run_name_config, results in all_simulation_results.items():
        time_steps = results['time_steps_array']
        f_tilde = results['f_tilde']

        plt.figure(figsize=(10, 6))
     
        # We'll use distinct, fixed colors for Joint 1 and Joint 2 error within each plot
        plt.plot(time_steps, f_tilde[0, :],
                 label='$\\tilde{f}_1$', color='blue', linestyle='-', linewidth=1.5)
        plt.plot(time_steps, f_tilde[1, :],
                 label='$\\tilde{f}_2$', color='red', linestyle='--', linewidth=1.5)

        plt.xlabel('Time [s]', fontsize=plt.rcParams['axes.labelsize'])
        plt.ylabel('$\\tilde{f}$ Error', fontsize=plt.rcParams['axes.labelsize'])
        # Concatenate f-string for title
        plt.title('Function Approximation Error ($\\tilde{f}$) for ' + f'{run_name_config} ({trajectory_type} Trajectory)', fontsize=plt.rcParams['axes.titlesize'])
        plt.legend(loc='upper right', fontsize=plt.rcParams['legend.fontsize'])
        apply_clean_ax_style(plt.gca())
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        plt.show()


    # =============================================================
    # PRETTY Comparative Plot: f_tilde Error (Norm)
    # =============================================================
    plt.figure(figsize=(10, 6))
    plt.title('$\|f(x,\\dot{x})-\\Phi(X,\\hat{\\theta})\|$' , fontsize=plt.rcParams['axes.titlesize'])

    # Define a mapping for display names
    # display_names = {
    #     "CLDNN1": "Baseline [21, Eqn. 44] $k_{2}=0$",
    #     "CLOE": "Developed method $k_{2}=0$",
    #     "CLOE-Robust": "Developed method $k_{2}=1$",
    #     "OG_DNN": "Baseline Method [6] $k_{2}=0$",
    #     "OG_DNN-Robust": "Baseline Method [6] $k_{2}=1$",
    # }
    # Define a mapping for display names
    display_names = {
        "CLDNN1": "Baseline [2, Eqn. 44] $k_{2}=0$",
        "CLOE": "Developed method $k_{2}=0$",
        "CLOE-Robust": "Developed method $k_{2}=1$",
        "OG_DNN": "Baseline [1] $k_{2}=0$",
        "OG_DNN-Robust": "Baseline [1] $k_{2}=1$",
    }


    # Store handles and labels to reorder them later
    lines = []
    labels = []

    current_color_idx = 0
    for run_name_config, results in all_simulation_results.items():
        time_steps = results['time_steps_array']
        f_tilde = results['f_tilde']
        f_tilde_norm = np.linalg.norm(f_tilde, axis=0)

        color_for_run = color_map.get(run_name_config, default_colors[current_color_idx % len(default_colors)])
        
        # Determine transparency based on run_name_config
        alpha_value = 1.0 # Default is fully opaque
        if run_name_config in ['OG_DNN', 'OG_DNN-Robust']:
            alpha_value = 0.25 # Make these lines 25% transparent

        if run_name_config not in color_map:
            current_color_idx += 1

        # Get the display name, defaulting to run_name_config if not found in mapping
        plot_label_name = display_names.get(run_name_config, run_name_config)
        
        # Corrected: Construct the full label string WITHOUT the norm prefix
        full_legend_label = plot_label_name 

        # Store the line object and its label
        line, = plt.plot(time_steps, f_tilde_norm,
                         label=full_legend_label, # Use the full label here
                         color=color_for_run, linestyle='-', linewidth=1.5, alpha=alpha_value)
        lines.append(line)
        labels.append(full_legend_label) # Store the full label for reordering

    plt.xlabel('Time [s]', fontsize=plt.rcParams['axes.labelsize'])
    plt.ylabel('$\|f(x,\\dot{x})-\\Phi(X,\\hat{\\theta})\|$', fontsize=plt.rcParams['axes.labelsize'])
    
    # Define the desired order of labels for the legend
    # Ensure these strings exactly match the labels generated by 'full_legend_label' (now without the prefix)
    # desired_order_labels = [
    #     'Developed method $k_{2}=0$',        # CLOE
    #     'Baseline [21, Eqn. 44] $k_{2}=0$', # CLDNN1
    #     'Baseline Method [6] $k_{2}=0$',    # OG_DNN
    #     'Developed method $k_{2}=1$',        # CLOE-Robust
    #     'Baseline Method [6] $k_{2}=1$',    # OG_DNN-Robust',
    # ]
    desired_order_labels = [
        'Developed method $k_{2}=0$',        # CLOE
        'Baseline [2, Eqn. 44] $k_{2}=0$', # CLDNN1
        'Baseline [1] $k_{2}=0$',    # OG_DNN
        'Developed method $k_{2}=1$',        # CLOE-Robust
        'Baseline [1] $k_{2}=1$',    # OG_DNN-Robust',
    ]

    # Create lists for handles and labels in the desired order
    ordered_handles = []
    ordered_labels = []
    for label_text in desired_order_labels:
        try:
            idx = labels.index(label_text)
            ordered_handles.append(lines[idx])
            ordered_labels.append(labels[idx])
        except ValueError:
            # Handle cases where a label in desired_order_labels might not exist in the plot
            print(f"Warning: Label '{label_text}' not found in plot.")

    # Pass the ordered handles and labels to plt.legend()
    plt.legend(handles=ordered_handles, labels=ordered_labels,
               loc='upper left', fontsize=plt.rcParams['legend.fontsize'], ncol=1) 
    
    apply_clean_ax_style(plt.gca())
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()
    
    # --- Added line to save the plot as SVG with additional parameters ---
    plt.savefig('f_tilde_error_norm.svg', format='svg', bbox_inches='tight', dpi=300, transparent=True)
    # --- Re-enabled plt.show() for debugging ---
    plt.show()
    
    # =============================================================
    # PRETTY Comparative Plot: f_tilde Error (Norm) - Robust Controllers Only
    # =============================================================
    plt.figure(figsize=(10, 6))
    #plt.title('$\|f(x,\\dot{x})-\\Phi(X,\\hat{\\theta})\|$' , fontsize=plt.rcParams['axes.titlesize'])
     
    # Define a mapping for display names
    display_names = {
        "CLDNN1": "Baseline [2, Eqn. 44] $k_{2}=0$",
        "CLOE": "Developed method $k_{2}=0$",
        "CLOE-Robust": "Developed method",
        "OG_DNN": "Baseline [1] $k_{2}=0$",
        "OG_DNN-Robust": "Baseline method",
    }
     
    # Define a specific color map for the two runs you want to plot
    desired_colors = {
        'CLOE-Robust': 'blue',
        'OG_DNN-Robust': 'red'
    }
     
    # Store handles and labels to reorder them later
    lines = []
    labels = []
     
    current_color_idx = 0
    for run_name_config, results in all_simulation_results.items():
        # Only plot if the run name is 'CLOE-Robust' or 'OG_DNN-Robust'
        if run_name_config in ['CLOE-Robust', 'OG_DNN-Robust']:
            time_steps = results['time_steps_array']
            f_tilde = results['f_tilde']
            f_tilde_norm = np.linalg.norm(f_tilde, axis=0)
     
            # Get the color from the new 'desired_colors' map
            color_for_run = desired_colors.get(run_name_config)
            
            # Determine transparency based on run_name_config
            alpha_value = 1.0 # Default is fully opaque
            if run_name_config in ['OG_DNN', 'OG_DNN-Robust']:
                alpha_value = 0.75 # Make these lines 25% transparent
     
            # The rest of the logic remains the same
            if run_name_config not in color_map:
                current_color_idx += 1
     
            plot_label_name = display_names.get(run_name_config, run_name_config)
            full_legend_label = plot_label_name 
     
            line, = plt.plot(time_steps, f_tilde_norm,
                             label=full_legend_label,
                             color=color_for_run, linestyle='-', linewidth=1.5, alpha=alpha_value)
            lines.append(line)
            labels.append(full_legend_label)
     
    plt.xlabel('Time [s]', fontsize=plt.rcParams['axes.labelsize'])
    plt.ylabel('$\|f(x,\\dot{x})-\\Phi(X,\\hat{\\theta})\|$', fontsize=plt.rcParams['axes.labelsize'])
     
    # Define the desired order of labels for the legend.
    # Corrected: Match the display_names exactly.
    desired_order_labels = [
        'Developed method',    # Must match the generated label for CLOE-Robust
        'Baseline method',               # Must match the generated label for OG_DNN-Robust
    ]
     
    # Create lists for handles and labels in the desired order
    ordered_handles = []
    ordered_labels = []
    for label_text in desired_order_labels:
        try:
            idx = labels.index(label_text)
            ordered_handles.append(lines[idx])
            ordered_labels.append(labels[idx])
        except ValueError:
            print(f"Warning: Label '{label_text}' not found in plot.")
     
    # Pass the ordered handles and labels to plt.legend()
    plt.legend(handles=ordered_handles, labels=ordered_labels,
               loc='best', fontsize=plt.rcParams['legend.fontsize'], ncol=1) 
     
    apply_clean_ax_style(plt.gca())
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()
     
    plt.savefig('robust_only_f_tilde_error_norm.svg', format='svg', bbox_inches='tight', dpi=300, transparent=True)
    plt.show()
    
    # =============================================================
    # Consolidated Plot for Neural Network Weights across Adaptation Laws
    # =============================================================
    num_adaptation_laws = len(all_simulation_results)
    fig, axes = plt.subplots(num_adaptation_laws, 1, figsize=(10, 4 * num_adaptation_laws), sharex=True)
    if num_adaptation_laws == 1:
        axes = [axes] # Ensure iterable

    fig.suptitle('Neural Network Weight Convergence for Each Adaptation Law (' + f'{trajectory_type} Trajectory)', fontsize=plt.rcParams['axes.titlesize'])

    weight_colors = cm.get_cmap('tab10', 10).colors
    weight_line_styles = ['-', '--', ':', '-.']

    for i, (run_name_config, results) in enumerate(all_simulation_results.items()):
        time_steps = results['time_steps_array']
        weights = results['weights_history']
        num_weights = weights.shape[0]

        ax = axes[i]
        for w_idx in range(num_weights):
            ax.plot(time_steps, weights[w_idx, :],
                    label=f'Weight {w_idx+1}', # Keeping labels here for completeness if you ever want them back
                    color=weight_colors[w_idx % len(weight_colors)],
                    linestyle=weight_line_styles[w_idx % len(weight_line_styles)], linewidth=1)

        ax.set_title('Weights for: ' + f'{run_name_config}', fontsize=plt.rcParams['axes.titlesize'] * 0.9)
        ax.set_ylabel('Weight Value', fontsize=plt.rcParams['axes.labelsize'])
        # ax.legend(loc='best', fontsize=plt.rcParams['legend.fontsize'], ncol=2) # REMOVED THIS LINE
        apply_clean_ax_style(ax)

    axes[-1].set_xlabel('Time [s]', fontsize=plt.rcParams['axes.labelsize'])
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()

    # =============================================================
    # Comparative Plot: Minimum Eigenvalue of grad_hist_sum_history
    # =============================================================
    plt.figure(figsize=(10, 6))
    plt.title('Comparative Evolution of $\\lambda_{min}(\\sum \\Phi^{\\prime T} \\Phi^{\\prime})$ for ' + f'{trajectory_type} Trajectory', fontsize=plt.rcParams['axes.titlesize'])

    current_color_idx = 0
    for i, (run_name_config, results) in enumerate(all_simulation_results.items()):
        min_eig_values = results['min_eig_grad_hist_sum_history']
        time_steps_array = results['time_steps_array']

        color_for_run = color_map.get(run_name_config, default_colors[current_color_idx % len(default_colors)])
        if run_name_config not in color_map:
            current_color_idx += 1

        plt.plot(time_steps_array, min_eig_values,
                 label='$\\lambda_{min}(\\sum \\Phi^{\\prime T} \\Phi^{\\prime})$ - ' + f'{run_name_config}',
                 color=color_for_run,
                 linestyle=line_styles[i % len(line_styles)], linewidth=1.5)

    plt.xlabel('Time [s]', fontsize=plt.rcParams['axes.labelsize'])
    plt.ylabel('Minimum Eigenvalue of $\\sum \\Phi^{\\prime T} \\Phi^{\\prime}$', fontsize=plt.rcParams['axes.labelsize'])
    plt.legend(loc='upper right', fontsize=plt.rcParams['legend.fontsize'], ncol=1)
    apply_clean_ax_style(plt.gca())
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()


 

    # =============================================================
    # Consolidated Plot: f_actual vs f_hat for Each Update Law and Joint
    # =============================================================
    num_adaptation_laws = len(all_simulation_results)
    state_size = all_simulation_results[list(all_simulation_results.keys())[0]]['f_actual'].shape[0]

    fig, axes = plt.subplots(num_adaptation_laws, state_size, figsize=(10 * state_size, 4 * num_adaptation_laws), sharex=True)
    if num_adaptation_laws == 1 and state_size == 1:
        axes = np.array([[axes]])
    elif num_adaptation_laws == 1:
        axes = np.expand_dims(axes, axis=0)
    elif state_size == 1:
        axes = np.expand_dims(axes, axis=1)

    fig.suptitle('Actual $f(x)$ vs. Estimated $\\hat{f}(x)$ Over Time (' + f'{trajectory_type} Trajectory)', fontsize=plt.rcParams['axes.titlesize'])

    current_color_idx_fact_fhat = 0

    for i, (run_name_config, results) in enumerate(all_simulation_results.items()):
        time_steps = results['time_steps_array']
        f_actual = results['f_actual']
        f_hat = results['f_hat']

        color_for_run = color_map.get(run_name_config, default_colors[current_color_idx_fact_fhat % len(default_colors)])
        if run_name_config not in color_map:
            current_color_idx_fact_fhat += 1


        for joint_idx in range(state_size):
            ax = axes[i, joint_idx]

            plt.plot(time_steps, f_actual[joint_idx, :],
                             label=f'$f_{{{joint_idx+1}}}(x)$ Actual',
                             color=color_for_run, linestyle='-', linewidth=1.5)

            ax.plot(time_steps, f_hat[joint_idx, :],
                             label='$\\hat{f}_{' + f'{joint_idx+1}' + '}(x)$ Estimated',
                             color=color_for_run, linestyle=':', linewidth=1.5, alpha=0.8)

            ax.set_title(f'{run_name_config}: Joint {joint_idx+1} Dynamics', fontsize=plt.rcParams['axes.titlesize'] * 0.9)
            ax.set_ylabel('Value', fontsize=plt.rcParams['axes.labelsize'])
            ax.legend(loc='best', fontsize=plt.rcParams['legend.fontsize'])
            apply_clean_ax_style(ax)

    for j in range(state_size):
        axes[-1, j].set_xlabel('Time [s]', fontsize=plt.rcParams['axes.labelsize'])

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    #plt.savefig('f_actual_vs_f_hat_plot.png')
    plt.show()

    # =============================================================
    # NEW Comparative Plot: f_tilde_integral_history
    # =============================================================
    plt.figure(figsize=(10, 6))
    plt.title('Comparative $\\int \\tilde{f} dt$ for ' + f'{trajectory_type} Trajectory', fontsize=plt.rcParams['axes.titlesize'])

    current_color_idx = 0
    for run_name_config, results in all_simulation_results.items():
        time_steps = results['time_steps_array']
        f_tilde_integral = results['f_tilde_integral_history']

        color_for_run = color_map.get(run_name_config, default_colors[current_color_idx % len(default_colors)])
        if run_name_config not in color_map:
            current_color_idx += 1

        plt.plot(time_steps, f_tilde_integral[0, :],
                 label='$\\int \\tilde{f}_1 dt$ - ' + f'{run_name_config}',
                 color=color_for_run, linestyle=get_joint_linestyle(0), linewidth=1.5)
        plt.plot(time_steps, f_tilde_integral[1, :],
                 label='$\\int \\tilde{f}_2 dt$ - ' + f'{run_name_config}',
                 color=color_for_run, linestyle=get_joint_linestyle(1), linewidth=1.5)

    plt.xlabel('Time [s]', fontsize=plt.rcParams['axes.labelsize'])
    plt.ylabel('$\\int \\tilde{f} dt$', fontsize=plt.rcParams['axes.labelsize'])
    plt.legend(loc='upper right', fontsize=plt.rcParams['legend.fontsize'], ncol=1)
    apply_clean_ax_style(plt.gca())
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()
    
   # =============================================================
    # NEW: Comparative Plot for Norms of f_hat and (tau + f_hat) with Subplots
    # =============================================================
    print("\n--- Plotting Comparative Norms of f_hat and (tau + f_hat) ---")
    
    # Create a figure with two subplots, sharing the x-axis
    fig, (ax_f_hat_norm, ax_tau_plus_f_hat_norm) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    fig.suptitle(f'Comparison of Dynamics Approximations for {trajectory_type} Trajectory', fontsize=plt.rcParams['axes.titlesize'])

    current_color_idx = 0
    # Line styles for different quantities on the same plot (e.g., solid for f_hat, dashed for tau+f_hat)
    norm_line_styles = ['-', '--', ':', '-.']

    for run_name_config, results in all_simulation_results.items():
        time_steps = results['time_steps_array']
        f_hat = results['f_hat']
        tau = results['tau']
        
        # Calculate norms
        f_hat_norm = np.linalg.norm(f_hat, axis=0)
        tau_plus_f_hat_norm = np.linalg.norm(tau + f_hat, axis=0)

        # Get color for the current run
        color_for_run = color_map.get(run_name_config, default_colors[current_color_idx % len(default_colors)])
        if run_name_config not in color_map:
            current_color_idx += 1
        
        # Plot f_hat norm on the top subplot
        ax_f_hat_norm.plot(time_steps, f_hat_norm,
                label=r'$\|\hat{f}(x)\|_2$ - ' + f'{run_name_config}',
                color=color_for_run,
                linestyle='-', # Solid line for f_hat norm
                linewidth=1.5)

        # Plot (tau + f_hat) norm on the bottom subplot
        ax_tau_plus_f_hat_norm.plot(time_steps, tau_plus_f_hat_norm,
                label=r'$\|\tau + \hat{f}(x)\|_2$ - ' + f'{run_name_config}',
                color=color_for_run,
                linestyle='--', # Dashed line for tau + f_hat norm
                linewidth=1.5)

    # Set titles and labels for the subplots
    ax_f_hat_norm.set_title(r'Norm of Estimated Dynamics $\|\hat{f}(x)\|$', fontsize=plt.rcParams['axes.titlesize'] * 0.9)
    ax_f_hat_norm.set_ylabel('Norm Value', fontsize=plt.rcParams['axes.labelsize'])
    ax_f_hat_norm.legend(loc='best', fontsize=plt.rcParams['legend.fontsize'])
    apply_clean_ax_style(ax_f_hat_norm)

    ax_tau_plus_f_hat_norm.set_title(r'Norm of Combined Control Input $\|\tau + \hat{f}(x)\|$', fontsize=plt.rcParams['axes.titlesize'] * 0.9)
    ax_tau_plus_f_hat_norm.set_xlabel('Time [s]', fontsize=plt.rcParams['axes.labelsize'])
    ax_tau_plus_f_hat_norm.set_ylabel('Norm Value', fontsize=plt.rcParams['axes.labelsize'])
    ax_tau_plus_f_hat_norm.legend(loc='best', fontsize=plt.rcParams['legend.fontsize'])
    apply_clean_ax_style(ax_tau_plus_f_hat_norm)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Adjust rect to make space for suptitle and bottom labels
    plt.show() # Display the plot

    # =============================================================
    # NEW: Comparative Plot for Norms of f_hat and (tau - f_hat) with Subplots
    # =============================================================
    print("\n--- Plotting Comparative Norms of f_hat and (tau - f_hat) ---")
    
    # Create a figure with two subplots, sharing the x-axis
    fig, (ax_f_hat_norm, ax_tau_minus_f_hat_norm) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Updated suptitle to reflect (tau - f_hat)
    fig.suptitle(f'Comparison of Dynamics Approximations for {trajectory_type} Trajectory', fontsize=plt.rcParams['axes.titlesize'])

    current_color_idx = 0

    for run_name_config, results in all_simulation_results.items():
        time_steps = results['time_steps_array']
        f_hat = results['f_hat']
        tau = results['tau']
        
        # Calculate norms for f_hat and (tau - f_hat)
        f_hat_norm = np.linalg.norm(f_hat, axis=0)
        tau_minus_f_hat_norm = np.linalg.norm(tau - f_hat, axis=0) # Changed to tau - f_hat

        # Get color for the current run
        color_for_run = color_map.get(run_name_config, default_colors[current_color_idx % len(default_colors)])
        if run_name_config not in color_map:
            current_color_idx += 1
        
        # Plot f_hat norm on the top subplot
        ax_f_hat_norm.plot(time_steps, f_hat_norm,
                label=r'$\|\hat{f}(x)\|_2$ - ' + f'{run_name_config}',
                color=color_for_run,
                linestyle='-', # Solid line for f_hat norm
                linewidth=1.5)

        # Plot (tau - f_hat) norm on the bottom subplot
        ax_tau_minus_f_hat_norm.plot(time_steps, tau_minus_f_hat_norm,
                label=r'$\|\tau - \hat{f}(x)\|_2$ - ' + f'{run_name_config}', # Label changed to tau - f_hat
                color=color_for_run,
                linestyle='--', # Dashed line for tau - f_hat norm
                linewidth=1.5)

    # Set titles and labels for the subplots
    ax_f_hat_norm.set_title(r'Norm of Estimated Dynamics $\|\hat{f}(x)\| $', fontsize=plt.rcParams['axes.titlesize'] * 0.9) # Title changed
    ax_f_hat_norm.set_ylabel('Norm Value', fontsize=plt.rcParams['axes.labelsize'])
    ax_f_hat_norm.legend(loc='best', fontsize=plt.rcParams['legend.fontsize'])
    apply_clean_ax_style(ax_f_hat_norm)

    ax_tau_minus_f_hat_norm.set_title(r'Norm of Combined Control Input $\|\tau - \hat{f}(x)\|$', fontsize=plt.rcParams['axes.titlesize'] * 0.9) # Title changed
    ax_tau_minus_f_hat_norm.set_xlabel('Time [s]', fontsize=plt.rcParams['axes.labelsize'])
    ax_tau_minus_f_hat_norm.set_ylabel('Norm Value', fontsize=plt.rcParams['axes.labelsize'])
    ax_tau_minus_f_hat_norm.legend(loc='best', fontsize=plt.rcParams['legend.fontsize'])
    apply_clean_ax_style(ax_tau_minus_f_hat_norm)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Adjust rect to make space for suptitle and bottom labels
    plt.show() # Display the plot
         
    # =============================================================
    # RMS Values
    # =============================================================

    print("\n--- Comparative RMS Error Results ---")
    rms_results_table = []
    # Added new column for the divided value
    header = ["Update Law", "RMS Tracking Error", "RMS f_tilde Error", "RMS f_tilde_integral_history", "RMS tau", "RMS f_hat", "RMS_robust_tau", "%Adaptive", "Gamma5", "RMS Int History / Gamma5"]

    def calculate_rms(data_array):
        flattened_data = data_array.flatten()
        cleaned_data = flattened_data[~np.isnan(flattened_data)]
        if cleaned_data.size == 0:
            return np.nan
        return np.sqrt(np.mean(cleaned_data**2))


    for run_name_config, results in all_simulation_results.items():
        rms_tracking_error = calculate_rms(results['tracking_error'])
        rms_f_tilde = calculate_rms(results['f_tilde'])
        rms_f_tilde_integral_history = calculate_rms(results['f_tilde_integral_history'])
        rms_tau = calculate_rms(results['tau'])
        rms_f_hat = calculate_rms(results['f_hat'])
        rms_robust_tau = calculate_rms(results['tau']+results['f_hat'])
        AdaptPercent = rms_f_hat/rms_tau

        # Hardcode gamma5_value as a float
        gamma5_numeric = np.nan # Initialize as NaN for non-CLOE runs
        gamma5_display = "N/A" 
        gamma5_numeric = 0.0001 # Hardcoded numeric value for CLOE
        gamma5_display = f"{gamma5_numeric:.4f}" # Formatted for display
        
        # Calculate rms_f_tilde_integral_history divided by gamma5_numeric
        rms_int_hist_divided_by_gamma5 = np.nan
        if not np.isnan(gamma5_numeric) and gamma5_numeric != 0:
            rms_int_hist_divided_by_gamma5 = rms_f_tilde_integral_history / gamma5_numeric

        row_data = {
            "Update Law": run_name_config,
            "RMS Tracking Error": f"{rms_tracking_error:.4f}",
            "RMS f_tilde Error": f"{rms_f_tilde:.4f}",
            "RMS f_tilde_integral_history": f"{rms_f_tilde_integral_history:.4f}",
            "RMS tau": f"{rms_tau:.4f}",
            "RMS f_hat": f"{rms_f_hat:.4f}",
            "RMS robust_tau": f"{rms_robust_tau:.4f}",
            "% Adaptive": f"{AdaptPercent:.4f}",
            "Gamma5": gamma5_display, 
            "RMS Int History / Gamma5": f"{rms_int_hist_divided_by_gamma5:.4f}" if not np.isnan(rms_int_hist_divided_by_gamma5) else "N/A",
        }
        rms_results_table.append(row_data)

    # Adjusted formatting string for the new column
    # Increased total length and adjusted column widths as needed for clarity
    print("{:<20} {:<25} {:<20} {:<30} {:<15} {:<15} {:<15} {:<15} {:<15} {:<28}".format(*header))
    print("-" * 168) # Increased separator length to match new column

    for row in rms_results_table:
        print("{:<20} {:<25} {:<20} {:<30} {:<15} {:<15}  {:<15} {:<15} {:<15} {:<28}".format(
            row["Update Law"], row["RMS Tracking Error"], row["RMS f_tilde Error"],
            row["RMS f_tilde_integral_history"], row["RMS tau"], row["RMS f_hat"], 
            row["RMS robust_tau"], row["% Adaptive"], row["Gamma5"],
            row["RMS Int History / Gamma5"]))
        