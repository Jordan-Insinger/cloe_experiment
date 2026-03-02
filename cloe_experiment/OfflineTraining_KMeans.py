# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 14:05:58 2025

@author: rebecca.hart
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 15:47:57 2025

@author: rebecca.hart
"""

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans # Import KMeans
import matplotlib.pyplot as plt
import os
import datetime

# ==============================================================================
# --- USER CONFIGURATION ---
# Adjust these parameters to customize data generation, model architecture,
# training, and data saving.
# ==============================================================================

# --- General Settings ---
RANDOM_SEED = 42 # Seed for reproducibility

# --- Data Generation Parameters (Interpolation Range) ---
N_TRAIN_SAMPLES =20000 # Number of samples for the training dataset
TRAIN_RANGE_START = 0 #-2*np.pi # Start of the data generation range
TRAIN_RANGE_END = 8 #2*np.pi # End of the data generation range

# --- DNN Model Architecture ---
# List of hidden layer neuron counts (e.g., [128, 64] means two hidden layers: 128 then 64 neurons)
DNN_HIDDEN_LAYER_NEURONS = [128, 128, 64] #
DNN_OUTPUT_LAYER_NEURONS = 2 # Output dimension (fx1, fx2)
DNN_ACTIVATION_HIDDEN = 'relu' # Activation function for hidden layers
DNN_ACTIVATION_OUTPUT = 'linear' # Activation function for the output layer (implicitly 'linear' for regression)

# --- Model Compilation Parameters ---
OPTIMIZER = 'adam' # Optimizer for training (e.g., 'adam', 'sgd')
LOSS_FUNCTION = 'mse' # Loss function (Mean Squared Error)
METRICS = ['mae'] # Evaluation metrics (Mean Absolute Error)

# --- Training Parameters ---
EPOCHS = 5000 # Maximum number of training epochs
BATCH_SIZE = 32 # Number of samples per gradient update
VALIDATION_SPLIT_RATIO = 0.15 # Percentage of training data to use for validation (0.0 to 1.0)

# --- Early Stopping Callback Parameters ---
ES_MONITOR = 'val_loss' # Metric to monitor for early stopping
ES_PATIENCE = 200 # Number of epochs with no improvement after which training will be stopped
ES_MIN_DELTA = 1e-5 # Minimum change in the monitored metric to qualify as an improvement
ES_MODE = 'min' # 'min' for loss, 'max' for accuracy
ES_RESTORE_BEST_WEIGHTS = True # Restores model weights from the epoch with the best value of the monitored metric

# --- ReduceLROnPlateau Callback Parameters ---
LR_MONITOR = 'val_loss' # Metric to monitor for learning rate reduction
LR_FACTOR = 0.5 # Factor by which the learning rate will be reduced. new_lr = lr * factor
LR_PATIENCE = 50 # Number of epochs with no improvement after which learning rate will be reduced
LR_MIN_LR = 1e-7 # Lower bound on the learning rate

# --- Offline Data Saving Parameters ---
NUM_POINTS_TO_SAVE_OFFLINE_DATA = 50 # Number of random points from training range to save with predicted fx
OFFLINE_DATA_OUTPUT_DIR_PREFIX = "offline_data_output" # Prefix for the timestamped directory for saving
OFFLINE_CSV_FILENAME = "offline_predicted_training_data.csv" # Name of the CSV file to save
# New parameter: Choose sampling method
SAMPLING_METHOD = 'kmeans_diversity' # Options: 'random', 'velocity_excitation', 'kmeans_diversity'

# --- Visualization Parameters ---
NUM_PLOT_SAMPLES = 1000 # Number of samples to plot for predictions vs true values


# ==============================================================================
# --- SCRIPT EXECUTION ---
# No user changes typically needed below this line.
# ==============================================================================

# --- Set Seeds for Reproducibility (using user-defined seed) ---
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# --- Configuration for Saving (using user-defined prefix) ---
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = os.path.join(OFFLINE_DATA_OUTPUT_DIR_PREFIX, f"{OFFLINE_DATA_OUTPUT_DIR_PREFIX}_{timestamp}")
os.makedirs(output_dir, exist_ok=True)
print(f"Saving generated offline data to: {output_dir}/")


# --- 1. Data Generation: Training Set (Interpolation Range) ---
x1_train_data = np.random.uniform(TRAIN_RANGE_START, TRAIN_RANGE_END, N_TRAIN_SAMPLES)
x2_train_data = np.random.uniform(TRAIN_RANGE_START, TRAIN_RANGE_END, N_TRAIN_SAMPLES)
xdot1_train_data = np.random.uniform(TRAIN_RANGE_START, TRAIN_RANGE_END, N_TRAIN_SAMPLES)
xdot2_train_data = np.random.uniform(TRAIN_RANGE_START, TRAIN_RANGE_END, N_TRAIN_SAMPLES)


# SCALED TRIG DYNAMICS
# # Fixed scaling factor as per the _scaled_complex_trig_dynamics function
# scaling_factor = 1.0

# # Calculate fx for training using NumPy and apply the scaling factor
# fx1_train = (np.sin(x1_train_data + x2_train_data) * np.cos(xdot1_train_data - xdot2_train_data) +
#              np.cos(x1_train_data) * np.sin(x2_train_data) * np.cos(xdot1_train_data) * np.sin(xdot2_train_data)) * scaling_factor

# fx2_train = (np.cos(x1_train_data) * np.sin(x2_train_data) * np.cos(xdot1_train_data) * np.sin(xdot2_train_data) -
#              np.sin(x1_train_data + x2_train_data) * np.cos(xdot1_train_data - xdot2_train_data) * np.sin(x1_train_data)) * scaling_factor

# DUFFING DYNAMICS
beta = .35  # Linear stiffness coefficient (dominates for small x)
alpha = 0.015 # Nonlinear cubic stiffness coefficient (dominates for large x)
delta = 0.3 # Damping coefficient (proportional to velocity)

 # Dynamics Function (f(x) = -beta*x - alpha*x^3 - delta*x_dot)
 # This represents the total "natural" acceleration of the system.
fx1_train = -(beta * x1_train_data) - (alpha * x1_train_data**3) - (delta * xdot1_train_data) - (beta*xdot1_train_data**2)
fx2_train = -(beta * x2_train_data) - (alpha * x2_train_data**3) - (delta * xdot2_train_data) - (beta*xdot2_train_data**2)

X_train_raw = np.array([x1_train_data, x2_train_data, xdot1_train_data, xdot2_train_data]).T # Shape (N, 4)
fx_train_raw = np.array([fx1_train, fx2_train]).T # Shape (N, 2)


# --- 2. Data Preprocessing: Scaling ---
input_scaler = MinMaxScaler(feature_range=(0, 1))
X_train_scaled = input_scaler.fit_transform(X_train_raw)

output_scaler = MinMaxScaler(feature_range=(-1, 1))
fx_train_scaled = output_scaler.fit_transform(fx_train_raw)

print(f"X_train_raw range: {X_train_raw.min():.2f} to {X_train_raw.max():.2f}")
print(f"X_train_scaled range: {X_train_scaled.min():.2f} to {X_train_scaled.max():.2f}")

# Splitting scaled training data into training and validation sets
X_train_final, X_val_final, fx_train_final, fx_val_final = train_test_split(
    X_train_scaled, fx_train_scaled, test_size=VALIDATION_SPLIT_RATIO, random_state=RANDOM_SEED
)

print(f"\nFinal Training Data Shape: {X_train_final.shape}, {fx_train_final.shape}")
print(f"Validation Data Shape: {X_val_final.shape}, {fx_val_final.shape}")


# --- 3. Define the DNN Model ---
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(DNN_HIDDEN_LAYER_NEURONS[0], activation=DNN_ACTIVATION_HIDDEN, input_shape=(4,))) # First hidden layer
for neurons in DNN_HIDDEN_LAYER_NEURONS[1:]: # Remaining hidden layers
    model.add(tf.keras.layers.Dense(neurons, activation=DNN_ACTIVATION_HIDDEN))
model.add(tf.keras.layers.Dense(DNN_OUTPUT_LAYER_NEURONS, activation=DNN_ACTIVATION_OUTPUT)) # Output layer


# --- 4. Compile the model ---
model.compile(optimizer=OPTIMIZER, loss=LOSS_FUNCTION, metrics=METRICS)


# --- 5. Callbacks for Best Training (Early Stopping and Learning Rate Reduction) ---
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor=ES_MONITOR,
    patience=ES_PATIENCE,
    min_delta=ES_MIN_DELTA,
    mode=ES_MODE,
    verbose=1,
    restore_best_weights=ES_RESTORE_BEST_WEIGHTS
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor=LR_MONITOR,
    factor=LR_FACTOR,
    patience=LR_PATIENCE,
    min_lr=LR_MIN_LR,
    verbose=1
)


# --- 6. Train the Model ---
print("\nStarting training on INTERPOLATION range (with Early Stopping)...")
history = model.fit(X_train_final, fx_train_final,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    verbose=1,
                    validation_data=(X_val_final, fx_val_final),
                    callbacks=[early_stopping, reduce_lr])
print("Training finished.")


# --- 7. Evaluate Performance on Training/Validation Sets ---

# Evaluate on Interpolation Training Set
train_loss_interp, train_mae_interp = model.evaluate(X_train_final, fx_train_final, verbose=0)
print(f"\n--- Interpolation Range (0 to 2*pi) Performance ---")
print(f"Train Set MSE: {train_loss_interp:.6f}, MAE: {train_mae_interp:.6f}, RMSE: {np.sqrt(train_loss_interp):.6f}")

# Evaluate on Interpolation Validation Set
val_loss_interp, val_mae_interp = model.evaluate(X_val_final, fx_val_final, verbose=0)
print(f"Validation Set MSE: {val_loss_interp:.6f}, MAE: {val_mae_interp:.6f}, RMSE: {np.sqrt(val_loss_interp):.6f}")


# --- 8. Save User-Selected Predicted Data to CSV ---
print("\n--- Saving User-Selected Predicted Data to CSV ---")

selected_indices_for_save = []

if SAMPLING_METHOD == 'random':
    print(f"Randomly sampling {NUM_POINTS_TO_SAVE_OFFLINE_DATA} points for saving...")
    selected_indices_for_save = np.random.choice(len(X_train_raw), NUM_POINTS_TO_SAVE_OFFLINE_DATA, replace=False)

elif SAMPLING_METHOD == 'velocity_excitation':
    print(f"Selecting {NUM_POINTS_TO_SAVE_OFFLINE_DATA} points based on velocity magnitude excitation...")
    velocities_raw = X_train_raw[:, 2:] # q_dot1, q_dot2
    velocity_magnitudes = np.linalg.norm(velocities_raw, axis=1)
    sorted_indices = np.argsort(velocity_magnitudes)[::-1]
    selected_indices_for_save = sorted_indices[:NUM_POINTS_TO_SAVE_OFFLINE_DATA]

elif SAMPLING_METHOD == 'kmeans_diversity':
    print(f"Selecting {NUM_POINTS_TO_SAVE_OFFLINE_DATA} points based on K-Means diversity sampling...")
    
    # K-Means works best with scaled data. We can use X_train_scaled directly.
    # X_train_scaled already contains both q and q_dot scaled (4 features)
    
    # Initialize KMeans. Set n_init to avoid warnings in newer scikit-learn versions.
    # n_init='auto' means it will run KMeans multiple times with different centroid seeds
    # and choose the best result (more robust clustering).
    # random_state for reproducibility.
    kmeans = KMeans(n_clusters=NUM_POINTS_TO_SAVE_OFFLINE_DATA, random_state=RANDOM_SEED, n_init='auto')
    
    # Fit KMeans to the scaled training data
    kmeans.fit(X_train_scaled)
    
    # The centroids represent the centers of the clusters.
    # We want to find the data point from our original raw data that is closest to each centroid.
    # This ensures that the selected points are actual data points, not abstract centroids.
    
    selected_indices_for_save = []
    for i in range(NUM_POINTS_TO_SAVE_OFFLINE_DATA):
        # Get the centroid for the current cluster
        centroid = kmeans.cluster_centers_[i]
        
        # Find all points belonging to this cluster
        cluster_points_indices = np.where(kmeans.labels_ == i)[0]
        
        if len(cluster_points_indices) > 0:
            # Find the point in the cluster closest to the centroid
            # Transform centroid back to original scale if comparing to X_train_raw
            # Or, compare in scaled space (X_train_scaled)
            
            # For simplicity and correctness, let's compare in the scaled space
            # X_train_scaled[cluster_points_indices] contains the scaled points in this cluster
            distances = np.linalg.norm(X_train_scaled[cluster_points_indices] - centroid, axis=1)
            
            # Get the index of the point closest to the centroid within the cluster_points_indices list
            closest_point_in_cluster_idx = cluster_points_indices[np.argmin(distances)]
            selected_indices_for_save.append(closest_point_in_cluster_idx)
        else:
            # This case means a cluster was empty, which is rare but possible
            # if n_clusters > number of unique points or data is very sparse.
            # Fallback: just append any remaining unselected index, or a random one.
            # For now, we'll just skip it, meaning we might get fewer than NUM_POINTS_TO_SAVE_OFFLINE_DATA
            print(f"Warning: Cluster {i} has no points. May save fewer than {NUM_POINTS_TO_SAVE_OFFLINE_DATA} points.")

    # Ensure selected_indices_for_save is a NumPy array for consistent indexing
    selected_indices_for_save = np.array(selected_indices_for_save)
    
    # Handle the case where KMeans might produce fewer than expected unique points if clusters are empty
    # or if some clusters are very small and don't yield a 'closest' point effectively.
    # If len(selected_indices_for_save) < NUM_POINTS_TO_SAVE_OFFLINE_DATA, you might want a fallback
    # (e.g., fill with random points) or just accept the smaller number.
    # For this example, we assume it generally works.

else:
    raise ValueError(f"Unknown sampling method: {SAMPLING_METHOD}")

# --- Retrieve the raw data for these selected indices and get predictions ---
# This part is common to all sampling methods
X_sampled_raw = X_train_raw[selected_indices_for_save, :]

# Get model predictions for these selected input points (scaled)
X_selected_scaled = input_scaler.transform(X_sampled_raw)
fx_predictions_sampled_scaled = model.predict(X_selected_scaled)
fx_predictions_sampled_raw = output_scaler.inverse_transform(fx_predictions_sampled_scaled)

# Combine X_raw and predicted fx for saving
offline_predicted_data_to_save = np.hstack((X_sampled_raw, fx_predictions_sampled_raw))


# Define the filename and save the combined data to a CSV file
csv_filename_predicted = os.path.join(output_dir, OFFLINE_CSV_FILENAME)
header_predicted = "q1,q2,q_dot1,q_dot2,f_predicted_x1,f_predicted_x2"
np.savetxt(csv_filename_predicted, offline_predicted_data_to_save, delimiter=',', fmt='%.6f', header=header_predicted, comments='')

print(f"User-selected predicted offline data saved to: {csv_filename_predicted}")


# --- 9. Visualization of Training Progress ---
plt.figure(figsize=(14, 6))

# Plot MSE Loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train MSE')
plt.plot(history.history['val_loss'], label='Validation MSE')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True)

# Plot MAE
plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Model MAE Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
