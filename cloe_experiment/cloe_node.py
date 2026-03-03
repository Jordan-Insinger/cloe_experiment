# -*- coding: utf-8 -*-
"""
authors: Rebecca Hart, Jordan Insinger
Created on Mon Mar 2 2026
"""

#--------Beccas dependencies-------------#
import numpy as np
import matplotlib 
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

import copy
import os
import datetime 
import asyncio

from cloe_experiment.GeneralDynamics import _duffing_squared_dynamics
from cloe_experiment.DesiredTrajectories import generate_trajectory
from cloe_experiment.Config import Config
from cloe_experiment.Entity import Entity
from cloe_experiment.DNN_Try1 import NeuralNetwork
#-----------------------------------------#

# ROS2 / PX4 Dependencies
import rclpy 
from rclpy.node import Node
from std_srvs.srv import Empty
from geometry_msgs import PoseStamped, TwistStamped
from mavros_msgs import State, SetMode

class Cloe(Node):
    def __init__(self):
        super().__init__('cloe')

        self.initialize_base_parameters()
        self.initialize_update_law_parameters()
        self.get_offline_data()

        self.initialize_subscribers()

        self.start_experiment = False
        self.offboard_mode = False

        self.set_mode_client = self.create_client(SetMode, 'set_mode')
        self.start_experiment_srv = self.create_service(
            Empty, 'start_experiment', self.start_experiment_callback) 

        self.get_logger().info('Initialized cloe node')

    def initialize_subscribers(self):
        self.pose_sub = self.create_subscription(
                PoseStamped, 
                'autonomy_park/pose', 
                self.pose_callback, 
                qos_profile=qos_profile_sensor_data
        )

        self.vel_sub = self.create_subscription(
                TwistStamped,
                'local_position/velocity_local',
                self.velocity_callback,
                qos_profile=qos_profile_sensor_data
        )

    def initialize_base_parameters(self):
        # Define Trajectory parameters in a dictionary
        self.traj_params = {
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
        self.base_sim_params = {
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
            "trajectory_params": self.traj_params,
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

        self.get_logger().info('Initialized base parameters')
    
    def initialize_update_law_parameters(self):
        self.update_law_configs = [
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
        self.get_logger().info('Initialized update law parameters')

    def get_offline_data(self):
        offline_data_file_path = r'C:\Users\rebecca.hart\OneDrive\Documents\NCR Research\[20XX_XXX] - Using Offline Learning\Sims\Sims for Dixon Draft V2\SimFiles\offline_data_output\VaryingPositive_trainingRegion_duffingSqd_delta03\offline_data_output_50pts_8_00\offline_predicted_training_data.csv'

        offline_training_data_full = None
        try:
            offline_training_data_full = np.loadtxt(offline_data_file_path, delimiter=',', skiprows=1)
            print(f"Loaded offline data with {offline_training_data_full.shape[0]} points.")
        except FileNotFoundError:
            print(f"Error: Offline data file not found at {offline_data_file_path}")
            # Generate dummy data if the file is not found, for demonstration purposes
            num_dummy_points = 100
            state_size = self.base_sim_params["state_size"]
            dummy_offline_q = np.random.rand(num_dummy_points, state_size) * 2 - 1
            dummy_offline_q_dot = np.random.rand(num_dummy_points, state_size) * 0.5 - 0.25
            dummy_offline_f_true = np.random.rand(num_dummy_points, state_size) * 10 - 5
            offline_training_data_full = np.hstack((dummy_offline_q, dummy_offline_q_dot, dummy_offline_f_true))
            print("Using dummy offline data instead.")

        offline_training_data_combined = offline_training_data_full
        self.base_sim_params["offline_training_data"] = offline_training_data_combined
        # Set history_window_size to the total number of loaded points for offline data
        self.base_sim_params["history_window_size"] = offline_training_data_combined.shape[0]

    def start_experiment_callback(self, request, response):
        self.start_experiment = True
        return response
        
    async def set_offboard(self):
        """Set to offboard mode"""

        # Send a few setpoints before starting
        for i in range(100):
            self.send_command(0.0, 0.0, 0.0, 0.0, 0.0)
            await self.sleep(0.05)

        last_request_time = self.get_clock().now()


        while rclpy.ok():
            current_time = self.get_clock().now()
            
            if not self.offboard_mode and (current_time - last_request_time).nanoseconds > 2e9:
                self.get_logger().info("Trying to set OFFBOARD mode...")
                req = SetMode.Request()
                req.custom_mode = "OFFBOARD"
                future = self.set_mode_client.call_async(req)
                await self.spin_until_future_complete(future)
                
                if future.result().mode_sent:
                    self.get_logger().info("OFFBOARD mode set")
                    self.offboard_mode = True
                    return True
                last_request_time = self.get_clock().now()
                
            self.send_command(0.0, 0.0, 0.0, 0.0, 0.0)  # Send neutral commands while waiting
            await self.sleep(0.05)

    async def sleep(self, seconds: float) -> None:
        """Sleep while still processing callbacks"""
        start = self.get_clock().now()
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.01)
            if (self.get_clock().now() - start).nanoseconds / 1e9 > seconds:
                break

# Callback functions
    def pose_callback(self, msg: PoseStamped) -> None:
        self.position[0] = msg.pose.position.x
        self.position[1] = msg.pose.position.y
        self.position[2] = msg.pose.position.z

        quat = msg.pose.orientation
        siny_cosp = 2 * (quat.w * quat.z + quat.x * quat.y)
        cosy_cosp = 1 - 2 * (quat.y * quat.y + quat.z * quat.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        self.orientation = yaw

    def vel_callback(self, msg:TwistStamped) -> None:
        self.velocity[0] = msg.twist.linear.x
        self.velocity[1] = msg.twist.linear.y
        self.velocity[2] = msg.twist.linear.z

    async def run_trajectory(self) -> None:
        while self.start_experiment == False:
            await self.sleep(0.5)
            
        self.get_logger().info("Running Trajectory")

        self.set_offboard() # switch to offboard and run controller

        traj_start_time = self.get_clock().now()
        while rclpy.ok():
            # get current time
            t = (self.get_clock().now - traj_start_time).nanoseconds / 1e9

            # check against final sim time

            # compute control input

            # send command to px4

    async def run_experiment(self, sim_params_current) -> None:
        try:
            self.get_logger().info(f"\nStarting Simulation for: {sim_params_current['update_law_name']} with params: {sim_params_current['update_law_params']}")

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
            await self.run_trajectory()

        except Exception as e:
            self.get_logger().error(f"Error in experiment: {e}")
            # Print more details about the error
            self.get_logger().error(traceback.format_exc())

        finally:
            self.get_logger().info("Experiment Finished") # Land when done or if interrupted
            # await self.land()

def main(args=None):
    rclpy.init(args=args)
    cloe = Cloe()

    # Create the event loop
    loop = asyncio.get_event_loop()
    
    try:
        for config_data in cloe.update_law_configs:
            sim_params_for_run = copy.deepcopy(cloe.base_sim_params)
            # Override specific parameters for the current update law configuration
            sim_params_for_run["update_law_name"] = config_data["update_law_name"]
            sim_params_for_run["update_law_params"] = config_data["update_law_params"]
            sim_params_for_run["gamma_update_law_params"] = config_data["gamma_update_law_params"]
            sim_params_for_run["controller_name"] = config_data["controller_name"]
            sim_params_for_run["controller_params"] = config_data["controller_params"]
            run_name = config_data["name"]

        loop.run_until_complete(cloe.run_experiment(sim_params_for_run))
        

    except KeyboardInterrupt:
        pass
    finally:
        # Clean shutdown
        cloe.destroy_node()
        rclpy.shutdown()
        loop.close()

if __name__ == '__main__':
    main()
