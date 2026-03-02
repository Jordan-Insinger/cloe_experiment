# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 12:03:18 2025

@author: rebecca.hart
"""

import numpy as np
from cloe_experiment.UpdateLaws import get_weights_dot
from cloe_experiment.GammaUpdateLaws import get_gamma_dot

class NeuralNetwork:
    def __init__(self, nn_input, config):
        # General parameters
        self.config = config        
        self.time_step_delta = config['dt']
        self.time_steps = int(config['T_sim'] / self.time_step_delta)
        self.nn_input = nn_input

        # Neural network parameters
        self.num_layers = config['num_layers']
        self.num_neurons = config['num_neurons']
        self.num_inputs = config['num_inputs']
        self.num_outputs = config['num_outputs']
        
        if 'activation_functions' not in config or len(config['activation_functions']) != 2:
            raise ValueError(
                "The 'activation_functions' parameter must be a list containing "
                "exactly two strings: [inner_layer_function, outer_layer_function]."
            )
        self.activation_functions = config['activation_functions']
        
        # Neural network weights
        self.initialize_weights()
        self.neural_network_gradient_wrt_weights = None
        
        gamma_gain = self.config['update_law_params']['gamma']
        num_weights = self.weights.size  # Get the total number of weights
        self.learning_rate =  gamma_gain * np.eye(num_weights)
        self.weight_bounds = self.config['update_law_params']['weight_bounds']
        self.gamma1 = self.config['update_law_params']['gamma1']
        self.gamma2 = self.config['update_law_params']['gamma2']
        self.gamma3 = self.config['update_law_params']['gamma3']
        self.gamma4 = self.config['update_law_params']['gamma4']
        self.gamma5 = self.config['update_law_params']['gamma5']
        
        self.last_history_stack_sum = np.zeros_like(self.weights)
        self.last_grad_hist_sum = np.zeros_like(self.learning_rate)
        self.last_grad_f_tilde_sum = np.zeros_like(self.weights) #
        self.last_grad_f_tilde_dot_sum = np.zeros_like(self.weights) #
        self.last_cumulative_f_tilde_integral_at_point = np.zeros_like(self.num_outputs)
        
        

    # def initialize_weights(self):
    #     np.random.seed(0)
    #     weights_list = []
    #     weights_list.append(self.kaiming_he_initialization(self.num_inputs, self.num_neurons))
    #     for _ in range(self.num_layers - 1):
    #         weights_list.append(self.kaiming_he_initialization(self.num_neurons, self.num_neurons))
    #     weights_list.append(self.kaiming_he_initialization(self.num_neurons, self.num_outputs))
    #     self.weights = np.vstack(weights_list)
    #     # self.weights = np.array([1,...,n]).reshape(-1,1)
        
    # def kaiming_he_initialization(self, input_size, output_size):
    #     return np.random.normal(0, np.sqrt(2 / input_size), output_size * (input_size + 1)).reshape(-1,1)
    
    def initialize_weights(self):
        np.random.seed(0)
        weights_list = []
        # Input layer to first hidden layer
        weights_list.append(self.xavier_initialization(self.num_inputs, self.num_neurons))
    
        # Hidden layers to hidden layers
        for _ in range(self.num_layers - 1):
            weights_list.append(self.xavier_initialization(self.num_neurons, self.num_neurons))
        
       # Last hidden layer to output layer
        weights_list.append(self.xavier_initialization(self.num_neurons, self.num_outputs))
    
        self.weights = np.vstack(weights_list)

    def xavier_initialization(self, input_size, output_size, distribution='normal'):
        """
        Xavier (Glorot) initialization for weights.
        Adds +1 to input_size for the bias term when calculating the fan_in for the weights.
        """
        fan_in = input_size + 1  # For weights including bias
        fan_out = output_size

        if distribution == 'uniform':
            limit = np.sqrt(6 / (fan_in + fan_out))
            # Reshape to (input_size + 1, output_size) to match your Kaiming He reshape logic
            return np.random.uniform(-limit, limit, output_size * (input_size + 1)).reshape(-1, 1)
        elif distribution == 'normal':
            std_dev = np.sqrt(2 / (fan_in + fan_out))
            # Reshape to (input_size + 1, output_size)
            return np.random.normal(0, std_dev, output_size * (input_size + 1)).reshape(-1, 1)
        else:
            raise ValueError("Distribution must be 'uniform' or 'normal'")
    
    def get_input_with_bias(self, step):
        return np.append(self.nn_input(step), 1).reshape(-1, 1)
        #return np.append(self.nn_input, 1).reshape(-1, 1)

    def construct_transposed_weight_matrices(self):
        weight_matrices = []
        current_index = 0
        biased_num_inputs = self.num_inputs + 1
        biased_num_neurons = self.num_neurons + 1
        
        # Create V1.T (num_neurons x num_inputs + 1 matrix)
        matrix = np.array(self.weights[current_index:current_index + biased_num_inputs * self.num_neurons]).reshape(biased_num_inputs, self.num_neurons, order='F')
        weight_matrices.append(matrix.T)
        current_index += biased_num_inputs * self.num_neurons

        # Create V2.T to VL.T (num_neurons x num_neurons + 1 matrices)
        for _ in range(1, self.num_layers):
            matrix = np.array(self.weights[current_index:current_index + biased_num_neurons * self.num_neurons]).reshape(biased_num_neurons, self.num_neurons, order='F')
            weight_matrices.append(matrix.T)
            current_index += biased_num_neurons * self.num_neurons

        # Create V(L+1).T (num_outputs x num_neurons + 1 matrix)
        matrix = np.array(self.weights[current_index:current_index + biased_num_neurons * self.num_outputs]).reshape(biased_num_neurons, self.num_outputs, order='F')
        weight_matrices.append(matrix.T)
        return weight_matrices
    
    def update_neural_network_weights(self, loss_signal, entity, step):
       
        # Get the name and parameters for the chosen update law from config
        law_name = self.config['update_law_name']
        law_params = self.config['update_law_params']
        
        # Call the dispatcher to get the rate of change of the weights
        weights_dot = get_weights_dot(law_name, self, loss_signal, law_params, entity, step)
        
        projected_weights_dot = self.proj(weights_dot, self.weights, self.weight_bounds)
        
        self.weights += self.time_step_delta * projected_weights_dot
        projected_weights = self.weights
        return projected_weights
        
        
    def proj(self, Theta, thetaHat, thetaBar):
        max_term = max(0.0, np.dot(thetaHat.T, thetaHat) - thetaBar**2)
        dot_term = np.dot(thetaHat.T, Theta)
        numerator = max_term**2 * (dot_term + np.sqrt(dot_term**2 + 1.0)) * thetaHat
        denominator = 2.0 * (1.0 + 2.0 * thetaBar)**2 * thetaBar**2
        return Theta - (numerator / denominator)
    
    def update_learning_rate(self, step):
        # Get the name and parameters for the chosen update law from config
        law_name = self.config['update_law_name']
        law_params = self.config['update_law_params']
        
        # Call the dispatcher to get the rate of change of the weights
        gamma_dot = get_gamma_dot(law_name, self, law_params, step)
               
        self.learning_rate += self.time_step_delta * gamma_dot

    def perform_forward_propagation(self, transposed_weight_matrices, neural_network_input_with_bias):
        activated_output_layers = [neural_network_input_with_bias]
        unactivated_output_layers = []
        activated_output = []
        for layer_index in range(0, self.num_layers + 1):
            product = transposed_weight_matrices[layer_index] @ activated_output_layers[layer_index]
            unactivated_output_layers.append(product)
            if layer_index != self.num_layers:
                # if layer_index == self.num_layers - 1:
                #     activated_output = self.apply_activation_function_and_bias(product, 'tanh')
                # else:
                #     activated_output = self.apply_activation_function_and_bias(product, 'swish')
                #activated_output_layers.append(activated_output)
                if layer_index == self.num_layers - 1:
                    activation_func = self.activation_functions[1]
                else:
                    activation_func = self.activation_functions[0]
                activated_output = self.apply_activation_function_and_bias(product, activation_func)
                activated_output_layers.append(activated_output)     
                

        return activated_output_layers, unactivated_output_layers

    def perform_backward_propagation(self, activated_output_layers, unactivated_output_layers, transposed_weight_matrices):
        neural_network_gradient_wrt_weights, product = None, None
        for layer in range(self.num_layers, -1, -1):
            if layer == self.num_layers:
                transposed_layer_outputs = activated_output_layers[layer].T
                neural_network_gradient_wrt_weights = np.kron(np.eye(self.num_outputs), transposed_layer_outputs)
                product = transposed_weight_matrices[layer] @ self.apply_activation_function_derivative_and_bias(unactivated_output_layers[layer-1], self.activation_functions[1])
            else:
                transposed_layer_outputs = activated_output_layers[layer].T
                neural_network_gradient_wrt_weights = np.hstack((product @ np.kron(np.eye(self.num_neurons), transposed_layer_outputs), neural_network_gradient_wrt_weights))

                if layer != 0:
                    product = product @ transposed_weight_matrices[layer] @ self.apply_activation_function_derivative_and_bias(unactivated_output_layers[layer-1], self.activation_functions[0])
       # self.neural_network_gradient_wrt_weights = neural_network_gradient_wrt_weights -- DON"T TURN ON
        return neural_network_gradient_wrt_weights

    def compute_neural_network_output(self, step, loss_signal, entity):
        transposed_weight_matrices = self.construct_transposed_weight_matrices()
        activated_output_layers, unactivated_output_layers = self.perform_forward_propagation(transposed_weight_matrices, self.get_input_with_bias(step))
        neural_network_gradient_wrt_weights = self.perform_backward_propagation(activated_output_layers, unactivated_output_layers, transposed_weight_matrices)
        nn_output = unactivated_output_layers[-1]
    
        # # Print transposed weight matrices
        # for i, matrix in enumerate(transposed_weight_matrices):
        #     print(f"V{i+1}.T = \n{matrix}\n")
        # # print activated output layers
        # for i, layer in enumerate(activated_output_layers):
        #     print(f"Activated output layer {i} = \n{layer}\n")
        # # print unactivated output layers
        # for i, layer in enumerate(unactivated_output_layers):
        #     print(f"Unactivated output layer {i} = \n{layer}\n")
        # # print neural network gradient w.r.t. weights
        # print(f"Neural network gradient w.r.t. weights = \n{self.neural_network_gradient_wrt_weights}\n")
        # # print neural network output
        # print(f"Neural network output = \n{nn_output}\n")

        # # Breakpoint here
        # self.update_neural_network_weights(velocity, nn_output)
        # Need to update here instead of backward propagation because then we can use backward_propagation in NN update laws bc it's a pure function now
        self.neural_network_gradient_wrt_weights = neural_network_gradient_wrt_weights
        self.update_learning_rate(step)
        self.update_neural_network_weights(loss_signal, entity, step)
        return nn_output
       

    @staticmethod
    def apply_activation_function_and_bias(x, activation_function):
       if activation_function == 'tanh': result = np.tanh(x)
       elif activation_function == 'swish': result = x * (1.0 / (1.0 + np.exp(-x)))
       elif activation_function == 'identity': result = x
       elif activation_function == 'relu': result = np.maximum(0, x)
       elif activation_function == 'sigmoid': result = 1 / (1 + np.exp(-x))
       elif activation_function == 'leaky_relu': result = np.where(x > 0, x, 0.01 * x)
       return np.vstack((result, [[1]]))

    @staticmethod
    def apply_activation_function_derivative_and_bias(x, activation_function):
       if activation_function == 'tanh': result = 1 - np.tanh(x)**2
       elif activation_function == 'swish':
           sigmoid = 1.0 / (1.0 + np.exp(-x))
           swish = x * sigmoid
           result = swish + sigmoid * (1 - swish)
       elif activation_function == 'identity': result = np.ones_like(x)
       elif activation_function == 'relu': result = (x > 0).astype(float)
       elif activation_function == 'sigmoid':
           sigmoid = 1 / (1 + np.exp(-x))
           result = sigmoid * (1 - sigmoid)
       elif activation_function == 'leaky_relu': result = np.where(x > 0, 1, 0.01)
       diag_result = np.diag(result.flatten())
       return np.vstack((diag_result, np.zeros(diag_result.shape[1])))
