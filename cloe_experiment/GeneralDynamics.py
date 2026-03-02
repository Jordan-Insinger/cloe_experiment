# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 13:50:18 2025

@author: rebecca.hart
"""

import numpy as np

def _complex_trig_dynamics(q, q_dot, tau):
    x1, x2, xdot1, xdot2 = q[0], q[1], q_dot[0], q_dot[1]
    fx1 = np.sin(x1 + x2) * np.cos(xdot1 - xdot2) + np.cos(x1) * np.sin(x2) * np.cos(xdot1) * np.sin(xdot2)
    fx2 = np.cos(x1) * np.sin(x2) * np.cos(xdot1) * np.sin(xdot2) - np.sin(x1 + x2) * np.cos(xdot1 - xdot2) * np.sin(x1)
    fx = np.array([fx1, fx2])
    fx_flat = fx.flatten()
    tau_flat = tau.flatten()
    q_dotdot = fx_flat + tau_flat

    return q_dotdot, fx_flat

def _scaled_complex_trig_dynamics(q, q_dot, tau):
    # Fixed scaling factor
    scaling_factor = 10.0

    x1, x2, xdot1, xdot2 = q[0], q[1], q_dot[0], q_dot[1]

    # Calculate the components of fx
    fx1 = np.sin(x1 + x2) * np.cos(xdot1 - xdot2) + np.cos(x1) * np.sin(x2) * np.cos(xdot1) * np.sin(xdot2)
    fx2 = np.cos(x1) * np.sin(x2) * np.cos(xdot1) * np.sin(xdot2) - np.sin(x1 + x2) * np.cos(xdot1 - xdot2) * np.sin(x1)

    # Combine into a NumPy array and apply the scaling factor
    fx = np.array([fx1, fx2]) * scaling_factor

    # Flatten fx and tau for element-wise addition
    fx_flat = fx.flatten()
    tau_flat = tau.flatten()

    # Calculate joint accelerations
    q_dotdot = fx_flat + tau_flat

    return q_dotdot, fx_flat


def _more_complex_dynamics(q, q_dot, tau):
    x1, x2, xdot1, xdot2 = q[0], q[1], q_dot[0], q_dot[1]
    fx1_initial = (np.sin(x1 + x2) * np.cos(xdot1 - xdot2) + np.cos(x1**2) * np.sin(x2**2) * np.cos(xdot1**3) * np.sin(xdot2**2))
    fx2_initial = (np.cos(x1) * np.sin(x2) * np.cos(xdot1**2) * np.sin(xdot2**3) - np.sin(x1 + x2) * np.cos(xdot1**2 - xdot2**2) * np.sin(x1))
    fx = np.array([fx1_initial, fx2_initial])
    fx[0] += np.exp(x1 * xdot1) * np.tanh(np.sin(x2) * xdot2)
    fx[1] += np.tanh(xdot2 - xdot1) * np.cos(x1 * xdot2) * np.exp(-x2**2)
    fx_flat = fx.flatten()
    tau_flat = tau.flatten()
    q_dotdot = fx_flat + tau_flat

    return q_dotdot, fx_flat
    
def _tanh_dynamics(q, q_dot, tau):
    x1, x2, xdot1, xdot2 = q[0], q[1], q_dot[0], q_dot[1]
    fx1 = x1 * xdot2 * np.tanh(x2) + (1 / np.cosh(x1))**2
    fx2 = (1 / np.cosh(xdot1 + xdot2))**2 - (1 / np.cosh(x2))**2
    fx = np.array([fx1, fx2])
    fx_flat = fx.flatten()
    tau_flat = tau.flatten()
    q_dotdot = fx_flat + tau_flat

    return q_dotdot, fx_flat

def _saturation_dynamics(q, q_dot, tau):
    # Unpack state variables (q represents position, q_dot represents velocity)
    x1, x2, xdot1, xdot2 = q[0], q[1], q_dot[0], q_dot[1]

    # Model Parameters
    r = 1.0   # Base linear growth rate (for small x)
    K = 1.0  # Saturation/Carrying Capacity (Limiting magnitude)


    fx1 = r * x1 * (1.0 - np.abs(x1) / K)
    fx2 = r * x2 * (1.0 - np.abs(x2) / K)

    # Combine into a NumPy array
    fx = np.array([fx1, fx2])

    # Standard output packaging
    fx_flat = fx.flatten()
    tau_flat = tau.flatten()
    q_dotdot = fx_flat + tau_flat

    return q_dotdot, fx_flat

# --------------------------------------------------------------------------------------------------
## 2. Resonance and Oscillation Dynamics (Duffing-like Nonlinear Restoring Force)
# Behavior: Dynamics dominated by linear stiffness (small state) then by cubic nonlinearity (large state).
# f(x) represents a nonlinear restoring force, similar to the -kx - alpha*x^3 terms.
def _duffing_dynamics(q, q_dot, tau):
    # Unpack state variables
    x1, x2, xdot1, xdot2 = q[0], q[1], q_dot[0], q_dot[1]

    # Model Parameters
    beta = 0.35  # Linear stiffness coefficient (dominates for small x)
    alpha = 0.015 # Nonlinear cubic stiffness coefficient (dominates for large x)
    delta = 0.5 # Damping coefficient (proportional to velocity)

    # Dynamics Function (f(x) = -beta*x - alpha*x^3 - delta*x_dot)
    # This represents the total "natural" acceleration of the system.
    fx1 = -(beta * x1) - (alpha * x1**3) - (delta * xdot1)
    fx2 = -(beta * x2) - (alpha * x2**3) - (delta * xdot2)

    # Combine into a NumPy array
    fx = np.array([fx1, fx2])

    # Standard output packaging
    fx_flat = fx.flatten()
    tau_flat = tau.flatten()
    q_dotdot = fx_flat + tau_flat

    return q_dotdot, fx_flat

def _duffing_squared_dynamics(q, q_dot, tau):
    # Unpack state variables
    x1, x2, xdot1, xdot2 = q[0], q[1], q_dot[0], q_dot[1]

    # Model Parameters
    beta = 0.35  # Linear stiffness coefficient (dominates for small x)
    alpha = 0.015 # Nonlinear cubic stiffness coefficient (dominates for large x)
    delta = 0.3 # Damping coefficient (proportional to velocity)

    # Dynamics Function (f(x) = -beta*x - alpha*x^3 - delta*x_dot)
    # This represents the total "natural" acceleration of the system.
    fx1 = -(beta * x1) - (alpha * x1**3) - (delta * xdot1) - (beta*xdot1**2)
    fx2 = -(beta * x2) - (alpha * x2**3) - (delta * xdot2) - (beta*xdot2**2)

    # Combine into a NumPy array
    fx = np.array([fx1, fx2])

    # Standard output packaging
    fx_flat = fx.flatten()
    tau_flat = tau.flatten()
    q_dotdot = fx_flat + tau_flat

    return q_dotdot, fx_flat