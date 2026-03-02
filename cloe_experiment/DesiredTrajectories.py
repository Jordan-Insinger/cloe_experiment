import numpy as np

def _circular_trajectory(t, params):
    """Calculates the position, velocity, and acceleration for a circular path."""
    A, f1 = params['A'], params['f1']
    qd = np.array([A * np.cos(f1 * t), A * np.sin(f1 * t)])
    qd_dot = np.array([-A * f1 * np.sin(f1 * t), A * f1 * np.cos(f1 * t)])
    qd_ddot = np.array([-A * f1**2 * np.cos(f1 * t), -A * f1**2 * np.sin(f1 * t)])
    return qd, qd_dot, qd_ddot

def _figure_eight_trajectory(t, params):
    """Calculates the position, velocity, and acceleration for a figure-eight path."""
    A, B, f1, f2 = params['A'], params['B'], params['f1'], params['f2']
    qd = np.array([A * np.sin(f1 * t), B * np.sin(f2 * t) * np.cos(f1 * t)])
    qd_dot = np.array([A * f1 * np.cos(f1 * t), B * (f2 * np.cos(f2 * t) * np.cos(f1 * t) - f1 * np.sin(f2 * t) * np.sin(f1 * t))])
    qd_ddot = np.array([-A * f1**2 * np.sin(f1 * t), B * (-f2**2 * np.sin(f2 * t) * np.cos(f1 * t) - 2 * f1 * f2 * np.cos(f2 * t) * np.sin(f1 * t) - f1**2 * np.sin(f2 * t) * np.cos(f1 * t))])
    return qd, qd_dot, qd_ddot

def _multi_sinusoid_trajectory(t, params):
    """Calculates the position, velocity, and acceleration for a multi-sinusoid path."""
    A, f1, f2 = params['A'], params['f1'], params['f2']
    qd = np.array([A * np.sin(f1 * t), f1 * np.sin(f2 * t)])
    qd_dot = np.array([A * f1 * np.cos(f1 * t), f1 * f2 * np.cos(f2 * t)])
    qd_ddot = np.array([-A * f1**2 * np.sin(f1 * t), -f1 * f2**2 * np.sin(f2 * t)])
    return qd, qd_dot, qd_ddot

def _spiral_trajectory(t, params):
    """Calculates the position, velocity, and acceleration for a spiral path."""
    # --- Trajectory Parameters ---
    A = params['A']   # Initial radius of the spiral
    f1 = params['f1'] # Angular velocity of the spiral

    # --- Spiral Trajectory ---
    radius = A + 0.1 * t
    qd_x = radius * np.cos(f1 * t)
    qd_y = radius * np.sin(f1 * t)

    # --- Velocity components ---
    qd_dot_x = -f1 * radius * np.sin(f1 * t) + 0.1 * np.cos(f1 * t)
    qd_dot_y =  f1 * radius * np.cos(f1 * t) + 0.1 * np.sin(f1 * t)

    # --- Acceleration components ---
    qd_ddot_x = -f1**2 * radius * np.cos(f1 * t) - 2 * f1 * 0.1 * np.sin(f1 * t)
    qd_ddot_y = -f1**2 * radius * np.sin(f1 * t) + 2 * f1 * 0.1 * np.cos(f1 * t)

    # --- Assemble the output vectors ---
    qd = np.array([qd_x, qd_y])
    qd_dot = np.array([qd_dot_x, qd_dot_y])
    qd_ddot = np.array([qd_ddot_x, qd_ddot_y])
    
    return qd, qd_dot, qd_ddot

def _growing_sinusoid_trajectory(t, params):
    """
    Calculates a trajectory where the amplitude grows over time.
    This follows the product rule for derivatives: (uv)' = u'v + uv'
    """
    # --- Parameters ---
    A, B, f1, f2 = params['A'], params['B'], params['f1'], params['f2']

    # --- Define the components of the product rule ---
    # For a trajectory like qd = u(t) * v(t)
    
    # u(t): The growth factor
    growth_factor = 1 - np.exp(-0.1 * t)
    
    # v(t): The base sinusoidal vector
    base_vector = np.array([A * np.sin(f1 * t), B * np.sin(f2 * t)])
    
    # u'(t): The derivative of the growth factor
    growth_derivative = 0.1 * np.exp(-0.1 * t)
    
    # v'(t): The derivative of the base vector
    base_derivative = np.array([f1 * A * np.cos(f1 * t), f2 * B * np.cos(f2 * t)])
    
    # --- Position: qd = u * v ---
    qd = growth_factor * base_vector

    # --- Velocity: qd_dot = u'v + uv' ---
    qd_dot = (growth_derivative * base_vector) + (growth_factor * base_derivative)
    
    # --- Acceleration: qd_ddot = u''v + 2u'v' + uv'' ---
    growth_accel = -0.01 * np.exp(-0.1 * t) # u''(t)
    base_accel = np.array([-f1**2 * A * np.sin(f1 * t), -f2**2 * B * np.sin(f2 * t)]) # v''(t)

    term1 = growth_accel * base_vector          # u''v
    term2 = 2 * growth_derivative * base_derivative # 2u'v'
    term3 = growth_factor * base_accel          # uv''
    
    qd_ddot = term1 + term2 + term3

    return qd, qd_dot, qd_ddot

# --- UPDATED: The main dispatcher function ---
def generate_trajectory(trajectory_name: str, t: float, params: dict):
    """
    General dispatcher to generate a point on a desired trajectory.
    """
    if trajectory_name == 'circular':
        return _circular_trajectory(t, params)
    
    elif trajectory_name == 'figure_eight':
        return _figure_eight_trajectory(t, params)
    
    elif trajectory_name == 'multi_sinusoid':
        return _multi_sinusoid_trajectory(t, params)
    elif trajectory_name == 'spiral':
        return _spiral_trajectory(t, params)
    elif trajectory_name == 'growing_sinusoid':
        return _growing_sinusoid_trajectory(t, params)
        
    else:
        raise ValueError(f"Unknown trajectory name: '{trajectory_name}'")