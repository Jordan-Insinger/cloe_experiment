# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 08:45:41 2025

@author: rebecca.hart
"""

import numpy as np

class Config:
    def __init__(self, **kwargs):
        """
        A flexible config class that accepts any key-value pairs.
        """
        # Loop through all the key-value pairs passed in from the dictionary
        for key, value in kwargs.items():
            # Set an attribute on this object with the same name as the key
            # e.g., if key='alpha1', this does 'self.alpha1 = 15'
            setattr(self, key, value)

        # You can still have it perform post-initialization calculations
        # This assumes 'T_sim' and 'dt' will always be in the dictionary
        self.total_steps = int(self.T_sim / self.dt)
        self.time_steps_array = np.arange(0, self.T_sim, self.dt)

