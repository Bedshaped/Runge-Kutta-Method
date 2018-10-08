# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 12:17:56 2018

@author: moreaua2
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define our function

def wrapper(t, y):
    return Nonlinear1(t, y, params)

def Nonlinear1(t, y, params):
    a, b = params
    dydt = -a*y**3 + b*np.sin(t)
    return dydt

# Parameters
   
params = [1, 1] # Our a and b values

t_0 = 0
t_max = 20
dt = 0.2
n = int(t_max/dt)

# Initial values

y_0 = np.array([0])

# Set up our interval and use our integration function

t = [t_0, t_max]

res = solve_ivp(wrapper, t, y_0, method="RK45", t_eval=np.linspace(t_0, t_max, n))

y = res.y[0]
t = res.t

plt.scatter(t, y)
plt.show()