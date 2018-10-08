# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 14:50:09 2018

@author: moreaua2
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define our function

def RLCircuit(t, i, params):
    V, R, L = params
    didt = (V - R*i)/L
    return didt

def Analytical(t, params):
    V, R, L = params
    return (V/R)*(1 - np.exp((-R*t)/L))

# Parameters

t_0 = 0
t_max = 30
n = int(100)

# Initial values

V = 10 # Volts
R = 50 # Ohms
L = 100 # Henries
   
params = np.array([V, R, L])

i_0 = np.array([0])

# Set up our interval and use our integration function

t = np.linspace(t_0, t_max, n)

res = solve_ivp(lambda t, i : RLCircuit(t, i, params), [t_0, t_max], i_0, method="RK45", t_eval=t)

y = res.y[0]
t = res.t

# Solve the same problem analytically for comparison using the same time interval

x = Analytical(t, params)

plt.figure(1, figsize=(9, 6))
plt.plot(t, y, label=r"RK Solution")
plt.plot(t, x, 'k.', label=r"Exact Solution")
plt.xlabel("Time")
plt.legend()

plt.figure(2, figsize=(9, 6))
plt.plot(t, abs(x - y), label=r"Deviation")
plt.legend()