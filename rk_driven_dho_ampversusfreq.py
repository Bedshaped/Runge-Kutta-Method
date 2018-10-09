# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 10:55:31 2018

@author: moreaua2
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def DampedDrivenOscillator(t, y, omega_D, params):
    x, v = y
    a, b, omega_0 = params
    dxdt = v
    dvdt = -b*v - (omega_0**2)*x - a*np.sin(omega_D*t)
    dydt = np.array([dxdt, dvdt])
    return dydt

# Parameters
    
t_0 = 0
t_max = 50*np.pi
dt = np.pi/50
n = int(t_max / dt)

# Initial values

params = [1, 0.05, 1] # A, b, omega_0
y_0 = [0, 1]

amplitudes = []
drivingFrequencies = np.linspace(0, 2*params[2], 100)

for omega_d in drivingFrequencies:
    func = lambda t, y : DampedDrivenOscillator(t, y, omega_d, params)
    res = solve_ivp(func, [t_0, t_max], y_0, method="RK45", t_eval=np.linspace(0.8*t_max, t_max, n))

    x = res.y[0]
    t = res.t
    
    amplitudes.append((max(x)-min(x))/2)
        
textstr = '\n'.join((
    r'$a=%.2f$' % (params[0], ),
    r'$b=%.2f$' % (params[1], ),
    r'$\omega_0=%.2f$' % (params[2], )))
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

plt.figure(1, figsize=(9, 6))
plt.plot(drivingFrequencies, amplitudes, "r-")
plt.ylabel(r"Amplitude")
plt.xlabel(r"$\omega_D$")
ax = plt.gca()
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)