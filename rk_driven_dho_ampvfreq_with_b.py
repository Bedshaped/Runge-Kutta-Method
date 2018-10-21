# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 11:31:51 2018

@author: moreaua2
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def DampedDrivenOscillator(t, y, b, omega_D, params):
    x, v = y
    a, omega_0 = params
    dxdt = v
    dvdt = -b*v - (omega_0**2)*x - a*np.sin(omega_D*t)
    dydt = np.array([dxdt, dvdt])
    return dydt

# Parameters
    
t_0 = 0
t_max = 50*np.pi
n = 100

# Initial values

params = [0.9, 1.2] # A, omega_0
y_0 = [0, 1]

amplitudes = []
drivingFrequencies = np.linspace(0, 2*params[1], 100)
i = 0

for b in (0.1, 0.25, 0.5, 0.75, 0.9):
    for omega_d in drivingFrequencies:
        func = lambda t, y : DampedDrivenOscillator(t, y, b, omega_d, params)
        res = solve_ivp(func, [t_0, t_max], y_0, method="RK45",
                        t_eval=np.linspace(0.8*t_max, t_max, n))
    
        x = res.y[0]
        t = res.t
        
        amplitudes.append((max(x)-min(x))/2)
        
    plt.figure(1, figsize=(9, 6))
    plt.plot(drivingFrequencies, amplitudes, label=r"b = $%.2f$" % b)
    amplitudes = []
        
textstr = '\n'.join((
    r'$a=%.2f$' % (params[0], ),
    r'$\omega_0=%.2f$' % (params[1], )))
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)


plt.ylabel(r"Amplitude")
plt.xlabel(r"Driving frequency $\omega_D$")
plt.legend()
ax = plt.gca()
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
plt.savefig("rk_ampvfreq.png", dpi=300)