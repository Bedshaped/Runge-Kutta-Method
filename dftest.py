# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 12:01:49 2018

@author: moreaua2
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def DampedDrivenOscillator(t, y, params):
    x, v = y
    a, b, omega_0, omega_D = params
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

params = [1, 0.1, 1, 0.9] # A, b, omega_0, omega_D
y_0 = [0, 1]

for omega_d in (params[2], 0.75*params[2], 0.25*params[2]):
    params[3] = omega_d
    func = lambda t, y : DampedDrivenOscillator(t, y, params)
    res = solve_ivp(func, [t_0, t_max], y_0, method="RK45", t_eval=np.linspace(t_0, t_max, n))

    x = res.y[0]
    t = res.t
        
    plt.figure(figsize=(9, 6))
    plt.plot(t, x, label=r"$x , \omega_D = %.2f$" % omega_d)
    plt.plot(t, params[0]*np.sin(omega_d*t), label=r"$f_d , \omega_D = %.2f$" % omega_d)
    plt.legend()

    
plt.xlabel("Time")
plt.savefig("rk_harmonic_analysis.png", dpi=300)