# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 12:18:30 2018

@author: moreaua2
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import correlate


def DampedDrivenOscillator(t, y, params):
    x, v = y
    a, b, omega_0, omega_D = params
    dxdt = v
    dvdt = -b*v - (omega_0**2)*x - a*np.sin(omega_D*t)
    dydt = np.array([dxdt, dvdt])
    return dydt

def DrivenElement(t, a, omega_d):
    return a*np.sin(omega_d*t)

# Parameters
    
t_0 = 0
t_max = 50*np.pi
dt = np.pi/50
n = int(t_max / dt)

# Initial values

params = [0.2, 0.1, 1, 0.9] # A, b, omega_0, omega_D
y_0 = [0, 1]

for omega_d in (params[2], 0.75*params[2], 0.25*params[2]):
    params[3] = omega_d
    func = lambda t, y : DampedDrivenOscillator(t, y, params)
    res = solve_ivp(func, [t_0, t_max], y_0, method="RK45", t_eval=np.linspace(0.8*t_max, t_max, n))

    x = res.y[0]
    t = res.t
        
    #driven = DrivenElement(t, params[1], omega_d)
    
    #nsamples = x.size
    
    #x -= x.mean(); x /= x.std()
    #driven -= driven.mean(); driven /= driven.std()
    
    #corr = correlate(x, driven)
    
    #dt = np.arange(1-nsamples, nsamples)
    #recovered_time_shift = dt[corr.argmax()]
    
    
    plt.figure(figsize=(9, 6))
    plt.plot(t, x, label=r"$x , \omega_D = %.2f$" % omega_d)
    plt.plot(t, DrivenElement(t, params[1], omega_d), label=r"$F_D , \omega_D = %.2f$" % omega_d)
    plt.legend()
    