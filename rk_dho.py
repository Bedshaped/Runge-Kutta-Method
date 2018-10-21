# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 12:44:21 2018

@author: moreaua2
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def DampedOscillator(t, y, params):
    x, v = y
    b, omega_0 = params
    dxdt = v
    dvdt = -b*v-(omega_0**2)*x
    dydt = np.array([dxdt, dvdt])
    return dydt

# Parameters
    
t_0 = 0
t_max = 50*np.pi
dt = np.pi/50
n = int(t_max / dt)

# Initial values

params = [0.1, 1.2] # b, omega_0
y_0 = [0, 1]

res = solve_ivp(lambda t, y : DampedOscillator(t, y, params), [t_0, t_max], y_0, method="RK45", t_eval=np.linspace(t_0, t_max, n))

x = res.y[0]
v = res.y[1]
t = res.t

textstr = '\n'.join((
    r'$b=%.2f$' % (params[0], ),
    r'$\omega_0=%.2f$' % (params[1], )))
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

plt.figure(1, figsize=(9, 6))
plt.plot(t, x, label=r"$x , x_0 = %d$" % y_0[0])
plt.plot(t, v, label=r"$v , v_0 = %d$" % y_0[1])
plt.xlabel("Time")
plt.legend()
ax = plt.gca()
ax.text(0.05, 0.25, textstr, transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
plt.savefig("rk_dho.png", dpi=300)

plt.figure(2, figsize=(9, 6))
plt.plot(x, v, 'k')
plt.axis('equal')
plt.xlabel(r"$x$")
plt.ylabel(r"$v$")
plt.savefig("rk_dho_phase.png", dpi=300)
