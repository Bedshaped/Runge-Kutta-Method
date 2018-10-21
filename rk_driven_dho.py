# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 10:09:08 2018

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

params = [0.9, 0.1, 1.2, 1.1] # A, b, omega_0, omega_D
y_0 = [0, 1]

res = solve_ivp(lambda t, y : DampedDrivenOscillator(t, y, params), [t_0, t_max], y_0, method="RK45", t_eval=np.linspace(t_0, t_max, n))

x = res.y[0]
v = res.y[1]
t = res.t

textstr = '\n'.join((
    r'$a=%.2f$' % (params[0], ),
    r'$b=%.2f$' % (params[1], ),
    r'$\omega_0=%.2f$' % (params[2], ),
    r'$\omega_D=%.2f$' % (params[3], )))
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

plt.figure(1, figsize=(9, 6))
plt.plot(t, x, label=r"$x , x_0 = %d$" % y_0[0])
plt.plot(t, v, label=r"$v , v_0 = %d$" % y_0[1])
plt.xlabel(r"Time $t$")
plt.ylabel(r"Amplitude")
plt.legend()
ax = plt.gca()
ax.text(0.05, 0.25, textstr, transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
plt.savefig("rk_ddho.png", dpi=300)


plt.figure(2, figsize=(9, 6))
plt.plot(x, v, 'k')
plt.axis('equal')
plt.xlabel(r"$x$")
plt.ylabel(r"$v$")
plt.savefig("rk_ddho_phase.png", dpi=300)