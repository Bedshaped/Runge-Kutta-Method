# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 12:17:56 2018

@author: moreaua2
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define our function

def Nonlinear1(t, y, params):
    a, b = params
    dydt = -a*y**3 + b*np.sin(t)
    return dydt

# Parameters
   
params = [0.2, 0.2] # Our a and b values

t_0 = 0
t_max = 600
dt = 0.02
n = int(t_max/dt)

# Initial values

y_0 = np.array([0])

# Set up our interval and use our integration function

t = [t_0, t_max]

res = solve_ivp(lambda t, y : Nonlinear1(t, y, params), t, y_0, method="RK45", t_eval=np.linspace(t_0, t_max, n))

y = res.y[0]
t = res.t

textstr = '\n'.join((
    r'$a=%.2f$' % (params[0], ),
    r'$b=%.2f$' % (params[1], )))
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

plt.plot(t, y, 'b')
plt.xlabel("t")
plt.ylabel("y(t)")
ax = plt.gca()
ax.text(0.05, 0.25, textstr, transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
plt.savefig("rk_nonlinear_firstorder.png", dpi=300)