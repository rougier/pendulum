# Double pendulum formula translated from the C code at
# http://www.physics.usyd.edu.au/~wheat/dpend_html/solve_dpend.c

from numpy import sin, cos
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation
G = 9.81  # acceleration due to gravity, in m/s^2
L1 = 1.0  # length of pendulum 1 in m
L2 = 1.0  # length of pendulum 2 in m
M1 = 1.0  # mass of pendulum 1 in kg
M2 = 1.0  # mass of pendulum 2 in kg


def derivs(state, t):

    dydx = np.zeros_like(state)
    dydx[0] = state[1]
    del_ = state[2] - state[0]
    den1 = (M1 + M2)*L1 - M2*L1*cos(del_)*cos(del_)
    dydx[1] = (M2*L1*state[1]*state[1]*sin(del_)*cos(del_) +
               M2*G*sin(state[2])*cos(del_) +
               M2*L2*state[3]*state[3]*sin(del_) -
               (M1 + M2)*G*sin(state[0]))/den1
    dydx[2] = state[3]
    den2 = (L2/L1)*den1
    dydx[3] = (-M2*L2*state[3]*state[3]*sin(del_)*cos(del_) +
               (M1 + M2)*G*sin(state[0])*cos(del_) -
               (M1 + M2)*L1*state[1]*state[1]*sin(del_) -
               (M1 + M2)*G*sin(state[2]))/den2
    return dydx

# create a time array from 0..100 sampled at 0.015 second steps
dt = 0.01
t = np.arange(0.0, 100.0, dt)

# th1 and th2 are the initial angles (degrees)
# w10 and w20 are the initial angular velocities (degrees per second)
th1 = 120.0
w1 = 0.0
th2 = -10.0
w2 = 0.0

# initial state
state = np.radians([th1, w1, th2, w2])

# integrate your ODE using scipy.integrate.
y = integrate.odeint(derivs, state, t)


P1 = np.dstack([L1*sin(y[:, 0]), -L1*cos(y[:, 0])]).squeeze()
P2 = P1 + np.dstack([L2*sin(y[:, 2]), -L2*cos(y[:, 2])]).squeeze()
               

fig = plt.figure(figsize=(5,5), facecolor=".85")
ax = plt.axes([0,0,1,1], frameon=False)
#subplot(1,1,1, aspect=1, frameon = False, xlim=(-2, 2), ylim=(-2, 2))

n = 250
colors= np.zeros((n,4))
colors[:,3] = np.linspace(0, 1, n, endpoint=True)
scatter = ax.scatter(np.zeros(n), np.zeros(n), s = 10,
                     facecolor = colors, edgecolor='none', zorder=-100)

line1, = ax.plot([], [], '-', color='k', lw=12, solid_capstyle='round')
line2, = ax.plot([], [], '-', color='w', lw=10, solid_capstyle='round', zorder=10)
line3, = ax.plot([], [], 'o', color='k', markersize=2, zorder=20)

line4, = ax.plot([], [], '-', color='.75', lw=12, solid_capstyle='round', zorder=-50)
line5, = ax.plot([], [], '-', color='.90', lw=10, solid_capstyle='round', zorder=-40)
line6, = ax.plot([], [], 'o', color='.75', markersize=2, zorder=-30)


ax.set_xlim(-2,+2)
ax.set_xticks([])
ax.set_ylim(-2,+2)
ax.set_yticks([])

def animate(i):

    j = max(i-50,0)
    X = [0, P1[j,0], P2[j,0]]
    Y = [0, P1[j,1], P2[j,1]]
    line4.set_data(X,Y)
    line5.set_data(X,Y)
    line6.set_data(X,Y)

    
    X = [0, P1[i,0], P2[i,0]]
    Y = [0, P1[i,1], P2[i,1]]
    line1.set_data(X,Y)
    line2.set_data(X,Y)
    line3.set_data(X,Y)

    
    scatter.set_offsets(P2[max(i-n,0):i])

ani = animation.FuncAnimation(fig, animate, np.arange(1, len(y)), interval=25)
#ani.save('pendulum.mp4', fps=30)
plt.show()
