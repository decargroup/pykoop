import numpy as np
from scipy import integrate
from matplotlib import pyplot as plt
from dynamics import pendulum

pend = pendulum.Pendulum(mass=1, length=1, damping=0.1)
t_range = (0, 10)
t_step = 0.01
x0 = pend.x0(np.array([np.pi/2, 0]))


def u(t):
    return 5 * np.exp(-t / 5) * np.sin(2 * np.pi * t)


sol = integrate.solve_ivp(lambda t, x: pend.f(t, x, u(t)), t_range, x0,
                          t_eval=np.arange(*t_range, t_step), rtol=1e-8,
                          atol=1e-8)

fig, ax = plt.subplots(3, 1)
ax[0].plot(sol.t, sol.y[0, :])
ax[1].plot(sol.t, sol.y[1, :])
ax[2].plot(sol.t, u(sol.t))
plt.show()
