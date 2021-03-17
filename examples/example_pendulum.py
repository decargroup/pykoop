import numpy as np
from scipy import integrate
from matplotlib import pyplot as plt
from dynamics import pendulum
from pykoop import koopman_pipeline, lifting_functions, lmi, dmd
from sklearn import preprocessing

pend = pendulum.Pendulum(mass=1, length=1, damping=0.1)
t_range = (0, 10)
t_step = 0.01
x0 = pend.x0(np.array([np.pi/2, 0]))


def u(t):
    return 5 * np.exp(-t / 5) * np.sin(2 * np.pi * t)


sol = integrate.solve_ivp(lambda t, x: pend.f(t, x, u(t)), t_range, x0,
                          t_eval=np.arange(*t_range, t_step), rtol=1e-8,
                          atol=1e-8)

X = np.vstack((
    np.zeros((1, sol.y.shape[1])),
    sol.y,
    u(sol.t),
))

kp = koopman_pipeline.KoopmanPipeline(
    preprocessing=preprocessing.StandardScaler(),
    # preprocessing=preprocessing.MinMaxScaler((-1, 1)),
    delay=lifting_functions.Delay(n_delay_x=2, n_delay_u=2),
    lifting_function=lifting_functions.PolynomialLiftingFn(order=2),
    estimator=dmd.Edmd()
)

kp.fit(X.T, n_u=1)

crash_index = None
n_samp = kp.delay_.n_samples_needed_
u_sim = np.reshape(u(sol.t), (1, -1))

X_sim = np.empty(sol.y.shape)
X_sim[:, :n_samp] = sol.y[:, :n_samp]
for k in range(n_samp, X.shape[1]):
    Xk = np.vstack((
        np.zeros((1, n_samp)),
        X_sim[:, (k-n_samp):k],
        u_sim[:, (k-n_samp):k]
    ))
    try:
        Xp = kp.predict(Xk.T).T
    except ValueError:
        crash_index = k
        break
    X_sim[:, [k]] = Xp[1:, [-1]]

if crash_index is not None:
    X_sim[:, crash_index:] = np.nan * np.ones((X_sim.shape[0], 1))
    X_sim = np.ma.masked_invalid(X_sim)

fig, ax = plt.subplots(3, 1)
ax[0].plot(sol.t, sol.y[0, :])
ax[0].plot(sol.t, X_sim[0, :])
ax[1].plot(sol.t, sol.y[1, :])
ax[1].plot(sol.t, X_sim[1, :])
ax[2].plot(sol.t, u(sol.t))

ax[0].set_ylim((-10, 10))
ax[1].set_ylim((-10, 10))
plt.show()
