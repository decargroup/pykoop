import numpy as np
from scipy import integrate, linalg
from pykoop import lmi, koopman_pipeline, lifting_functions
from dynamics import mass_spring_damper
from matplotlib import pyplot as plt

plt.rc('lines', linewidth=2)
plt.rc('axes', grid=True)
plt.rc('grid', linestyle='--')


def main():
    # Set up problem
    t_range = (0, 5)
    t_step = 0.1
    msd = mass_spring_damper.MassSpringDamper(
        mass=0.5,
        stiffness=0.7,
        damping=0.6
    )

    def u(t):
        return 0.1 * np.sin(t)

    def ivp(t, x):
        return msd.f(t, x, u(t))

    # Solve ODE for training data
    x0 = msd.x0(np.array([0, 0]))
    sol = integrate.solve_ivp(ivp, t_range, x0,
                              t_eval=np.arange(*t_range, t_step),
                              rtol=1e-8, atol=1e-8)

    # Split the data
    X = np.vstack((
        sol.y[:, :-1],
        np.reshape(u(sol.t), (1, -1))[:, :-1]
    ))

    kp = koopman_pipeline.KoopmanPipeline(
        delay=lifting_functions.Delay(),
        estimator=lmi.LmiEdmd(),
    )
    kp.fit(X.T, n_u=1)
    breakpoint()


if __name__ == '__main__':
    main()
