import numpy as np
from scipy import integrate
from dynamics import pendulum
from pykoop import koopman_pipeline, lifting_functions, lmi, dmd
from sklearn import preprocessing
import cProfile
import pstats

pend = pendulum.Pendulum(mass=1, length=1, damping=0.1)
t_range = (0, 10)
t_step = 0.01
x0 = pend.x0(np.array([np.pi/2, 0]))


def u(t):
    return 5 * np.exp(-t / 5) * np.sin(2 * np.pi * t)


sol = integrate.solve_ivp(lambda t, x: pend.f(t, x, u(t)), t_range, x0,
                          t_eval=np.arange(*t_range, t_step), rtol=1e-8,
                          atol=1e-8)

X_raw = np.vstack((
    np.zeros((1, sol.y.shape[1])),
    sol.y,
    u(sol.t),
))
X = lifting_functions.AnglePreprocessor().fit_transform(
    X_raw.T, angles=np.array([False, True, False, False])).T

kp = koopman_pipeline.KoopmanPipeline(
    preprocessing=preprocessing.StandardScaler(),
    delay=lifting_functions.Delay(n_delay_x=1, n_delay_u=1),
    lifting_function=lifting_functions.PolynomialLiftingFn(order=2),
    # estimator=dmd.Edmd()
    estimator=lmi.LmiEdmdTikhonovReg(alpha=1e-6, inv_method='sqrt',
                                     solver='mosek')
)

# Set up profiling
# pr = cProfile.Profile()
# pr.enable()
# Run profiled code
kp.fit(X.T, n_u=1)
score = kp.score(X.T)
print(f'score = {score}')
# Print profiling stats
# ps = pstats.Stats(pr)
# ps.strip_dirs().sort_stats('cumtime').print_stats()
