import pykoop
from sklearn.preprocessing import MaxAbsScaler, StandardScaler

# Get sample mass-spring-damper data
X_msd = pykoop.example_data_msd()

# Create pipeline
kp = pykoop.KoopmanPipeline(
    lifting_functions=[
        pykoop.SkLearnLiftingFn(MaxAbsScaler()),
        pykoop.PolynomialLiftingFn(order=2),
        pykoop.SkLearnLiftingFn(StandardScaler())
    ],
    regressor=pykoop.Edmd(alpha=0.1),
)

# Fit the pipeline
kp.fit(X_msd, n_inputs=1, episode_feature=True)

# Predict using the pipeline
X_pred = kp.predict_multistep(X_msd)

# Score using the pipeline
score = kp.score(X_msd)
