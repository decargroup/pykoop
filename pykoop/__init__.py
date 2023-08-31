"""Koopman operator identification library in Python."""

from ._sklearn_config.config import config_context, get_config, set_config
from .centers import (
    Centers,
    ClusterCenters,
    DataCenters,
    GaussianMixtureRandomCenters,
    GaussianRandomCenters,
    GridCenters,
    QmcCenters,
    UniformRandomCenters,
)
from .kernel_approximation import (
    KernelApproximation,
    RandomBinningKernelApprox,
    RandomFourierKernelApprox,
)
from .koopman_pipeline import (
    EpisodeDependentLiftingFn,
    EpisodeIndependentLiftingFn,
    KoopmanLiftingFn,
    KoopmanPipeline,
    KoopmanRegressor,
    SplitPipeline,
    combine_episodes,
    extract_initial_conditions,
    extract_input,
    score_trajectory,
    shift_episodes,
    split_episodes,
    strip_initial_conditions,
    unique_episodes,
)
from .lifting_functions import (
    BilinearInputLiftingFn,
    ConstantLiftingFn,
    DelayLiftingFn,
    KernelApproxLiftingFn,
    PolynomialLiftingFn,
    RbfLiftingFn,
    SkLearnLiftingFn,
)
from .regressors import DataRegressor, Dmd, Dmdc, Edmd, EdmdMeta
from .tsvd import Tsvd
from .util import (
    AnglePreprocessor,
    example_data_duffing,
    example_data_msd,
    example_data_pendulum,
    example_data_vdp,
    random_input,
    random_state,
)
