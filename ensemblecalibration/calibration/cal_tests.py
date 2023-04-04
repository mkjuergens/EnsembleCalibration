import random
import numpy as np

from scipy.stats import halfnorm

from ensemblecalibration.calibration.helpers import sample_m
from ensemblecalibration.sampling import (
    mhar_sampling_p,
    uniform_weight_sampling,
    rejectance_sampling_p,
)
from ensemblecalibration.calibration.minimization import (
    solve_cobyla1D,
    solve_cobyla2D,
    solve_neldermead1D,
    solve_neldermead2D,
)


_MAX_RNORM = np.max(halfnorm.rvs(size=1000 * 10000).reshape(-1, 10000), axis=0)


def npbetest_alpha(P, y, params):
    """
    updated, faster version of the npbe test:  calculate sampling distribution on,ly ocne,
    then use it for every alpha
    """
    stats = np.zeros(params["n_resamples"])
    for b in range(params["n_resamples"]):
        # extract bootstrap sample
        P_b = random.sample(P.tolist(), P.shape[0])
        P_b = np.stack(P_b)
        # sample predictions using a predefined sampling method
        if params["sampling"] == "lambda":
            # sample convex combinations of ensemle predictions
            P_bar_b = uniform_weight_sampling(P_b)  # of shape (N, M)
        elif params["sampling"] == "mcmc":
            P_bar_b = mhar_sampling_p(P_b, transform=params["transform"])
        elif params["sampling"] == "rejectance":
            P_bar_b = rejectance_sampling_p(P_b)
        else:
            raise NameError("check sampling method in configuration dictionary")
        # round to 5 decimal digits for numerical stability
        P_bar_b = np.trunc(P_bar_b * 10**3) / (10**3)
        P_bar_b = np.clip(P_bar_b, 0, 1)
        y_b = np.apply_along_axis(sample_m, 1, P_bar_b)
        # perform test
        stats[b] = params["test"](P_bar_b, y_b, params)

    # calculate 1-alpha quantile from sampling distribution
    q_alpha = np.quantile(stats, 1 - (np.array(params["alpha"])))

    # solve minimization problem
    if params["optim"] == "neldermead":
        if params["x_dependency"]:
            l = solve_neldermead2D(P, y, params)
        else:
            l = solve_neldermead1D(P, y, params)
    elif params["optim"] == "cobyla":
        if params["x_dependency"]:
            l = solve_cobyla2D(P, y, params)
        else:
            l = solve_cobyla1D(P, y, params)

    else:
        raise NotImplementedError

    minstat = params["obj"](l, P, y, params)
    decision = list(map(int, np.abs(minstat) > q_alpha))

    return decision, l


def _npbetest_alpha(P, y, alpha, params):
    params["alpha"] = alpha
    dec, l = npbetest_alpha(P, y, params)

    return dec
