from ensemblecalibration.sampling.lambda_sampling import (
    multinomial_label_sampling,
    uniform_weight_sampling,
)
from ensemblecalibration.sampling.mcmc_sampling import mhar_sampling_p
from ensemblecalibration.sampling.rejectance_sampling import rejectance_sampling_p
from ensemblecalibration.calibration.experiments import experiment_h0_feature_dependency


def sample_p_bar(p_probs, params: dict):
    """
    function which uses a predefined method to sample from the polytope of convex combinations 
    of ensemble predictions
    Parameters
    ----------
        p_probs: np.ndarray of shape (N, M, K) containing predictions of each predictor for each instance
        params: dict containing test parameters
    Returns
    -------
        P_bar_b: np.ndarray of shape (N, K) containing sampled predictions
    """
    if params["sampling"] == "lambda":
        # sample convex combinations of ensemle predictions
        P_bar_b = uniform_weight_sampling(p_probs)  # of shape (N, M)
    elif params["sampling"] == "mcmc":
        P_bar_b = mhar_sampling_p(p_probs, transform=params["transform"])
    elif params["sampling"] == "rejectance":
        P_bar_b = rejectance_sampling_p(p_probs)
    else:
        raise NameError("check sampling method in configuration dictionary")

    return P_bar_b


if __name__ == "__main__":
    p_probs, y = experiment_h0_feature_dependency(100, 10, 10, 0.01)
    params = {"sampling": "mcmc", "transform": "isometric"}
    p_bar = sample_p_bar(p_probs=p_probs, params=params)
    print(p_bar)
