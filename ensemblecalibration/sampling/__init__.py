from ensemblecalibration.sampling.lambda_sampling import multinomial_label_sampling, uniform_weight_sampling
from ensemblecalibration.sampling.mcmc_sampling import mhar_sampling_p
from ensemblecalibration.sampling.rejectance_sampling import rejectance_sampling_p


def sample_p_bar(p_probs, params: dict):

    if params["sampling"] == "lambda":
        # sample convex combinations of ensemle predictions
        P_bar_b = uniform_weight_sampling(p_probs) # of shape (N, M)
    elif params["sampling"] == "mcmc":
        P_bar_b = mhar_sampling_p(p_probs, transform=params["transform"])
    elif params["sampling"] == "rejectance":
        P_bar_b = rejectance_sampling_p(p_probs)
    else:
        raise NameError("check sampling method in configuration dictionary")
    
    return P_bar_b
