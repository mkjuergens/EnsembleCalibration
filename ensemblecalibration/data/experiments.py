import torch

from ensemblecalibration.data.gp_binary import exp_gp

def get_experiment(params: dict, h0: bool = True):

    if params["experiment"] == "gp":
        raise NotImplementedError("GP experiment not implemented yet.")