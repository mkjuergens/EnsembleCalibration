from src.meta_model.primary_model import *
from src.meta_model.secondary_model import *
from src.meta_model.train import *


def create_calibrator(calibrator_name: str) -> nn.Module:

    calibrator_name = calibrator_name.lower()
    if calibrator_name == "linear":
        return LinearCalibrator
    elif calibrator_name == "dirichlet":
        return DirichletCalibrator
    elif calibrator_name == "temperature":
        return TemperatureScalingCalibrator
    # elif ...
    else:
        raise ValueError(f"Unknown calibrator name: {calibrator_name}")


def create_comb_model(comb_model_name: str) -> nn.Module:

    comb_model_name = comb_model_name.lower()
    if comb_model_name == "mlp":
        return MLPCalW
    elif comb_model_name == "conv":
        return MLPCalWConv
    else:
        raise ValueError(f"Unknown comb model name: {comb_model_name}")
