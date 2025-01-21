import pytest

from ensemblecalibration.meta_model.secondary_model import LinearCalibrator
from ensemblecalibration.meta_model.primary_model import MLPCalW, MLPCalWConv
from ensemblecalibration.data.experiments_cal_test import get_experiment
from ensemblecalibration.data.ensemble.dataset_utils import load_results, load_dataset


def test_MLPCalWConv():
    # generate data
    config = {
        "experiment": "gp",
        "params": {"n_samples": 100, "bounds_p": [0, 1], "deg": 2},
    }

    # load parts of CIFAR 10 dataset
    trainloader, valloader, testloader, n_classes = load_dataset(
        "CIFAR10", batch_size=128
    )
    # get sample from CIFAR10 tot est as input from model
    x_inst, y_labels = next(iter(trainloader))

    # create model
    model = MLPCalWConv()
    # forward pass
    out = model(x_inst)
    print(out.shape)
    assert out.shape == (128, 10)

