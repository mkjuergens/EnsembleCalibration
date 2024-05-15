import numpy as np
import torch

from ensemblecalibration.utils.minimization import calculate_min
from ensemblecalibration.cal_test import npbe_test_ensemble



class CalibrationTestNPBE:

    def __init__(self, params: dict):
        self.params = params
        self.results = {}
        self.test = params