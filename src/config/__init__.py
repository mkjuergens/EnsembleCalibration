from ensemblecalibration.config.config_cal_test import *
from ensemblecalibration.config.config_recal import *

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config