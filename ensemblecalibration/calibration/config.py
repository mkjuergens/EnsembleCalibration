from ensemblecalibration.calibration.distances import tv_distance, l2_distance
from ensemblecalibration.calibration.cal_tests import skce_ul_obj, skce_uq_obj, hl_obj, classece_obj, confece_obj
from ensemblecalibration.calibration.cal_tests import skceul, skceuq, confece, classece

config_tests= {


    "SKCEul":{
        "test": "npce",
        "params": {
            "sampling": "lambda", # other options: mcmc, rejectance,
            "optim": "cobyla",
            "dist": tv_distance,
            "sigma": 2,
            "obj": skce_ul_obj,
            "test": skceul
            }
    },
    "SKCEuq":{
        "test": "npce",
        "params": {
            "sampling": "lambda", # other options: mcmc, rejectance,
            "optim": "cobyla",
            "dist": tv_distance,
            "sigma": 2,
            "obj": skce_uq_obj,
            "test": skceuq
            }
    },
    "CONFECE":{
        "test": "npce",
        "params": {
            "sampling": "lambda", # other options: mcmc, rejectance,
            "optim": "cobyla",
            "dist": tv_distance,
            "sigma": 2,
            "obj": confece_obj,
            "test": confece
            }
    },
    "CLASSECE":{
        "test": "npce",
        "params": {
            "sampling": "lambda", # other options: mcmc, rejectance,
            "optim": "cobyla",
            "dist": tv_distance,
            "sigma": 2,
            "obj": confece_obj,
            "test": confece
        }
    }
    } 