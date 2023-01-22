from ensemblecalibration.calibration.distances import tv_distance, l2_distance
from ensemblecalibration.calibration.cal_tests import skceul, skceuq, confece, classece
from ensemblecalibration.calibration.cal_tests import skce_ul_obj, skce_uq_obj, confece_obj, classece_obj
from ensemblecalibration.calibration.cal_tests import _npbe_test
config_tests= {


    "SKCEul":{
        "test": _npbe_test,
        "params": {
            "sampling": "rejectance", # other options: mcmc, rejectance,
            "optim": "cobyla",
            "n_resamples": 100,
            "dist": tv_distance,
            "sigma": 2.0, # to be used in the matrix valued kernel
            "obj": skce_ul_obj,
            "test": skceul,
            "transform": 'additive'
 }
    },

    "CONFECE":{
        "test": _npbe_test,
        "params": {
            "sampling": "rejectance", # other options: mcmc, rejectance,
            "optim": "cobyla",
            "n_resamples": 100,
            "n_bins": 15,
            "obj": confece_obj,
            "test": confece,
            "transform": 'additive' # needs to be in ['sqrt', 'additive', 'isometric'],
                                # only to be used for mcmc sampling
            }
    },
    "CLASSECE":{
        "test": _npbe_test,
        "params": {
            "sampling": "rejectance", # other options: mcmc, rejectance,
            "optim": "cobyla",
            "n_resamples": 100,
            "n_bins": 15,
            "obj": classece_obj,
            "test": classece,
            "transform": 'additive' # needs to be in ['sqrt', 'additive', 'isometric'],
                                # only to be used for mcmc sampling
        }
    }
    } 

settings = {
    "N": [100],
    "M": [10],
    "K": [3],
    "R": [1000],
    "u": [0.01],
    "alpha": [0.05, 0.20, 0.50]
}

if __name__ == "__main__":
    test = 'CONFECE'