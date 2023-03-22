from copy import deepcopy

from ensemblecalibration.calibration.distances import tv_distance
from ensemblecalibration.calibration.cal_tests import skceul, confece, classece, hl
from ensemblecalibration.calibration.cal_tests import skce_ul_obj, confece_obj, classece_obj, hl_obj
from ensemblecalibration.calibration.test_objectives import confece_obj_new, classece_obj_new, hl_obj_new, skce_ul_obj_new
from ensemblecalibration.calibration.cal_tests import  _npbetest_alpha
from ensemblecalibration.calibration.cal_test_new import _npbe_test_new_alpha

config_tests_new_cobyla_2d = {

    "HL5": {
        "test": _npbe_test_new_alpha,
        "params": {
            "l_prior": 1,
            "optim": "cobyla",
            "n_resamples": 100,
            "n_bins": 5,
            "test": hl, # test used for the calibration measure of the #perfectly# calibrated model
            "obj": hl_obj_new, # objective function for the minimzation part
            "x_dependency": True

        },
    },

    "HL10": {
        "test": _npbe_test_new_alpha,
        "params": {
            "l_prior": 1,
            "optim": "cobyla",
            "n_resamples": 100,
            "n_bins": 10,
            "test": hl, # test used for the calibration measure of the #perfectly# calibrated model
            "obj": hl_obj_new, # objective function for the minimzation part
            "x_dependency": True

        },
    },

     "SKCEul": {
        "test": _npbe_test_new_alpha,
        "params": {
            "l_prior": 1, 
            "optim": "cobyla",
            "n_resamples": 100,
            "dist": tv_distance,
            "sigma": 2.0, # to be used in the matrix valued kernel
            "test": skceul,
            "obj": skce_ul_obj_new,
            "x_dependency": True
        }
    },

    "ECEconf5": {
        "test": _npbe_test_new_alpha,
        "params": {
            "l_prior": 1,
            "optim": "cobyla",
            "n_resamples": 100,
            "n_bins": 5,
            "test": confece,
            "obj": confece_obj_new,
            "x_dependency": True
        }
    },

    "ECEconf10": {
        "test": _npbe_test_new_alpha,
        "params": {
            "l_prior": 1,
            "optim": "cobyla",
            "n_resamples": 100,
            "n_bins": 10,
            "test": confece,
            "obj": confece_obj_new,
            "x_dependency": True
        }
    },

    "ECEclass5": {
        "test": _npbe_test_new_alpha,
        "params": {
            "l_prior": 1,
            "optim": "cobyla",
            "n_resamples": 100,
            "n_bins": 5,
            "test": classece,
            "obj": classece_obj_new,
            "x_dependency": True
        }
    },

    "ECEclass10": {
        "test": _npbe_test_new_alpha,
        "params": {
            "l_prior": 1,
            "optim": "cobyla",
            "n_resamples": 100,
            "n_bins": 10,
            "test": classece,
            "obj": classece_obj_new,
            "x_dependency": True
        }
    },

}

config_tests_new_cobyla_1d = {

    "HL5": {
        "test": _npbe_test_new_alpha,
        "params": {
            "l_prior": 1,
            "optim": "cobyla",
            "n_resamples": 100,
            "n_bins": 5,
            "test": hl, # test used for the calibration measure of the #perfectly# calibrated model
            "obj": hl_obj, # objective function for the minimzation part
            "x_dependency": False

        },
    },

    "HL10": {
        "test": _npbe_test_new_alpha,
        "params": {
            "l_prior": 1,
            "optim": "cobyla",
            "n_resamples": 100,
            "n_bins": 10,
            "test": hl, # test used for the calibration measure of the #perfectly# calibrated model
            "obj": hl_obj, # objective function for the minimzation part
            "x_dependency": False

        },
    },

     "SKCEul": {
        "test": _npbe_test_new_alpha,
        "params": {
            "l_prior": 1, 
            "optim": "cobyla",
            "n_resamples": 100,
            "dist": tv_distance,
            "sigma": 2.0, # to be used in the matrix valued kernel
            "test": skceul,
            "obj": skce_ul_obj,
            "x_dependency": False
        }
    },

    "ECEconf5": {
        "test": _npbe_test_new_alpha,
        "params": {
            "l_prior": 1,
            "optim": "cobyla",
            "n_resamples": 100,
            "n_bins": 5,
            "test": confece,
            "obj": confece_obj,
            "x_dependency": False
        }
    },

    "ECEconf10": {
        "test": _npbe_test_new_alpha,
        "params": {
            "l_prior": 1,
            "optim": "cobyla",
            "n_resamples": 100,
            "n_bins": 10,
            "test": confece,
            "obj": confece_obj,
            "x_dependency": False
        }
    },

    "ECEclass5": {
        "test": _npbe_test_new_alpha,
        "params": {
            "l_prior": 1,
            "optim": "cobyla",
            "n_resamples": 100,
            "n_bins": 5,
            "test": classece,
            "obj": classece_obj,
            "x_dependency": False
        }
    },

    "ECEclass10": {
        "test": _npbe_test_new_alpha,
        "params": {
            "l_prior": 1,
            "optim": "cobyla",
            "n_resamples": 100,
            "n_bins": 10,
            "test": classece,
            "obj": classece_obj,
            "x_dependency": False
        }
    },

}


### configuration with new test, nelder-mead, x dependecy
config_tests_new_neldermead_2d = deepcopy(config_tests_new_cobyla_2d)
for test in config_tests_new_neldermead_2d:
    config_tests_new_neldermead_2d[test]["params"]["optim"] = "neldermead"

### configuration with new test, nelder-mead, no x dependecy
config_tests_new_neldermead_1d = deepcopy(config_tests_new_cobyla_1d)
for test in config_tests_new_neldermead_1d:
    config_tests_new_neldermead_1d[test]["params"]["optim"] = "neldermead"


config_tests_cobyla_2d= {

    "HL5": {
            "test": _npbetest_alpha,
            "params": {
                "sampling": "lambda",
                "l_prior": 1,
                "optim": "cobyla", 
                "n_resamples": 100, 
                "n_bins": 5,
                "test": hl, 
                "obj": hl_obj_new,
                "transform": "additive",
                "x_dependency": True
                
                },
        },
    "HL10": {
        "test": _npbetest_alpha,
        "params": {
            "sampling": "lambda",
            "l_prior": 1,
            "optim": "cobyla", 
            "n_resamples": 100, 
            "n_bins": 10,
            "test": hl, 
            "obj": hl_obj_new,
            "transform": "additive",
            "x_dependency": True
            
    },

        }, 
    "SKCEul":{
        "test": _npbetest_alpha,
        "params": {
            "sampling": "lambda", #  options: mcmc, mcmc, rejectance,
            "l_prior": 1,
            "optim": "cobyla", # TODO: add other option here
            "n_resamples": 100,
            "dist": tv_distance,
            "sigma": 2.0, # to be used in the matrix valued kernel
            "obj": skce_ul_obj_new,
            "test": skceul,
            "transform": 'additive',
            "x_dependency": True
 }
    },

    "CONFECE5":{
        "test": _npbetest_alpha,
        "params": {
            "sampling": "lambda", # other options: mcmc, rejectance,
            "l_prior": 1,
            "optim": "cobyla",
            "n_resamples": 100,
            "n_bins": 5,
            "obj": confece_obj_new,
            "test": confece,
            "transform": 'additive', # needs to be in ['sqrt', 'additive', 'isometric'],
            "x_dependency": True
            }
    },
    "CONFECE10":{
        "test": _npbetest_alpha,
        "params": {
            "sampling": "lambda", # other options: mcmc, rejectance,
            "l_prior": 1,
            "optim": "cobyla",
            "n_resamples": 100,
            "n_bins": 10,
            "obj": confece_obj_new,
            "test": confece,
            "transform": 'additive', # needs to be in ['sqrt', 'additive', 'isometric'],
            "x_dependency": True
            }
    },
    "CLASSECE5":{
        "test": _npbetest_alpha,
        "params": {
            "sampling": "lambda", # options: lambda, mcmc, rejectance,
            "l_prior": 1,
            "optim": "cobyla",
            "n_resamples": 100,
            "n_bins": 5,
            "obj": classece_obj_new,
            "test": classece,
            "transform": 'additive', # needs to be in ['sqrt', 'additive', 'isometric'],
            "x_dependency": True
        }
    },

    "CLASSECE10":{
        "test": _npbetest_alpha,
        "params": {
            "sampling": "lambda", # other options: mcmc, rejectance,
            "l_prior": 1,
            "optim": "cobyla",
            "n_resamples": 100,
            "n_bins": 10,
            "obj": classece_obj_new,
            "test": classece,
            "transform": 'additive', # needs to be in ['sqrt', 'additive', 'isometric'],
            "x_dependency": True
        }
    }
    } 


config_tests_cobyla_1d= {

    "HL5": {
            "test": _npbetest_alpha,
            "params": {
                "sampling": "lambda",
                "l_prior": 1,
                "optim": "cobyla", 
                "n_resamples": 100, 
                "n_bins": 5,
                "test": hl, 
                "obj": hl_obj,
                "transform": "additive",
                "x_dependency": False
                
                },
        },
    "HL10": {
        "test": _npbetest_alpha,
        "params": {
            "sampling": "lambda",
            "l_prior": 1,
            "optim": "cobyla", 
            "n_resamples": 100, 
            "n_bins": 10,
            "test": hl, 
            "obj": hl_obj,
            "transform": "additive",
            "x_dependency": False
            
    },

        }, 
    "SKCEul":{
        "test": _npbetest_alpha,
        "params": {
            "sampling": "lambda", #  options: mcmc, mcmc, rejectance,
            "l_prior": 1,
            "optim": "cobyla", # TODO: add other option here
            "n_resamples": 100,
            "dist": tv_distance,
            "sigma": 2.0, # to be used in the matrix valued kernel
            "obj": skce_ul_obj,
            "test": skceul,
            "transform": 'additive',
            "x_dependency": False
 }
    },

    "CONFECE5":{
        "test": _npbetest_alpha,
        "params": {
            "sampling": "lambda", # other options: mcmc, rejectance,
            "l_prior": 1,
            "optim": "cobyla",
            "n_resamples": 100,
            "n_bins": 5,
            "obj": confece_obj,
            "test": confece,
            "transform": 'additive', # needs to be in ['sqrt', 'additive', 'isometric'],
            "x_dependency": False
            }
    },
    "CONFECE10":{
        "test": _npbetest_alpha,
        "params": {
            "sampling": "lambda", # other options: mcmc, rejectance,
            "l_prior": 1,
            "optim": "cobyla",
            "n_resamples": 100,
            "n_bins": 10,
            "obj": confece_obj,
            "test": confece,
            "transform": 'additive', # needs to be in ['sqrt', 'additive', 'isometric'],
            "x_dependency": False
            }
    },
    "CLASSECE5":{
        "test": _npbetest_alpha,
        "params": {
            "sampling": "lambda", # options: lambda, mcmc, rejectance,
            "l_prior": 1,
            "optim": "cobyla",
            "n_resamples": 100,
            "n_bins": 5,
            "obj": classece_obj,
            "test": classece,
            "transform": 'additive', # needs to be in ['sqrt', 'additive', 'isometric'],
            "x_dependency": False
        }
    },

    "CLASSECE10":{
        "test": _npbetest_alpha,
        "params": {
            "sampling": "lambda", # other options: mcmc, rejectance,
            "l_prior": 1,
            "optim": "cobyla",
            "n_resamples": 100,
            "n_bins": 10,
            "obj": classece_obj,
            "test": classece,
            "transform": 'additive', # needs to be in ['sqrt', 'additive', 'isometric'],
            "x_dependency": False
        }
    }
    } 



### configuration with x dependency, nelder-mead, old test
config_tests_neldermead_2d = deepcopy(config_tests_cobyla_2d)
for test in config_tests_neldermead_2d:
    config_tests_neldermead_2d[test]["params"]["optim"] = "neldermead"

### configuration without x dependency, nelder-mead, old test
config_tests_neldermead_1d = deepcopy(config_tests_cobyla_1d)
for test in config_tests_neldermead_1d:
    config_tests_neldermead_1d[test]["params"]["optim"] = "neldermead"



config_tests_reduced = {

    "CONFECE5":{
        "test": _npbetest_alpha,
        "params": {
            "sampling": "lambda", # other options: mcmc, rejectance,
            "l_prior": 1,
            "optim": "cobyla",
            "n_resamples": 100,
            "n_bins": 5,
            "obj": confece_obj,
            "test": confece,
            "transform": 'additive' # needs to be in ['sqrt', 'additive', 'isometric'],
                                # only to be used for mcmc sampling
            }
    },
    "CLASSECE5":{
        "test": _npbetest_alpha,
        "params": {
            "sampling": "lambda", # other options: mcmc, rejectance,
            "l_prior": 1,
            "optim": "cobyla",
            "n_resamples": 100,
            "n_bins": 5,
            "obj": classece_obj,
            "test": classece,
            "transform": 'additive' # needs to be in ['sqrt', 'additive', 'isometric'],
                                # only to be used for mcmc sampling
        }
    },

    "CLASSECE10":{
        "test": _npbetest_alpha,
        "params": {
            "sampling": "lambda", # other options: mcmc, rejectance,
            "l_prior": 1,
            "optim": "cobyla",
            "n_resamples": 100,
            "n_bins": 10,
            "obj": classece_obj,
            "test": classece,
            "transform": 'additive' # needs to be in ['sqrt', 'additive', 'isometric'],
                                # only to be used for mcmc sampling
        }
    },
    "CONFECE10":{
        "test": _npbetest_alpha,
        "params": {
            "sampling": "lambda", # other options: mcmc, rejectance,
            "l_prior": 1,
            "optim": "cobyla",
            "n_resamples": 100,
            "n_bins": 10,
            "obj": confece_obj,
            "test": confece,
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
    "alpha": [0.05, 0.13, 0.21, 0.30, 0.38, 0.46, 0.54, 0.62, 0.70, 0.78, 0.87, 0.95]
}

settings_2 = {
    "N": [100],
    "M": [10],
    "K": [3],
    "R": [100],
    "u": [0.01],
    "alpha": [0.05]
},

settings_3 = {
    "N": [100],
    "M": [10],
    "K": [3],
    "R": [100],
    "u": [0.01],
    "alpha": [0.05, 0.2, 0.5]
}

if __name__ == "__main__":
    conf = config_tests_cobyla_2d
    for i in range(len(list(conf.keys()))):
        conf[list(conf.keys())[i]]["params"]["sampling"] = 'lambda'
    print(conf[list(config_tests_cobyla_2d.keys())[0]]["params"]["sampling"])
    conf_2 = config_tests_new_neldermead_1d
    print(conf_2)
    