from copy import deepcopy

from ensemblecalibration.calibration.calibration_estimates.distances import tv_distance

from ensemblecalibration.calibration.calibration_estimates import *
from ensemblecalibration.calibration.cal_tests import  _npbetest_alpha
from ensemblecalibration.calibration.cal_test_new import _npbe_test_new_alpha, _npbe_test_v3_alpha
from ensemblecalibration.calibration.p_value_analysis import npbe_test_p_values, npbe_test_v3_p_values
from ensemblecalibration.nn_training.losses import SKCELoss
from ensemblecalibration.nn_training.distances import tv_distance_tensor, skce_ul_tensor, skce_uq_tensor

config_p_value_analysis = {
    "SKCEul2": {
    "test": npbe_test_v3_p_values,
    "params": {
        "take_square": True,
        "n_predictors": 100,
        "l_prior": 1, 
        "optim": "cobyla",
        "n_resamples": 1000,
        "dist": tv_distance,
        "sigma": 2.0, # to be used in the matrix valued kernel
        "test": skceul,
        "obj": skce_ul_obj,
        "obj_lambda": skce_ul_obj_lambda,
        "sampling": "lambda",
        "transform": "isometric",
        "x_dependency": False,
        "alpha": 0.05,
    },
    },
    "SKCEul": {
    "test": npbe_test_v3_p_values,
    "params": {
        "take_square": False,
        "n_predictors": 100,
        "l_prior": 1, 
        "optim": "cobyla",
        "n_resamples": 1000,
        "dist": tv_distance,
        "sigma": 2.0, # to be used in the matrix valued kernel
        "test": skceul,
        "obj": skce_ul_obj,
        "obj_lambda": skce_ul_obj_lambda,
        "sampling": "lambda",
        "transform": "isometric",
        "x_dependency": False,
        "alpha": 0.05,
    },
    },
    "HL5": {
    "test": npbe_test_v3_p_values,
        "params": {
            "n_predictors": 100,
            "l_prior": 1,
            "optim": "cobyla",
            "n_resamples": 1000,
            "n_bins": 5,
            "test": hltest, # test used for the calibration measure of the #perfectly# calibrated model
            "obj": hl_obj, # objective function for the minimzation part
            "obj_lambda": hl_obj_lambda,
            "sampling": "lambda",
            "x_dependency": False,
            "alpha": 0.05,

        },
    },

    "HL10": {
    "test": npbe_test_v3_p_values,
        "params": {
            "n_predictors": 100,
            "l_prior": 1,
            "optim": "cobyla",
            "n_resamples": 1000,
            "n_bins": 10,
            "test": hltest, # test used for the calibration measure of the #perfectly# calibrated model
            "obj": hl_obj, # objective function for the minimzation part
            "obj_lambda": hl_obj_lambda,
            "sampling": "lambda",
            "x_dependency": False,
            "alpha": 0.05,

        },
    },
    "ECEconf5": {
    "test": npbe_test_v3_p_values,
        "params": {
            "n_predictors": 100, # bad wording here, should be n_iterations
            "l_prior": 1,
            "optim": "cobyla",
            "n_resamples": 1000, # for bootstrapping the null distribution
            "n_bins": 5,
            "test": confece,
            "obj": confece_obj,
            "obj_lambda": confece_obj_lambda,
            "sampling": "lambda",
            "x_dependency": False,
            "alpha": 0.05,
        }
    },

    "ECEconf10": {
    "test": npbe_test_v3_p_values,
        "params": {
            "n_predictors": 100,
            "l_prior": 1,
            "optim": "cobyla",
            "n_resamples": 1000,
            "n_bins": 10,
            "test": confece,
            "obj": confece_obj,
            "obj_lambda": confece_obj_lambda,
            "sampling": "lambda",
            "x_dependency": False,
            "alpha": 0.05,
        }
    },

    "ECEclass5": {
    "test": npbe_test_v3_p_values,
        "params": {
            "n_predictors": 1000,
            "l_prior": 1,
            "optim": "cobyla",
            "n_resamples": 1000,
            "n_bins": 5,
            "test": classece,
            "obj": classece_obj,
            "obj_lambda": classece_obj_lambda,
            "sampling": "lambda",
            "x_dependency": False,
            "alpha": 0.05,
        }
    },

    "ECEclass10": {
    "test": npbe_test_v3_p_values,
        "params": {
            "n_predictors": 100,
            "l_prior": 1, 
            "optim": "cobyla",
            "n_resamples": 1000,
            "n_bins": 10,
            "test": classece,
            "obj": classece_obj,
            "obj_lambda": classece_obj_lambda,
            "sampling": "lambda",
            "x_dependency": False,
            "alpha": 0.05,
        }
    },
}

config_new_v3 = {
    "SKCEul2": {
        "test": _npbe_test_v3_alpha,
        "params": {
            "take_square": True,
            "l_prior": 1,
            "optim": "cobyla",
            "n_resamples": 100,
            "n_predictors": 100, # number of "guesses" for the c.c.
            "dist": tv_distance,
            "sigma": 2.0, # to be used in the matrix valued kernel
            "test": skceul, # test used for the calibration measure of the #perfectly# calibrated model
            "obj": skce_ul_obj, # objective function for the minimzation part
            "sampling": "lambda",
            "transform": "sqrt",
            "x_dependency": False
        }
    },
    "SKCEul": {
        "test": _npbe_test_v3_alpha,
        "params": {
            "l_prior": 1, 
            "optim": "cobyla",
            "n_resamples": 100,
            "n_predictors": 100,
            "dist": tv_distance,
            "sigma": 2.0, # to be used in the matrix valued kernel
            "test": skceul,
            "obj": skce_ul_obj_new,
            "sampling": "lambda",
            "transform": "sqrt",
            "x_dependency": False
        }
    },
    "ECEconf5": {
        "test": _npbe_test_v3_alpha,
        "params": {
            "l_prior": 1,
            "optim": "cobyla",
            "n_resamples": 100,
            "n_predictors": 100,
            "n_bins": 5,
            "test": confece,
            "obj": confece_obj,
            "sampling": "lambda",
            "transform": "sqrt",
            "x_dependency": False
        }      
},
    "ECEconf10": {
        "test": _npbe_test_v3_alpha,
        "params": {
            "l_prior": 1,
            "optim": "cobyla",
            "n_resamples": 100,
            "n_predictors": 100,
            "n_bins": 10,
            "test": confece,
            "obj": confece_obj,
            "sampling": "lambda",
            "transform": "sqrt",
            "x_dependency": False
        }
    },
    "ECEclass5": {
        "test": _npbe_test_v3_alpha,
        "params": {
            "l_prior": 1,
            "optim": "cobyla",
            "n_resamples": 100,
            "n_predictors": 100,
            "n_bins": 5,
            "test": classece,
            "obj": classece_obj,
            "sampling": "lambda",
            "transform": "sqrt",
            "x_dependency": False
        }
    },
    "ECEclass10": {
        "test": _npbe_test_v3_alpha,
        "params": {
            "l_prior": 1,
            "optim": "cobyla",
            "n_resamples": 100,
            "n_predictors": 100,
            "n_bins": 10,
            "test": classece,
            "obj": classece_obj,
            "sampling": "lambda",
            "transform": "sqrt",
            "x_dependency": False
        }
    }

}

config_new_mlp = {
    "SKCEuq": {
        "test": _npbe_test_new_alpha,
        "params": {
            "l_prior": 1, 
            "optim": "mlp",
            "n_resamples": 100,
            "dist": tv_distance,
            "sigma": 2.0, # to be used in the matrix valued kernel
            "test": skceul,
            "obj": skce_ul_obj_new,
            "loss": SKCELoss(use_median_bw=True, dist_fct=tv_distance_tensor, 
                             tensor_miscal=skce_uq_tensor),
            "n_epochs": 100,
            "lr": 0.001,
            "x_dependency": True
        }
    },
    "SKCEul": {
        "test": _npbe_test_new_alpha,
        "params": {
            "l_prior": 1, 
            "optim": "mlp",
            "n_resamples": 100,
            "dist": tv_distance,
            "sigma": 2.0, # to be used in the matrix valued kernel
            "test": skceul,
            "obj": skce_ul_obj_new,
            "loss": SKCELoss(use_median_bw=True, dist_fct=tv_distance_tensor, 
                             tensor_miscal=skce_ul_tensor),
            "n_epochs": 100,
            "lr": 0.001,
            "x_dependency": True
        }
    }      
}

config_tests_new_cobyla_2d = {

    "HL5": {
        "test": _npbe_test_new_alpha,
        "params": {
            "l_prior": 1,
            "optim": "cobyla",
            "n_resamples": 100,
            "n_bins": 5,
            "test": hltest, # test used for the calibration measure of the #perfectly# calibrated model
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
            "test": hltest, # test used for the calibration measure of the #perfectly# calibrated model
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
            "test": hltest, # test used for the calibration measure of the #perfectly# calibrated model
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
            "test": hltest, # test used for the calibration measure of the #perfectly# calibrated model
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
                "test": hltest, 
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
            "test": hltest, 
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
                "test": hltest, 
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
            "test": hltest, 
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
    