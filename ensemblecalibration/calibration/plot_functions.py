from ast import literal_eval

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_t1_erros_analysis(df: pd.DataFrame, list_errors: list = ['CONFECE', 'CLASSECE'], sampling_method: str = 'lambda',
                            take_avg: bool = True, plot_ha: bool = False):

    alphas = df['alpha'].values
    results = np.zeros((len(list_errors), len(df)))
    for i in range(len(list_errors)):
        results_i = df[list_errors[i]]
        for j in range(len(df)):
            if take_avg:
                val_ij = (sum(literal_eval(results_i[j]))/len(literal_eval(results_i[j])))
            else: # averagea are already saved in the dataframe
                val_ij = results_i[j]
            results[i, j] = val_ij

    
    if not plot_ha:
        fig, ax = plt.subplots(len(list_errors), 1, figsize=(8, 12))
        for j in range(len(list_errors)):
            ax[j].plot(alphas, results[j])
            ax[j].plot(alphas, alphas, '--')
            ax[j].set_title(f'{list_errors[j]}')
            ax[j].set_xlabel(r'$\alpha$')
            ax[j].set_ylabel(r'Type $1$ error')
            ax[j].grid()
        
        plt.suptitle(r'Type $1$ error analysis, sampling: {}'.format(sampling_method))

    else:
        raise NotImplementedError

    return fig