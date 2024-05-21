from typing import Optional
import numpy as np
import pandas as pd
from ast import literal_eval
import matplotlib.pyplot as plt

def plot_t1_erros_analysis(
    df: pd.DataFrame,
    list_errors: list,
    take_avg: bool = False,
    plot_ha: bool = False,
    figsize: tuple = (8, 12),
    title: Optional[str] = None,
):
    if "alpha" in df:
        alphas = df["alpha"].values
    else:
        alphas = np.array(
            [0.05, 0.13, 0.21, 0.30, 0.38, 0.46, 0.54, 0.62, 0.70, 0.78, 0.87, 0.95]
        )
    if take_avg:
        results = np.zeros((len(list_errors), len(df)))
    else:
        results = np.zeros((len(list_errors), len(alphas)))
    for i in range(len(list_errors)):
        results_i = df[list_errors[i]]
        if take_avg:
            for j in range(len(df)):
                val_ij = sum(literal_eval(results_i[j])) / len(
                    literal_eval(results_i[j])
                )
                results[i, j] = val_ij
        else:  # averagea are already saved in the dataframe
            for j in range(len(alphas)):
                val_ij = literal_eval(results_i[0])[j]
                results[i, j] = val_ij
    if not plot_ha:
        fig, ax = plt.subplots(len(list_errors), 1, figsize=figsize)
        for j in range(len(list_errors)):
            ax[j].plot(alphas, results[j])
            ax[j].plot(alphas, alphas, "--")
            ax[j].set_title(f"{list_errors[j]}")
            ax[j].set_xlabel(r"$\alpha$")
            ax[j].set_ylabel(r"Type $1$ error")
            ax[j].grid()
    else:
        fig, ax = plt.subplots(
            len(list_errors), len(df), figsize=figsize, sharex=True, sharey=True
        )
        if len(list_errors) > 1:
            for i in range(len(list_errors)):
                ax[i, 0].set_ylabel(f"{list_errors[i]}")
                for j in range(len(df)):
                    ax[i, j].plot(alphas, literal_eval(df[list_errors[i]][j]))
                    if j == 0:
                        ax[i, j].plot(alphas, alphas, "--")
                    #ax[i, j].spines[["right", "top"]].set_visible(False)
                    ax[i, j].grid()

            for i in range(len(df)):
                ax[0, i].set_title(f"S_{i}")
        else:
            ax[0].set_ylabel(f"{list_errors[0]}")
            for j in range(len(df)):
                ax[j].plot(alphas, literal_eval(df[list_errors[0]][j]))
                if j == 0:
                    ax[j].plot(alphas, alphas, "--")
                ax[j].spines[["right", "top"]].set_visible(False)
                ax[j].grid()
    fig.supxlabel(r"$\alpha$")
    fig.supylabel(r"Type $1$/$2$ error")
    plt.tight_layout()
    if title is not None:
        plt.suptitle(title)
    return fig
