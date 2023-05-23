"""Usefule functions for plotting"""
from typing import Optional
from ast import literal_eval

import numpy as np
import pandas as pd
import ternary
import matplotlib
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
                    ax[i, j].spines[["right", "top"]].set_visible(False)
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


def plot_heatmap_analysis(
    res: dict, params: dict, l_real: np.ndarray, figsize: tuple = (15, 6)
):
    """function for plotting the results of the distance analysis as heatmaps for each test

    Parameters
    ----------
    res : dict
        dictionary containing the results of the distance analysis, as returned by run_distance_analysis
    params : dict
        dictionary containing the parameters of the test
    L_real : np.ndarray
        array of shape (n_ensembles,) containing the real weights
    figsize : tuple, optional
        size of the figure, by default (15, 6)

    Returns
    -------
    fig
        matplotlib figure
    """

    n_cols = len(res) // 2
    n_rows = len(res) // n_cols
    fig, ax = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=True, sharey=True)

    for i, test in enumerate(params):
        # set color scheme dependent of value of
        col = res[test]["stats"]

        tax = ternary.TernaryAxesSubplot(ax=ax[i // n_cols, i % n_cols])
        tax.gridlines(color="blue", multiple=0.2, linewidth=0.5)
        tax.set_background_color(color="whitesmoke")
        # plot real lambda in each plot
        tax.scatter(
            [tuple(l_real)], color="red", s=40, label="chosen $\lambda$", zorder=2
        )
        # use logarithmic transformation for colorbar
        tax.scatter(
            [tuple(res[test]["l_all"][i]) for i in range(len(res[test]["l_all"]))],
            alpha=0.5,
            c=col,
            colormap=plt.cm.viridis,
            colorbar=True,
            vmax=max(col),
            norm=matplotlib.colors.SymLogNorm(linthresh=0.05),
        )
        ax[i // n_cols, i % n_cols].set_title(test)

    plt.legend()
    plt.tight_layout()

    return fig


def plot_scatterplot_ternary(
    list_points: list,
    list_cols: Optional[list] = None,
    title: Optional[str] = None,
    scale: int = 1
):
    """function for plotting a scatterplot of points in the 3-simplex

    Parameters
    ----------
    list_points : list of arrays
        list of points  to plot
    list_cols : Optional[list], optional
        list of colors for each point, by default None. If None, default matplotlib colors are used
    title : Optional[str], optional
        title of the figure, by default None
    scale : int, optional
        scale of the figure, by default 1

    Returns
    -------
    fig
        matplotlib figure
    """

    fig, tax = ternary.figure(scale=scale)

    tax.boundary(linewidth=1.5)
    tax.gridlines(color="blue", multiple=0.2, linewidth=0.5)
    tax.set_background_color(color="whitesmoke", alpha=0.5)
    if list_cols is None:
        list_cols = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for idx, arr in enumerate(list_points):
        tax.scatter(
            [tuple(arr[i]) for i in range(len(arr))],
            marker="o",
            color=list_cols[idx],
            alpha=0.5,
            label=f"points {idx}",
            zorder=2,
        )

    tax.legend()
    tax.clear_matplotlib_ticks()
    tax.get_axes().axis("off")
    plt.tight_layout()
    if title is not None:
        plt.title(title)

    return fig


if __name__ == "__main__":
    results_final = pd.read_csv(
        "final_results_experiments_t1t2_alpha_100_10_10_0.01_lambda.csv"
    )
    list_errors_results = list(results_final.keys())
