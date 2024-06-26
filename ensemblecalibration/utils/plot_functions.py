from typing import Optional
import numpy as np
import pandas as pd
import torch
from ast import literal_eval
import matplotlib.pyplot as plt
import ternary
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull


from ensemblecalibration.utils.projections import project_points2D


def plot_t1_erros_analysis(
    df: pd.DataFrame,
    list_errors: list,
    take_avg: bool = False,
    plot_ha: bool = False,
    figsize: tuple = (8, 12),
    title: Optional[str] = None,
    list_col_titles: list = ["S_0", "S_1", "S_2", "S_3"],
    n_type_1: int = 1,
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
            ax[j].set_title(f"{list_errors[j]}", fontsize=15)
            ax[j].set_xlabel(r"$\alpha$")
            # y label closer to plot
            ax[j].yaxis.set_label_coords(-0.1, 0.5)
            ax[j].set_ylabel(r"Type $1$ error", fontsize=14)
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
                    if j in range(n_type_1):
                        ax[i, j].plot(alphas, alphas, "--")
                    # ax[i, j].spines[["right", "top"]].set_visible(False)
                    ax[i, j].grid()

            for i, col_title in enumerate(list_col_titles):
                ax[0, i].set_title(col_title, fontsize=15)
        else:
            ax[0].set_ylabel(f"{list_errors[0]}")
            for j in range(len(df)):
                ax[j].plot(alphas, literal_eval(df[list_errors[0]][j]))
                if j == 0:
                    ax[j].plot(alphas, alphas, "--")
                ax[j].spines[["right", "top"]].set_visible(False)
                ax[j].grid()
    fig.supxlabel(r"$\alpha$", fontsize=15)
    fig.supylabel(r"Type $1$/$2$ error", fontsize=15)
    # plt.tight_layout()
    if title is not None:
        plt.suptitle(title, fontsize=16)
    return fig


def plot_heatmap_dirichlet_2D(alpha: torch.tensor, scale: int = 100):
    """plots a simplex of the Dirichlet distribution with the specified alpha parameters.

    Parameters
    ----------
    alpha : torch.tensor of shape (3,)
        parameters of the Dirichlet distribution
    scale : int, optional
       resolution of the heatmap, by default 100

    Returns
    -------
    plt.figure
        figure of the heatmap
    """

    heatmap_dict = {}
    for i in range(scale):
        for j in range(scale - i):
            k = scale - i - j - 1
            if k >= 0:
                point = torch.tensor([i, j, k]) / (scale - 1)
                pdf_value = (
                    torch.distributions.Dirichlet(alpha).log_prob(point).exp().item()
                )
                heatmap_dict[(i, j, k)] = pdf_value

    # Initialize the ternary plot
    figure, tax = ternary.figure(scale=scale - 1)

    # Plot the heatmap
    tax.heatmap(
        heatmap_dict,
        scale=scale - 1,
        cmap="viridis",
        style="hexagonal",
        scientific=True,
    )

    # Customize the plot
    tax.boundary()
    tax.gridlines(color="black", multiple=10)
    tax.set_title("Heatmap of Dirichlet Distribution")

    # Set axis labels
    fontsize = 12
    tax.left_axis_label("X", fontsize=fontsize)
    tax.right_axis_label("Y", fontsize=fontsize)
    tax.bottom_axis_label("Z", fontsize=fontsize)

    # return the figure
    return figure


def plot_polytope_2D(points: np.ndarray, title: str, extra_points: np.ndarray = None):

    # project points to 2D plane
    points_2d = project_points2D(points)
    # create convex hull
    hull = ConvexHull(points_2d)
    # get vertices of the convex hull
    hull_vertices = hull.vertices
    # intialize ternary plot
    figure, tax = ternary.figure(scale=1.0)
    tax.boundary()
    tax.gridlines(color="black", multiple=0.1)

    # Plot the points
    tax.scatter(
        list(map(tuple, points)), marker="o", color="red", label="Polytope Vertices"
    )
    # Create a polygon patch
    polygon = Polygon(
        list(map(tuple, project_points2D(hull_vertices))),
        closed=True,
        color="grey",
        alpha=0.5,
    )
    tax.get_axes().add_patch(polygon)

    # Plot the boundary edges from the Convex Hull
    for simplex in hull.simplices:
        point1 = tuple(points[simplex[0]])
        point2 = tuple(points[simplex[1]])
        tax.line(point1, point2, linewidth=1.0, color="red", alpha=0.5)

    # remove outer axes
    tax.clear_matplotlib_ticks()
