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
from ensemblecalibration.utils.helpers import process_df


def plot_error_analysis(
    df: pd.DataFrame,
    list_errors: list,
    figsize: tuple = (8, 12),
    title: Optional[str] = None,
    list_col_titles: list = ["S_0", "S_1", "S_2", "S_3"],
    type_1: bool = True,
):
    if "alpha" in df:
        alphas = df["alpha"].values
    else:
        alphas = np.array(
            [0.05, 0.13, 0.21, 0.30, 0.38, 0.46, 0.54, 0.62, 0.70, 0.78, 0.87, 0.95]
        )
    fig, ax = plt.subplots(
        len(list_errors), len(df), figsize=figsize, sharex=True, sharey=True
    )
    # process the dataframe if needed
    df = process_df(df)
    if len(list_errors) > 1:
        for i in range(len(list_errors)):
            ax[i, 0].set_ylabel(f"{list_errors[i]}", fontsize=15)
            # set y lim to (0,1)
            ax[i, 0].set_ylim(0, 1)
            for j in range(len(df)):
                # plot thick lines, with crosses where the data points are
                label = (
                    r"$\frac{\#(H_1)}{\#(H_1) + \#(H_0)}$"
                    if type_1
                    else r"$\frac{\#(H_0)}{\#(H_1) + \#(H_0)}$"
                )
                ax[i, j].plot(
                    alphas,
                    df[list_errors[i]].iloc[j],
                    linewidth=3,
                    marker="x",
                    markersize=10,
                    color="black",
                    label=label,
                )
                if type_1:
                    ax[i, j].plot(alphas, alphas, "--", color="black", alpha=0.5)
                # ax[i, j].spines[["right", "top"]].set_visible(False)
                ax[i, j].grid()

        for i, col_title in enumerate(list_col_titles):
            ax[0, i].set_title(col_title, fontsize=15)
        ax[0,0].legend(fontsize=15)
    else:
        ax[0].set_ylabel(f"{list_errors[0]}")
        for j in range(len(df)):
            ax[j].plot(alphas, literal_eval(df[list_errors[0]].iloc[j]))
            if j == 0:
                # plot lines with crosses (data points)
                ax[j].plot(alphas, alphas, "--")
            ax[j].spines[["right", "top"]].set_visible(False)
            ax[j].grid()
    fig.supxlabel(r"$\alpha$", fontsize=15, y=0.01)
    # fig.supylabel(r"Type $1$/$2$ error", fontsize=15)
    # plt.tight_layout()
    if title is not None:
        plt.suptitle(title, fontsize=18, y=0.99)
    fig.subplots_adjust(hspace=0.1)
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
