import os
import re
import json

from typing import Optional
import numpy as np
import pandas as pd
import torch
from ast import literal_eval
import matplotlib.pyplot as plt
import seaborn as sns
import ternary
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull
import argparse


from src.utils.projections import project_points2D
from src.utils.helpers import process_df
from src.cal_estimates.utils import create_heatmap_data


def plot_ens_comb_cal(
    experiment,
    model,
    file_name: str,
    x_inst=None,
    ens_preds=None,
    p_true=None,
    n_plot=1000,
    device="cpu",
    title: Optional[str] = None,
    output_path: str = "../../figures/",
    alpha_ens=0.05,
    alpha_comb=0.5,
    marker='x',
    max_ens_to_plot=3,
    output_pbar: str = "weighted"
):
    """
    Plots an experiment's data and predictions:
      - True probabilities vs. x
      - Ensemble predictions
      - Combined (p_bar) predictions
      - Calibrated (p_cal) predictions

    This version uses a discrete color palette ('tab10') so that each set of points 
    has distinct, high-contrast colors. We also limit the number of ensemble members 
    plotted to `max_ens_to_plot` so we don't run out of distinct colors or clutter the plot.

    Parameters
    ----------
    experiment : object
        The experiment instance, which typically has attributes:
          - experiment.x_inst : shape (N,1) or (N,)
          - experiment.p_true : shape (N,2), the 'true' probabilities for each x
          - experiment.ens_preds : shape (N,K,2), the ensemble predictions
    model : nn.Module
        A model that when called with (x, p_preds), returns e.g. (p_cal, p_bar, weights).
    x_inst : np.ndarray or torch.Tensor, optional
        If not provided, we'll use experiment.x_inst.
    ens_preds : np.ndarray or torch.Tensor, optional
        If not provided, we'll use experiment.ens_preds.
    p_true : np.ndarray, optional
        If not provided, we'll use experiment.p_true.
    n_plot : int, optional
        Number of points to plot (from the start), by default 1000.
    device : str, optional
        "cpu" or "cuda" for moving data before passing to model, by default "cpu".
    title : str, optional
        Title for the plot, by default None.
    save_path : str, optional
        If provided, the figure is saved to this path.
    alpha_ens : float, optional
        Alpha (transparency) for plotting individual ensemble predictions, by default 0.02.
    alpha_comb : float, optional
        Alpha (transparency) for the combined p_bar scatter, by default 0.5.
    marker : str, optional
        Marker style for scatter points, by default 'x'.
    max_ens_to_plot : int, optional
        How many ensemble members to individually plot, by default 3.
    output_pbar : str, optional
        how to aggregate p_bar, options: "weighted", which takes the weights of the comb model, or "average",

    Returns
    -------
    (fig, ax) : tuple
        The matplotlib figure and axis handles.
    """

    # If any is None, fall back to experiment attributes
    if x_inst is None:
        x_inst = experiment.x_inst
    if ens_preds is None:
        ens_preds = experiment.ens_preds
    if p_true is None:
        p_true = experiment.p_true

    # Convert x_inst, ens_preds to torch if needed
    if not isinstance(x_inst, torch.Tensor):
        x_inst_torch = torch.tensor(x_inst, dtype=torch.float32)
    else:
        x_inst_torch = x_inst

    if not isinstance(ens_preds, torch.Tensor):
        ens_preds_torch = torch.tensor(ens_preds, dtype=torch.float32)
    else:
        ens_preds_torch = ens_preds

    # Slice to n_plot
    n_plot = min(n_plot, x_inst_torch.shape[0])
    # x_inst_torch = x_inst_torch[:n_plot]
    # ens_preds_torch = ens_preds_torch[:n_plot]

    # slice randomly
    idx = np.random.choice(x_inst_torch.shape[0], n_plot, replace=False)
    x_inst_torch = x_inst_torch[idx].to(device)
    ens_preds_torch = ens_preds_torch[idx].to(device)


    # Call model => (p_cal, p_bar, weights) or similar
    outputs = model(x_inst_torch, ens_preds_torch)
    if isinstance(outputs, tuple):
        if len(outputs) == 3:
            p_cal, p_bar, weights = outputs
        elif len(outputs) == 2:
            p_cal, p_bar = outputs
            weights = None
        else:
            # fallback
            p_cal = outputs[0]
            p_bar = outputs[-1]
            weights = None
    else:
        # if just a single thing returned
        p_cal = None
        p_bar = outputs
        weights = None

    if output_pbar == "average":
        # take average of enssemble predictions
        p_bar = ens_preds_torch.mean(dim=1)

    # Convert to numpy for plotting
    p_bar_np = p_bar.detach().cpu().numpy()
    if p_cal is not None:
        p_cal_np = p_cal.detach().cpu().numpy()
    else:
        p_cal_np = None

    x_np = x_inst_torch.cpu().numpy().squeeze()

    # Also get p_true for class 0
    if p_true is not None:
        # select n_plot points of p_true
        # use the indices from x_inst_torch
        p_true_np = p_true[idx, 0]
    else:
        p_true_np = None

    # Subset ensemble predictions for plotting
    # # use again the indices from x_inst_torch
    # ens_preds_subset = ens_preds_torch[idx].cpu().numpy()
    K = ens_preds.shape[1]

    #######################################################
    # Create a discrete color palette with more "jumps"
    # We'll use 'tab10' or 'Set2' for distinct colors
    # We'll need up to max_ens_to_plot + 3 distinct colors
    #######################################################
    color_list = sns.color_palette("viridis", 3)
    color_true = color_list[0]
    color_comb = color_list[1]
    color_cal  = color_list[2]


    # Plot
    fig, ax = plt.subplots(figsize=(7,5))

    # 1) True
    if p_true_np is not None:
        ax.scatter(
            x_np,
            p_true_np,
            label=r"$\mathbb{P}(Y=0|X)$",
            marker=marker,
            color=color_true
        )

    # 2) Combined p_bar
    ax.scatter(
        x_np,
        p_bar_np[:,0],
        label=r"$f_{\lambda}(x)$",
        alpha=alpha_comb,
        marker=marker,
        color=color_comb
    )

    # 3) Calibrated p_cal (if available)
    if p_cal_np is not None:
        ax.scatter(
            x_np,
            p_cal_np[:,0],
            label=r"$g(f_{\lambda}(x))$",
            marker=marker,
            color=color_cal
        )

    # 4) Ensemble members
    # We'll limit ourselves to max_ens_to_plot for distinct colors
    n_plot_members = min(K, max_ens_to_plot)
    for k in range(n_plot_members):
        ens_color = "grey" # pick next color
        ax.scatter(
            x_np,
            ens_preds_torch[:,k,0],
            alpha=alpha_ens,
            label=(f"ens" if k==0 else None),
            marker="o",
            color=ens_color
        )
    # If you wanted to label each ensemble member differently, you could remove the
    # condition 'if k==0 else None' for the label, or create separate legends.

    if title is not None:
        ax.set_title(title)
    ax.set_xlabel(r"$x$", fontsize=15)
    ax.set_ylabel(r"$\mathbb{P}(Y=0|X)$", fontsize=15)
    ax.legend(fontsize=15)

    # set x ticks and y tick size
    ax.tick_params(axis="both", labelsize=15)

    save_path = os.path.join(output_path, file_name)
    # make output path if it does not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Figure saved to: {save_path}")

    return fig, ax


def read_and_plot_error_analysis(
    file_path,
    save_name: str,
    output_path: str = "../../figures/",
    list_col_titles: list = [
        r"$\lambda=const$",
        r"$\lambda=f(x)$"
    ],
    title: str = None,
    figsize: tuple = (8, 10),
    type_1: bool = True,
    alpha: np.ndarray = None
):
    # make output path if it does not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    df_results = pd.read_csv(file_path)

    # plot error analysis
    fig = plot_error_analysis(
        df_results,
        list_errors=df_results.columns,
        list_col_titles=list_col_titles,
        title=title,
        figsize=figsize,
        type_1=type_1,
        alpha=alpha
    )
    fig.savefig(output_path + save_name + ".png", dpi=300)
    # fig_t2.savefig(output_path + title_2 + ".png", bbox_inches="tight")

    return fig


def read_and_plot_error_analysis_full(
    file_path,
    output_path: str = "../../figures/",
    list_col_titles: list = [
        r"$\lambda=const$",
        r"$\lambda=f(x)$",
        r"$S_1$",
        r"$S_2$",
        r"$S_3$",
    ],
    title_1: str = None,
    title_2: str = None,
    save_name_1: str = "error_analysis_t1",
    save_name_2: str = "error_analysis_t2",
    figsize_1: tuple = (7, 10),
    figsize_2: tuple = (10, 10),
    n_type_1: int = 2,
    alpha_1: np.ndarray = None,
    alpha_2: np.ndarray = None,
):
    # make output path if it does not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    df_results = pd.read_csv(file_path)
    # split into data used for type 1 and type 2 error analysis
    df_results_t1 = df_results.iloc[:n_type_1, :]
    df_results_t2 = df_results.iloc[n_type_1:, :]

    # plot error analysis
    fig_t1 = plot_error_analysis_syn(
        df_results_t1,
        list_errors=df_results_t1.columns,
        list_col_titles=list_col_titles[:n_type_1],
        title=title_1,
        figsize=figsize_1,
        alpha=alpha_1
    )
    fig_t2 = plot_error_analysis_syn(
        df_results_t2,
        list_errors=df_results_t2.columns,
        list_col_titles=list_col_titles[n_type_1:],
        title=title_2,
        figsize=figsize_2,
        type_1=False,
        alpha=alpha_2
    )

    # save figures
    save_path_1 = os.path.join(output_path, save_name_1 + ".png")
    save_path_2 = os.path.join(output_path, save_name_2 + ".png")
    fig_t1.savefig(save_path_1, bbox_inches="tight", dpi=300)
    fig_t2.savefig(save_path_2, bbox_inches="tight", dpi=300)

    return fig_t1, fig_t2


from typing import Optional

def safe_literal_eval(val):
    """Convert a string with np.float64 wrappers to a valid Python literal.
    
    If the value is a string containing entries like np.float64(0.0),
    this function removes the 'np.float64(' and the closing ')', so that
    literal_eval can parse it.
    """
    if isinstance(val, str):
        fixed = re.sub(r'np\.float64\((.*?)\)', r'\1', val)
        return literal_eval(fixed)
    return val


def plot_error_analysis_syn(
    df: pd.DataFrame,
    list_errors: list,
    figsize: tuple = (8, 12),
    title: Optional[str] = None,
    list_col_titles: list = None,
    type_1: bool = True,
    alpha: Optional[np.ndarray] = None,
):
    """
    Flexible plotting: if the dataframe contains an 'alpha' column
    with JSON-encoded lists, each row can have its own alpha grid.
    """

    # -------- extract per-row alpha vectors -----------------------
    def get_alpha(row):
        if "alpha" in row and isinstance(row["alpha"], str):
            return np.asarray(json.loads(row["alpha"]))
        return alpha                      # fallback for legacy CSV

    alphas_per_row = df.apply(get_alpha, axis=1).tolist()
    # --------------------------------------------------------------

    # fall back if list_col_titles not supplied
    if list_col_titles is None:
        list_col_titles = [f"S_{i}" for i in range(len(df))]

    n_rows = len(df)
    n_metrics = len(list_errors)

    # --- figure/axes boilerplate ---------------------------------
    fig, ax = plt.subplots(
        n_metrics, n_rows, figsize=figsize, sharex=True, sharey=True
    ) if n_metrics > 1 else plt.subplots(
        n_rows, figsize=figsize, sharex=True, sharey=True
    )

    # make ax 2D for uniform indexing
    if n_metrics == 1:
        ax = np.atleast_2d(ax)

    # -------- main plotting loop ---------------------------------
    for i, metric in enumerate(list_errors):
        for j in range(n_rows):
            # y_values = safe_literal_eval(df[metric].iloc[j])
            y_values = json.loads(df[metric].iloc[j])
            # a = alphas_per_row[j]
            a = json.loads(df["alpha"].iloc[j])
            if not isinstance(y_values, (list, tuple, np.ndarray)):
                y_values = [float(y_values)] 
            # truncate / pad alpha to match y length
            if len(y_values) < len(a):
                a = a[: len(y_values)]
            elif len(y_values) > len(a):
                a = np.pad(a, (0, len(y_values) - len(a)), "edge")

            ax[i, j].plot(a, y_values, linewidth=3, marker="x",
                          markersize=8, color="black")
            if type_1:
                ax[i, j].plot(a, a, "--", color="grey", alpha=0.5)

            ax[i, j].grid(True, alpha=0.3)
            ax[i, j].spines[["right", "top"]].set_visible(False)

        ax[i, 0].set_ylabel(metric, fontsize=13)

    # ---- titles & labels ----------------------------------------
    for j, col_title in enumerate(list_col_titles):
        if j < ax.shape[1]:
            ax[0, j].set_title(col_title, fontsize=14)

    fig.supxlabel(r"$\alpha$", fontsize=15, y=0.04)
    if title:
        fig.suptitle(title, fontsize=16, y=0.98)
    fig.tight_layout()
    return fig


def plot_error_analysis(
    df: pd.DataFrame,
    list_errors: list,
    figsize: tuple = (8, 12),
    title: Optional[str] = None,
    list_col_titles: list = ["S_0", "S_1", "S_2", "S_3"],
    type_1: bool = True,
    alpha: Optional[np.ndarray] = None,
):
    # Determine alpha values.
    if "alpha" in df:
        alphas = df["alpha"].values
    elif alpha is not None:
        alphas = alpha
    else:
        alphas = np.array(
            [0.05, 0.13, 0.21, 0.30, 0.38, 0.46, 0.54, 0.62, 0.70, 0.78, 0.87, 0.95]
        )
    
    # Create subplots.
    if len(list_errors) > 1:
        fig, ax = plt.subplots(len(list_errors), len(df), figsize=figsize, sharex=True, sharey=True)
    else:
        fig, ax = plt.subplots(len(df), figsize=figsize, sharex=True, sharey=True)
        # Ensure ax is iterable (list) in the single error case.
        if not hasattr(ax, '__iter__'):
            ax = [ax]
        else:
            ax = list(ax)
    
    # Process the dataframe if needed.
    df = process_df(df)
    
    if len(list_errors) > 1:
        # Multiple error metrics: ax is 2D.
        for i in range(len(list_errors)):
            ax[i, 0].set_ylabel(f"{list_errors[i]}", fontsize=15)
            ax[i, 0].set_ylim(0, 1)
            for j in range(len(df)):
                # Get value and process if it is a string.
                val = df[list_errors[i]].iloc[j]
                y_values = safe_literal_eval(val) if isinstance(val, str) else val
                label = (
                    r"$\frac{\#(H_1)}{\#(H_1) + \#(H_0)}$"
                    if type_1
                    else r"$\frac{\#(H_0)}{\#(H_1) + \#(H_0)}$"
                )
                ax[i, j].plot(
                    alphas,
                    y_values,
                    linewidth=3,
                    marker="x",
                    markersize=10,
                    color="black",
                    label=label,
                )
                if type_1:
                    ax[i, j].plot(alphas, alphas, "--", color="black", alpha=0.5)
                ax[i, j].grid()
                ax[i, j].spines[["right", "top"]].set_visible(False)

        for i, col_title in enumerate(list_col_titles):
            ax[0, i].set_title(col_title, fontsize=18)
        ax[0, 0].legend(fontsize=15)
    else:
        # Single error metric: ax is a 1D array.
        ax[0].set_ylabel(f"{list_errors[0]}", fontsize=15)
        for j in range(len(df)):
            val = df[list_errors[0]].iloc[j]
            y_values = safe_literal_eval(val) if isinstance(val, str) else val
            ax[j].plot(alphas, y_values, linewidth=3, marker="x", markersize=10, color="black")
            if j == 0:
                ax[j].plot(alphas, alphas, "--", color="black", alpha=0.5)
            ax[j].grid()
            ax[j].spines[["right", "top"]].set_visible(False)
    
    fig.supxlabel(r"$\alpha$", fontsize=20, y=0.03)
    if title is not None:
        plt.suptitle(title, fontsize=18, y=0.97)
    fig.subplots_adjust(hspace=0.1)
    
    # Set tick parameters appropriately.
    if len(list_errors) > 1:
        for i in range(len(list_errors)):
            for j in range(len(df)):
                ax[i, j].tick_params(axis="both", labelsize=15)
    else:
        for j in range(len(df)):
            ax[j].tick_params(axis="both", labelsize=15)
    
    return fig



def plot_heatmaps(
    data_dicts,
    scale,
    ternary_point=None,
    out_path="heatmap_cal_estimates.png"
):
    """
    Plot multiple heatmaps in a 2x2 or NxN grid, one per measure
    within data_dicts. Writes to 'out_path'.
    """
    measures = list(data_dicts.keys())
    n_measures = len(measures)

    # We'll produce up to 4 or 6 subplots
    fig_cols = 2
    fig_rows = int(np.ceil(n_measures / fig_cols))

    fig, axes = plt.subplots(fig_rows, fig_cols, figsize=(8*fig_cols, 6*fig_rows))
    axes = axes.ravel() if n_measures>1 else [axes]

    for i, measure in enumerate(measures):
        ax = axes[i]
        ax.set_title(f"{measure}", fontsize=14, pad=10)

        # Convert the dictionary data => use ternary.heatmap
        # Possibly we do log scale? Then we do np.log...
        # For demonstration, let's skip log transform:
        measure_data = data_dicts[measure]

        tax = ternary.TernaryAxesSubplot(ax=ax, scale=scale)
        tax.gridlines(multiple=5, color="black")

        # heatmap wants a dict (i,j)-> val with i+j <= scale
        # Just pass measure_data directly:
        tax.heatmap(
            measure_data,
            scale=scale,
            style="hexagonal",
            cmap="inferno",
            colorbar=True
        )

        if ternary_point is not None:
            # e.g. (0.1,0.1,0.8) * scale
            # Convert to scaled coords
            scaled_point = (
                ternary_point[0]*scale,
                ternary_point[1]*scale,
                ternary_point[2]*scale
            )
            tax.scatter([scaled_point], marker='o', color='red', s=70, label="p^*")

        tax.set_axis_limits({'b': [0, 1], 'l': [0, 1], 'r': [0, 1]})
        tax.get_ticks_from_axis_limits(multiple=5)
        tax.set_custom_ticks(tick_formats="%.2f", offset=0.03, fontsize=10)
        tax.clear_matplotlib_ticks()
        tax.get_axes().axis('off')

    # If we put a legend for the point:
    if ternary_point is not None:
        axes[0].legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"Saved heatmaps to {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate and plot calibration heatmaps over the 2-simplex."
    )
    parser.add_argument("--scale", type=int, default=40,
                        help="Resolution of the simplex grid.")
    parser.add_argument("--nsamples", type=int, default=2000,
                        help="Number of random samples for miscalibration estimation.")
    parser.add_argument("--measures", nargs="+", default=["Brier","L2","SKCE","MMD","KL"],
                        help="Which calibration measures to compute.")
    parser.add_argument("--out", type=str, default="heatmap_cal_estimates.png",
                        help="Output path for the figure.")
    parser.add_argument("--ptrue", type=float, nargs="+", default=[0.1,0.1,0.8],
                        help="Ground-truth distribution, e.g. '--ptrue 0.1 0.1 0.8'")

    args = parser.parse_args()

    scale = args.scale
    n_samples = args.nsamples
    measures = args.measures
    out_path = args.out
    ptrue = np.array(args.ptrue, dtype=float)

    # Compute the data
    data_dicts = create_heatmap_data(
        scale=scale,
        n_samples=n_samples,
        p_true=ptrue,
        measures=measures
    )

    # Plot
    plot_heatmaps(
        data_dicts=data_dicts,
        scale=scale,
        ternary_point=ptrue,
        out_path=out_path
    )




# def plot_error_analysis(
#     df: pd.DataFrame,
#     list_errors: list,
#     figsize: tuple = (8, 12),
#     title: Optional[str] = None,
#     list_col_titles: list = ["S_0", "S_1", "S_2", "S_3"],
#     type_1: bool = True,
#     alpha: Optional[np.ndarray] = None,
# ):
#     if "alpha" in df:
#         alphas = df["alpha"].values
#     elif alpha is not None:
#         alphas = alpha
#     else:
#         alphas = np.array(
#             [0.05, 0.13, 0.21, 0.30, 0.38, 0.46, 0.54, 0.62, 0.70, 0.78, 0.87, 0.95]
#         )
#     fig, ax = plt.subplots(
#         len(list_errors), len(df), figsize=figsize, sharex=True, sharey=True
#     )
#     # process the dataframe if needed
#     df = process_df(df)
#     if len(list_errors) > 1:
#         for i in range(len(list_errors)):
#             ax[i, 0].set_ylabel(f"{list_errors[i]}", fontsize=15)
#             # set y lim to (0,1)
#             ax[i, 0].set_ylim(0, 1)
#             for j in range(len(df)):
#                 # plot thick lines, with crosses where the data points are
#                 label = (
#                     r"$\frac{\#(H_1)}{\#(H_1) + \#(H_0)}$"
#                     if type_1
#                     else r"$\frac{\#(H_0)}{\#(H_1) + \#(H_0)}$"
#                 )
#                 ax[i, j].plot(
#                     alphas,
#                     df[list_errors[i]].iloc[j],
#                     linewidth=3,
#                     marker="x",
#                     markersize=10,
#                     color="black",
#                     label=label,
#                 )
#                 if type_1:
#                     ax[i, j].plot(alphas, alphas, "--", color="black", alpha=0.5)
#                 # ax[i, j].spines[["right", "top"]].set_visible(False)
#                 ax[i, j].grid()
#                 ax[i, j].spines[["right", "top"]].set_visible(False)

#         for i, col_title in enumerate(list_col_titles):
#             ax[0, i].set_title(col_title, fontsize=18)
#         ax[0,0].legend(fontsize=15)
#     else:
#         ax[0].set_ylabel(f"{list_errors[0]}")
#         for j in range(len(df)):
#             ax[j].plot(alphas, literal_eval(df[list_errors[0]].iloc[j]))
#             if j == 0:
#                 # plot lines with crosses (data points)
#                 ax[j].plot(alphas, alphas, "--")
#             ax[j].spines[["right", "top"]].set_visible(False)
#             ax[j].grid()
#     fig.supxlabel(r"$\alpha$", fontsize=20, y=0.03)
#     # fig.supylabel(r"Type $1$/$2$ error", fontsize=15)
#     # plt.tight_layout()
#     if title is not None:
#         plt.suptitle(title, fontsize=18, y=0.97)
#     fig.subplots_adjust(hspace=0.1)
#     # set x and y ticks and labels to be big
#     for i in range(len(list_errors)):
#         for j in range(len(df)):
#             ax[i, j].tick_params(axis="both", labelsize=15)
#     return fig


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



if __name__ == "__main__":
    main()