import datetime
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MultipleLocator

SEABORN_0_13_OR_NEWER = tuple(int(part) for part in sns.__version__.split(".")[:2]) >= (0, 13)


# Below are functions for analyzing and plotting neural network data

def make_results_directory():
    """
    This function makes (if needed) a result folder named according to the current date and time
    in the 'results' directory of the model, also creating 'results' if it is missing, and returns its path.
    Call it once so the plot and the table of one training run end up in the same folder.
    """

    parent_dir = os.getcwd() + "/results/"
    child_dir = datetime.datetime.now().strftime("%Y-%m-%d %Hh%Mm") + " results"
    os.makedirs(parent_dir + child_dir, exist_ok=True)
    return parent_dir + child_dir


def plot_loss_over_time(loss_over_time_list, session_list, results_directory=None):
    """
    This function takes a list of loss over time (which contains the losses calculated for
    each image in a given session) and a list of session numbers that correspond to each loss
    value. It calculates the mean loss for each session using its associated losses repeatedly
    for bootstrapped 95% confidence intervals. These means with confidence intervals are plotted
    over the session numbers found in the session list.

    The plot mentioned above is saved as 'training_loss_plot' in the result folder given as
    'results_directory'. If none is given, a result folder named according to the date and time of
    training is made in the 'results' directory of the model (see make_results_directory()).

    Parameters
    ----------
    loss_over_time_list
    session_list
    """

    # seaborn 0.13+ replaced 'scale' and 'errwidth' with Line2D keywords and err_kws, and markers must be a
    # single style (not a list) when there is no hue; the older keywords are kept for the pinned seaborn 0.12
    if SEABORN_0_13_OR_NEWER:
        style = dict(markersize=8, linewidth=2.3, err_kws={"linewidth": 1.75})
    else:
        style = dict(scale=0.85, errwidth=1.75)
    sns.pointplot(x=session_list, y=loss_over_time_list, estimator="mean", errorbar=("ci", 95),
                  markers="o", linestyles="--", capsize=0.15, **style)
    plt.xlabel("Session", fontsize=12)
    plt.ylabel("Mean Loss", fontsize=12)

    ax = plt.gca()
    ax.spines["left"].set_linewidth(2)
    ax.spines["bottom"].set_linewidth(2)

    x_ticks = MultipleLocator(9)
    ax.xaxis.set_major_locator(x_ticks)
    plt.xlim(0, max(session_list))
    sns.despine()

    y_max = 2
    if max(loss_over_time_list) > y_max:
        y_max = max(loss_over_time_list)
    plt.ylim(0, y_max)

    if results_directory is None:
        results_directory = make_results_directory()
    file_name = "/training_loss_plot.png"
    plt.savefig(results_directory + file_name, dpi=300, bbox_inches="tight")


def export_loss_table(loss_table_list, session_list, results_directory=None):
    """
    This function takes a loss table list comprised of dictionaries with image class (key)-list
    of image class losses (value) pairs and iterates over it (list comprehension) to convert
    each dictionary to a pandas dataframe. These dataframes are concatenated for a final
    combined dataframe with session number in  the left most column, image set index in
    the second leftmost column, and calculated loss according to image class in the remaining
    columns (column headers reflect image class names).

    The dataframe mentioned above is saved as 'training_loss_table' in the result folder given as
    'results_directory'. If none is given, a result folder named according to the date and time of
    training is made in the 'results' directory of the model (see make_results_directory()).

    Parameters
    ----------
    loss_table_list
    session_list
    """

    loss_dfs = [pd.DataFrame(x) for x in loss_table_list]
    final_df = pd.concat(loss_dfs, keys=sorted(set(session_list)))

    if results_directory is None:
        results_directory = make_results_directory()
    file_name = "/training_loss_table.xlsx"
    final_df.to_excel(results_directory + file_name, sheet_name="Loss per Class per Session")
