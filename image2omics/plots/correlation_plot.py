"""
Copyright 2023 Arash Mehrjou, Patrick Schwab, GSK plc

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import ticker
from matplotlib.offsetbox import AnchoredText
from scipy import stats
from slingpy.utils.logging import info, warn


def plot_correlation(
    data_df: pd.DataFrame,
    gene_name: str,
    file_name: str,
    directory: Path,
    y_pred_name: str = "predicted from images (absolute)",
    y_true_name: str = "measured (absolute)",
):
    import warnings

    warnings.simplefilter(action="ignore", category=FutureWarning)

    try:
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            data_df[y_pred_name], data_df[y_true_name]
        )
    except ValueError:
        warn(
            f"Couldn't plot correlation for {file_name} - most likely because there aren't enough unique values."
        )
        return
    g = sns.jointplot(
        x=y_pred_name,
        y=y_true_name,
        data=data_df.reset_index(),
        kind="reg",
        color="xkcd:black",
        scatter_kws={"s": 25},
    )
    # g = g.plot_joint(sns.regplot, color="xkcd:black")
    ax = g.ax_joint
    anc = AnchoredText(
        f"r={r_value:.2f} p={p_value:.4f} n={len(data_df):d}",
        loc="upper left",
        frameon=True,
        prop={"size": 15},
    )
    anc2 = AnchoredText(
        f"{gene_name}",
        loc="lower right",
        frameon=False,
        pad=0.001,
        borderpad=0.05,
        prop={"size": 35, "fontweight": "bold"},
    )
    ax.set_xlabel("Predicted [abundance]")
    ax.set_ylabel("Measured [abundance]")
    ax.add_artist(anc)
    ax.add_artist(anc2)
    ax.set_axisbelow(True)
    ax.grid(linestyle="--", color="grey", linewidth=0.25)

    ax.xaxis.set_major_formatter(ticker.EngFormatter())
    ax.yaxis.set_major_formatter(ticker.EngFormatter())

    plt.tight_layout()
    data_folder = os.path.join(directory, "corr")
    if not os.path.exists(data_folder):
        os.mkdir(data_folder)
    plot_path = os.path.join(data_folder, file_name)
    plt.savefig(plot_path)
    info(f"Saved plot to {plot_path}.")
    plt.clf()
    plt.cla()
    plt.close()
