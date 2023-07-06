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
from collections import Counter
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from slingpy.utils.logging import info


def plot_violins(
    data_df,
    plot_title,
    file_name,
    group_column: str,
    error_column: str,
    directory: Path,
    seed: int = 909,
    topk: int = -1,
):
    subset_df = data_df.dropna(subset=group_column)
    counts = Counter(subset_df[group_column])
    top_locations = sorted(counts.items(), key=lambda x: x[1])
    if topk != -1:
        top_locations = top_locations[-topk:]
    top_locations = list(map(lambda x: x[0], top_locations))
    top_locations = list(
        sorted(
            top_locations,
            key=lambda x: np.median(
                subset_df[subset_df[group_column] == x][error_column].values
            ),
        )
    )
    errs = [
        subset_df[subset_df[group_column] == location][error_column].values
        for location in top_locations
    ]
    if len(errs) == 0:
        info(f"Skipped {file_name} because no locations given.")
        return

    def adjacent_values(vals, q1, q3):
        upper_adjacent_value = q3 + (q3 - q1) * 1.5
        upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

        lower_adjacent_value = q1 - (q3 - q1) * 1.5
        lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
        return lower_adjacent_value, upper_adjacent_value

    parts = plt.violinplot(errs, showmeans=False, showmedians=False, showextrema=False)
    quartile1, medians, quartile3 = list(
        zip(*[np.percentile(err, [25, 50, 75], axis=0) for err in errs])
    )
    whiskers = np.array(
        [
            adjacent_values(sorted_array, q1, q3)
            for sorted_array, q1, q3 in zip(errs, quartile1, quartile3)
        ]
    )
    whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]

    inds = np.arange(1, len(medians) + 1)
    plt.vlines(inds, quartile1, quartile3, color="k", linestyle="-", lw=5)
    plt.vlines(inds, whiskers_min, whiskers_max, color="k", linestyle="-", lw=1)
    plt.hlines(medians, inds - 0.07, inds + 0.07, color="white", linestyle="-", lw=0.5)

    colors = [
        "e3f2fd",
        "bbdefb",
        "90caf9",
        "64b5f6",
        "42a5f5",
        "2196f3",
        "1e88e5",
        "1976d2",
        "1565c0",
        "0d47a1",
    ]
    for c, pc in zip(colors, parts["bodies"]):
        pc.set_facecolor("#" + c.upper())
        pc.set_edgecolor("black")
        pc.set_alpha(1)

    def set_axis_style(labels):
        plt.gca().xaxis.set_tick_params(direction="out")
        plt.gca().xaxis.set_ticks_position("bottom")
        plt.gca().set_xticks(np.arange(1, len(labels) + 1), labels=labels)
        plt.gca().set_xlim(0.25, len(labels) + 0.75)
        # plt.gca().set_xlabel(group_column)
        plt.xticks(rotation=90)

    jitter_amt = 0.15
    random_state = np.random.RandomState(seed)
    for idx, (loc, err_i, c_i) in enumerate(zip(top_locations, errs, colors)):
        plt.scatter(
            [idx + 1] * len(err_i)
            + random_state.uniform(-jitter_amt, jitter_amt, size=len(err_i)),
            err_i,
            s=0.35,
            linewidths=0.15,
            marker="X",
            edgecolors="black",
            alpha=0.5,
            c="#" + c_i.upper(),
        )

    set_axis_style(top_locations)
    plt.ylabel("Correlation [R^2]")
    plt.ylim([0, 1.0])
    plt.title(plot_title)
    plt.gca().set_axisbelow(True)
    plt.grid(linestyle="--", color="grey", linewidth=0.25, axis="y")
    plt.gca().spines["right"].set_color("none")
    plt.gca().spines["top"].set_color("none")
    plt.gcf().subplots_adjust(bottom=0.5)
    plt.tight_layout()
    plot_path = os.path.join(directory, file_name)
    plt.savefig(plot_path)
    plt.clf()
    plt.cla()
    plt.close()
    info(f"Saved plot to {plot_path}.")
