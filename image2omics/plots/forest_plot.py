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

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from matplotlib import pyplot as plt
from slingpy.utils.logging import info


def plot_forest(
    df: pd.DataFrame,
    file_name: str,
    directory: Path,
    num_significant_location_betas: int,
    title: str,
    max_beta_magnitude: int = 5,
    xlim: Optional[Tuple[int, int]] = (-5.5, 5.5),
):
    plt.figure(figsize=(6, 4 * (len(df) / 25)), dpi=150)
    ci = [
        df.iloc[::-1]["beta"] - df.iloc[::-1]["2.5%"].values,
        df.iloc[::-1]["97.5%"].values - df.iloc[::-1]["beta"],
    ]
    plt.errorbar(
        x=df.iloc[::-1]["beta"],
        y=df.iloc[::-1].index.values,
        xerr=ci,
        color="black",
        capsize=2,
        linestyle="None",
        linewidth=0.5,
        marker="D",
        markersize=2,
        mfc="black",
        mec="black",
    )
    plt.gca().spines["right"].set_color("none")
    plt.gca().spines["left"].set_color("none")
    plt.gca().spines["top"].set_color("none")
    plt.axvline(x=0, linewidth=0.8, linestyle="--", color="black")
    plt.axhline(y=0.5, linewidth=0.3, linestyle=":", color="black")
    if num_significant_location_betas != 0:
        plt.axhline(
            y=num_significant_location_betas + 0.5,
            linewidth=0.3,
            linestyle=":",
            color="black",
        )
    plt.tick_params(axis="both", which="major", labelsize=5)
    plt.tick_params(axis="y", which="both", length=0)
    plt.xlabel("Betas [95% Confidence Interval, higher = more predictable]", fontsize=8)
    plt.tight_layout()
    if xlim is not None:
        plt.xlim(xlim)
    plt.title(title)
    save_dir = directory / file_name
    plt.savefig(save_dir, bbox_inches="tight")
    plt.clf()
    plt.cla()
    plt.close()
    info(f"Saved {save_dir}")
