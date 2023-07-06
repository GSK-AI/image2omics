"""
Copyright 2023 Rahil Mehrizi, Cuong Nguyen, GSK plc

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
from typing import List

import numpy as np
import pandas as pd
import seaborn as sns
import scanpy as sc
import anndata as ad
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from slingpy.utils.logging import info


def plot_embedding_scanpy(
    df: pd.DataFrame,
    file_name: str,
    save_folder: str = "",
):
    features_col = [c for c in df.columns if c.startswith("feat_")]
    metadata_col = [c for c in df.columns if c not in features_col]

    df=df[df.target != "RELA"]

    adata = ad.AnnData(df[features_col].values)
    adata.var_names = features_col
    for col in metadata_col:
        adata.obs[col] = df[col].values

    adata_latent=adata
    sc.tl.pca(adata_latent)
    sc.pp.neighbors(adata_latent,use_rep='X', n_neighbors=15)
    sc.tl.leiden(adata_latent)
    sc.tl.paga(adata_latent)
    sc.pl.paga(adata_latent, plot=False)
    sc.tl.umap(adata_latent, spread=1, min_dist=0.5, negative_sample_rate=20,init_pos="paga",random_state=0)

    def gene_mapping(row):
        gene = row["target"]
        stim = row["cell_state"]
        col = row['Column_x']
        if gene == "RELA" and stim == "m1":
            return "M1-RELA"
        if gene == "STAT1" and stim == "m1":
            return "M1-STAT1"
        elif gene == "STAT6" and stim == "m2":
            return "M2-STAT6"
        elif gene == "NT":
            if col == 13:
                return "M0"
            else:
                if stim == "m1":
                    return "M1-NT"
                else:
                    return "M2-NT"
        else:
            return None

    adata_latent.obs["Stimulation - Gene KO"] = adata_latent.obs.apply(gene_mapping, axis=1).values

    with plt.rc_context({'figure.figsize': (8,8)}):
        sc.pl.umap(
            adata_latent, color="Stimulation - Gene KO",title=None,legend_loc='on data',wspace=0.5,
            legend_fontsize=15, legend_fontoutline=4,size=200,frameon=False,na_in_legend=False,
            return_fig=True
        )
        plt.title("")
        plt.tight_layout()
        plot_path = os.path.join(save_folder, file_name)
        plt.savefig(plot_path)
        info(f"Saved plot to {plot_path}.")
        plt.clf()
        plt.cla()
        plt.close()
