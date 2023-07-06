"""
Copyright 2023 Rahil Mehrizi, Arash Mehrjou, Cuong Nguyen, Patrick Schwab, GSK plc

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

import itertools
import os
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import List
import math
import h5py

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import slingpy as sp
import statsmodels.api as sm
from scipy import stats
from scipy.stats import linregress
from sklearn.preprocessing import MinMaxScaler
from slingpy.utils.logging import info, warn
from skimage import exposure

from image2omics.datasets.hgnc_names import HGNCNames
from image2omics.datasets.reactome_pathway_embedding import (
    ReactomePathwayEmbedding,
)
from image2omics.datasets.subcellular_location import SubcellularLocation
from image2omics.plots.correlation_plot import plot_correlation
from image2omics.plots.embedding_plot import plot_embedding_scanpy
from image2omics.plots.forest_plot import plot_forest
from image2omics.plots.pie_plot import plot_pie
from image2omics.plots.violin_plot import plot_violins


def smape(y_true, y_pred, smape_normalisation_factor: float = 1.0):
    return (
        np.abs(
            (
                (y_pred - y_true)
                / ((np.abs(y_true) + np.abs(y_pred)) / smape_normalisation_factor)
            )
        )
        * 100.0
    )


class PlotApplication:
    FONT_FILE = f"{os.environ['DATA_DIR']}/third-party/Open_Sans.zip"
    META_FILE = (f"{os.environ['DATA_DIR']}/data/metadata.csv")

    DRUGSEQ_M1_FILE = (f"{os.environ['SAVE_DIR']}/transcriptomics_m1/transcriptomics_m1_predictions.csv")
    DRUGSEQ_M2_FILE = (f"{os.environ['SAVE_DIR']}/transcriptomics_m2/transcriptomics_m2_predictions.csv")
    PROTEOMICS_M1_FILE = (f"{os.environ['SAVE_DIR']}/proteomics_m1/proteomics_m1_predictions.csv")
    PROTEOMICS_M2_FILE = (f"{os.environ['SAVE_DIR']}/proteomics_m2/proteomics_m2_predictions.csv")
    ICF_FILES = (f"{os.environ['DATA_DIR']}/data/ICF")
    TILES_FILES = (f"{os.environ['DATA_DIR']}/data/TILES")

    ALL_FILES = [
        DRUGSEQ_M1_FILE,
        DRUGSEQ_M2_FILE,
        PROTEOMICS_M1_FILE,
        PROTEOMICS_M2_FILE
    ]
    EMBEDDING = f"{os.environ['DATA_DIR']}/third-party/embedding.csv"
    ALL_LOCATIONS = [
        "Actin filaments",
        "Cell Junctions",
        "Centriolar satellite",
        "Centrosome",
        "Cytokinetic bridge",
        "Cytosol",
        "Endoplasmic reticulum",
        "Golgi apparatus",
        "Intermediate filaments",
        "Microtubules",
        "Mitochondria",
        "Nuclear bodies",
        "Nuclear membrane",
        "Nuclear speckles",
        "Nucleoli",
        "Nucleoli fibrillar center",
        "Nucleoli rim",
        "Nucleoplasm",
        "Plasma membrane",
        "Vesicles",
    ]
    MEAN_MEASURED_SCALED_NAME = "Mean abundance"
    SELECTED_GENES = ["DDI2", "HLA-A", "PDCD6IP", "S100A11", "TFPI2"]
    TOTAL_GENES = {
        "Proteomics (M1)": 4986,
        "Proteomics (M2)": 4986,
        "Transcriptomics (M1)": 26317,
        "Transcriptomics (M2)": 26317,
    }

    def __init__(self, cache_folder: str, data_folder: str, seed: int = 909):
        self.cache_folder = Path(cache_folder)
        self.cache_folder.mkdir(exist_ok=True)
        self.data_folder = Path(data_folder)
        self.data_folder.mkdir(exist_ok=True)
        self.seed = seed
        self.hgnc = HGNCNames(str(self.cache_folder))
        self.subcellular_location = SubcellularLocation(str(self.cache_folder)).load()
        self.pathways = ReactomePathwayEmbedding(str(self.cache_folder)).load()

    def _setup(self):
        from matplotlib import font_manager
        font_file = PlotApplication.FONT_FILE
        font_dir = Path(self.cache_folder) / "fonts"
        font_dir.mkdir(exist_ok=True)
        with zipfile.ZipFile(font_file, "r") as zip_ref:
            zip_ref.extractall(font_dir)

        font_dirs = [font_dir]
        font_files = font_manager.findSystemFonts(fontpaths=font_dirs)

        for font_file in font_files:
            font_manager.fontManager.addfont(font_file)

        # set font
        plt.rcParams["font.family"] = "Open Sans"

    def _read_df(self, file_path: Path) -> pd.DataFrame:
        column_names = list(pd.read_csv(file_path, nrows=1))
        include_column_names = [
            col for col in column_names if "plate_barcode" not in col
        ]
        df = pd.read_csv(
            file_path,
            usecols=include_column_names,
            index_col=0,
            dtype={
                "well_id": "category",
                "train/test/val fold": "category",
                "measured (absolute)": np.float32,
                "predicted from images (absolute)": np.float32,
                "seed": np.uint8,
                "donor": np.uint8,
            },
        )
        return df

    def _get_locations(self, gene_symbols: List[str]) -> List[str]:
        locations = []
        for name in gene_symbols:
            if name:
                ccl_idx = self.subcellular_location.get_by_row_name(name)
                if not math.isnan(ccl_idx[0]):
                    resolved = self.subcellular_location.get_column_code_lists()[0][
                        int(ccl_idx[0])
                    ]
                    locations.append(resolved)
                else:
                    locations.append(None)
            else:
                locations.append(None)
        return locations

    def _get_association_with_r2(
        self, df: pd.DataFrame, x_labels: List[str], y_label: str
    ) -> pd.DataFrame:
        df_x = df[x_labels]
        model = sm.OLS(df[[y_label]], sm.add_constant(df_x)).fit()
        params = model.params
        conf = model.conf_int(0.05)
        conf["beta"] = params
        conf.columns = ["2.5%", "97.5%", "beta"]
        odds_df = pd.DataFrame(conf)
        odds_df["pvalues"] = model.pvalues
        odds_df["significant?"] = [
            "significant" if pval <= 0.05 else "not significant"
            for pval in model.pvalues
        ]
        return odds_df.sort_values("beta", ascending=False)

    def _get_pathways(self, gene_symbols: List[str]) -> List[str]:
        blank_vector = np.zeros((len(self.pathways.get_column_names()),))
        pathways = []
        for name in gene_symbols:
            if name:
                pathway_vector = self.pathways.get_by_row_name(name)
                if not math.isnan(pathway_vector[0][0]):
                    pathways.append(pathway_vector[0])
                else:
                    pathways.append(blank_vector)
            else:
                pathways.append(blank_vector)
        return np.array(pathways, dtype=float)

    def _get_corr_df(
        self,
        df: pd.DataFrame,
        is_protein: bool = False,
        seed: int = 0,
        key: str = "gene/protein",
    ):
        df = df[(df["seed"] == seed)]
        df = df[(df["train/test/val fold"] == "test")]
        df = df.set_index(key)

        if is_protein:
            df["gene_symbol"] = df.index
        else:
            df["gene_symbol"] = self.hgnc.convert_ensembl_ids_to_gene_names(df.index)
        df = df.set_index("gene_symbol")
        df = df.loc[
            [gene for gene in PlotApplication.SELECTED_GENES if gene in df.index]
        ]
        return df

    def _calculate_r2(
        self,
        df: pd.DataFrame,
        is_protein: bool = False,
        seed: int = 0,
        key: str = "gene/protein",
    ) -> pd.DataFrame:
        df = df[(df["seed"] == seed)]
        train_set_means = df[(df["train/test/val fold"] == "train")].set_index(key)
        train_set_means = train_set_means.groupby(train_set_means.index)[
            ["measured (absolute)"]
        ].mean()
        df = df[(df["train/test/val fold"] == "test")]
        df = df.set_index(key)
        col_names = [
            "r2",
            "p",
            "n",
            "smape_pred",
            "smape_mean",
            "mean_measured",
            "p_smape",
        ]
        index = list(sorted(set(df.index)))
        r2_df = pd.DataFrame(
            data=np.zeros((len(index), len(col_names))),
            index=pd.Index(index, name=key),
            columns=col_names,
        )

        gene_index = defaultdict(list)
        for gene, i in zip(df.index, range(len(df.index))):
            gene_index[gene] += [i]

        col_measured_idx = df.columns.tolist().index("measured (absolute)")
        col_predicted_idx = df.columns.tolist().index(
            "predicted from images (absolute)"
        )

        values = df.values

        def _process_gene(i, gene):
            locations = list(sorted(gene_index[gene]))
            mean_measured = train_set_means.loc[gene]
            x = values[locations, col_measured_idx].astype(float)
            y = values[locations, col_predicted_idx].astype(float)
            slope, intercept, r_value, p_value, std_err = linregress(x, y)
            smape_mean = smape(x, np.array([mean_measured] * len(x))[:, 0])
            smape_pred = smape(x, y)
            pd.DataFrame(smape_mean).to_csv("/home/rm426130/test.csv")
            pd.DataFrame(smape_pred).to_csv("/home/rm426130/test1.csv")
            p_smape = stats.wilcoxon(smape_pred, smape_mean, alternative="less")[1]
            r2_df.iloc[i] = (
                r_value**2,
                p_value,
                len(x),
                np.mean(smape_pred),
                np.mean(smape_mean),
                mean_measured.values[0],
                p_smape,
            )
        for i, gene in enumerate(index):
            _process_gene(i, gene)
        if is_protein:
            r2_df["gene_symbol"] = r2_df.index
        else:
            r2_df["gene_symbol"] = self.hgnc.convert_ensembl_ids_to_gene_names(
                r2_df.index
            )
        
        r2_df["subcellular_location"] = self._get_locations(r2_df["gene_symbol"])
        r2_df[self.pathways.get_column_names()] = self._get_pathways(r2_df["gene_symbol"])
        location_indicators = (
            r2_df.subcellular_location.str.split("\s*;\s*", expand=True)
            .stack()
            .str.get_dummies()
            .sum(level=0)
        )
        r2_df[location_indicators.columns] = location_indicators
        r2_df[location_indicators.columns] = r2_df[location_indicators.columns].fillna(
            0.0
        )
        scaler = MinMaxScaler()
        r2_df[PlotApplication.MEAN_MEASURED_SCALED_NAME] = scaler.fit_transform(
            r2_df[["mean_measured"]]
        )
        return r2_df.sort_values("r2", ascending=False)

    def _print_most_and_least_predictable(
        self,
        median_df: pd.DataFrame,
        p_val_threshold: float = 0.05,
        topk: int = 10,
        key: str = "gene/protein",
        is_protein: bool = True,
    ):
        q25_df = median_df[median_df["quantile"] == 0.025].set_index(key)
        q975_df = median_df[median_df["quantile"] == 0.975].set_index(key)
        median_df = median_df[median_df["quantile"] == 0.5].set_index(key)
        median_df = median_df.sort_values("r2", ascending=False)
        median_df = median_df[
            (median_df["p_smape"] < p_val_threshold)
            & (median_df["p"] < p_val_threshold)
        ]
        top_k_df = median_df.head(topk)
        bottom_k_df = median_df.tail(topk)

        for gene_name, row in itertools.chain(
            top_k_df.iterrows(), bottom_k_df.iterrows()
        ):
            q25_row = q25_df.loc[gene_name]
            q975_row = q975_df.loc[gene_name]

            def _get_ci(x, is_percent=True):
                unit = "\\%" if is_percent else ""
                return f"{row[x]:.2f}{unit} (95\\% CI: {q25_row[x]:.2f}{unit}, {q975_row[x]:.2f}{unit})"

            mapped_symbol = gene_name
            if not is_protein:
                mapped_symbol = self.hgnc.convert_ensembl_ids_to_gene_names(
                    mapped_symbol
                )[0]
            print(
                f"{mapped_symbol} & {_get_ci('r2', is_percent=False)} & "
                f"{_get_ci('smape_pred')} \\\\"
            )

    def _plot_pathway_violins(
        self, df: pd.DataFrame, directory: Path, pathway_names: List[str]
    ):
        for pathway_name in pathway_names:
            plot_violins(
                df,
                file_name=f"r2.by.{pathway_name.replace(' ', '_').replace('/', '--')}.pdf",
                plot_title=f"Pathway membership {pathway_name}",
                group_column=pathway_name,
                error_column="r2",
                directory=directory,
            )

    def _get_r2_df(self, file_path: Path, seed: int = 0, df: pd.DataFrame = None):
        print(file_path.split("/")[-2])
        is_protein = "proteo" in file_path.split("/")[-2]
        is_m1 = "m1" in file_path.split("/")[-2]
        r2_df_file_path = self.data_folder / (str(os.path.basename(file_path)) + f".r2.{seed}.csv")
        state = "M1" if is_m1 else "M2"
        omics_layer = "Proteomics" if is_protein else "Transcriptomics"
        if not r2_df_file_path.exists():
            if df is None:
                df = self._read_df(file_path)
            r2_df = self._calculate_r2(df, is_protein, seed=seed)
            r2_df["state"] = state
            r2_df["layer"] = omics_layer
            r2_df["type"] = f"{omics_layer} ({state})"
            r2_df.to_csv(r2_df_file_path)
            info(f"Wrote {r2_df_file_path}")
        else:
            r2_df = pd.read_csv(r2_df_file_path, index_col=0)
            info(f"Read {r2_df_file_path}")
        return r2_df, r2_df_file_path, is_protein, is_m1, state, omics_layer

    def _calculate_all_r2(self, min_pathway_size: int = 10, num_top_pathways: int = 5):
        all_dfs_path = self.data_folder / "all.r2.csv"

        all_dfs = []
        for file_path in PlotApplication.ALL_FILES:
            (
                r2_df,
                r2_df_file_path,
                is_protein,
                is_m1,
                state,
                omics_layer,
            ) = self._get_r2_df(file_path)
            pathway_violins_directory = self.data_folder / f"{omics_layer}_{state}"
            pathway_violins_directory.mkdir(exist_ok=True)
            corr_df_file_path = self.data_folder / (str(file_path) + ".corr.csv")
            if not corr_df_file_path.exists():
                # if df is None:
                df = self._read_df(file_path)
                corr_df = self._get_corr_df(df, is_protein=is_protein)
                corr_df.to_csv(corr_df_file_path)
                info(f"Wrote {corr_df_file_path}")
            else:
                corr_df = pd.read_csv(corr_df_file_path, index_col=0)
                info(f"Read {corr_df_file_path}")

            plot_violins(
                r2_df,
                file_name=f"{'proteomics' if is_protein else 'transcriptomics'}_{'m1' if is_m1 else 'm2'}.location.pdf",
                plot_title=f"{'Proteomics' if is_protein else 'Transcriptomics'} {'M1' if is_m1 else 'M2'}",
                group_column="subcellular_location",
                error_column="r2",
                directory=self.data_folder,
                topk=8,
            )
            index_list = [int(i) for i in np.where(r2_df[self.pathways.get_column_names()].sum() > min_pathway_size)[0]]
            pathway_names = [self.pathways.get_column_names()[i] for i in index_list]
            betas_df = self._get_association_with_r2(
                r2_df,
                x_labels=PlotApplication.ALL_LOCATIONS
                + pathway_names
                + [PlotApplication.MEAN_MEASURED_SCALED_NAME],
                y_label="r2",
            )
            significant_location_betas_df = betas_df.loc[PlotApplication.ALL_LOCATIONS][
                betas_df["significant?"] == "significant"
            ].sort_values("beta", ascending=False)
            significant_pathway_betas_df = betas_df.loc[pathway_names][
                betas_df["significant?"] == "significant"
            ].sort_values("beta", ascending=False)
            topk_significant_pathway_betas_df = pd.concat(
                [
                    significant_pathway_betas_df.head(num_top_pathways),
                    significant_pathway_betas_df.tail(num_top_pathways),
                ]
            )
            topk_significant_pathway_betas_plus_abundance_df = pd.concat(
                [
                    topk_significant_pathway_betas_df,
                    significant_location_betas_df,
                    betas_df.loc[[PlotApplication.MEAN_MEASURED_SCALED_NAME]],
                ]
            )
            plot_forest(
                topk_significant_pathway_betas_plus_abundance_df,
                f"{omics_layer}.{state}.forest.pdf",
                pathway_violins_directory,
                num_significant_location_betas=len(significant_location_betas_df),
                title=f"{omics_layer} ({state})",
            )
            self._plot_pathway_violins(
                r2_df,
                pathway_violins_directory,
                significant_pathway_betas_df.index,
            )
            for gene in PlotApplication.SELECTED_GENES:
                if gene in corr_df.index:
                    plot_correlation(
                        corr_df.loc[gene],
                        gene_name=gene,
                        file_name=f"{gene}.corr.pdf",
                        directory=self.data_folder / f"{omics_layer}_{state}",
                    )
                else:
                    warn(f"{gene} not in {r2_df_file_path}")
            all_dfs.append(r2_df)
        all_dfs = pd.concat(all_dfs)
        all_dfs.to_csv(all_dfs_path)

        plot_violins(
            all_dfs,
            file_name=f"violin.type.pdf",
            plot_title=f"Overall prediction performance",
            group_column="type",
            error_column="r2",
            directory=self.data_folder,
        )
        return all_dfs

    def _get_meta_dataframe(self) -> pd.DataFrame:
        meta_file_path = PlotApplication.META_FILE
        meta = pd.read_csv(meta_file_path, encoding="utf-8")
        return meta

    def _read_embedding(self, file_path: str):
        column_names = list(pd.read_csv(file_path, nrows=1))
        include_column_names = [
            col for col in column_names if "plate_barcode" not in col
        ]
        df = pd.read_csv(
            file_path,
            usecols=include_column_names,
            index_col=0,
            dtype={
                "well_id": "category",
                "train/test/val fold": "category",
                "measured (absolute)": np.float32,
                "predicted from images (absolute)": np.float32,
                "seed": np.uint8,
                "donor": np.uint8,
            },
        )
        return df

    @staticmethod
    def gmt_to_dataframe(fname):
        res = []
        with open(fname) as inf:
            for line in inf:
                f = line.rstrip().split("\t")
                name = f[0]
                description = f[1]
                members = f[2:]
                for member in members:
                    res.append(
                        pd.Series(
                            {"name": name, "description": description, "member": member}
                        )
                    )
        return pd.DataFrame(res)

    def _calculate_embeddings(self):
        df = self._read_embedding(PlotApplication.EMBEDDING)
        meta_df = self._get_meta_dataframe()
        df["PLATE_ID"] = df["PLATE_BARCODE"].str.rsplit("_", expand=True)[1]
        df = df.rename(columns={"COLUMN": "Column", "ROW": "Row"})
        col_letters = [chr(ord("a") + x - 1).upper() for x in df["Row"]]
        df["well_id"] = [f"{a}{b:d}" for a, b in zip(col_letters, df["Column"])]
        pre_len = len(df)
        df = pd.merge(df, meta_df, on=["well_id", "PLATE_ID"])
        assert len(df) == pre_len, "we should not drop any entries here."

        plot_embedding_scanpy(
            df,
            f"embedding.pdf",
            save_folder=str(self.data_folder),
        )

    def _calculate_summary_predictability_stats(self, all_seeds_df: pd.DataFrame):
        summary_df = (
            all_seeds_df.groupby(["gene/protein", "layer", "state"])
            .quantile([0.025, 0.5, 0.975])
            .reset_index()
        ).rename(columns={"level_3": "quantile"})
        return summary_df

    def _calculate_predictability_pie(self):
        all_dfs_path = self.data_folder / "all.r2.seeds.csv"
        if not all_dfs_path.exists():
            all_r2_dfs = []
            for file_path in PlotApplication.ALL_FILES:
                df = self._read_df(file_path)
                seeds = sorted(set(df["seed"]))
                for seed in seeds:
                    all_r2_dfs.append(self._get_r2_df(file_path, seed=seed, df=df)[0])
            all_dfs = pd.concat(all_r2_dfs)
            all_dfs.to_csv(all_dfs_path)
            info(f"Wrote {all_dfs_path}.")

        included_column_names = list(pd.read_csv(all_dfs_path, nrows=1))[:8] + [
            "state",
            "layer",
            "type",
        ]
        all_dfs = pd.read_csv(all_dfs_path, usecols=included_column_names)
        info(f"Read {all_dfs_path}.")
        summary_df = self._calculate_summary_predictability_stats(all_dfs)
        summary_df.to_csv("/home/rm426130/test.csv")

        data = []
        for layer in ["Transcriptomics", "Proteomics"]:
            for state in ["M1", "M2"]:
                type_string = f"{layer} ({state})"
                print(f"=== {layer} ({state}) ===")
                this_summary_df = summary_df[
                    (summary_df["state"] == state) & (summary_df["layer"] == layer)
                ]
                self._print_most_and_least_predictable(
                    this_summary_df, is_protein=layer == "Proteomics"
                )
                key: str = "gene/protein"
                q25_df = this_summary_df[
                    this_summary_df["quantile"] == 0.025
                ].set_index(key)
                q975_df = this_summary_df[
                    this_summary_df["quantile"] == 0.975
                ].set_index(key)
                median_df = this_summary_df[
                    this_summary_df["quantile"] == 0.5
                ].set_index(key)

                pp_not_filtered = (
                    len(median_df) / PlotApplication.TOTAL_GENES[type_string]
                )
                pp_signif = (
                    len(median_df[median_df["p_smape"] < 0.05])
                    / PlotApplication.TOTAL_GENES[type_string]
                )
                pp_signif_q25 = (
                    len(q25_df[q25_df["p_smape"] < 0.05])
                    / PlotApplication.TOTAL_GENES[type_string]
                )
                pp_signif_q975 = (
                    len(q975_df[q975_df["p_smape"] < 0.05])
                    / PlotApplication.TOTAL_GENES[type_string]
                )
                data_row = [
                    type_string,
                    pp_signif,
                    pp_signif_q25,
                    pp_signif_q975,
                    pp_not_filtered,
                ]
                data.append(data_row)
                pp_not_signif = pp_not_filtered - pp_signif
                plot_pie(
                    [
                        pp_signif,
                        pp_not_signif,
                        1 - pp_not_signif - pp_signif,
                    ],
                    labels=[
                        f"Significant {pp_signif*100:.2f}%\n(95% CI: {pp_signif_q975*100:.2f}, {pp_signif_q25*100:.2f})",
                        f"Not significant {(pp_not_signif)*100:.2f}%"
                        f"\n(95% CI: {(pp_not_filtered - pp_signif_q25)*100:.2f}, "
                        f"{(pp_not_filtered - pp_signif_q975)*100:.2f})",
                        f"Low/no expression\n(filtered) {(1 - pp_not_signif - pp_signif)*100:.2f}%",
                    ],
                    plot_title=f"{type_string} [n={PlotApplication.TOTAL_GENES[type_string]}]",
                    file_name=f"{layer}.{state}.pie.pdf",
                    directory=self.data_folder,
                )
        return all_dfs

    def _selected_tiles_for_embeddings(self):
        d = {"M1_RELA": (9,9), "M1_NT": (7,12), "M1_STAT1": (8,15), "M0_NT":(7,13), "M2_STAT6": (9,15), "M2_NT":(7,12)}
        fov = 1

        for k, (c,r) in d.items():
            if k.split("_")[0] == 'M1':
                icf_file = f"{PlotApplication.ICF_FILES}/ICF.X{c}.Y{r}.P1.2021159114145.ELN27012_G212LFO.h5"
                icf = h5py.File(icf_file,"r")
                tile_file = f"{PlotApplication.TILES_FILES}/TILES.X{c}.Y{r}.P1.2021159114145.ELN27012_G212LFO.h5"
                f = h5py.File(tile_file,"r")
            else:
                icf_file = f"{PlotApplication.ICF_FILES}/ICF.X{c}.Y{r}.P1.2021155130232.ELN27012_G212LFI.h5"
                icf = h5py.File(icf_file,"r")
                tile_file = f"{PlotApplication.TILES_FILES}/TILES.X{c}.Y{r}.P1.2021155130232.ELN27012_G212LFI.h5"
                f = h5py.File(tile_file,"r")

            img_b = icf['HOECHST 33342_em456_ex405']['fov_1'][...]
            img_g = icf['MitoTracker Orange_em599_ex561']['fov_1'][...]
            img_r = icf['Alexa 647_em706_ex640']['fov_1'][...]

            img_b = np.clip(img_b,*np.percentile(img_b, (0.5, 99.5)))
            img_g = np.clip(img_g,*np.percentile(img_g, (0.5, 99.5)))
            img_r = np.clip(img_r,*np.percentile(img_r, (0.5, 99.5)))
            
            img_b = img_b - np.amin(img_b)
            img_g = img_g - np.amin(img_g)
            img_r = img_r - np.amin(img_r)
            
            img_b = exposure.rescale_intensity(img_b, in_range='uint16', out_range="float")
            img_g = exposure.rescale_intensity(img_g, in_range='uint16', out_range="float")
            img_r = exposure.rescale_intensity(img_r, in_range='uint12', out_range="float")
            
            img_b = exposure.adjust_gamma(img_b, gain=15)
            img_g = exposure.adjust_gamma(img_g, gain=3.5)
            img_r = exposure.adjust_gamma(img_r, gain=1.5)

            img_r = np.expand_dims(img_r, axis=2)
            img_g = np.expand_dims(img_g, axis=2)
            img_b = np.expand_dims(img_b, axis=2)

            img = np.append(img_r, img_g, axis=2)
            img = np.append(img, img_b, axis=2)
            img = exposure.equalize_adapthist(img)

            fig, axes = plt.subplots(3,3,figsize=(10,9.99))
            for i in range(3):
                for j in range(3):
                    x = int(f[f"fov_1"][...][i*3+j,1])
                    y = int(f[f"fov_1"][...][i*3+j,0])
                    axes[i,j].imshow(img[y:y+128,x:x+128])
                    axes[i,j].axis('off')

            plt.subplots_adjust(wspace=0, hspace=0.02)
            output_file_path = self.data_folder / f"{k}.pdf"
            if not output_file_path.exists():
                plt.savefig(output_file_path)

    def run(self):
        self._setup()
        self._calculate_all_r2()
        self._calculate_predictability_pie()
        self._calculate_embeddings()
        self._selected_tiles_for_embeddings()


if __name__ == "__main__":
    app = sp.instantiate_from_command_line(PlotApplication)
    results = app.run()
