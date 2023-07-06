"""
Copyright (C) 2021  Patrick Schwab, GlaxoSmithKline plc
"""
import os
from typing import AnyStr

import numpy as np
import pandas as pd
import slingpy as sp
from image2omics.datasets.hgnc_names import HGNCNames


class SubcellularLocation(sp.AbstractHDF5Dataset):
    """
    SOURCE: https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-018-03106-1/MediaObjects/41467_2018_3106_MOESM5_ESM.xlsx
    PROJECT: https://www.nature.com/articles/s41467-018-03106-1#Sec25

    CITE:
    https://www.nature.com/articles/s41467-018-03106-1#Sec25

    LICENSE: public domain
    """

    ORIGINAL_FILE_URL = (
        "https://www.proteinatlas.org/download/subcellular_location.tsv.zip"
    )

    def __init__(
        self,
        save_directory: AnyStr,
        in_memory: bool = False,
        fill_missing: bool = True,
        fill_missing_value: float = float("nan"),
        duplicate_merge_strategy: sp.AbstractMergeStrategy = sp.NoMergeStrategy(),
    ):
        super().__init__(
            save_directory=save_directory,
            in_memory=in_memory,
            fill_missing=fill_missing,
            fill_missing_value=fill_missing_value,
            duplicate_merge_strategy=duplicate_merge_strategy,
        )

    def _load(self) -> sp.DatasetLoadResult:
        file_path = os.path.join(self.save_directory, "SubcellularLocation/subcellular_location.tsv.zip")
        df = pd.read_csv(file_path, sep="\t", compression="zip", index_col="Gene")
        df = df[df["Reliability"] == "Approved"]["Main location"]
        name_converter = HGNCNames(self.save_directory)
        df.index = name_converter.convert_ensembl_ids_to_gene_names(df.index)
        df = df[~df.index.isnull()]
        unique_locations = list(set(df.values))
        code_list = dict(zip(unique_locations, range(len(unique_locations))))
        column_code_lists = [dict(zip(range(len(unique_locations)), unique_locations))]
        df = df.apply(lambda x: code_list[x])
        load_result = sp.DatasetLoadResult(
            df.values[:, np.newaxis],
            type(self).__name__,
            column_names=["location"],
            row_names=df.index.values.tolist(),
            column_code_lists=column_code_lists,
        )
        return load_result


if __name__ == "__main__":
    app = sp.instantiate_from_command_line(SubcellularLocation)
    results = app.run()
