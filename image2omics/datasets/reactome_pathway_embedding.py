"""
Copyright (C) 2021  Patrick Schwab, GlaxoSmithKline plc
"""
import os
import zipfile
from typing import AnyStr, Dict, List

import numpy as np
import slingpy as sp
from image2omics.datasets.hgnc_names import HGNCNames


class ReactomePathwayEmbedding(sp.AbstractHDF5Dataset):
    """
    SOURCE:
    https://reactome.org/download/current/ReactomePathways.gmt.zip

    PROJECT: https://reactome.org/download-data

    CITE:
    ?

    ACKNOWLEDGE:
    ?

    LICENSE: https://creativecommons.org/licenses/by/4.0/ (see https://reactome.org/license)
    """

    ORIGINAL_FILE_URL = "https://reactome.org/download/current/ReactomePathways.gmt.zip"

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

    @staticmethod
    def _read_gmt_into_pathway_dict(
        file_path: str, name_converter: HGNCNames
    ) -> Dict[str, List[str]]:
        all_pathways: Dict[str, List[str]] = {}
        with zipfile.ZipFile(file_path) as zp:
            with zp.open(zp.namelist()[0]) as fp:
                for line in fp:
                    tokens = line.decode("utf-8").rstrip().split("\t")
                    pathway_name, members = str(tokens[0]), tokens[2:]
                    members = name_converter.update_outdated_gene_names(
                        list(map(str, members)), verbose=False
                    )
                    all_pathways[pathway_name] = members
        return all_pathways

    def _load(self) -> sp.DatasetLoadResult:
        gmt_file_path = os.path.join(self.save_directory, "ReactomePathwayEmbedding/ReactomePathways_23Nov2022.gmt.zip")
        name_converter = HGNCNames(self.save_directory)
        reference_names = name_converter.get_gene_names()
        pathway_dict = ReactomePathwayEmbedding._read_gmt_into_pathway_dict(
            gmt_file_path, name_converter
        )

        indicator_matrix = np.zeros(
            (
                len(reference_names),
                len(pathway_dict),
            )
        )
        pathway_names = list(map(str, pathway_dict.keys()))
        gene_index_map = dict(zip(reference_names, range(len(reference_names))))
        for pathway_index, (pathway_name, gene_set) in enumerate(
            list(pathway_dict.items())
        ):
            gene_indices = [
                gene_index_map[gene_name]
                for gene_name in gene_set
                if gene_name in gene_index_map
            ]
            indicator_matrix[gene_indices, pathway_index] = 1

        row_names = reference_names
        col_names = pathway_names

        load_result = sp.DatasetLoadResult(
            indicator_matrix,
            type(self).__name__,
            column_names=col_names,
            row_names=row_names,
        )
        return load_result


if __name__ == "__main__":
    app = sp.instantiate_from_command_line(ReactomePathwayEmbedding)
    results = app.run()
