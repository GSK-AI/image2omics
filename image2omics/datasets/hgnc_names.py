"""
Copyright (C) 2021  Patrick Schwab, GlaxoSmithKline plc
"""
import gzip
import os
import traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import AnyStr, Dict, List, Optional, TypeVar

import numpy as np
import pandas as pd
from gtfparse import read_gtf
from slingpy.utils.logging import warn

StrListOrSeries = TypeVar("StrListOrSeries", List[str], pd.Series)


class HGNCNames:

    def __init__(self, cache_directory: AnyStr):
        self.cache_directory = cache_directory
        self._name_updates_mapping: Optional[pd.DataFrame] = None
        self._report_csv_path: Optional[Path] = None

    def get_gene_names(self):
        mapping_data = self._load_hgnc()
        gene_names = mapping_data.symbol.to_list()
        return gene_names

    def _load_ensembl_symbol_mapping(self):
        # ensembl_symbol_file = self._get_hgnc_ensembl_symbol_file()
        ensembl_symbol_file = os.path.join(self.cache_directory, "ensembl_symbol.csv")
        mapping_data = pd.read_csv(ensembl_symbol_file)
        return mapping_data

    def _load_hgnc(self):
        # tsv_file = self._get_hgnc_master_file()
        tsv_file = os.path.join(self.cache_directory, "HGNCNames/hgnc_mapping.tsv")
        mapping_data = pd.read_csv(
            tsv_file,
            sep="\t",
            keep_default_na=False,
            na_values=[""],
            low_memory=False,
        )

        return mapping_data

    def _get_gene_name_updates_mapping(self):
        """
        Builds a table mapping old symbols to new symbols, including debug columns for
        the exception report. Columns:
            <index>: old symbol
            rename_to: best candidate to rename to (same as <index> if no rename is needed)
            next_symbol: comma-separated list of symbols whose `prev_symbol` includes this symbol
            alias_of: comma-separated list of symbols whose `aliases` includes this symbol
            is_renamed: True if a rename is necessary
            is_ambiguous: True if the new symbol couldn't be unambiguously chosen

        Renaming uses these rules:
        1. If `symbol` is already in HGNC: don't rename it
        2. Otherwise, if `symbol` is in another symbol's `prev_symbol`: rename to that other symbol
        3. Otherwise, if `symbol` is in another symbol's `aliases`: rename to that other symbol
        """
        if self._name_updates_mapping is None:
            hgnc = self._load_hgnc().set_index("symbol")

            # Make inverse prev_symbol lookup
            next_symbols = defaultdict(list)
            prev_symbols = hgnc.prev_symbol[hgnc.prev_symbol.notna()]
            for symbol, prev_symbols in prev_symbols.items():
                for prev_symbol in prev_symbols.split("|"):
                    next_symbols[prev_symbol].append(symbol)

            # Make inverse aliases lookup
            alias_ofs = defaultdict(list)
            alias_symbols = hgnc.alias_symbol[hgnc.alias_symbol.notna()]
            for symbol, aliases in alias_symbols.items():
                for alias in aliases.split("|"):
                    alias_ofs[alias].append(symbol)

            # Add an alias to fix incorrectly upper-cased symbols (e.g. "C11ORF1" should be "C11orf1").
            # Among protein-coding genes, currently only 262 genes containing "orf" (="open reading frame") are
            # affected, and they can be renamed unambiguously. If HGNC changes or the inclusion of non-protein-coding
            # genes on this codepath cause these aliases to become ambiguous, it may be necessary to rethink this
            # solution, e.g. by only applying this renaming on datasets that consistently use the incorrect case.
            mixed_case_symbols = hgnc.index[hgnc.index.str.contains("[a-z]", case=True)]
            if hgnc.index.isin(mixed_case_symbols.str.upper()).any():
                raise AssertionError("HGNC contains ambiguous mixed-case genes")
            for symbol in mixed_case_symbols:
                alias_ofs[symbol.upper()].append(symbol)

            all_symbols = set(hgnc.index).union(next_symbols.keys(), alias_ofs.keys())

            mappings = []
            for symbol in sorted(all_symbols):
                next_candidates = next_symbols.get(symbol, None)
                alias_candidates = alias_ofs.get(symbol, None)
                if symbol in hgnc.index:
                    rename_to = symbol
                    is_ambiguous = False
                elif next_candidates is not None:
                    rename_to = next_candidates[0]
                    is_ambiguous = len(next_candidates) > 1
                else:
                    rename_to = alias_candidates[0]
                    is_ambiguous = len(alias_candidates) > 1

                next_symbol = (
                    np.nan if next_candidates is None else ",".join(next_candidates)
                )
                alias_of = (
                    np.nan if alias_candidates is None else ",".join(alias_candidates)
                )
                mappings.append(
                    {
                        "symbol": symbol,
                        "rename_to": rename_to,
                        "next_symbol": next_symbol,
                        "alias_of": alias_of,
                        "is_renamed": rename_to != symbol,
                        "is_ambiguous": is_ambiguous,
                    }
                )

            self._name_updates_mapping = pd.DataFrame(mappings).set_index("symbol")

        return self._name_updates_mapping

    def get_hgnc_mapping(
        self, from_id: str, to_id: str, exclude_nan=True
    ) -> Dict[str, str]:
        if (from_id == "ensembl_gene_id" and to_id == "symbol") or (
            from_id == "symbol" and to_id == "ensembl_gene_id"
        ):
            mapping_data = self._load_ensembl_symbol_mapping()
            # Update to latest gene names without producing an exception report
            # as many rows won't necessarily be used
            gene_name_mappings = self._get_gene_name_updates(mapping_data.symbol)
            mapping_data = mapping_data.assign(symbol=gene_name_mappings.rename_to)
        else:
            mapping_data = self._load_hgnc()

        mapping = mapping_data.set_index(from_id)[to_id]

        if exclude_nan:
            mapping = mapping[mapping.index.notna() & mapping.notna()]

        return mapping.to_dict()

    def get_hgnc_multimapping(self, from_id: str, to_id: str) -> Dict[str, List[str]]:
        mapping = self.get_hgnc_mapping(from_id, to_id)

        multimapping = {
            key: value.split("|") if isinstance(value, str) else [value]
            for key, value in mapping.items()
        }

        return multimapping

    @staticmethod
    def _get_caller_and_stacktrace(skip_stack):
        stack = traceback.extract_stack()[:-skip_stack]
        try:
            caller_name = Path(stack[-1].filename).stem
        except:
            caller_name = None

        stacktrace = "".join(traceback.format_list(stack))

        return caller_name, stacktrace

    def _get_gene_name_updates(self, gene_names):
        name_updates_mapping = self._get_gene_name_updates_mapping()
        gene_names = pd.Series(gene_names, name="symbol")

        # Select rows in the same order as gene_names
        mapped_names = (
            gene_names.to_frame()
            .merge(name_updates_mapping, on="symbol", how="left")
            .set_index(gene_names.index)
        )
        mapped_names["is_missing"] = mapped_names.is_renamed.isna()

        # Fix boolean columns that have become Object-typed due to nans
        mapped_names["is_renamed"] = mapped_names["is_renamed"] == True
        mapped_names["is_ambiguous"] = mapped_names["is_ambiguous"] == True

        # Keep the original symbol if there's no `rename_to`
        mapped_names["rename_to"] = mapped_names.rename_to.where(
            mapped_names.rename_to.notna(), mapped_names.symbol
        )

        return mapped_names

    def update_outdated_gene_names(
        self, gene_names: StrListOrSeries, verbose=False
    ) -> StrListOrSeries:
        caller, stacktrace = HGNCNames._get_caller_and_stacktrace(skip_stack=2)
        mapped_names = self._get_gene_name_updates(gene_names)

        self._report_name_update_exceptions(mapped_names, verbose, caller, stacktrace)

        if isinstance(gene_names, pd.Series):
            assert (gene_names.index == mapped_names.index).all()
            return mapped_names.rename_to
        else:
            return mapped_names.rename_to.to_list()

    def convert_ensembl_ids_to_gene_names(
        self,
        ensembl_ids: StrListOrSeries,
        verbose=False,
        preserve_unmapped=False,
    ) -> StrListOrSeries:
        """Maps Ensembl IDs to HGNC symbols
        Args:
            ensembl_ids: list of IDs to map
            verbose: if True, all mappings will be saved to the exception report
            preserve_unmapped: if False, IDs that could not be mapped to an HGNC symbol
                will be None. If True, they will keep their original Ensembl ID.
        Returns:
            A list-of-strings or pd.Series, depending on the type of ensembl_ids
        """
        caller, stacktrace = HGNCNames._get_caller_and_stacktrace(skip_stack=2)

        # Truncate to 15 characters to remove version suffix e.g.
        # ENSG00000223972.5 -> ENSG00000223972
        clean_ids = pd.Series(ensembl_ids, name="ensembl_gene_id").str.slice(None, 15)

        # Convert using original Ensembl mapping (with potentially outdated names)
        mapping_data = self._load_ensembl_symbol_mapping()[
            ["ensembl_gene_id", "symbol"]
        ]
        converted_names = (
            clean_ids.to_frame()
            .merge(mapping_data, on="ensembl_gene_id", how="left")
            .set_index(clean_ids.index)
            # Restore original ID for report / preserve_unmapped
            .assign(ensembl_gene_id=ensembl_ids)
        )

        # Update any out-of-date names & generate report
        name_updates = self._get_gene_name_updates(converted_names.symbol)
        report_data = converted_names[["ensembl_gene_id"]].join(name_updates)
        self._report_name_update_exceptions(report_data, verbose, caller, stacktrace)

        # Fill missing values (nan) with their original ID or None
        fill_value = report_data.ensembl_gene_id.values if preserve_unmapped else None
        rename_to = report_data.rename_to.where(
            report_data.rename_to.notna(), fill_value
        )

        if isinstance(ensembl_ids, pd.Series):
            assert (ensembl_ids.index == rename_to.index).all()
            return rename_to
        else:
            return rename_to.to_list()

    def _report_name_update_exceptions(self, mapped_names, verbose, caller, stacktrace):
        # Find cases where two different input symbols map to the same `rename_to`
        collisions = set(
            mapped_names[~mapped_names.is_missing][["rename_to"]]
            .reset_index()
            .drop_duplicates()
            .groupby("rename_to")
            .filter(lambda grp: len(grp) > 1)
            .rename_to
        )
        mapped_names = mapped_names.assign(
            is_collision=mapped_names.rename_to.isin(collisions)
        )

        # "== True" is needed here to coerce nans to False so that np.count_nonzero
        # counts Trues instead of non-None values
        n_renamed = np.count_nonzero(mapped_names.is_renamed == True)
        n_ambiguous = np.count_nonzero(mapped_names.is_ambiguous == True)
        n_missing = np.count_nonzero(mapped_names.is_missing == True)
        n_collisions = np.count_nonzero(mapped_names.is_collision == True)
        n_unchanged = np.count_nonzero(
            (mapped_names.is_renamed == False) & (mapped_names.is_missing == False)
        )

        if verbose or n_ambiguous > 0 or n_missing > 0 or n_collisions > 0:

            report_data = mapped_names
            if not verbose:
                report_data = report_data[
                    report_data.is_missing
                    | report_data.is_ambiguous
                    | report_data.is_collision
                ]
                report_data = report_data[~report_data.index.duplicated()]

            if self._report_csv_path is None:
                # First call with this instance. Create a new report with the stacktrace
                report_path = (
                    Path(self.cache_directory) / f"{type(self).__name__}_exceptions"
                )
                report_path.mkdir(parents=True, exist_ok=True)
                report_name = datetime.now().isoformat().replace(":", "_")
                if caller:
                    report_name += f"_{caller}"

                self._report_csv_path = report_path / f"{report_name}.csv"
                stacktrace_path = report_path / f"{report_name}_stacktrace.txt"

                report_data.to_csv(self._report_csv_path)
                stacktrace_path.write_text(stacktrace)
            else:
                # Append to existing file
                report_data.to_csv(self._report_csv_path, header=False, mode="a")

            warn(
                f"HGNCNames renamed {n_renamed} genes, "
                f"{n_unchanged} were up-to-date, "
                f"{n_ambiguous} were ambiguous, "
                f"{n_missing} were unrecognized, "
                f"{n_collisions} became non-unique after renaming. "
                f"Report written to {self._report_csv_path}."
            )
