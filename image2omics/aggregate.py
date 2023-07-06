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

import logging
import sys
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd


def aggregate_array(
    by: Union[str, List[str]],
    features: Dict[str, List],
    metadata: List[dict],
    op: Union[dict, str] = "mean",
    array_col: str = "feat",
    metadata_to_dict_arg: str = "records",
) -> Tuple[np.ndarray, Union[dict, list]]:
    """Aggregate feature based on specified metadata keys and operation
    This function uses pandas GroupBy for aggregation
    Parameters
    ----------
    by : List[str]
        Keys in metadata to aggregate by
    features : Dict[str, List],
        a dictiory containing 2D Array to be aggregated with shape (N, F) for each features
    metadata : List[dict]
        Metadata of the array with length N. Each element is a
        dictionary containing metadata for each row of the array.
    op : str, optional
        Aggregation operation, by default "mean"
        Must be in the namespace of pd.core.groupby.GroupBy.
        Currently supports ['mean', 'median', 'max', 'sum']
    metadata_to_dict_arg : str, optional
        Argument for pd.DataFrame.to_dict() method, by default "records"
        Currently supports ['records', 'list']
    Returns
    -------
    Tuple[np.ndarray, Union[dict, list]]
        Tuple of (aggregated array, aggregated metadata)
    """
    if isinstance(by, str):
        by = [by]

    for key in features.keys():
        if len(features[key]) != len(metadata):
            logging.error(
                f"Number of elements in feature ({len(features[key])}) "
                + f"and metadata ({len(metadata)}) are not equal."
            )
            sys.exit(1)

    if metadata_to_dict_arg not in ["records", "list"]:
        logging.error(
            f"Unsupported metadata_to_dict_arg ({metadata_to_dict_arg}). "
            + "Must be in ['records', 'list']"
        )
        sys.exit(1)

    if isinstance(op, dict) and "ARRAY" not in op:
        logging.error("When op is dict, the key ARRAY must also be provided.")
        sys.exit(1)

    df = pd.DataFrame(metadata)

    missing_cols = set(by) - set(df.columns)
    if missing_cols:
        logging.error(
            f"Keys to aggregate by ({list(missing_cols)}) are not found in provided metadata."
        )
        sys.exit(1)

    metadata_col = list(df.columns)
    for key in features.keys():
        if features[key].ndim > 2:
            features[key] = features[key].reshape(features[key].shape[0], -1)
        feature_col = [f"{key}_{i}" for i in range(len(features[key].T))]
        df = pd.concat([df, pd.DataFrame(features[key], columns=feature_col)], axis=1)

        if isinstance(op, dict):
            # Expand the key ARRAY to all feature_col
            op = {**op, **{col: op["ARRAY"] for col in feature_col}}
    del op["ARRAY"]
    if op == "var":
        df_grouped = aggregate_df_pooled_variance(df=df, by=by)
    else:
        df_grouped = aggregate_df(df=df, by=by, op=op)

    array_agg = {}
    for key in features.keys():
        feature_col = [f"{key}_{i}" for i in range(len(features[key].T))]
        array_agg[key] = df_grouped[feature_col].values
    metadata_agg = df_grouped[metadata_col]
    metadata_agg = metadata_agg.to_dict(metadata_to_dict_arg)

    return array_agg, metadata_agg


def aggregate_df(df: pd.DataFrame, by: Union[str, List[str]], op: Union[Dict, str]):
    if isinstance(by, str):
        by = [by]

    df_grouped = df.groupby(by=by, as_index=False)
    if isinstance(op, str):
        if op not in ["mean", "median", "max", "sum"]:
            logging.error(
                f"Unsupported op ({op}). Must be in ['mean', 'median', 'max', 'sum']"
            )
            sys.exit(1)
        df_grouped = getattr(df_grouped, op)()
    elif isinstance(op, dict):
        df_grouped = df_grouped.agg(op)
    else:
        logging.error(
            f"Unsupported type ({type(op)}) for op. Must be Union[Dict, str]."
        )
        sys.exit(1)

    # select non-overlapping columns and take the first of metadata
    cols_to_use = list(df.columns.difference(df_grouped.columns)) + by
    df = df[cols_to_use].groupby(by=by, as_index=False)
    df = df.first().reset_index(drop=True)

    df_grouped = df_grouped.merge(df, how="left", on=by).reset_index(drop=True)
    return df_grouped


def aggregate_df_pooled_variance(df: pd.DataFrame, by: Union[str, List[str]]):
    if isinstance(by, str):
        by = [by]

    # first, calculate the variance on the tile level to determine model uncertainty
    df_grouped = df.groupby(
        by=["ROW", "COLUMN", "FOV", "PLATE_BARCODE", "TILE"], as_index=False
    )
    df_grouped = getattr(df_grouped, "var")()

    # then, calculate the pooled variance on the "by" level to aggregate results
    df_grouped = df_grouped.groupby(by=by, as_index=False)
    df_grouped = getattr(df_grouped, "mean")()

    # select non-overlapping columns and take the first of metadata
    cols_to_use = list(df.columns.difference(df_grouped.columns)) + by
    df = df[cols_to_use].groupby(by=by, as_index=False)
    df = df.first().reset_index(drop=True)

    df_grouped = df_grouped.merge(df, how="left", on=by).reset_index(drop=True)
    return df_grouped
