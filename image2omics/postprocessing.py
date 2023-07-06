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

import argparse
import os

import pandas as pd


def main(gt_dir: str, features_dir: str) -> None:
    """Merging the predictions with the gt data, reverse standardization of predictions and saving the results in csv format

    Parameters
    ----------
    gt_dir : str
        Path to the groundtruth data
    features_dir : str
        Path to the predictions
    """
    # read data
    gt = pd.read_csv(gt_dir)
    targets = gt["gene/protein"].unique()
    df_all = []
    for f in os.listdir(features_dir):
        if f.startswith("metadata_features_model"):
            model_seed = f.split("_")[-1].split(".")[0]
            df = pd.read_csv(os.path.join(features_dir, f))
            cols = [col for col in df.columns if col.startswith("head.final")]
            df = df[['PLATE_BARCODE','ROW','COLUMN']+cols]
            df.columns = ['PLATE_BARCODE','ROW','COLUMN'] + list(targets)

            # defining well id and updating plate barcodes
            df["well_id"] = df["ROW"].apply(lambda x: chr(x + 64))
            df["well_id"] = df["well_id"] + df["COLUMN"].astype(str)
            df.drop(["COLUMN", "ROW"], axis=1, inplace=True)

            # merging predictions with gt data
            df.set_index(["PLATE_BARCODE", "well_id"], inplace=True)
            df = df.stack()
            df = df.reset_index()
            df.rename(
                columns={
                    "PLATE_BARCODE": "plate_barcode",
                    "level_2": "gene/protein",
                    0: "predicted from images (absolute)",
                },
                inplace=True,
            )
            df = df.merge(gt, on=["plate_barcode", "well_id", "gene/protein"], how="left")
            df = df.dropna()

            # reverting the prediction back to the original space
            Mean = (
                df[df["train/test/val fold"] == "train"]
                .groupby("gene/protein")["measured (absolute)"]
                .mean()
            )
            Std = (
                df[df["train/test/val fold"] == "train"]
                .groupby("gene/protein")["measured (absolute)"]
                .std()
            )
            Mean = Mean.to_dict()
            Std = Std.to_dict()
            df["predicted from images (absolute)"] = df[
                ["gene/protein", "predicted from images (absolute)"]
            ].apply(lambda x: x[1] * Std[x[0]] + Mean[x[0]], axis=1)

            df["seed"] = int(model_seed)
            df_all.append(df)

    #concatenating predictions
    df_all = pd.concat(df_all, ignore_index=True)
    df_all = df_all[['plate_barcode','well_id','seed','gene/protein','train/test/val fold','measured (absolute)','predicted from images (absolute)']]
    
    # saving predictions
    filename = features_dir.split("/")[-1] + "_predictions.csv"
    df_all.to_csv(os.path.join(features_dir, filename))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--gt-dir", help="Path to the groudtruth data", required=True
    )
    parser.add_argument(
        "-o", "--features-dir", help="Path to the features.h5 file", required=True
    )

    args = parser.parse_args()

    main(args.gt_dir, args.features_dir)
