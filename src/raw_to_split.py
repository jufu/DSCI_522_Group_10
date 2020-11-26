# author: Justin Fu
# date: 2020-11-19

"""
Usage: raw_to_split.py --in_file=<in_file> --out_file=<out_file> 
 
Options:
<in_path>           path including filename of the input file to process
<out_path>          path to where the processed data will be written to

This command/script will take the input file, cleanse, process, transform and split data to files in the path specified in the argument out_path.
It will create the folder processed and output the processed files prefixed with the original filename.

Example:
    python raw_to_split.py --in_file="../data/raw/bank-additional-full.csv" --out_file="../data"

"""

import os
from io import BytesIO
import pandas as pd
from docopt import docopt

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
import numpy as np
import pandas as pd
from sklearn.compose import (
    ColumnTransformer,
    TransformedTargetRegressor,
    make_column_transformer,
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
import ntpath

opt = docopt(__doc__)


def main(in_filename, out_path):
    """[summary]

    Parameters
    ----------
    in_filename : string
        path including filename of the input file to process
    out_path : string
        path to where the processed data will be written to

    Example
    ----------
    main(f"../data/raw/bank-additional-full.csv", "../data")
    """

    bank_add_full = None
    try:

        bank_add_full = pd.read_csv(in_filename, sep=";")

    except Exception as e:
        print("Error: ", e)

    df = bank_add_full
    train_df, test_df = train_test_split(df, test_size=0.20, random_state=123)
    X_train = train_df.drop(columns=["y"])
    y_train = train_df["y"]
    X_test = test_df.drop(columns=["y"])
    y_test = test_df["y"]

    drop_features = []
    numeric_features = [
        "age",
        "duration",
        "campaign",
        "pdays",
        "previous",
        "emp.var.rate",
        "cons.price.idx",
        "cons.conf.idx",
        "euribor3m",
        "nr.employed",
    ]
    categorical_features = [
        "job",
        "marital",
        "default",
        "housing",
        "loan",
        "poutcome",
    ]
    ordinal_features = ["education"]
    ordering = [
        "illiterate",
        "basic.4y",
        "basic.6y",
        "basic.9y",
        "high.school",
        "professional.course",
        "university.degree",
        "unknown",
    ]
    target = ["y"]

    numeric_transformer = make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler()
    )
    ordinal_transformer = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OrdinalEncoder(categories=[ordering] * len(ordinal_features)),
    )
    categorical_transformer = make_pipeline(
        SimpleImputer(strategy="constant", fill_value="missing"),
        OneHotEncoder(handle_unknown="ignore", sparse=False),
    )

    preprocessor = make_column_transformer(
        ("drop", drop_features),
        (numeric_transformer, numeric_features),
        (ordinal_transformer, ordinal_features),
        (categorical_transformer, categorical_features),
    )

    preprocessor.fit(X_train)

    ohe_columns = list(
        preprocessor.named_transformers_["pipeline-3"]
        .named_steps["onehotencoder"]
        .get_feature_names(categorical_features)
    )
    new_columns = numeric_features + ordinal_features + ohe_columns

    X_train_enc = pd.DataFrame(
        preprocessor.transform(X_train), index=X_train.index, columns=new_columns
    )

    output_folder = out_path + "/processed/"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # get base filename
    prefix_file = ntpath.basename(in_filename).split(".")[0]

    X_train_enc.to_csv(output_folder + prefix_file + "_X_train_enc.csv")
    X_train.to_csv(output_folder + prefix_file + "_X_train.csv")
    y_train.to_csv(output_folder + prefix_file + "_y_train.csv")
    X_test.to_csv(output_folder + prefix_file + "_X_test.csv")
    y_test.to_csv(output_folder + prefix_file + "_y_test.csv")
    print("Successfully cleansed, pre-processed, transformed and split to the folder: " + output_folder)


if __name__ == "__main__":
    main(opt["--in_file"], opt["--out_file"])
