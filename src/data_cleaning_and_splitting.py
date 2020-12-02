# author: Justin Fu
# date: 2020-11-19

"""
Usage: data_cleaning_and_splitting.py --in_file=<in_file> --out_file=<out_file> 
 
Options:
<in_path>           path including filename of the input file to process
<out_path>          path to where the processed data will be written to

This command/script will take the input file, cleanse, process, transform and split data to files in the path specified in the argument out_path.
It will create the folder processed and output the processed files prefixed with the original filename.

Example:
    python src/data_cleaning_and_splitting.py --in_file="data/raw/bank-additional-full.csv" --out_file="data"

"""

import os
import pandas as pd
from docopt import docopt


from sklearn.model_selection import train_test_split
import ntpath

opt = docopt(__doc__)


def main(in_filename, out_path):
    """This function takes user inputs of raw data file and
    performs data cleaning, preprocessing, and splitting
    and saves output in specified path

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

    # Transform targets to numeric so they can be supported by recall, precision and f1 score metrics
    df.loc[df.y == "no", "y"] = 0
    df.loc[df.y == "yes", "y"] = 1
    df["y"] = pd.to_numeric(df["y"])

    # Change feature names
    column_names = {
        "duration": "last_contact_duration",
        "campaign": "contacts_during_campaign",
        "pdays": "days_after_previous_contact",
        "previous": "previous_contacts",
        "emp.var.rate": "employment_variation_rate",
        "cons.price.idx": "consumer_price_index",
        "cons.conf.idx": "consumer_confidence_index",
        "euribor3m": "euribor_3_month_rate",
        "nr.employed": "number_of_employees",
        "marital": "marital_status",
        "poutcome": "previous_outcome",
        "y": "target",
    }

    df.rename(columns=column_names, inplace=True)

    # split data into train and test
    train_df, test_df = train_test_split(df, test_size=0.20, random_state=123)

    output_folder = out_path + "/processed/"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # get base filename
    prefix_file = ntpath.basename(in_filename).split(".")[0]

    train_df.to_csv(output_folder + prefix_file + "_train.csv", index=False)
    test_df.to_csv(output_folder + prefix_file + "_test.csv", index=False)

    print(
        "Successfully cleaned, pre-processed, transformed and split to the folder: "
        + output_folder
    )


if __name__ == "__main__":
    main(opt["--in_file"], opt["--out_file"])
