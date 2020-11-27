"""author: Asma Al-Odaini
date: 2020-11-25
This script imports the 'bank_add_full.csv' file and generates visualizations to explore the training data.
Usage: data_vis.py <data_path> [--image_path=<image_path>]

Options:
<data_path>      Takes a path of the training data csv (this is a required positional argument)
[--image_path=<image_path>]     Takes a path for images (this is an optional option)

Example:
    python src/data_vis.py "data/raw/bank-additional-full.csv" --image_path="results/"

"""

import pandas as pd
import altair as alt
from sklearn.model_selection import train_test_split
from docopt import docopt

alt.renderers.enable("mimetype")
alt.data_transformers.enable("data_server")

opt = docopt(__doc__)

bank_add_full = pd.read_csv(opt["<data_path>"], sep=";")

df = bank_add_full
train_df, test_df = train_test_split(df, test_size=0.20, random_state=123)

alt.data_transformers.disable_max_rows()


def main(opt):
    target = "y"

    categorical_features = {
        "job": "Job",
        "marital": "Marital Status",
        "default": "Default",
        "housing": "Housing",
        "loan": "Loan",
        "poutcome": "Previous Outcome",
        "contact": "Contact",
        "education": "Education",
        "day_of_week": "Day of the week",
        "month": "Month",
    }

    education_ordering = [
        "illiterate",
        "basic.4y",
        "basic.6y",
        "basic.9y",
        "high.school",
        "professional.course",
        "university.degree",
        "unknown",
    ]

    month_ordering = [
        "jan",
        "feb",
        "mar",
        "apr",
        "may",
        "jun",
        "jul",
        "aug",
        "sep",
        "oct",
        "nov",
        "dec",
    ]

    day_ordering = ["mon", "tue", "wed", "thu", "fri"]

    for feature, name in categorical_features.items():
        counts_y = train_df.groupby(by=[feature, target])[target].count()
        percent_purchased = (
            pd.DataFrame(counts_y)
            .rename(columns={"y": "count"})
            .reset_index()
            .pivot(index=feature, columns="y", values="count")
            .reset_index()
        )

        percent_purchased = percent_purchased.assign(
            percent_purchased=round(
                100
                * (
                    percent_purchased["yes"]
                    / (percent_purchased["no"] + percent_purchased["yes"])
                ),
                2,
            )
        )
        if feature == "education":
            plot = (
                alt.Chart(percent_purchased)
                .mark_bar()
                .encode(
                    x=alt.X(
                        "percent_purchased:Q",
                        scale=alt.Scale(domain=(0, 100)),
                        title="Percent Purchased",
                    ),
                    y=alt.Y(feature, sort=education_ordering, title=name),
                )
            )
        else:
            percent_purchased.sort_values("percent_purchased", inplace=True)
            plot = (
                alt.Chart(percent_purchased)
                .mark_bar()
                .encode(
                    x=alt.X(
                        "percent_purchased:Q",
                        scale=alt.Scale(domain=(0, 100)),
                        title="Percent Purchased",
                    ),
                    y=alt.Y(feature, sort="x", title=name),
                )
            )
        path = str(opt["--image_path"]) + feature + ".svg"
        plot.save(path, scale_factor=3.0)


if __name__ == "__main__":
    main(opt)
