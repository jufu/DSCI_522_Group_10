"""author: Asma Al-Odaini
date: 2020-11-25
This script imports the training data file and generates visualizations to explore the features.
Usage: data_vis.py --data_path=<data_path> --image_path=<image_path>

Options:
--data_path=<data_path>      Takes a path of the training data csv 
--image_path=<image_path>     Takes a path for images 

Example:
    python src/data_vis.py --data_path="data/processed/bank-additional-full_train.csv" --image_path="results/"

"""

import pandas as pd
import altair as alt
from docopt import docopt
from altair_saver import save

# from selenium import webdriver

# driver = webdriver.Chrome("chromedriver_linux64/chromedriver")

alt.renderers.enable("mimetype")
alt.data_transformers.enable("data_server")

opt = docopt(__doc__)


alt.data_transformers.disable_max_rows()


def main(opt):

    train_df = pd.read_csv(opt["--data_path"])
    train_df.loc[train_df.y == 0, "y"] = "no"
    train_df.loc[train_df.y == 1, "y"] = "yes"

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

    ordinal_features = {
        "education": education_ordering,
        "day_of_week": day_ordering,
        "month": month_ordering,
    }

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
        if feature in ordinal_features.keys():
            sort = ordinal_features[feature]
        else:
            sort = "x"

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
                y=alt.Y(feature, sort=sort, title=name),
            )
        )
        path = str(opt["--image_path"]) + feature + ".svg"
        plot.save(path, scale_factor=3.0)

    numeric_features = {
        "age": "Age",
        "duration": "Last Contact Duration",
        "campaign": "Number of Contacts During Campaign",
        "pdays": "Days After Previous Contact",
        "previous": "Number of Previous Contacts",
        "emp.var.rate": "Employment Variation Rate",
        "cons.price.idx": "Consumer Price Index",
        "cons.conf.idx": "Consumer Confidence Index",
        "euribor3m": "Euribor 3 Month Rate",
        "nr.employed": "Number of Employees",
    }

    for feature, name in numeric_features.items():
        plot = (
            alt.Chart(train_df)
            .transform_density(feature, groupby=["y"], as_=[feature, "density"])
            .mark_area(opacity=0.4)
            .encode(x=alt.X(feature, title=name), y="density:Q", color="y")
            .properties(height=100, width=100)
        )
        path = str(opt["--image_path"]) + feature + ".svg"

        # save(plot, path, method="selenium", webdriver=driver)

    print(
        "Visualizations for categorical variables have been creates in the declared path"
    )


if __name__ == "__main__":
    main(opt)
