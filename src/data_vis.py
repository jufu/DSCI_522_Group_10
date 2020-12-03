"""author: Asma Al-Odaini
date: 2020-11-25
This script imports the training data file and generates visualizations to explore the features.
You can choose from the following features:  ["job", "marital_status", "default", "housing", "previous_outcome", "contact": "education", "day_of_week", "month", "last_contact_duration", "contacts_during_campaign", "days_after_previous_contact", "previous_contacts", "employment_variation_rate", "consumer_price_index", "consumer_confidence_index", "euribor_3_month_rate", "number_of_employees"]. If no feature was selected, all features will be visualized. 

Usage: data_vis.py --data_path=<data_path> --image_path=<image_path> [--feature=<feature>] 

Options:
--data_path=<data_path>      Takes a path of the training data csv 
--image_path=<image_path>     Takes a path for images 
[--feature=<feature>]     Takes a feature to generate a plot for


Example:
    python src/data_vis.py --data_path="data/processed/bank-additional-full_train.csv" --image_path="results/" --feature="job"

"""

import pandas as pd
import altair as alt
from docopt import docopt
from altair_saver import save


# alt.renderers.enable("mimetype")
# alt.data_transformers.enable("data_server")

opt = docopt(__doc__)


alt.data_transformers.disable_max_rows()


def main(opt):
    """This function reads user inputs and calls the appropriate
    plotting function depending on the type of feature specified
    in the --feature option (categorical or numeric) or
    creates all plots if no feature is specified.

    Parameters
    ----------
    opt : Dict
        docopt dictionary that contains user input

    Example
    ----------
    main(opt)
    """

    train_df = pd.read_csv(opt["--data_path"])
    train_df.loc[train_df.target == 0, "target"] = "Not Purchased"
    train_df.loc[train_df.target == 1, "target"] = "Purchased"

    feature_option = opt["--feature"]

    categorical_features = [
        "job",
        "marital_status",
        "default",
        "housing",
        "loan",
        "previous_outcome",
        "contact",
        "education",
        "day_of_week",
        "month",
    ]

    numeric_features = [
        "age",
        "last_contact_duration",
        "contacts_during_campaign",
        "days_after_previous_contact",
        "previous_contacts",
        "employment_variation_rate",
        "consumer_price_index",
        "consumer_confidence_index",
        "euribor_3_month_rate",
        "number_of_employees",
    ]

    if feature_option in numeric_features:
        create_numeric_plots(train_df, [feature_option])
    elif feature_option in categorical_features:
        create_categorical_plots(train_df, [feature_option])
    elif feature_option is None:
        create_numeric_plots(train_df, numeric_features)
        create_categorical_plots(train_df, categorical_features)


def create_categorical_plots(train_df, categorical_features):
    """Creates percent purchased plots given a dataframe and a list of categorical feature/s
    and saves it in the specified folder.

    Parameters
    ----------
    train_df : DataFrame
        Dataframe containing the training split of the Bank Additional Full dataset
    categorical_features : list
        A list containing the categorical feature/s of which to create percent purchased plots

    Example
    ----------
    create_categorical_plots(dataframe, [categorical_feature])
    """
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

    for feature in categorical_features:
        name = feature.replace("_", " ").title()
        counts_y = train_df.groupby(by=[feature, "target"])["target"].count()
        percent_purchased = (
            pd.DataFrame(counts_y)
            .rename(columns={"target": "count"})
            .reset_index()
            .pivot(index=feature, columns="target", values="count")
            .reset_index()
        )

        percent_purchased = percent_purchased.assign(
            percent_purchased=round(
                100
                * (
                    percent_purchased["Purchased"]
                    / (
                        percent_purchased["Not Purchased"]
                        + percent_purchased["Purchased"]
                    )
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
                    axis=alt.Axis(grid=False),
                ),
                y=alt.Y(feature, sort=sort, title=name),
            )
        )
        path = str(opt["--image_path"]) + feature + ".png"
        plot.save(path)
        print(name + " plot was created and saved")

    print(
        "Visualizations for categorical variables have been created in the declared path"
    )


def create_numeric_plots(train_df, numeric_features):
    """Creates density plots given a dataframe and a list of numeric
    feature/s and saves it in the specified folder.

    Parameters
    ----------
    train_df : DataFrame
        Dataframe containing the training split of the Bank Additional Full dataset
    numeric_features : list
        A list containing the numeric feature/s of which to create density plots

    Example
    ----------
    create_numeric_plots(dataframe, [numeric_feature])
    """
    for feature in numeric_features:
        name = feature.replace("_", " ").title()
        plot = (
            alt.Chart(train_df)
            .transform_density(feature, groupby=["target"], as_=[feature, "density"])
            .mark_area(opacity=0.4)
            .encode(
                x=alt.X(feature, title=name, axis=alt.Axis(grid=False)),
                y=alt.Y("density:Q", title="Density", axis=alt.Axis(grid=False)),
                color=alt.Color("target", legend=alt.Legend(title="")),
            )
            .properties(height=100, width=100)
        )
        path = str(opt["--image_path"]) + feature + ".png"
        plot.save(path)
        print(name + " plot was created and saved")

    print("Visualizations for numeric variables have been created in the declared path")


if __name__ == "__main__":
    main(opt)
