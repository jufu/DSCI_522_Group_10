"""
author: Chuck Ho
date: 2020-11-27

This script imports the train and test csv from the proccessed data folder  and performing machine learning modelling and alaysis.

Usage: machine_learning_analysis.py --in_train=<in_train> --in_test=<in_test> --out_path=<out_path> 
 
Options:
--in_train=<in_train>         path including filename of the input train data file to process (this is a required option)
--in_test=<in_test>           path including filename of the input test data file to process (this is a required option)
--out_path=<out_path>         path to where the figures and tables will be written to (this is a required option)

Example:
    python machine_learning_analysis.py --in_train="data/processed/bank-additional-full_train.csv" --in_test="data/processed/bank-additional-full_test.csv" --out_path="results/"
    

"""

#Standards
import os
import numpy as np
import pandas as pd
import string
from collections import deque
from docopt import docopt


#Plots
import matplotlib.pyplot as plt
import altair as alt
import seaborn as sns



# classifiers / models
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, RidgeCV, RidgeClassifier
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, SVR



# Data, preprocessing and pipeline

#Pro
from sklearn import datasets
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    PolynomialFeatures,
    StandardScaler,
)
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import (
    ColumnTransformer,
    TransformedTargetRegressor,
    make_column_transformer,
)


# metrics for class imbalance
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix, 
    plot_confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)


# hyperparameter optimization

from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    ShuffleSplit,
    cross_val_score,
    cross_validate,
    train_test_split,
)

# other
# ignore warning 
import warnings
warnings.filterwarnings('ignore')




opt = docopt(__doc__)


def main(in_train, in_test, out_path):
    """
    Take in the data, perform preprocessing to model fitting, generate analysis with figures and table and output to the given path.
    
    Parameters
    ----------
    in_train : string
        path including filename of the input train data file to process
    in_test : string
        path including filename of the input test data file to process    
    out_path : string
        path to where the processed data will be written to
    Example
    ----------
    main("../data/processed/bank-additional-full_train.csv", "../data/processed/bank-additional-full_test.csv", "../result")
    """

    # load in data (should be full data before split)   ##need update once clean data script is finalized.
    train_df = pd.read_csv(in_train, sep=',')
    
    test_df = pd.read_csv(in_test, sep=',')
    
    # Define types of features: numeric, categorical, ordinal for now. No drop features  ## need update on drop feature after data clean.

    
    numeric_features = ["age", "contacts_during_campaign", "days_after_previous_contact", "previous_contacts", "employment_variation_rate", 
                        "consumer_price_index", "consumer_confidence_index", "euribor_3_month_rate", "number_of_employees", "last_contact_duration"]
    categorical_features = ["job", "previous_outcome", "month", "day_of_week", "contact","marital_status", "default", "housing", "loan"]
    ordinal_features = ["education"]
    education_ordering = ['illiterate', 'basic.4y','basic.6y','basic.9y','high.school',
                'professional.course','university.degree', 'unknown']
    drop_features = []
    target = ["target"]
    
    
    # drop target for train and test data.
    X_train = train_df.drop(columns=target)
    y_train = train_df[target]
    X_test = test_df.drop(columns=target)
    y_test = test_df[target]
    
    
    # Define preprocessing transformers (preprocessors - column transformer)
    
    
    numeric_transformer = make_pipeline(       
        SimpleImputer(strategy="median"), 
        StandardScaler()   
        )  
    ordinal_transformer = make_pipeline(       
        SimpleImputer(strategy="most_frequent"),       
        OrdinalEncoder(categories=[education_ordering])   
        )   
    categorical_transformer = make_pipeline(       
        SimpleImputer(strategy="constant", fill_value="missing"),       
        OneHotEncoder(handle_unknown="ignore", sparse=False)  
        )
    
    preprocessor = make_column_transformer(
        ("drop", drop_features),
        (numeric_transformer, numeric_features),
        (ordinal_transformer, ordinal_features),
        (categorical_transformer, categorical_features)
        )

    
    # A function to store mean cross-validation validation values 
    def store_cross_val_results(model_name, scores, results_dict):
        results_dict[model_name] = {
            "Accuracy": "{:0.3f}".format(np.mean(scores["test_accuracy"])),
    #         "mean_fit_time (s)": "{:0.4f}".format(np.mean(scores["fit_time"])),   #since it's not critical to get the result within an hour or so, fit and score time would not matter much
    #         "mean_score_time (s)": "{:0.4f}".format(np.mean(scores["score_time"])),
            "Recall": "{:0.3f}".format(np.mean(scores["test_recall"])),
            "Precision": "{:0.3f}".format(np.mean(scores["test_precision"])),
            "f1": "{:0.3f}".format(np.mean(scores["test_f1"])),
            "AP": "{:0.3f}".format(np.mean(scores["test_average_precision"])),
            "Roc_Auc": "{:0.3f}".format(np.mean(scores["test_roc_auc"])),
        }
    
    
    # A summary dictionary to store the scores for different models.
    results_df = {}
    
    
    # Define model metrics, fit and score the baseline model: Dummy Classifier
    scoring=["accuracy", "f1", "recall", "precision", "average_precision", "roc_auc"] 
    pipe = make_pipeline(preprocessor, DummyClassifier(strategy="most_frequent"))
    scores = cross_validate(pipe, X_train, y_train, return_train_score=True, scoring=scoring)
    summary = store_cross_val_results("Dummy", scores, results_df)
    pd.DataFrame(results_df)
    
    
    # Fit and score differnet model to see if Logistic Regression is the better classifier in this case.
    models = {
        "Decision Tree": DecisionTreeClassifier(),
        "Naive Bayes": GaussianNB(),
        "RBF SVM": SVC(),
        "Logistic Regression": LogisticRegression(),  
        "Logistic Regression (balanced)": LogisticRegression(class_weight="balanced"),
        "Ridge Classifier": RidgeClassifier(),
        "Ridge Classifier(balanced)": RidgeClassifier(class_weight="balanced"),
        "Random Forest": RandomForestClassifier(),
        "Random Forest (balanced)": RandomForestClassifier(class_weight="balanced")
    }
    
    for model_name, model in models.items():
        pipe = make_pipeline(preprocessor, model)
        scores = cross_validate(pipe, X_train, y_train, cv=5, scoring=scoring, n_jobs=-1)
        summary = store_cross_val_results(model_name, scores, results_df)
    model_selection = pd.DataFrame(results_df).T 
    model_selection.style.set_properties(**{'text-align': 'center'})


    # Generated the model summary to out_path folder (if not exist, create it)
    
    output_folder = out_path
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    

    model_selection.to_html(output_folder + "model_selection.html", justify = "center")
    
    
    
    # Selected the Logistic Regression model
    
    pipe_lr_balance = make_pipeline(preprocessor, LogisticRegression(class_weight="balanced"))
    
    
    # Hyperparameter optimization on C and max_iter with random search CV base on f1 score
    
    param_grid = {
        "logisticregression__C": 10.0 ** np.arange(-3, 3),
        "logisticregression__max_iter": np.arange(200,2000,200),
    }
    random_search = RandomizedSearchCV(pipe_lr_balance, param_distributions=param_grid, n_jobs=-1, n_iter=30, cv=5, scoring = "f1", verbose=1)
    random_search.fit(X_train, y_train)
    
    
    #  Generate top 5 hyperparameter combination table.
    
    hyper_opt_result = pd.DataFrame(random_search.cv_results_)[
        [
            "mean_test_score",
            "param_logisticregression__C",
            "param_logisticregression__max_iter",
            "rank_test_score",
        ]
    ].set_index("rank_test_score").sort_index().reset_index().head(5)
    

    hyper_opt_result.rename(columns= {'mean_test_score':'f1', 'rank_test_score':'rank'}, inplace=True)
    hyper_opt_result.style.set_properties(**{'text-align': 'center'}).set_table_styles([dict(selector='th', props=[('text-align', 'center')])])
    hyper_opt_result.to_html(output_folder + "hyperparameter_optimization_result.html", justify = "center", index=False)
    
    
    # Bulid pipeline for best model with optimized hyperparameters
    
    pipe_lr_best = random_search.best_estimator_
    pipe_lr_best.fit(X_train, y_train)
    
    
    # Plot confusion matrix and generated the figure in the folder
    cm = plot_confusion_matrix(pipe_lr_best, X_test, y_test, display_labels=["Not Subscribed", "Subscribed"], values_format="d", cmap=plt.cm.Blues)
    
    
    # path2 = "../data/confusion_matrix.png"
    plt.tight_layout()
    plt.savefig(output_folder + "confusion_matrix.svg", bbox_inches = "tight")
    

    
    # Generate classification report in the folder
    
    c_report = classification_report(y_test, pipe_lr_best.predict(X_test), target_names=["Not Subscribed", "Subscribed"], output_dict=True)
    report_df = pd.DataFrame(c_report)
    cr_plot = sns.heatmap(report_df.iloc[:-1, :].T, annot=True, cmap="Blues")
    
    #path3 = "../data/classification_report.png"
    plt.savefig(output_folder + "classification_report.svg")
    
    
    # Extract Predict name and coefficient from fitted pipeline
    
    weights = pipe_lr_best.named_steps['logisticregression'].coef_.flatten()
    ohe_c_features = list(pipe_lr_best.named_steps['columntransformer'].named_transformers_['pipeline-3'].named_steps['onehotencoder'].get_feature_names(categorical_features))
    transformed_columns = numeric_features + ordinal_features + ohe_c_features
    
    # Put them in pd dataframe
    
    data={'Predictors':transformed_columns, 'Coefficient':weights}
    feature_importance = pd.DataFrame(data)
    
    
    # Extract Predictor Importance and generate the top 10 table
    
    feature_importance["abs"] = abs(feature_importance["Coefficient"])
    feature_importance_top10 = feature_importance.sort_values(by="abs", ascending=False).nlargest(10,'abs')
    feature_importance_top10['Predictors'] = feature_importance_top10['Predictors'].replace({   
        "duration" : "Last Contact Duration",
        "emp.var.rate": "Employment Variation Rate",
        "cons.price.idx": "Consumer Price Index",
        "euribor3m": "Euribor 3 Month Rate",
        "poutcome_failure" : "Failed in Previous Contact",
        "month_mar" : "March",
        "month_may" : "May",
        "month_jun" : "June",
        "month_nov" : "Nov",
        "month_aug" : "Aug"
    })
    feature_importance_top10 = feature_importance_top10.round(2)
    feature_importance_top10.to_html(output_folder +  "top10_predictors_table.html", justify = "center", index=False)
    
    
    # Plotting the top 10 influential predictor (by absolute value of the coefficient) and save it to the folder
    plot = alt.Chart(feature_importance_top10).mark_bar().encode(alt.X("abs:Q", type='quantitative', scale=alt.Scale(domain=(0, 4)),title="Feature Coefficients"), alt.Y("Predictors", sort="-x", title="Features"))

        
    # path = "../data/top10_features.svg"
    plot.save(output_folder +  "top10_predictors_disregard_direction.svg")


if __name__ == "__main__":
    main(opt["--in_train"], opt["--in_test"], opt["--out_path"])
    
