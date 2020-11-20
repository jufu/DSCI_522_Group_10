# Post Campaign Bank Term Deposit Subscription Predictor
- author: DSCI 522 Group10
- Group Member: Justin Fu, Junting He, Chuck Ho, Asma Odaini

A data analysis project for DSCI 522 (Data Science workflows); a course in the Master of Data Science program at the University of British Columbia.

## Introduction

For this project we are trying to answer the question: given detail records of outreach targets of this tele-marketing campaign, will the contacted client subscribe to the promoting term deposit product. Answering this question is important to bank as they can better estimate and measure the effectiveness of the campaign for the pool of remaining targets or potential new clients. Furthermore, we would also want to identify the key attributes of clients who will be more susceptible to this type of campaign. We have hypothesis that loan experience may be a negative driver to the subscription while some social and economic indicator such as euribor rate may be a positive driver to subscription. Answers for these questions will provide guidance to the bank in order to prioritize resources in targeting the higher potential, avoid potential customer dissatisfaction and adjust campaign period based on social and economic indicators. 


The data set used in this project is created by S. Moro, P. Cortez and P. Rita. It was sourced from the UCI Machine Learning Repository and can be found [here](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing).   Each row in the data set represents summary statistics with detail information of the contacted client, including bank client info (e.g. age, job, loan experience, etc.), other campaign attributes (e.g. number of contact, previous campaign outcome, etc) and social and economic attributes. (e.g. consumer confidence index, euribor rate, etc.)


To answer the predictive question posed above, we plan to build a predictive classification model. Before building our model we will partition the data into a training and test set (split 80%:20%) and perform exploratory data analysis to assess whether there is a strong class imbalance problem that we might need to address, as well as explore whether there are any predictors whose distribution looks very similar between the two classes and having the initial validation on the hypothesis on correlation for above mentioned predictors. It also helps determine whether certain predictors should be omitted from the modelling. The class counts will be presented as a table and used to inform whether we think there is a class imbalance problem. The predictor distributions across classes will be plotted as histograms (by predictor) where the distribution will be coloured by class.

Before doing any encoding to our mixed type predictors, we discovered that the data contains quite some unknown entries for certain predictors (such as previous outcome, etc), we will need to do imputation or drop it from the model. Having done the imputation, we will need to conduct other pre-processing such as One Hot Encoder and Ordinal scaling for categorical and binary predictors and standard scaler for numeric predictors. While we are keen in predicting one of two classes and directional importance of the predictor, one suitable and simple approach that we plan to first explore is using a Logistic Regression classification algorithm. We will try hyperparameter optimization on $C$ and carry out cross-validation using ~ 100 folds because this data set is very large, having 45211 observations. We will use overall validation accuracy to choose C. A table of overall accuracy for $C$ will be included as part of the final report for this project. 

After selecting our final model, we will re-fit the model on the entire training data set, and then evaluate its performance on the test data set. At this point we will look at overall accuracy as well as misclassification errors (from the confusion matrix) to assess prediction performance. These values will be reported as a table in the final report.

Thus far we have performed some exploratory data analysis, and the report for that can be found [here](src/breast_cancer_eda.md).

## Usage

To replicate the analysis, clone this GitHub repository, install the [dependencies](#dependencies) listed below, and run the following commands at the command line/terminal from the root directory of this project:

```
python src/download_and_extract_zip.py --url=https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip --out_file="data/raw/"

Rscript -e "rmarkdown::render('src/xxx_eda.Rmd')"
```

## Dependencies
channels:
  - conda-forge
  - defaults
dependencies:
  - ipykernel
  - matplotlib>=3.2.2
  - scikit-learn>=0.23.2
  - pandas>=1.1.3
  - requests>=2.24.0
  - graphviz
  - python-graphviz
  - altair>=4.1.0
  - jinja2
  - pip>=20
  - pandas-profiling>=1.4.3
  - pip:
    - psutil>=5.7.2
    - xgboost>=1.*
    - lightgbm>=3.*
    - git+git://github.com/mgelbart/plot-classifier.git
- Python 3.7.3 and Python packages:
  - docopt==0.6.2
  - requests==2.22.0
  - pandas==0.24.2
  - feather-format==0.4.0
- R version 3.6.1 and R packages:
  - knitr==1.26
  - feather==0.3.5
  - tidyverse==1.2.1
  - caret==6.0-84
  - ggridges==0.5.1
  - ggthemes==4.2.0
  
  
We are providing you with a `conda` environment file which is available [here](env-bank_marketing.yaml). You can download this file and create a conda environment for the course and activate it as follows. 

```
conda env create -f env-bank_marketing.yaml
conda activate bank
```
  
## License
The Post Campaign Bank Term Deposit Subscription Predictor materials here are licensed under the Creative Commons Attribution 2.5 Canada License (CC BY 2.5 CA). If re-using/re-mixing please provide attribution and link to this webpage.

# References

[Moro et al., 2014] S. Moro, P. Cortez and P. Rita.  2014. “UCI Machine Learning Repository.” University of California, Irvine, School of Information; Computer Sciences. http://archive.ics.uci.edu/ml.
