# Post Campaign Bank Term Deposit Subscription Predictor
- author: DSCI 522 Group10
- Group Member: Justin Fu, Junting He, Chuck Ho, Asma Odaini

A data analysis project for DSCI 522 (Data Science workflows); a course in the Master of Data Science program at the University of British Columbia.

## Introduction

For this project we are trying to answer the question: given detail records of the this tele-marketing campaign outreach, will the contacted customers subscribe to the promoting term deposit product. Answering this question is important to bank as they can better estimate potential subscription for the pool of remaining targets, or even for next similar campaign. Furthermore, we would also want to identify the key attributes of customers (e.g demographics) and the nature of the call (e.g. the month, day of the week, contact method) to help the telem-marketing team to prioritize resources in calling the higher potential customers and adjusting time and medium for the calling. For example, it is found that subscription seems more associated with students in our EDA.  


The data set used in this project is created by S. Moro, P. Cortez and P. Rita. It was sourced from the UCI Machine Learning Repository and can be found [here](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing).   Each row in the data set represents summary statistics with detail information of the contacted client, including bank client info (e.g. age, job, loan experience, etc.), other campaign attributes (e.g. number of contact, previous campaign outcome, etc) and social and economic attributes. (e.g. consumer confidence index, euribor rate, etc.)


To answer the predictive question posed above, we plan to build a predictive classification model. Before building our model we will partition the data into a training and test set (split 80%:20%) and perform exploratory data analysis to assess whether there is a strong class imbalance problem that we might need to address, as well as explore whether there are any predictors whose distribution looks very similar between the two classes. It also helps determine whether certain predictors should be omitted from the modelling. The class counts will be presented as a table and used to inform whether we think there is a class imbalance problem and list out potential approach if class imblance is identified. The numeric predictor distributions across classes will be plotted as histograms where the distribution will be coloured by class. The categorical predictors distribution the percentage of subscription(positive class) will be plotted by bar plot.

While we are keen in predicting one of two classes and learning the importance of the predictors, one suitable and simple approach that we plan to first explore is using a Logistic Regression classification algorithm. Since our predictors are mixed with different types of variables and contains unknown values (e.g. education), we will perform imputation and pre-processing such as One Hot Encoder for categorical predictors and standard scaler for numeric predictors before model fitting and hyperparameter optimization. We will try hyperparameter optimization on C and carry out cross-validation using ~ 100 folds because the train data set is very large, having 32950 observations. We will use overall accuracy, f1, recall and precision scores to choose C. A table of these metrics for C will be included as part of the final report for this project. 

After selecting our final model, we will re-fit the model on the entire training data set, and then evaluate its performance on the test data set. At this point we will look at overall accuracy as well as misclassification errors (from the confusion matrix) to assess prediction performance. These values will be reported as a confusion matrix plot and table in the final report. The importance of the predictors will also be reported as a table in the final report.

Thus far we have performed some exploratory data analysis, and the report for that can be found [here](src/bank_marketing_data_eda.ipynb).

## Usage

To replicate the analysis, clone this GitHub repository, install the [dependencies](#dependencies) listed below, and run the following commands at the command line/terminal from the root directory of this project:

```
python src/download_and_extract_zip.py --url=https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip --out_file="data/raw/"

python src/bank_marketing_data_eda.ipynb
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
  - altair_saver
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
  
  
We are providing you with a `conda` environment file which is available [here](env-bank_marketing.yaml). You can download this file and create a conda environment for this project and activate it as follows. 

```
conda env create -f env-bank_marketing.yaml
conda activate bank
```
  
## License
The Post Campaign Bank Term Deposit Subscription Predictor materials here are licensed under the Creative Commons Attribution 2.5 Canada License (CC BY 2.5 CA). If re-using/re-mixing please provide attribution and link to this webpage.

## References

[Moro et al., 2014] S. Moro, P. Cortez and P. Rita.  2014. “UCI Machine Learning Repository.” University of California, Irvine, School of Information; Computer Sciences. http://archive.ics.uci.edu/ml.
