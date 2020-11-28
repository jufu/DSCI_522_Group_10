# Post Campaign Bank Term Deposit Subscription Predictor
- author: DSCI 522 Group10
- Group Member: Justin Fu, Junting He, Chuck Ho, Asma Al-Odaini

A data analysis project for DSCI 522 (Data Science workflows); a course in the Master of Data Science program at the University of British Columbia.

## Introduction

For this project we are trying to answer the question: given detail records of the this tele-marketing campaign outreach, will the contacted customers subscribe to the promoting term deposit product. Answering this question is important to bank as they can better estimate potential subscription for the pool of remaining targets, or even for next similar campaign. Furthermore, we would also want to identify the key attributes of customers (e.g demographics) and the nature of the call (e.g. the month, day of the week, contact method) to help the telem-marketing team to prioritize resources in calling the higher potential customers and adjusting time and medium for the calling. For example, it is found that subscription seems more associated with students in our EDA.  


The data set used in this project is created by S. Moro, P. Cortez and P. Rita. It was sourced from the UCI Machine Learning Repository and can be found [here](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing).   Each row in the data set represents summary statistics with detail information of the contacted client, including bank client info (e.g. age, job, loan experience, etc.), other campaign attributes (e.g. number of contact, previous campaign outcome, etc) and social and economic attributes. (e.g. consumer confidence index, euribor rate, etc.)


To answer the predictive question posed above, we plan to build a predictive classification model. Before building our model we will partition the data into a training and test set (split 80%:20%) and perform exploratory data analysis to assess whether there is a strong class imbalance problem that we might need to address, as well as explore whether there are any predictors whose distribution looks very similar between the two classes. It also helps determine whether certain predictors should be omitted from the modelling. The class counts will be presented as a table and used to inform whether we think there is a class imbalance problem and list out potential approach if class imblance is identified. The numeric predictor distributions across classes will be plotted as histograms where the distribution will be coloured by class. The categorical predictors distribution the percentage of subscription(positive class) will be plotted by bar plot.

While we are keen in predicting one of two classes and learning the importance of the predictors, one suitable and simple approach that we plan to first explore is using a Logistic Regression classification algorithm. Since our predictors are mixed with different types of variables and contains unknown values (e.g. education), we are considering to perform imputation and other pre-processing such as One Hot Encoder for categorical predictors and standard scaler for numeric predictors before model fitting and hyperparameter optimization. We will try hyperparameter optimization on C and carry out cross-validation using ~ 100 folds because the train data set is very large, having 32950 observations. We will use overall accuracy, f1, recall and precision scores to choose C. A table of these metrics for C will be included as part of the final report for this project. 

After selecting our final model, we will re-fit the model on the entire training data set, and then evaluate its performance on the test data set. At this point we will look at overall accuracy as well as misclassification errors (from the confusion matrix) to assess prediction performance. These values will be reported as a confusion matrix plot and table in the final report. The importance of the predictors will also be reported as a table in the final report.

Thus far we have performed some exploratory data analysis, and the report for that can be found [here](http://htmlpreview.github.io/?https://raw.githubusercontent.com/UBC-MDS/DSCI_522_Group_10/main/src/bank_marketing_data_eda.html).

## Usage

To replicate the analysis, clone this GitHub repository, install the [dependencies](#dependencies) listed below, and run the following commands at the command line/terminal from the root directory of this project:

```
#activate our conda environment
conda activate bank

# download data and unzip data folder 
python src/download_and_extract_zip.py --url=https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip --out_file="data/raw/"

# split data 
python src/raw_to_split.py --in_file="data/raw/bank-additional-full.csv" --out_file="data"

# create exploratory data analysis figure for numeric features and write to file 
Rscript src/data_vis_continous.R --data_path='data/processed/bank-additional-full_train.csv' --image_path='results/'  

# create exploratory data analysis figure for categorical features and write to file 
python src/data_vis.py --data_path="data/processed/bank-additional-full_train.csv" --image_path="results/"

# Render the EDA report
jupyter nbconvert src/bank_marketing_data_eda.ipynb --no-input --to html

# create, train, and test model
python src/machine_learning_analysis.py --in_train="data/processed/bank-additional-full_train.csv" --in_test="../data/processed/bank-additional-full_test.csv" --out_path="results/"

# render final report
Rscript -e "rmarkdown::render('doc/bank_marketing_predict_report.Rmd', output_format = 'github_document')" 
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
  - docopt
  - jinja2  
  - pip>=20
  - pandas-profiling>=1.4.3  
  - seaborn
  - pip:
    - psutil>=5.7.2
    - xgboost>=1.*
    - lightgbm>=3.*
    - git+git://github.com/mgelbart/plot-classifier.git
    

    
We are providing you with a `conda` environment file which is available [here](env-bank_marketing.yaml). You can download this file and create a conda environment for this project and activate it as follows. 

```
conda env create -f env-bank_marketing.yaml
conda activate bank
```
  
## License
The Post Campaign Bank Term Deposit Subscription Predictor materials here are licensed under the Creative Commons Attribution 2.5 Canada License (CC BY 2.5 CA). If re-using/re-mixing please provide attribution and link to this webpage.

## References

[Moro et al., 2014] S. Moro, P. Cortez and P. Rita.  2014. “UCI Machine Learning Repository.” University of California, Irvine, School of Information; Computer Sciences. http://archive.ics.uci.edu/ml.

<div id="refs" class="references hanging-indent">

<div id="ref-barich1991framework">

Barich, Howard, and Philip Kotler. 1991. “A Framework for Marketing
Image Management.” *MIT Sloan Management Review* 32 (2): 94.

</div>

<div id="ref-Dua2019">

Dua, Dheeru, and Casey Graff. 2017. “UCI Machine Learning Repository.”
University of California, Irvine, School of Information; Computer
Sciences. <http://archive.ics.uci.edu/ml>.

</div>

<div id="ref-Hunter">

Hunter, J. D. 2007. “Matplotlib: A 2D Graphics Environment.” *Computing
in Science & Engineering* 9 (3): 90–95.
<https://doi.org/10.1109/MCSE.2007.55>.

</div>

<div id="ref-docoptpython">

Keleshev, Vladimir. 2014. *Docopt: Command-Line Interface Description
Language*. <https://github.com/docopt/docopt>.

</div>

<div id="ref-mckinney-proc-scipy-2010">

McKinney. 2010. “Data Structures for Statistical Computing in Python.”
In *Proceedings of the 9th Python in Science Conference*, edited by
Stéfan van der Walt and Jarrod Millman, 56–61.
[https://doi.org/ 10.25080/Majora-92bf1922-00a](https://doi.org/%2010.25080/Majora-92bf1922-00a%20).

</div>

<div id="ref-moro2014data">

Moro, Sérgio, Paulo Cortez, and Paulo Rita. 2014. “A Data-Driven
Approach to Predict the Success of Bank Telemarketing.” *Decision
Support Systems* 62: 22–31.

</div>

<div id="ref-numpy">

Oliphant, Travis. n.d. “NumPy: A Guide to NumPy.” USA: Trelgol
Publishing. <http://www.numpy.org/>.

</div>

<div id="ref-R">

R Core Team. 2020. *R: A Language and Environment for Statistical
Computing*. Vienna, Austria: R Foundation for Statistical Computing.
<https://www.R-project.org/>.

</div>

<div id="ref-rust2010spotlight">

Rust, Roland T, Christine Moorman, and Gaurav Bhalla. 2010. “Spotlight
on Reinvention: Rethinking Marketing.” *Harvard Business Review* 88 (1):
2–8.

</div>

<div id="ref-Altair2018">

VanderPlas, Jacob, Brian Granger, Jeffrey Heer, Dominik Moritz, Kanit
Wongsuphasawat, Arvind Satyanarayan, Eitan Lees, Ilia Timofeev, Ben
Welsh, and Scott Sievert. 2018. “Altair: Interactive Statistical
Visualizations for Python.” *Journal of Open Source Software*, December.
<https://doi.org/10.21105/joss.01057>.

</div>

<div id="ref-Python">

Van Rossum, Guido, and Fred L. Drake. 2009. *Python 3 Reference Manual*.
Scotts Valley, CA: CreateSpace.

</div>

<div id="ref-michael_waskom_2017_883859">

Waskom, Michael, Olga Botvinnik, Drew O’Kane, Paul Hobson, Saulius
Lukauskas, David C Gemperline, Tom Augspurger, et al. 2017.
*Mwaskom/Seaborn: V0.8.1 (September 2017)* (version v0.8.1). Zenodo.
<https://doi.org/10.5281/zenodo.883859>.

</div>

<div id="ref-knitr">

Xie, Yihui. 2020. *Knitr: A General-Purpose Package for Dynamic Report
Generation in R*. <https://yihui.org/knitr/>.

</div>

</div>
