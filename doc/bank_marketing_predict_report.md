Predicting deposit product subscription from contacted customers
================
Asma Al-Odaini, Chuck Ho, Justin Fu, Junting He
2020/11/26 (updated: 2020-11-27)

  - [Summary](#summary)
  - [Introduction](#introduction)
  - [Methods](#methods)
      - [Data](#data)
      - [Analysis](#analysis)
  - [Results & Discussion](#results-discussion)
  - [References](#references)

# Summary

For this project we are trying to answer the question: given detail
records of the this telemarketing campaign outreach, will the contacted
customers subscribe to the promoting term deposit product. To answer the
predictive question posed above, we plan to build a predictive
classification model using logistic regression algorithm. Our final
classification model have a reasonable performance on test data, with a
final f1-score of 0.6 and an overall accuracy of 0.86. However, this
model is not good enough to be used in the industry since the accuray
and f-1 score still have a lot to improve. To further improve the
classification performance on the model, we will look more closely into
the features and do some feature engineering.

# Introduction

Telemarketing is a method of selling products or services to potential
customers using the telephone or the Internet and it is commonly used in
banks for promotion. Detecting and focusing more on the target customers
could have great impact on save time and money (Barich and Kotler 1991).

In this project, we ask if a machine learning algorithm can be used to
predict whether a newly contacted customer subscribe to the promoting
term deposit product. Answering this question is important to bank as
they can better estimate potential subscription for the pool of
remaining targets, or even for next similar campaign (Moro, Cortez, and
Rita 2014). Furthermore, we would also want to identify the key
attributes of customers (e.g demographics) and the nature of the call
(e.g. the month, day of the week, contact method) to help the
tele-marketing team to prioritize resources in calling the higher
potential customers and adjusting time and medium for the calling.
Therefore, if we can use a machine learning algorithm to make an
accurate and effective prediction on whether a newly contacted customer
subscribe to the promoting term deposit product, this could may be
beneficial to building longer and tighter relations and enhancing
business demand (Rust, Moorman, and Bhalla 2010).

By doing exploratory analysis, we noticed that some of the predictors
might be useful to predict the subscription target. For the categorical
features, some features are similar in the proportion subscribed, while
others seem to be promising in predicting the positive class. The
poutcome (previous outcome) feature seems to be a good candidate as
previous success is highly correlated with the positive class.

<div class="figure">

<img src="../results/poutcome.svg" alt="Figure 1.Distribution of previous outcome features in the training set for subscribers to the bank's term deposit product." width="50%" />

<p class="caption">

Figure 1.Distribution of previous outcome features in the training set
for subscribers to the bank’s term deposit product.

</p>

</div>

In addition, the features values month is also of great possibility to
be correlated with the target.

<div class="figure">

<img src="../results/month.svg" alt="Figure 2.Distribution of month features in the training set for subscribers to the bank's term deposit product." width="30%" />

<p class="caption">

Figure 2.Distribution of month features in the training set for
subscribers to the bank’s term deposit product.

</p>

</div>

For numeric features, we plotted the distributions of each predictor
from the training data set and coloured the distribution by different
class (didn’t subscribe: blue and subscribed: orange). Although the
distributions for all of these numeric features overlap to a certain
degree, they do show a difference in their centers and spreads, for
example, employment variation rate, last contact duration, euribor 3
month rate, and consumer price index.

<div class="figure">

<img src="../results/numeric.png" alt="Figure 3.Distribution of numeric features in the training set for subscribers and non-subscribers to the bank's term deposit product." width="70%" />

<p class="caption">

Figure 3.Distribution of numeric features in the training set for
subscribers and non-subscribers to the bank’s term deposit product.

</p>

</div>

# Methods

## Data

The data set used in this project is from from a marketing campaign of a
Portuguese bank and created by S. Moro, P. Cortez and P. Rita (Moro,
Cortez, and Rita 2014). It was sourced from the UCI Machine Learning
Repository (@ Dua and Graff 2017) and can be found
[here](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing). Each row
in the data set represents summary statistics with detail information of
the contacted client, including bank client info (e.g. age, job, loan
experience, etc.), other campaign attributes (e.g. number of contact,
previous campaign outcome, etc) and social and economic attributes.
(e.g. consumer confidence index, euribor rate, etc.) They were using
telemarketing to attempt to get customer to sign up for the bank’s term
deposit product. The target in this dataset is yes or no to subscribing
to the term deposit product.

## Analysis

The logistic regression (lr) algorithm was used to build a
classification model to predict whether whether a newly contacted
customer subscribe to the promoting term deposit product (found in the y
column of the data set). We used all variables in the original data set
to fit the model and carried out cross-validation to choose the
hyperparameter C and max\_iter with f1-score as the scoring metric. The
R and Python programming languages (R Core Team 2020; Van Rossum and
Drake 2009) and the following R and Python packages were used to perform
the analysis: knitr (Xie 2020), matplotlib (Hunter 2007), seaborn
(Waskom et al. 2017), numpy(Oliphant, n.d.), os(Van Rossum and Drake
2009), warnings(McKinney 2019), Pandas (McKinney 2010),
altair(VanderPlas et al. 2018), docopt (Keleshev 2014) The code used to
perform the analysis and create this report can be found here:
<https://github.com/UBC-MDS/DSCI_522_Group_10>

# Results & Discussion

Since our predictors are mixed with different types of variables and
contains unknown values (e.g. education), firstly, we performed simple
imputation and other pre-processing such as One Hot Encoder for ordinal
and categorical predictors, and standard scaler for numeric predictors
before model fitting and hyperparameter optimization. Then, we decided
to base our decisions on the f1\_score with a bias towards recall over
precision and performed randomizedsearchcv to choose a model. Based on
the cross-validation f1\_score, we got into the conclusion that we chose
a simple classification model using the logistic regression
(balanced）algorithm. Additionally, this algorithm is very
interpretable and easier to communicate to higher level management, and
balanced class weight can help to deal with our class imbalance.

<!--html_preserve-->

<table border="1" class="dataframe">

<thead>

<tr style="text-align: center;">

<th>

</th>

<th>

Accuracy

</th>

<th>

Recall

</th>

<th>

Precision

</th>

<th>

f1

</th>

<th>

AP

</th>

<th>

Roc\_Auc

</th>

</tr>

</thead>

<tbody>

<tr>

<th>

Dummy

</th>

<td>

0.888

</td>

<td>

0.000

</td>

<td>

0.000

</td>

<td>

0.000

</td>

<td>

0.112

</td>

<td>

0.500

</td>

</tr>

<tr>

<th>

Decision Tree

</th>

<td>

0.888

</td>

<td>

0.522

</td>

<td>

0.501

</td>

<td>

0.511

</td>

<td>

0.315

</td>

<td>

0.728

</td>

</tr>

<tr>

<th>

Naive Bayes

</th>

<td>

0.810

</td>

<td>

0.654

</td>

<td>

0.327

</td>

<td>

0.436

</td>

<td>

0.387

</td>

<td>

0.828

</td>

</tr>

<tr>

<th>

RBF SVM

</th>

<td>

0.908

</td>

<td>

0.364

</td>

<td>

0.670

</td>

<td>

0.471

</td>

<td>

0.613

</td>

<td>

0.920

</td>

</tr>

<tr>

<th>

Logistic Regression

</th>

<td>

0.911

</td>

<td>

0.421

</td>

<td>

0.659

</td>

<td>

0.514

</td>

<td>

0.591

</td>

<td>

0.933

</td>

</tr>

<tr>

<th>

Logistic Regression (balanced)

</th>

<td>

0.859

</td>

<td>

0.879

</td>

<td>

0.437

</td>

<td>

0.584

</td>

<td>

0.583

</td>

<td>

0.936

</td>

</tr>

<tr>

<th>

Ridge Classifier

</th>

<td>

0.906

</td>

<td>

0.321

</td>

<td>

0.672

</td>

<td>

0.435

</td>

<td>

0.594

</td>

<td>

0.932

</td>

</tr>

<tr>

<th>

Ridge Classifier(balanced)

</th>

<td>

0.863

</td>

<td>

0.841

</td>

<td>

0.443

</td>

<td>

0.580

</td>

<td>

0.577

</td>

<td>

0.930

</td>

</tr>

<tr>

<th>

Random Forest

</th>

<td>

0.912

</td>

<td>

0.453

</td>

<td>

0.655

</td>

<td>

0.535

</td>

<td>

0.634

</td>

<td>

0.940

</td>

</tr>

<tr>

<th>

Random Forest (balanced)

</th>

<td>

0.911

</td>

<td>

0.405

</td>

<td>

0.668

</td>

<td>

0.504

</td>

<td>

0.629

</td>

<td>

0.941

</td>

</tr>

</tbody>

</table>

<!--/html_preserve-->

Figure 4.Scoring on different models.

After that, we did the hyperparameter optimization on C and max\_iter by
carrying out 5-fold cross-validation and using f1-score as our metric of
model prediction performance. We observed that the optimal combination
of C C and max\_iter were 1 and 1600 respectively..

<!--html_preserve-->

<table border="1" class="dataframe">

<thead>

<tr style="text-align: center;">

<th>

rank

</th>

<th>

f1

</th>

<th>

param\_logisticregression\_\_C

</th>

<th>

param\_logisticregression\_\_max\_iter

</th>

</tr>

</thead>

<tbody>

<tr>

<td>

1

</td>

<td>

0.583735

</td>

<td>

1

</td>

<td>

200

</td>

</tr>

<tr>

<td>

2

</td>

<td>

0.583629

</td>

<td>

1

</td>

<td>

1800

</td>

</tr>

<tr>

<td>

2

</td>

<td>

0.583629

</td>

<td>

1

</td>

<td>

800

</td>

</tr>

<tr>

<td>

2

</td>

<td>

0.583629

</td>

<td>

1

</td>

<td>

400

</td>

</tr>

<tr>

<td>

2

</td>

<td>

0.583629

</td>

<td>

1

</td>

<td>

600

</td>

</tr>

</tbody>

</table>

<!--/html_preserve-->

Figure 5. Results of hyperparameter optimization.

Our classification model have a reasonable performance on test data,
with a final f1-score of 0.6 and an overall accuracy of 0.86. Since we
would rather to make the mistakes on classifying the costumers who is
actually not going to subscribe into “yes” class than make the mistakes
on predicting the costumers who is actually going to subscribe into “no”
class, it is acceptable the precision is relatively low with the purpose
to maximize recall. However, we think that this model is not good enough
to be used in the industry since the accuray and f-1 score still have a
lot to improve.

<div class="figure">

<img src="../results/confusion_matrix.svg" alt="Figure 6. Comfusion matrix on the test result." width="60%" />

<p class="caption">

Figure 6. Comfusion matrix on the test result.

</p>

</div>

To make a better intepretation on the model, we can look at the the top
10 features identified by model. They seem to overlap with the ones
identified in explanatory data analysis, which means the model training
is follow our expectation and on the right track.

<div class="figure">

<img src="../results/top10_predictors_disregard_direction.svg" alt="Figure 7. Top 10 features identified by model." width="60%" />

<p class="caption">

Figure 7. Top 10 features identified by model.

</p>

</div>

To further improve the classification performance on the model, instead
of focusing on choosing the best model, we will look more closely into
the features and do some feature engineering. For example, we can use
the recursive feature elimination (RFE) to eliminate unimportant
features or we can use search and score to select the only the important
features.

# References

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
