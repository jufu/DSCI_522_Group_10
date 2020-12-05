Predicting deposit product subscription from contacted customers
================
Asma Al-Odaini, Chuck Ho, Justin Fu, Junting He
2020/11/26 (updated: 2020-12-04)

# Summary

With a banking institution’s telemarketing campaign data, we are
attempting to build a classification model to help us identify and
pursue customers that will subscribe to a term deposit product. We plan
to build a predictive classification model to help us predict customers
that will subscribe to the term deposit product.

Our final classification model has reasonable performance on the test
data, with a f1-score of 0.6 and a recall score of 0.90. On the 8238
test data cases, we correctly predicted 7091. However, it incorrectly
predicted 1147 cases with 1056 being false positives and 91 being false
negatives. We were more focused on reducing false negatives to capture
more revenue. The false positives are also important as it may require
more resources to reach out to potential customers, but our bias was
towards revenue (recall).

This model is not sufficient to be used in the industry since the f-1
score still has a lot to improve on. We recommend continuing to improve
the classification performance before we use this in production.

# Introduction

Telemarketing is a method of selling products or services to potential
customers using the telephone or the Internet and it is commonly used in
banks for promotion. Detecting and focusing more on the target customers
could have great impact on save time and money (Barich and Kotler 1991).

In this project, we investigate if a machine learning algorithm can be
used to predict whether a customer will potentially subscribe to the
term deposit product. Answering this question is important to bank as
they can better estimate potential subscription for the pool of
remaining targets, or even for next similar campaign (Moro, Cortez, and
Rita 2014). Furthermore, we would also want to identify the key
attributes of customers and the nature of the call (e.g. the month, day
of the week, contact method) to help the telemarketing team to
prioritize resources in calling the higher potential customers and
adjusting time and medium for the calls. Therefore, if we can use a
classification model to make an accurate and effective prediction, this
will be beneficial to expand the bank’s value chain to the customer and
enhance business demand (Rust, Moorman, and Bhalla 2010).

By doing exploratory analysis, we identified some of the features might
be more useful to predict the subscription target. For the categorical
features, a couple features looked promising to the model. The “previous
outcome” feature seems to be a good candidate as previous success is
highly correlated with those subscribing to the term deposit product.

<div class="figure" style="text-align: center">

<img src="../results/previous_outcome.png" alt="Figure 1.Distribution of previous outcome features in the training set for subscribers to the bank's term deposit product." width="50%" />
<p class="caption">
Figure 1.Distribution of previous outcome features in the training set
for subscribers to the bank’s term deposit product.
</p>

</div>

In addition, the “month” feature also shows potential. The time of year
may impact when term deposit products are more interesting for the
customers.

<div class="figure" style="text-align: center">

<img src="../results/month.png" alt="Figure 2.Distribution of month features in the training set for subscribers to the bank's term deposit product." width="30%" />
<p class="caption">
Figure 2.Distribution of month features in the training set for
subscribers to the bank’s term deposit product.
</p>

</div>

For numeric features, we plotted the distributions of each predictor
from the training data set and coloured the distribution by different
class (did not subscribe: blue and subscribed: orange). Although the
distributions for all of these numeric features overlap to a certain
degree, they also show a difference in their centers and spreads, for
example, employment variation rate, last contact duration, euribor 3
month rate, and consumer price index.

<div class="figure" style="text-align: center">

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
telemarketing to attempt to get customers to sign up for the bank’s term
deposit product. The target in this dataset is yes or no to subscribing
to the term deposit product.

## Analysis

The logistic regression (lr) algorithm was used to build the
classification model to predict whether whether a customer will
subscribe to the term deposit product (found in the y column of the data
set). We used all variables in the original data set to fit the model
and carried out cross-validation to choose the hyperparameter C and
max\_iter with f1-score as the scoring metric. The R and Python
programming languages (R Core Team 2020; Van Rossum and Drake 2009) and
the following R and Python packages were used to perform the analysis:
knitr (Xie 2020), matplotlib (Hunter 2007), seaborn (Waskom et al.
2017), numpy(Oliphant 2006–), os(Van Rossum and Drake 2009),
warnings(McKinney 2019), Pandas (McKinney 2010), altair(VanderPlas et
al. 2018), docopt (Keleshev 2014) The code used to perform the analysis
and create this report can be found here:
<https://github.com/UBC-MDS/DSCI_522_Group_10>

# Results & Discussion

With this first version of the classification model, we included all
features in the model. Future versions of this model will include
enhancements such as feature elimination to try and improve our results.
The model’s pipeline first performs simple imputation and other
pre-processing such as One Hot Encoder for ordinal and categorical
predictors, and standard scaler for numeric predictors before model
fitting and hyperparameter optimization.

Our decisions in choosing the model and also hyperparameter optimization
was based on the f1\_score with a bias towards recall over precision.
Based on the cross-validation f1\_score, our conclusion was to use the
logistic regression (balanced) algorithm. One advantage of this
algorithm is that it is very interpretable, allows us to understand
feature importance, and easier to communicate to higher level
management. We also included class weight balancing to help deal with
our class imbalance in this spotting a class problem.

<p align="center">
<iframe src="data:text/html;charset=utf-8, %3Ctable%20border=%221%22%20class=%22dataframe%22%3E%0A%20%20%3Cthead%3E%0A%20%20%20%20%3Ctr%20style=%22text-align:%20center;%22%3E%0A%20%20%20%20%20%20%3Cth%3E%3C/th%3E%0A%20%20%20%20%20%20%3Cth%3EAccuracy%3C/th%3E%0A%20%20%20%20%20%20%3Cth%3ERecall%3C/th%3E%0A%20%20%20%20%20%20%3Cth%3EPrecision%3C/th%3E%0A%20%20%20%20%20%20%3Cth%3Ef1%3C/th%3E%0A%20%20%20%20%20%20%3Cth%3EAP%3C/th%3E%0A%20%20%20%20%20%20%3Cth%3ERoc_Auc%3C/th%3E%0A%20%20%20%20%3C/tr%3E%0A%20%20%3C/thead%3E%0A%20%20%3Ctbody%3E%0A%20%20%20%20%3Ctr%3E%0A%20%20%20%20%20%20%3Cth%3EDummy%3C/th%3E%0A%20%20%20%20%20%20%3Ctd%3E0.888%3C/td%3E%0A%20%20%20%20%20%20%3Ctd%3E0.000%3C/td%3E%0A%20%20%20%20%20%20%3Ctd%3E0.000%3C/td%3E%0A%20%20%20%20%20%20%3Ctd%3E0.000%3C/td%3E%0A%20%20%20%20%20%20%3Ctd%3E0.112%3C/td%3E%0A%20%20%20%20%20%20%3Ctd%3E0.500%3C/td%3E%0A%20%20%20%20%3C/tr%3E%0A%20%20%20%20%3Ctr%3E%0A%20%20%20%20%20%20%3Cth%3EDecision%20Tree%3C/th%3E%0A%20%20%20%20%20%20%3Ctd%3E0.889%3C/td%3E%0A%20%20%20%20%20%20%3Ctd%3E0.528%3C/td%3E%0A%20%20%20%20%20%20%3Ctd%3E0.506%3C/td%3E%0A%20%20%20%20%20%20%3Ctd%3E0.517%3C/td%3E%0A%20%20%20%20%20%20%3Ctd%3E0.320%3C/td%3E%0A%20%20%20%20%20%20%3Ctd%3E0.731%3C/td%3E%0A%20%20%20%20%3C/tr%3E%0A%20%20%20%20%3Ctr%3E%0A%20%20%20%20%20%20%3Cth%3ENaive%20Bayes%3C/th%3E%0A%20%20%20%20%20%20%3Ctd%3E0.810%3C/td%3E%0A%20%20%20%20%20%20%3Ctd%3E0.654%3C/td%3E%0A%20%20%20%20%20%20%3Ctd%3E0.327%3C/td%3E%0A%20%20%20%20%20%20%3Ctd%3E0.436%3C/td%3E%0A%20%20%20%20%20%20%3Ctd%3E0.387%3C/td%3E%0A%20%20%20%20%20%20%3Ctd%3E0.828%3C/td%3E%0A%20%20%20%20%3C/tr%3E%0A%20%20%20%20%3Ctr%3E%0A%20%20%20%20%20%20%3Cth%3ERBF%20SVM%3C/th%3E%0A%20%20%20%20%20%20%3Ctd%3E0.908%3C/td%3E%0A%20%20%20%20%20%20%3Ctd%3E0.364%3C/td%3E%0A%20%20%20%20%20%20%3Ctd%3E0.670%3C/td%3E%0A%20%20%20%20%20%20%3Ctd%3E0.471%3C/td%3E%0A%20%20%20%20%20%20%3Ctd%3E0.613%3C/td%3E%0A%20%20%20%20%20%20%3Ctd%3E0.920%3C/td%3E%0A%20%20%20%20%3C/tr%3E%0A%20%20%20%20%3Ctr%3E%0A%20%20%20%20%20%20%3Cth%3ELogistic%20Regression%3C/th%3E%0A%20%20%20%20%20%20%3Ctd%3E0.911%3C/td%3E%0A%20%20%20%20%20%20%3Ctd%3E0.421%3C/td%3E%0A%20%20%20%20%20%20%3Ctd%3E0.659%3C/td%3E%0A%20%20%20%20%20%20%3Ctd%3E0.514%3C/td%3E%0A%20%20%20%20%20%20%3Ctd%3E0.591%3C/td%3E%0A%20%20%20%20%20%20%3Ctd%3E0.933%3C/td%3E%0A%20%20%20%20%3C/tr%3E%0A%20%20%20%20%3Ctr%3E%0A%20%20%20%20%20%20%3Cth%3ELogistic%20Regression%20(balanced)%3C/th%3E%0A%20%20%20%20%20%20%3Ctd%3E0.859%3C/td%3E%0A%20%20%20%20%20%20%3Ctd%3E0.879%3C/td%3E%0A%20%20%20%20%20%20%3Ctd%3E0.437%3C/td%3E%0A%20%20%20%20%20%20%3Ctd%3E0.584%3C/td%3E%0A%20%20%20%20%20%20%3Ctd%3E0.583%3C/td%3E%0A%20%20%20%20%20%20%3Ctd%3E0.936%3C/td%3E%0A%20%20%20%20%3C/tr%3E%0A%20%20%20%20%3Ctr%3E%0A%20%20%20%20%20%20%3Cth%3ERidge%20Classifier%3C/th%3E%0A%20%20%20%20%20%20%3Ctd%3E0.906%3C/td%3E%0A%20%20%20%20%20%20%3Ctd%3E0.321%3C/td%3E%0A%20%20%20%20%20%20%3Ctd%3E0.672%3C/td%3E%0A%20%20%20%20%20%20%3Ctd%3E0.435%3C/td%3E%0A%20%20%20%20%20%20%3Ctd%3E0.594%3C/td%3E%0A%20%20%20%20%20%20%3Ctd%3E0.932%3C/td%3E%0A%20%20%20%20%3C/tr%3E%0A%20%20%20%20%3Ctr%3E%0A%20%20%20%20%20%20%3Cth%3ERidge%20Classifier(balanced)%3C/th%3E%0A%20%20%20%20%20%20%3Ctd%3E0.863%3C/td%3E%0A%20%20%20%20%20%20%3Ctd%3E0.841%3C/td%3E%0A%20%20%20%20%20%20%3Ctd%3E0.443%3C/td%3E%0A%20%20%20%20%20%20%3Ctd%3E0.580%3C/td%3E%0A%20%20%20%20%20%20%3Ctd%3E0.577%3C/td%3E%0A%20%20%20%20%20%20%3Ctd%3E0.930%3C/td%3E%0A%20%20%20%20%3C/tr%3E%0A%20%20%20%20%3Ctr%3E%0A%20%20%20%20%20%20%3Cth%3ERandom%20Forest%3C/th%3E%0A%20%20%20%20%20%20%3Ctd%3E0.912%3C/td%3E%0A%20%20%20%20%20%20%3Ctd%3E0.463%3C/td%3E%0A%20%20%20%20%20%20%3Ctd%3E0.652%3C/td%3E%0A%20%20%20%20%20%20%3Ctd%3E0.541%3C/td%3E%0A%20%20%20%20%20%20%3Ctd%3E0.635%3C/td%3E%0A%20%20%20%20%20%20%3Ctd%3E0.940%3C/td%3E%0A%20%20%20%20%3C/tr%3E%0A%20%20%20%20%3Ctr%3E%0A%20%20%20%20%20%20%3Cth%3ERandom%20Forest%20(balanced)%3C/th%3E%0A%20%20%20%20%20%20%3Ctd%3E0.911%3C/td%3E%0A%20%20%20%20%20%20%3Ctd%3E0.410%3C/td%3E%0A%20%20%20%20%20%20%3Ctd%3E0.666%3C/td%3E%0A%20%20%20%20%20%20%3Ctd%3E0.507%3C/td%3E%0A%20%20%20%20%20%20%3Ctd%3E0.630%3C/td%3E%0A%20%20%20%20%20%20%3Ctd%3E0.941%3C/td%3E%0A%20%20%20%20%3C/tr%3E%0A%20%20%3C/tbody%3E%0A%3C/table%3E " style="border: none; seamless:seamless; width: 800px; height: 300px">
</iframe>
</p>

Figure 4. Scoring results on the algorithms tested

As a result, we are using Logistic Regression (balanced) with
class\_weight equal to “balanced” which deals with the class imbalance
originally identified in our proposal. Then, we conducted hyperparameter
optimization for the parmeters C, the inverse of regularization
strength, and max\_iter, the maximum number of iterations, by carrying
out 5-fold cross-validation. We used the f1-score as our metric of model
prediction performance. We observed that the optimal combination of C
and max\_iter were 1 and 200, respectively. There were also other
combinations that also had the same score, so we arbitrarily chose the
first one.

<p align="center">
<iframe src="data:text/html;charset=utf-8, %3Ctable%20border=%221%22%20class=%22dataframe%22%3E%0A%20%20%3Cthead%3E%0A%20%20%20%20%3Ctr%20style=%22text-align:%20center;%22%3E%0A%20%20%20%20%20%20%3Cth%3Erank%3C/th%3E%0A%20%20%20%20%20%20%3Cth%3Ef1%3C/th%3E%0A%20%20%20%20%20%20%3Cth%3Eparam_logisticregression__C%3C/th%3E%0A%20%20%20%20%20%20%3Cth%3Eparam_logisticregression__max_iter%3C/th%3E%0A%20%20%20%20%3C/tr%3E%0A%20%20%3C/thead%3E%0A%20%20%3Ctbody%3E%0A%20%20%20%20%3Ctr%3E%0A%20%20%20%20%20%20%3Ctd%3E1%3C/td%3E%0A%20%20%20%20%20%20%3Ctd%3E0.583629%3C/td%3E%0A%20%20%20%20%20%20%3Ctd%3E1%3C/td%3E%0A%20%20%20%20%20%20%3Ctd%3E800%3C/td%3E%0A%20%20%20%20%3C/tr%3E%0A%20%20%20%20%3Ctr%3E%0A%20%20%20%20%20%20%3Ctd%3E1%3C/td%3E%0A%20%20%20%20%20%20%3Ctd%3E0.583629%3C/td%3E%0A%20%20%20%20%20%20%3Ctd%3E1%3C/td%3E%0A%20%20%20%20%20%20%3Ctd%3E1800%3C/td%3E%0A%20%20%20%20%3C/tr%3E%0A%20%20%20%20%3Ctr%3E%0A%20%20%20%20%20%20%3Ctd%3E1%3C/td%3E%0A%20%20%20%20%20%20%3Ctd%3E0.583629%3C/td%3E%0A%20%20%20%20%20%20%3Ctd%3E1%3C/td%3E%0A%20%20%20%20%20%20%3Ctd%3E1000%3C/td%3E%0A%20%20%20%20%3C/tr%3E%0A%20%20%20%20%3Ctr%3E%0A%20%20%20%20%20%20%3Ctd%3E4%3C/td%3E%0A%20%20%20%20%20%20%3Ctd%3E0.583338%3C/td%3E%0A%20%20%20%20%20%20%3Ctd%3E100%3C/td%3E%0A%20%20%20%20%20%20%3Ctd%3E200%3C/td%3E%0A%20%20%20%20%3C/tr%3E%0A%20%20%20%20%3Ctr%3E%0A%20%20%20%20%20%20%3Ctd%3E5%3C/td%3E%0A%20%20%20%20%20%20%3Ctd%3E0.583334%3C/td%3E%0A%20%20%20%20%20%20%3Ctd%3E10%3C/td%3E%0A%20%20%20%20%20%20%3Ctd%3E200%3C/td%3E%0A%20%20%20%20%3C/tr%3E%0A%20%20%3C/tbody%3E%0A%3C/table%3E " style="border: none; seamless:seamless; width: 800px; height: 180px">
</iframe>
</p>

Figure 5. Top 5 results from hyperparameter optimization.

Our classification model has reasonable performance on the test data,
with a final f1-score of 0.6. As we were focused on minimizing false
negatives compared to false positives, as mentioned in the summary, we
were using the f1-score as the main metric but had a bias of recall over
precision as we focused on capturing as many customers as we can. With
the current f1-score, we believe there is more room to improve on this
model.

<div class="figure" style="text-align: center">

<img src="../results/confusion_matrix.svg" alt="Figure 6. Comfusion matrix on the test result." width="60%" />
<p class="caption">
Figure 6. Comfusion matrix on the test result.
</p>

</div>

From this stage of analysis, we reviewed the features with the top 10
weights identified by the model. They did overlap with the some of
features identified in our explanatory data analysis.

<div class="figure" style="text-align: center">

<img src="../results/top10_predictors_disregard_direction.svg" alt="Figure 7. Top 10 features identified by model." width="60%" />
<p class="caption">
Figure 7. Top 10 features identified by model.
</p>

</div>

To further improve on this model, we can look into the false negatives
and false positives to see if we can compare them to the ones that are
correctly predicted. From there we can potentially identify features
that have more influence on this incorrect prediction and explore
different feature engineering techniques to improve our model. We will
also spend more effort on feature selection and attempt recursive
feature elimination and/or feature selection. In addition to providing
the call agents with predictions on whether a customer would potentially
subscribe to the term deposit product, we could provide probability
estimates for the prediction so the bank’s call agents can use their own
judgment to whether or not they will spend their time to engage a
particular customer. If they are not busy, they may be more willing to
contact those with lower probabilities. If they are busy, they can skip
particular customers to try and maximize the revenue stream.

# References

<div id="refs" class="references csl-bib-body hanging-indent">

<div id="ref-barich1991framework" class="csl-entry">

Barich, Howard, and Philip Kotler. 1991. “A Framework for Marketing
Image Management.” *MIT Sloan Management Review* 32 (2): 94.

</div>

<div id="ref-Dua2019" class="csl-entry">

Dua, Dheeru, and Casey Graff. 2017. “UCI Machine Learning Repository.”
University of California, Irvine, School of Information; Computer
Sciences. <http://archive.ics.uci.edu/ml>.

</div>

<div id="ref-Hunter" class="csl-entry">

Hunter, J. D. 2007. “Matplotlib: A 2d Graphics Environment.” *Computing
in Science & Engineering* 9 (3): 90–95.
<https://doi.org/10.1109/MCSE.2007.55>.

</div>

<div id="ref-docoptpython" class="csl-entry">

Keleshev, Vladimir. 2014. *Docopt: Command-Line Interface Description
Language*. <https://github.com/docopt/docopt>.

</div>

<div id="ref-mckinney-proc-scipy-2010" class="csl-entry">

McKinney, Wes. 2010. “Data Structures for Statistical Computing in
Python.” In *Proceedings of the 9th Python in Science Conference*,
edited by Stéfan van der Walt and Jarrod Millman, 56–61.
<https://doi.org/10.25080/Majora-92bf1922-00a>.

</div>

<div id="ref-moro2014data" class="csl-entry">

Moro, Sérgio, Paulo Cortez, and Paulo Rita. 2014. “A Data-Driven
Approach to Predict the Success of Bank Telemarketing.” *Decision
Support Systems* 62: 22–31.

</div>

<div id="ref-numpy" class="csl-entry">

Oliphant, Travis. 2006–. “NumPy: A Guide to NumPy.” USA: Trelgol
Publishing. <http://www.numpy.org/>.

</div>

<div id="ref-R" class="csl-entry">

R Core Team. 2020. *R: A Language and Environment for Statistical
Computing*. Vienna, Austria: R Foundation for Statistical Computing.
<https://www.R-project.org/>.

</div>

<div id="ref-rust2010spotlight" class="csl-entry">

Rust, Roland T, Christine Moorman, and Gaurav Bhalla. 2010. “Spotlight
on Reinvention: Rethinking Marketing.” *Harvard Business Review* 88 (1):
2–8.

</div>

<div id="ref-Python" class="csl-entry">

Van Rossum, Guido, and Fred L. Drake. 2009. *Python 3 Reference Manual*.
Scotts Valley, CA: CreateSpace.

</div>

<div id="ref-Altair2018" class="csl-entry">

VanderPlas, Jacob, Brian Granger, Jeffrey Heer, Dominik Moritz, Kanit
Wongsuphasawat, Arvind Satyanarayan, Eitan Lees, Ilia Timofeev, Ben
Welsh, and Scott Sievert. 2018. “Altair: Interactive Statistical
Visualizations for Python.” *Journal of Open Source Software*, December.
<https://doi.org/10.21105/joss.01057>.

</div>

<div id="ref-michael_waskom_2017_883859" class="csl-entry">

Waskom, Michael, Olga Botvinnik, Drew O’Kane, Paul Hobson, Saulius
Lukauskas, David C Gemperline, Tom Augspurger, et al. 2017.
*Mwaskom/Seaborn: V0.8.1 (September 2017)* (version v0.8.1). Zenodo.
<https://doi.org/10.5281/zenodo.883859>.

</div>

<div id="ref-knitr" class="csl-entry">

Xie, Yihui. 2020. *Knitr: A General-Purpose Package for Dynamic Report
Generation in r*. <https://yihui.org/knitr/>.

</div>

</div>
