# Post Campaign Bank Term Deposit Subscription Predictor
- author: DSCI 522 Group10
- Group Member: Justin Fu, Junting He, Chuck Ho, Asma Al-Odaini

A data analysis project for DSCI 522 (Data Science workflows); a course in the Master of Data Science program at the University of British Columbia.

## About

With a banking institution's telemarketing campaign data, we are attempting to build a classification model using the Logistice Regression algorithm to help us identify and pursue customers that will subscribe to a term deposit product. Our final classification model has reasonable performance on the test data, with a f1-score of 0.6 and a recall score of 0.90. On the 8238 test data cases, we correctly predicted 7091. However, it incorrectly predicted 1147 cases with 1056 being false positives and 91 being false negatives. We were more focused on reducing false negatives to capture more revenue. The false positives are also important as it may require more resources to reach out to potential customers, but our bias was towards revenue (recall). This model is not sufficient to be used in the industry since the f-1 score still has a lot to improve on. We recommend continuing to improve the classification performance before we use this in production.  


The data set used in this project is from from a marketing campaign of a Portuguese bank and created by S. Moro, P. Cortez and P. Rita (Moro, Cortez, and Rita 2014). It was sourced from the UCI Machine Learning Repository (@ Dua and Graff 2017) and can be found [here](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing). Each row in the data set represents summary statistics with detail information of the contacted client, including bank client info (e.g. age, job, loan experience, etc.), other campaign attributes (e.g. number of contact, previous campaign outcome, etc) and social and economic attributes. (e.g. consumer confidence index, euribor rate, etc.) They were using telemarketing to attempt to get customers to sign up for the bank's term deposit product. The target in this dataset is yes or no to subscribing to the term deposit product.

## Report

The report can be found [here](https://htmlpreview.github.io/?https://raw.githubusercontent.com/UBC-MDS/DSCI_522_Group_10/main/doc/bank_marketing_predict_report.html).

## Usage

To replicate the analysis, clone this GitHub repository, install the [dependencies](#dependencies) listed below, and run the following commands at the command line/terminal from the root directory of this project:

```
#activate our conda environment
conda activate bank

make all
```

## Dependencies

### Python

We are providing you with a `conda` environment file which is available [here](env-bank_marketing.yaml). You can download this file and create a conda environment for this project and activate it as follows. 

```
conda env create -f env-bank_marketing.yaml
conda activate bank
```

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
    
### R
R version 3.6.1 and R packages:
  - knitr==1.26
  - tidyverse==1.2.1
  - caret==6.0-84
  - ggridges==0.5.1
  - ggthemes==4.2.0
  - docopt==0.7.1
  - rmarkdown==2.5
    
  
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
