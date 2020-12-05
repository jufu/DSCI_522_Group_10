# Makefile
# Asma Al-Odaini, Dec 2020

# This driver script extracts Bank marketing data from a source, 
# preprocess and splits the data into training and test sets, 
# generates visualizations of features and class distribution, 
# and performs a machine learning analysis to predict 
# the outcome of a term deposit bank marketing call 
# (purchase or no purchase). 

# example usage:
# make all

#all : doc/count_report.md

# preprocess and split data 
data/processed/bank-additional-full_train.csv data/processed/bank-additional-full_test.csv : src/data_cleaning_and_splitting.py data/raw/bank-additional-full.csv
    python src/data_cleaning_and_splitting.py --in_file=data/raw/bank-additional-full.csv --out_file="data"


# create exploratory data analysis figure for categorical features and write to file 
results/age.png : src/data_vis.py data/processed/bank-additional-full_train.csv
  python src/data_vis.py --data_path="data/processed/bank-additional-full_train.csv" --image_path="results/" --feature=age

	


