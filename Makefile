# bank marketing campaign customer prediction
# author: Group 10 (Justin Fu, Junting He, Chuck Ho, Asma Al-Odaini)
# date: 2020-12-04

# This driver script extracts Bank marketing data from a source, 
# preprocess and splits the data into training and test sets, 
# generates visualizations of features and class distribution, 
# and performs a machine learning analysis to predict 
# the outcome of a term deposit bank marketing call 
# (purchase or no purchase). 

# example usage:
# make all

all:  src/bank_marketing_data_eda.html doc/bank_marketing_predict_report.html 

#down and extract data from zip of the data source url
data/raw/bank-additional/bank-additional-full.csv : src/download_and_extract_zip.py 
	python src/download_and_extract_zip.py --url=https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip --out_file="data/raw/"

# preprocess and split data 
data/processed/bank-additional-full_train.csv data/processed/bank-additional-full_test.csv : src/data_cleaning_and_splitting.py data/raw/bank-additional/bank-additional-full.csv src/data_cleaning_and_splitting.py
	python src/data_cleaning_and_splitting.py --in_file=data/raw/bank-additional/bank-additional-full.csv --out_file="data"


# Repurposed the pattern % functionality to simulate grouped targets. If we use Make version 4.3, we can use "&:" 
# create exploratory data anlysis figures for categorical features
results/age%png results/last_contact_duration%png results/contacts_during_campaign%png results/days_after_previous_contact%png results/previous_contacts%png results/employment_variation_rate%png results/consumer_price_index%png results/consumer_confidence_index%png results/euribor_3_month_rate%png results/number_of_employees%png results/contact%png results/day_of_week%png results/default%png results/education%png results/housing%png results/job%png results/loan%png results/marital_status%png results/month%png : data/processed/bank-additional-full_train.csv src/data_vis.py	
	mkdir -p results
	python src/data_vis.py --data_path="data/processed/bank-additional-full_train.csv" --image_path="results/"


# create preliminary exploratory data analysis report
src/bank_marketing_data_eda.html: src/bank_marketing_data_eda.ipynb data/raw/bank-additional/bank-additional-full.csv data/processed/bank-additional-full_train.csv data/processed/bank-additional-full_test.csv results/age.png results/last_contact_duration.png results/contacts_during_campaign.png results/days_after_previous_contact.png results/previous_contacts.png results/employment_variation_rate.png results/consumer_price_index.png results/consumer_confidence_index.png results/euribor_3_month_rate.png results/number_of_employees.png results/job.png results/marital_status.png results/default.png results/housing.png results/loan.png results/previous_outcome.png results/contact.png results/education.png results/day_of_week.png  results/month.png
	jupyter nbconvert src/bank_marketing_data_eda.ipynb --no-input --to html --TemplateExporter.exclude_input=True --no-prompt

# create, train, and test model
results/model_selection%html results/hyperparameter_optimization_result%html results/classification_report%svg results/top10_predictors_table%html results/top10_predictors_disregard_direction%svg  : src/machine_learning_analysis.py data/processed/bank-additional-full_train.csv data/processed/bank-additional-full_test.csv
	python src/machine_learning_analysis.py --in_train="data/processed/bank-additional-full_train.csv" --in_test="data/processed/bank-additional-full_test.csv" --out_path="results/"

# render final report
doc/bank_marketing_predict_report.html : results/previous_outcome.png results/month.png results/consumer_price_index.png results/euribor_3_month_rate.png results/last_contact_duration.png results/employment_variation_rate.png results/model_selection.html results/hyperparameter_optimization_result.html results/confusion_matrix.svg results/top10_predictors_disregard_direction.svg doc/bank_marketing_predict_report.Rmd
	Rscript -e "rmarkdown::render('doc/bank_marketing_predict_report.Rmd', output_format = 'github_document')" 

clean: 
	rm -rf data
	rm -rf results	
	rm -f src/bank_marketing_data_eda.html		
	rm -f doc/bank_marketing_predict_report.html	
	rm -f doc/bank_marketing_predict_report.md	

			
