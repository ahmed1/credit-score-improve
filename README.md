MetaData:



00_raw_data.csv — csv file produced after parsing training folder

00_raw_val_data.csv — csv file produced after parsing validation folder

App — folder with home.html and index.php for frontend app on heroku

BaseLineModel Prod.ipynb — notebook used to create baseline model

clean_data.py -- includes all functions used to clean / transform the data

Deploy-app — includes template.yaml (for cloudformation — used to declare resources required and define necessary roles) and app.py / requirements.txt files I dockerized and used to build AWS infrastructure necessary for data pipeline to parse and send data to model deployed on AWS.

> this also includes the test.sh and deploy.sh scripts I wrote to test the pipeline and  deploy it on AWS, as well as events/event.json that I used to test the pipeline locally.

Eda_prod.ipynb -- parsing the files, exploring the train dataset, creating visualizations in this notebook

**final_report.pdf** — Data Science report I wrote explaining everything I did.

Modelv2.ipynb — notebook I used to build second iteration of model

parser.py -- includes all the functions used to parse the train / validation files and transform them to dataframes

validation_scores.txt — all the predicted scores, using the baseline model, for the validation set.