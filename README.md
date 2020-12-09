## Disaster-Response-MultiLabelClassifcation with ETL pipeline, ML pipeline and dash based web application
In this project we will do a multi label classification of disaster related messages.

### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>
The project uses the following libraries in addition to Anaconda with python version:3.7.4.
- pandas
- numpy
- seaborn
- plotly
- dash
- sklearn
- xgboost
- nltk
- pickle
- TextBlob
- sqlalchemy
- re

## Project Motivation <a name="motivation"></a>
In this project, we will dive into the Figure-8(now Appen) disaster response message which is available on Appen[Disaster Response dataset](https://appen.com/datasets/combined-disaster-response-data/)). We will implement ETL(Extract, Transform and Load) pipeline, ML pipeline and dash based web application which will show a data dashboard for visualization and also a text classification which will take input from user and give probabilites of different labels.

## Web Application Sample Snapshots
![file1](https://github.com/sanketg186/Disaster-Response-MultiLabelClassifcation/blob/main/visualization1.png)

![file2](https://github.com/sanketg186/Disaster-Response-MultiLabelClassifcation/blob/main/visualization2.png)

![file3](https://github.com/sanketg186/Disaster-Response-MultiLabelClassifcation/blob/main/visualization3.png)

![file4](https://github.com/sanketg186/Disaster-Response-MultiLabelClassifcation/blob/main/visualization4.png)

## File Descriptions <a name="files"></a>
This folder contains to jupyter notebooks.
1. **etl.ipnyb**: This notebook contains the ETL(Extract,Transform, Load) pipeline for cleaning the data and saving the data into the database.
2. **ML_Pipeline.ipnyb** : This notebook contains machine learning implementations for pipeline and multilabel classification.
3. **app** : This folder contains the web application code. 
4. **data**: This folder contains the data files, data cleaning and database creation code.
5. **models**: This folder contains the model training code.

## Run
In the terminal go to the directory Disaster-Response-MultiLabelClassifcation/ and run the following commands in sequence.
- python data/process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db : This will create DisasterResponse.db database.
- python models/train_classifier.py DisasterResponse.db classifier.pkl: This will train classifier on database and save the model as classifier.pkl.
- python app/run.py : This will run a web application.

Run the web application Go to http://127.0.0.1:3001/
The web app provides data dashboard and text-message classification page for mutli-label classification. The F1-score for labels falls in the range of .62 and .93. The imbalance issue has effected the F1-score. Proper model selection and paramter tuning is needed to get optimum results.

## Results <a name="results"></a>
The dataset has class imbalance issue. This resul
## Licensing, Authors, Acknowledgements <a name="licensing"></a>
Must give credit to Figure-8 for the data.
