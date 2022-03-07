import pandas as pd
import numpy as np
import timeit
import os
import json
import joblib
import subprocess

from data import preprocess
from training import train_model
from ingestion import merge_multiple_dataframe

# initialize logging
import logging 
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

# Exception handling

##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 


##################Function to get model predictions
def model_predictions(model_pth: str, dataset: pd.DataFrame) -> pd.array:
    #read the deployed model and a test dataset, calculate predictions

    # load the trained model
    model = joblib.load(model_pth)
    
    # prepare data, no dataset split 
    X, _, y, _ = preprocess(dataset)
    
    return model.predict(X)
    

##################Function to get summary statistics
def dataframe_summary(dataset: pd.DataFrame) -> dict:
    # calculate summary statistics here
    # means, medians, and standard deviations
    # determine the numeric columns of the dataset
    num_cols = dataset.select_dtypes(include=np.number).columns.tolist()
    logger.info(f"Numeric columns: {num_cols}")

    # save statistics per numeric column in a dict
    summary = {}
    for item in num_cols:
        summary[item] = {'mean': dataset[item].mean(), 'median':dataset[item].median(),'std':dataset[item].std()}

    #return value should be a list containing all summary statistics
    return summary


def missing_data(dataset: pd.DataFrame) -> pd.Series:
    
    # calculate the percentage of missing values
    na_percent = dataset.isna().sum()/len(dataset)    
    return na_percent


##################Function to get timings
def execution_time() -> dict:
    
    #calculate timing of training.py and ingestion.py
    start = timeit.default_timer()
    merge_multiple_dataframe()
    end_ingestion = timeit.default_timer()
    train_model(
            os.path.join(config['output_folder_path'], config['cleaned_data']),
            os.path.join(config['output_model_path'], config['scores']), 
            os.path.join(config['output_model_path'])
    )
    end_training = timeit.default_timer()

    #return a list of 2 timing values in seconds
    return {"ingestion":(end_ingestion-start), "training":(end_training-end_ingestion)}


##################Function to check dependencies
def outdated_packages_list():
    #get a list of
    outdated = subprocess.check_output(['python','-m','pip', 'list', '--outdated'])
    logger.info(f"Check: {outdated}")
    return 


if __name__ == '__main__':
    try:
        # model predictions
        df = pd.read_csv(os.path.join(config['test_data_path'], config['test_data']))
        preds = model_predictions(os.path.join(config['prod_deployment_path'], config['model']), df)
        logger.info(f"Predictions: {preds}")

        # statistic summary
        df = pd.read_csv(os.path.join(config['output_folder_path'], config['cleaned_data']))
        statistics = dataframe_summary(df)
        logger.info(f"Dataset statistics: {statistics}")
        na_percentage = missing_data(df)
        logger.info(f"Missing data: {na_percentage}")
        
        # measure the execution of ingestion and training process (versioning and logging not considered ;-)
        duration = execution_time()
        logger.info(f"Execution time: {duration}")

        outdated_packages_list()
    
    except Exception as err:
        print(f"Diagnositcs Main Error: {err}")   
