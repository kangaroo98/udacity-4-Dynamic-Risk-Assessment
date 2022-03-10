'''
Step 3 - Diagnostics

Author: Oliver
Date: 2022, March

'''
import pandas as pd
import numpy as np
import timeit
import os
import json
import subprocess

from training import train_model
from ingestion import merge_multiple_dataframe
from training import model_predictions

from shared import Score

# initialization
import logging 
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

config = {}
def init():
    global config
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    with open('config.json','r') as f:
         config = json.load(f)
    logger.info(f"Current working dir: {os.getcwd()}")
    logger.info(f"Config dictionary: {config}")
   

def dataframe_summary(dataset: pd.DataFrame) -> dict:
    '''
    Calculates summary statistics (means, medians, and standard deviations).
    Only numeric columns are considered. Returns a dict accessible by the col name. 
    '''
    # determine the numeric columns of the dataset
    num_cols = dataset.select_dtypes(include=np.number).columns.tolist()
    logger.info(f"Numeric columns: {num_cols}")

    # save statistics per numeric column in a dict
    summary = {}
    for item in num_cols:
        summary[item] = {'mean': dataset[item].mean(), 'median':dataset[item].median(),'std':dataset[item].std()}

    #return value should be a list containing all summary statistics
    return summary


def missing_data(dataset: pd.DataFrame) -> dict:
    '''
    Calculates the percentage of missing values. 
    '''
    # calculate the percentage of missing values
    na_percent = dataset.isna().sum()/len(dataset)    
    return na_percent.to_dict()


def execution_time() -> dict:
    '''
    Function to get timings of the ingestion and training processs. 
    Returns a dict accessible by 'ingestion' re. 'training' key and the corresponding timings.
    '''
    #calculate timing of training.py and ingestion.py
    start = timeit.default_timer()
    os.system('python3 ingestion.py')
    end_ingestion = timeit.default_timer()
    os.system('python3 training.py')
    end_training = timeit.default_timer()

    # return a list of 2 timing values in seconds
    return {"ingestion":(end_ingestion-start), "training":(end_training-end_ingestion)}


def outdated_packages_list():
    '''
    Function to check dependencies
    '''
    #get a list of
    outdated = subprocess.check_output(['python','-m','pip', 'list', '--outdated'])
    with open('outdated.txt', 'wb') as f:
       f.write(outdated)
    
    return outdated


def diagnostics(testdata_pth: str, prod_depl_model_pth: str, test_dataset_pth: str):

    # model predictions
    df = pd.read_csv(testdata_pth)
    preds, _ = model_predictions(prod_depl_model_pth, df)

    # statistic summary
    df = pd.read_csv(test_dataset_pth)
    statistics = dataframe_summary(df)
    logger.info(f"Dataset statistics: {statistics}")
    na_percentage = missing_data(df)
    logger.info(f"Missing data: {na_percentage}")
    
    # measure the execution of ingestion and training process (versioning and logging not considered ;-)
    duration = execution_time()
    logger.info(f"Execution time: {duration}")

    #outdated = outdated_packages_list()
    #logger.info(f"Check: {outdated}")
    


if __name__ == '__main__':
    try:
        init()
        diagnostics(
            os.path.join(config['test_data_path'], config['test_data']),
            os.path.join(config['prod_deployment_path'], config['model']),
            os.path.join(config['output_folder_path'], config['cleaned_data'])
        )
    except Exception as err:
        print(f"Diagnositcs Main Error: {err}")   
