import os
import json
import pandas as pd
import numpy as np

from ingestion import merge_directory, load_ingestedfiles, merge_multiple_dataframe
from scoring import get_model_last_version_score, get_model_scores,score_model, get_model_last_version
from config import Score


# initialize logging
import logging 
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

# Exception handling

#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 


def check_4_newdata() -> list:
    '''
    Check for new data to ingest. Returns a list of new files.
    '''
    ##################Check and read new data
    #first, read ingestedfiles.txt
    ingested_files = load_ingestedfiles(os.path.join(config['output_folder_path'], config['ingested_files']))
    logger.info(f"Ingested file name list: {ingested_files}")

    #second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
    current_files, _ = merge_directory(os.path.join(config['input_folder_path']))
    logger.info(f"Current file name list: {current_files}")

    return  list(set(current_files).difference(set(ingested_files)))


def check_4_modeldrift(hist_score_list: list, new_score: Score) -> bool:
    '''
    Check if model drift occurred.
    '''
    scores = []
    for item in hist_score_list:
        scores.append(item['score'])
    iqr = np.quantile(scores,0.75)-np.quantile(scores,0.25)
    logger.info(f"historical scores: {scores} iqr: {iqr}")

    if (new_score['score'] > np.quantile(scores,0.75)+iqr*1.5):
        return True
    elif (new_score['score'] < np.quantile(scores,0.25)-iqr*1.5):
        return True
    
    return False



def process_automation():
    
    new_files = check_4_newdata()
    logger.info(f"Files not yet ingested: {new_files}")

    if len(new_files) > 0:

        # ingest the new data
        merge_multiple_dataframe()
        logger.info("New data merged and ingested.")

        ##################Checking for model drift
        #check whether the score from the deployed model is different from the score from 
        # recent_model_score = get_model_last_version_score(os.path.join(config['prod_deployment_path'], config['scores']))
        # logger.info(f"Last version score: {recent_model_score}")

        # the model that uses the newest ingested datas
        ingested_model_score = score_model(
            os.path.join(config['prod_deployment_path'], config['model']),
            os.path.join(config['output_folder_path'], config['cleaned_data'])
        )
        logger.info(f"Ingested version score: {ingested_model_score}")

        ##################Deciding whether to proceed, part 2
        #if you found model drift, you should proceed. otherwise, do end the process here
        hist_version_scores = get_model_scores(
                                os.path.join(config['prod_deployment_path'], config['scores']),
                                get_model_last_version(os.path.join(config['prod_deployment_path'], config['scores'])))
        if check_4_modeldrift(hist_version_scores, ingested_model_score):
            pass
            ##################Re-deployment
            #if you found evidence for model drift, re-run the deployment.py script

            ##################Diagnostics and reporting
            #run diagnostics.py and reporting.py for the re-deployed model


if __name__ == '__main__':
    try:
        process_automation()
    except Exception as err:
        print(f"Process Automation Main Error: {err}")



