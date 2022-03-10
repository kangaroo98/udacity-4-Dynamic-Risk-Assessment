import os
import json
import pandas as pd
import numpy as np

from ingestion import merge_directory, load_ingestedfiles, merge_multiple_dataframe
from reporting import reporting
from scoring import get_model_last_version_score, get_model_scores,score_model, get_model_last_version
from training import train_model
from deployment import deploy_model
from diagnostics import diagnostics
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

    # check if F1score is a low outlier
    if (new_score['score'] < np.quantile(scores,0.25)-iqr*1.5):
        return True
    # elif (new_score['score'] > np.quantile(scores,0.75)+iqr*1.5):
    #     return True
    
    return False


def process_automation():
    
    new_files = check_4_newdata()
    logger.info(f"Files not yet ingested: {new_files}")

    if len(new_files) > 0:

        # ingest the new data
        merge_multiple_dataframe(
            os.path.join(config["output_folder_path"], config['cleaned_data']),
            os.path.join(config['input_folder_path']),
            os.path.join(config['output_folder_path']),
            config['cleaned_data'],
            config['ingested_files']
        )
        logger.info("New data merged and ingested.")

        ##################Checking for model drift
        # check whether the score from the deployed model is different from the score from 
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
        
        if check_4_modeldrift(hist_version_scores, ingested_model_score) or True:
            # Re-train the model
            train_score, version = train_model(
                os.path.join(config['output_folder_path'], config['cleaned_data']),
                os.path.join(config['output_model_path'], config['scores']), 
                os.path.join(config['output_model_path']),
                os.path.join(config['model'])
            )
            logger.info(f"Re-train score: {train_score}")

            ##################Re-deployment
            #if you found evidence for model drift, re-run the deployment.py script
            deploy_model(
                version, 
                os.path.join(config['output_model_path']),
                os.path.join(config['model']),
                os.path.join(config['output_folder_path'], config['ingested_files']),
                os.path.join(config['output_model_path']),
                os.path.join(config['scores']),
                os.path.join(config['prod_deployment_path'])
            )
            logger.info(f"Model version {version} deployed in {config['prod_deployment_path']}.")

            ##################Diagnostics and reporting
            #run diagnostics.py and reporting.py for the re-deployed model and the api calls
            diagnostics(
                os.path.join(config['test_data_path'], config['test_data']),
                os.path.join(config['prod_deployment_path'], config['model']),
                os.path.join(config['output_folder_path'], config['cleaned_data'])
            )
            logger.info(f"Diagnostics based on model {os.path.join(config['prod_deployment_path'], config['model'])}")

            reporting(
                os.path.join(config['prod_deployment_path'], config['model']),
                os.path.join(config['test_data_path'], config['test_data']),
                os.path.join(config['output_model_path'], config['confusionmatrix'])
            )
            logger.info(f"Reporting based on model {os.path.join(config['prod_deployment_path'], config['model'])}")

            os.system("python3 apicalls.py")


if __name__ == '__main__':
    try:
        init()
        process_automation()
    except Exception as err:
        print(f"Process Automation Main Error: {err}")



