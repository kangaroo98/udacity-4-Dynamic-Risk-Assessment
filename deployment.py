'''
Step 2 - Training, Scoring, and Deploying an ML Model

Author: Oliver
Date: 2022, March

'''
import os
import json
import shutil

from shared import FileInvalid, DeploymentFailed
from scoring import get_model_last_version, get_model_scores, save_score_list

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


####################function for deployment

def deploy_model_score(scores_pth_src: str, version: int, scores_pth_dest: str):
    '''
    Deploy the model score by version number to the dest path.
    '''
    # get all train and test scores of the model version and save in dest
    scores = get_model_scores(scores_pth_src, version)
    save_score_list(scores_pth_dest, scores)
    #save_score_list(os.path.join(dest, f"v{version}_{config['scores']}"), scores)
    logger.info(f"Model version {version} with train and test scores {scores} are saved as {scores_pth_dest}")


def deploy_model(
    version: int, 
    model_dir_src: str, 
    model_name: str, 
    ingested_pth_src: str, 
    scores_dir_src: str, 
    scores_name:str,
    dir_dest: str):
    '''
    Deploy all model artifacts by version.
    '''
    # model file
    shutil.copy(os.path.join(model_dir_src, f"v{version}_{model_name}"), os.path.join(dir_dest, model_name))

    # model score
    deploy_model_score(os.path.join(scores_dir_src, scores_name), version, os.path.join(dir_dest, scores_name))
    
    # ingested files
    shutil.copy(ingested_pth_src, dir_dest)

        
        
if __name__ == '__main__':
    try:
        init() 

        #copy the latest pickle file, the latestscore.txt value, and the ingestfiles.txt file into the deployment directory
        version = get_model_last_version(os.path.join(config['output_model_path'], config['scores']))
        logger.info(f"Version to deploy: {version}")
        deploy_model(
            version, 
            os.path.join(config['output_model_path']),
            os.path.join(config['model']),
            os.path.join(config['output_folder_path'], config['ingested_files']),
            os.path.join(config['output_model_path']),
            os.path.join(config['scores']),
            os.path.join(config['prod_deployment_path'])
        )

    except Exception as err:
        print(f"Deployment Main Error: {err}")

