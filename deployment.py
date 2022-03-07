import os
import json
import shutil

from config import FileInvalid, DeploymentFailed
from scoring import get_model_last_version, get_model_scores, save_score_list

# initialize logging
import logging 
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


##################Load config.json and correct path variable
with open('config.json','r') as f:
    config = json.load(f) 


score_list_path = os.path.join(config['output_model_path'], config['scores'])


####################function for deployment

def deploy_model_score(score_list_pth: str, version: int, dest: str):

    # get all train and test scores of the model version and save in dest
    logger.info(f"Model version: {version} destination: {dest}")
    scores = get_model_scores(score_list_pth, version)
    save_score_list(os.path.join(dest, config['scores']), scores)
    #save_score_list(os.path.join(dest, f"v{version}_{config['scores']}"), scores)
    logger.info(f"Model version {version} with train and test scores {scores} are saved in {dest}")


def deploy_model(version: int, model_src: str, ingest_src: str, score_list_pth: str, dest: str):
    
    # model file
    shutil.copy(model_src, dest)

    # model score
    deploy_model_score(score_list_pth, version, dest)
    
    # ingested files
    shutil.copy(ingest_src, dest)
    
    logger.info(f"Model version {version} deployed in {dest}.")

        
        
if __name__ == '__main__':
    try:
        #copy the latest pickle file, the latestscore.txt value, and the ingestfiles.txt file into the deployment directory
        working_dir = "." #os.getcwd()
        version = get_model_last_version(os.path.join(config['output_model_path'], config['scores']))
        deploy_model(
            version, 
            os.path.join(working_dir, config['output_model_path'], config['model']),
            #os.path.join(working_dir, config['output_model_path'], f"v{version}_{config['model']}"),
            os.path.join(working_dir, config['output_folder_path'], config['ingested_files']),
            os.path.join(config['output_model_path'], config['scores']),
            os.path.join(config['prod_deployment_path'])
        )

    except Exception as err:
        print(f"Deployment Main Error: {err}")

