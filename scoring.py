import os
from xmlrpc.client import boolean
import joblib
import json
from datetime import datetime

from sklearn.metrics import f1_score

from config import Score 
from data import load_prepare_data

# initialize logging
import logging 
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

# Exception handling
from config import FileInvalid 

#################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 


def get_model_scores(score_list_pth: str, version) -> list:

    # load versioned scores
    scores = load_score_list(score_list_pth)
    rec_scores = []

    if (len(scores) > 0):
        for item in scores:
            if (item['version'] == version):
                    rec_scores.append(item) 
        return rec_scores
    
    return None


def get_model_last_version(score_list_pth: str) -> int:

    # load versioned scores
    scores = load_score_list(score_list_pth)

    # determine highest version number in the list
    if (len(scores) <= 0):
        raise FileInvalid("Score list file does not exist or is empty.")

    rec_score = max(scores, key=lambda x:x['version']) 
    logger.info(f"Most recent version (list count: {len(scores)}): {rec_score}")
    return rec_score['version']


def load_score_list(score_list_pth: str) -> list:
    
    # load all model scores
    scores=[]
    if os.path.isfile(score_list_pth):
        with open(score_list_pth, 'r') as f:
            for item in f:
                scores = json.loads(item)
    else:
        logger.info(f"Score list file ({score_list_pth}) does not exist, creating a new file.")
    return scores


def save_score_list(score_list_pth: str, score_list: list):
    # save the score_list in the pth dir
    with open(score_list_pth, 'w') as f:
        json.dump(score_list, f)


def version_score(score: Score, score_list_pth: str, new_version: boolean=True) -> int:

    # read current scores 
    scores = load_score_list(score_list_pth)

    # set most recent version
    new_score = json.loads(score.json()) 
    if (len(scores) > 0):
        rec_score = max(scores, key=lambda x:x['version']) 
        logger.info(f"Most recent version (list count: {len(scores)}): {rec_score}")
        new_score['version'] = (rec_score['version'] + 1) if new_version else rec_score['version']

    # add/save the new score
    scores.append(new_score)
    save_score_list(score_list_pth, scores)

    logger.info(f"Score ({new_score}) added to list {scores} and saved in {score_list_pth}")
    return new_score['version']


#################Function for model scoring
def scoring(mode, model, X_test, y_test) -> Score:

    #this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    #it should write the result to the latestscore.txt file

    # scoring
    preds = model.predict(X_test)
    score = Score(mode=mode, metric='F1', score=f1_score(y_test, preds), timestamp=datetime.now())

    # write the score to file for further reference
    return score


def score_model(model_path: str, score_list_pth: str):

    #this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    #it should write the result to the latestscore.txt file

    # load the trained model
    model = joblib.load(model_path)
    
    # load and prepare test data, no dataset split 
    X_test, _, y_test, _ = load_prepare_data(os.path.join(config['test_data_path'], config['test_data']))
    
    # score the model
    score = scoring('test', model, X_test, y_test)

    # write the score to file for further reference
    version = version_score(score, score_list_pth, False)#
    logger.info(f"Tested model with F1 score {score} saved as version: {version}")


if __name__ == '__main__':
    try:
        score_model(
            os.path.join(config['output_model_path'], config['model']), 
            os.path.join(config['output_model_path'], config['scores'])
        )
    except Exception as err:
        print(f"Scoring Main Error: {err}")