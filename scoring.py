'''
Step 2 - Training, Scoring, and Deploying an ML Model

Author: Oliver
Date: 2022, March

'''
import os
import joblib
import json
from datetime import datetime

from sklearn.metrics import f1_score

from shared import Score 
from shared import load_prepare_data

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

# Exception handling
from shared import FileInvalid 


def get_model_scores(score_list_pth: str, version: int) -> list:
    '''
    Read and return the list of all scores from score_list_pth,
    containing all scores captured during training and testing.
    '''
    # load versioned scores
    scores = load_score_list(score_list_pth)
    rec_scores = []

    if (len(scores) > 0):
        for item in scores:
            if (item['version'] == version):
                    rec_scores.append(item) 
        return rec_scores
    
    return None

def get_model_last_version_score(score_list_pth: str) -> Score:
    '''
    Return the scor of the most recent version.
    '''
    # load versioned scores
    scores = load_score_list(score_list_pth)

    # determine highest version number in the list
    if (len(scores) <= 0):
        raise FileInvalid("Score list file does not exist or is empty.")

    rec_score = max(scores, key=lambda x:x['version']) 
    logger.info(f"Most recent version (list count: {len(scores)}): {rec_score}")
    
    return rec_score

def get_model_last_version(score_list_pth: str) -> int:
    '''
    Return the most recent version number.
    '''
    score = get_model_last_version_score(score_list_pth)
    
    return score['version']

def load_score_list(score_list_pth: str) -> list:
    '''
    Load and return the list of scores.
    '''
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
    '''
    Save the list of scores.
    '''
    # save the score_list in the pth dir
    with open(score_list_pth, 'w') as f:
        json.dump(score_list, f)


def version_score(score: Score, score_list_pth: str, new_version: bool=True) -> int:
    '''
    Add a new score. By default the score is versioned with a new version number increment. 
    But if new_version is set to False the recent version number is used. This is used to 
    save a history of test scores to the most recent model version (just for myself to practice ;-).  
    '''
    # read current scores 
    scores = load_score_list(score_list_pth)

    # set most recent version
    new_score = json.loads(score.json()) 
    if (len(scores) > 0):
        rec_score = max(scores, key=lambda x:x['version']) 
        logger.info(f"Most recent version (list count: {len(scores)}): {rec_score}")
        new_score['version'] = (rec_score['version'] + 1) if new_version else rec_score['version']
    else:
        new_score['version'] = 1 if new_version else 0


    # add/save the new score
    scores.append(new_score)
    save_score_list(score_list_pth, scores)

    logger.info(f"Score ({new_score}) added to list (version_flag {new_version}) and saved in {score_list_pth}")
    return new_score['version']


def scoring(mode, model, X_test, y_test) -> Score:
    '''
    Predicting a feature dataset and returning a Score (pydantic model defined in config.py) 
    '''
    #this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    #it should write the result to the latestscore.txt file

    # scoring
    preds = model.predict(X_test)
    test=f1_score(y_test, preds)
    score_res = Score(mode=mode, metric='F1', score=f1_score(y_test, preds), timestamp=datetime.now())
    logger.info(f"Tested model with X_test {X_test} resulted in preds {preds} vs act {y_test} and score {test}")

    # write the score to file for further reference
    return score_res


def score_model(model_path: str, data_pth: str) -> Score:
    '''
    #this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    #it should write the result to the latestscore.txt file
    '''    
    # load the trained model
    model = joblib.load(model_path)
    
    # load and prepare test data, no dataset split 
    X_test, _, y_test, _ = load_prepare_data(data_pth)
    
    # score the model
    return scoring('test', model, X_test, y_test)


if __name__ == '__main__':
    try:
        init() 

        last_version = get_model_last_version(os.path.join(config['output_model_path'], config['scores']))

        score_obj = score_model(
            os.path.join(config['output_model_path'], f"v{last_version}_{config['model']}"),
            os.path.join(config['test_data_path'], config['test_data'])
        )

        # version the score to file for further reference
        new_version = version_score(score_obj, os.path.join(config['output_model_path'], config['scores']), False)
        logger.info(f"Tested model version {last_version} with F1 score {score_obj}. Score (mode: {score_obj['mode']}) saved as version: {new_version}")

    except Exception as err:
        print(f"Scoring Main Error: {err}")