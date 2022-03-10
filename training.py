'''
Step 2 - Training, Scoring, and Deploying an ML Model

Author: Oliver
Date: 2022, March

'''
import os
import joblib
import json
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

from scoring import scoring, version_score
from shared import load_prepare_data, preprocess
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

# Exception handling


###################Load config.json and get path variables
config = {}
def init():
    global config
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    logger.info(f"Current working dir: {os.getcwd()}")
    with open('config.json','r') as f:
         config = json.load(f)


def model_predictions(model_pth: str, dataset: pd.DataFrame) -> (np.ndarray, np.ndarray):
    '''
    Function to get model prediction. Returns prediction array for each row of the 
    input feature dataset.
    '''
    #read the deployed model and a test dataset, calculate predictions

    # load the trained model
    model = joblib.load(model_pth)
    
    # prepare data, no dataset split 
    X, _, y, _ = preprocess(dataset)
    
    return model.predict(X), y

#################Function for training the model
def train_model(clean_data_pth: str, score_list_pth: str, model_dir: str, model_name: str) -> Score:
    '''
    Train the model with the ingested and cleaned data. The model is scored and versioned if score 
    is higher as a threshold (currently static). Model version is added to the filename.
    '''
    X_train, X_val, y_train, y_val = load_prepare_data(clean_data_pth, 0.2)

    logger.info(f"Train Features: {X_train.shape}")
    logger.info(f"Train Label: {y_train.shape}")
    logger.info(f"Validation Features: {X_val.shape}")
    logger.info(f"Validation Label: {y_val.shape}")

    logger.info(f"Validation Label: {X_train}")
    logger.info(f"Validation Label: {y_train}")
    logger.info(f"Validation Label: {X_val}")
    logger.info(f"Validation Label: {y_val}")


    #use this logistic regression for training
    lr = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                    intercept_scaling=1, l1_ratio=None, max_iter=100,
                    multi_class='auto', n_jobs=None, penalty='l2',
                    random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                    warm_start=False)
    # fit the logistic regression to your data
    model = lr.fit(X_train, y_train.ravel())

    # model validation
    score = scoring('train', model, X_val, y_val)

    # version the model if the score exceeds threshold
    if (score['score'] > 0.6):
        version = version_score(score, score_list_pth)

        #write the trained model to your workspace in a file called trainedmodel.pkl (see config.json)
        joblib.dump(model, os.path.join(model_dir, f"v{version}_{model_name}"))
        logger.info(f"Trained model with F1 score {score} saved as version: {version} in {model_dir}")
        return score, version
    logger.info(f"Trained model with F1 score {score} is too bad and NOT saved as version (cliplevel score <0.6).")
    return score, None
    
    
if __name__ == '__main__':
    try:
        init() 

        # train the model
        train_model(
            os.path.join(config['output_folder_path'], config['cleaned_data']),
            os.path.join(config['output_model_path'], config['scores']), 
            os.path.join(config['output_model_path']),
            os.path.join(config['model'])
        )
    except Exception as err:
        print(f"Training Main Error: {err}")