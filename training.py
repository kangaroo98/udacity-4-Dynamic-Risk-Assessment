import os
import joblib
import json
from sklearn.linear_model import LogisticRegression

from scoring import scoring, version_score
from data import load_prepare_data
from config import Score

# initialize logging
import logging 
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

# Exception handling


###################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 


#################Function for training the model
def train_model(clean_data_pth: str, score_list_pth: str, model_dir: str):

    X_train, X_val, y_train, y_val = load_prepare_data(clean_data_pth, 0.2)

    logger.info(f"Train Features: {X_train.shape}")
    logger.info(f"Train Label: {y_train.shape}")
    logger.info(f"Validation Features: {X_val.shape}")
    logger.info(f"Validation Label: {y_val.shape}")


    #use this logistic regression for training
    lr = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                    intercept_scaling=1, l1_ratio=None, max_iter=100,
                    multi_class='auto', n_jobs=None, penalty='l2',
                    random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                    warm_start=False)
    
    #fit the logistic regression to your data
    model = lr.fit(X_train, y_train.ravel())

    # model validation
    score = scoring('train', model, X_val, y_val)

    # version the model if the score exceeds threshold
    if (score['score'] > 0.6):
        version = version_score(score, score_list_pth)

        #write the trained model to your workspace in a file called trainedmodel.pkl (see config.json)
        joblib.dump(model, os.path.join(model_dir, f"v{version}_{config['model']}"))
        logger.info(f"Trained model with F1 score {score} saved as version: {version} in {model_dir}")

    
    
if __name__ == '__main__':
    try:
        # train the model
        train_model(
            os.path.join(config['output_folder_path'], config['cleaned_data']),
            os.path.join(config['output_model_path'], config['scores']), 
            os.path.join(config['output_model_path'])
        )
    except Exception as err:
        print(f"Training Main Error: {err}")