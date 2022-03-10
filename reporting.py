'''
Step 4 - Reporting

Author: Oliver
Date: 2022, March

'''
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

from diagnostics import model_predictions

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

###############Load config.json and get path variables
config = {}
def init():
    global config
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    logger.info(f"Current working dir: {os.getcwd()}")
    with open('config.json','r') as f:
         config = json.load(f)


def plot_confusion_matrix(save_pth: str, act: np.array, preds: np.array, ):
    '''
    Plot/save a confusion matrix.
    '''
    # plot
    cm = confusion_matrix(act, preds)
    fig = plt.figure()

    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)    
    fig.colorbar(cax)
    labels = ['False', 'True']
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.title('Confusion matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(save_pth)


def reporting(model_pth: str, data_pth: str, plot_pth: str):
    '''
    Create a confusion matrix based on a dataset stored in data_path. Plot saved in model_path.
    '''
    #calculate a confusion matrix using the test data and the deployed model
    #write the confusion matrix to the workspace
    df = pd.read_csv(data_pth)
    preds, act = model_predictions(model_pth, df)
    plot_confusion_matrix(plot_pth, act, preds)
    logger.info(f"Predictions: {preds}, Actuals: {act}")


if __name__ == '__main__':
    try:
        init()
        reporting(
            os.path.join(config['prod_deployment_path'], config['model']),
            os.path.join(config['test_data_path'], config['test_data']),
            os.path.join(config['output_model_path'], config['confusionmatrix'])
        )
    except Exception as err:
        print(f"Reporting Main Error: {err}")   
