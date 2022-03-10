'''
Step 2 - Training, Scoring, and Deploying an ML Model

Author: Oliver
Date: 2022, March

'''
import os
import json
from datetime import datetime
import pandas as pd
from pydantic import BaseModel
from typing import Literal, Union 
from shtab import Optional

from sklearn.model_selection import train_test_split

# initialize logging
import logging 
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


# pydantic model
METRIC = Literal['F1', 'MAE', 'SSE']
MODE = Literal['train','test']

class Score(BaseModel):
    version: int = None 
    mode: MODE
    timestamp: datetime
    metric: METRIC
    score: float
    
    def __getitem__(self, item):
        return getattr(self, item)

# # Exception handling - define user-defined exceptions
class AppError(Exception):
    """Base class for other exceptions"""
    pass

class UnsupportedFileType(AppError):
    """Base class for other exceptions"""
    pass

class FileInvalid(AppError):
    """Base class for other exceptions"""
    pass

class DeploymentFailed(AppError):
    """Base class for other exceptions"""
    pass


# data preparation
def load_prepare_data(csv_pth: str, test_data_split: float = 0) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
    '''
    Load the dataset from csv_pth. If the test_data_split attribute is not >0, the dataset is preprocessed 
    and returned as a whole. 
    '''
    # Actually the cleaned_data should be split into train/validate/test data. But since test data 
    # is provided, I assumed that the initially ingested data is supposed to be the training/validation data only,
    # testing is done by using the provided testdata.

    # load the test dataset
    df = pd.read_csv(csv_pth)

    # preprocess the test dataset and return features and labels
    X_train, X_test, y_train, y_test = preprocess(df, test_data_split)

    return X_train, X_test, y_train, y_test


def preprocess(df: pd.DataFrame, test_data_split: float = 0) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
    '''
    Feature extracting and train and validation/test dataset split
    '''
    # Feature and label extraction
    y = df['exited'].values.reshape(-1,1)
    X = df[['lastmonth_activity','lastyear_activity','number_of_employees']].values.reshape(-1,3)
    
    if (test_data_split > 0):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_data_split, random_state=42)
        return X_train, X_test, y_train, y_test
    
    return X, None, y, None
