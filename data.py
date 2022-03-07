import pandas as pd
from sklearn.model_selection import train_test_split

# initialize logging
import logging 
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

# Exception handling


def load_prepare_data(csv_pth: str, test_data_split: float = 0) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):

    # Actually the cleaned_data should be split into train/validate/test data. But since test data 
    # is provided, I assumed that the initially ingested data is supposed to be the training/validation data only,
    # testing is done by using the provided testdata.
    # load the test dataset
    df = pd.read_csv(csv_pth)

    # preprocess the test dataset and return features and labels
    X_train, X_test, y_train, y_test = preprocess(df, test_data_split)

    return X_train, X_test, y_train, y_test


def preprocess(df: pd.DataFrame, test_data_split: float = 0) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):

    # Feature and label extraction
    y = df['exited'].values.reshape(-1,1)
    X = df[['lastmonth_activity','lastyear_activity','number_of_employees']].values.reshape(-1,3)
    
    if (test_data_split > 0):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_data_split, random_state=42)
        return X_train, X_test, y_train, y_test
    
    return X, None, y, None
