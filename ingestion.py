'''
Step 1 - Data Ingestion

Author: Oliver
Date: 2022, March

'''
import pandas as pd
import os
import json

# Shared items
from shared import UnsupportedFileType  

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


def check_and_append(file_name: str, append: bool, dataset: pd.DataFrame) -> pd.DataFrame:
    '''
    Scan the filename for an appropriate type and append the new dataset
    Input:
    file_name: str
    dataset: pd.DataFrame
    
    Output:
    pd.DataFrame
    '''
    logger.info(f"File name: {str(file_name)}")            
    if (file_name[-3:] == 'csv'):
        if append:
            tmp = pd.read_csv(file_name)
            appended_dataset = tmp if (dataset is None) else dataset.append(tmp) 
            logger.info(f"Appended csv file: {file_name} Size of dataset: {tmp.shape}")
    elif (file_name[-4:] == 'json'):
        if append:
            tmp = pd.read_json(file_name)
            appended_dataset = tmp if (dataset is None) else dataset.append(tmp)
            logger.info(f"Appended json file: {file_name} Size of dataset: {tmp.shape}")
    else:
        raise UnsupportedFileType("Detected unsupported file type")
    
    return appended_dataset


def check_and_merge(file_name: str, merge: bool, dataset: pd.DataFrame) -> pd.DataFrame:
    '''
    Scan the filename for an appropriate type and merge the new dataset
    Input:
    file_name: str 
    dataset: pd.DataFrame
    
    Output:
    pd.DataFrame
    '''
    logger.info(f"File name: {str(file_name)}")            
    if (file_name[-3:] == 'csv'):
        if merge:    
            tmp = pd.read_csv(file_name)
            merged_dataset = tmp if (dataset is None) else dataset.merge(tmp, how='outer')
            logger.info(f"Merged csv file: {file_name} with size {tmp.shape}")
    elif (file_name[-4:] == 'json'):
        if merge:
            tmp = pd.read_json(file_name)
            merged_dataset = tmp if (dataset is None) else dataset.merge(tmp, how='outer')
            logger.info(f"Merged json file: {file_name} with size {tmp.shape}")
    else:
        logger.error(f"Unsupported file type. Clean up the file {file_name}")
        raise UnsupportedFileType("Detected unsupported file type")
    
    return merged_dataset


def merge_directory(
        directory_name: str, 
        create_dataset: bool = False, 
        dataset: pd.DataFrame = None, 
        file_list: list = None) -> (list, pd.DataFrame):
    '''
    Recursively scan the directory for files and append / merge the data into one dataset
    '''
    logger.info(f"CurrentDirList: {os.listdir(directory_name)}")
    logger.info(f"File List: {file_list}")
    if file_list is None:
        file_list = [] 

    for item in os.listdir(directory_name):
        
        logger.info(f"Current directory: {directory_name} Item: {item}")
        item_pth = os.path.join(directory_name, item)

        if (os.path.isfile(item_pth)):
            try:    
                logger.info(f"File detected.")
                if (create_dataset):
                    dataset = check_and_merge(item_pth, create_dataset, dataset) 
                    #dataset = check_and_append(item_pth, create_dataset, dataset)
                file_list.append(item_pth) 
                logger.info(f"Data appended (create_dataset: {create_dataset}")
            except UnsupportedFileType as err:
                logger.error("Error: unsupported data type")                

        elif (os.path.isdir(item_pth)):
            logger.info("Directory detected")
            file_list, dataset = merge_directory(item_pth, create_dataset, dataset, file_list)
    
    return file_list, dataset 


def clean_dataset(dataset: pd.DataFrame):
    '''
    Clean the dataset by removing duplicates. The dataset is changed 'inplace'. 
    '''
    # drop all duplicated records
    logger.info(f"Cleaning dataset size: {dataset.shape}")
    if (dataset.duplicated().any()):
        logger.info(f"Identified duplicated records in the merged dataset (current size: {dataset.shape}). Keeping the first occurance.")
        dataset.drop_duplicates(keep='first',inplace = True)
        dataset.reset_index(drop=True, inplace=True)
        logger.info(f"Duplicated records removed. New size: {dataset.shape}")
    
    logger.info(f"Cleaned dataset size: {dataset.shape}")


def load_ingestedfiles(pth: str) -> list:
    ingested_files = []
    with open(pth, 'r') as f:
        for line in f:
            item = line[:-1]
            ingested_files.append(item)
    return ingested_files


def save_ingestedfiles(pth: str, file_list: list):
    with open(pth, 'w') as f:
        for item in file_list:
            f.write("%s\n" % item)


def save_dataset(
        dataset: pd.DataFrame, file_list: list, 
        directory_name: str, dataset_file_name: str, ingested_file_name: str):
    '''
    Save the datset to file (dataset -> directory_name/dataset_file_name). 
    Also save the list of file names the dataset was built of (file_list -> directory_name/ingested_file_name).
    '''
    # write the merged dataset to file (define the dir in config.json)
    cleaned_pth = os.path.join(directory_name, dataset_file_name)
    logger.info(f"directory: {directory_name} file: {dataset_file_name} path: {cleaned_pth}")
    dataset.to_csv(cleaned_pth, index=False)

    # write the merged files to file for further reference
    save_ingestedfiles(os.path.join(directory_name, ingested_file_name), file_list)
    

def merge_multiple_dataframe(dataset_pth: str, src_dir: str, dest_dir: str, dest_data_filename: str, dest_ingest_filename: str):
    '''
    Merge all files and corresponding dataset to one dataset, clean and save it for further processing. 
    '''
    # Merging the data
    df = pd.read_csv(dataset_pth) if (os.path.isfile(dataset_pth)) else None
    merged_files, merged_dataset = merge_directory(src_dir, True, df)
    logger.info(f"Merged files in ingest_directory: {merged_files}")

    # Cleaning the merged dataset            
    clean_dataset(merged_dataset)

    # Save the cleand dataaset
    save_dataset(
        merged_dataset, merged_files, dest_dir, dest_data_filename, dest_ingest_filename )
 

if __name__ == '__main__':
    try:
        init()
        merge_multiple_dataframe(
            os.path.join(config["output_folder_path"], config['cleaned_data']),
            os.path.join(config['input_folder_path']),
            os.path.join(config['output_folder_path']),
            config['cleaned_data'],
            config['ingested_files']
        )

    except Exception as err:
        print(f"Ingestion Main Error: {err}")
