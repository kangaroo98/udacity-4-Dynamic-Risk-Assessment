import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
import typing


# initialize logging
import logging 
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

# Exception handling
from config import UnsupportedFileType


#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']
cleaned_data = config['cleaned_data']
ingested_files = config['ingested_files']



def append_dataset(file_name: str, dataset: pd.DataFrame) -> pd.DataFrame:
    
    logger.info(f"File name: {str(file_name)}")            
    if (file_name[-3:] == 'csv'):
        logger.info(f"Append csv file: {file_name}")
        tmp = pd.read_csv(file_name)
        logger.info(f"Append dataset: {tmp.shape}")
        appended_dataset = tmp if dataset.empty else dataset.append(tmp)
    elif (file_name[-4:] == 'json'):
        logger.info(f"Append json file: {file_name}")
        tmp = pd.read_json(file_name)
        logger.info(f"Append dataset: {tmp.shape}")
        appended_dataset = tmp if dataset.empty else dataset.append(tmp)
    else:
        raise UnsupportedFileType("Detected unsupported file type")
    
    return appended_dataset


def merge_dataset(file_name: str, dataset: pd.DataFrame) -> pd.DataFrame:
    
    logger.info(f"File name: {str(file_name)}")            
    if (file_name[-3:] == 'csv'):
        logger.info(f"Merge csv file: {file_name}")
        tmp = pd.read_csv(file_name)
        logger.info(f"Merged dataset: {tmp.shape}")
        merged_dataset = tmp if dataset.empty else dataset.merge(tmp, how='outer')
    elif (file_name[-4:] == 'json'):
        logger.info(f"Merge json file: {file_name}")
        tmp = pd.read_json(file_name)
        logger.info(f"Merged dataset: {tmp.shape}")
        merged_dataset = tmp if dataset.empty else dataset.merge(tmp, how='outer')
    else:
        raise UnsupportedFileType("Detected unsupported file type")
    
    return merged_dataset


def ingest_directory(directory_name: str, dataset: pd.DataFrame = pd.DataFrame(), file_list: list = []) -> (pd.DataFrame,list):
    
    logger.info(f"CurrentDirList: {os.listdir(directory_name)}")

    for item in os.listdir(directory_name):
        
        logger.info(f"Current directory: {directory_name} Item: {item}")
        item_pth = os.path.join(directory_name, item)

        if (os.path.isfile(item_pth)):
            try:    
                logger.info("File detected")
                dataset = merge_dataset(item_pth, dataset)
                #dataset = append_dataset(item_pth, dataset)
                file_list.append(item_pth) 
                logger.info(f"Data appended, new dataset size: {dataset.shape}")
            except UnsupportedFileType as err:
                logger.error("Error: unsupported data type")                

        elif (os.path.isdir(item_pth)):
            logger.info("Directory detected")
            dataset, file_list = ingest_directory(item_pth, dataset, file_list)
           
    
    return dataset, file_list


def clean_dataset(dataset: pd.DataFrame):
    
    # drop all duplicated records
    logger.info(f"Cleaning dataset size: {dataset.shape}")
    if (dataset.duplicated().any()):
        logger.info(f"Identified duplicated records in the merged dataset (current size: {dataset.shape}). Keeping the first occurance.")
        dataset.drop_duplicates(keep='first',inplace = True)
        dataset.reset_index(drop=True, inplace=True)
        logger.info(f"Duplicated records removed. New size: {dataset.shape}")
    
    logger.info(f"Cleaned dataset size: {dataset.shape}")


def save_dataset(
        dataset: pd.DataFrame, file_list: list, 
        directory_name: str, dataset_file_name: str, ingested_file_name: str):
    
    # write the merged dataset to file (define the dir in config.json)
    cleaned_pth = os.path.join(directory_name, dataset_file_name)
    logger.info(f"directory: {directory_name} file: {dataset_file_name} path: {cleaned_pth}")
    dataset.to_csv(cleaned_pth, index=False)

    # write the merged files to file for further reference
    with open(os.path.join(directory_name, ingested_file_name), 'w') as f:
        for item in file_list:
            f.write("%s\n" % item)


def merge_multiple_dataframe():
    
    # Merging the data 
    working_dir = "." #os.getcwd()
    merged_dataset, merged_files = ingest_directory(os.path.join(working_dir, input_folder_path))
    logger.info(f"Initial working directory '{working_dir}' and folder {input_folder_path} are merged.")
    logger.info(f"Merged files: {merged_files}")

    # Cleaning the merged dataset            
    clean_dataset(merged_dataset)

    # Save the cleand dataaset
    save_dataset(
        merged_dataset, merged_files,
        os.path.join(working_dir, output_folder_path),
        cleaned_data, ingested_files
    )
 

if __name__ == '__main__':
    try:
        merge_multiple_dataframe()
    except Exception as err:
        print(f"Ingestion Main Error: {err}")
