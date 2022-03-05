import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from typing import Optional

# initialize logging
import logging 
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

# Exception handling
from error import UnsupportedFileType


#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']


def merge_dataset(file_name: str, dataset: pd.DataFrame) -> pd.DataFrame:
    
    logger.info(f"File name: {str(file_name)}")            
    if (file_name[-3:] == 'csv'):
        logger.info(f"Merge csv file: {file_name}")
        merged_dataset = pd.read_csv(file_name) if dataset.empty else dataset.merge(pd.read_csv(file_name), how='outer')
        logger.info(f"Merged dataset: {merged_dataset.shape}")
    elif (file_name[-4:] == 'json'):
        logger.info(f"Merge json file: {file_name}")
        merged_dataset = pd.read_json(file_name) if dataset.empty else dataset.merge(pd.read_json(file_name), how='outer')
        logger.info(f"Merged dataset: {merged_dataset.shape}")
    else:
        raise UnsupportedFileType("Detected unsupported file type")
    
    return merged_dataset


def ingest_directory(directory_name: str, dataset: pd.DataFrame = pd.DataFrame(), file_list: list = []) -> (pd.DataFrame,list):
    logger.info(f"CurrentDirList: {os.listdir(directory_name)}")
    os.chdir(directory_name)

    for item in os.listdir(directory_name):
        logger.info(f"Current directroy: {directory_name} Item: {item}")
        if (os.path.isfile(item)):

            try:    
                logger.info("File detected")
                dataset = merge_dataset(item, dataset)
                file_list.append(os.path.join(directory_name,item)) 
                logger.info(f"Data appended, new dataset size: {dataset.shape}")
            except UnsupportedFileType as err:
                logger.error("Error: unsupported data type")                

        elif (os.path.isdir(item)):
            logger.info("Directory detected")
            dataset, file_list = ingest_directory(os.path.join(directory_name, item), dataset, file_list)
            os.chdir(directory_name)
    
    return dataset, file_list


def clean_dataset(dataset: pd.DataFrame):
    
    # drop all duplicated records
    logger.info(f"Dataset: {dataset}")
    if (dataset.duplicated().any()):
        logger.info(f"Identified duplicated records in the merged dataset (current size: {dataset.shape}). Keeping the first occurance.")
        dataset.drop_duplicates(keep='first',inplace = True)
        dataset.reset_index(drop=True, inplace=True)
        logger.info(f"Duplicated records removed. New size: {dataset.shape}")
    
    logger.info("clean_dataset() done.")


def save_dataset(
        dataset: pd.DataFrame, file_list: list, 
        directory_name: str, dataset_file_name: str, ingested_file_name: str):
    
    # write the merged dataset to file (define the dir in config.json)
    cleaned_pth = os.path.join(directory_name, dataset_file_name)
    logger.info(f"dir: {directory_name} file: {dataset_file_name} path: {cleaned_pth}")
    dataset.to_csv(cleaned_pth, index=False)

    # write the merged files to file for further reference
    #ast.file_list

def merge_multiple_dataframe():
    
    # Merging the data 
    initial_wd = os.getcwd()
    merged_dataset, merged_files = ingest_directory(os.path.join(initial_wd, input_folder_path))
    logger.info(f"Initial working directory: {initial_wd} and folder {input_folder_path} are merged.")
    logger.info(f"Merged files: {merged_files}")

    # Cleaning the merged dataset            
    clean_dataset(merged_dataset)

    # Save the cleand dataaset
    save_dataset(
        merged_dataset, merged_files,
        os.path.join(initial_wd, output_folder_path),
        "finaldata.csv", "ingestedfiles.txt"
    )


if __name__ == '__main__':
    try:
        merge_multiple_dataframe()
    except Exception as err:
        print(f"Ingestion Error: {err}")
