import requests
import os
import json
import pandas as pd
import pickle

# initialize logging
import logging 
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

# Exception handling

#Specify a URL that resolves to your workspace
URL = "http://127.0.0.1:8000/"

##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

def write_json(response, pth):
    with open(pth,'w') as f:
        json.dump(response, f)



# #Call each API endpoint and store the responses

# Prediction - POST
df = pd.read_csv(os.path.join(config['test_data_path'], "testdata.csv"))
response_post = requests.post(f"{URL}prediction", data=pickle.dumps(df))
logger.info(f"Prediction2: {response_post.json()}")

# Prediction - POST 2
dataset_name = os.path.join(config['test_data_path'], "testdata.csv")
with open(dataset_name, 'rb') as f:
    content = f.read()
response_post = requests.post(f"{URL}prediction2", data=content)
logger.info(f"Prediction: {response_post.json()}")

# Prediction - POST 3
dataset_name = os.path.join(config['test_data_path'], "testdata.csv")
with open(dataset_name, 'rb') as f:
    files = {'dataset': f}
    response_post = requests.post(f"{URL}prediction3",files=files)
logger.info(f"Prediction3: {response_post.json()}")

# Prediction - GET
response1 = requests.get(f"{URL}prediction?filename=testdata/testdata.csv").json()
logger.info(f"Prediction: {response1}")
write_json(response1, os.path.join(config['output_model_path'], "apireturns1.txt"))

# Scoring - GET
response2 = requests.get(f"{URL}scoring").json()
logger.info(f"Scoring: {response2}")
write_json(response2, os.path.join(config['output_model_path'], "apireturns2.txt"))

# Statistics - GET
response3 = requests.get(f"{URL}summarystats").json()
logger.info(f"Statistics: {response3} response: {type(response3)}")
write_json(response3, os.path.join(config['output_model_path'], "apireturns3.txt"))

# Diagnositc - GET
response4 = requests.get(f"{URL}diagnostics").json()
logger.info(f"Diagnostic: {response4}")
write_json(response4, os.path.join(config['output_model_path'], "apireturns4.txt"))

# combine all API responses
responses = {'r1_prediction':response1, 'r2_scoring': response2, 'r3_summarystats': response3, 'r4_diagnostic': response4}

# write the responses to your workspace
write_json(responses, os.path.join(config['output_model_path'], "apireturns.txt"))

