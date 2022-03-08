'''
Step 4 - Reporting

Author: Oliver
Date: 2022, March

'''
from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import json
import os

# import create_prediction_model
# import diagnosis 
# import predict_exited_from_saved_model
from diagnostics import dataframe_summary
from diagnostics import execution_time
from diagnostics import missing_data
from diagnostics import outdated_packages_list
from training import model_predictions
from scoring import score_model
from config import Score

######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 

UPLOAD_DIRECTORY = config['prod_deployment_path']

#######################Prediction Endpoint
@app.route("/prediction", methods=['GET','POST','OPTIONS'])
def predict():        
    if request.method == 'POST':
        df = pickle.loads(request.data)
    else:
        filename = request.args.get('filename')
        df = pd.read_csv(filename)

    #call the prediction function you created in Step 3
    preds, _ = model_predictions(os.path.join(config['prod_deployment_path'], config['model']), df)

    #add return value for prediction outputs
    return jsonify(preds.tolist())

#######################Prediction Endpoint2
@app.route("/prediction2", methods=['GET','POST','OPTIONS'])
def predict2():        
    if request.method == 'POST':
        upload_pth = os.path.join(UPLOAD_DIRECTORY, "tmp.csv")
        with open(upload_pth, "wb") as f:
            f.write(request.data)
        df = pd.read_csv(upload_pth)
    else:
        filename = request.args.get('filename')
        df = pd.read_csv(filename)

    #call the prediction function you created in Step 3
    preds, _ = model_predictions(os.path.join(config['prod_deployment_path'], config['model']), df)

    #add return value for prediction outputs
    return jsonify(preds.tolist())

#######################Prediction Endpoint3
@app.route("/prediction3", methods=['GET','POST','OPTIONS'])
def predict3():        
    if request.method == 'POST':
        
        tmp_pth = os.path.join(UPLOAD_DIRECTORY, "tmp3.csv")
        content = request.files['dataset'].read()
        # save the file  
        with open(tmp_pth, "wb") as f:
            f.write(content)
        df = pd.read_csv(tmp_pth)

    else:
        filename = request.args.get('filename')
        df = pd.read_csv(filename)

    #call the prediction function you created in Step 3
    preds, _ = model_predictions(os.path.join(config['prod_deployment_path'], config['model']), df)

    #add return value for prediction outputs
    return jsonify(preds.tolist())

#######################Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def score():        
    #check the score of the deployed model
    score_obj = score_model(
            os.path.join(config['prod_deployment_path'], config['model']),
            os.path.join(config['test_data_path'], config['test_data'])
    )
    #add return value (a single F1 score number)
    return str(score_obj['score'])

#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def statistics():        
    #check means, medians, and modes for each column
    df = pd.read_csv(os.path.join(config['output_folder_path'], config['cleaned_data']))
    
    #return a list of all calculated summary statistics
    return dataframe_summary(df) 

#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def timing():        
    #check timing and percent NA values
    result_dict = {}
    result_dict['timing'] = execution_time()

    df = pd.read_csv(os.path.join(config['output_folder_path'], config['cleaned_data']))
    result_dict['na_percentage'] = missing_data(df)

    #result_dict['outdated'] = outdated_packages_list()

    #add return value for all diagnostics
    return jsonify(result_dict) 

if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
