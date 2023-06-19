from flask import Flask, request
import pandas as pd
import numpy as np
import pickle
#import create_prediction_model
import diagnostics as dg 
import scoring as sc
#import predict_exited_from_saved_model
import json
import os



######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 

prediction_model = None


#######################Prediction Endpoint
@app.route("/prediction", methods=['POST','OPTIONS'])
def predict():        
    #call the prediction function you created in Step 3
    data = request.get_json(force=True)
    return dg.model_predictions(pd.DataFrame.from_dict(data)) #add return value for prediction outputs

#######################Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def scoring():        
    #check the score of the deployed model
    score = sc.score_model()
    return str(score) #add return value (a single F1 score number)

#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def summary():        
    summary = dg.dataframe_summary() #check means, medians, and modes for each column
    return summary #return a list of all calculated summary statistics

#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diag():        
    results = {}

    results['missing_data'] = dg.check_missing_data()
    results['timings'] = dg.execution_time()
    results['outdated_pkg_list'] = dg.outdated_packages_list()

    return results #add return value for all diagnostics

if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
