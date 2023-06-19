import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json
import shutil



##################Load config.json and correct path variable
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
prod_deployment_path = os.path.join(config['prod_deployment_path']) 
model_path = os.path.join(config['output_model_path']) 


if not os.path.exists(prod_deployment_path):
    os.makedirs(prod_deployment_path)

####################function for deployment
def store_model_into_pickle():
    #copy the latest pickle file, the latestscore.txt value, and the ingestfiles.txt file into the deployment directory

#    with open(model_path+'/latestscore.txt', 'rb') as file:
#        score = file.read()    
#    with open(model_path+'/trainedmodel.pkl', 'rb') as file:
#        model = pickle.load(file) 

#    file_names = []   
#    with open(dataset_csv_path+'ingestedfiles.txt', 'r') as f:
#    # Read each line
#    for line in f:
#        # Remove newline characters
#        line = line.strip()
#        # Append the file name to the list
#        file_names.append(line)

    shutil.copy(model_path+'/latestscore.txt', prod_deployment_path+'/latestscore.txt')
    shutil.copy(model_path+'/trained_model.pkl', prod_deployment_path+'/trained_model.pkl')
    shutil.copy(dataset_csv_path+'/ingestedfiles.txt', prod_deployment_path+'/ingestedfiles.txt')
        

if __name__ == '__main__':
    store_model_into_pickle()