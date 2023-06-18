from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import json



#################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 
model_path = os.path.join(config['output_model_path']) 


#################Function for model scoring
def score_model():
    #this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    #it should write the result to the latestscore.txt file
    test_data = pd.read_csv(test_data_path+'/testdata.csv')
    with open(model_path+'/trainedmodel.pkl', 'rb') as file:
        model = pickle.load(file)

    predictions = model.predict(test_data.iloc[:,:-1])
    score = f1_score(test_data.iloc[:,-1],predictions)
    with open(model_path+'/latestscore.txt', 'wb') as file:
        file.write(str(score))