import pandas as pd
import numpy as np
import pickle
import os
import sys
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
def score_model(model=None,data=None):
    #this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    #it should write the result to the latestscore.txt file
    if not isinstance(data, pd.DataFrame):
        data = pd.read_csv(test_data_path+'/testdata.csv')
    if model!='':
        try:
            with open(model_path+'/trained_model.pkl', 'rb') as file:
                model = pickle.load(file)
        except:
            with open('practice'+model_path+'/trained_model.pkl', 'rb') as file:
                model = pickle.load(file)

    predictions = model.predict(data.iloc[:,1:-1])
    score = f1_score(data.iloc[:,-1],predictions)
    try:
        with open(model_path+'/latestscore.txt', 'w') as file:
            file.write(str(score))
    except:
        with open('practice'+model_path+'/latestscore.txt', 'w') as file:
            file.write(str(score))

    return score


if __name__ == '__main__':
    # Check if command-line arguments are provided
    if len(sys.argv) >= 3:
        model_argument = sys.argv[1]
        data_argument = sys.argv[2]

        # Load the model and test data from arguments
        with open(model_argument, 'rb') as file:
            model = pickle.load(file)
        test_data = pd.read_csv(data_argument)

        # Call the scoring function with the provided model and data
        score_model(model, test_data)
    else:
        score_model()