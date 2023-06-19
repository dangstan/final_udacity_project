import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import diagnostics as dg



###############Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config['test_data_path'])  
model_path = os.path.join(config['output_model_path']) 



##############Function for reporting
def score_model():
    #calculate a confusion matrix using the test data and the deployed model
    #write the confusion matrix to the workspace
    test_data = pd.read_csv(test_data_path+'/testdata.csv')
    preds = dg.model_predictions(test_data)
    y_true = test_data.iloc[:,-1]
    
    # Create the confusion matrix
    cm = confusion_matrix(y_true, preds)

    # Use Seaborn's heatmap for a more visually appealing confusion matrix
    plt.figure(figsize=(10,7))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('Truth')

    # Save the plot
    plt.savefig(f'{model_path}/confusionmatrix.png')



if __name__ == '__main__':
    score_model()
