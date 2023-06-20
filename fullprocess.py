import os
import json
import pandas as pd
import pickle
from scoring import score_model

##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

prod_deployment_path = config['prod_deployment_path']
output_folder_path = config['output_folder_path']
source_path = config['input_folder_path']

if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

def pipeline():

    ##################Check and read new data
    #first, read ingestedfiles.txt
    file_names = []
    with open(prod_deployment_path+'/ingestedfiles.txt', 'r') as f:
        # Read each line
        for line in f:
            # Remove newline characters
            line = line.strip()
            # Append the file name to the list
            file_names.append(line)

    #second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
    source_files = os.listdir(os.getcwd()+'/'+source_path)

    if any(x not in file_names for x in source_files):
        os.system('python3 ingestion.py')
    else:
        print(file_names,source_files)
        return


    ##################Deciding whether to proceed, part 1
    #if you found new data, you should proceed. otherwise, do end the process here


    ##################Checking for model drift
    #check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
    with open(prod_deployment_path+'/latestscore.txt', 'rb') as file:
        old_score = file.read()
    with open(prod_deployment_path+'/trained_model.pkl', 'rb') as file:
        model = pickle.load(file)

    data = pd.read_csv(output_folder_path+'/finaldata.csv')

    new_score = score_model(model,data)

    ##################Deciding whether to proceed, part 2
    #if you found model drift, you should proceed. otherwise, do end the process here
    print(new_score,old_score)
    if new_score > float(old_score):
        print(new_score,old_score)
        return

    ##################Re-deployment
    #if you found evidence for model drift, re-run the deployment.py script
    os.system('python3 training.py')
    os.system('python3 scoring.py')
    os.system('python3 deployment.py')

    ##################Diagnostics and reporting
    #run diagnostics.py and reporting.py for the re-deployed model
    os.system('python3 apicalls.py')
    os.system('python3 diagnostics.py')
    os.system('python3 reporting.py')

if __name__ == '__main__':
    pipeline()