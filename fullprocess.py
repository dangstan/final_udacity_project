
import os
import training
import scoring
import deployment
import diagnostics
import reporting
import pandas as pd
import pickle

##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

prod_deployment_path = os.path.join(config['prod_deployment_path']) 
source_path = os.path.join(config['sourcedata']) 

##################Check and read new data
#first, read ingestedfiles.txt
file_names = []
with open(prod_deployment_path+'ingestedfiles.txt', 'r') as f:
    # Read each line
    for line in f:
        # Remove newline characters
        line = line.strip()
        # Append the file name to the list
        file_names.append(line)

#second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
source_files = os.listdir(os.getcwd()+source_path)

if any(x not in file_names for x in source_files):
    os.system('python3 ingestion.py')


##################Deciding whether to proceed, part 1
#if you found new data, you should proceed. otherwise, do end the process here


##################Checking for model drift
#check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
with open(prod_deployment_path+'/latestscore.txt', 'rb') as file:
    score = file.read()   
with open(prod_deployment_path+'/trainedmodel.pkl', 'rb') as file:
    model = pickle.load(file)




##################Deciding whether to proceed, part 2
#if you found model drift, you should proceed. otherwise, do end the process here



##################Re-deployment
#if you found evidence for model drift, re-run the deployment.py script

##################Diagnostics and reporting
#run diagnostics.py and reporting.py for the re-deployed model







