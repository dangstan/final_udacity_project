import pandas as pd
import numpy as np
import os
import json
from datetime import datetime




#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']

if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

#############Function for data ingestion
def merge_multiple_dataframe():
    path = os.getcwd()+'/'+input_folder_path
    filenames = os.listdir(path)
    df_ = pd.DataFrame()
    
    for each_filename in filenames:
        if '.csv' not in each_filename:
            continue
        dfl = pd.read_csv(path+'/'+each_filename)
        df_=pd.concat([df_,dfl], ignore_index=True).drop_duplicates() 

    df_.to_csv(output_folder_path+'/finaldata.csv', index=False) 
    with open(output_folder_path+'/ingestedfiles.txt', 'w') as f:
        for file in filenames:
            f.write(f"{file}\n")

if __name__ == '__main__':
    merge_multiple_dataframe()
