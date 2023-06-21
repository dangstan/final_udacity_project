
import pandas as pd
import numpy as np
import timeit
import os
import json
import subprocess
import pickle
import requests

##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 
model_path = os.path.join(config['prod_deployment_path']) 

##################Function to get model predictions
def model_predictions(data):
    #read the deployed model and a test dataset, calculate predictions
    with open(model_path+'/trained_model.pkl', 'rb') as file:
        model = pickle.load(file)

    predictions = model.predict(data.iloc[:,1:-1]).tolist()
    return predictions #return value should be a list containing all predictions

##################Function to get summary statistics
def dataframe_summary():
    #calculate summary statistics here

    df = pd.read_csv(dataset_csv_path+'/finaldata.csv')

    # Select numeric columns only
    df_numeric = df.select_dtypes(include=['float64', 'int64'])

    # Create an empty list to store the statistics
    statistics = []

    # Calculate statistics for each numeric column
    for column in df_numeric.columns:
        mean = df_numeric[column].mean()
        median = df_numeric[column].median()
        std_dev = df_numeric[column].std()

        # Append the statistics to the list
        statistics.append({'column': column, 'mean': mean, 'median': median, 'std_dev': std_dev})

    return statistics #return value should be a list containing all summary statistics


def check_missing_data():

    df = pd.read_csv(dataset_csv_path+'/finaldata.csv')

    # Create an empty list to store the missing data percentages
    missing_data = []

    # Calculate missing data for each column
    for column in df.columns:
        na_count = df[column].isna().sum()
        total_count = df[column].shape[0]
        na_percentage = (na_count / total_count) * 100

        # Append the missing data percentage to the list
        missing_data.append({'column': column, 'na_percentage': na_percentage})

    return missing_data

##################Function to get timings
def execution_time():

    time_list = []
    #calculate timing of training.py and ingestion.py
    starttime = timeit.default_timer()
    os.system('python3 ingestion.py')
    timing=timeit.default_timer() - starttime
    time_list.append({'ingestion':timing})

    starttime = timeit.default_timer()
    os.system('python3 training.py')
    timing=timeit.default_timer() - starttime
    time_list.append({'training':timing})

    return time_list #return a list of 2 timing values in seconds

##################Function to check dependencies
def outdated_packages_list():
    
    # Getting the installed packages and their versions using pip freeze
    installed_packages = subprocess.check_output(["pip", "freeze"]).decode().splitlines()

    packages = {}

    # Splitting the package name and version and storing them in a dictionary
    for package in installed_packages:
        name, version = package.split('==')
        packages[name] = {"Current Version": version}

    # Getting the latest version of each package from PyPi
    for package in packages.keys():
        url = f"https://pypi.org/pypi/{package}/json"
        try:
            response = requests.get(url)
            data = json.loads(response.text)
            packages[package]["Latest Version"] = data["info"]["version"]
        except Exception as e:
            packages[package]["Latest Version"] = "Unable to fetch"

    return packages


if __name__ == '__main__':
    data = pd.read_csv(test_data_path+'/testdata.csv')
    model_predictions(data)
    dataframe_summary()
    check_missing_data()
    execution_time()
    outdated_packages_list()





    
