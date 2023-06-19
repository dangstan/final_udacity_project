import requests
import json
import os
import pandas as pd

##################Load config.json and correct path variable
with open('config.json','r') as f:
    config = json.load(f) 

# Specify a URL that resolves to your workspace
BASE_URL = "http://127.0.0.1:8000/"

# Specify the endpoint URLs
PREDICTION_URL = BASE_URL + "prediction"
SCORING_URL = BASE_URL + "scoring"
SUMMARYSTATS_URL = BASE_URL + "summarystats"
DIAGNOSTICS_URL = BASE_URL + "diagnostics"

# Load the test data for the prediction endpoint
test_data_path = os.path.join(config['test_data_path']) 
test_data = pd.read_csv(test_data_path+'/testdata.csv')

# Call each API endpoint and store the responses
response1 = requests.post(PREDICTION_URL, json=test_data.to_dict())
response2 = requests.get(SCORING_URL)
response3 = requests.get(SUMMARYSTATS_URL)
response4 = requests.get(DIAGNOSTICS_URL)

# Combine all API responses
responses = {
    'prediction': response1.json(),
    'scoring': response2.json(),
    'summarystats': response3.json(),
    'diagnostics': response4.json(),
}

# Write the responses to a file
with open('config.json','r') as f:
    config = json.load(f)

output_file_path = os.path.join(config['output_model_path'], 'apireturns.txt')

with open(output_file_path, 'w') as f:
    f.write(json.dumps(responses, indent=4))



