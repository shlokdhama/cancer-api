import pandas as pd
import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data = pd.read_csv(os.path.join(BASE_DIR, 'data.csv'), index_col=0)
patient = data.iloc[0].tolist()

with open('test_patient.json', 'w') as f:
    json.dump({"expression_values": patient}, f)

print("Saved to test_patient.json")