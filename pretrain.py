

from V3 import Linear_Regression ,Random_forest ,Lstm_channel  ,anomalies

import json

import os

channels = anomalies.index.unique().to_list()

# print(anomalies.loc["P-2"])
# print(anomalies.index[anomalies.index == "p-2"])

calibration = {}

for i in channels :
    print(f'Pretraining {i}')
    calibration[i] = {}

    calibration[i]['linear_regression'] = Linear_Regression(i ,mode= 'pretrain')
    calibration[i]['random_forest'] = Random_forest(i, mode= 'pretrain')
    calibration[i]['lstm'] = Lstm_channel(i,mode='pretrain')



os.makedirs("models",exist_ok=True)

with open("models/calibration.json","w") as file :
    json.dump(calibration,file,indent=4)


print("pretraining complete")