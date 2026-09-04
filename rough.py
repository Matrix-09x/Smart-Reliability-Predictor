import pandas as pd 
import numpy as np 
import ast
anomalies = pd.read_csv('labeled_anomalies.csv').reset_index(drop=True)

anomalies = anomalies.set_index('chan_id')

anomaly_zone = anomalies.loc['P-4','anomaly_sequences']

list_data = ast.literal_eval(anomaly_zone)
anomaly_array = np.array(list_data,dtype=int)

Total_anomaly = 0
for i in anomaly_array  :
    start , end = i[0] , i[1]   
    Total_anomaly += end - start + 1 
    

    


    