# import pandas as pd 
# import numpy as np 
# import ast
# anomalies = pd.read_csv('labeled_anomalies.csv').reset_index(drop=True)

# anomalies = anomalies.set_index('chan_id')

# anomaly_zone = anomalies.loc['P-4','anomaly_sequences']

# list_data = ast.literal_eval(anomaly_zone)
# anomaly_array = np.array(list_data,dtype=int)

# Total_anomaly = 0
# for i in anomaly_array  :
#     start , end = i[0] , i[1]
#     for x in range(start , end) :
#         print(x)
    
    
# for anomaly in anomalies :
#     start ,end = anomaly[0],anomaly[1]


for i in range(1,5) :
    
    for j in range(1,i+1) :
        print(j+1,end=" ")
    print()