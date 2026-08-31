
import numpy  as np 
import pandas as pd 
import math as m  
import ast

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score


np.set_printoptions(threshold=np.inf)

anomalies = pd.read_csv('labeled_anomalies.csv')
Training = np.load('data\data/train/E-11.npy')
Training_signal = Training[:,0]



window_size = 10
input = []
output = []

for i,value in enumerate(Training_signal) :
    if i < len(Training_signal) - window_size : 
        input.append(Training_signal[i: i+window_size])
        output.append(Training_signal[i+window_size])

input = np.array(input)
output = np.array(output)

Testing  =  np.load('data\data/test/E-11.npy')
testing_signal =  Testing[:,0]


x_test = []
y_test = []




for i ,value in enumerate(testing_signal) :
    if i < len(testing_signal) - window_size :
        x_test.append(testing_signal[i:i+window_size])
        y_test.append(testing_signal[i+window_size])


x_test = np.array(x_test)
y_test = np.array(y_test)


anomalies = anomalies.set_index('chan_id')

anomaly_zone = anomalies.loc['P-7','anomaly_sequences']

list_data = ast.literal_eval(anomaly_zone)
anomaly_array  = np.array(list_data,dtype= int)

print(type(anomaly_array))
print(anomaly_array)
# print(anomaly_zone[0])
# print(anomaly_array[0,1])
# Total_anoamly = anomaly_array[0,1] - anomaly_array[0,0]  +  1
# print(Total_anoamly)

# model = LinearRegression()

# model_2 = RandomForestRegressor()


# def precision_recall(model) : 


#     model.fit(input,output)

#     predictions = model.predict(x_test)


#     error = np.abs(y_test - predictions)

#     error_mean = np.mean(error)
#     # print(error_mean)
#     error_std = np.std(error)
#     # print(error_std)
#     thrshold = error_mean + 0.5  *  error_std 
#     # print(thrshold)


#     anomaly_indices = np.where(error > thrshold)[0]
#     anomaly_indices = anomaly_indices + window_size
#     # print(anomaly_indices)

#     tp =  len(anomaly_indices[(anomaly_indices >= anomaly_array[0,0]) & (anomaly_indices <= anomaly_array[0,1])])
#     fp = len(anomaly_indices[(anomaly_indices < anomaly_array[0,0]) | (anomaly_indices > anomaly_array[0,1])])
#     fn = Total_anoamly - tp
#     tn = len(y_test) - tp - fp - fn

#     precision  = tp/(tp+fp)
#     recall = tp/(tp+fn)
   
#     return precision , recall
    



# precision_l , recall_l = precision_recall(model)
# print(f'Linear regression model | Precision : {precision_l} and Recall : {recall_l}')
# precision_r , recall_r  = precision_recall(model_2)
# print(f'Random Forest model | Precision : { precision_r} and Recall : {recall_r}')











# print(anomalies.loc[9])



# r2 = r2_score(y_test,predictions)
# print(r2)





# print(predictions[:100])
# print(y_test[:100])
# Changes = np.diff(Telemetry)




# for i in Telemetry :

# magnitude = np.abs(Changes)





# mag_mean = np.mean(magnitude)
# # print(mag_mean)
# mag_std = np.std(magnitude)
# # print(mag_std)

# anomaly_values = magnitude[magnitude > mag_mean + 3 *mag_std]
# indices = np.where(np.isin(magnitude,anomaly_values))[0]
# # print(indices)
# print(anomalies.loc[9])

# indices_inside = indices[(indices >= 5400) & (indices <= 6022)]
# indices_inside_magn = magnitude[indices_inside]
# indices_outside = indices[(indices < 5400) | (indices > 6022) ]
# indices_outside_magn = magnitude[(indices_outside)]

# print(f'Inside anomaly values mean | {np.mean(indices_inside_magn)} and maximum value | {np.max(indices_inside_magn)}')
# print(f'Outside anomaly values mean | {np.mean(indices_outside_magn)} and maximum value | {np.max(indices_outside_magn)}')


# print(Changes[3400:3900])

# zero_count = []
# zero = 0

# zero_start_index = []
# zero_end_index = []
# for index,i in enumerate(Changes) :
#     if i == 0 :
#         zero += 1

#         if zero == 1 :
#             zero_start_index.append(index)

#         if index < len(Changes)-1 :
#            if Changes[index+1] != 0  :
#                zero_end_index.append(index)

#         else :
#             if Changes[index] == 0:
#                zero_end_index.append(index)

            

        

#     else :
#         zero = 0

    
#     zero_count.append(zero)
        
# zero_count = np.array(zero_count)
# zero_start_index = np.array(zero_start_index)
# zero_end_index = np.array(zero_end_index)


# zero_lenth = np.array(zero_end_index) - np.array(zero_start_index) + 1




# window_size = 50 

# windows = np.lib.stride_tricks.sliding_window_view(Telemetry,window_shape= window_size)

# rolling_range = windows.max(axis=1) - windows.min(axis = 1)

# time_index = np.arange(len(rolling_range)) + window_size//2

# row = anomalies.loc[9]
# anomaly_zone = eval(row['anomaly_sequences'])

# plt.figure(figsize=(12,4))
# plt.plot(Telemetry, linewidth = 0.8 , color = 'steelblue')


# for start,end in anomaly_zone :
#     plt.axvspan(start,end, color = 'red' ,alpha = 0.3)
#     print(start,end)

# plt.xlabel('Time step')
# plt.ylabel('Zero_count check for anomaly')
# plt.tight_layout()

# plt.show()
















# print(zero_count)
# print(zero_start_index[zero_start_index >= 5000 and zero_start_index <=5051])

# print(zero_end_index)
# print(zero_lenth)



# sorted_zero= np.sort(zero_lenth)[::-1]
# sorted_zero_index = np.argsort(zero_lenth)[::-1]
   



# print(np.argmax(magnitude))
# print(np.max(magnitude))
# print(anomalies.loc[4])

# print(Changes)

    

# print(zero_count[2149 : 2350])
# print(zero_count[2000 : 2500])
# print(zero_count[4536 : 4844])
# print(np.max(zero_count[4536 : 4844]))
# print(zero_count[3539 : 3779])
# print(np.max(zero_count[3539: 3779]))

# print(np.argmax(zero_count))
# print(np.max(zero_count))

# print(Telemetry[2139:2159])
# print(Changes[4000:8500])
# print(Changes[2000:2400])
# print(Changes[2149:2349])
# print(Changes[4400:4900])


# print(Changes[:100])


# print(zero_count[2000:3000])





        








# p = float(input('Enter the probability of failure : '))
# n = int(input('Enter the number of request : '))
# rng = np.random.default_rng()

# Accuracy_list = []



# def stimualtion_request(n ,p):
#    result_list = []
#    for i in range(1,n+1) :
      
#       number = rng.random()

#       if number < p  : 
#          outcome = 0 # 0  represent failure broo as 0 is 0 and its failure
#          result_list.append(outcome)

#       else :

#          outcome = 1 # 1 represent success like no particualar reason just prob and stat things
#          result_list.append(outcome)

#    return result_list

# result = stimualtion_request(n,p)


# failure_cause = []

# causes = ['Db','Network','Server']
# for i in result :
#    if i == 0 :
#       cause = rng.choice(causes)
#       failure_cause.append(cause)

# evidences = []





# for i in failure_cause :
#    if i  == 'Db' :
#       db_num = int(rng.integers(70,101))
#       n_num = int(rng.integers(10,131))
#       S_num = int(rng.integers(30,76))
#       evidences.append({'db load' : db_num , 'network latency' : n_num , 'server load' : S_num })

#    elif i == 'Network' :
#       db_num = int(rng.integers(30,76))
#       n_num = int(rng.integers(125,501))
#       S_num = int(rng.integers(30,76))
#       evidences.append({'db load' : db_num , 'network latency' : n_num , 'server load' : S_num })

#    else :
#       db_num = int(rng.integers(30,76))
#       n_num = int(rng.integers(10,131))
#       S_num = int(rng.integers(70,101))
#       evidences.append({'db load' : db_num , 'network latency' : n_num , 'server load' : S_num })




   





# ranges = [[(70,100) , (10,130) , (30,75)] , [(30 ,75)  , (125,500) , (30,75)] , [(30,75) , (10,130) , (70,100)]]



# Causes_list = []
# for  evidence in evidences :
   
#    contributions = {}
#    total_evidence = 0
#    for cause_index , rang in enumerate(ranges) :
#       likelihood = 1
#       for index, r in  enumerate(rang):
#          denominator = 0
#          if index == 0 :
#             min_r , max_r = r[0] , r[1]
#             denominator = (max_r - min_r) + 1
#             if evidence['db load'] >= min_r and evidence['db load'] <= max_r :
               
#                likelihood *= 1/denominator 
               

#             else :
#                likelihood *= 0
               

            
#          if index == 1 :
#             min_r , max_r = r[0] , r[1]
#             denominator = (max_r - min_r) + 1
#             if evidence['network latency'] >= min_r and evidence['network latency'] <= max_r :
#                likelihood *= 1/denominator 
               

#             else :
#                likelihood *= 0
               
         
            
#          if index == 2 :
#             min_r , max_r = r[0] , r[1]
#             denominator = (max_r - min_r) + 1
#             if evidence['server load'] >= min_r and evidence['server load'] <= max_r :
#                likelihood *= 1/denominator 
               

#             else :
#                likelihood *= 0
   
#       contribution = likelihood * 1/3

#       if cause_index == 0 :
#          contributions['Db'] = contribution 

#       elif cause_index == 1 :
#          contributions['Network'] = contribution 

#       elif cause_index == 2 :
#          contributions['Server'] = contribution 

   
#       total_evidence += contribution
#    if total_evidence != 0 :
#       db_prob = contributions['Db']/total_evidence 
#       Net_prob = contributions['Network']/total_evidence 
#       Server_prob = contributions['Server']/total_evidence 

#       if db_prob > Net_prob and db_prob > Server_prob :
#          Causes_list.append("Db")

#       elif Net_prob >  db_prob and Net_prob > Server_prob :
#          Causes_list.append('Network')
#       elif Server_prob >  db_prob and  Server_prob > Net_prob :
#          Causes_list.append('Server')

#       else :
#          Causes_list.append('Tied')

#    else :
#       Causes_list.append('Unknown')
      
# # print(failure_cause)
# # print(Causes_list)

# Db_correceted = 0
# Network_corrected = 0
# Server_corrected = 0
# Db_Server_incorrect = 0
# Db_Network_incorrect = 0
# Server_Db_incorrected = 0
# Server_Network_incorrect = 0
# Network_Db_incorrected = 0
# Network_Server_incorrected = 0
# correct_prediction = 0 


# for i, cause in enumerate(failure_cause) :
#    if cause == Causes_list[i] :
#       correct_prediction += 1
#       if cause == 'Db' :
#          Db_correceted += 1
#       elif cause == 'Network' :
#          Network_corrected += 1

#       else :
#          Server_corrected += 1


#    else :
#       if cause == 'Db' :
#          if Causes_list[i] == 'Server' :
#             Db_Server_incorrect += 1

#          elif Causes_list[i] == 'Network' :
#             Db_Network_incorrect += 1

#       elif cause == 'Server' :
#          if Causes_list[i] ==  'Db' :
#             Server_Db_incorrected += 1

#          elif Causes_list[i] == 'Network' :
#             Server_Network_incorrect += 1


#       else :
#          if Causes_list[i] == 'Db' :
#             Network_Db_incorrected += 1

#          elif Causes_list[i] == 'Server' :
#             Network_Server_incorrected += 1


# Accuracy = (correct_prediction /len(failure_cause)) * 100 

# confusion_matrix  = {
#    'Db' : {'Db correct prediction' : Db_correceted ,
#             'Server incorrect prediction for Db' :  Db_Server_incorrect ,
#             'Network incorrected prediction for Db ' : Db_Network_incorrect
#             } ,

#    'Server' : {'Server correct prediction' : Server_corrected ,
#                'Db incorrect prediction for Server' : Server_Db_incorrected ,
#                'Network incorrect prediction for Server ' : Server_Network_incorrect 
#                } ,

#    'Network' : {'Network correct prediction' : Network_corrected  , 
#                'Db incorrect prediction for Network' : Network_Db_incorrected , 
#                'Server incorrect prediction for Network' : Network_Server_incorrected
               
#    }


# }





# if n > 0 and p > 0 :

#    print('SYSTEM REPORT')
#    print(f'Request | {n}')
#    print(f'Failure Proabability | {p}')



#    expected_mean = n*p

#    observed_failure = len(failure_cause)
   


#    difference = observed_failure - expected_mean


#    standard_deviation = m.sqrt(expected_mean)


#    if  expected_mean < 10  :

#       total_prob = 0 

#       for k in range(0 , observed_failure) :
#          prob = (expected_mean**k * m.e**(-expected_mean))/ m.factorial(k)
#          total_prob += prob

#       Final_prob = 1 - total_prob


#       print(f'Failures Expected |  {expected_mean}')
#       print(f'Failure observed | {observed_failure}')
#       print(f'Anomaly Probability | {Final_prob:.4%}')

#       if Final_prob < 0.01 :
#          print('Abnormal , Risky')

#       elif  Final_prob < 0.05  :
#          print('Unusual , Alert')

#       else : 
#          print('Normal')


#    elif expected_mean >=10 : 

#       z_score = difference/standard_deviation
#       z_score_fabs = m.fabs(z_score)


   

#       print(f'Failures Expected |  {expected_mean}')
#       print(f'Failure observed | {observed_failure}')
#       print(f'Diffrerence | {difference}')
#       print(f'Standard Deviation |  {standard_deviation}')
#       print(f'Anomaly score | {z_score}')

#       if z_score_fabs < 2 :
#           print('Normal , evrything is fine')

#       elif z_score_fabs >= 2 and z_score_fabs < 3 :
#           print('Unusual , Alert check the sytem')

#       elif  z_score_fabs >= 3 :
#           print('Abnormal  , Risky situation')


#    print('CAUSE ATTRIBUTION (BAYESIAN)')
#    print(f'Accuracy  | {Accuracy:.2f}%')
   
#    col  =  ["","Db","Network","Server"]

#    table = [
#       ["DB" , Db_correceted , Db_Network_incorrect , Db_Server_incorrect] ,
#       ["Network" , Network_Db_incorrected , Network_corrected , Network_Server_incorrected],
#       ["Server" , Server_Db_incorrected , Server_Network_incorrect , Server_corrected]

#    ]
#    print('-------------Confusion Matrix---------------')
#    print(f"{col[0]:<8} | {col[1]:<8} | {col[2]:<8} | {col[3]:<8}")
#    print('-' * 41)

#    for row in table :
#       print(f"{row[0]:<8} | {row[1]:<8} | {row[2]:<8} | {row[3]:<8}")



# else :
#    print('Error ,Values of Probability of failure or Number of requests is inappropriate ')



# print(Accuracy_list)



                 










              

            
           

           

            

               
                 
                 
                 
                 
           
         
        
        

             

