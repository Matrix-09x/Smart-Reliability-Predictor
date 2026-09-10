
import numpy  as np 
import pandas as pd 
import math as m  
import ast
import tensorflow   as tf


import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from keras.models import Sequential
from keras.layers import LSTM , Dense

from sklearn.metrics import r2_score



anomalies = pd.read_csv('labeled_anomalies.csv').reset_index(drop=True)
anomalies = anomalies.set_index('chan_id')


np.set_printoptions(threshold=np.inf)
def Lstm_channel(channel) :

    Training = np.load(f'data\data/train/{channel}.npy')
    Training_signal = Training[:,0]



    split = int(len(Training_signal) * 0.8 )

    train_signal = Training_signal[:split]

    validation_signal = Training_signal[split:]

  
  

    window_size = 10
    x_train = []
    y_train = []


    for i,value in enumerate(train_signal) :
        if i < len(train_signal) - window_size : 
            x_train.append(train_signal[i: i+window_size])
            y_train.append(train_signal[i+window_size])
            



    x_train = np.array(x_train)
    input_lstm  = x_train.reshape(x_train.shape[0] , x_train.shape[1],1)

    y_train = np.array(y_train)

    Testing  =  np.load(f'data\data/test/{channel}.npy')
    testing_signal =  Testing[:,0]


    x_test = []
    y_test = []




    for i ,value in enumerate(testing_signal) :
        if i < len(testing_signal) - window_size :
            x_test.append(testing_signal[i:i+window_size])
            y_test.append(testing_signal[i+window_size])


    x_test = np.array(x_test)
    x_test_lstm = x_test.reshape(x_test.shape[0] , x_test.shape[1],1)
    y_test = np.array(y_test)


    # print(np.shape(x_test))
    # print(np.shape(x_test_lstm))
    x_validation = []
    y_validation = []

    for i,value in enumerate(validation_signal) :
        if i < len(validation_signal) - window_size :
            x_validation.append(validation_signal[i:i+window_size])
            y_validation.append(validation_signal[i+window_size])


    x_validation = np.array(x_validation)
    x_validation_lstm = x_validation.reshape(x_validation.shape[0] , x_validation.shape[1],1)
    y_validation = np.array(y_validation)


    anomaly_zone = anomalies.loc[channel,'anomaly_sequences']

    list_data = ast.literal_eval(anomaly_zone)
    anomaly_array  = np.array(list_data,dtype= int)


    # Total_anoamly = anomaly_array[0,1] - anomaly_array[0,0]  +  1



    lstm_model = Sequential([tf.keras.Input(shape = (window_size,1)) ,LSTM(50),Dense(1)])
    lstm_model.compile(optimizer= 'adam' , loss = 'mse')

    lstm_model.fit(input_lstm , y_train , epochs= 10 , batch_size= 32)

    validation_predictions = lstm_model.predict(x_validation_lstm)

    validation_predictions = validation_predictions.flatten()

    error = np.abs(y_validation - validation_predictions)

    mean = np.mean(error)
    std = np.std(error)

    threshold = mean + 2 * std


    test_predictions = lstm_model.predict(x_test_lstm)
    test_predictions = test_predictions.flatten()

    y_test_anomaly = y_test[1240:1440]
    test_predictions_anomaly = test_predictions[1240 : 1440]




    plt.plot(y_test,color= 'blue')
    plt.plot(test_predictions,color = 'green')
    plt.axhline(threshold,color = 'red' , linewidth = 0.35)
    plt.xlabel('Time')
    plt.ylabel('Telemetry')
    plt.legend()
    
    plt.show()

    test_error = np.abs(y_test - test_predictions)
    # test_error_anomaly = test_error[5400 : 6022]
    # plt.plot(test_error_anomaly)
    # plt.axhline(threshold,linewidth = 0.4,color = 'red')
    # plt.xlabel('Time')
    # plt.ylabel('Test error')
    
   
    # plt.show()


    continouous_anomalies = []
  
    test_error_count = 0
    streak_start = None
    
    for index,err in enumerate(test_error) :
        if err > threshold :

            if test_error_count == 0 :
                streak_start  = index 
            test_error_count +=1


        else :
            if test_error_count >= 2 :
                continouous_anomalies.extend(range(streak_start , index))

            streak_start = None
            test_error_count = 0 

           
                


    if test_error_count >=2 :
        continouous_anomalies.extend(range(streak_start,index+1))

        
    continouous_anomalies  = np.array(continouous_anomalies)


    
    # anomaly_indices = np.where(test_error > threshold)[0]

    continouous_anomalies = continouous_anomalies + window_size
    fn_indices_list = []
    for index  in range(5400,6023) :
        
            if index not in continouous_anomalies :
                fn_indices_list.append(index)
  
     
    
    # fn_error_values = []
    # for i in fn_indices_list :
    #     fn_error_values.append(test_error[i])


    # print(f'Fn error min : {np.min(fn_error_values)}')
    # print(f'Fn error max : {np.max(fn_error_values)}')
    # print(f'threshold : {threshold}')
    
    print(f'Total fn indices : {len(fn_indices_list)}')
    print(f'fn {fn_indices_list}')

    print(f'Total anomalies dedected | {len(continouous_anomalies)}')

    fig,ax = plt.subplots(figsize= (12,4))
    

    ax.plot(test_error)
    ax.axhline(y=threshold , linewidth = 2 , color = 'green' )

    row  = anomalies.loc[channel]
    anomaly_zone = eval(row['anomaly_sequences'])

    for start,end in anomaly_zone :
        ax.axvspan(start ,end , alpha = 0.3 , color = 'red')
        # print(start,end)

    ax.set_xlabel('Test index')
    ax.set_ylabel('Test error')
    fig.tight_layout()
  

    Total_anomaly = 0

    for zone in anomaly_zone :
        Total_anomaly += zone[1] - zone[0] + 1


    tp = 0 
    fp = 0
    for anomaly in continouous_anomalies : 

       found = False
      
       for i in anomaly_array : 
            if anomaly >= i[0] and anomaly <= i[1] :
                found = True
                tp +=  1
                break

       if not found :
           fp +=1
        
    fn = Total_anomaly - tp 

    
    precision = tp/(tp+fp)  if (tp+fp) > 0 else 0 
    recall = tp/(tp+fn) if (tp+fn) > 0 else 0


    print(f'Tp : {tp}')
    print(f'Fp : {fp}')
    print(f'Fn : {fn}')
    print(f'Precision : {precision}')
    print(f'Recall : {recall}')
    print(f'Threshold : {threshold}')

    result_list = [testing_signal , test_error , threshold , continouous_anomalies , anomaly_zone , precision , recall, fig]
    
    # return result_list

   




Lstm_channel('E-8')








# def precision_recall(model , x_train , x_test_data,**fitkwargs) : 


    # model = LinearRegression()

    # model_2 = RandomForestRegressor(random_state= 42)

#     model.fit(x_train , output,**fitkwargs)

#     predictions = model.predict(x_train)
#     predictions = predictions.flatten()

#     training_error  = np.abs(output - predictions)
#     training_error_mean  = np.mean(training_error)
#     training_error_std  = np.std(training_error)
#     threshold = training_error_mean + 0.5 * training_error_std

#     test_predictions  = model.predict(x_test_data)
#     test_predictions = test_predictions.flatten()
#     testing_error = np.abs(y_test - test_predictions)

    
#     anomaly_indices = np.where(testing_error > threshold)[0]
#     anomaly_indices = anomaly_indices + window_size


#     tp =  len(anomaly_indices[(anomaly_indices >= anomaly_array[0,0]) & (anomaly_indices <= anomaly_array[0,1])])
#     fp = len(anomaly_indices[(anomaly_indices < anomaly_array[0,0]) | (anomaly_indices > anomaly_array[0,1])])
#     fn = Total_anoamly - tp
#     tn = len(y_test) - tp - fp - fn

#     print(f'Total anomaly dedected : {len(anomaly_indices)}')
#     print(f'Tp : {tp}')
#     print(f'Tn : {tn}')
#     print(f'Fp : {fp}')
#     print(f'Fn : {fn}')

    
#     precision  = tp/(tp+fp)
#     recall = tp/(tp+fn)
   
#     return precision , recall
    





# precision_l , recall_l = precision_recall(model,input,x_test)
# print(f'Linear regression model | Precision : {precision_l} and Recall : {recall_l}')
# precision_r , recall_r  = precision_recall(model_2,input,x_test)
# print(f'Random Forest model | Precision : { precision_r} and Recall : {recall_r}')

# precision_lstm , recall_lstm = precision_recall(lstm_model , input_lstm , x_test_lstm , epochs = 10 , batch_size = 32)
# print(f'Lstm | Precision : {precision_lstm} , Recall | {recall_lstm}')







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





        




                 










              

            
           

           

            

               
                 
                 
                 
                 
           
         
        
        

             

