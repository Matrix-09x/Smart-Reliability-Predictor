
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

    validation_predictions = lstm_model.predict(x_validation_lstm,verbose= 0)

    validation_predictions = validation_predictions.flatten()

    validation_error = np.abs(y_validation - validation_predictions)

    mean = np.mean(validation_error)
    std = np.std(validation_error)

    threshold_list = [1,1.5,2,2.5,3,3.5]

    values_list = []
    
    for i in threshold_list :
        threshold =  mean + i * std
        values_list.append(threshold)



    Predictions_diff_thresholds = []
    
    for i in values_list :
        anomalies_indices = np.where(validation_error > i)[0]
        Predictions_diff_thresholds.append(anomalies_indices)


    flaggged_points_count = []
    for i in Predictions_diff_thresholds :
        count = len(i)
        flaggged_points_count.append(count)


    flaggged_points_count = np.array(flaggged_points_count)

    Flagges_rate = flaggged_points_count/len(validation_error)


    # print(values_list)
    # print(Flagges_rate)s
    
    # print(f'mean | {validation_mean}')
    # print(f'Std | {validation_std}')
    # print(f'Total length | {len(copy_validation_Signal)}')
    

    injected_anomalies  = [[150,155],[304,309],[405,410]]
    persistance = [1,2,3,5]
    all_scenario_metrics = []

    for scenrio_id in injected_anomalies :
        copy_validation_Signal = np.array(validation_signal)    
        validation_std = np.std(copy_validation_Signal)
        validation_mean = np.mean(copy_validation_Signal) 


          
        first , second = scenrio_id[0],scenrio_id[1]
        copy_validation_Signal[first:second] += 4* validation_std
        
        # print(len(copy_validation_Signal))

        # print(f'Copy validation_signal | {copy_validation_Signal[290:320]}')

        x_injected = []
        y_injected = []

        for i,value in enumerate(copy_validation_Signal) :
            if i < len(copy_validation_Signal) - window_size :
                x_injected.append(copy_validation_Signal[i:i+window_size])
                y_injected.append(copy_validation_Signal[i+window_size])

        x_injected = np.array(x_injected)
        y_injected = np.array(y_injected)

        x_injected_lstm = x_injected.reshape(x_injected.shape[0],x_injected.shape[1],1)
        injected_predictions = lstm_model.predict(x_injected_lstm,verbose=0)

        injected_predictions = injected_predictions.flatten()

        injection_error = np.abs(y_injected - injected_predictions)

        injected_threshold_results = []

        for i in values_list :
            indices = np.where(injection_error > i)[0]

            original_indices = indices + window_size
            injected_threshold_results.append(original_indices)

        persistance = [1,2,3,5]

        final_experiment_continous_anomalies = []
        for n in persistance :

            threshold_continuous_anomalies = []
            for threshold in values_list :
                validation_continuous_anomalies = []
                validation_error_count = 0 
                streak_start = None 
                for index , err in  enumerate(injection_error) :
                    if err  >  threshold :
                        if validation_error_count == 0 :
                            streak_start = index 
                        validation_error_count += 1

                    else :
                        if validation_error_count >= n :
                            validation_continuous_anomalies.extend(range(streak_start,index))
                        streak_start = None 
                        validation_error_count = 0

                if validation_error_count >= n :
                    validation_continuous_anomalies.extend(range(streak_start , index+1))

                threshold_continuous_anomalies.append(validation_continuous_anomalies)

            
            final_experiment_continous_anomalies.append(threshold_continuous_anomalies)

        
        # print(f'Persistance and threshold combianation | {final_experiment_continous_anomalies}') 
        

        
        

        combination_list_1  = []
        combination_list_2 = []
        combination_list_3 = []
        combination_list_4 = []
        comnination_list_5 = []
        combination_list_6= []
        for index,experiment in enumerate(final_experiment_continous_anomalies) :
            persistance_value = persistance[index]
        
            tp_list = []
            fp_list = []
            fn_list = []
            precision_list = []
            recall_list = []
            dedected_list_thresholds = []
            delay = []
            for theshold in experiment :
                tp = 0
                fp = 0
                dedected_list_points = []
                for  i in theshold:
            
                    if i >= first - 10 and i < second - 10 :
                        tp+=1  
                        dedected_list_points.append(i) 

                    else :
                        fp +=1 

                fn = 5 - tp 

                precision = tp/(tp+fp) if (tp+fp > 0) else  0 
                recall = tp/(tp+fn) if (tp+fn > 0) else 0

                tp_list.append(tp)
                fp_list.append(fp)
                fn_list.append(fn)
                precision_list.append(precision)
                recall_list.append(recall)
                if len(dedected_list_points)  >=  persistance_value :
                    delay_value = dedected_list_points[persistance_value-1] - (first-10)
                    dedected_list_thresholds.append(dedected_list_points[0])
                    delay.append(delay_value)
                else :
                    dedected_list_thresholds.append('missed')
                    delay.append('Not found')

                
            # print(f'Tp | {tp_list}')
            # print(f'Fp | {fp_list}')
            # print(f'Fn | {fn_list}')
            # print(f'Precision | {precision_list}')
            # print(f'Recall | {recall_list}')
            # print(f'Dedected_points | {dedected_list_thresholds}')
            # print(f'Delay | {delay}')
            combination_list_1.append([tp_list[0],fp_list[0],fn_list[0],precision_list[0],recall_list[0],dedected_list_thresholds[0],delay[0]])
            combination_list_2.append([tp_list[1],fp_list[1],fn_list[1],precision_list[1],recall_list[1],dedected_list_thresholds[1],delay[1]])
            combination_list_3.append([tp_list[2],fp_list[2],fn_list[2],precision_list[2],recall_list[2],dedected_list_thresholds[2],delay[2]])
            combination_list_4.append([tp_list[3],fp_list[3],fn_list[3],precision_list[3],recall_list[3],dedected_list_thresholds[3],delay[3]])
            comnination_list_5.append([tp_list[4],fp_list[4],fn_list[4],precision_list[4],recall_list[4],dedected_list_thresholds[4],delay[4]])
            combination_list_6.append([tp_list[5],fp_list[5],fn_list[5],precision_list[5],recall_list[5],dedected_list_thresholds[5],delay[5]])

        Total_combination_list = [combination_list_1,combination_list_2,combination_list_3,combination_list_4,comnination_list_5,combination_list_6]

        # print(combination_list_1)
        # print(combination_list_2)
        # print(combination_list_3)
        # print(combination_list_4)
        # print(comnination_list_5)
        # print(combination_list_6)
        # print()
        
        metrics_list = []
        
        for index ,  threshold_list in enumerate(Total_combination_list):
            threshold_index = index

            for index,i in enumerate(threshold_list) :
                tp , fp , fn , precision , recall , detected_thresholds , delay = i 
                persistance__value = persistance[index]

                if delay != 'Not found' :
                    delay = delay 
                else :
                    delay = float('inf')


                metrics_list.append({
                    'threshold' : threshold_index,
                    'persistance' : persistance__value,
                    'Tp' : tp ,
                    'Fp' : fp ,
                    'fn' : fn,
                    'precison' : precision,
                    'recall' : recall,
                    'Dedected Threshold' : detected_thresholds,
                    'Delay' : delay
                })

        
        all_scenario_metrics.append({
            'scenario' : [first,second],
            'metrics' :metrics_list.copy()
        })


    # print(all_scenario_metrics)
        
    combined_metrics = []

    for  candidate in all_scenario_metrics[0]['metrics'] :
        combined_metrics.append({
            'threshold' : candidate['threshold'],
            'persistance' : candidate['persistance'],
            'combined tp' : 0,
            'combined fp' : 0,
            'combined fn' : 0,
        })


    for scenario in all_scenario_metrics :
        metrics = scenario['metrics']
        
        for index,candidate in enumerate(metrics):
            combined_metrics[index]['combined tp'] +=  candidate['Tp']
            combined_metrics[index]['combined fp'] += candidate['Fp']
            combined_metrics[index]['combined fn'] += candidate['fn']
    

    for candidate in combined_metrics :
        tp = candidate['combined tp']
        fp = candidate['combined fp']
        fn = candidate['combined fn']

        precision = tp/(tp+fp) if (tp+fp) != 0 else 0 
        recall = tp/(tp+fn)  if (tp+fn) != 0 else 0 

        candidate['combined precison'] = precision 
        candidate['combined recall'] = recall

    print(len(combined_metrics))
    print(combined_metrics[0])
    print(combined_metrics[-1])



    

    

      


            

    

    



            








            
            

                


            


        
   
        
        # print(f'Index : {index}')

                    


    
            



          
    

    # for threshold , indices in zip(values_list , injected_threshold_results) :
    #     correct_indices = len(indices[(indices >= 304) & (indices < 310)])


    #     print(f'Threshold | {threshold}')
    #     print(f'Detected Points | {indices}')
    #     print(f'Correctly dedected points | {correct_indices}')
    #     print(f'Number_dedected | {len(indices)}')
    #     print()
   



    

    
    # threshold = mean + 2 * std


    test_predictions = lstm_model.predict(x_test_lstm)
    test_predictions = test_predictions.flatten()


    # y_test_anomaly = y_test[1240:1440]
    # test_predictions_anomaly = test_predictions[1240 : 1440]




    # plt.plot(y_test,color= 'blue')
    # plt.plot(test_predictions,color = 'green')
    # plt.axhline(threshold,color = 'red' , linewidth = 0.35)
    # plt.xlabel('Time')
    # plt.ylabel('Telemetry')
    # plt.legend()
    
    # plt.show()

    test_error = np.abs(y_test - test_predictions)
    # test_error_anomaly = test_error[5400:6023]
    # plt.plot(test_error)
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
            if test_error_count >= 1 :
                continouous_anomalies.extend(range(streak_start , index))

            streak_start = None
            test_error_count = 0 

           
                


    if test_error_count >=1 :
        continouous_anomalies.extend(range(streak_start,index+1))

        
    continouous_anomalies  = np.array(continouous_anomalies)


    
    # anomaly_indices = np.where(test_error > threshold)[0]

     
    
    continouous_anomalies = continouous_anomalies + window_size
    # fn_indices_list = []

    # for i in anomaly_array :
    #     start ,end = i[0] , i[1]

    #     for index  in range(start ,end+1) :
            
    #             if index not in continouous_anomalies :
    #                 fn_indices_list.append(index)
        
    # fn_error_values = []
    # for i in fn_indices_list :
    #     fn_error_values.append(test_error[i])


    # print(f'Fn error min : {np.min(fn_error_values)}')
    # print(f'Fn error max : {np.max(fn_error_values)}')
    # print(f'threshold : {threshold}')
    
    # print(f'Total fn indices : {len(fn_indices_list)}')
    # print(f'fn {fn_indices_list}')

    # print(f'Total anomalies dedected | {len(continouous_anomalies)}')

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


    # print(f'Tp : {tp}')
    # print(f'Fp : {fp}')
    # print(f'Fn : {fn}')
    # print(f'Precision : {precision}')
    # print(f'Recall : {recall}')
    # print(f'Threshold : {threshold}')
    # print(f'Continuoas anomalies : {continouous_anomalies}')

    result_list = [testing_signal , test_error , threshold , continouous_anomalies , anomaly_zone , precision , recall, fig]
    return result_list

   


Lstm_channel('F-7')


def Random_forest(channel) :
    Training  = np.load(f'data\data/train{channel}.npy')
    testing = np.load(f'data\data/train{channel}.npy')

    
    Training_signal = Training[:,0]

    split = int(len(Training_signal * 0.8))

    train_signal = Training_signal[:split]
    validation_signal = Training_signal[split:]

    window_size = 10
    x_train = []
    y_train = []

    for i,value in enumerate(train_signal) :
        if i <len(train_signal)- window_size :
            x_train.append(train_signal[i,i+window_size])
            y_train.append(train_signal[i+window_size])

    x_train = np.array(x_train)
    y_train = np.array(y_train)


    x_validation = []
    y_validation = []

    for i,value in enumerate(validation_signal):
        if i < len(validation_signal) + window_size :
            x_validation.append(i,i+window_size)
            y_validation.append(i+window_size)

    x_validation = np.array(x_validation)
    y_validation = np.array(y_validation)

    testing_signal = testing[:0]
     
    x_test = []
    y_test = []

    for i,value in enumerate(testing_signal) :
        if i < len(testing_signal)- window_size :
            x_test.append(i,i+window_size)
            y_test.append(i+window_size)



    x_test = np.array(x_test)
    y_test = np.array(y_test)

    anomaly_zone = anomalies.loc[channel,'anomaly_sequences']

    list_data = ast.literal_eval(anomaly_zone)
    anomaly_array = np.array(list_data,dtype=int)


    model = RandomForestRegressor(n_estimators=50,random_state=42,n_jobs=-1)
    model.fit(x_train,y_train)

    validation_predictions = model.predict(validation_signal)

    validation_predictions = validation_predictions.flatten()

    validation_error = np.abs(y_validation - validation_predictions)

    mean = np.mean(validation_error)
    std = np.std(validation_error)

    threshold = mean + 3*std 
    

    test_predictions = model.predict(y_test)
    test_predictions = test_predictions.flatten()

    test_error = np.abs(y_test-test_predictions)

    continuous_anomalies = []
    test_error_count = 0
    streak_Start = None

    for index,value in enumerate(test_error) :
        if value > threshold :
            if test_error_count == 0 :
                streak_Start =  index
            test_error_count += 1

        else :
          if test_error_count >= 3 :
              continuous_anomalies.extend(range(streak_Start,index))


          streak_Start = None 
          test_error_count = 0


    if test_error_count >= 3 :
        continuous_anomalies.extend(range(streak_Start,index+1))


    continuous_anomalies = np.array(continuous_anomalies) + window_size 

    fig ,ax = plt.subplots(figsize = (12,4))

    ax.plot(test_error)
    ax.axhline(y=threshold, linewidth = 2 , color = 'green' )

    row = anomalies.loc[channel]
    anomaly_zone = eval(row['anomaly_sequences'])

    for starr , end in anomaly_zone :
        ax.axvspan(starr , end , alpha = 0.3 , color = 'red')

    ax.set_xlabel('Test_index')
    ax.set_ylabel('Test error')
    fig.tight_layout()

    total_anomaly = 0

    for zone in anomaly_zone :
        total_anomaly += zone[1] - zone[0] + 1

    tp = 0
    fp = 0

    for i in continuous_anomalies :
        found = False 

        for x in anomaly_array :
            if i > x[0] and i < x[1] :
                tp +=1
                break 

        if not found :
            fp+=1 


    fn = total_anomaly - tp



    precision = tp/(tp+fp) if (tp+fp) > 0 else 0 
    recall = tp(tp+fn) if (tp +fn) > 0 else 0 

    result_list = [testing_signal, test_error , threshold , continuous_anomalies , anomaly_zone , precision , recall , fig]
    
    return result_list 



def Linear_Regression(channel) :
    training = np.load(f'data\data/train{channel}.npy')
    testing = np.load(f'data\data/train{channel}.npy')

    training_signal = training[:,0]
    testing_signal = testing[:0]

    split = int(len(training_signal * 0.8))

    train_set = training_signal[:split]
    validation_set = training_signal[split:]

    window_size = 10 

    x_train =[]
    y_train = []

    for i , value in enumerate(train_set) :
        if i < len(train_set) - window_size :
            x_train.append(i,i+window_size)
            y_train.append(i+window_size)

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    x_validation = []
    y_validation = []

    for i , value in enumerate(validation_set) :
        if i < len(train_set) - window_size :
            x_validation.append(i,i+window_size)
            y_validation.append(i+window_size)

    x_validation = np.array(x_validation)
    y_validation = np.array(y_validation)

    x_test = []
    y_test = []

    for i , value in enumerate(testing_signal) :
        if i < len(train_set)  - window_size :
            x_test.append(i,i+window_size)
            y_test.append(i+window_size)

    x_test = np.array(x_test)
    y_test = np.array(y_test) 

    anomaly_zone = anomalies.loc[channel,'anomaly_sequences']

    list_data = ast.literal_eval(anomaly_zone)
    anomaly_array = np.array(list_data,dtype=int)


    model = LinearRegression()
    model.fit(x_train,y_train)

    validation_predictions = model.predict(validation_set)
    validation_predictions = validation_predictions.flatten()

    validation_err = np.abs(y_validation - validation_predictions)

    mean = np.mean(validation_err)
    std = np.std(validation_err)

    threshold = mean + 3*std 
    
    test_predictions = model.predict(y_test)
    test_predictions = test_predictions.flatten()

    test_error = np.abs(y_test - test_predictions)

    continuous_anomalies = []
    test_error_count = 0
    streak_start = None

    for index,value in enumerate(test_error) :
        if value > threshold : 
            if test_error_count == 0 :
                streak_start = index 
            test_error_count += 1

        else :
            if test_error_count >= 3 :
                continuous_anomalies.extend(range(streak_start,index))
    

    
    if  test_error_count >= 3 :
        continuous_anomalies.extend(range(streak_start,index+1))

    
    continuous_anomalies = np.array(continuous_anomalies) + window_size

    fig , ax = plt.subplots(figsize = (12,4))

    ax.plot(test_error)
    ax.axhline(y=threshold , linewidth = 2 , color = 'green')

    row = anomalies.loc[channel]
    anomaly_zone = eval(row['anomaly_sequences'])

    for start , end in anomaly_zone :
        ax.axvspan(start,end,alpha = 0.3 , color = 'red')

    ax.set_xlabel('Test index')
    ax.set_ylabel('Test error')
    fig.tight_layout()

    total_anomaly = 0

    for i in anomaly_zone :
        total_anomaly += i[1] - i[0] + 1

    tp = 0
    fp = 0

    for i in continuous_anomalies :
        found = False

        for x in anomaly_array :
            if i > x[0] and i < x[1] :
                tp += 1
                break
        if not found :
            fp += 1

    fn = total_anomaly - tp

    precision = tp/(tp+fp) if (tp+fp) > 0 else 0 

    recall = tp(tp+fn) if (tp+fn) > 0 else 0 

    result_list = [testing_signal,test_error,threshold,continuous_anomalies,anomaly_zone,precision,recall,fig]

    return result_list
     

    
    

    
    


    






    



   






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





        




                 










              

            
           

           

            

               
                 
                 
                 
                 
           
         
        
        

             

