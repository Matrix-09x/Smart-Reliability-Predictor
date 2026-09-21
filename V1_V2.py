import numpy as np
import math as m


rng = np.random.default_rng()


def Proability_stimulator(n,p) : 

   if n <= 0 or p <= 0 :
      
      return {'error' : 'Values of Probability of failure or Number of requests is inappropriate '}

   def stimualtion_request(n,p):
      result_list = []
      for i in range(1,n+1) :
         
         number = rng.random()

         if number < p  : 
            outcome = 0 # 0  represent failure broo as 0 is 0 and its failure
            result_list.append(outcome)

         else :

            outcome = 1 # 1 represent success like no particualar reason just prob and stat things
            result_list.append(outcome)

      return result_list

   result = stimualtion_request(n,p)


   failure_cause = []

   causes = ['Db','Network','Server']
   for i in result :
      if i == 0 :
         cause = rng.choice(causes)
         failure_cause.append(cause)

   evidences = []





   for i in failure_cause :
      if i  == 'Db' :
         db_num = int(rng.integers(70,101))
         n_num = int(rng.integers(10,131))
         S_num = int(rng.integers(30,76))
         evidences.append({'db load' : db_num , 'network latency' : n_num , 'server load' : S_num })

      elif i == 'Network' :
         db_num = int(rng.integers(30,76))
         n_num = int(rng.integers(125,501))
         S_num = int(rng.integers(30,76))
         evidences.append({'db load' : db_num , 'network latency' : n_num , 'server load' : S_num })

      else :
         db_num = int(rng.integers(30,76))
         n_num = int(rng.integers(10,131))
         S_num = int(rng.integers(70,101))
         evidences.append({'db load' : db_num , 'network latency' : n_num , 'server load' : S_num })




      





   ranges = [[(70,100) , (10,130) , (30,75)] , [(30 ,75)  , (125,500) , (30,75)] , [(30,75) , (10,130) , (70,100)]]



   Causes_list = []
   for  evidence in evidences :
      
      contributions = {}
      total_evidence = 0
      for cause_index , rang in enumerate(ranges) :
         likelihood = 1
         for index, r in  enumerate(rang):
            denominator = 0
            if index == 0 :
               min_r , max_r = r[0] , r[1]
               denominator = (max_r - min_r) + 1
               if evidence['db load'] >= min_r and evidence['db load'] <= max_r :
                  
                  likelihood *= 1/denominator 
                  

               else :
                  likelihood *= 0
                  

               
            if index == 1 :
               min_r , max_r = r[0] , r[1]
               denominator = (max_r - min_r) + 1
               if evidence['network latency'] >= min_r and evidence['network latency'] <= max_r :
                  likelihood *= 1/denominator 
                  

               else :
                  likelihood *= 0
                  
            
               
            if index == 2 :
               min_r , max_r = r[0] , r[1]
               denominator = (max_r - min_r) + 1
               if evidence['server load'] >= min_r and evidence['server load'] <= max_r :
                  likelihood *= 1/denominator 
                  

               else :
                  likelihood *= 0
      
         contribution = likelihood * 1/3

         if cause_index == 0 :
            contributions['Db'] = contribution 

         elif cause_index == 1 :
            contributions['Network'] = contribution 

         elif cause_index == 2 :
            contributions['Server'] = contribution 

      
         total_evidence += contribution
      if total_evidence != 0 :
         db_prob = contributions['Db']/total_evidence 
         Net_prob = contributions['Network']/total_evidence 
         Server_prob = contributions['Server']/total_evidence 

         if db_prob > Net_prob and db_prob > Server_prob :
            Causes_list.append("Db")

         elif Net_prob >  db_prob and Net_prob > Server_prob :
            Causes_list.append('Network')
         elif Server_prob >  db_prob and  Server_prob > Net_prob :
            Causes_list.append('Server')

         else :
            Causes_list.append('Tied')

      else :
         Causes_list.append('Unknown')
         
   # print(failure_cause)
   # print(Causes_list)

   Db_correceted = 0
   Network_corrected = 0
   Server_corrected = 0
   Db_Server_incorrect = 0
   Db_Network_incorrect = 0
   Server_Db_incorrected = 0
   Server_Network_incorrect = 0
   Network_Db_incorrected = 0
   Network_Server_incorrected = 0
   correct_prediction = 0 


   for i, cause in enumerate(failure_cause) :
      if cause == Causes_list[i] :
         correct_prediction += 1
         if cause == 'Db' :
            Db_correceted += 1
         elif cause == 'Network' :
            Network_corrected += 1

         else :
            Server_corrected += 1


      else :
         if cause == 'Db' :
            if Causes_list[i] == 'Server' :
               Db_Server_incorrect += 1

            elif Causes_list[i] == 'Network' :
               Db_Network_incorrect += 1

         elif cause == 'Server' :
            if Causes_list[i] ==  'Db' :
               Server_Db_incorrected += 1

            elif Causes_list[i] == 'Network' :
               Server_Network_incorrect += 1


         else :
            if Causes_list[i] == 'Db' :
               Network_Db_incorrected += 1

            elif Causes_list[i] == 'Server' :
               Network_Server_incorrected += 1


   Accuracy = (correct_prediction /len(failure_cause)) * 100 

   confusion_matrix  = {
      'Db' : {'Db correct prediction' : Db_correceted ,
               'Server incorrect prediction for Db' :  Db_Server_incorrect ,
               'Network incorrected prediction for Db ' : Db_Network_incorrect
               } ,

      'Server' : {'Server correct prediction' : Server_corrected ,
                  'Db incorrect prediction for Server' : Server_Db_incorrected ,
                  'Network incorrect prediction for Server ' : Server_Network_incorrect 
                  } ,

      'Network' : {'Network correct prediction' : Network_corrected  , 
                  'Db incorrect prediction for Network' : Network_Db_incorrected , 
                  'Server incorrect prediction for Network' : Network_Server_incorrected
                  
      }


   }





   if n > 0 and p > 0 :

      # print('SYSTEM REPORT')
      # print(f'Request | {n}')
      # print(f'Failure Proabability | {p}')



      expected_mean = n*p

      observed_failure = len(failure_cause)
      


      difference = observed_failure - expected_mean


      standard_deviation = m.sqrt(expected_mean)


      output = {
         'expected_mean' : expected_mean,
         'observed_failure' : observed_failure,
         'accuracy' : Accuracy,
         'confusion matrix' : confusion_matrix,
      }

      if  expected_mean < 10  :

         total_prob = 0 

         for k in range(0 , observed_failure) :
            prob = (expected_mean**k * m.e**(-expected_mean))/ m.factorial(k)
            total_prob += prob

         Final_prob = 1 - total_prob

         output['metric_name'] = 'Anomaly_probability'
         output['metric_value'] = f'{Final_prob:.4%}'
         output['status'] = 'Abnormal,Risky' if Final_prob < 0.01 else('Unusual,Alert' if Final_prob < 0.05 else 'Normal')
        


      elif expected_mean >=10 : 

         z_score = difference/standard_deviation
         z_score_fabs = m.fabs(z_score)


         output['metric_name'] = 'Anomaly Score'
         output['metric_value'] = f'{z_score:.3f}'
         output['status'] = 'Normal' if  z_score_fabs < 2 else('Unusual,Alert' if z_score_fabs < 3 else('Abnormal, Risky'))

   return output


   

         # print(f'Failures Expected |  {expected_mean}')
         # print(f'Failure observed | {observed_failure}')
         # print(f'Diffrerence | {difference}')
         # print(f'Standard Deviation |  {standard_deviation}')
         # print(f'Anomaly score | {z_score}')

         


      # print('CAUSE ATTRIBUTION (BAYESIAN)')
      # print(f'Accuracy  | {Accuracy:.2f}%')
      
      # col  =  ["","Db","Network","Server"]

      # table = [
      #    ["DB" , Db_correceted , Db_Network_incorrect , Db_Server_incorrect] ,
      #    ["Network" , Network_Db_incorrected , Network_corrected , Network_Server_incorrected],
      #    ["Server" , Server_Db_incorrected , Server_Network_incorrect , Server_corrected]

      # ]
      # print('-------------Confusion Matrix---------------')
      # print(f"{col[0]:<8} | {col[1]:<8} | {col[2]:<8} | {col[3]:<8}")
      # print('-' * 41)

      # for row in table :
      #    print(f"{row[0]:<8} | {row[1]:<8} | {row[2]:<8} | {row[3]:<8}")






   