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
    
    

    
# A = int(input('Enter the Quantity of keyboard'))
# a = int(input('Enter the price of keyboard'))
# B = int(input('Enter the Quantity of Monitor'))
# b = int(input('Enter the price of Monitor'))
# C = int(input('Enter the Quantity of Mouse'))
# c = int(input('Enter the price of Mouse'))

# keybaord_total = A * a 
# Monitor_total = B * b
# mouse_total = C * c

# discount_monitor = Monitor_total * (1-0.05)
# discount_keyboard = keybaord_total * (1-0.03)
# discount_mouse = mouse_total * (1-0.02)

# print('Total price after discount',discount_monitor+discount_keyboard+discount_mouse)

# a = int(input('Enter the number :'))

# f = a % 10 
# a = a // 10 
# s = a % 10 
# t = a //10
# print(f+s+t)


# tp = int(input('Enter the number :'))
# dis = int(input('Enter the discount'))

# if dis < 10 :
#     price =  tp - (tp*dis)/100
#     print('discount price',price)
# else :
#     print(tp)




n = int(input('Enter the number '))

p = n

s = 0

while n > 0 :
    
    r = n % 10 
    s += r ** 3
    n = n // 10 

if p == s :
    print('armstrong')
else :
    print('not')
  






