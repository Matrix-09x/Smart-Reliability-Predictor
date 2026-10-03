# SMART RELIABILITY PREDICTOR
This project analyzes the signals coming from the system and  detect and flags the anomlies using Probability and Statistics
concepts .This is an predictor working on real NASA dataset  simulating the anomalies from different channels and give the 
real analysis. This is basically my research project to find the nature of the dataset and make a system so that it can 
flag the anomalies

# Link for the project

[Url](https://smart-reliability-predictor.vercel.app/)

# This Project is divided into three parts --

## V1 Stimulating Predictor
In this section  i made an system to analyze the failure probability  where you enter the no of requests and failure probability
and for a system and it gives you the analysis . I used  probability and statistics concepts like z score  to find the unusual 
failure counts (for those who are not familiar it basically tells how far the value appears from the mean value  of the system if 
it is too far its an anomaly) .But if the probability of an failure is too small and there are large amount of requests it z score 
doesnt work properly because of the mean value , in this case i have used poisson ditribution . You can experiment it yourself in the 
live project .

## V2 Bayes theorem
After v1 i introduced bayes formula , to find the most probable cause for the anomaly by seeing the evidence and than concluding the 
cause behind the failure . For those who dont know bayes theorem is an mathematical formula that helps you update the chance of something
happening when you get new information . 

## V3 NASA DATASET 
From here i introduced a real NASA dataset SMAP and MSL . Basically this dataset has different channels and consist of anomalies happened on 
that channels .So i started researching on the dataset to find similarity and pattern in those channels . In there labeled_anomalies.csv
(you can check it yourself) there are two types of channels Contextual and point . So i reseatched on those channels and tested three models
Linear regression , Random forest and an RNN - Long term short memory (LSTM) . You can see the results in the live project i have provided 
options to choose different models .

# How it works 

First go on the live dashbord . There are 2 section  -
* V1/V2 - This section comprised of first 2 versions of my project , you can enter number of requests and probability failure for the system
  run  and can see the  results .

* V3 - This section include my research on real NASA dataset . You can choose different channels and ml models provided and can see the 
  analysis of that run .

# Credits 
* Stack overflow for documentation
* Official NASA SMAP and MSL  Dataset for  research and stimulation
* Vercel for deployment

# Reason behind making this project
I was studying Probaility and statistics concepts for a long time . I wanted to implement it and see the real results . I also watched Spiderman
Brand new day at that time so my inner engineer was rising and i wanted to build something so i decided to make this project .

## Made by -- 
Touheed (Matrix) ambition of becoming Elite AI engineer .



