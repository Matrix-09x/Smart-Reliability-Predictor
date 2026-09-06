import streamlit as st
import numpy as np
import pandas as pd

from V3 import Lstm_channel , anomalies

from V1_V2 import  Proability_stimulator





channels = anomalies.index.to_list()
# print(channels)

st.set_page_config(
    page_title= 'NASA Spacecraft Reliability Engine',
    page_icon= '🚀',
    layout='wide'
)


module = st.sidebar.radio(
    "Choose a module :",
    ["V1/V2 : Synthethic Reliability Stimulator" , "V3 NASA Telemetry Deep Learning Analysis : "]
)

st.sidebar.title("🚀 Mission Control")

st.sidebar.markdown('---')

if module  ==  "V1/V2 : Synthethic Reliability Stimulator" :
    st.title("📊 Probabilistic Failure Stimulator and Bayesian Attribution")
    st.markdown("""
    This section evaluates incoming request streams , calcualting  if anomaly rates 
    defy normal variance using **Poisson distribution** .It then applies a custom
    **Bayesian Causal Attribution Matrix** to determine root failure causes.
    """)
    st.markdown("---")

    n = st.number_input('Enter number of requests :',min_value= 1,value=100,step=1)
    p = st.number_input('Enter the probability for failure :',min_value=0.1,max_value=1.0)


    if st.button('Run Stimulation') :
        with st.spinner('Running stimulation....') :
            output = Proability_stimulator(n,p)

        if 'error' in output :
            st.error(output['error'])

        else :
            col1, col2 ,col3 =  st.columns(3)
            col1.metric('Expected Failures' , f'{output["expected_mean"]:.1f}')
            col2.metric('Observed Failures' , output['observed_failure'])
            col3.metric('Accuracy', f'{output['accuracy']:.1f}%')


            st.subheader("🚨 Anomaly Assesment")
            col4 , col5 = st.columns(2)
            col4.metric(output['metric_name'],output['metric_value'])
            col5.metric('Status',output['status'])


            st.subheader('🎯 Bayesian failure Attribution')

            confusion_matrix  =output['confusion matrix']

            data  = {
                'Actual Db' : [confusion_matrix['Db']['Db correct prediction'],confusion_matrix['Db']['Server incorrect prediction for Db'],confusion_matrix['Db']['Network incorrected prediction for Db ']],
                "Actual Server" : [confusion_matrix['Server']['Server correct prediction'],confusion_matrix['Server']['Db incorrect prediction for Server'],confusion_matrix['Server']['Network incorrect prediction for Server ']],
                'Actual Network' : [confusion_matrix['Network']['Network correct prediction'],confusion_matrix['Network']['Db incorrect prediction for Network'],confusion_matrix['Network']['Server incorrect prediction for Network']]
 
            }


            df = pd.DateOffset(data,index=['Predicted:Db','Predicted:Network','Predicted:Server'])
            st.table(df)


else :
    st.title("🛰️ NASA Spacecraft  Telemetry  Deep Learning Analysis")
    st.markdown("""This section runs a multi channel **Recurent Neural Network (LSTM)** to analyze
    continuous  sequenct-to-sequence telemetry data  from the **NASA SMAP/MSL DATASET** .
    It tracks real-time sensor shifts where classical linear model fails.
    """)
    st.markdown("---")



    selected_channel = st.selectbox(
        'Choose Telemetry Channel:',
        channels

    )

    if st.button("Run Analysis") :
        with st.spinner('Training LSTM and analyzing telemetry..... this may take  a while'):
            result = Lstm_channel(selected_channel)


    

        testing_signal , test_error , threshold , anomalies_indices , anomaly_zone , precison , recall , fig = result


        col1 , col2 , col3  = st.columns(3)
        col1.metric('Precision' , f'{precison:.1%}')
        col2.metric('Recall' , f'{recall:.1%}')
        col3.metric('Anomalies Flagged',len(anomalies_indices))

        st.pyplot(fig)

    
