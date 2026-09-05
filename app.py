import streamlit as st
import numpy as np
import pandas as pd

from V3 import Lstm_channel , anomalies






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

    pass




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

    
