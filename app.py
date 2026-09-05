import streamlit as st
import numpy as np
import pandas as pd


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

if module  ==  "V1/V2 : Synthetic Reliability Stimulator" :
    st.title("📊 Probabilistic Failure Stimulator and Bayesian Attribution")
    st.markdown("""
    This section evaluates incoming request streams , calcualting  if anomaly rates 
    defy normal variance using **Poisson distribution** .It then applies a custom
    **Bayesian Causal Attribution Matrix** to determine root failure causes.
    """)
    st.markdown("---")





else :
    st.title("🛰️ NASA Spacecraft  Telemetry  Deep Learning Analysis")
    st.markdown("""This section runs a multi channel **Recurent Neural Network (LSTM)** to analyze
    continuous  sequenct-to-sequence telemetry data  from the **NASA SMAP/MSL DATASET** .
    It tracks real-time sensor shifts where classical linear model fails.
    """)
    st.markdown("---")


    
