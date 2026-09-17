import streamlit as st
import numpy as np
import pandas as pd

from V3 import Lstm_channel , anomalies , Linear_Regression ,Random_forest

from V1_V2 import  Proability_stimulator




models = ['Linear Regression','Random Forest','Long Term Short Memory(LSTM)']
channels = anomalies.index.to_list()
# print(channels)

st.set_page_config(
    page_title= 'NASA Spacecraft Reliability Engine',
    page_icon= '🚀',
    layout='wide'
)


module = st.sidebar.radio(
    "Choose a module :",
    ["📊V1/V2 : Reliability Simulator" , "🛰️V3 — NASA Telemetry Analysis : "]
)

st.sidebar.title("🚀 Mission Control")

st.sidebar.markdown('---')

if module  ==  "📊V1/V2 : Reliability Simulator" :
    st.title("📊 Probabilistic Reliability Simulator and Bayesian Failure Attribution")
    st.markdown("""
    This section evaluates incoming request streams , calcualting  if anomaly rates 
    defy normal variance using **Poisson distribution** .It then applies a custom
    **Bayesian Causal Attribution Matrix** to determine root failure causes.
    """)
    st.markdown("---")

    n = st.number_input('Number of Requests :',min_value= 1,value=100,step=1)
    p = st.number_input('Failure of Probability :',min_value=0.1,max_value=1.0)


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
                'Actual Db' : [confusion_matrix['Db']['Db correct prediction'],confusion_matrix['Db']['Network incorrected prediction for Db '],confusion_matrix['Db']['Server incorrect prediction for Db']],
                'Actual Network' : [confusion_matrix['Network']['Db incorrect prediction for Network'],confusion_matrix['Network']['Network correct prediction'],confusion_matrix['Network']['Server incorrect prediction for Network']],
                "Actual Server" : [confusion_matrix['Server']['Db incorrect prediction for Server'],confusion_matrix['Server']['Network incorrect prediction for Server '],confusion_matrix['Server']['Server correct prediction']]
 
            }


            df = pd.DataFrame(data,index=['Predicted : Db','Predicted : Network','Predicted : Server'])
            st.dataframe(df, use_container_width=True)

            st.markdown('---')
            st.subheader('📝 Summary')

            st.markdown(
                f"""
                The stimulator processed **{n:,}** requests with an assumed 
                faliure probability of **{p:.1%}**.

                - Expected failures : **{output['expected_mean']:.1f}**
                - Observed failures : **{output['observed_failure']}**
                - Bayesian attribution accuracy : **{output["accuracy"]:.1f}%**
                - System assessment : **{output['status']}**

                The Bayesian attribution matrix  estimated whether failures are
                more consistent with **Database , Network and Server** conditions.

                """
            )









else :
    st.title("🛰️ NASA Spacecraft  Telemetry  Deep Learning Analysis")
    st.markdown("""This section runs a multi channel **Recurent Neural Network (LSTM)** to analyze
    continuous  sequenct-to-sequence telemetry data  from the **NASA SMAP/MSL DATASET** .
    It tracks real-time sensor shifts where classical linear model fails.
    """)
    st.markdown("---")

    model_selected = st.selectbox('Choose Model for the Anaylsis :',models)
    

    st.markdown("---")

    selected_channel = st.selectbox(
        'Choose Telemetry Channel:',
        channels

    )

    if st.button("Run Analysis") :
        if model_selected == 'Long Term Short Memory(LSTM)' :

            
            with st.spinner('Training LSTM and analyzing telemetry..... this may take  a while'):
                result = Lstm_channel(selected_channel)


        

            testing_signal , test_error , threshold , anomalies_indices , anomaly_zone , precison , recall , fig = result


            col1 , col2 , col3  = st.columns(3)
            col1.metric('Precision' , f'{precison:.1%}')
            col2.metric('Recall' , f'{recall:.1%}')
            col3.metric('Anomalies Flagged',len(anomalies_indices))

            st.info(f'Analysis completed for telemetry channel: **{selected_channel}**')

            st.subheader('📡 Telemetry channel')
            st.line_chart(testing_signal,use_container_width=True)

            st.subheader('🚨 Anomaly detection')

            st.pyplot(fig,use_container_width=True)




            st.markdown('---')

            st.subheader('📝 Summary')

            st.markdown(
                f"""
                
                The LSTM model NASA Telemetry Channel analyzed **{selected_channel} and flagged **{len(anomalies_indices):,}**
                observations as potentially anomalous.

                - **Precision:** {precison:.1%}
                - **Recall** {recall:.1%}
                - **Detection threshold:** {threshold:.6f}   
                    
                An observation is flagged when the models prediction error exceeds the  automatically  calculated
                anomaly threshold

                """


            )


        if model_selected == 'Linear Regression' :

            with st.spinner('Training Linear Regression and analyzing telemetry... this may take a while') :
                result = Linear_Regression(selected_channel)

            testing_signal , test_error , threshold , anomalies_indices , anomaly_zone , precison , recall , fig = result

            col1 , col2,col3 = st.columns(3)
            col1.metric('Precision' , f'{precison:.1%}')
            col2.metric('Recall',f'{recall:.1%}')
            col3.metric('Anomalies Flagged',len(anomalies_indices))

            st.info(f'Analysis completed for telemetry channel : **{selected_channel}**')

            st.subheader('📡 Telemetry channel')
            st.line_chart(testing_signal,use_container_width=True)

            st.subheader('🚨 Anomaly detection')

            st.pyplot(fig,use_container_width=True)


            st.markdown('---')

            st.subheader('📝 Summary')

            st.markdown(
                f"""

                The Linear regression model NASA Telemetry Channel analyzes **{selected_channel} and flagged **{len(anomalies_indices):,}**
                observations as potentially anomalous.

                - **Precision** {precison:.1%}
                - **Recall** {recall:.1%}
                - **Detection threshold:** {threshold:.6f}

                An observation is flagged when the models prediction error ex

          




         """
            )
        


