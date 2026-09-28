import streamlit as st
import numpy as np
import pandas as pd
import time 

from V3 import Lstm_channel , anomalies , Linear_Regression ,Random_forest

from V1_V2 import  Proability_stimulator




models = ['Linear Regression','Random Forest','Long Term Short Memory(LSTM)']
channels = anomalies.index.to_list()


st.set_page_config(
    page_title= 'NASA Spacecraft Reliability Engine',
    page_icon= '🚀',
    layout='wide'
)

st.markdown("""
<style>
 

    /*Main background */
    .stApp {
        background : linear-gradient(
           135deg,
           #050b14 0%,
           #0b1625 50%,
           #07111f 100%
        );
        color: #e6edf3;
    }

    /*Main content */
    .main .block-container {
      padding-top: 2rem;
      padding-bottom: 3rem;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
       background : linear-gradient(
        180deg,
        #07111f 0%,
        #0b1728 100%
       );
       border-right: 1px solid #1d344d;
       
    }

    /* Sidebar text */
    [data-testid="stSidebar"] * {
      color: #dce7f3;
    }

    /*Headings */
    h1 {
      color: #f1f7ff !important;
      font-weight: 700;
      letter-spacing: -0.5px;
    }

    h2,h3 {
      color: #d9ecff !important;
    }

    /* Normal text */
    p,li, label {
       color: #c7d5e3 !important;
      }

    /* Metric cards */
    [data-testid="stMetric"]{
      background: rgba(17,31,49,0.85);
      border: 1px solid #1f3a55;
      border-radius: 12px;
      padding: 16px;
      box-shadow: 0 4px 15px rgba(0,0,0,0.25);
    }

    [data-testid="stMetricValue"] {
      color: #e8f5ff !important;
    }

    /* Buttons */
    .stButton > button {
       width: 100%;
       border-radius: 10px;
       border : 1px solid #1976a8;
       background: linear-gradient(
          90deg,
          #075985,
          #0e7490
        );

        color: white;
        font-weight: 600;
        padding: 0.6rem 1rem;
        transition: 0.2s;
      }

      .stButton > button:hover {
       border-color: #38bdf8;
       background: linear-gradient(
         90deg,
         #0e7490,
         #0369a1
        );
        color: white;
      }

      /*Select boxes and number inputs */
      div[data-baseweb="select"] > div,
      div[data-baseweb="input"] > div {
        background-color: #0d1b2a;
        border: 1px solid #27445f;
        border-radius: 8px;
       }

       /* Horizontal separators */
       hr {
        border-color: #1d344d;

       }

       /* Info boxes */
       [data-testid="stDataFrame"] {
         background-color: rgba(13,27,42,0.8);
         border: 1px solid #1f3a55;
         border-radius: 10px;
        }

        /* Captions */
        .stCaption {
         color: #8198ad !important;

        }

</style>
""",unsafe_allow_html=True)


    







    
    
      












module = st.sidebar.radio(
    "Choose a module :",
    ["📊V1/V2 : Reliability Simulator" , "🛰️ V3 — NASA Telemetry Analysis : "]
)

st.sidebar.title("🚀 Mission Control")

st.sidebar.markdown('---')

if module  ==  "📊V1/V2 : Reliability Simulator" :
    st.title("📊 Probabilistic Reliability Simulator and Bayesian Failure Attribution")
    st.markdown("""
    This module simulates a sysnthetic system reliability environment.
    It first evaluates whether the observed failure rate is statistically unusual,
    then uses Bayesian attribution to estimate the most likely failure cause.
""")
    




    st.markdown("---")


    st.subheader('⚙️ Simulation Configuration')

    col1,col2 = st.columns(2)

    with col1 :
        n = st.number_input('Number of Requests',min_value=1,value=100,step=1)

    with col2 :
        p = st.number_input('Failure Probability',min_value=0.1,max_value=1.0,value=0.1,step=0.01)

    st.caption('The simulator generates a synthetic request stream using the selected'
               'request count and assumed failure probability')

    run_stimulation = st.button('🚀 Run Reliability Simulation',use_container_width=True)

    if run_stimulation :

        with st.spinner('Running reliability simulation....') :
            output = Proability_stimulator(n,p)

        if  'error' in output :

            st.error(output['error'])

        else :
            expected_failures = output['expected_mean']
            observed_failures = output['observed_failure']

            accuracy = output['accuracy']
            metric_name = output['metric_name']
            metric_value = output['metric_value']
            status = output['status']
            confusion_matrix = output['confusion matrix']


            st.markdown('---')

            st.subheader('🚦 System Status')

            if isinstance(status,str) :
                status_text = status.lower()

            else :
                status_text = str(status).lower()

            if ('normal' in status_text or 'safe' in status_text or 'stable' in status_text) :
                st.success(f'🟢 System Status : {status}')

            elif ('anomaly' in status_text or 'abnormal' in status_text or 'high' in status_text) :
                st.error(f'🔴 System Status: {status}')

            else :
                st.warning(f'🟡 System Status: {status}')


            st.info(
                f'Simulation completed for **{n:,} requests**'
                f'with an assumed failure probability of **{p:.1%}**.'

            )

            st.subheader('📊 Reliability Metrics')

            col1,col2,col3,col4 = st.columns(4)

            col1.metric('Expected Failures',f'{expected_failures:.1f}')

            col2.metric('Observed Failures',f'{observed_failures:.1f}')

            col3.metric('Accuracy',f'{accuracy:.1f}%')

            failure_rate = observed_failures/n

            col4.metric('Observed failure Rate',f'{failure_rate:.1%}')


            st.markdown('---')

            st.subheader('📈 Failure Behaviour')

            failure_chart = pd.DataFrame(
                {
                    "Metric" : [
                        'Expected Failures','ObservednFailures'
                    ],
                    "Failures" : [
                        expected_failures,observed_failures
                    ]
                }
            )

            st.bar_chart(failure_chart.set_index('Metric'),use_container_width=True)

            st.caption('Comparison between the expected number of failures'
                       'and the failures observed during the simulation')


            st.subheader('🚨 Statistical Anomaly Assessment')

            col1 , col2 = st.columns(2)

            col1.metric(metric_name,metric_value)

            col2.metric('Assessment',status)

            st.markdown(
                """
                The simulator  compares the observed failure behaviour with
                the expected statisitical behaviour of the system.

                This stage answers:

                **"IS the observed failure activity unusual?"**
"""
            )


            st.markdown('---')
            st.subheader('🎯 Bayesian Failure Attribution')

            st.markdown(
                """
                After dedecting unusual failure behaviour, the system attempts
                to determine which sysnthetic failure cause is most consistent
                with the available evidence.
"""
            )

            data = {
                'Actual Database' : [
                    confusion_matrix['Db']['Db correct prediction'],
                    confusion_matrix['Db']['Network incorrected prediction for Db '],
                    confusion_matrix['Db']['Server incorrect prediction for Db']
                                     
                ],

                'Actual Network' : [
                    confusion_matrix['Network']['Db incorrect prediction for Network'],
                    confusion_matrix['Network']['Network correct prediction'],
                    confusion_matrix['Network']['Server incorrect prediction for Network']

                ],

                'Actual Server' :[
                    confusion_matrix['Server']['Db incorrect prediction for Server'],
                    confusion_matrix['Server']['Network incorrect prediction for Server '],
                    confusion_matrix['Server']['Server correct prediction']
                ]    
                
            }

            confusion_df = pd.DataFrame(data,index=[
                'Predicted: Database',
                'Predicted: Network',
                'Predicted: Server'
            ])

            st.dataframe(confusion_df,use_container_width=True)

            st.subheader('🧠 How V2 Attribution Works')

            st.markdown(
                """
                **Observed failure --> Evidence --> Bayesian update --> Most likely cause**

                The Bayesian attribution stage evaluates the available evidence
                against three possible synthetic causes :

                - 🗄️ Database
                - 🌐 Network
                - 🖥️ Server

                The confusion matrix above shows how accurately the attribution
                system identifies the underlying simulated cause.

"""
            )


            st.subheader('🎯 Attribution performance')

            col1 , col2 = st.columns(2)

            col1.metric('Bayesian Accuracy',f'{accuracy:.1f}%')

            total_predictions = confusion_df.to_numpy().sum()

            correct_predictions = (confusion_df.iloc[0,0] + confusion_df.iloc[1,1] + confusion_df.iloc[2,2])

            if total_predictions > 0 :
                calculated_acc = (correct_predictions/total_predictions) * 100

            else :
                calculated_acc = 0


            col2.metric('Confusion Matrix Accuracy',f'{calculated_acc:.1f}%')

            st.markdown('---')

            st.subheader('🔍 Simulation Details')

            detail_col1 , detail_col2 = st.columns(2)

            with detail_col1 :
                st.markdown(
                    f'''
                    **Simulation Inputs**

                    - Requests: **{n:,}**
                    - Assumed failure probability: **{p:.1%}**
                    - Expected failures: **{expected_failures:.1f}**
                    

'''
                )


            with detail_col2 :

                st.markdown(
                    f'''
                    **Observed Behaviour**

                    - Observed failures: **{observed_failures:,}**
                    - Observed failure rate: **{failure_rate:.1%}**
                    - Statistical assessment: **{status}**

'''
                )


            st.markdown('---')
            st.subheader('📝 Analysis Summary')

            st.markdown(
                f"""
                The reliability simulator processed **{n:,} requests** with an
                assumed failure probability of **{p:.1%}**.

                **Reliability analysis:**

                - Expected failures: **{expected_failures:.1f}**
                - Observed failures: **{observed_failures:,}**
                - Observed failure rate: **{failure_rate:.1%}**
                - Statistical assessment: **{status}**

                **Bayesian attribution**

                - Attribution accuracy: **{accuracy:.1f}%**
                - Candidate causes : **Database, Network, Server**

                V1 established the statistical reliability baseline , while V2
                extend the system by attempting to attribute failures to their
                most likely underlying cause.

"""
            )


            with st.expander('⚠️ Important Notes & Limitations'):

                st.markdown(
                    """
                    
                    - V1/V2 use sysnthetic failure data rather than real
                      spacecraft telemetry (for real NASA data analysis 
                      switch to V3 section).

                    - Failure causes are limited to Database , Network and 
                      Server conditions.

                    - The Bayesian attribution depends on the evidence 
                      distribution used by the simulator.

                    - Statistical anomaly detection indicates unusual 
                      behaviour, it does not prove a physical system failure.

                    - V1/V2 are experimental foundations for the later
                      real world NASA telemetry analysis in V3
"""
                )

                

                      
                    















                                     
                                     
                                     















else :
    st.title("🛰️ NASA Spacecraft  Telemetry  Deep Learning Analysis")
    st.markdown("""This section runs a multi channel **Recurent Neural Network (LSTM)** to analyze
    continuous  sequenct-to-sequence telemetry data  from the **NASA SMAP/MSL DATASET** .
    It tracks real-time sensor shifts where classical linear model fails.
    """)
    st.markdown("---")


    st.subheader("🎛️ Analysis Configuration")

    col1 , col2 = st.columns(2)

    with col1 :
        selected_channel = st.selectbox('📡 Telemetry Channel',channels)

    with col2 :
        model_selected = st.selectbox('🤖 Detection Model',models)

    st.markdown("")

    run_analysis = st.button('🚀 Run Analysis',use_container_width=True)


    if run_analysis :

        if model_selected == 'Long Term Short Memory(LSTM)' :

            with st.spinner('Training LSTM model and analyzing Telemetry.... this may take a while') :
                time.sleep(2.5)
                result = Lstm_channel(selected_channel)


        elif model_selected == 'Random Forest' :

            with st.spinner('Training Random Forest and analyzing Telemetry... this may take a while') :
                 time.sleep(2.5)
                 result = Random_forest(selected_channel)

        elif model_selected == 'Linear Regression' :

            with st.spinner('Training Lineare Regression and analyzing Telemetry... this may take a while') :
                time.sleep(2.5)
                result = Linear_Regression(selected_channel)


        testing_signal , test_error , threshold , anomalies_indices , anomaly_zone , precision , recall , fig = result

        if precision + recall > 0 :
            f1 = 2 * precision * recall / (precision + recall)
        else : 
            f1 = 0


        st.markdown('---')

        st.subheader('🚨 Mission Status')

        if len(anomalies_indices) > 0 :
            st.error(
                f'🔴 Anomaly Activity Dedected - '
                f'({len(anomalies_indices):,} observation flagged)'
            )

        else :
            st.success(f'No Significant Anomalous Activity Detected')


        st.info(
            f'Analysis completed  for **{selected_channel}**'\
            f'using **{model_selected}**'
        )


        st.subheader('📊 Dedection Performance')

        col1 , col2 , col3 = st.columns(3)

        col1.metric('Precision',f'{precision:.1%}')
        col2.metric('Recall',f'{recall:.1%}')
        col3.metric('F1 score',f'{f1:.1%}')

        col1 , col2 , col3 = st.columns(3)

        col1.metric('Anomalous Flagged',f'{len(anomalies_indices):,}')

        col2.metric('Detection Threshold',f'{threshold:.6f}')

        if model_selected == 'Long Term Short Memory(LSTM)' :
            persistance_display = 'Channel calibrated'

        else :
            persistance_display = '3 observations'


        col3.metric('Persistance',persistance_display)

        st.markdown('---')
        st.subheader('📡 Telemetry Signal')


        st.caption(f'Raw telemetry signal from NASA channel {selected_channel}.')

        st.line_chart(testing_signal,use_container_width= True)



        st.subheader('🚨 Prediction Error & Anomaly Dedection')

        st.caption("Prediction error is compared against the models's detection" 
                   "threshold , NASA-labeled anomaly intervals are shown on the graph")

        st.pyplot(fig,use_container_width=True)


        st.markdown(
             f'''
             The **{model_selected}** model analyze a rolling window of 
             previous telemetry values and predict the next value.

             **Prediction --> Prediction Error --> Threshold --> Persistance   --> Anomaly

             The system flags telemetry behaviour when the prediction error
             exceeds the calculated  dedection threshold and satisfies the
             persistance requirement.

'''
        )



        st.subheader('🔍 Detection Details')


        if len(anomalies_indices) > 0 :
            detail_col1 , detail_col2  = st.columns(2)

            detail_col1.metric(
                'First flagged index',int(anomalies_indices[0])
            )

            detail_col2.metric(
                'Last flagges index',int(anomalies_indices[-1])
            )

            st.caption(
                'These are individual flagges telemetry observations.'
                'They should not automatically be interpreted as separate spacecraft failures.'

            )

        else :
            st.success('No telemetry observations exceeded the final anomaly criteria')


        st.markdown('---')

        st.subheader('🎯 Model Performance')

        performance_data = pd.DataFrame(
            {
                'Metric' : ['Precision','recall','F1 Score'],
                'Value' : [precision,recall,f1]

            }
        )

        performance_data['Value'] = (performance_data['Value']*100).round(2)

        st.dataframe(performance_data,hide_index=True,use_container_width=True)

        st.markdown('---')

        st.subheader('📝 Analysis Summary')

        st.markdown(
            f'''
            The **{model_selected}** model analyzed NASA telemetry channel
            **{selected_channel}** and flagged
            **{len(anomalies_indices):,}** potentially anomalous observations.**

            The detector achieved:

            - **Precision:** {precision:.1%}
            - **Recall:** {recall:.1%}
            - **F1 Score:** {f1:.1%}
            - **Dedection threshold:** {threshold:.6f}

            The detection system is based on **prediction error** rather than
            a fixed raw-value rule. This allows the detector to identify
            telemetry behaviour that differs from what the selected predictive
            model expects .

            The model selected by the user was **{model_selected}**.

'''
        )

        with st.expander('⚠️ Important Notes & limitations') :

            st.markdown(
                """
                - NASA anomaly labels are used for evaluation rather than
                  directly training the anomaly detector.
                - Detection thresholds are calculated from telemetry
                prediction behaviour.
                - Different telemetry channels can have different statistical
                  behaviour.
                - A falgged observation indicates anomalous telemetry
                  behaviour , not a confirmed spacecraft failure.
                - Precision and recall depend on how the model's detected
                  observations overlap with the the available NASA anomaly labels.
                - Prediction-error based detection can be affected  by distribution
                  shifts between training , validation, and test telemetry.

"""
            )

















