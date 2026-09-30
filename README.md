# Smart Reliability and Failure Predictor

A project that started as a way to practice probability and statistics and ended up as an anomaly detection system running on actual NASA spacecraft telemetry.

It tracks system behavior figures out the likely cause of failures using Bayesian logic and flags strange patterns in telemetry data using machine learning.



## Live Demo

Check out the Live Dashboard: https://smart-reliability-predictor.vercel.app/

Quick heads up: The status badge shows API Offline when you first load the page. That is normal. It starts up as soon as you click Run Analysis.



## My Story

I was studying probability and stats for AI engineering but reading theory gets boring fast. I wanted to actually build something hands on and honestly I was feeling inspired after watching Spider-Man Brand New Day.

I decided to build this as part of StarDance. The goal was simple  create a reliability monitor that does not just catch errors but actually works out why they are happening.



## How the Project Evolved

### V1 — Basic Statistical Monitoring
The first version simulated a system receiving requests with a specific failure rate.

* First try: I tried using Z-scores to spot high failure counts. That broke down on small sample sizes because the distribution was not a neat normal curve.
* The fix: I switched to a Poisson distribution where lambda equals n times p which handles rare low count events much better.
* What it does: V1 tells you if a sudden jump in failures is actually unusual or just normal randomness.

### V2 — Finding the Cause with Bayes
Once V1 flags an issue V2 tries to answer a harder question: what caused it?

I used Bayes Theorem to calculate the likelihood of different root causes:

P(C|E) = P(E|C) * P(C) / P(E)

The system checks the evidence against three common problems:
* Database overload
* Server overload
* Network issues

I tested this with a 10000 run simulation and got an average classification accuracy of about 98 percent.

### V3 — Real NASA Telemetry
For V3 I dropped simulated data and loaded up the real NASA SMAP and MSL anomaly dataset.

Real spacecraft telemetry is messy. Instead of writing rigid rules for what counts as an anomaly I trained models to look at the last 10 data points and predict what the next value should be.

If the real reading is far off from the predicted reading it flags an anomaly. I also added a persistence rule so a single weird spike does not trigger a false alarm. The error has to stay high over multiple readings.



## Models Tested

* Linear Regression: A simple baseline using the last 10 readings.
* Random Forest: Good at picking up non-linear patterns across the window.
* LSTM: Worked best for sequential time series data.
* Structure: 10 inputs to LSTM with 50 units to Dense layer to Next value prediction



## Architecture and Deployment

* Tech Stack: Python NumPy Pandas Scikit-learn TensorFlow Keras FastAPI HTML CSS JS Vercel
* How it runs:
  1. Models are trained offline using pretrain.py
  2. Saved model files and thresholds are stored
  3. FastAPI loads the saved models and serves predictions
  4. The frontend dashboard talks to FastAPI and displays the results

By separating training from inference the app runs quickly on free hosting without needing to retrain every time.



## Running Locally

1. Install dependencies:
   pip install -r requirements.txt

2. Generate model files:
   python pretrain.py

3. Start the server:
   python -m uvicorn api.index:app --port 8000

4. Open http://127.0.0.1:8000 in your browser.


## Limitations

* Anomalies do not equal Hardware Failures: The system flags mathematical deviations in telemetry but that does not always mean a physical part on the spacecraft broke.
* Single Channel Focus: Some complex spacecraft issues can only be caught by looking at multiple sensor channels at the same time.



## AI Usage

I used AI throughout the project as a tutor to help break down statistical concepts and debug code when I got stuck.

I also used AI to help build the HTML dashboard. I originally built the interface using Streamlit but Streamlit could not be used for the final deployment. Since I had limited time and was not experienced with HTML and JS I used AI to port the interface so I could stay focused on the backend algorithms and ML logic.



Built by Touheed Matrix as part of my journey toward becoming an AI engineer.