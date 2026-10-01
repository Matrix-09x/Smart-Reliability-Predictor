# Smart Reliability and Failure Predictor

This project started when I was learning probability and statistics for AI engineering. I wanted to use what I was learning in a real project instead of only studying theory.
It started as a simple failure simulation and later became an anomaly detection system using real NASA spacecraft telemetry.

## Live Demo

[Open the Live Dashboard](https://smart-reliability-predictor.vercel.app/)

Note: The API status can show Offline when the dashboard is opened for the first time. This is normal for my deployment. It starts working when an analysis is run.

## My Story

I was studying probability and statistics for AI engineering, but I wanted to learn by building something.
Like i also recently watched Spiderman brand new day so my inner engineer was rising and i wanted to build something .
I made this project as part of StarDance. My main goal was to build a system that can detect problems and also try to find their possible cause.

## How the Project Evolved

### V1 - Statistical Monitoring

The first version was a failure simulation.
I first tried using Z-scores to find unusual failure counts. It did not work well for smaller sample sizes, so I changed it to a Poisson distribution.
V1 checks if the observed number of failures is unusual compared to the expected number.

### V2 - Finding the Cause

After V1, I wanted to find out why a failure might be happening.
I used Bayes Theorem:
The system checks three possible causes:

- Database overload
- Server overload
- Network issues

I tested it with a 10,000-run simulation and got an average classification accuracy of about 98%.

### V3 - NASA Telemetry

For V3, I moved from simulated data to real NASA telemetry data from the **SMAP and MSL anomaly datasets**.
The dataset contain many channels and they were classified into 2 types - point and contextual .
The models use the previous 10 readings to predict the next value.
If the actual value is far from the prediction, it can be marked as an anomaly.
I also added a persistence rule so that one random spike does not immediately create an alert. The error needs to stay high across multiple readings.

## Models Tested

I tested three models:

- Linear Regression - Used as a simple baseline.
- Random Forest - Used to find non-linear patterns.
- LSTM - Used for the sequential time-series data.

### Tech Stack

- Python
- NumPy
- Pandas
- Scikit-learn
- TensorFlow / Keras
- FastAPI
- HTML
- CSS
- JavaScript
- Vercel

### How It Works

1. Models are trained using pretrain.py.
2. The trained models and thresholds are saved.
3. FastAPI loads the saved models.
4. The dashboard sends requests to the API.
5. The results are shown on the dashboard.

Training and prediction are kept separate so the deployed app does not need to train the models every time.

## Running Locally

Install the dependencies:
pip install -r requirements.txt

Train the models:
python pretrain.py


Start the server:
python -m uvicorn api.index:app --port 8000


Then open:
[http://127.0.0.1:8000](http://127.0.0.1:8000)



## AI Usage
I used AI mainly as a tutor during the project. It helped me understand some probability, statistics, and machine learning concepts, as well as programming and documentation questions. I tested the code myself, fixed errors, and made changes while building the project.



### Final Note
This project started as a way to practice probability and statistics and grew into a project about anomaly detection, time-series data, machine learning, and deployment.
Built by Touheed (Matrix) as part of my journey toward becoming an AI engineer.

