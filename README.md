# Smart Reliability & Failure Predictor

A reliability and anomaly-detection system that evolved from probability-based failure analysis into a machine-learning system for real NASA spacecraft telemetry.

The project combines probability, statistics, Bayesian inference, and machine learning to detect unusual behavior, investigate possible causes, and identify anomalies in real telemetry data.



---

##  Live Demo

**Try the deployed application:**
[Smart Reliability & Failure Predictor](https://smart-reliability-predictor-xf8m7suhusz4mu9tp8lert.streamlit.app/)

The deployed dashboard contains the V1/V2 reliability simulator and the V3 NASA telemetry anomaly-detection system.


---

## My Story

I had been studying probability and statistics for AI engineering, but I was mostly just consuming theory. I wanted to actually build something—my inner engineer was coming out (especially after watching *Spider-Man: Brand New Day*!).

That led me to **StarDance**, where I decided to build a reliability system that could detect abnormal behavior and eventually understand why it was happening.

---

# Project Evolution

## V1 — Probability-Based Reliability Analysis

The first version simulated a system receiving requests with a certain probability of failure.

I used the **Binomial distribution** to model failures:

$$
X \sim \text{Binomial}(n,p)
$$

with expected failures:

$$
E[X] = np
$$

I initially used a z-score to detect unusually large failure counts:

$$
z = \frac{x-\mu}{\sigma}
$$

However, I discovered that z-scores were not reliable when expected failure counts were small because the underlying distribution was not well-approximated by a symmetric normal distribution.

So I switched to the **Poisson distribution** for appropriate cases:

$$
X \sim \text{Poisson}(\lambda)
$$

where:

$$
\lambda=np
$$

This allowed the system to calculate the probability of observing a particular number of failures directly.

###  In simple terms

V1 answers:

> **"Is this system behaving unusually?"**

Instead of simply saying that the number of failures looks high, the system calculates what should normally happen and compares that with what actually happened.

---

## V2 — Bayesian Failure Attribution

After detecting abnormal behavior, I wanted the system to answer a harder question:

> **"What is probably causing it?"**

I introduced **Bayesian inference**:

$$
P(C|E)=\frac{P(E|C)P(C)}{P(E)}
$$

where:

* C = possible cause
* E = observed evidence

The system considered possible causes such as:

* Database overload
* Server overload
* Network issues

and used the available evidence to calculate the probability of each cause.

I also added a 3×3 confusion matrix and ran the simulator 10,000 times. The average classification accuracy was approximately **98%**.

###  In simple terms

* **V1 asks:** *"Something is wrong — how unusual is it?"*
* **V2 asks:** *"Something is wrong — what is the most likely reason?"*

Instead of hardcoding the answer, the system uses evidence to determine the most probable cause.

---

## V3 — NASA Telemetry Anomaly Detection

For V3, I moved from simulated data to the real **NASA SMAP/MSL anomaly detection dataset**.

The challenge became much harder because different telemetry channels behaved differently. Some contained clear point anomalies, while others had contextual patterns that could not be captured using simple rules.

I experimented with:

* Linear Regression
* Random Forest
* LSTM

The models use a rolling window of **10 previous observations** to predict the next value.

For a prediction \(\hat{x}_t\), the prediction error is:

$$
e_t = |x_t-\hat{x}_t|
$$

An anomaly is detected when the error becomes sufficiently large.

The threshold is based on validation error:

$$
T=\mu_e+k\sigma_e
$$

I experimented with multiple threshold values and different persistence requirements.

### In simple terms

Instead of asking:

> **"Is this telemetry value unusually large?"**

the system asks:

> **"Could the model have predicted this value from the recent behavior?"**

If the prediction is very wrong, the behavior may be anomalous.

---

# Models

### Linear Regression

Uses the previous 10 observations to predict the next telemetry value.

### Random Forest

Uses an ensemble of decision trees to model the relationship between recent telemetry values and the next value.

### LSTM

A recurrent neural network designed for sequential data.

The architecture is approximately:

```text
10 previous values
       ↓
   LSTM(50)
       ↓
    Dense(1)
       ↓
Next-value prediction
```

The LSTM was particularly useful for modeling sequential telemetry behavior.

---

# Threshold & Persistence

I experimented with combinations of:

* **Threshold multipliers:** 1, 1.5, 2, 2.5, 3, 3.5
* **Persistence:** 1, 2, 3, 5

For the LSTM, these parameters were calibrated using validation data and synthetic anomaly scenarios, and the selected parameters were saved alongside the pretrained model. The other models use their calibrated detection threshold with a persistence requirement.

The final deployed system uses the saved calibration parameters during inference rather than recalculating them every time the application runs.

###  In simple terms

The system does not panic because of one strange measurement. It looks for unusual behavior that **continues**.

---

# Architecture

The project is divided into three main stages:

* **V1/V2 — Reliability Simulation**

  * Simulates system failures
  * Statistical anomaly assessment
  * Bayesian failure-cause attribution

* **V3 — NASA Telemetry**

  * Loads real spacecraft telemetry
  * Uses predictive models for next-value prediction
  * Calculates prediction error
  * Applies threshold + persistence-based anomaly detection
  * Evaluates detections against NASA anomaly labels

* **Deployment**

  * Models are pretrained offline
  * Model artifacts and calibration parameters are saved
  * Streamlit loads the saved artifacts during inference

---

# Deployment

Training the LSTM models every time the application starts would be too expensive for a free cloud deployment. So I separated pretraining from inference:

```text
Pretraining
     ↓
Saved models
     ↓
Calibration parameters
     ↓
Streamlit App
     ↓
Dashboard Inference
```

* `pretrain.py` trains the models and saves them inside the `models/` directory.
* The Streamlit application then loads those saved models instead of retraining them.

### Live application

The final system is deployed here:

**[Open the Smart Reliability & Failure Predictor](https://smart-reliability-predictor-xf8m7suhusz4mu9tp8lert.streamlit.app/)**

---

# Tech Stack

* **Language:** Python
* **Data Processing:** NumPy, Pandas
* **Machine Learning:** Scikit-learn, TensorFlow / Keras, Joblib
* **UI & Visuals:** Streamlit, Matplotlib
* **Dataset:** NASA SMAP/MSL dataset

---

# Running Locally

Install the dependencies:

```bash
pip install -r requirements.txt
```

Pretrain the models:

```bash
python pretrain.py
```

Then start the dashboard:

```bash
streamlit run app.py
```

---

# Limitations

This project detects anomalous telemetry behavior. It does not directly prove that a spacecraft or physical system has failed.

The NASA dataset is also challenging: different channels have different behavior, and some contextual anomalies are difficult to detect from a single telemetry channel. The project is therefore primarily an exploration of how statistical reasoning and machine learning can be combined for reliability analysis.

---

# AI Usage

I used AI as a learning, planning, and documentation assistant throughout the project. I also used AI to help me understand concepts and approaches when I was stuck.

A small CSS section in `app.py` was initially generated with AI because I was unfamiliar with CSS. I studied how it worked, then modified and customized it myself for the final dashboard.

---

# Author

Built by **Touheed (Matrix)** as a StarDance project and as part of my journey toward becoming an AI engineer.
