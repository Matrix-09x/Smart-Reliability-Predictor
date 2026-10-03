# Smart Reliability and Failure Predictor 

I started this project  when i was stuck in learning probability and statistics for ai . Like i wanted to implement what i learned  in real projects , that was more fun . So i builded this project that initially started as a simple faliure stimulation but later turned into an anomaly detection system and i literally used an real NASA dataset for it .

# Dashboard 
<<<<<<< HEAD

=======
![Dashboard for the project] (https://github.com/Matrix-09x/Smart-Reliability-Predictor/blob/1d7bd21e5b0d5b3e0fbbccfeedbbaf65066d6b6a/Screenshot%202026-10-01%20195046.png)
>>>>>>> 45d53ba4aa0ae26b8765487c01ab0c5c247bd04b


## Live demo or url 

[Url for the dashboard](https://smart-reliability-predictor.vercel.app/)

Note: The API status can show offline when the dashboard is opened for the first time. This is normal for my deployment as it starts working when an analysis is run or you
have to wait sometime.

## My Story

I was studying probability and statistics for AI engineering, but I wanted to learn by building something.
Like i also recently watched Spiderman brand new day so my inner engineer was rising and i wanted to build something .
I made this project as part of StarDance. My main goal was to build a system that can detect problems and also try to find their possible cause.

## Project Evolution from start to end 

### V1 Statistical Monitoring -- 

The first version was a failure simulation.
I first tried using Z-scores to find unusual failure counts. It did not work well for smaller sample sizes, so I changed it to a Poisson distribution , as this was used in special cases . 
V1 checks if the observed number of failures is unusual compared to the expected number .

### V2 Finding the cause (Bayes theorem) -- 

After V1, I wanted to find out why a failure might be happening so i implemented bayes which is basically an theorem through which we can find the cause on the basis of evidence .

The system checks three possible causes:

- Database overload
- Server overload
- Network issues

I tested it with a 10000 run simulation and got an average classification accuracy of about 98% , try it yourself  you sure will have fun . 

### V3 NASA Telemetry (Working on real nasa dataset) --

For v3 i wanted to do something big so i introduced an real official  NASA dataset **NASA SMAP and MSL dataset** . 
The dataset contain many channels and they were classified into 2 types - point and contextual .
The models use the previous 10 readings to predict the next value and if the next  actual value is far from the 
prediction, it can be marked as an anomaly.
I also added a persistence rule so that one random spike does not immediately create an alert. The error needs to stay high across multiple readings.

## Models Tested

I tested three models 

- Linear Regression - Used as a simple baseline , first model that i used but it was basically good at finding line graphs.
- Random Forest - Used to find non-linear patterns , Work on lots of random trees but not the best one .
- LSTM - Used for the sequential time-series data , this is an actual rnn and i think its one of the best .

Remember you might see different results on different channel as the dataset is vast and anomalies dont have similar pattern .

### Tech Stack

Most of the code is basically written using python and numpy.
For frontend Html , CSS and Js has been used .

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

Install the dependencies to run the project locally  -

pip install -r requirements.txt

Train the models -
python pretrain.py


Start the server -

python -m uvicorn api.index:app --port 8000


Then open:
[http://127.0.0.1:8000](http://127.0.0.1:8000)



## AI Usage
I used AI mainly as a tutor during the project. It helped me understand some probability, statistics, and machine learning concepts, as well as programming and documentation questions. I tested the code myself, fixed errors, and made changes while building the project.



###  Builder (Engineer behind this project) 
This project started as a way to practice probability and statistics and grew into a project about anomaly detection, time-series data, machine learning, and deployment.
Built by Touheed (Matrix) as part of my journey toward becoming an AI engineer.

