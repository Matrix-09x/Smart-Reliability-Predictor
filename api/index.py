from io import  BytesIO

import base64 

import numpy as np

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel

from V3 import Lstm_channel ,anomalies ,Linear_Regression ,Random_forest 

from V1_V2 import Proability_stimulator 

app = FastAPI(title="NASA Spacecraft Reliability Engine")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Models = ["Linear Regression","Random Forest","Long Term Short Memory(LSTM)"]

channels = anomalies.index.to_list()


class Reliabilityrequest(BaseModel) :
    n : int = 100 
    p : float = 0.1


class TelemetryRequest(BaseModel):
    channel: str
    model: str


def safe(value) :

    if isinstance(value,np.ndarray) :
        return value.tolist()

    if isinstance(value, np.generic) :
        return value.item()

    if isinstance(value,(list,tuple)) :
        return [safe(x) for x in value]

    if isinstance(value,dict) :
        return{
            str(k): safe(v)
            for k,v in value.items()
        }

    return value


def figure_to_base64(fig):

    if fig is None:
        return None

    buffer = BytesIO()

    fig.savefig(
        buffer,
        format="png",
        bbox_inches="tight",
        dpi=120
    )

    buffer.seek(0)

    encoded = base64.b64encode(
        buffer.read()
    ).decode("utf-8")

    try:

        import matplotlib.pyplot as plt

        plt.close(fig)

    except Exception:
        pass

    return encoded




@app.get("/api/health") 
def health() :

    return{"status" : "ok",
           "channels" : safe(channels),
           "models" : Models
           }


@app.post("/api/reliability") 
def reliability(request: Reliabilityrequest) :

    if request.n < 1 :
        raise HTTPException(status_code=400,detail="Number of requests must be >= 1")

    if not 0.1 <= request.p <= 1.0 :
        raise HTTPException(status_code=400,detail="Failure probability must be between 0.1 and 1.0")


    output = Proability_stimulator(request.n , request.p)

    if "error" in output :
        return safe(output)

    confusion_matrix = output[
        "confusion matrix"
    ]


    matrix = [

        [
            confusion_matrix["Db"][
                "Db correct prediction"
            ],

            confusion_matrix["Network"][
                "Db incorrect prediction for Network"
            ],

            confusion_matrix["Server"][
                "Db incorrect prediction for Server"
            ]
        ],

        [
            confusion_matrix["Db"][
                "Network incorrected prediction for Db "
            ],

            confusion_matrix["Network"][
                "Network correct prediction"
            ],

            confusion_matrix["Server"][
                "Network incorrect prediction for Server "
            ]
        ],

        [
            confusion_matrix["Db"][
                "Server incorrect prediction for Db"
            ],

            confusion_matrix["Network"][
                "Server incorrect prediction for Network"
            ],

            confusion_matrix["Server"][
                "Server correct prediction"
            ]
        ]

    ]


    total = sum(
        map(sum, matrix)
    )


    correct = (
        matrix[0][0]
        + matrix[1][1]
        + matrix[2][2]
    )


    confusion_accuracy = (correct/total * 100 if total else 0)

    return safe({

        "expected_failures":
            output["expected_mean"],

        "observed_failures":
            output["observed_failure"],

        "accuracy":
            output["accuracy"],

        "metric_name":
            output["metric_name"],

        "metric_value":
            output["metric_value"],

        "status":
            output["status"],

        "failure_rate":
            output["observed_failure"]
            / request.n,

        "matrix":
            matrix,

        "confusion_matrix_accuracy":
            confusion_accuracy

    })


@app.post("/api/telemetry")
def telemetry(request: TelemetryRequest) :

    if request.channel not in channels :

        raise HTTPException(status_code= 400, detail="Unknown telemetry channel.")

    if request.model not in Models :

        raise HTTPException(status_code=400 , detail="Unknown detection model.")



    if request.model == "Long Term Short Memory(LSTM)":

        result = Lstm_channel(
            request.channel
        )

    elif request.model == "Random Forest":

        result = Random_forest(
            request.channel
        )

    else:

        result = Linear_Regression(
            request.channel
        )


    (
        testing_signal,
        test_error,
        threshold,
        anomalies_indices,
        anomaly_zone,
        precision,
        recall,
        fig

    ) = result


    if precision + recall > 0 :
        f1  = 2 * (precision*recall)/(precision + recall)

    else :
        f1 = 0 


    return safe({

        "channel":
            request.channel,

        "model":
            request.model,

        "testing_signal":
            testing_signal,

        "test_error":
            test_error,

        "threshold":
            threshold,

        "anomalies_indices":
            anomalies_indices,

        "anomaly_zone":
            anomaly_zone,

        "precision":
            precision,

        "recall":
            recall,

        "f1":
            f1,

        "persistence":
            (
                "Channel calibrated"
                if request.model
                == "Long Term Short Memory(LSTM)"
                else
                "3 observations"
            ),

        "figure_png_base64":
            figure_to_base64(fig)

    })