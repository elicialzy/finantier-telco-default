# Finantier Telco Default Prediction

This repository contains ML model that predicts the likelihood of customers defaulting on telco payment based on their telco data. 

## Model Performance
Model used: [TabNet](https://github.com/dreamquark-ai/tabnet) <br>
Model Performance:
```
ROC-AUC: 0.75641831
F1-score: 0.61674009
```


## Installation

### Requirements
* Docker
* Python

------
### Create Virtual Environment

Install Python virtual environment 
```
$ pip3 install virtualenv
```

Create a virtual environment named _venv_ for this project
```
$ virtualenv venv
```

Start the environment 
```
$ source venv/bin/activate
```

------
### Method 1: Running model on Docker
Build the docker image
```
$ docker build -t finantier-telco-default .
```

Run docker container
```
$ docker run -d -p 3000:3000 finantier-telco-default
```

Check if container is running
```
$ docker ps -a
```
_Copy hostname to use API (e.g. http://0.0.0.0:3000/)_

------
### Method 2: Running model on local computer
Install requirements
```
$ pip install -r ./requirements.txt
```

Run the API service
```
$ python app.py
```
_Copy hostname to use API (e.g. http://0.0.0.0:3000/)_

------
## API Usage
`POST /predict`
`e.g. http://0.0.0.0:3000/predict`

**API Parameters**
Name  | Required | Type | Supported Values
------------- | ------------- | ------------- | ------------- |
gender | required | string | `Female` or `Male`
seniorCitizen | required | float | `1` (is senior citizen) or `0` (is not senior citizen)
Partner | required | string | `Yes` or `No`
Dependents | required | string | `Yes` or `No`
tenure | required | float | 
PhoneService | required | str | `Yes` or `No`
MultipleLines | required | string | `Yes` or `No` or `No phone service`
InternetService | required | string | `DSL` or `Fiber optic` or `No`
OnlineSecurity | required | string | `Yes` or `No` or `No internet service`
OnlineBackup | required | string | `Yes` or `No` or `No internet service`
DeviceProtection | required | string | `Yes` or `No` or `No internet service`
TechSupport | required | string | `Yes` or `No` or `No internet service`
StreamingTV | required | string | `Yes` or `No` or `No internet service`
StreamingMovies | required | string | `Yes` or `No` or `No internet service`
Contract | required | string | `Month-to-month` or `One year` or `Two year`
PaperlessBilling | required | string | `Yes` or `No`
PaymentMethod | required | string | `Electronic check` or `Mailed check` or `Bank transfer (automatic)` or `Credit card (automatic)`
MonthlyCharges | required | float | 
TotalCharges | required | float | 

------
**Sample request**
```
[
  {
    "gender": "Male",
    "SeniorCitizen": 0.0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 34.0,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "Yes",
    "OnlineBackup": "No",
    "DeviceProtection": "Yes",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "One year",
    "PaperlessBilling": "No",
    "PaymentMethod": "Mailed check",
    "MonthlyCharges": 56.95,
    "TotalCharges": "1889.5"
  }
]
```

------
**Sample response**
```
{
    "prediction": "[0]"
}
```
1: Customer is likely to default on telco payment<br>
0: Customer is unlikely to default on telco payment
