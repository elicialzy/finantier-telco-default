import pandas as pd
import numpy as np
from flask import Flask, jsonify, request, render_template
import joblib
from sklearn import preprocessing

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():

    # Load model & model columns
    model = joblib.load('model.pkl')
    model_columns = joblib.load('model_columns.pkl')

    # Load dataset from API, and convert to DataFrame
    json_ = request.json
    query_df = pd.DataFrame(json_)

    # Data Pre-processing to drop unimportant features from feature selection
    query_df = query_df.drop(columns=['PhoneService', 'gender', 'MultipleLines'])
    # Data Pre-processing to convert categorical data to numerical columns
    ohe = preprocessing.OneHotEncoder()
    arr_x_train = ohe.fit_transform(query_df[categorical_cols]).toarray()
    x_train_labels = ohe.get_feature_names_out()
    x_train_labels = np.array(x_train_labels).ravel()
    x_train = pd.DataFrame(arr_x_train, columns=x_train_labels)
    add_x_train = query_df[['tenure', 'MonthlyCharges', 'TotalCharges']].reset_index(drop=True)
    x_train = pd.concat([x_train, add_x_train], axis=1) 

    # Reindex columns based on loaded model columns
    query = x_train.reindex(columns=model_columns, fill_value=0)
    query = query.to_numpy().astype(np.float32)
    
    # Predict results
    prediction = model.predict(query)

    # Return predicted result
    return jsonify({'prediction': str(list(prediction))})


categorical_cols = ['SeniorCitizen', 'Partner', 'Dependents',
       'InternetService', 'OnlineSecurity', 'OnlineBackup', 
       'DeviceProtection', 'TechSupport', 'StreamingTV',
       'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']


@app.route('/')
def home():
    return "Yellow Banana"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='3000', debug=True)
