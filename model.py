# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import joblib

import warnings
from pandas.core.common import SettingWithCopyWarning
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)


# Read dataset
df = pd.read_csv('finantier_data_technical_test_dataset.csv', index_col=False)


# Data Pre-processing
# Drop rows with missing values and empty strings
df = df.dropna()
df['TotalCharges'] = df['TotalCharges'].replace(' ', 0)

# Convert TotalCharges to numerical column
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])

# Separate input variables and labels
X = df.drop(columns=['Default', 'customerID'], axis=1)
y = df[['Default']]

# 70-30 Train-test split
x_train_ori, x_test_ori, y_train, y_test = train_test_split(X, y, test_size=0.3)

# print("x_train shape:", x_train_ori.shape)
# print("y_train shape:", y_train.shape)
# print("x_test shape:", x_test_ori.shape)
# print("y_test shape:", y_test.shape)


# Feature Engineering
# Encode labels
le = preprocessing.LabelEncoder()
y_train['Default'] = le.fit_transform(y_train['Default'])
y_test['Default'] = le.transform(y_test['Default'])

categorical_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents',
       'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
       'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
       'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']

# Convert input variables to categorical using One Hot Encoder
ohe = preprocessing.OneHotEncoder()
arr_x_train = ohe.fit_transform(x_train_ori[categorical_cols]).toarray()
x_train_labels = ohe.get_feature_names_out()
x_train_labels = np.array(x_train_labels).ravel()
x_train = pd.DataFrame(arr_x_train, columns=x_train_labels)
add_x_train = x_train_ori[['tenure', 'MonthlyCharges', 'TotalCharges']].reset_index(drop=True)
x_train = pd.concat([x_train, add_x_train], axis=1)

arr_x_test = ohe.transform(x_test_ori[categorical_cols]).toarray()
x_test_labels = ohe.get_feature_names_out()
x_test_labels = np.array(x_test_labels).ravel()
x_test = pd.DataFrame(arr_x_test, columns=x_test_labels)
add_x_test = x_test_ori[['tenure', 'MonthlyCharges', 'TotalCharges']].reset_index(drop=True)
x_test = pd.concat([x_test, add_x_test], axis=1)

# Oversampling data using SMOTE since data is imbalanced, 73.5% default cases
smt = SMOTE()
x_train, y_train = smt.fit_resample(x_train, y_train)
#print(y_train['Default'].value_counts())

# Feature Selection
# Using Chi-square test for feature selection as it involves mainly categorical features, and categorical labels
fs = SelectKBest(score_func=chi2)
fs.fit(x_train, y_train)

# Print chi-square stats for each feature column
rows=[]
for i in range(len(fs.scores_)):
    rows.append([x_train.columns[i], fs.scores_[i]])
    
fs_list = pd.DataFrame(rows, columns=["column name", "chi-squared stats"])
# print(fs_list)

# Plot the scores
# plt.figure(figsize=(20,10))
# plt.bar([x_train.columns[i] for i in range(len(fs.scores_))], fs.scores_)
# plt.xticks(rotation=90)
# plt.show()

# Selecting only important features
drop_columns=['gender_Male', 'gender_Female', 
             'PhoneService_Yes', 'MultipleLines_No phone service',
             'PhoneService_No', 'MultipleLines_No', 'MultipleLines_Yes']
x_train = x_train.drop(columns=drop_columns, axis=1)
x_test = x_test.drop(columns=drop_columns, axis=1)
model_columns = list(x_train.columns)


# Model Building
x_train = x_train.to_numpy()
y_train = y_train.to_numpy().squeeze()
x_test = x_test.to_numpy()

tabnet = TabNetClassifier(verbose=0,seed=1)
tabnet.fit(X_train=x_train, y_train=y_train,
               patience=5,max_epochs=100,
               eval_metric=['roc_auc'])


# Results Prediction
# Helper function to print out evaluation metrics
def evaluate_results(y_test, y_pred, model):
    '''
    This is a helper function that we will call to print basic results statistics.
    '''
    cm = confusion_matrix(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    
    print(f"Performance of ", model, ":")
    print("Confusion Matrix: \n", cm)
    print("Accuracy: ", round(acc, 8))
    print("Precision: ", round(prec, 8))
    print("Recall: ", round(recall, 8))
    print("F1 score: ", round(f1, 8))
    print("ROC AUC: ", round(auc, 8))

y_pred = tabnet.predict(x_test)
evaluate_results(y_test, y_pred, 'TabNet')


# Load Models & Model Columns to pickle files
joblib.dump(tabnet,'model.pkl')
print('Model dumped!')

joblib.dump(model_columns, 'model_columns.pkl')
print('Model columns dumped!')
