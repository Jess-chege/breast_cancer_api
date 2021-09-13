from flask import Flask, request
from flask.templating import render_template
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.datasets import make_classification
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd 
import os
from sklearn.metrics import r2_score
import pickle

app = Flask("cancer-prediction")

q = ""

@app.route('/')# methods=['GET']) 
def loadPage():
    return render_template('index.html', query="")

@app.route('/', methods=['POST'])
def cancerPrediction():
    #data = pd.read_csv('data.csv')
    #data.head()
    #data.drop(['id', 'Unnamed: 32'], inplace=True, axis=1)

    inputQuery1 = request.form['query1']
    inputQuery2 = request.form['query2']
    inputQuery3 = request.form['query3']
    inputQuery4 = request.form['query4']
    inputQuery5 = request.form['query5']
    inputQuery6 = request.form['query6']
    inputQuery7 = request.form['query7']
    inputQuery8 = request.form['query8']
    inputQuery9 = request.form['query9']
    inputQuery10 = request.form['query10']
    inputQuery11 = request.form['query11']
    inputQuery12 = request.form['query12']
    inputQuery13 = request.form['query13']
    inputQuery14 = request.form['query14']

    
    #data['diagnosis'] = data['diagnosis'].map( {'B': 1, 'M': 0} )

    #feature_cols = [c for c in data.columns if c not in ['diagnosis']]
    #X = data[feature_cols]
    #y = data['diagnosis']
    #pd.pandas.set_option('display.max_columns', None)

    #feature_sel_model = SelectFromModel(Lasso(alpha=0.005, random_state=23, max_iter=3000,tol=30.295954819192826))
    #feature_sel_model.fit(data[feature_cols], data['diagnosis'])
    #feature_sel_model.get_support()
    #selected_feat = data[feature_cols].columns[(feature_sel_model.get_support())]
    #print('total features: {}'.format((data[feature_cols].shape[1])))
    #print('selected features: {}'.format(len(selected_feat)))
    #print(selected_feat)
    #X = X[selected_feat]
    #y = data['diagnosis'] 
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    #X_test = X_test.fillna(X_train.mean())
    #first_model = RandomForestClassifier(n_estimators = 10, criterion = 'entropy',
                                  #random_state = 42)
   # first_model.fit(X_train, y_train)

    #pred_y = first_model.predict(X_test)

    #preds = first_model.predict(X_train)
   # with open('./model_files/model_pickle', 'rb') as f:
    #model = pickle.load(open('model_pickle', 'rb'))
    with open('model_save', 'rb') as f:
        model = pickle.load(f)
    


    data = [[inputQuery1, inputQuery2, inputQuery3, inputQuery4, inputQuery5, inputQuery6, inputQuery7, inputQuery8, inputQuery9, inputQuery10, inputQuery11, inputQuery12, inputQuery13, inputQuery14,]]

    new_df = pd.DataFrame(data, columns =['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean','compactness_mean', 'concavity_mean', 'perimeter_se', 'area_se','radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst','compactness_worst', 'concavity_worst'])
    #print('new_df') 

    single = model.predict(new_df)
    probability = model.predict_proba(new_df)[:,1]

    if single==1:
        o1 = "The patient is diagnosed with breast cancer"
        o2 = "Confidence: {}".format(probability*100)
    else:
        o1= "The patient is  not diagnosed with breast cancer"
        o2= "Confidence: {}".format(probability*100)

    



    #print("Accuracy:", accuracy_score(y_test, pred_y))

   


    return render_template('index.html', output1=o1, output2=o2, query1 = request.form['query1'], query2=request.form['query2'], query3=request.form['query3'], query4=request.form['query4'], query5=request.form['query5'], query6=request.form['query6'], query7=request.form['query7'], query8=request.form['query8'], query9=request.form['query9'], query10=request.form['query10'], query11=request.form['query11'], query12=request.form['query12'], query13=request.form['query13'], query14=request.form['query14'] )


if __name__ == '__main__':
    app.run(debug=True)
    #app.run(debug=True, host='0.0.0.0', port=9696)