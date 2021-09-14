from flask import Flask, request
from flask_restful import reqparse, abort, Api, Resource
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
class Predictcancer(Resource):
    def cancerPrediction():
   
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

  
      with open('model_save', 'rb') as f:
        model = pickle.load(f)
    


        data = [[inputQuery1, inputQuery2, inputQuery3, inputQuery4, inputQuery5, inputQuery6, inputQuery7, inputQuery8, inputQuery9, inputQuery10, inputQuery11, inputQuery12, inputQuery13, inputQuery14,]]

        new_df = pd.DataFrame(data, columns =['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean','compactness_mean', 'concavity_mean', 'perimeter_se', 'area_se','radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst','compactness_worst', 'concavity_worst'])
    

        single = model.predict(new_df)
        probability = model.predict_proba(new_df)[:,1]

        if single==1:
          o1 = "The patient is diagnosed with breast cancer"
          o2 = "Confidence: {}".format(probability*100)
        else:
          o1= "The patient is  not diagnosed with breast cancer"
          o2= "Confidence: {}".format(probability*100)

    



   

   


        return render_template('index.html', output1=o1, output2=o2, query1 = request.form['query1'], query2=request.form['query2'], query3=request.form['query3'], query4=request.form['query4'], query5=request.form['query5'], query6=request.form['query6'], query7=request.form['query7'], query8=request.form['query8'], query9=request.form['query9'], query10=request.form['query10'], query11=request.form['query11'], query12=request.form['query12'], query13=request.form['query13'], query14=request.form['query14'] )
        api.add_resource(Predictcancer, '/' )

if __name__ == '__main__':
    app.run(debug=True)
    #app.run(debug=True, host='0.0.0.0', port=9696)