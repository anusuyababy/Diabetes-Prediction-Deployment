# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 16:30:21 2020

@author: DELL
"""
from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the Random Forest CLassifier model

clf = pickle.load(open('model.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        Pregnancies = int(request.form['Pregnancies'])
        Glucose = int(request.form['Glucose'])
        BP = int(request.form['BloodPressure'])
        SkinThickness = int(request.form['SkinThickness'])
        Insulin = int(request.form['Insulin'])
        BMI = float(request.form['BMI'])
        DiabetesPedigreeFunction = float(request.form['DiabetesPedigreeFunction'])
        Age = int(request.form['Age'])
        
        data = np.array([[Pregnancies, Glucose, BP, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
        my_prediction = clf.predict(data)
        
        if(int(my_prediction)==1):
            prediction="Sorry ! You Have Diabetes"
        else:
            prediction="Congrats ! You dont have Diabetes" 
        
        return (render_template('index.html', prediction=prediction))


        
if __name__ == '__main__':
    app.run(debug=True)