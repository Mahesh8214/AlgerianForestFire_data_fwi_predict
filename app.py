from flask import Flask,jsonify,render_template,request
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle

app = Flask(__name__)

ridge_model = pickle.load(open('models/ridge.pkl' , 'rb'))
scaler_model = pickle.load(open('models/scaler.pkl' , 'rb'))

@app.route('/')
def home():
  jls_extract_var = 'index.html'
  return render_template(jls_extract_var)

@app.route("/predict_datapoint" , methods= ['POST' , 'GET'])
def index():
  if request.method=='POST':
   
    Temperature = float(request.form.get('Temperature'))
    RH = float(request.form.get('RH'))
    Ws = float(request.form.get('Ws'))
    Rain = float(request.form.get('Rain'))
    FFMC = float(request.form.get('FFMC'))	
    DMC	= float(request.form.get('DMC'))
    ISI	= float(request.form.get('ISI'))
    Classes	= float(request.form.get('Classes'))
    Region = float(request.form.get('Region'))

    new_scaled_data = scaler_model.transform([[Temperature,RH,Ws,Rain,FFMC,DMC ,ISI,Classes,Region]])
    result = ridge_model.predict(new_scaled_data)
    return render_template('index.html',result = result[0])

  else:
        return render_template('index.html')



if __name__=="__main__":
    app.run(host="0.0.0.0")

