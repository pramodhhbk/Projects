#!pip install flask-ngrok
#!pip install gevent
from __future__ import division, print_function
from flask_ngrok import run_with_ngrok
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from flask import Flask, redirect, url_for
import sys
import os
import glob
import re
import json
import librosa
import numpy as np 
import pandas as pd
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras
model=keras.models.load_model("/content/drive/MyDrive/Pramodh_Project.h5")
import os
directory = '/content/drive/MyDrive/ips'

def genre_classifier(filename):
    print(filename)
    audio, sample_rate = librosa.load(filename, res_type='kaiser_fast') 
    print(audio,sample_rate)
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=90)
    mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
    mfccs_scaled_features=mfccs_scaled_features.reshape(1,-1)
    predicted_label=model.predict_classes(mfccs_scaled_features)
    #prediction_class=labelencoder.inverse_transform(prediction_label)
    return predicted_label
app=Flask(__name__,template_folder='/content/drive/MyDrive/templates')
UPLOAD_FOLDER='/content/drive/MyDrive/ips'
app.config['UPLOAD_FOLDER']=UPLOAD_FOLDER
run_with_ngrok(app)

@app.route("/")
def index():
  return  render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        print("Image recieved on server")
        f = request.files['file']   # Get the file from post request
        file_path = os.path.join(app.config['UPLOAD_FOLDER'] ,f.filename)
        f.save(file_path)
        preds = genre_classifier(file_path)
        print("preds=",preds)
        li=["blues","classical","country","disco","hiphop","jazz","metal","pop","reggae","rock"]
        print(li[preds[0]])
        return li[preds[0]]
    return None
    
app.run()
