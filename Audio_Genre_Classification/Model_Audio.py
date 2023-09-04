import librosa
import numpy as np 
import pandas as pd

librosa.__version__

# Commented out IPython magic to ensure Python compatibility.
import matplotlib.pyplot as plt
# %matplotlib inline

filename ='/content/drive/MyDrive/music/blues/blues.00000.wav'

import IPython.display as ipd
import librosa.display

plt.figure(figsize=(14,5))
data,sample_rate=librosa.load(filename)
librosa.display.waveplot(data,sr=sample_rate)
ipd.Audio(filename)

librosa_audio_data,librosa_sample_rate=librosa.load(filename)
lb=librosa.feature.chroma_stft(y=librosa_audio_data, sr=librosa_sample_rate)
lb_arr=np.mean(lb,axis=0)
lb_var=np.var(lb,axis=0)
lb_new=np.mean(lb_arr,axis=0)
lb_lat=np.var(lb_arr,axis=0)
print(lb_new,lb_lat)

print(librosa_audio_data)

plt.figure(figsize=(12, 4))
plt.plot(librosa_audio_data)

mfccs = librosa.feature.mfcc(y=librosa_audio_data, sr=librosa_sample_rate, n_mfcc=90)
print(mfccs.shape)

mfccs

def features_extractor(file_name):
    librosa_audio_data,librosa_sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
    mfccs_features = librosa.feature.mfcc(y=librosa_audio_data, sr=librosa_sample_rate, n_mfcc=90)
    mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
    
    return mfccs_scaled_features

feature_set=[]
import os
directory = '/content/drive/MyDrive/music'

for filename in os.listdir(directory):
  for musicfile in os.listdir(directory+'/'+filename):
    print(musicfile)
    if(musicfile=='.ipynb_checkpoints'):
      continue
    data = features_extractor(directory+'/'+filename+'/'+musicfile)
    feature_set.append([data,filename])

features_extracted_df=pd.DataFrame(feature_set,columns=['feature','class'])
features_extracted_df.tail()

X=np.array(features_extracted_df['feature'].tolist())
y=np.array(features_extracted_df['class'].tolist())

X.shape

features_extracted_df['class'].value_counts()

from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
y=to_categorical(labelencoder.fit_transform(y))

y

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)



X_train

X_train.shape

y_train.shape

import tensorflow as tf
print(tf.__version__)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten
from tensorflow.keras.optimizers import Adam
from sklearn import metrics

num_labels=y.shape[1]

import keras

model = Sequential()

model.add(Flatten(input_shape=(90,)))
model.add(Dense(512, activation='relu', kernel_regularizer = keras.regularizers.l2(0.001)))
model.add(Dropout(0.3))
model.add(Dense(256, activation='relu', kernel_regularizer = keras.regularizers.l2(0.003)))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu', kernel_regularizer = keras.regularizers.l2(0.01)))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(10, activation='softmax'))
model.summary()

from tensorflow.keras import optimizers
adam = optimizers.Adam(lr=1e-4)

model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')
model.fit(X_train, y_train, batch_size=32, epochs=200,verbose=1)

test_accuracy=model.evaluate(X_test,y_test,verbose=0)
print(test_accuracy[1])

import os
directory = '/content/drive/MyDrive/music'

for filename in os.listdir(directory):
  for musicfile in os.listdir(directory+'/'+filename):
    if(musicfile=='.ipynb_checkpoints'):
      continue
    audio, sample_rate = librosa.load(directory+'/'+filename+'/'+musicfile, res_type='kaiser_fast') 
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=90)
    mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
    mfccs_scaled_features=mfccs_scaled_features.reshape(1,-1)
    predicted_label=model.predict_classes(mfccs_scaled_features)
    prediction_class = labelencoder.inverse_transform(predicted_label) 
    print(prediction_class[0],filename)

model.save('Pramodh_Project.h5')



