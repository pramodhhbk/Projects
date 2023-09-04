
# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np
import plotly.express as px
import tensorflow as tf
# %matplotlib inline

data = pd.read_csv('Churn_Modelling.csv')

data.head()

### Exited -> 1 -> left the bank
### Exited -> 0 -> stayed with the bank

data['Exited'].value_counts()

tf.__version__

x = data.iloc[:,3:-1].values
y=data.iloc[:,-1].values

print(x)

print(y)

data.isnull().sum()

## Label Encoding the Gender Column
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
x[:,2] = le.fit_transform(x[:,2])

### Label Encoder has encoded females as 0 and males as 1
print(x)

### One hot encoding the Geo locations
## in Column Transformer , the parameter is taken as a list of tuple [(name , transformer , columns)]
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[1])],remainder='passthrough')
x = np.array(ct.fit_transform(x))

### Once Onehot encoding is done , the encoded values have moved to the first position
x

### Split Train and Test set
from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.2,random_state=0)

### Standardizing the input variables is key in Neural Networks
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)

print(x_train)

### Now we are going to build the sequential ANN for bank churn prediction
ann = tf.keras.models.Sequential()

## Input and first Hidden Layer
## There is no rule of thumb to decide the neurons in the hidden layer , so try experimenting different values , I go with 6
ann.add(tf.keras.layers.Dense(units=6,activation='relu'))

## Second hidden Layers
ann.add(tf.keras.layers.Dense(units=6,activation='relu'))

### Output Layer
### only one output neuron is required for the o/p layer
ann.add(tf.keras.layers.Dense(units=1,activation ='sigmoid'))

## Compiling the ANN
## optimizer are those who use GD or SGD , adam is SGD
ann.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

### Training the ANN on our x_train
### Batch size , we compare batch_sized pred with real time values rather than 1 by 1
ann.fit(x_train,y_train,batch_size=64,epochs=150)

### Probablity that certain customer leaves the bank
print(ann.predict(ss.transform([[1,0,0,600,1,40,3,60000,2,1,1,50000]])))

## Confusion matrix
## Converting the probs to zero or 1
y_hat = ann.predict(x_test)
y_hat = np.round(y_hat)
from sklearn.metrics import confusion_matrix,accuracy_score
print(confusion_matrix(y_test,y_hat))
print(accuracy_score(y_test,y_hat))

