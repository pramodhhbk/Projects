import pandas as pd
import numpy as np
import seaborn as sns
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

tf.__version__

###Preprocessing the Images of Training Set to avoid Overfitting
###Image Augementation
### Rescale is like feature scaling where the pixels are modified between 0 and 1
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip=True

)
training_set = train_datagen.flow_from_directory(
    'Section 40 - Convolutional Neural Networks (CNN)/dataset/training_set',
    target_size = (64,64), ## Final Image size fed into the Conv layer
    batch_size=32,
    class_mode='binary'
)

### We need to rescale the Test Set images as well
test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory(
    'Section 40 - Convolutional Neural Networks (CNN)/dataset/test_set',
    target_size=(64,64),
    batch_size=32,
    class_mode='binary'
)

### Building the CNN
### Initializing the CNN as sequence of layers
cnn = tf.keras.models.Sequential()

### Conv Layer
cnn.add(
    tf.keras.layers.Conv2D(
        filters=32 ,kernel_size=3 , activation='relu',input_shape = [64,64,3]
    )
)

##Pooling Layer to identify the imp features from the feature map
cnn.add(
    tf.keras.layers.MaxPool2D(
        pool_size=2,
        strides=2
    )
)

### Second Conv Layer
### Input shape is only added to the first layer
cnn.add(
    tf.keras.layers.Conv2D(
        filters=32 ,kernel_size=3 , activation='relu'
    )
)
cnn.add(
    tf.keras.layers.MaxPool2D(
        pool_size=2,
        strides=2
    )
)

### Flattening
cnn.add(
    tf.keras.layers.Flatten()
)

### Full Connected Layer
cnn.add(
    tf.keras.layers.Dense(
        units=128,
        activation='relu'
    )
)

### Output Layer
cnn.add(
    tf.keras.layers.Dense(
        units=1,
        activation = 'sigmoid'
    )
)

### Compiling the CNN
cnn.compile(
    optimizer='adam',
    loss = 'binary_crossentropy' ,
    metrics = ['accuracy' , tf.keras.metrics.F1Score()]
)

cnn.fit(x= training_set , validation_data = test_set , epochs = 25)

### Making a single Prediction
test_image = image.load_img(
    'Section 40 - Convolutional Neural Networks (CNN)/dataset/single_prediction/cat_or_dog_2.jpg',
     target_size=(64,64)
)
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image ,axis=0)
result = cnn.predict(test_image/255.0)
print(training_set.class_indices)
print(result[0][0])
if result[0][0] > 0.5:
    prediction = 'dog'
    print(prediction)
else:
    prediction = 'cat'
    print(prediction)

