import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import log_loss, accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tensorflow as tf
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint



n_features = 28
#model architecture:

def create_deep_model(input_dim, hidden_units=2048, num_layers=5, dropout_rate=0.5, activation='relu'):

    def dense_block(units, activation, dropout_rate):
        def make(inputs):
            x = tf.keras.layers.Dense(units)(inputs)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation(activation)(x)
            x = tf.keras.layers.Dropout(dropout_rate)(x)
            return x
        return make
        
    inputs = tf.keras.layers.Input(shape=(input_dim,))
    x = inputs
    for _ in range(num_layers):
        x = dense_block(hidden_units, activation, dropout_rate)(x)
    
    outputs = tf.keras.layers.Dense(1)(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model



def create_wide_model():
    return tf.keras.experimental.LinearModel()


wide = create_wide_model()

deep = create_deep_model(input_dim=n_features)

model = tf.keras.experimental.WideDeepModel(linear_model=wide, dnn_model=deep, activation='sigmoid')

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'] )

#loading model: 

model = tf.keras.models.load_model('./deepandWide_1024Batch_default_lr')

#scale input data:

#scaler = StandardScaler()
#X = scaler.fit_transform(X)

#predicting:

#y_pred = (model.predict(X) >= 0.5).astype("int")

#accuracy = accuracy_score(y, y_pred)

#print(f'Accuracy: {accuracy}')
