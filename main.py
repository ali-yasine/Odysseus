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

print(tf.__version__)

cols = [    
  'class_label', 'lepton_pt', 'lepton_eta', 'lepton_phi', 'missing_energy_magnitude',
  'missing_energy_phi', 'jet_1_pt', 'jet_1_eta', 'jet_1_phi', 'jet_1_b-tag',
  'jet_2_pt', 'jet_2_eta', 'jet_2_phi', 'jet_2_b-tag', 'jet_3_pt',
  'jet_3_eta', 'jet_3_phi', 'jet_3_b-tag', 'jet_4_pt', 'jet_4_eta',
  'jet_4_phi', 'jet_4_b-tag', 'm_jj', 'm_jjj', 'm_lv', 'm_jlv',
  'm_bb', 'm_wbb', 'm_wwbb'
] 

filename = "./data/HIGGS_train.csv"

df = pd.read_csv(filename, header=None, names=cols)
for col in df.columns:  
    df[col] = pd.to_numeric(df[col], errors='coerce')
#remove rows with missing values
df.dropna(inplace=True)

scaler = StandardScaler()
cols_to_scale = df.columns[1:]

scaler.fit(df[cols_to_scale])


df[cols_to_scale] = scaler.transform(df[cols_to_scale])
X = df.iloc[:, 1:].values 
y = df.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=0)        
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=0) 



# Compute the shape of the in   put data
n_samples_train, n_features = X_train.shape
n_samples_test, _ = X_test.shape
n_samples_val, _ = X_val.shape
n_channels = 1  # We have a single channel since we have a single feature dimension

early_stop = EarlyStopping(monitor='val_loss', patience=200, restore_best_weights=True)

model_checkpoint_callback = ModelCheckpoint(
    filepath='./tmp/checkpoint',
    save_weights_only=True,
    monitor='val_loss',
    mode='max',
    save_best_only=True)


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

with tf.device('/GPU:0'):

    hist = model.fit(X_train, y_train, epochs=1000, 
                 validation_data=(X_val, y_val),
                callbacks=[early_stop, model_checkpoint_callback], batch_size=1024, verbose=2)

    train_acc = model.evaluate(X_train, y_train)
    test_acc = model.evaluate(X_test, y_test)
    val_acc = model.evaluate(X_val, y_val)

threshold = 0.5
y_train_pred = (model.predict(X_train) >= threshold).astype(int)
y_test_pred = (model.predict(X_test) >= threshold).astype(int)
y_val_pred = (model.predict(X_val) >= threshold).astype(int)


print('Train accuracy: ', accuracy_score(y_train, y_train_pred))
print('Test accuracy: ', accuracy_score(y_test, y_test_pred))
print('Val accuracy: ', accuracy_score(y_val, y_val_pred))
#save history
hist_df = pd.DataFrame(hist.history)
hist_df.to_csv('./hist/deepandWide_history_1024B.csv')

#save model
model.save('./models/deepandWide_1024Batch_default_lr', save_format='tf')
