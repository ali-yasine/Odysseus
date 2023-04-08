#this file is used for octopus purposes change main.ipynb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.pretraining import TabNetPretrainer
import torch.optim 

from typing import List #for type hinting

def loadData(filename:str, cols: List[str]) -> pd.DataFrame:
    df = pd.read_csv(filename, usecols=cols)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.isnull().sum()
    #remove rows with missing values
    df.dropna(inplace=True)
    return df

def main():
    cols = [
        'class_label', 'lepton_pt', 'lepton_eta', 'lepton_phi', 'missing_energy_magnitude',
        'missing_energy_phi', 'jet_1_pt', 'jet_1_eta', 'jet_1_phi', 'jet_1_b-tag',
        'jet_2_pt', 'jet_2_eta', 'jet_2_phi', 'jet_2_b-tag', 'jet_3_pt',
        'jet_3_eta', 'jet_3_phi', 'jet_3_b-tag', 'jet_4_pt', 'jet_4_eta',
        'jet_4_phi', 'jet_4_b-tag', 'm_jj', 'm_jjj', 'm_lv', 'm_jlv',
        'm_bb', 'm_wbb', 'm_wwbb'
    ] 
    data = "data/HIGGS_train.csv"
    df = loadData(data, cols)

    X = df.iloc[:, 1:].values 
    y = df.iloc[:, 0].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)        

    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=0)

    unsupervised_M = TabNetPretrainer(
        optimizer_fn=torch.optim.Adam(lr=0.02),
        n_a=128,
        n_d=128,
        n_steps=20
    )

    unsupervised_M.fit(
        X_train=X_train,
        eval_set=[X_val],
        pretraining_ratio=0.8,
        batch_size=8192,
        virtual_batch_size=256,
    )
    unsupervised_M.save_model("models/unsupervised_M")

    model_M = TabNetClassifier(n_d=96, n_a=32, lambda_sparse=0.000001,  n_steps=8, gamma=2.0, device_name='gpu')
    model_M.fit(X_train, y_train, eval_set=[(X_val, y_val)], batch_size=8192, virtual_batch_size=256)

    model_M.save_model("models/model_M")
