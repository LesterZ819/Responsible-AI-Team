import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython import display
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Input
from keras.models import Model




def biased_classifier(n_features):
    inputs = Input(shape=(n_features,))
    dense1 = Dense(64, activation='relu')(inputs)
    dropout1 = Dropout(0.2)(dense1)
    dense2 = Dense(64, activation='relu')(dropout1)
    dropout2 = Dropout(0.2)(dense2)
    dense3 = Dense(64, activation="relu")(dropout2)
    dropout3 = Dropout(0.2)(dense3)
    dense4 = Dense(64, activation="relu")(dropout3)
    dropout4 = Dropout(0.2)(dense4)
    dense5 = Dense(64, activation="relu")(dropout4)
    dropout5 = Dropout(0.2)(dense5)
    outputs = Dense(1, activation='sigmoid')(dropout5)
    nnmodel = Model(inputs=[inputs], outputs=[outputs])
    nnmodel.compile(loss='binary_crossentropy', optimizer='adam')
    return nnmodel
