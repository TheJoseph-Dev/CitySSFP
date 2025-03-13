import tensorflow
from tensorflow import keras
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Plotter import plot_segment

LOOKBACK = 24
BATCH_SIZE = 1
class NeuralNetwork:
    def __init__(self, model_version: str):
        self.model = load_model(f"Models/{model_version}")
    
    def predict(self, x, y, steps: int = 1, plot: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        #plot_segment(i, [train_x, test_x, test_x[LOOKBACK:]], [train_y, test_y, predictions])
        p_y: np.ndarray = y[len(y)-LOOKBACK-1:]
        for _ in [None] * steps:
            generator = TimeseriesGenerator(p_y, p_y, length  = LOOKBACK,
                                                sampling_rate = 1,
                                                stride        = 1,
                                                batch_size    = BATCH_SIZE)
            predictions = self.model.predict(generator, verbose=0)
            p_y = np.append(p_y, min(max(predictions[-1], 0.0001), 1))

            if(plot): plot_segment(0, [x, x[len(x)-len(p_y):]], [y, p_y], ["Real", "Prediciton"], ['-', '--'])
        
        return x, p_y, p_y[-1]