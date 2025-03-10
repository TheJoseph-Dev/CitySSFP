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
    
    def predict(self, x, y, steps: int = 1, plot: bool = True) -> np.ndarray:
        generator = TimeseriesGenerator(y, y, length        = LOOKBACK,
                                              sampling_rate = 1,
                                              stride        = 1,
                                              batch_size    = BATCH_SIZE)
        predictions = self.model.predict(generator, verbose=0)
        
        #plot_segment(i, [train_x, test_x, test_x[LOOKBACK:]], [train_y, test_y, predictions])
        if(plot): plot_segment(0, [x, x[LOOKBACK:]], [y, predictions])
        return x, y, predictions