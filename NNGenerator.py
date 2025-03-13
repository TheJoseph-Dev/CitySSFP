from tensorflow import keras
from keras import Sequential
from keras.layers import Dense, LSTM
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from DataFetch import get_dataframe_content, split_data #, get_newest_test_sample
from Plotter import plot_segment
from datetime import datetime

print(keras.__version__)
print(keras.backend.backend())

MAX_SPEED = 60

'''
#Multiple Predictions (Recursive) Test
from NeuralNetwork import NeuralNetwork
nn = NeuralNetwork("Model-10x-20250308-003101-AC-55.h5")
x, y, compressed_segments = get_newest_test_sample(MAX_SPEED)
t_y = y.transpose()[1]
p_y: np.ndarray = t_y[:25]
for i in range(0, 7):
    nx, ny, p = nn.predict(x[i:25+i], p_y, 1, False)
    p_y = np.append(p_y, p[-1])
    plot_segment(1, [x, x[:25+i+1]], [t_y, p_y[:25+i+1]])

exit()
'''

data = get_dataframe_content(100)
print(data.head())

def split_test_data(compressed_segment: int, x, y, p: float):
    single_y = [0] * len(y)
    for t in range(len(y)):
        single_y[t] = y[t][compressed_segment]
    single_y = np.array(single_y)
    partition = int(len(x)*p) + 1
    return x[:partition], single_y[:partition], x[partition:], single_y[partition:]

x, y, c = split_data(data, MAX_SPEED)
train_x, train_y, test_x, test_y = split_test_data(0, x, y, 0.7)

SEGMENTS = y.shape[1]
print(SEGMENTS)

LOOKBACK = 24
BATCH_SIZE = 1
generator = TimeseriesGenerator(train_y, train_y, length        = LOOKBACK,
                                                  sampling_rate = 1,
                                                  stride        = 1,
                                                  batch_size    = BATCH_SIZE)
def get_model(ts_generator, epochs: int = 32) -> Sequential:
    regressor = Sequential([
        LSTM(2, input_shape=(LOOKBACK,1)),
        Dense(1)
    ])
   
    # optimizer: is the algorithm to reduce the loss
    opt = Adam()
    regressor.compile(optimizer=opt, loss='mean_squared_error', metrics=['accuracy'])

    #earlyStop = EarlyStopping(monitor='loss', patience=4)
    regressor.fit(ts_generator, epochs=epochs, verbose=1, callbacks=[])

    return regressor

print(f"x: {train_x.shape} {test_x.shape} | y: {train_y.shape} {test_y.shape}")
model = get_model(generator, 32)


generator = TimeseriesGenerator(test_y, test_y, length = LOOKBACK,
                                                sampling_rate = 1,
                                                stride        = 1,
                                                batch_size    = BATCH_SIZE)

predictions = model.predict(generator)
print(predictions.shape)

plot_segment(0, [train_x, test_x, test_x[LOOKBACK:]], [train_y, test_y, predictions], lineStyles=['-', '-', '--'])

for i in range(1, 6):
    train_x, train_y, test_x, test_y = split_test_data(i, x, y, 0.7)
    generator = TimeseriesGenerator(test_y, test_y, length = LOOKBACK,
                                                    sampling_rate = 1,
                                                    stride        = 1,
                                                    batch_size    = BATCH_SIZE)

    predictions = model.predict(generator)
    plot_segment(i, [train_x, test_x, test_x[LOOKBACK:]], [train_y, test_y, predictions], lineStyles=['-', '-', '--'])

model.summary()
scores = model.evaluate(generator)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
model.save(f"Models/Model-10x-{datetime.now().strftime('%Y%m%d-%H%M%S')}-AC-{int(scores[1]*100)}.h5")
print("Model Saved!")