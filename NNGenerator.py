from tensorflow import keras
from keras import Sequential
from keras.layers import Dense, LSTM
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from DataFetch import get_dataframe_content, split_data
from Plotter import plot_segment
from datetime import datetime

print(keras.__version__)
print(keras.backend.backend())

MAX_SPEED = 60
data = get_dataframe_content(2)
print(data.head())

def split_test_data(compressed_segment: int, x, y, p: float):
    single_y = [0] * len(y)
    for t in range(len(y)):
        single_y[t] = y[t][compressed_segment]
    single_y = np.array(single_y)
    partition = int(len(x)*p) + 1
    return x[:partition], single_y[:partition], x[partition:], single_y[partition:]

x, y = split_data(data, MAX_SPEED)
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

#output = model.predict(test_x.reshape((-1, LOOKBACK, 1)))
# x_inputs = []
# for i in range(LOOKBACK, len(test_x)):
#     # For each step, we extract the last LOOKBACK values
#     x_input = test_x[i-LOOKBACK:i].reshape((LOOKBACK, 1))  # Shape: (LOOKBACK, 1)
#     x_inputs.append(x_input)

# # Convert the list of inputs into a numpy array with shape (num_samples, LOOKBACK, 1)
# x_inputs = np.array(x_inputs)  # Shape: (num_samples, LOOKBACK, 1)

generator = TimeseriesGenerator(test_y, test_y, length = LOOKBACK,
                                                sampling_rate = 1,
                                                stride        = 1,
                                                batch_size    = BATCH_SIZE)

predictions = model.predict(generator)
print(predictions.shape)

# def evaluate_accuracy(test_y, predictions):
#     mse = np.mean((test_y - predictions) ** 2)
#     print(f"MSE: {mse}")
#evaluate_accuracy(test_y[LOOKBACK:], predictions)

plot_segment(0, [train_x, test_x, test_x[LOOKBACK:]], [train_y, test_y, predictions])

for i in range(1, 6):
    train_x, train_y, test_x, test_y = split_test_data(i, x, y, 0.7)
    generator = TimeseriesGenerator(test_y, test_y, length = LOOKBACK,
                                                    sampling_rate = 1,
                                                    stride        = 1,
                                                    batch_size    = BATCH_SIZE)

    predictions = model.predict(generator)
    plot_segment(i, [train_x, test_x, test_x[LOOKBACK:]], [train_y, test_y, predictions])

model.summary()
scores = model.evaluate(generator)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
model.save(f"Models/Model-10x-{datetime.now().strftime('%Y%m%d-%H%M%S')}-AC-{int(scores[1]*100)}.h5")
print("Model Saved!")