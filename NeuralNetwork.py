from tensorflow import keras
from keras import Sequential
from keras.layers import Dense, LSTM
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from DataFetch import get_dataframe_content, normalize_speed, normalize_time, timestr_to_int
from collections import defaultdict

print(keras.__version__)
print(keras.backend.backend())

MAX_SPEED = 60
data = get_dataframe_content(200)
print(data.head())

def split_data(data: pd.DataFrame):
    '''
        The splitted and normalized data to train/test the neural network

        Returns
        -----
        x : A float array [0, 1] which represent the timestamps of each change.

        y : A float array [0, 1] which represent the current speed of each segment per timestamp
    '''

    norm_data = normalize_speed(data, MAX_SPEED)
    print(norm_data[["iTime", "speed"]].head())

    # Compresses the segment original id to a more compact index to store in a list
    segment_compress = defaultdict(int)
    speed_map: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for index, row, in norm_data.iterrows():
        segment_compress[row["segment_id"]]
        speed_map[row["iTime"]].append((row["segment_id"], row["speed"]))

    coord = 0
    for key in segment_compress:
        segment_compress[key] = coord
        coord += 1

    def update_row(data: list[tuple[int, float]], row_y: list[int], computated_segments: int):
        for t in data:
            computated_segments += int(row_y[segment_compress[t[0]]] == -1)
            row_y[segment_compress[t[0]]] = t[1]
        return row_y, computated_segments
        
    computated_segments = 0
    row_y = [-1] * len(segment_compress)

    train_y = []
    for key in speed_map:
        #speed_map[key].sort()
        row_y, computated_segments = update_row(speed_map[key], row_y, computated_segments)

        if(computated_segments != len(segment_compress)): train_y = [row_y] # Update the 1st row while any segment is not defined 
        else: train_y.append(row_y.copy())
        #print(f"{key}: {len(speed_map[key])}")

    train_x = np.unique(data["iTime"].to_numpy())
    train_x = normalize_time(train_x[len(train_x)-len(train_y):])
    print(f"segments: {len(segment_compress)}")
    print(f"train_x: {train_x.shape}, train_y: {np.array(train_y).shape}")
    return train_x, np.array(train_y)

def split_test_data(x, y, p: float):
    single_y = [0] * len(y)
    for t in range(len(y)):
        single_y[t] = y[t][0]
    single_y = np.array(single_y)
    partition = int(len(x)*p) + 1
    return x[:partition], single_y[:partition], x[partition:], single_y[partition:]

x, y = split_data(data)
train_x, train_y, test_x, test_y = split_test_data(x, y, 0.7)

SEGMENTS = y.shape[1]
print(SEGMENTS)

def plot(compressed_segment: int, x_axes: list, y_axes: list):
    for i in range(len(y_axes)):
        plot_x = np.reshape(x_axes[i], (-1))
        #plot_y = y_axes[i].transpose()[compressed_segment]
        plot_y = y_axes[i]
        #plot_y *= MAX_SPEED

        # Create the plot
        plt.plot(plot_x, plot_y, label=r'0')

    # plot_y = y.transpose()[1]
    # plot_y *= MAX_SPEED
    # plt.plot(plot_x, plot_y, label=r'1')

    # Add labels and title
    plt.xlabel('timestamp (t)')
    plt.ylabel('speed (t)')
    plt.title('Segments: [0]')

    # Add a legend
    plt.legend()

    # Show the plot
    plt.grid(True)
    plt.show()

#plot(1)

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
    regressor.compile(optimizer=opt, loss='mean_squared_error')

    earlyStop = EarlyStopping(monitor='loss', patience=4)
    regressor.fit(ts_generator, epochs=epochs, verbose=1, callbacks=[earlyStop])

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

plot(0, [train_x, test_x, test_x[LOOKBACK:]], [train_y, test_y, predictions[0]])
plot(1, [train_x, test_x, test_x[LOOKBACK:]], [train_y, test_y, predictions[1]])