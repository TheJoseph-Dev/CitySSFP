import pandas as pd
from datetime import datetime
import numpy as np
from collections import defaultdict

# SPEED: Estimated traffic speed in miles per hour. A value of -1 means no estimate is available.
# Data Info: https://data.cityofchicago.org/api/assets/3F039704-BD76-4E6E-8E42-5F2BB01F0AF8?download=true

def fetch(rows: int, timeout: int = 3000):
    from sodapy import Socrata

    env = open("CREDENTIALS.env", "r")
    APP_KEY = env.readline().removesuffix('\n')
    EMAIL = env.readline().removesuffix('\n')
    PW = env.readline().removesuffix('\n')
    env.close()

    client = Socrata("data.cityofchicago.org", APP_KEY, username=EMAIL, password=PW, timeout=timeout)
    results = client.get("sxs8-h27x", limit=rows, content_type="csv")
    results_df = pd.DataFrame.from_records(results)
    print(results_df.head())
    print(results_df.size)
    results_df.to_csv(f"./Data/chicago-city-x{int(rows/1000)}k.csv")

def timestr_to_int(timestr: str) -> int:
    return int(datetime.fromisoformat(timestr).timestamp())

def get_dataframe_content(rows: int = 100, skiprows: int = 0):
    MILE2KM = 1.609344
    df = pd.read_csv(f"Data/chicago-city-x{rows}k.csv", skiprows=skiprows)
    content = df[["segment_id", "from_street", "to_street", "hour", "day_of_week", "time", "speed", "length", "bus_count"]]
    content["length"] *= MILE2KM
    for index, row, in content.iterrows():
        if int(row["speed"]) > 0: content.at[index, "speed"] *= MILE2KM
    
    iTimes = []
    for index, row, in content.iterrows():
        iTimes.append(timestr_to_int(row["time"]))

    content = content.join(pd.DataFrame({"iTime" : iTimes}))

    return content

# 0-1 Normalization
def normalize_speed(df: pd.DataFrame, MAX_SPEED: int):
    for index, row, in df.iterrows():
        if int(row["speed"]) < 0: df.at[index, "speed"] = MAX_SPEED
        df.at[index, "speed"] /= MAX_SPEED
    return df
    
def normalize_time(iTime):
    MIN_TIME = iTime[0]
    MAX_TIME = iTime[-1]-MIN_TIME
    iTime = iTime.astype(float)
    for i in range(len(iTime)):
        iTime[i] -= MIN_TIME
        iTime[i] /= (MAX_TIME/10)
    return iTime

def split_data(data: pd.DataFrame, MAX_SPEED: int):
    '''
        The splitted and normalized data to train/test the neural network

        Returns
        -----
        x : A float array [0, 1] which represent the timestamps of each change.

        y : A float array [0, 1] which represent the current speed of each segment per timestamp

        compressed_segments : A dictionary that stores the compressed segments IDs 
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
    return train_x, np.array(train_y), segment_compress

def get_newest_test_sample(MAX_SPEED: int):
    return split_data(get_dataframe_content(3), MAX_SPEED)


#fetch(3000)

# Benchmark | LOOKBACK = 24
# 2500 rows : 305s => Barely Enough Timestamps (26)
# 5000 rows : 260s