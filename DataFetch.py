import pandas as pd
from datetime import datetime
import numpy as np

# SPEED: Estimated traffic speed in miles per hour. A value of -1 means no estimate is available.
# Data Info: https://data.cityofchicago.org/api/assets/3F039704-BD76-4E6E-8E42-5F2BB01F0AF8?download=true

def fetch(rows: int, timeout: int = 3000):
    from sodapy import Socrata

    client = Socrata("data.cityofchicago.org", None, timeout=timeout)
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
        iTime[i] /= (MAX_TIME/1)
    return iTime

#fetch(300000)