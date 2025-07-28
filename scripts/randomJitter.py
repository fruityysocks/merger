import numpy as np
import pandas as pd

def addJitter(df, lat, long):
    df = df.copy()
    
    latJitterDeg = .8 / 111
    
    latJitter = np.random.uniform(-latJitterDeg, latJitterDeg, size=len(df))
    
    df[f'{lat}Jittered'] = df[lat] + latJitter
    cosLat = np.cos(np.radians(df[lat]))
    cosLat = np.clip(cosLat, 0.0001, None)
    longJitterDeg = .8 / (111 * cosLat)
    
    longJitter = np.random.uniform(-1, 1, size=len(df)) * longJitterDeg
    
    df[f'{long}Jittered'] = df[long] + longJitter
    
    return df


df = pd.read_csv("data/demographics.csv")
dfJittered = addJitter(df, lat='childLat', long='childLong')
output = 'data/demographicsJittered.csv'
dfJittered.to_csv(output, index=False)

print(f"Saved jittered coordinates to {output}")
