from math import radians, cos, sin, asin, sqrt
import pandas as pd

def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Haversine formula
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    km = 6371 * c 
    return km

def find_neighbors(df, radius_km):
    speakers = list(zip(df['speaker'], df['childLatJittered'], df['childLongJittered']))
    neighbors = []
    n = len(speakers)
    for i in range(n):
        id_a, lat_a, lon_a = speakers[i]
        for j in range(i+1, n):
            id_b, lat_b, lon_b = speakers[j]
            distance = haversine(lat_a, lon_a, lat_b, lon_b)
            if distance <= radius_km:
                neighbors.append((id_a, id_b, distance))
    return neighbors

df = pd.read_csv("data/demographicsJittered.csv")
neighbors = find_neighbors(df, radius_km=2)
print(f"Found {len(neighbors)} neighbor pairs within 2 km.")
print()
