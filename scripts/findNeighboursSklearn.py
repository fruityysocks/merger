import numpy as np
import pandas as pd
import networkx as nx
from sklearn.neighbors import BallTree

df = pd.read_csv('data/demographicsJittered.csv')
latitudes = df['childLatJittered'].values
longitudes = df['childLongJittered'].values
coordsRad = np.vstack((np.radians(latitudes), np.radians(longitudes))).T

tree = BallTree(coordsRad, metric='haversine')
k = 10 
distances, indices = tree.query(coordsRad, k=k)

earthRadiusKM = 6371

G = nx.Graph()

for i, speaker in enumerate(df['speaker']):
    G.add_node(speaker)

for i, speaker in enumerate(df['speaker']):
    for j in range(1, k): 
        neighbor_idx = indices[i, j]
        neighbor_speaker = df.iloc[neighbor_idx]['speaker']
        distance_km = distances[i, j] * earthRadiusKM
        G.add_edge(speaker, neighbor_speaker, weight=distance_km)

matching = nx.algorithms.matching.min_weight_matching(G)

print(f"Found {len(matching)} unique neighbor pairs (each speaker only once).")

neighbor_pairs = []
for spk1, spk2 in matching:
    dist = G[spk1][spk2]['weight']
    neighbor_pairs.append((spk1, spk2, dist))

neighborsDF = pd.DataFrame(neighbor_pairs, columns=['speaker1', 'speaker2', 'distance_km'])
output = 'data/neighborPairs.csv'
neighborsDF.to_csv(output, index=False)
print(f"Saved unique neighbor pairs to {output}")
print(f"Total unique pairs: {len(neighborsDF)}")