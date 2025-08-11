import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import pandas as pd

def move10miles(
    df, latCol, longCol, max, stdDev):
    """
    Adds normal-distributed jitter to df coordinates in-place,
    constrained to New England polygon boundaries.
    
    - max: maximum allowed jitter from original position (approx)
    - stdDev: std dev of normal distribution in miles
    """
    # Load New England polygon
    shapefile_path = "data/naturalearth_lowres/ne_110m_admin_1_states_provinces.shp"

    gdf = gpd.read_file(shapefile_path)

    new_england_states = ['Connecticut', 'Massachusetts', 'Maine', 'New Hampshire', 'Rhode Island', 'Vermont']
    ne_states = gdf[gdf['name'].isin(new_england_states)]
    ne_union = ne_states.geometry.unary_union
    milesPerDegLat = 69.0

    jitteredLats = []
    jitteredLongs = []

    for lat, long in zip(df[latCol], df[longCol]):
        point = Point(long, lat)
        jitteredPoint = None
        attempts = 0
        maxAttempts = 100

        while attempts < maxAttempts:
            # Normal distributed offsets in miles
            lat_offset_miles = np.random.normal(loc=0, scale=stdDev)
            lon_offset_miles = np.random.normal(loc=0, scale=stdDev)

            # Clip to max ± limit to prevent extremes
            lat_offset_miles = np.clip(lat_offset_miles, -max, max)
            lon_offset_miles = np.clip(lon_offset_miles, -max, max)

            # Convert miles to degrees (lon corrected by cos(lat))
            milesPerDegLong = milesPerDegLat * np.cos(np.radians(lat))
            lat_offset_deg = lat_offset_miles / milesPerDegLat
            lon_offset_deg = lon_offset_miles / milesPerDegLong

            new_lat = lat + lat_offset_deg
            new_lon = long + lon_offset_deg
            jitteredPoint = Point(new_lon, new_lat)

            if ne_union.contains(jitteredPoint):
                break  # valid point within New England
            attempts += 1

        if attempts == maxAttempts:
            jitteredPoint = point  # fallback to original

        jitteredLats.append(jitteredPoint.y)
        jitteredLongs.append(jitteredPoint.x)

    # Add jittered coordinates to dataframe in place
    df["childLatJittered_day3"] = jitteredLats
    df["childLongJittered_day3"] = jitteredLongs
    
    return df

df = pd.read_csv('data/demographicInfo.csv')  

df = move10miles(df, latCol="childLatJittered_day2", longCol="childLongJittered_day2", max=10, stdDev=3)
df.to_csv("data/demographicInfo.csv", index=False)
print(df[[ 'childLatJittered_day2', 'childLatJittered_day3', 'childLongJittered_day2', 'childLongJittered_day3']].head())