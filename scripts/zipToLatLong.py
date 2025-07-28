from geopy.geocoders import Nominatim
import pandas as pd
import time
import os

geolocator = Nominatim(user_agent="my_geocoder")

def geocode_city_state(row):
    location_str = f"{row['adultcity']}, {row['adultstate']}, USA"
    try:
        location = geolocator.geocode(location_str, timeout=10)
        if location:
            result = pd.Series({'adult_latitude': location.latitude, 'adult_longitude': location.longitude})
        else:
            result = pd.Series({'adult_latitude': None, 'adult_longitude': None})
    except Exception as e:
        print(f"Geocoding error for {location_str}: {e}")
        result = pd.Series({'adult_latitude': None, 'adult_longitude': None})
    time.sleep(1)  
    return result

df = pd.read_csv("data/demographics.csv")

output_file = "data/demographics_geocoded.csv"

if os.path.exists(output_file):
    os.remove(output_file)  

header_df = pd.DataFrame(columns=list(df.columns) + ['adult_latitude', 'adult_longitude'])
header_df.to_csv(output_file, index=False)

for idx, row in df.iterrows():
    coords = geocode_city_state(row)
    combined_row = pd.concat([row, coords])
    
    combined_row.to_frame().T.to_csv(output_file, mode='a', header=False, index=False)
    
    print(f"Processed row {idx+1} / {len(df)}")