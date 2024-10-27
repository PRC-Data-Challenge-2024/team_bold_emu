import pandas as pd
import csv
import os
import argparse
import matplotlib.pyplot as plt

def ground_speed_data(df, time_column, groundspeed_column, altitude_column):
    """
    Extracts lift-off data from the flight DataFrame and calculates time to lift-off,
    ground speed at lift-off, and ground speed delta.

    Parameters:
    - df: Pandas DataFrame with flight data
    - time_column: str, the column name for the timestamp
    - groundspeed_column: str, the column name for the groundspeed
    - altitude_column: str, the column name for the altitude

    Returns:
    - A dictionary with time_to_lift_off (seconds), ground_speed_at_lift_off, and ground_speed_delta.
    """
    # Convert the time column to datetime if necessary
    df[time_column] = pd.to_datetime(df[time_column])
    

    # Calculate elapsed time in seconds
    df['elapsed_time_seconds'] = (df[time_column] - df[time_column].iloc[0]).dt.total_seconds()
    
    
    # Cut off all data before the first appearance of a value in groundspeed
    first_groundspeed_idx = df[groundspeed_column].gt(0).idxmax()
    df = df.loc[first_groundspeed_idx:]
    

    # Find the lift-off start (when groundspeed > 20 knots) and the end (altitude +100 meters)
    lift_off_start_idx = df[groundspeed_column].gt(20).idxmax()
    initial_altitude = df.loc[lift_off_start_idx, altitude_column]
    #print(initial_altitude)
    #print(initial_altitude + 328)
    #print(df[df[altitude_column].gt(initial_altitude)]["altitude"])
    lift_off_end_idx = df[df[altitude_column].gt(initial_altitude + 328)].index[0]
    lift_off_df = df.loc[lift_off_start_idx:lift_off_end_idx]

    # Calculate time_to_lift_off
    time_to_lift_off = lift_off_df['elapsed_time_seconds'].iloc[-1] - lift_off_df['elapsed_time_seconds'].iloc[0]

    # Calculate ground_speed_at_lift_off (groundspeed at the final data point)
    ground_speed_at_lift_off = lift_off_df[groundspeed_column].iloc[-1]

    # Calculate ground_speed_delta (difference between first and last groundspeed values)
    ground_speed_delta = lift_off_df[groundspeed_column].iloc[-1] - lift_off_df[groundspeed_column].iloc[0]

    return {
        'time_to_lift_off': time_to_lift_off,
        'ground_speed_at_lift_off': ground_speed_at_lift_off,
        'ground_speed_delta': ground_speed_delta
    }
    
def jet_stream_data(df, altitude_column, latitude_column, longitude_column):
    #sets a divider for the amount of chunks we want to have
    divider = 100

    # Calculate the average altitude from the given column.
    avg_altitude = int(df[altitude_column].mean())
    
    # Determine the step size `n` for 100 equidistant values.
    # We use `max(1, len(df) // 100)` to ensure we don't get a step size of 0.
    n = int(max(1, len(df) // divider))
    
    # Slice the DataFrame using `n` as the step size and make a copy to avoid the SettingWithCopyWarning.
    df_sampled = df.iloc[::n].reset_index(drop=True).copy()
    
    # Calculate the difference between each [latitude, longitude] pair and the next.
    # This will create a DataFrame with the differences.
    df_sampled['lat_diff'] = df_sampled[latitude_column].diff().shift(-1)
    df_sampled['lon_diff'] = df_sampled[longitude_column].diff().shift(-1)
    
    # Construct the vectors as lists of [lat_diff, lon_diff].
    df_sampled['vector'] = df_sampled.apply(
        lambda row: [row['lat_diff'], row['lon_diff']], axis=1
    )
    
    # Drop rows where the difference is NaN (usually the last row).
    df_vectors = df_sampled.dropna(subset=['lat_diff', 'lon_diff']).copy()

    # Define polar_jet_stream_on and subtropical_jet_stream_on.
    #The polar jet stream has strength 1, the subtropical is weaker with 0.75!
    df_vectors['polar_jet_stream_on'] = df_vectors[latitude_column].between(40, 60).astype(int)
    df_vectors['subtropical_jet_stream_on'] = df_vectors[latitude_column].between(20, 30).astype(float) * 0.75
        
    #longitude diff negative means east to west, longitude diff positive means west to east
    #Calculate the jet_stream_coefficient as the weighted sum of longitude differences.
    df_vectors['jet_stream_coefficient'] = (
        df_vectors['lon_diff'] * df_vectors['polar_jet_stream_on'] +
        df_vectors['lon_diff'] * df_vectors['subtropical_jet_stream_on']
    )
    
    # Display the relevant columns.
    #print(df_vectors[['vector', 'polar_jet_stream_on', 'subtropical_jet_stream_on', 'jet_stream_coefficient']])
    
    # Sum up the jet_stream_coefficient column to get the total coefficient.
    jet_stream_coeff = df_vectors['jet_stream_coefficient'].sum()
    
    return {
        'avg_altitude': avg_altitude,
        'jet_stream_coeff': jet_stream_coeff
    }
    
def weather_data(df):
    # Temperature: Calculate departure_temp (first 60 points) and arrival_temp (last 60 points)
    departure_temp = int(df['temperature'].head(60).median())
    arrival_temp = int(df['temperature'].tail(60).median())
    
    # Wind: Calculate u_wind and v_wind as the sum of all values in each column
    u_wind = df['u_component_of_wind'].sum()
    v_wind = df['v_component_of_wind'].sum()
    
    # Vertical rate: Sort and take the highest and lowest 100 values, then take the median for both.
    sorted_vertical_rate = df['vertical_rate'].sort_values()
    vertical_ascend = sorted_vertical_rate.tail(100).median()  # Highest 100 values (ascending)
    vertical_descend = sorted_vertical_rate.head(100).median()  # Lowest 100 values (descending)
    
    # Humidity: Sort and take the difference between the median of the highest 100 and lowest 100 values
    sorted_humidity = df['specific_humidity'].sort_values()
    highest_humidity = sorted_humidity.tail(100).median()  # Highest 100 values
    lowest_humidity = sorted_humidity.head(100).median()  # Lowest 100 values
    humidity_diff = highest_humidity - lowest_humidity
    
    # Return results in a dictionary similar to the previous style
    return {
        'departure_temp': departure_temp,
        'arrival_temp': arrival_temp,
        'u_wind': u_wind,
        'v_wind': v_wind,
        'vertical_ascend': vertical_ascend,
        'vertical_descend': vertical_descend,
        'humidity_diff': humidity_diff
    }
    
def append_weather_data_to_csv(csv_file, flight_id, result):
    csv_file = 'weather_data_' + csv_file + '.csv'

    # Define the headers and the row with flight_id
    headers = ['flight_id', 'departure_temp', 'arrival_temp', 'u_wind', 'v_wind', 'vertical_ascend', 'vertical_descend', 'humidity_diff']
    row = [
        flight_id,
        result['departure_temp'],
        result['arrival_temp'],
        result['u_wind'],
        result['v_wind'],
        result['vertical_ascend'],
        result['vertical_descend'],
        result['humidity_diff']
    ]
    
    # Write header only if the file doesn't exist
    write_header = not os.path.exists(csv_file)
    
    # Append the result to the CSV file
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if write_header:
            writer.writerow(headers)  # Write the header only once
        writer.writerow(row)
    
    
def append_jet_stream_data_to_csv(csv_file,flight_id, result):
    csv_file = 'jet_stream_data_' + csv_file + '.csv'

    # Define the header and the row with flight_id
    headers = ['flight_id', 'avg_altitude', 'jet_stream_coeff']
    row = [flight_id, result['avg_altitude'], result['jet_stream_coeff']]
    
    # Write header only if the file doesn't exist
    write_header = not os.path.exists(csv_file)
    
    # Append the result to the CSV file
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if write_header:
            writer.writerow(headers)  # Write the header only once
        writer.writerow(row)

def append_ground_speed_data_to_csv(csv_file,flight_id, result):
    csv_file = 'ground_speed_data_' + csv_file + '.csv'

    # Define the header and the row with flight_id
    headers = ['flight_id', 'time_to_lift_off', 'ground_speed_at_lift_off', 'ground_speed_delta']
    row = [flight_id, result['time_to_lift_off'], result['ground_speed_at_lift_off'], result['ground_speed_delta']]
    
    # Write header only if the file doesn't exist
    write_header = not os.path.exists(csv_file)
    
    # Append the result to the CSV file
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if write_header:
            writer.writerow(headers)  # Write the header only once
        writer.writerow(row)