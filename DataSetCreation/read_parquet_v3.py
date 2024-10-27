import pandas as pd
import os
import matplotlib.pyplot as plt
import time
import dask.dataframe as dd
import pyarrow.parquet as pq

# Set number of threads explicitly
#pl.Config.set_global_threads(14)



#extracts the flight dataframe for a given date and flight id and a defined folder
def get_flight_data(flight_id, flight_date, parquet_folder_path):
    """
    Extracts flight data for a specific flight_id and date from the respective Parquet file.

    Parameters:
    - flight_id: str or int, the unique flight identifier.
    - flight_date: str, the date of the flight in 'YYYY-MM-DD' format.
    - parquet_folder_path: str, path to the folder containing parquet files.

    Returns:
    - flight_data: pandas DataFrame with data points for the specific flight, or empty DataFrame if not found.
    """
    # Generate the path to the respective parquet file
    parquet_file_path = os.path.join(parquet_folder_path, f"{flight_date}.parquet")
    
    # Check if the parquet file exists
    if not os.path.exists(parquet_file_path):
        print(f"Parquet file for date {flight_date} does not exist.")
        return pd.DataFrame()  # Return an empty DataFrame if file is missing
    
    # Read the parquet file into a DataFrame
    try:
        parquet_df = pd.read_parquet(parquet_file_path)
    except Exception as e:
        print(f"Error reading parquet file: {e}")
        return pd.DataFrame()  # Return empty DataFrame on error
    
    # Filter the DataFrame to only include rows where flight_id matches
    flight_data = parquet_df[parquet_df['flight_id'] == flight_id]
    
    if flight_data.empty:
        print(f"No data found for flight_id {flight_id} on {flight_date}.")
        
    '''
    The returned data set has the following columns
    Index(['flight_id', 'timestamp', 'latitude', 'longitude', 'altitude',
       'groundspeed', 'track', 'vertical_rate', 'u_component_of_wind',
       'v_component_of_wind', 'temperature', 'specific_humidity', 'icao24'],
      dtype='object')
    '''
    
    return flight_data
    
def get_flight_data_lift_off(flight_id, flight_date, parquet_folder_path):
    """
    Extracts flight data for a specific flight_id and date from the respective Parquet file,
    returning only 'flight_id', 'timestamp', 'altitude', and 'groundspeed' columns.

    Parameters:
    - flight_id: str or int, the unique flight identifier.
    - flight_date: str, the date of the flight in 'YYYY-MM-DD' format.
    - parquet_folder_path: str, path to the folder containing parquet files.

    Returns:
    - flight_data: pandas DataFrame with data points for the specific flight, or empty DataFrame if not found.
    """
    # Generate the path to the respective parquet file
    parquet_file_path = os.path.join(parquet_folder_path, f"{flight_date}.parquet")
    
    # Check if the parquet file exists
    if not os.path.exists(parquet_file_path):
        print(f"Parquet file for date {flight_date} does not exist.")
        return pd.DataFrame()  # Return an empty DataFrame if file is missing
    
    # Read the parquet file into a DataFrame
    try:
        # Only read in the specified columns
        selected_columns = ['flight_id', 'timestamp', 'altitude', 'groundspeed']
        parquet_df = pd.read_parquet(parquet_file_path, columns=selected_columns)
    except Exception as e:
        print(f"Error reading parquet file: {e}")
        return pd.DataFrame()  # Return empty DataFrame on error
    
    # Filter the DataFrame to only include rows where flight_id matches
    flight_data = parquet_df[parquet_df['flight_id'] == flight_id]
    
    if flight_data.empty:
        print(f"No data found for flight_id {flight_id} on {flight_date}.")
        
    return flight_data
    
def get_flight_data_jet(flight_id, flight_date, parquet_folder_path):
    """
    Extracts flight data for a specific flight_id and date from the respective Parquet file,
    returning only 'flight_id', 'timestamp', 'latitude', and 'longitude' columns.

    Parameters:
    - flight_id: str or int, the unique flight identifier.
    - flight_date: str, the date of the flight in 'YYYY-MM-DD' format.
    - parquet_folder_path: str, path to the folder containing parquet files.

    Returns:
    - flight_data: pandas DataFrame with data points for the specific flight, or empty DataFrame if not found.
    """
    # Generate the path to the respective parquet file
    parquet_file_path = os.path.join(parquet_folder_path, f"{flight_date}.parquet")
    
    # Check if the parquet file exists
    if not os.path.exists(parquet_file_path):
        print(f"Parquet file for date {flight_date} does not exist.")
        return pd.DataFrame()  # Return an empty DataFrame if file is missing
    
    # Read the parquet file into a DataFrame
    try:
        # Only read in the specified columns
        selected_columns = ['flight_id', 'altitude', 'latitude', 'longitude']
        parquet_df = pd.read_parquet(parquet_file_path, columns=selected_columns)
    except Exception as e:
        print(f"Error reading parquet file: {e}")
        return pd.DataFrame()  # Return empty DataFrame on error
    
    # Filter the DataFrame to only include rows where flight_id matches
    flight_data = parquet_df[parquet_df['flight_id'] == flight_id]
    
    if flight_data.empty:
        print(f"No data found for flight_id {flight_id} on {flight_date}.")
        
    return flight_data
    
def get_flight_data_weather(flight_id, flight_date, parquet_folder_path):
    """
    Extracts flight data for a specific flight_id and date from the respective Parquet file,
    returning only 'flight_id', 'timestamp', 'latitude', and 'longitude' columns.

    Parameters:
    - flight_id: str or int, the unique flight identifier.
    - flight_date: str, the date of the flight in 'YYYY-MM-DD' format.
    - parquet_folder_path: str, path to the folder containing parquet files.

    Returns:
    - flight_data: pandas DataFrame with data points for the specific flight, or empty DataFrame if not found.
    """
    # Generate the path to the respective parquet file
    parquet_file_path = os.path.join(parquet_folder_path, f"{flight_date}.parquet")
    
    # Check if the parquet file exists
    if not os.path.exists(parquet_file_path):
        print(f"Parquet file for date {flight_date} does not exist.")
        return pd.DataFrame()  # Return an empty DataFrame if file is missing
    
    # Read the parquet file into a DataFrame
    try:
        # Only read in the specified columns
        selected_columns = ['flight_id', 'altitude', 'vertical_rate', 'u_component_of_wind', 'v_component_of_wind', 'temperature', 'specific_humidity']
        parquet_df = pd.read_parquet(parquet_file_path, columns=selected_columns)
    except Exception as e:
        print(f"Error reading parquet file: {e}")
        return pd.DataFrame()  # Return empty DataFrame on error
    
    # Filter the DataFrame to only include rows where flight_id matches
    flight_data = parquet_df[parquet_df['flight_id'] == flight_id]
    
    if flight_data.empty:
        print(f"No data found for flight_id {flight_id} on {flight_date}.")
        
    return flight_data
    
    
    
def remove_outliers(df, column='altitude', threshold=1000):
    """
    Removes outliers from a Pandas DataFrame based on sudden jumps in the specified column.
    This method removes rows where the difference between consecutive rows exceeds a given threshold.

    Parameters:
    - df: Pandas DataFrame with flight data.
    - column: str, the column name to check for outliers (default is 'altitude').
    - threshold: int, the maximum allowed difference between consecutive rows to consider as valid (default is 1000).

    Returns:
    - df: Pandas DataFrame with outliers removed.
    """
    # Calculate the difference between consecutive rows in the specified column
    df['diff'] = df[column].diff().abs()

    # Keep rows where the difference is less than the threshold or NaN (for the first row)
    df_cleaned = df[(df['diff'] < threshold) | (df['diff'].isna())]

    # Drop the 'diff' column used for filtering
    df_cleaned = df_cleaned.drop(columns=['diff'])
    
    #This sort of cleans a few of the weird altitude patterns
    #It tries to find the lowest altitude and cuts off everything before if its in the first half of the flight
    # Define the midpoint of the DataFrame
    midpoint = len(df_cleaned) // 2
    # Find the first index of the lowest altitude in the first half of the DataFrame
    lowest_altitude_idx = df_cleaned.iloc[:midpoint][column].idxmin()
    # Apply the cutoff using the lowest altitude index found in the first half
    # Cut off all data before the first occurrence of the lowest altitude
    df_cleaned = df_cleaned.loc[lowest_altitude_idx:]

    return df_cleaned


'''
# Read the parquet file
df = pd.read_parquet('2022-01-01.parquet')

# Display the first few rows
print(df.head())

import pandas as pd
'''

if __name__ == "__main__":

    # Example usage
    parquet_folder = 'parquet_files/'
    flight_id = 248763780
    flight_date = '2022-01-01'

    # Get the flight data
    start_time = time.time()
    flight_data = get_flight_data(flight_id, flight_date, parquet_folder)
    aquisition_time = time.time()

    print(f"took {aquisition_time-start_time:.2f} to retrieve flight")

    # Display the flight data
    print(flight_data)
    #print(flight_data.columns)
    plot_flight_data_overview(flight_data)
