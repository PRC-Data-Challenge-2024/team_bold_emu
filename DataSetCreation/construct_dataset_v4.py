import pandas as pd
import os
from tqdm import tqdm
from read_parquet_v3 import get_flight_data, remove_outliers, get_flight_data_lift_off, get_flight_data_jet, get_flight_data_weather
import argparse
from plotting import plot_flight_data_overview, plot_take_off, plot_lift_off, plot_whole_take_off
from total_g_v3 import ground_speed_data, append_ground_speed_data_to_csv, jet_stream_data, append_jet_stream_data_to_csv, weather_data, append_weather_data_to_csv

# Set up argument parsing
parser = argparse.ArgumentParser(description='Process flight data.')
parser.add_argument('--challenge', action='store_true', help='Use challenge set CSV.')
parser.add_argument('--submission', action='store_true', help='Use submission set CSV.')
parser.add_argument('--submissionv2', action='store_true', help='Use final submission set CSV.')

# Add the -L flag for lift-off calculations
parser.add_argument('-L', '--lift_off', action='store_true', help='Perform lift off calculations.')

# Add the -J flag for Jet-stream calculations
parser.add_argument('-J', '--jet_stream', action='store_true', help='Perform jet stream calculations.')

# Add the -W flag for Weather calculations
parser.add_argument('-W', '--weather', action='store_true', help='Perform weather and climb calculations.')

args = parser.parse_args()

# Determine which dataset to use based on the arguments
if args.challenge:
    df = pd.read_csv('challenge_set.csv')
    filename = 'challenge_set'
    output_filename = 'ground_speed_data_challenge.csv'
elif args.submission:
    df = pd.read_csv('submission_set.csv')
    filename = 'submission_set'
    output_filename = 'ground_speed_data_submission.csv'
elif args.submissionv2:
    df = pd.read_csv('final_submission_set.csv')
    filename = 'final_submission_set'
    output_filename = 'ground_speed_data_submission_v2.csv'
else:
    raise ValueError('Please specify either --challenge or --submission.')

# Ensure the 'rip' directory exists
if not os.path.exists('rip'):
    os.makedirs('rip')

# Set the folder path for parquet files
parquet_folder_path = "parquet_files"

#cheap manual restart code here
# Define the cutoff_flight_id
#cutoff_flight_id = 251824649  # Replace with the desired flight ID
# Filter the DataFrame to keep only rows where flight_id is greater than or equal to cutoff_flight_id
#df = df[df['flight_id'] >= cutoff_flight_id]

#counter = 0

ripperonis = []
ripperoni_counter = 0

# Loop through each row with tqdm progress bar
for index, row in tqdm(df.iterrows(), total=df.shape[0]):

    #counter += 1
    #if counter > 50:
    #    break

    try:
        # Extract flight_id and date
        flight_id = row['flight_id']
        flight_date = row['date']
        
        #printing shit for debugging
        #print(flight_id)
        #print(flight_date)
        
        # Use flight_id and date in your custom method
        # e.g., some_method(flight_id, date)
        #print(f"Processing flight_id: {flight_id}, date: {date}")     
       
        
        #Lift off stuff
        if args.lift_off:
            # Get the flight data from the parquet files for each flight
            flight_data = get_flight_data_lift_off(flight_id, flight_date, parquet_folder_path)
            #Remove possible outliers
            df_pandas = remove_outliers(flight_data)
            #calculate time to lift off, and ground speed at lift off
            results = ground_speed_data(df_pandas, 'timestamp', 'groundspeed', 'altitude')
            #print(results)
            #finally append the results to a csv
            append_ground_speed_data_to_csv(filename,flight_id, results)
            
            
        if args.jet_stream:
            #Jet stream calculations here
            #Just for shits and giggles this will also include the average altitude
            # Get the flight data from the parquet files for each flight
            flight_data = get_flight_data_jet(flight_id, flight_date, parquet_folder_path)
            #Remove possible outliers
            df_pandas = remove_outliers(flight_data)
            
            #calculate average altitude, and jet stream coefficient
            #this method needs other positional column arguments
            results = jet_stream_data(df_pandas, 'altitude', 'latitude', 'longitude')           
            #finally append the results to a csv
            append_jet_stream_data_to_csv(filename,flight_id, results)
            
        if args.weather:
            #Jet stream calculations here
            #Just for shits and giggles this will also include the average altitude
            # Get the flight data from the parquet files for each flight
            #returns columns 'vertical_rate', 'u_component_of_wind', 'v_component_of_wind', 'temperature', 'specific_humidity'
            flight_data = get_flight_data_weather(flight_id, flight_date, parquet_folder_path)
            #Remove possible outliers
            df_pandas = remove_outliers(flight_data)
                    
            #calculate average altitude, and jet stream coefficient
            #this method needs other positional column arguments
            #got tired of named columns, decided for hard coded ones
            results = weather_data(df_pandas) 
            #finally append the results to a csv
            append_weather_data_to_csv(filename,flight_id, results)
            

        #If we want to create the take off plots, this is the method
        #plot_whole_take_off(df_pandas, 'timestamp', 'groundspeed', 'altitude', flight_id, pre_lift_off_seconds=60, time_limit=30)
    except:
        print("rip")
        ripperoni_counter += 1
        ripperonis.append(flight_id)  # Assuming you're appending the first flight_id or whichever flight_id is relevant
        
        # Save ripperonis to a file called 'ripperoni.txt'
        with open('rip/ripperoni_' + filename + '.txt', 'w') as file:
            for flight_id in ripperonis:
                file.write(f"{flight_id}\n")
        
        # Save ripperoni_counter to a file called 'ripperoni_counter.txt'
        with open('rip/ripperoni_counter_' + filename + '.txt', 'w') as file:
            file.write(str(ripperoni_counter))