import pandas as pd

# Load the main challenge set
challenge_set = pd.read_csv('challenge_set.csv')

# Load the additional datasets
ground_speed_data = pd.read_csv('ground_speed_data_challenge_set.csv')
jet_stream_data = pd.read_csv('jet_stream_data_challenge_set.csv')
weather_data = pd.read_csv('weather_data_challenge_set.csv')
oil_prices = pd.read_csv('oil_prices_2022.csv')
msci_world = pd.read_csv('msci_world_2022.csv')

# Merge ground_speed_data with challenge_set on 'flight_id'
challenge_set = pd.merge(challenge_set, ground_speed_data, on='flight_id', how='inner')

# Drop rows with missing values
challenge_set = challenge_set.dropna()

# Merge jet_stream_data with challenge_set on 'flight_id'
challenge_set = pd.merge(challenge_set, jet_stream_data, on='flight_id', how='inner')

# Drop rows with missing values
challenge_set = challenge_set.dropna()

# Merge weather_data with challenge_set on 'flight_id'
challenge_set = pd.merge(challenge_set, weather_data, on='flight_id', how='inner')

# Drop rows with missing values
challenge_set = challenge_set.dropna()

# Merge oil_prices with challenge_set on 'date', filling missing dates with nearest available values
oil_prices['date'] = pd.to_datetime(oil_prices['date'])
challenge_set['date'] = pd.to_datetime(challenge_set['date'])
challenge_set = pd.merge_asof(challenge_set.sort_values('date'), oil_prices.sort_values('date'), on='date', direction='nearest')

# Merge msci_world with challenge_set on 'date', filling missing dates with nearest available values
msci_world['date'] = pd.to_datetime(msci_world['date'])
challenge_set = pd.merge_asof(challenge_set.sort_values('date'), msci_world.sort_values('date'), on='date', direction='nearest')

# Export the final combined dataset to a new CSV file
challenge_set.to_csv('challenge_set_v6.csv', index=False)

print("Data successfully merged and saved as 'challenge_set_v6.csv'")
