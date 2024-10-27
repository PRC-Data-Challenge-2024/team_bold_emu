import yfinance as yf
import pandas as pd

# Define the MSCI World ticker symbol (using ISIN or WKN).
ticker = "URTH"  # This represents MSCI World

# Set the date range for the year 2022
start_date = "2022-01-01"
end_date = "2022-12-31"

# Download the data using yfinance
msci_world_data = yf.download(ticker, start=start_date, end=end_date)

# Display the MSCI World data
print(msci_world_data)

# Save the data to a CSV file
msci_world_data.to_csv("msci_world_2022.csv")
