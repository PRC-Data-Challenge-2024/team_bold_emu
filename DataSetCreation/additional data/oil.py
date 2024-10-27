import yfinance as yf
import pandas as pd

# Define the oil ticker symbol. WTI Crude Oil is commonly represented as 'CL=F'.
ticker = "CL=F"

# Set the date range for the year 2022
start_date = "2022-01-01"
end_date = "2022-12-31"

# Download the data using yfinance
oil_data = yf.download(ticker, start=start_date, end=end_date)

# Display the oil price data
print(oil_data)

# If you want to save the data to a CSV file, you can use this line
oil_data.to_csv("oil_prices_2022.csv")