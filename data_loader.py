
import yfinance as yf
import pandas as pd

def download_data(ticker, start_date, end_date):
    """
    Downloads historical stock data from Yahoo Finance.
    
    
    Args:
        ticker (str): The stock ticker symbol (e.g., 'AAPL').
        start_date (str): The start date for the data in 'YYYY-MM-DD' format.
        end_date (str): The end date for the data in 'YYYY-MM-DD' format.
        
    Returns:
        pd.DataFrame: A pandas DataFrame containing the historical stock data,
                      or None if the download fails.
    """
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            print(f"No data found for ticker {ticker} from {start_date} to {end_date}.")
            return None
        print(f"Successfully downloaded data for {ticker}.")
        return data
    except Exception as e:
        print(f"An error occurred during data download: {e}")
        return None

if __name__ == '__main__':
    # Example usage:
    ticker_symbol = 'AAPL'
    start = '2020-01-01'
    end = '2023-12-31'
    stock_data = download_data(ticker_symbol, start, end)
    
    if stock_data is not None:
        print("\nData Head:")
        print(stock_data.head())
        print("\nData Tail:")
        print(stock_data.tail())

