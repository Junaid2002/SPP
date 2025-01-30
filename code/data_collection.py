import yfinance as yf
import pandas as pd

def collect_data(symbol, start_date, end_date):
    data = yf.download(symbol, start=start_date, end=end_date)
    data.to_csv(f'{symbol}_data.csv')
    print(f"Data for {symbol} saved as {symbol}_data.csv")
    return data

if __name__ == '__main__':
    stock_symbol = 'AAPL'
    collect_data(stock_symbol, '2010-01-01', '2025-01-01')
