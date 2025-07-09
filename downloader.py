import yfinance as yf
import pandas as pd

# 下载数据
dow = yf.download('^DJI', start='1985-01-01')
sp500 = yf.download('^GSPC', start='1985-01-01')
nasdaq = yf.download('^IXIC', start='1985-01-01')

# 保存为CSV
dow.to_csv('dow_jones_data.csv')
sp500.to_csv('sp500_data.csv')
nasdaq.to_csv('nasdaq_data.csv')