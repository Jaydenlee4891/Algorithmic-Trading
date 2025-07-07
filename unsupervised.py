from statsmdoels.regression.rolling import RollingOLS
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
import numpy as np
import datetime as dt
import yfinance as yf
import pandas_ta
import warnings

warnings.filterwarnings('ignore')

sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]

sp500['Symbol'] = sp500['Symbol'].str.replace('.','-')

symbols_list = sp500['Symbol'].unique().tolist()

end_date = '2025-07-06'

start_date = pd.to_datetime(end_date)-pd.DateOffset(365*8)

df = yf.download(tickers=symbols_list, start=start_date,end=end_date)

df = df.stack()

#Garman-Klass
df['garman_klass_vol'] = (
    0.5 * (np.log(df['high'] / df['low']))**2 
    - (2 * np.log(2) - 1) * (np.log(df['close'] / df['open']))**2
)

#rsi
df['rsi'] = df.groupby(level=1)['close'].transform(lambda x: pandas_ta.rsi(close=x,length=20))

#Bollinger Bands

