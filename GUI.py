# Import the necessary libraries
import yfinance as yf
import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
#from mpl_finance import candlestick_ohlc

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from scipy.stats import randint

import streamlit as st
import sys
import subprocess
import os

import joblib
from datetime import datetime
import time

from mplfinance.original_flavor import candlestick_ohlc
import mplfinance as mpf
import matplotlib.dates as mpl_dates

import warnings
warnings.filterwarnings('ignore')

# define variables
crypt_coins = "BTC,BNB,ETH,DOT,XRP,ADA,LTC,XLM,DASH,BCH,DOGE,AAVE,SOL,MIOTA,NEXO,ETC,XMR,WAVES,ICX,EOS"
ccoin_list = crypt_coins.split(",")  # ['BTC', ' BNB'...]
ccoin_dictionary = {coin: ccoin_list.index(coin) for coin in ccoin_list}

# load model, encoder, and scaler
scaler_ = joblib.load('scaler.joblib')
label_encoder_ = joblib.load('label_encoder.joblib')
model_ = joblib.load('model.joblib')


# define functions
def getcoindata(coin_name_, interval_="1d"):
    df2 = yf.Ticker(f"{coin_name_}-USD").history(start='2022-04-01', end=str(datetime.today().date()),interval=interval_)
    df2 = pd.DataFrame(df2)
    df2['crypto_name'] = coin_name_

    df2.reset_index()
    df2.drop(["Dividends", "Stock Splits"], axis=1, inplace=True)
    return df2


def myScaler_prod(data):
    global scaler_
    data_ = data.copy()
    data_scaled_ = scaler_.transform(data_)
    data_scaled_ = pd.DataFrame(data_scaled_, columns=data_.columns)
    data_scaled_.index = data_.index
    return data_scaled_


def myInverseScale_prod(data):
    global scaler_
    data_scaler_inverse = scaler_.inverse_transform(data)
    data_scaler_inverse = pd.DataFrame(data_scaler_inverse, columns=data.columns)
    data_scaler_inverse.index = data.index
    return data_scaler_inverse


def execute_program(coin="BTC", time_string="1d"):
    global model_
    # fetch data
    df_test = getcoindata(coin_name_=coin, interval_=time_string)

    # label encoding
    df_test["crypto_name_encoded"] = label_encoder_.transform(df_test["crypto_name"])

    # scale
    to_scale_ = df_test[["Open", "High", "Low", "Close", "crypto_name_encoded"]]
    df_scaled_ = myScaler_prod(to_scale_)

    # predict
    predicted_ = model_.predict(df_scaled_[["Open", "High", "Low", "crypto_name_encoded"]])

    # inverse scale
    inverse_predicted = df_scaled_.copy()
    inverse_predicted["Close"] = predicted_
    inverse_predicted = myInverseScale_prod(inverse_predicted[["Open", "High", "Low", "Close", "crypto_name_encoded"]])

    # merge
    df_test["predicted"] = inverse_predicted["Close"]

    # return dataset
    return df_test


# Streamlit sidebar
with st.sidebar:
    st.header("MENU BAR")
    select_option = st.selectbox('Select Coin', tuple(ccoin_list))
    interval = st.selectbox('Select timeframe', tuple(['1d', '1wk', '1mo', '3mo']))
    submit_button = st.button('Predict', use_container_width=True)

# Streamlit main content
st.header("SoliGence Cryptocurrency Price Prediction")
if submit_button:
    with st.spinner('Loading Please wait...'):
        data = execute_program(coin=select_option, time_string=interval)
        data.at[data.index[-1], 'Close'] = data.iloc[-1]["predicted"]
        starting_price = data.iloc[-1]["Open"]
        target_price = data.iloc[-1]["Close"]
        time.sleep(5)
    st.success('Completed ✅')
    st.write("The last candlestick in yellow color represents the predicted data.")
    st.write(f"Starting price: {round(starting_price, 2)} USD, Target price: {round(target_price, 2)} USD")

    # Plot
    mc = mpf.make_marketcolors(up='yellow', down='yellow')
    s = mpf.make_mpf_style(base_mpf_style='yahoo', marketcolors=mc)
    nans = [float('nan')] * len(data)
    cdf = pd.DataFrame(dict(Open=nans, High=nans, Low=nans, Close=nans), index=data.index)
    cdf.loc[str(data.tail(1).index[0])] = data.loc[str(data.tail(1).index[0])]
    fig, ax1 = mpf.plot(data[-30:], type='candle', style='yahoo', returnfig=True, title=select_option)
    mpf.plot(cdf[-30:], type='candle', style=s, ax=ax1[0])

    st.pyplot(fig)