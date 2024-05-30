import streamlit as st 
import pandas as pd 
import numpy as np 
import yfinance as yf 
import plotly_express as px 
import altair as alt

#######################
# Page configuration
st.set_page_config(
    page_title="Sentiment Analyzer Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")

#######################

#######################
# Graph/Chart/Plot

#pie chart


#######################

#######################
#   Dashboard
st.title('📈 Stock Prices Dashboard')

col = st.columns((1.5, 4.5, 2), gap='medium')

df_aapl = pd.read_csv('Historical stock prices/AAPL_historical_data.csv')

with col[0]:
    st.markdown('#### Data Visualization')
    # need to resize the graph and create a function
    apple_data = yf.download('AAPL', start='2021-01-01', end='2023-12-31')
    fig = px.line(apple_data, x = apple_data.index, y = apple_data['Adj Close'], title = 'AAPL')
    st.plotly_chart(fig)

with col[1]:
    st.markdown('#### Data Visualization')
    apple_data = yf.download('AAPL', start='2021-01-01', end='2023-12-31')
    fig = px.line(apple_data, x = apple_data.index, y = apple_data['Adj Close'], title = 'AAPL')
    st.plotly_chart(fig)
#######################

pricing_data, news = st.tabs(['Pricing Data', 'Top 10 News'])

with pricing_data:
    st.header('Price Movements')
    aapl = df_aapl
    st.write(aapl)

with news:
    st.header('News')