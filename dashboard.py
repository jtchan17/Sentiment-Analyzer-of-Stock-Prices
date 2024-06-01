import streamlit as st 
import pandas as pd 
import numpy as np 
import yfinance as yf 
import plotly_express as px 
import altair as alt
import matplotlib
import matplotlib.pyplot as plt

#######################
# Page configuration
st.set_page_config(
    page_title="Sentiment Analyzer Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed")

# Initialize connection.
conn = st.connection('mysql', type='sql')
df_fn = conn.query('SELECT * from dashboard.financialnews;', ttl=600)
df_sp = conn.query('SELECT * from dashboard.stockprice;', ttl=600)

#Loading the data
# @st.cache_data
# def load_data(url):
#     df = pd.read_csv(url)
#     return df

alt.themes.enable("dark")

#######################

#######################
# Side bar (Filtering)
with st.sidebar:
    st.title('Welcome! User')
    # st.sidebar.header('Select what company to display')
    # company = df_fn['company'].unique().tolist()
    # company_selected = st.sidebar.multiselect('Companies', company, company)
#######################

#######################
# Graph/Chart/Plot

#Sentiment Pie chart
    

#######################

#######################
#   Dashboard
st.title('📈 Stock Prices :blue[Dashboard]')

#search and filtering
fil_col1, fil_col2, fil_col3, fil_col4 = st.columns([1.25, 2, 4, 1])

with fil_col1:
    filter_expander = st.expander(label='Year')
    with filter_expander:
        st.checkbox('2021')
        st.checkbox('2022')
        st.checkbox('2023')

with fil_col2:
    filter_expander = st.expander(label='Company')
    with filter_expander:
        st.checkbox('AAPL', key='aapl', value=False)
        st.checkbox('AMZN', key='amzn', value=False)
        st.checkbox('META', key='meta', value=False)
        st.checkbox('MSFT', key='msft', value=False)
        st.checkbox('TSLA', key='tsla', value=False)
with fil_col3:
    btn_clear = st.button('Clear All Filter', key='clearFilter')
    # is_btn_clicked = st.session_state('clearFilter')

with fil_col4:
    st.button('Export 	:arrow_down_small:')

#Data Visualization/EDA
col = st.columns((2, 5, 3), gap='small')
with col[0]:
    st.subheader('Gain/Loss :small_red_triangle::small_red_triangle_down:')
    # need to resize the graph and create a function
    apple_data = yf.download('AAPL', start='2021-01-01', end='2023-12-31')
    fig = px.line(apple_data, x = apple_data.index, y = apple_data['Adj Close'], title = 'Stock Prices', width= 200, height=400)
    st.plotly_chart(fig, use_container_width=True)

with col[1]:
    st.subheader('Historical Stock Data')
    query = "SELECT date, adj_close, company FROM dashboard.stockprice;"
    df_sp1 = conn.query(query, ttl=600)
    fig = px.line(df_sp, x='date', y='adj_close', height=500, width = 300, template="gridon", color="company")
    st.plotly_chart(fig,use_container_width=True)

with col[2]:
    st.subheader('Sentiment score over time')
    query = "SELECT publisher, COUNT(*) AS sum FROM dashboard.financialnews GROUP BY publisher;"
    df_fn1 = conn.query(query, ttl=600)
    fig = px.pie(df_fn1, values='sum', names='publisher', template='plotly_dark', hole=0.5)
    fig.update_traces(text=df_fn1['publisher'], textposition='inside')
    st.plotly_chart(fig,use_container_width=False)
    # fig, ax = plt.subplots(figsize=(5, 5))
    # df = df_fn['sentiment_score']
    # colors = ['gray', 'red', 'green']
    # ax.pie(sentiment_score, labels=(sentiment_score.index + ' (' + sentiment_score.map(str)+ ')'), wedgeprops = { 'linewidth' : 3, 'edgecolor' : 'white'}, colors=colors, startangle=90)

pricing_data, news = st.tabs(['Pricing Data', 'News'])

with pricing_data:
    st.subheader('Price Movements')
    with st.container(height=500, border=False):
        st.table(df_sp)

with news:
    st.subheader('News')
    for i in range(10):
        company = df_fn['company']
        st.markdown('##### :green[Company]')
        st.write(company[i])
        news_title = df_fn['title']
        st.markdown('##### Headline:')
        st.write(news_title[i])

st.subheader('Summary')

#######################

