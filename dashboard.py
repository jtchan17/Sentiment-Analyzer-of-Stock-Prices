import streamlit as st 
import pandas as pd 
import numpy as np 
import yfinance as yf 
import plotly_express as px 
import altair as alt

#####################################################################
# Page configuration
st.set_page_config(
    page_title="Sentiment Analyzer Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed")

# Initialize connection.
conn = st.connection('mysql', type='sql')

# Loading the data
@st.cache_resource
def load_data(query):
    df = conn.query(query, ttl=600)
    return df
df_fn = load_data('SELECT * from dashboard.financialnews;')
df_sp = load_data('SELECT * from dashboard.stockprice;')
alt.themes.enable("dark")

#####################################################################

#####################################################################
# Side bar (Filtering)
with st.sidebar:
    st.title('Welcome! User')
#####################################################################

#####################################################################
#   Dashboard
st.title('📈 Sentiment Analyzer :blue[Dashboard] of Stock Prices')

#search and filtering
fil_col1, fil_col2, fil_col3, fil_col4 = st.columns([1.25, 2, 4, 1])

with fil_col1:
    years = ['All', '2021', '2022', '2023']
    select_year = st.selectbox('Select Year', years, key='select_year')

with fil_col2:
    company_expander = st.expander(label='Company')
    with company_expander:
        aapl = st.checkbox('AAPL', key='aapl', value=False)
        amzn = st.checkbox('AMZN', key='amzn', value=False)
        meta = st.checkbox('META', key='meta', value=False)
        msft = st.checkbox('MSFT', key='msft', value=False)
        tsla = st.checkbox('TSLA', key='tsla', value=False)
    companies = {'AAPL': aapl, 'AMZN': amzn, 'META': meta, 'MSFT': msft, 'TSLA': tsla}

if 'aapl' not in st.session_state:
    st.session_state['aapl'] = False
if 'amzn' not in st.session_state:
    st.session_state['amzn'] = False
if 'meta' not in st.session_state:
    st.session_state['meta'] = False
if 'msft' not in st.session_state:
    st.session_state['msft'] = False
if 'tsla' not in st.session_state:
    st.session_state['tsla'] = False
if 'select_year' not in st.session_state:
    st.session_state['select_year'] = 'All'

with fil_col3:
    def clear_filters():
        st.session_state['aapl'] = False
        st.session_state['amzn'] = False
        st.session_state['meta'] = False
        st.session_state['msft'] = False
        st.session_state['tsla'] = False
        st.session_state['select_year'] = 'All'

    btn_clear = st.button('Clear All Filter', key='clearFilter', on_click=clear_filters)

with fil_col4:
    st.button('Export 	:arrow_down_small:')

###### Filter based on Year and Company ######
def query(table, year, companies):
    query = f'SELECT * FROM dashboard.{table} WHERE '
    if year != 'All':
        query += f'YEAR(published_date) = {year} ' if table == 'financialnews' else f'YEAR(date) = {year} '
        if companies:
            companies_str = ', '.join(f'"{company}"' for company in companies)
            query += f'AND company IN ({companies_str})'
        return query
    else:
        if companies:
            companies_str = ', '.join(f'"{company}"' for company in companies)
            query += f'company IN ({companies_str})'
        return query

# List of selected companies
selected_companies = [company for company, selected in companies.items() if selected]

# Determine queries based on selected filters
if select_year == 'All' and not selected_companies:
    filtered_df_fn = df_fn
    filtered_df_sp = df_sp
else:
    fn_query = query('financialnews', select_year, selected_companies)
    sp_query = query('stockprice', select_year, selected_companies)
    filtered_df_fn = conn.query(fn_query, ttl=600)
    filtered_df_sp = conn.query(sp_query, ttl=600)

###### Data Visualization/EDA ######
r1c1, r1c2 = st.columns((7, 3), gap='small')
with r1c1:
    st.subheader('Historical Stock Data')
    st.markdown('###### currency in USD')
    # query = "SELECT date, adj_close, company FROM filtered_df_sp;"
    # df_sp1 = conn.query(query, ttl=600)
    fig = px.line(filtered_df_sp, x='date', y='adj_close', height=500, width = 300, template='gridon', color='company')
    st.plotly_chart(fig,use_container_width=True)

with r1c2:
    st.subheader('Highest Price Across Years')
    query = 'SELECT YEAR(date) as Year, company AS Companies, MAX(high) AS Highest FROM dashboard.stockprice WHERE YEAR(date) in (2021, 2022, 2023) GROUP BY Companies, Year ORDER BY Year DESC, Highest DESC;'
    df_highest = conn.query(query, ttl=600)
    st.table(df_highest)

r2c1, r2c2, r2c3 = st.columns((2, 5, 3), gap='small')
with r2c1:
    st.subheader('Gain/Loss :small_red_triangle::small_red_triangle_down:')
    # need to resize the graph and create a function
    # apple_data = yf.download('AAPL', start='2021-01-01', end='2023-12-31')
    # fig = px.line(apple_data, x = apple_data.index, y = apple_data['Adj Close'], title = 'Stock Prices', width= 200, height=400)
    df_fn1 = filtered_df_fn.groupby('sentiment_score').size().reset_index(name='Total')
    # fig = px.bar(df_fn1, y='Total', x='sentiment_score', template='plotly_dark', height=500)
    # st.plotly_chart(fig, use_container_width=True)
    st.area_chart(df_fn1, color='sentiment_score')

with r2c2:
    st.subheader('Historical Stock Data')
    fig = px.bar(filtered_df_sp, x='date', y='adj_close', height=500, width = 300, template='gridon', color='company')
    st.plotly_chart(fig,use_container_width=True)
    # st.bar_chart(df_sp, x='date', y='adj_close', height=500, width = 300, color='company')

with r2c3:
    st.subheader('Sentiment Score Over Time')
    #sentiment score
    df_fn1 = filtered_df_fn.groupby('sentiment_score').size().reset_index(name='Total')
    fig = px.pie(df_fn1, values='Total', names='sentiment_score', template='plotly_dark', hole=0.5)
    fig.update_traces(text=df_fn1['sentiment_score'], textposition='inside')
    st.plotly_chart(fig,use_container_width=True)

r3c1, r3c2 = st.columns((3, 10), gap='small')

with r3c1:
    st.subheader('Top 10 Publishers :newspaper:')
    df_fn1 = (filtered_df_fn.groupby('publisher').size().reset_index(name='Total'))
    df_fn1 = (df_fn1.sort_values(by="Total", ascending=False)).head(10)
    st.dataframe(df_fn1,
                 column_order=("publisher", "Total"),
                 hide_index=True,
                 width=None,
                 column_config={
                     "publisher": st.column_config.TextColumn("Publisher",),
                     "Total": st.column_config.ProgressColumn("Total",format="%f",min_value=0,max_value=max(df_fn1.Total),)
                     }
                 )

with r3c2:
    st.subheader('Publishers :newspaper:')
    df_fn1 = filtered_df_fn.groupby('publisher').size().reset_index(name='Total')
    fig = px.bar(df_fn1,x='Total', y='publisher', template='seaborn')
    fig.update_traces(text=df_fn1['publisher'], textposition='inside')
    st.plotly_chart(fig, use_container_width=True, height = 500)

pricing_data, news = st.tabs(['Pricing Data', 'News'])

with pricing_data:
    st.subheader('Price Movements')
    with st.container(height=500, border=True):
        st.table(filtered_df_sp)
    csv=filtered_df_sp.to_csv(index = False).encode('utf-8')
    st.download_button(label='Download Historical Data', data= csv, file_name='Historical Data.csv')

with news:
    st.subheader('News')
    with st.container(height=500, border=True):
        df_fn1 = filtered_df_fn.sort_values(by="published_date", ascending=True)
        st.table(df_fn)
    csv=filtered_df_fn.to_csv(index = False).encode('utf-8')
    st.download_button(label='Download Financial News', data= csv, file_name='Financial News.csv')

st.subheader('Summary')

#####################################################################

