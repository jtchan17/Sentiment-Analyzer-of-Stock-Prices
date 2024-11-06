import streamlit as st 
import pandas as pd 
import plotly_express as px 
import altair as alt
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import matplotlib.pyplot as plt
import pdfkit
import jinja2
# from jinja2 import Environment, PackageLoader, select_autoescape, FileSystemLoader
from streamlit.components.v1 import iframe
import plotly.io as pio
import matplotlib.pyplot as plt
import os

#####################################################################
PDF_TEMPLATE_FILE = 'PDFtemplate.html'
IMG_FOLDER = os.path.join(os.getcwd(), 'image')

#####################################################################
# Redirect to app.py if not logged in, otherwise show the navigation menu
# menu_with_redirect()

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
# Side bar (Login)
with st.sidebar:
    st.title(f'Welcome {st.session_state.role}')


#####################################################################

#####################################################################
#   Dashboard
st.title('📈 Sentiment Analyzer :blue[Dashboard] of Stock Prices')

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


#====================================================================
#ROW 1
r1c1, r1c2 = st.columns((7, 3), gap='small')
with r1c1:
    st.subheader('Historical Stock Data')
    st.markdown('###### currency in USD')
    chart_HistoricalStockData = px.line(df_sp, x='date', y='adj_close', template='gridon', color='company')
    st.plotly_chart(chart_HistoricalStockData, key='chart_HistoricalStockData', use_container_width=True)   

with r1c2:
    st.subheader('Highest Price Across Years')
    query = 'SELECT YEAR(date) as Year, company AS Companies, MAX(high) AS Highest FROM dashboard.stockprice WHERE YEAR(date) in (2021, 2022, 2023) GROUP BY Companies, Year ORDER BY Year DESC, Highest DESC;'
    df_highest = conn.query(query, ttl=600)

    # def filter_years(df):
    #     df['date'] = pd.to_datetime(df['date'])
    #     # if year != 'All':
    #     #     # df_filtered = df[df['date']].dt.year
    #     #     df['year'] = df['date'].dt.year
    #     # else:
    #     #     df['year'] = df['date'].dt.year
            
    #     result = df.groupby(['year', 'company']).agg({'high': 'max'}).reset_index()
    #     result.rename(columns={'year': 'Year', 'company': 'Companies', 'high': 'Highest'}, inplace=True)
    #     result = result.sort_values(by=['Year', 'Highest'], ascending=[True, False])
    #     result = result.reset_index(drop=True)
    #     return result
    
    # table_HighestPriceAcrossYear = filter_years(df_sp)
    st.table(df_highest)
#====================================================================
#ROW 2
r2c1, r2c2 = st.columns((3, 5), gap='small')
with r2c1:
    st.subheader('Number of News Across Companies')
    table_NumberofNewsAcrossCompanies = df_fn.groupby('company')['title'].count().reset_index(name='Total')
    st.table(table_NumberofNewsAcrossCompanies)

with r2c2:
    st.subheader('Frequency of News Over Time')
    df_article_freq = df_fn.groupby(['published_date', 'company']).size().unstack(fill_value=0)
    df_article_freq = df_article_freq.reset_index()
    df_melted = pd.melt(df_article_freq, id_vars='published_date', var_name='company', value_name='frequency')
    chart_FrequencyofNewsOverTime = px.line(df_melted, x='published_date', y="frequency", template='gridon', color='company')
    st.plotly_chart(chart_FrequencyofNewsOverTime,use_container_width=True)
#====================================================================
#ROW 3
r3c1, r3c2 = st.columns((5,5), gap='small')
with r3c1:
    st.subheader('Sentiment Score Over Time')

    #sentiment score
    def plot_pie():
        df_sentiment = df_fn.groupby('sentiment_score').size().reset_index(name='Total')
        chart_SentimentScoreOverTime = px.pie(df_sentiment, values='Total', names='sentiment_score', color="sentiment_score",
                                            color_discrete_map={'negative': '#EF553B', 
                                                                'positive': '#00CC96', 
                                                                'neutral': '#636EFA'},
                                            hole=0.5)
        chart_SentimentScoreOverTime.update_traces(textposition='inside')
        return chart_SentimentScoreOverTime
    chart_SentimentScoreOverTime = plot_pie()
    st.plotly_chart(chart_SentimentScoreOverTime, use_container_width=True)

with r3c2:
    st.subheader('Sentiment Score Across Companies')
    grouped_sentiment_df_fn = df_fn.groupby(['company', 'sentiment_score']).size().unstack(fill_value=0)
    df_sentiment_freq = grouped_sentiment_df_fn.reset_index()
    df_melted = pd.melt(df_sentiment_freq, id_vars='company', var_name='sentiment_score', value_name='frequency')
    chart_SentimentScoreAcrossCompanies = alt.Chart(df_melted).mark_bar().encode(
        x="sentiment_score",
        y="frequency",
        color="company"
    )
    st.altair_chart(chart_SentimentScoreAcrossCompanies, use_container_width=True)

    grouped_sentiment_df_fn = df_fn.groupby(['company', 'sentiment_score']).size().unstack(fill_value=0)
    table_SentimentFrequency = grouped_sentiment_df_fn.reset_index()
    grouped_sentiment_df_fn.rename(columns={'company': 'Companies', 'negative': 'Negative', 'neutral': 'Neutral', 'positive': 'Positive'}, inplace=True)
    table_SentimentFrequency = grouped_sentiment_df_fn
    st.table(table_SentimentFrequency)
    
#====================================================================
#ROW 4
r4c1, r4c2 = st.columns((3, 7), gap='small')

with r4c1:
    st.subheader('Top 10 Publishers :newspaper:')
    df_fn1 = (df_fn.groupby('publisher').size().reset_index(name='Total'))
    table_TopPublishers = (df_fn1.sort_values(by="Total", ascending=False)).head(10)
    st.dataframe(table_TopPublishers,
                column_order=("publisher", "Total"),
                hide_index=True,
                width=None,
                column_config={
                    "publisher": st.column_config.TextColumn("Publisher",),
                    "Total": st.column_config.ProgressColumn("Total",format="%f",min_value=0,max_value=max(df_fn1.Total),)
                    }
                )

with r4c2:
    st.subheader('Publishers :newspaper:')
    df_fn1 = df_fn.groupby('publisher').size().reset_index(name='Total')
    chart_Publishers = px.bar(df_fn1,x='Total', y='publisher', template='seaborn')
    chart_Publishers.update_traces(text=df_fn1['publisher'], textposition='inside')
    st.plotly_chart(chart_Publishers, use_container_width=True, height = 1000)

