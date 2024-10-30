import streamlit as st 
import pandas as pd 
import plotly_express as px 
import altair as alt
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import matplotlib.pyplot as plt
import io
import base64
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
# Side bar (Login)
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
    popover = st.popover("Select Year")
    select_year = popover.radio(label='Select Year',options=years, key='select_year', label_visibility="collapsed")

with fil_col2:
    popover = st.popover("Company")
    aapl = popover.checkbox('AAPL', key='aapl', value=True)
    amzn = popover.checkbox('AMZN', key='amzn', value=True)
    meta = popover.checkbox('META', key='meta', value=True)
    msft = popover.checkbox('MSFT', key='msft', value=True)
    tsla = popover.checkbox('TSLA', key='tsla', value=True)
    companies = {'AAPL': aapl, 'AMZN': amzn, 'META': meta, 'MSFT': msft, 'TSLA': tsla}

# if 'aapl' not in st.session_state:
#     st.session_state['aapl'] = True
# if 'amzn' not in st.session_state:
#     st.session_state['amzn'] = True
# if 'meta' not in st.session_state:
#     st.session_state['meta'] = True
# if 'msft' not in st.session_state:
#     st.session_state['msft'] = True
# if 'tsla' not in st.session_state:
#     st.session_state['tsla'] = True
# if 'select_year' not in st.session_state:
#     st.session_state['select_year'] = 'All'

with fil_col3:
    def clear_filters():
        st.session_state['aapl'] = True
        st.session_state['amzn'] = True
        st.session_state['meta'] = True
        st.session_state['msft'] = True
        st.session_state['tsla'] = True
        st.session_state['select_year'] = 'All'
        st.session_state['positive'] = True
        st.session_state['negative'] = True
        st.session_state['neutral'] = True

    btn_clear = st.button('Clear All Filter', key='clearFilter', on_click=clear_filters)

#filcol_4



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
    filtered_df_fn = conn.query(fn_query, ttl=10000)
    filtered_df_sp = conn.query(sp_query, ttl=10000)

#====================================================================
#ROW 1
r1c1, r1c2 = st.columns((7, 3), gap='small')
with r1c1:
    st.subheader('Historical Stock Data')
    st.markdown('###### currency in USD')
    chart_HistoricalStockData = px.line(filtered_df_sp, x='date', y='adj_close', template='gridon', color='company')
    st.plotly_chart(chart_HistoricalStockData,use_container_width=True)   

with r1c2:
    st.subheader('Highest Price Across Years')
    # query = 'SELECT YEAR(date) as Year, company AS Companies, MAX(high) AS Highest FROM dashboard.stockprice WHERE YEAR(date) in (2021, 2022, 2023) GROUP BY Companies, Year ORDER BY Year DESC, Highest DESC;'
    # df_highest = conn.query(query, ttl=600)

    def filter_years(df, year):
        df['date'] = pd.to_datetime(df['date'])
        if year != 'All':
            # df_filtered = df[df['date']].dt.year
            df['year'] = df['date'].dt.year
        else:
            df['year'] = df['date'].dt.year
            
        result = df.groupby(['year', 'company']).agg({'high': 'max'}).reset_index()
        result.rename(columns={'year': 'Year', 'company': 'Companies', 'high': 'Highest'}, inplace=True)
        result = result.sort_values(by=['Year', 'Highest'], ascending=[True, False])
        result = result.reset_index(drop=True)
        return result
    
    table_HighestPriceAcrossYear = filter_years(filtered_df_sp, select_year)
    st.table(table_HighestPriceAcrossYear)
#====================================================================
#ROW 2
r2c1, r2c2 = st.columns((3, 5), gap='small')
with r2c1:
    st.subheader('Number of News Across Companies')
    table_NumberofNewsAcrossCompanies = filtered_df_fn.groupby('company')['title'].count().reset_index(name='Total')
    st.table(table_NumberofNewsAcrossCompanies)

with r2c2:
    st.subheader('Frequency of News Over Time')
    df_article_freq = filtered_df_fn.groupby(['published_date', 'company']).size().unstack(fill_value=0)
    df_article_freq = df_article_freq.reset_index()
    df_melted = pd.melt(df_article_freq, id_vars='published_date', var_name='company', value_name='frequency')
    chart_FrequencyofNewsOverTime = px.line(df_melted, x='published_date', y="frequency", template='gridon', color='company')
    st.plotly_chart(chart_FrequencyofNewsOverTime,use_container_width=True)
#====================================================================
#ROW 3
popover = st.popover("Choose sentiments to display")
positive = popover.checkbox('Positive', key='positive', value=True)
negative = popover.checkbox('Negative', key='negative', value=True)
neutral = popover.checkbox('Neutral', key='neutral', value=True)

# List of selected companies
sentiments = {'positive': positive, 'negative': negative, 'neutral': neutral}
selected_sentiments = [sentiment for sentiment, selected in sentiments.items() if selected]

def filter_sentiment(df):
    if selected_sentiments:
        df = df[df['sentiment_score'].isin(selected_sentiments)]
    return df

r3c1, r3c2 = st.columns((5,5), gap='small')
with r3c1:
    st.subheader('Sentiment Score Over Time')

    #sentiment score
    def plot_pie():
        df_sentiment = filtered_df_fn.groupby('sentiment_score').size().reset_index(name='Total')
        df_fn1 = filter_sentiment(df_sentiment)
        chart_SentimentScoreOverTime = px.pie(df_fn1, values='Total', names='sentiment_score', template='plotly_dark', hole=0.5)
        chart_SentimentScoreOverTime.update_traces(textposition='inside')
        return chart_SentimentScoreOverTime
    chart_SentimentScoreOverTime = plot_pie()
    st.plotly_chart(chart_SentimentScoreOverTime, use_container_width=True)

with r3c2:
    st.subheader('Sentiment Score Across Companies')
    grouped_sentiment_df_fn = filtered_df_fn.groupby(['company', 'sentiment_score']).size().unstack(fill_value=0)
    df_sentiment_freq = grouped_sentiment_df_fn.reset_index()
    df_melted = pd.melt(df_sentiment_freq, id_vars='company', var_name='sentiment_score', value_name='frequency')
    df_fn1 = filter_sentiment(df_melted)
    chart_SentimentScoreAcrossCompanies = alt.Chart(df_fn1).mark_bar().encode(
        x="sentiment_score",
        y="frequency",
        color="company"
    )
    st.altair_chart(chart_SentimentScoreAcrossCompanies, use_container_width=True)

    df_fn1 = filter_sentiment(filtered_df_fn)
    grouped_sentiment_df_fn = df_fn1.groupby(['company', 'sentiment_score']).size().unstack(fill_value=0)
    table_SentimentFrequency = grouped_sentiment_df_fn.reset_index()
    st.table(table_SentimentFrequency)
    
#====================================================================
#ROW 4
r4c1, r4c2 = st.columns((3, 7), gap='small')

with r4c1:
    st.subheader('Top 10 Publishers :newspaper:')
    df_fn1 = (filtered_df_fn.groupby('publisher').size().reset_index(name='Total'))
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
    df_fn1 = filtered_df_fn.groupby('publisher').size().reset_index(name='Total')
    chart_Publishers = px.bar(df_fn1,x='Total', y='publisher', template='seaborn')
    chart_Publishers.update_traces(text=df_fn1['publisher'], textposition='inside')
    st.plotly_chart(chart_Publishers, use_container_width=True, height = 1000)

#====================================================================
#Sentiment Analyzer
st.subheader('Sentiment Analyzer')
st.markdown('#### Please put your financial headline here: ')
headline_input = st.text_input('headline', label_visibility="collapsed")

# Load the tokenizer and model
tokenizer = BertTokenizer.from_pretrained('./model')
model = BertForSequenceClassification.from_pretrained('./model')

# Tokenize the input
inputs = tokenizer(headline_input, return_tensors='pt')

# Perform inference
with torch.no_grad():
    outputs = model(**inputs)

# Get the prediction
logits = outputs.logits
prediction = torch.argmax(logits, dim=1).item()
sentiments = {0: 'Neutral', 1: 'Negative', 2: 'Positive'}

st.button('Predict')

if headline_input != '':
    st.markdown('Related company: ')
    st.markdown(f'Sentiment: {sentiments[prediction]}')

with fil_col4:
    templateLoader = jinja2.FileSystemLoader(searchpath="./")
    templateEnv = jinja2.Environment(loader=templateLoader)
    # env = Environment(loader=FileSystemLoader("."), autoescape=select_autoescape())
    template = templateEnv.get_template(PDF_TEMPLATE_FILE)

    # Convert chart to png image (base64 encoded)
    def getBase64Img(img):
        mybuff = io.StringIO()
        img.write_html(mybuff, include_plotlyjs='cdn')
        html_bytes = mybuff.getvalue().encode('utf8')
        newBuff = io.BytesIO(html_bytes)
        newBuff.seek(0)
        base64_jpg = base64.b64encode(newBuff.read()).decode('utf8')
        # href = f'<a href="data:image/png;charset=utf-8;base64, {base64_jpg}" download="plot.html">Download plot</a>'
        # st.markdown(href, unsafe_allow_html=True)
        html = f'data:image/png;base64,{base64_jpg}'
        return html
        # mybuff = io.StringIO()
        # b_publishers.write_html(mybuff, include_plotlyjs='cdn')
        # html_bytes = mybuff.getvalue().encode('utf8')
        # newBuff = io.BytesIO(html_bytes)
        # newBuff.seek(0)
        # base64_jpg = base64.b64encode(newBuff.read()).decode('utf8')

    def getTableHTML(table):
        table_html = table.to_html(index=False)
        newBuff = io.BytesIO(table_html)
        newBuff.seek(0)
        base64_jpg = base64.b64encode(newBuff.read()).decode('utf8')
        href = f'<a href="data:image/png;charset=utf-8;base64, {base64_jpg}" download="plot.html">Download table</a>'
        st.markdown(href, unsafe_allow_html=True)
        return table_html
    
    def save_plotly_plot(name, fig):
        file_name = os.path.join(IMG_FOLDER, f"{ name }.png")
        fig.write_image(file_name, engine="kaleido")
        return file_name
    
    def save_altair_plot(name, fig):
        file_name = os.path.join(IMG_FOLDER, f"{ name }.png")
        fig.save(file_name)
        return file_name
    
    # hsd_html = getBase64Img(chart_HistoricalStockData)
    # fnot_html = getBase64Img(chart_FrequencyofNewsOverTime)
    # ssot_html = getBase64Img(chart_SentimentScoreOverTime)
    # publisher = getBase64Img(chart_Publishers)
    hsd_html = save_plotly_plot('historicalprice_line', chart_HistoricalStockData)
    fnot_html = save_plotly_plot('news_line', chart_FrequencyofNewsOverTime)
    ssot_html = save_plotly_plot('sentiment_pie', chart_SentimentScoreOverTime)
    publisher = save_plotly_plot('publiser_bar', chart_Publishers)
    ssac_html = save_altair_plot('companies_sentiment_bar', chart_SentimentScoreAcrossCompanies)

    try:
        wkhtml_path = pdfkit.configuration(wkhtmltopdf = 'C:/Program Files/wkhtmltopdf/bin/wkhtmltopdf.exe')
        
        # hpay_html = getTableHTML(table_HighestPriceAcrossYear)

        html = template.render(
            hsd_url = hsd_html,
            fnot_url = fnot_html,
            ssot_url = ssot_html,
            ssac_url = ssac_html,
            publishers_url = publisher,
            # hpay_url = hpay_html,

        )

        pdf = pdfkit.from_string(html, configuration = wkhtml_path, options = {"enable-local-file-access": "", "zoom": "1.3"})

        submit = st.download_button(
                "Export⬇️ ",
                data=pdf,
                file_name="Stock Prices Report.pdf",
                mime="application/pdf",
            )
        
        if submit:
            st.balloons()

    except(ValueError, TypeError):
        st.button('Export⬇️')
        print('Button with label only')
    
#====================================================================
# Tab
pricing_data, news = st.tabs(['Stock Price', 'News'])
with pricing_data:
#     st.subheader('Stock Price')
#     with st.container(height=500, border=True):
#         st.table(filtered_df_sp)
    csv=filtered_df_sp.to_csv(index = False).encode('utf-8')
    st.download_button(label='Download Historical Data', data= csv, file_name='Historical Data.csv')

with news:
    # st.subheader('Financial News')
#     with st.container(height=500, border=True):
#         df_fn1 = filtered_df_fn.sort_values(by="published_date", ascending=True)
#         st.table(df_fn)
    csv=filtered_df_fn.to_csv(index = False).encode('utf-8')
    st.download_button(label='Download Financial News', data= csv, file_name='Financial News.csv')
#####################################################################