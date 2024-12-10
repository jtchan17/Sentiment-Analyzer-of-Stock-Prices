import streamlit as st 
import pandas as pd 
import plotly_express as px 
import altair as alt
import torch
import matplotlib.pyplot as plt
import pdfkit
import jinja2
# from jinja2 import Environment, PackageLoader, select_autoescape, FileSystemLoader
from streamlit.components.v1 import iframe
import plotly.io as pio
import matplotlib.pyplot as plt
import os
from wordcloud import WordCloud, STOPWORDS

##########################################################################################################################################
PDF_TEMPLATE_FILE = 'PDFtemplate.html'
IMG_FOLDER = os.path.join(os.getcwd(), 'image')

##########################################################################################################################################
# Redirect to app.py if not logged in, otherwise show the navigation menu
# menu_with_redirect()

# Initialize connection.
conn = st.connection('mysql', type='sql')

# Loading the data
@st.cache_data
def load_data(query):
    df = conn.query(query, ttl=600)
    return df
df_fn = load_data('SELECT * from dashboard.fnwithtopics;')
df_sp = load_data('SELECT * from dashboard.stockprice;')
alt.themes.enable("dark")

##########################################################################################################################################

##########################################################################################################################################
# Side bar (Login)
with st.sidebar:
    # st.title(f'Welcome {st.session_state.role}! :sunglasses:')
    st.title(f'Welcome {st.session_state.username}! :sunglasses:')
    custom_colors = {
        'Red': '#EF553B',
        'Blue': '#636EFA',
        'Green': '#00CC96',
        'Yellow': '#FECB52',
        'Pink': '#E45756',
        'Orange': '#FFA15A',
        'Purple': '#AB63FA',
        'Cyan': '#19D3F3',
        'Lime': '#B6E880',
        'Magenta': '#FF6692'
    }
    #------------------------------------------------------------------------
    #Text Colour
    font_colors = {
        'White': ':white',
        'Violet': ':violet',
        'Blue':':blue',
        'Green': ':green',
        'Orange': ':orange',
        'Red': ':red',
        'Purple': ':purple'
    }
    font_family = {
        'Sans-serif': 'sans-serif',
        'Serif': 'serif',
        'Monospace': 'monospace',
        'Arial': 'arial',
        'Lucida Console': 'Lucida Console',
        'Times New Roman': 'Times New Roman',
        'Courier New': 'Courier New'
    }
    font_style = ['normal', 'italic']

    st.subheader('Text', divider=True)
    # text_colour_selection = st.selectbox('Colour', options=list(font_colors.keys()), index=0)
    # final_font_colour = font_colors[text_colour_selection]
    font_family_selection = st.selectbox('Font Family', options=list(font_family.keys()), index=0)
    final_font_family = font_family[font_family_selection]
    font_style_selection = st.selectbox('Style', font_style, index=0)
    #------------------------------------------------------------------------
    #Companies Colour
    st.subheader('Companies', divider=True)
    companies_default_colors = {
        'AAPL': 'Blue',
        'AMZN': 'Orange',
        'TSLA': 'Green',
        'MSFT': 'Red',
        'META': 'Purple'
    }

    #set the default colours for each companies
    cust_aapl_selection = st.selectbox('AAPL', list(custom_colors.keys()), index=list(custom_colors.keys()).index(companies_default_colors['AAPL']))
    cust_amzn_selection = st.selectbox('AMZN', list(custom_colors.keys()), index=list(custom_colors.keys()).index(companies_default_colors['AMZN']))
    cust_tsla_selection = st.selectbox('TSLA', list(custom_colors.keys()), index=list(custom_colors.keys()).index(companies_default_colors['TSLA']))
    cust_msft_selection = st.selectbox('MSFT', list(custom_colors.keys()), index=list(custom_colors.keys()).index(companies_default_colors['MSFT']))
    cust_meta_selection = st.selectbox('META', list(custom_colors.keys()), index=list(custom_colors.keys()).index(companies_default_colors['META']))

    #final colour selection
    final_aapl_colour = custom_colors[cust_aapl_selection]
    final_amzn_colour = custom_colors[cust_amzn_selection]
    final_tsla_colour = custom_colors[cust_tsla_selection]
    final_msft_colour = custom_colors[cust_msft_selection]
    final_meta_colour = custom_colors[cust_meta_selection]

    #------------------------------------------------------------------------
    #Sentiment Colour
    st.subheader('Sentiment', divider=True)
    sentiment_default_colors = {
        'positive': 'Green',
        'negative': 'Red',
        'neutral': 'Blue'
    }
    cust_pos_selection = st.selectbox('Positive: ', list(custom_colors.keys()), index=list(custom_colors.keys()).index(sentiment_default_colors['positive']))
    cust_neg_selection = st.selectbox('Negative:', list(custom_colors.keys()), index=list(custom_colors.keys()).index(sentiment_default_colors['negative']))
    cust_neu_selection = st.selectbox('Neutral:', list(custom_colors.keys()), index=list(custom_colors.keys()).index(sentiment_default_colors['neutral']))

    final_pos_colour = custom_colors[cust_pos_selection]
    final_neg_colour = custom_colors[cust_neg_selection]
    final_neu_colour = custom_colors[cust_neu_selection]

    

##########################################################################################################################################

##########################################################################################################################################
#   Dashboard
# st.title(f'📈 {final_font_colour}[Dashboard of Stock Prices and Financial News]')
st.markdown(
    f"""
    <p style="font-family: {final_font_family}; font-size: 40px; font-style: {font_style_selection}; font-weight: bold;">
    📈 Dashboard of Stock Prices and Financial News
    </p>
    """,
    unsafe_allow_html=True,
)
#-------------------------------------------------------------------------------------------
#CSS Injection
#Button
st.markdown(
    f"""
    <style>
    .stButton{{
        background-color: #111111; /* Button background color */
        border-radius: 10px; /* Rounded corners */
        padding: 5px 10px;
    }}
    .stButton > button > div > p {{
        font-family: {final_font_family} !important; /* Change to desired font */
        font-size: 16px; /* Adjust font size */
        font-style: {font_style_selection}
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# #Download Button
st.markdown(
    f"""
    <style>
    .stDownloadButton {{
        background-color: #111111; /* Button background color */
        border-radius: 10px; /* Rounded corners */
        padding: 5px 10px;
    }}
    .stDownloadButton > button > div > p{{
        font-family: {final_font_family} !important; /* Change to desired font */
        font-size: 16px; /* Adjust font size */
        font-style: {font_style_selection}
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# #Popover
st.markdown(
    f"""
    <style>
    .stPopover {{
        background-color: #111111; /* Button background color */
        border-radius: 10px; /* Rounded corners */
        padding: 5px 10px;
    }}

    .stPopover > div > button > div > p{{
        font-family: {final_font_family}; /* Change to desired font */
        font-size: 16px; /* Adjust font size */
        font-style: {font_style_selection}
    }}
    </style>
    """,
    unsafe_allow_html=True
)

#Table
st.markdown(
    f"""
    <style>
    .stTable{{
        font-family: {final_font_family}; /* Change to desired font */
        font-style: {font_style_selection}
    }}
    </style>
    """,
    unsafe_allow_html=True
)

#DataFrame
# st.markdown(
#     f"""
#     <style>
#     div[data-testid="stDataFrame"] > div > div[class="stDataFrameGlideDataEditor gdg-wmyidgi"] > div > div[class="gdg-sldgczr6"] > div > canvas[data-testid="data-grid-canvas"]{{
#         background-color: #555555
#         font-family: {final_font_family}; /* Change to desired font */
#         font-style: {font_style_selection}
#     }}
#     </style>
#     """,
#     unsafe_allow_html=True
# )
st.markdown(
    f"""
    <style>
    div[data-testid="stDataFrame"] > div[data-testid="stDataFrameResizable"] > div[class="stDataFrameGlideDataEditor gdg-wmyidgi"] > div > div[class="gdg-s1dgczr6"] > div[class="dvn-underlay"] {{
        background-color: #555555
        font-family: {final_font_family}; /* Change to desired font */
        font-style: {font_style_selection}
    }}
    </style>
    """,
    unsafe_allow_html=True
)

#Tabs
st.markdown(
    f"""
    <style>
    .stTabs > div > div > div > button > div > p{{
        font-family: {final_font_family}; /* Change to desired font */
        font-style: {font_style_selection}
    }}
    </style>
    """,
    unsafe_allow_html=True
)

#Chart
def chart_update_layout(chart):
    chart.update_layout(
            font=dict(
                family=font_family_selection,
                style=font_style_selection
            )
        )
#-------------------------------------------------------------------------------------------

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
        st.session_state['aapl'] = True
        st.session_state['amzn'] = True
        st.session_state['meta'] = True
        st.session_state['msft'] = True
        st.session_state['tsla'] = True
        st.session_state['select_year'] = 'All'
        st.session_state['positive'] = True
        st.session_state['negative'] = True
        st.session_state['neutral'] = True
        st.session_state['politics'] = True
        st.session_state['economy'] = True
        st.session_state['technology'] = True
        st.session_state['health'] = True
        st.session_state['sports'] = True
        st.session_state['entertainment'] = True
        st.session_state['science'] = True
        st.session_state['business'] = True
        st.session_state['travel'] = True
        st.session_state['education'] = True
        st.session_state['lifestyle'] = True
        st.session_state['finance'] = True
        st.session_state['investing'] = True
        st.session_state['wellness'] = True

    btn_clear = st.button('Clear All Filter', key='clearFilter', on_click=clear_filters)

###### Filter based on Year and Company ######
def query(table, year, companies):
    query = f'SELECT * FROM dashboard.{table} WHERE '
    if year != 'All':
        query += f'YEAR(published_date) = {year} ' if table == 'fnwithtopics' else f'YEAR(date) = {year} '
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
    fn_query = query('fnwithtopics', select_year, selected_companies)
    sp_query = query('stockprice', select_year, selected_companies)
    filtered_df_fn = conn.query(fn_query, ttl=10000)
    filtered_df_sp = conn.query(sp_query, ttl=10000)

#======================================================================================================================================== 
#ROW 1
r1c1, r1c2 = st.columns((7, 3), gap='small')
with r1c1:
    # st.subheader(f'{final_font_colour}[Historical Stock Data]')
    st.markdown(
        f"""
        <p style="font-family: {final_font_family}; font-size: 24px; font-style: {font_style_selection}; font-weight: bold;">
            Historical Stock Data
        </p>
        """,
        unsafe_allow_html=True,
    )
    # st.markdown(f'###### {final_font_colour}[currency in USD]')
    chart_HistoricalStockData = px.line(filtered_df_sp, x='date', y='adj_close', template='gridon', color='company', 
                                        color_discrete_map={'AAPL': final_aapl_colour,
                                                            'AMZN': final_amzn_colour,
                                                            'TSLA': final_tsla_colour,
                                                            'MSFT': final_msft_colour,
                                                            'META': final_meta_colour})
    chart_update_layout(chart_HistoricalStockData)
    st.plotly_chart(chart_HistoricalStockData, key='chart_HistoricalStockData', use_container_width=True)   

with r1c2:
    # st.subheader(f'{final_font_colour}[Highest Price Across Years]')
    st.markdown(
        f"""
        <p style="font-family: {final_font_family}; font-size: 24px; font-style: {font_style_selection}; font-weight: bold;">
            Highest Price Across Years
        </p>
        """,
        unsafe_allow_html=True,
    )
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
    table_HighestPriceAcrossYear = table_HighestPriceAcrossYear
    st.table(table_HighestPriceAcrossYear)
    # st.table(table_HighestPriceAcrossYear.style.set_properties(**{"color": f"{text_colour_selection}"}))
#======================================================================================================================================== 
#ROW 2
r2c1, r2c2 = st.columns((3, 5), gap='small')
with r2c1:
    # st.subheader(f'{final_font_colour}[Number of News Across Companies]')
    st.markdown(
        f"""
        <p style="font-family: {final_font_family}; font-size: 24px; font-style: {font_style_selection}; font-weight: bold;">
            Number of News Across Companies
        </p>
        """,
        unsafe_allow_html=True,
    )
    table_NumberofNewsAcrossCompanies = filtered_df_fn.groupby('company')['title'].count().reset_index(name='Total')
    table_NumberofNewsAcrossCompanies = table_NumberofNewsAcrossCompanies
    st.table(table_NumberofNewsAcrossCompanies)
    # st.table(table_NumberofNewsAcrossCompanies.style.set_properties(**{"color": f"{text_colour_selection}"}))

with r2c2:
    # st.subheader(f'{final_font_colour}[Frequency of News Over Time]')
    st.markdown(
        f"""
        <p style="font-family: {final_font_family}; font-size: 24px; font-style: {font_style_selection}; font-weight: bold;">
            Frequency of News Over Time
        </p>
        """,
        unsafe_allow_html=True,
    )
    df_article_freq = filtered_df_fn.groupby(['published_date', 'company']).size().unstack(fill_value=0)
    df_article_freq = df_article_freq.reset_index()
    df_melted = pd.melt(df_article_freq, id_vars='published_date', var_name='company', value_name='frequency')
    chart_FrequencyofNewsOverTime = px.line(df_melted, x='published_date', y="frequency", template='gridon', color='company',
                                            color_discrete_map={'AAPL': final_aapl_colour,
                                                                'AMZN': final_amzn_colour,
                                                                'TSLA': final_tsla_colour,
                                                                'MSFT': final_msft_colour,
                                                                'META': final_meta_colour})
    chart_update_layout(chart_FrequencyofNewsOverTime)
    st.plotly_chart(chart_FrequencyofNewsOverTime,use_container_width=True)
#======================================================================================================================================== 
#ROW 3
pop_col1, pop_col2 = st.columns([5,5])
with pop_col1:
    sentiment_popover = st.popover('Choose sentiments')
    positive = sentiment_popover.checkbox('Positive', key='positive', value=True)
    negative = sentiment_popover.checkbox('Negative', key='negative', value=True)
    neutral = sentiment_popover.checkbox('Neutral', key='neutral', value=True)

    # List of selected companies
    sentiments = {'Positive': positive, 'Negative': negative, 'Neutral': neutral}
    selected_sentiments = [sentiment for sentiment, selected in sentiments.items() if selected]

    def filter_sentiment(df):
        if selected_sentiments:
            df = df[df['sentiment_score'].isin(selected_sentiments)]
        return df

with pop_col2:
    topic_popover = st.popover('Choose topics')
    politics = topic_popover.checkbox('Politics', key='politics', value=True)
    economy = topic_popover.checkbox('Economy', key='economy', value=True)
    technology = topic_popover.checkbox('Technology', key='technology', value=True)
    health = topic_popover.checkbox('Health', key='health', value=True)
    sports = topic_popover.checkbox('Sports', key='sports', value=True)
    entertainment = topic_popover.checkbox('Entertainment', key='entertainment', value=True)
    science = topic_popover.checkbox('Science', key='science', value=True)
    business = topic_popover.checkbox('Business', key='business', value=True)
    travel = topic_popover.checkbox('Travel', key='travel', value=True)
    education = topic_popover.checkbox('Education', key='education', value=True)
    lifestyle = topic_popover.checkbox('Lifestyle', key='lifestyle', value=True)
    finance = topic_popover.checkbox('Finance', key='finance', value=True)
    investing = topic_popover.checkbox('Investing', key='investing', value=True)
    wellness = topic_popover.checkbox('Wellness', key='wellness', value=True)

    # #List of Topics
    topics = {
        'Politics\r': politics,
        'Economy\r': economy,
        'Technology\r': technology,
        'Health\r': health,
        'Sports\r': sports,
        'Entertainment\r': entertainment,
        'Science\r': science,
        'Business\r': business,
        'Travel\r': travel,
        'Education\r': education,
        'Lifestyle\r': lifestyle,
        'Finance\r': finance,
        'Investing\r': investing,
        'Wellness\r': wellness,}

    # selected_topics = [topic for topic, select in topics.items() if select]
    selected_topics = [topic for topic, select in topics.items() if select]

    def filter_topics(dffn):
        if selected_topics:
            dffn = dffn[dffn['topics'].isin(selected_topics)]
        return dffn
    
r3c1, r3c2 = st.columns((5,5), gap='small')
with r3c1:
    # st.subheader(f'{final_font_colour}[Sentiment Score Over Time]')
    st.markdown(
        f"""
        <p style="font-family: {final_font_family}; font-size: 24px; font-style: {font_style_selection}; font-weight: bold;">
            Sentiment Score Over Time
        </p>
        """,
        unsafe_allow_html=True,
    )

    #sentiment score
    def plot_pie():
        df_sentiment = filtered_df_fn.groupby('sentiment_score').size().reset_index(name='Total')
        df_fn1 = filter_sentiment(df_sentiment)
        chart_SentimentScoreOverTime = px.pie(df_fn1, values='Total', names='sentiment_score', color="sentiment_score",
                                            color_discrete_map={'Negative': final_neg_colour, 
                                                                'Positive': final_pos_colour, 
                                                                'Neutral': final_neu_colour},
                                            hole=0.5)
        chart_SentimentScoreOverTime.update_traces(textposition='inside')
        return chart_SentimentScoreOverTime
    
    chart_SentimentScoreOverTime = plot_pie()
    chart_update_layout(chart_SentimentScoreOverTime)
    st.plotly_chart(chart_SentimentScoreOverTime, use_container_width=True)

    #Sentiments Score Across Companies
    df_fn1 = filter_sentiment(filtered_df_fn)
    grouped_sentiment_df_fn = df_fn1.groupby(['company', 'sentiment_score']).size().unstack(fill_value=0)
    table_SentimentFrequency = grouped_sentiment_df_fn.reset_index()
    grouped_sentiment_df_fn.rename(columns={'company': 'Companies', 'negative': 'Negative', 'neutral': 'Neutral', 'positive': 'Positive'}, inplace=True)
    table_SentimentFrequency = grouped_sentiment_df_fn
    st.table(table_SentimentFrequency)
    # st.table(table_SentimentFrequency.style.set_properties(**{"color": f"{text_colour_selection}"}))

with r3c2:
    # st.subheader(f'{final_font_colour}[Sentiment Score Across Companies]')
    st.markdown(
        f"""
        <p style="font-family: {final_font_family}; font-size: 24px; font-style: {font_style_selection}; font-weight: bold;">
            Sentiments Distribution by Topic
        </p>
        """,
        unsafe_allow_html=True,
    )
    #Sentiments Distribution by Topic
    # st.dataframe(df_fn1)
    df_fn1 = filter_topics(df_fn1)
    chart_TopicFrequency = px.histogram(
        df_fn1, 
        x='sentiment_score', 
        color='topics', 
        labels={'sentiment_score': 'Sentiments', 'count': 'Total'},
        template='plotly_dark'
    )
    chart_update_layout(chart_TopicFrequency)
    st.plotly_chart(chart_TopicFrequency)

    #WordCloud
    # Start with one review:
    text = " ".join(title for title in filtered_df_fn.title)

    # Create stopword list:
    stopwords = set(STOPWORDS)
    stopwords.update(filtered_df_fn['company'].tolist())
    stopwords.update(filtered_df_fn['publisher'].tolist())
    stopwords.update(['Apple', 'Tesla', 'Meta', 'Amazon', 'Microsoft'])
    # stopwords.update(["META", "TSLA", "AMZN", "AAPL", "MSFT", 'Microsoft', 'Tesla', 'Apple', 'Amazon'])

    # Generate a word cloud image
    wordcloud = WordCloud(stopwords=stopwords, background_color="black").generate(text)
    # Display the generated image:
    # the matplotlib way:
    word_frequncy = plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    word_frequncy = st.pyplot(plt)
#======================================================================================================================================== 
#ROW 4
r4c1, r4c2 = st.columns((3, 7), gap='small')

with r4c1:
    # st.subheader(f'{final_font_colour}[Top 10 Publishers]'+':newspaper:')
    st.markdown(
        f"""
        <p style="font-family: {final_font_family}; font-size: 24px; font-style: {font_style_selection}; font-weight: bold;">
            Top 10 Publishers 📰
        </p>
        """,
        unsafe_allow_html=True,
    )
    df_fn1 = (filtered_df_fn.groupby('publisher').size().reset_index(name='Total'))
    table_TopPublishers = (df_fn1.sort_values(by="Total", ascending=False)).head(10)
    table_TopPublishers = table_TopPublishers
    df = st.dataframe(table_TopPublishers,
                column_order=("publisher", "Total"),
                hide_index=True,
                width=None,
                column_config={
                    "publisher": st.column_config.TextColumn("Publisher",),
                    "Total": st.column_config.ProgressColumn("Total",format="%f",min_value=0,max_value=max(df_fn1.Total),)
                    }
                )
with r4c2:
    # st.subheader(f'{final_font_colour}[Publishers]'+':newspaper:')
    st.markdown(
        f"""
        <p style="font-family: {final_font_family}; font-size: 24px; font-style: {font_style_selection}; font-weight: bold;">
            Publishers 📰
        </p>
        """,
        unsafe_allow_html=True,
    )
    df_fn1 = filtered_df_fn.groupby('publisher').size().reset_index(name='Total')
    chart_Publishers = px.bar(df_fn1,x='Total', y='publisher', template='seaborn')
    chart_Publishers.update_traces(text=df_fn1['publisher'], textposition='inside')
    chart_update_layout(chart_Publishers)
    st.plotly_chart(chart_Publishers, use_container_width=True, height = 1000)
    
#======================================================================================================================================== 
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


#======================================================================================================================================== 
with fil_col4:
    templateLoader = jinja2.FileSystemLoader(searchpath="./")
    templateEnv = jinja2.Environment(loader=templateLoader)
    # env = Environment(loader=FileSystemLoader("."), autoescape=select_autoescape())
    template = templateEnv.get_template(PDF_TEMPLATE_FILE)

    #save dataframe to html
    def getTableHTML(table, index, border):
        table_html = table.to_html(index=index, escape=False, border=border)
        return table_html
    
    #save plotly_express chart into png format
    def save_plotly_plot(name, fig):
        file_name = os.path.join(IMG_FOLDER, f"{ name }.png")
        fig.write_image(file_name, engine="kaleido")
        return file_name
    
    #save altair chart into png format
    def save_altair_plot(name, fig):
        file_name = os.path.join(IMG_FOLDER, f"{ name }.png")
        fig.save(file_name)
        return file_name
    
    def save_word_cloud(name, fig):
        file_name = os.path.join(IMG_FOLDER, f"{ name }.png")
        # Save the image in the img folder:
        wordcloud.to_file(file_name)
        return file_name
    
    hsd_html = save_plotly_plot('historicalprice_line', chart_HistoricalStockData)
    fnot_html = save_plotly_plot('news_line', chart_FrequencyofNewsOverTime)
    ssot_html = save_plotly_plot('sentiment_pie', chart_SentimentScoreOverTime)
    publisher = save_plotly_plot('publiser_bar', chart_Publishers)
    sdbt_html = save_plotly_plot('topics_bar', chart_TopicFrequency)
    wf_html = save_word_cloud('wordcloud', word_frequncy)
    # ssac_html = save_altair_plot('companies_sentiment_bar', chart_SentimentScoreAcrossCompanies)
    hpay_table_html = getTableHTML(table_HighestPriceAcrossYear, False, 1)
    nnac_table_html = getTableHTML(table_NumberofNewsAcrossCompanies, False, 1)
    ssac_table_html = getTableHTML(table_SentimentFrequency, True, 2)
    tp_table_html = getTableHTML(table_TopPublishers, False, 1)

    try:
        wkhtml_path = pdfkit.configuration(wkhtmltopdf = 'C:/Program Files/wkhtmltopdf/bin/wkhtmltopdf.exe')
        
        

        html = template.render(
            hsd_url = hsd_html,
            fnot_url = fnot_html,
            ssot_url = ssot_html,
            sdbt_url = sdbt_html,
            wf_url = wf_html,
            publishers_url = publisher,
            hpay_table = hpay_table_html,
            nnac_table = nnac_table_html,
            ssac_table = ssac_table_html,
            tp_table = tp_table_html,
            selected_font = final_font_family,
            selected_font_style = font_style_selection,
        )

        pdf = pdfkit.from_string(html, configuration = wkhtml_path, options = {"enable-local-file-access": "", "zoom": "1.3"})

        submit = st.download_button(
                "Export⬇️",
                data=pdf,
                file_name="Stock Prices Report.pdf",
                mime="application/pdf",
            )

        if submit:
            st.balloons()

    except(ValueError, TypeError):
        export_button = st.button('Export⬇️')
        print('Button with label only')

##########################################################################################################################################
