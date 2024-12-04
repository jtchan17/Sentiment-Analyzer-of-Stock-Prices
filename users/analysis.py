import streamlit as st 
import altair as alt
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly_express as px 
import pandas as pd
from wordcloud import WordCloud, STOPWORDS
import pandas as pd

# Initialize connection.
conn = st.connection('mysql', type='sql')

# Loading the data
@st.cache_resource
def load_data(query):
    df = conn.query(query, ttl=600)
    return df
df_fn = load_data('SELECT * from dashboard.financialnewswtopics;')
# df_sp = load_data('SELECT * from dashboard.stockprice;')
df_fnsp = pd.read_csv('MergeData.csv')
alt.themes.enable("dark")

#Calculate daily returns for each company's stock data
for company, data in df_fnsp.items():
    df_fnsp['daily_return'] = df_fnsp['Close'].pct_change()

with st.sidebar:
    # st.subheader('Companies', divider=True)
    companies_list = [
        'AAPL',
        'AMZN',
        'TSLA',
        'MSFT',
        'META'
    ]

    #set the default colours for each companies
    cust_company = st.selectbox('Select Company', companies_list)

company_data = df_fnsp[df_fnsp['company'] == cust_company]
company_data1 = df_fn[df_fn['company'] == cust_company]

#Sentiment score and Stock price over the time
fig = px.line(company_data,
    x='Date',
    y=['Adj Close', 'sentiment_score'],
    # color='sentiment_score',
    # labels={'value': 'Value', 'Date': 'Date'},
    title=f"Stock Price and Sentiment Score for {cust_company}",
)

fig.update_layout(
    yaxis=dict(title="Stock Price"),
    yaxis2=dict(title="Sentiment Score", overlaying="y", side="right"),
    template="gridon"
)

st.plotly_chart(fig)

# Sentiment vs Daiy Stock Returns
topic_counts = company_data1['topics'].value_counts().reset_index()
topic_counts.columns = ['Topic', 'Count']
fig2 = px.scatter(
    topic_counts,
    x='Topic',
    y='Count',
    color='Count',
    size='Count',
    # hover_data=['Date'],
    title=f"Sentiment vs Daily Returns for {cust_company}",
    labels={'sentiment_score': 'Sentiment Score', 'daily_return': 'Daily Return'}
)

fig2.update_layout(template="plotly_dark")
st.plotly_chart(fig2)

# #Topic Distribution
# topic_counts = company_data1['topics'].value_counts().reset_index()
# topic_counts.columns = ['Topic', 'Count']

fig3 = px.bar(
    topic_counts,
    x='Topic',
    y='Count',
    color='Count',
    title=f"Topic Distribution for {cust_company}",
    labels={'Topic': 'Topic', 'Count': 'Count'}
)

fig3.update_layout(template="plotly_dark", xaxis_tickangle=-45)
st.plotly_chart(fig3)

# #Impact of Topics on Stock Returns
fig4 = px.box(
    company_data1,
    x='topics',
    y='sentiment_score',
    color='topics',
    title=f"Impact of Topics on Sentiments for {cust_company}",
    labels={'topics': 'Topic', 'sentiment_score': 'Sentiments'}
)

fig4.update_layout(template="plotly_dark", xaxis_tickangle=-45)
st.plotly_chart(fig4)

#Correlation Heatmap
correlation_data = company_data[['sentiment_score', 'daily_return']].corr()

# Convert correlation matrix into long-form for Plotly Express
correlation_long = correlation_data.stack().reset_index()
correlation_long.columns = ['Variable1', 'Variable2', 'Correlation']

fig5 = px.imshow(
    correlation_data,
    text_auto=True,
    color_continuous_scale='Viridis',
    title=f"Correlation Heatmap for {cust_company}"
)

fig5.update_layout(template="plotly_dark")
st.plotly_chart(fig5)

#Word Cloud
# Start with one review:
text = " ".join(title for title in company_data1.title)

# Create stopword list:
stopwords = set(STOPWORDS)
stopwords.update(["META", "TSLA", "AMZN", "AAPL", "MSFT", 'Microsoft', 'Tesla', 'Apple', 'Amazon'])

# Generate a word cloud image
wordcloud = WordCloud(stopwords=stopwords, background_color="black").generate(text)
# Display the generated image:
# the matplotlib way:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

# Create and generate a word cloud image:
# lower max_font_size, change the maximum number of word and lighten the background:
# wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(text)
# plt.figure()
# plt.imshow(wordcloud, interpolation="bilinear")
# plt.axis("off")
st.pyplot(plt)

# Save the image in the img folder:
wordcloud.to_file("image/wordcloud.png")