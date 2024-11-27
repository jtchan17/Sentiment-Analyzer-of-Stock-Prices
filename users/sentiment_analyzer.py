import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch

st.title('📈 Sentiment Analyzer')

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
companies = ['aapl', 'meta', 'msft', 'amzn', 'tsla']
company_keywords = {
    'aapl': ['apple', 'aapl'],
    'meta': ['facebook', 'meta'],
    'msft': ['microsoft', 'msft'],
    'amzn': ['amazon', 'amzn'],
    'tsla': ['tesla', 'tsla']
}
related_company = '-'

if headline_input != '':
    for company, keywords in company_keywords.items():
        if any(keyword in headline_input.lower() for keyword in keywords):
            related_company = company
            break
    st.markdown(f'Related company: {related_company.upper()}')
    st.markdown(f'Sentiment: {sentiments[prediction]}')
