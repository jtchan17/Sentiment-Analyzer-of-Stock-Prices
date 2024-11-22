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

if headline_input != '':
    st.markdown('Related company: ')
    st.markdown(f'Sentiment: {sentiments[prediction]}')
