import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch

st.title('📈 Sentiment Analyzer')

#Sentiment Analyzer
st.subheader('Sentiment Analyzer')
st.markdown('#### Please put your financial headline here: ')
headline_input = st.text_input('headline', label_visibility="collapsed")

# Load the tokenizer and model
# tokenizer = BertTokenizer.from_pretrained('./model')
# model = BertForSequenceClassification.from_pretrained('./model')
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone', num_labels=3, output_hidden_states=True)

# Tokenize the input
# inputs = tokenizer(headline_input, return_tensors='pt')

def predict_sentiment(text):
    '''Function to predict the sentiment of a given text using a pre-trained BERT model.
    Args: the input text for sentiment prediction.
    Returns: the predicted sentiment ('negative', 'neutral', 'positive').
    '''

    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class = torch.argmax(predictions, dim=1).item()
    sentiment = {0: 'Neutral', 1: 'Negative', 2: 'Positive'}
    return sentiment[predicted_class]

# Example prediction
# example_text = "Apple (AAPL) Stock Significantly Outperforms S&P 500: A Strong Bullish Trend in 2024"
predicted_sentiment = predict_sentiment(headline_input)
# print(f"Predicted Sentiment: {predicted_sentiment}")

# # Perform inference
# with torch.no_grad():
#     outputs = model(**inputs)

# # Get the prediction
# logits = outputs.logits
# prediction = torch.argmax(logits, dim=1).item()
# sentiments = {0: 'Neutral', 1: 'Negative', 2: 'Positive'}

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
    # st.markdown(f'Sentiment: {sentiments[prediction]}')
    st.markdown(f'Predicted Sentiment: {predicted_sentiment}')
