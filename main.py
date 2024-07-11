from nltk.corpus import stopwords
from heapq import nlargest
from string import punctuation
from nltk.tokenize import word_tokenize, sent_tokenize
from transformers import T5Tokenizer, T5ForConditionalGeneration

import streamlit as st
import nltk


nltk.download('punkt')
nltk.download('stopwords')

stopWords = set(stopwords.words("english"))

st.title("Text - Summarizer")
input_text = st.text_area("Enter your text to Summarize: ", height=200)
col1, col2, col3 = st.columns(3)
with col1:
    option = st.selectbox(
        "Click for Different Types for Summarization Techniques",
        (
            "Extractive",
            "Abstractive",
        ),
    )

text = input_text

if st.button("Summarize"):
    if option == "Extractive":
        words = word_tokenize(text)

        for i in range(len(words)):
            words[i] = words[i].lower()

        freqTable = dict()
        for word in words:
            if word not in stopWords:
                if word not in punctuation:
                    if word in freqTable:
                        freqTable[word] += 1
                    else:
                        freqTable[word] = 1

        sentences = sent_tokenize(text)
        sentenceValue = dict()

        for sentence in sentences:
            for word, freq in freqTable.items():
                if word in sentence.lower():
                    if sentence in sentenceValue:
                        sentenceValue[sentence] += freq
                    else:
                        sentenceValue[sentence] = freq

        if sentenceValue:  # Check if sentenceValue is not empty
            maxValue = max(sentenceValue.values())
            for sentence in sentenceValue:
                sentenceValue[sentence] = sentenceValue[sentence] / maxValue

            length = int(len(sentenceValue) * 0.35)
            summary = nlargest(length, sentenceValue, key=sentenceValue.get)

            Summary = ""
            for sentence in summary:
                Summary += sentence
        else:
            Summary = "No summary could be generated. The input text might be too short or empty."

        st.write(Summary)

    else:
        mlength = int(len(text) * 0.40)

        model_name = 't5-small'
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name)

        inputs = tokenizer.encode("summarize: " + text, return_tensors="pt")
        summary_ids = model.generate(inputs, max_length=mlength, min_length=40, length_penalty=1.0, num_beams=10, early_stopping=True)
        Summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        st.write(Summary)
