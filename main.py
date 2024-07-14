from nltk.corpus import stopwords
from heapq import nlargest
from string import punctuation
from nltk.tokenize import word_tokenize, sent_tokenize
from transformers import T5Tokenizer, T5ForConditionalGeneration

import streamlit as st
import nltk

# importing all english letter stopwords int stopword
stopWords = set(stopwords.words("english"))

# setting up Streamlit interface
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

# extractive summarization part
if st.button("Summarize"):
    if option == "Extractive":

        # split the text into words
        words = word_tokenize(text)

        # lower case words of the text
        for i in range(len(words)):
            words[i] = words[i].lower()

        freqTable = dict()

        # get  the frequency of all the words other than the stopwords
        for word in words:
            if word not in stopWords:
                if word not in punctuation:
                    if word in freqTable:
                        freqTable[word] += 1
                    else:
                        freqTable[word] = 1

        # split the text into sentences
        sentences = sent_tokenize(text)

        sentenceValue = dict()

        # then we will get the frequency of all the sentences based on the frequency of words
        for sentence in sentences:
            for word, freq in freqTable.items():
                if word in sentence.lower():
                    if sentence in sentenceValue:
                        sentenceValue[sentence] += freq
                    else:
                        sentenceValue[sentence] = freq

        if sentenceValue:  # Check if sentenceValue is not empty
            maxValue = max(sentenceValue.values())  # to get the max value
            for sentence in sentenceValue:
                sentenceValue[sentence] = sentenceValue[sentence] / maxValue   # making sentence score

            length = int(len(sentenceValue) * 0.35)  # 35 % total text we will get in the summary
            summary = nlargest(length, sentenceValue, key=sentenceValue.get)
            # getting the largest sentence score based sentences

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
        # converts text into a format the model can understand.
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        # capable of generating text based on input prompts.
        inputs = tokenizer.encode("summarize: " + text, return_tensors="pt")
        # it instructs the model to perform summarization
        summary_ids = model.generate(inputs, max_length=mlength, min_length=40, length_penalty=1.0, num_beams=10, early_stopping=True)
        Summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        st.write(Summary)