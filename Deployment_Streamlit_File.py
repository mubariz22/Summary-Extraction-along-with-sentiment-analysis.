#importing the libraries
import streamlit as st
import numpy as np
import pandas as pd
import time
import PyPDF2 as pdf
import nltk
#nltk.download('all')          #THIS NEEDS TO BE RUN ONCE COMPULSARY. YOU CAN EXECUTE/INSTALL IT IN VIRTUAL ENV. USING PIP
import re
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from operator import itemgetter
import matplotlib.pyplot as plt
from io import StringIO
from heapq import nlargest
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

#Setting page options
st.set_page_config(page_title ="P-101 Team III",
                    page_icon="🔮")
st.markdown("<h1 style='text-align: center; color: black;'>e-book Summary Generator & Sentiment Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: purple;'>This project generates a summary of any user-supplied pdf e-book, as well as sentiment with a score.</h4>", unsafe_allow_html=True)
st.markdown("<h6 style='text-align: center; color: black;'>Developed by:</h6>", unsafe_allow_html=True)
st.markdown("<h6 style='text-align: center; color: black;'>- Mentors: Kartik & Pallavi and,</h6>", unsafe_allow_html=True)
st.markdown("<h6 style='text-align: center; color: black;'>- Project 101 Team 3</h6>", unsafe_allow_html=True)


#File uploader
uploaded_file = st.file_uploader("Choose a file", type="pdf")
if uploaded_file is not None:
    with st.spinner('Uploading your e-book...'):
        pdf_reader = pdf.PdfFileReader(uploaded_file)
        text=''
        for i in range(0,pdf_reader.numPages):
            pageObj = pdf_reader.getPage(i)
            text=text+pageObj.extractText()
    st.success("Successfully uploaded your e-book")


#Part 1 - Cleaning & formatting of extracted text for NLP
    new_text = text.split('                    ')
    list2 = [x.replace('\n', ' ').replace('\n\n', ' ').replace('\n\n\n', ' ').replace('\n\n\n\n', ' ').replace('\n\n\n\n\n', ' ').replace(' -- ',' ').replace('- ',' ') for x in new_text]
    def listToString(s):
        str1 = "" 
        for ele in s: 
            str1 += ele  
            return str1
    s = list2
    now_string = listToString(s)
    st.success("Successfully cleaned and formatted your e-book data")
    

    
#Part 2 - Generate the Summary of all input text
    st.warning('WARNING: Large PDF files take more time to process!')
    st.info('Started to generate Summary and Sentiment results for your ebook...')
    with st.spinner('Please wait for this operation to complete....'):
        stop_words = list(STOP_WORDS)
        nlp = spacy.load('en_core_web_lg')
     #nlp.max_length = 1230000
        nlp.max_length = 1638156
        doc = nlp(now_string)
        tokens = [token.text for token in doc]
        punctuation = punctuation + '\n'
        word_frequencies = {}
        for word in doc:
            if word.text.lower() not in stop_words:
                if word.text.lower() not in punctuation:
                    if word.text not in word_frequencies.keys():
                        word_frequencies[word.text] = 1
                    else:
                        word_frequencies[word.text] += 1
        sorted(word_frequencies.items(), key=itemgetter(1), reverse = True)
        max_frequency = max(word_frequencies.values())
        for word in word_frequencies.keys():
            word_frequencies[word] = word_frequencies[word]/max_frequency
        sentence_tokens = [sent for sent in doc.sents]
        sentence_scores = {}
        for sent in sentence_tokens:
            for word in sent:
                if word.text.lower() in word_frequencies.keys():
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent] = word_frequencies[word.text.lower()]
                    else:
                        sentence_scores[sent] += word_frequencies[word.text.lower()]
        select_length = int(len(sentence_tokens)*0.3)
        summary = nlargest(select_length, sentence_scores,key = sentence_scores.get)
        final_summary = [word.text for word in summary]
        summary_2 = ''.join(final_summary)
        percentage_of_text_in_summary = (len(summary_2)/len(text))*100
    st.success("Complete!")
    st.markdown("<h4 style='text-align: center; color: purple;'>The Summary of your e-book:</h4>", unsafe_allow_html=True)
    st.write("The Percentage of text in summary from the entire ebook is: ", percentage_of_text_in_summary)
    st.text_area(label ="",value=summary_2, height =500)

    
#Part 3 - Sentiment Analysis
    analyser = SentimentIntensityAnalyzer()
    scores =[]
    for sentence in final_summary:
        score = analyser.polarity_scores(sentence)
        scores.append(score)
    
    
#Converting List of Dictionaries into Dataframe
    dataFrame= pd.DataFrame(scores)
    df2= dataFrame.mean()
    if (df2.iloc[3] <= -0.05):
        #st.write("Overall Sentiment of the ebook: NEGATIVE")
        st.markdown("<h4 style='text-align: center; color: purple;'>Overall Sentiment of the ebook: NEGATIVE</h4>", unsafe_allow_html=True)
    elif ((df2.iloc[3] >= -0.05) and (df2.iloc[3] <= 0.05)):
        #st.write("Overall Sentiment of the ebook: NEUTRAL")
        st.markdown("<h4 style='text-align: center; color: purple;'>Overall Sentiment of the ebook: NEUTRAL</h4>", unsafe_allow_html=True)
    else:
        #st.write("Overall Sentiment of the ebook: POSITIVE")
        st.markdown("<h4 style='text-align: center; color: purple;'>Overall Sentiment of the ebook: POSITIVE</h4>", unsafe_allow_html=True)
    st.write("Final Sentiment Score is: ",df2.iloc[3])
    st.write("Sentiment Score for each sentence in the Summarized ebook:-\n")
    st.write(dataFrame)
    st.write("Mean Sentiment Score for the complete e-book :-\n",dataFrame.mean())