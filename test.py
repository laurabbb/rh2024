#jupytext --to py dataPart.ipynb

#to use the method in the other file, I should be able to say
#from file_name import function_name
!pip install datasets

import streamlit as st
from datasets import load_dataset
from dataPart import cleanDataset, preprocessing, fakeNewsDetector, testSplitAndVectorizing, intializeModels
import pandas as pd
from transformers import pipeline
from transformers import MarianMTModel, MarianTokenizer

#good streamlit tutorial: https://streamlitpython.com/#section-5
#streamlit deployments : https://docs.kanaries.net/topics/Streamlit/streamlit-config

st.title('FND.AI')

st.header('What is fake news detector?')
st.write('Introducing the groundbreaking Fake News Detector 3.0, a revolutionary tool designed to combat misinformation across linguistic boundaries. Utilizing state-of-the-art natural language processing algorithms, this advanced detector is capable of analyzing text in not one, not two, but three languages: English, Spanish, and French. Its sophisticated AI engine scours news articles, social media posts, and online content with unparalleled accuracy, identifying dubious claims, misleading information, and outright fabrications. With a user-friendly interface, this detector empowers individuals to discern fact from fiction effortlessly, helping to safeguard against the spread of misinformation in an increasingly interconnected world. Whether youre a journalist, educator, or concerned citizen, the Fake News Detector 3.0 is your indispensable ally in the fight for truth and transparency.')

st.header('')

st.header('Step 1: Choose your starting language')

languages = ['English', 'Russian', 'Spanish', 'Portuguese', 'Mandarin', 'Cantonese', 'Hindi', 'Arabic', 'Japanese', 'French']
chosenLanguage = st.selectbox('Select your starting language', languages)

st.header('Step 2: Input your news article')
chosenText = st.text_input('Paste Article')

@st.cache_data()
def model():
    dataset = load_dataset("GonzaloA/fake_news")
    df = cleanDataset(dataset)
    print("Dataset Cleanded!")
    df = preprocessing(df)
    print("Daset Preprocessed")
    xTrainCount, xTestCount, y_train, y_test, vectorizer = testSplitAndVectorizing(df)
    print("Triansplit")
    avgAccuracyScore, mnb, cnb, bnb, dtc, drc, percept, passiveAggressive = intializeModels(xTrainCount, xTestCount, y_train, y_test)
    print("models initialized")
    return avgAccuracyScore, mnb, cnb, bnb, dtc, drc, percept, passiveAggressive, vectorizer

avgAccuracyScore, mnb, cnb, bnb, dtc, drc, percept, passiveAggressive, vectorizer = model()

if chosenLanguage == 'Spanish' and chosenText:
    #Below adapted from ChatGPT
    #Load pre-trained translation model and tokenizer
    model_name = "Helsinki-NLP/opus-mt-es-en"
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    # Tokenize input text
    inputs = tokenizer(chosenText, return_tensors="pt", padding=True, truncation=True, max_length=512)
    # Translate text
    translated_ids = model.generate(inputs["input_ids"], max_length=512, num_beams=4, early_stopping=True)
    # Decode translated text
    chosenText = tokenizer.decode(translated_ids[0], skip_special_tokens=True)
    # Print the translated text
    st.header("The translated text is :")
    st.write(chosenText)

    #prediction part
    prediction = fakeNewsDetector(chosenText, vectorizer, mnb, cnb, bnb, dtc, drc, percept, passiveAggressive)

    if prediction:
        st.header("The article is legitimate.")
    else:
        st.header("The article is not legitimate.")

if chosenLanguage == 'Russian' and chosenText:
    #Below adapted from ChatGPT
    #Load pre-trained translation model and tokenizer
    model_name = "Helsinki-NLP/opus-mt-ru-en"
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    # Tokenize input text
    inputs = tokenizer(chosenText, return_tensors="pt", padding=True, truncation=True, max_length=512)
    # Translate text
    translated_ids = model.generate(inputs["input_ids"], max_length=512, num_beams=4, early_stopping=True)
    # Decode translated text
    chosenText = tokenizer.decode(translated_ids[0], skip_special_tokens=True)
    # Print the translated text
    st.header("The translated text is :")
    st.write(chosenText)

    #prediction part
    prediction = fakeNewsDetector(chosenText, vectorizer, mnb, cnb, bnb, dtc, drc, percept, passiveAggressive)

    if prediction:
        st.header("The article is legitimate.")
    else:
        st.header("The article is not legitimate.")

if chosenLanguage == 'Portuguese' and chosenText:
    #Below adapted from ChatGPT
    #Load pre-trained translation model and tokenizer
    model_name = "Helsinki-NLP/opus-mt-pt-en"
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    # Tokenize input text
    inputs = tokenizer(chosenText, return_tensors="pt", padding=True, truncation=True, max_length=512)
    # Translate text
    translated_ids = model.generate(inputs["input_ids"], max_length=512, num_beams=4, early_stopping=True)
    # Decode translated text
    chosenText = tokenizer.decode(translated_ids[0], skip_special_tokens=True)
    # Print the translated text
    st.header("The translated text is :")
    st.write(chosenText)

    #prediction part
    prediction = fakeNewsDetector(chosenText, vectorizer, mnb, cnb, bnb, dtc, drc, percept, passiveAggressive)

    if prediction:
        st.header("The article is legitimate.")
    else:
        st.header("The article is not legitimate.")


if (chosenLanguage == 'Mandarin' or chosenLanguage =='Cantonese') and chosenText:
    #Below adapted from ChatGPT
    #Load pre-trained translation model and tokenizer
    model_name = "Helsinki-NLP/opus-mt-zh-en"
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    # Tokenize input text
    inputs = tokenizer(chosenText, return_tensors="pt", padding=True, truncation=True, max_length=512)
    # Translate text
    translated_ids = model.generate(inputs["input_ids"], max_length=512, num_beams=4, early_stopping=True)
    # Decode translated text
    chosenText = tokenizer.decode(translated_ids[0], skip_special_tokens=True)
    # Print the translated text
    st.header("The translated text is :")
    st.write(chosenText)

    #prediction part
    prediction = fakeNewsDetector(chosenText, vectorizer, mnb, cnb, bnb, dtc, drc, percept, passiveAggressive)

    if prediction:
        st.header("The article is legitimate.")
    else:
        st.header("The article is not legitimate.")

if chosenLanguage == 'French' and chosenText:
    #Below adapted from ChatGPT
    #Load pre-trained translation model and tokenizer
    model_name = "Helsinki-NLP/opus-mt-fr-en"
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    # Tokenize input text
    inputs = tokenizer(chosenText, return_tensors="pt", padding=True, truncation=True, max_length=512)
    # Translate text
    translated_ids = model.generate(inputs["input_ids"], max_length=512, num_beams=4, early_stopping=True)
    # Decode translated text
    chosenText = tokenizer.decode(translated_ids[0], skip_special_tokens=True)
    # Print the translated text
    st.header("The translated text is :")
    st.write(chosenText)

    #prediction part
    prediction = fakeNewsDetector(chosenText, vectorizer, mnb, cnb, bnb, dtc, drc, percept, passiveAggressive)

    if prediction:
        st.header("The article is legitimate.")
    else:
        st.header("The article is not legitimate.")

if chosenLanguage == 'Hindi' and chosenText:
    #Below adapted from ChatGPT
    #Load pre-trained translation model and tokenizer
    model_name = "Helsinki-NLP/opus-mt-hi-en"
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    # Tokenize input text
    inputs = tokenizer(chosenText, return_tensors="pt", padding=True, truncation=True, max_length=512)
    # Translate text
    translated_ids = model.generate(inputs["input_ids"], max_length=512, num_beams=4, early_stopping=True)
    # Decode translated text
    chosenText = tokenizer.decode(translated_ids[0], skip_special_tokens=True)
    # Print the translated text
    st.header("The translated text is :")
    st.write(chosenText)

    #prediction part
    prediction = fakeNewsDetector(chosenText, vectorizer, mnb, cnb, bnb, dtc, drc, percept, passiveAggressive)

    if prediction:
        st.header("The article is legitimate.")
    else:
        st.header("The article is not legitimate.")

if chosenLanguage == 'Arabic' and chosenText:
    #Below adapted from ChatGPT
    #Load pre-trained translation model and tokenizer
    model_name = "Helsinki-NLP/opus-mt-ar-en"
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    # Tokenize input text
    inputs = tokenizer(chosenText, return_tensors="pt", padding=True, truncation=True, max_length=512)
    # Translate text
    translated_ids = model.generate(inputs["input_ids"], max_length=512, num_beams=4, early_stopping=True)
    # Decode translated text
    chosenText = tokenizer.decode(translated_ids[0], skip_special_tokens=True)
    # Print the translated text
    st.header("The translated text is :")
    st.write(chosenText)

    #prediction part
    prediction = fakeNewsDetector(chosenText, vectorizer, mnb, cnb, bnb, dtc, drc, percept, passiveAggressive)

    if prediction:
        st.header("The article is legitimate.")
    else:
        st.header("The article is not legitimate.")

if chosenLanguage == 'Japanese' and chosenText:
    #Below adapted from ChatGPT
    #Load pre-trained translation model and tokenizer
    model_name = "Helsinki-NLP/opus-mt-ja-en"
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    # Tokenize input text
    inputs = tokenizer(chosenText, return_tensors="pt", padding=True, truncation=True, max_length=512)
    # Translate text
    translated_ids = model.generate(inputs["input_ids"], max_length=512, num_beams=4, early_stopping=True)
    # Decode translated text
    chosenText = tokenizer.decode(translated_ids[0], skip_special_tokens=True)
    # Print the translated text
    st.header("The translated text is :")
    st.write(chosenText)

    #prediction part
    prediction = fakeNewsDetector(chosenText, vectorizer, mnb, cnb, bnb, dtc, drc, percept, passiveAggressive)

    if prediction:
        st.header("The article is legitimate.")
    else:
        st.header("The article is not legitimate.")

if chosenLanguage == 'English' and chosenText:
    prediction = fakeNewsDetector(chosenText, vectorizer, mnb, cnb, bnb, dtc, drc, percept, passiveAggressive)
    if prediction:
        st.header("The article is legitimate.")
    else:
        st.header("The article is not legitimate.")
