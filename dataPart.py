# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---



# +
#Train russian to english

#train fake news identifier

#allow web input on streamlit app so users can use the two models on a news article of their 
#choice that they copy and paste

#better yet, later we can allow a language drop down where they can select their language.


# +
import numpy as np
import pandas as pd
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import sent_tokenize, word_tokenize
from collections import Counter
from nltk.corpus import stopwords

from sklearn import linear_model
from sklearn.linear_model import Perceptron, LogisticRegression, PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB, ComplementNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split


# +
#CLEANS DATASET

def cleanDataset(dataset):
    #combines all of the dataset to 1 df
    df = pd.concat([dataset['train'].to_pandas(), dataset['validation'].to_pandas(), dataset['test'].to_pandas()])

    #drops uneeded columns
    df = df.drop('Unnamed: 0', axis=1)

    #we need to combine text and title since the title is important.
    df['text'] = df['title'] + ' ' + df['text']
    df = df.drop('title', axis=1)

    #resets index since they have been combined and drops old indexes.
    df = df.reset_index(drop=True)

    return df


# +
#FIX: Figure out if removing stopwords and then tokenizing or vice versa is better. it may differ for each language. https://www.quora.com/What-should-be-done-first-stopwords-removal-or-stemming-Why-In-weka-should-I-perform-stemming-to-stopwords-list-so-the-word-abl-can-be-removed
#PREPROCESSING

def preprocessing(df):
    #TOKENIZING & STOPWORDS
    
    nltk.download('stopwords')
    stop_words = stopwords.words('english')

    # Counts the frequency of each word in the original dataset
    word_freq = Counter()
    for row in df['text']:
        words = row.split()
        word_freq.update(words)

    # Adds the most frequent news words that are considered custom stopwords to the list.
    for word, freq in word_freq.most_common(100):
        if word.lower() not in stop_words:
            stop_words.append(word.lower())

    #Adding a few additional words that were not in already
    #FIX: add all major world leaders, maybe countries,
    #maybe look at stopwords for each language!
    stop_words.append('Putin')
    stop_words.append('Vladimir')

    #A table in order to remove punctuation
    translator = str.maketrans('', '', string.punctuation)

    #Removing the stopwords & other chars
    #FIX: not being super good at remmoving ' or - 
    for idx, row in df.iterrows():
        tokens = word_tokenize(row['text'])
        filtered = [w for w in tokens if not w.lower() in stop_words]
        filtered = [token.lower().translate(translator) for token in tokens if token not in string.punctuation or '*' or '–' or " ‘ " or " ’ "]
    
        filtered = ' '.join(filtered) #unsure if needed
            
        df.at[idx, 'text'] = filtered
    
    #LEMMATIZING

    lemmatizer = WordNetLemmatizer()

    # Iterate over each row in the DataFrame
    for idx, row in df.iterrows():
        filtered_text = lemmatize_text(row['text'])
        df.at[idx, 'text'] = filtered_text

    return df

def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return ' '.join(lemmatized_tokens)


# -

def testSplitAndVectorizing(df):
    #Splits into input and target variables
    x_df = df['text']
    y_df = df['label']

    #Splittign the data
    x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.1, random_state=42)

    # VECTORIZING THE DATA
    vectorizer = CountVectorizer()
    xTrainCount = vectorizer.fit_transform(x_train)
    xTestCount = vectorizer.transform(x_test)
    
    return xTrainCount, xTestCount, y_train, y_test, vectorizer


# +
def intializeModels(xTrainCount, xTestCount, y_train, y_test):
    arrayOfScores = []
    
    #_______________________________________
    # Training the Naive Bayes classifier
    mnb = MultinomialNB()
    mnb.fit(xTrainCount, y_train)

    # Calculating accuracy on the test set
    accuracy = mnb.score(xTestCount, y_test)
    #print("Multinomial Bayes Accuracy:", accuracy * 100)
    arrayOfScores.append(accuracy)
    
    #_______________________________________
    # Training the Complement Bayes classifier
    cnb = ComplementNB()
    cnb.fit(xTrainCount, y_train)

    # Calculating accuracy on the test set
    accuracy = cnb.score(xTestCount, y_test)
    #print("Complement Bayes Accuracy:", accuracy * 100)
    arrayOfScores.append(accuracy)
    
    #_______________________________________
    # Training the Bernoulli Naive Bayes
    bnb = BernoulliNB()
    bnb.fit(xTrainCount, y_train)

    # Calculating accuracy on the test set
    accuracy = bnb.score(xTestCount, y_test)
    #print("Bernoulli NB Accuracy:", accuracy * 100)
    arrayOfScores.append(accuracy)
    
    #_______________________________________
    # Training the Decision Tree classifier
    dtc = DecisionTreeClassifier()
    dtc.fit(xTrainCount, y_train)

    # Calculating accuracy on the test set
    accuracy = dtc.score(xTestCount, y_test)
    #print("Decision Tree Classifier Accuracy:", accuracy * 100)
    arrayOfScores.append(accuracy)
    
    #_______________________________________
    # Training the Decision Tree Regressor
    drc = DecisionTreeRegressor()
    drc.fit(xTrainCount, y_train)

    # Calculating accuracy on the test set
    accuracy = drc.score(xTestCount, y_test)
    #print("Decision Regressor Classifier Accuracy:", accuracy * 100)
    arrayOfScores.append(accuracy)
    
    #_______________________________________
    #Perceptron
    percept = Perceptron(tol=1e-3, random_state=0)
    percept.fit(xTrainCount, y_train)
    percept.score(xTrainCount, y_train)
    #print("Perceptron accuracy: ", accuracy * 100)
    arrayOfScores.append(accuracy)
    
    #_______________________________________
    #PassiveAggressive
    passiveAggressive = PassiveAggressiveClassifier(max_iter=1000, random_state=0, tol=1e-3)
    passiveAggressive.fit(xTrainCount, y_train)
    accuracy = passiveAggressive.score(xTestCount, y_test)
    #print("Passive Aggressive Accuracy:", accuracy * 100)
    arrayOfScores.append(accuracy)
    
    avgAccuracyScore = np.mean(arrayOfScores)
    return avgAccuracyScore, mnb, cnb, bnb, dtc, drc, percept, passiveAggressive
    
def fakeNewsDetector(text, vectorizer, mnb, cnb, bnb, dtc, drc, percept, passiveAggressive):
    #_______________________________________  
    predictions = []
    ans = []
    ans.append(text)
    ansVector = vectorizer.transform(ans)
    
    ansVector2D = ansVector.reshape(1,-1)
    
    #adds all models' predictions to list of predictions
    predictions.append(mnb.predict(ansVector2D)[0])
    predictions.append(cnb.predict(ansVector2D)[0])
    predictions.append(bnb.predict(ansVector2D)[0])
    predictions.append(dtc.predict(ansVector2D)[0])
    predictions.append(drc.predict(ansVector2D)[0])
    predictions.append(percept.predict(ansVector2D)[0])
    predictions.append(passiveAggressive.predict(ansVector2D)[0])
    
    #finds the median of the list of predictions
    prediction = int(np.median(predictions))
    
    return prediction
    
#0 is fake 1 is true

