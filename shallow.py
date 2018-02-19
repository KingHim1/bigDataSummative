import pandas as pd
import re
import nltk
import numpy as np
from nltk import f_measure


#downloaded "popular" collection
from nltk.corpus import stopwords
#create bag of words
from sklearn.feature_extraction.text import CountVectorizer
#create tf and tdidf data features
from sklearn.feature_extraction.text import TfidfTransformer
#Multinomial Naïve Bayes classifier
from sklearn.naive_bayes import MultinomialNB

stopws = set(stopwords.words("english"))
train = pd.read_csv("./news_ds.csv", header=0, sep=',', quoting=1, engine='python')


def preprocessText(text):
    letters_only = re.sub("[^a-zA-Z]", " ", text)
    letters_only.lower()
    words = letters_only.split()
    words = [word for word in words if not word in stopws]
    return " ".join(words)

def preprocessAllTexts(texts):
    cleanedTxts = []
    for text in texts:
        cleanedTxts.append(preprocessText(text))
    return cleanedTxts

cleanedTexts = preprocessAllTexts(train['TEXT'])

vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, max_features = 5000)

train_data_features = vectorizer.fit_transform(cleanedTexts)
train_data_features = train_data_features.toarray()
vocab = vectorizer.get_feature_names()

count = 0
for t in train_data_features[0]:
    count += t
print(count)

tfTransformer = TfidfTransformer(use_idf=False,smooth_idf=False, norm="l1")
tfIdfTransformer = TfidfTransformer(use_idf=True, smooth_idf=True)
ngram_vectorizer = CountVectorizer(analyzer='char_wb', ngram_range=(4, 4))

tf_data_features = tfTransformer.fit_transform(train_data_features)
# tf_data_features = tf_data_features.toarray()
tfidf_data_features = tfIdfTransformer.fit_transform(train_data_features)
# tfidf_data_features = tfidf_data_features.toarray()
ngram_data_features = ngram_vectorizer.fit_transform(cleanedTexts)

print(ngram_data_features.shape)

print(tf_data_features.shape)

print(tfidf_data_features.shape)

#learning
y = (train["LABEL"])
tfNBC = MultinomialNB()
tfNBC.fit(tf_data_features[:5000], y[:5000])
tfidfNBC = MultinomialNB()
tfidfNBC.fit(tfidf_data_features[:5000], y[:5000])
ngramNBC = MultinomialNB()
ngramNBC.fit(ngram_data_features[:5000], y[:5000])
refMask = np.array(y[5000:], dtype=bool)
y = set(train["ID"][5000:][refMask])
# print(y)
# print(tfNaiveBayesClassifier.predict(tf_data_features[2:3]))

def calcPrecRecallFMeasure(reference, prediction):
    precision = nltk.f_measure(reference, prediction, alpha=1.0)
    recall = nltk.f_measure(reference, prediction, alpha=0)
    f_measure = nltk.f_measure(reference, prediction, alpha=0.5)
    return [precision, recall, f_measure]

print("Precision, Recall and F_Measure for tf data features\n")
tfPredictedMask = np.array(tfNBC.predict(tf_data_features[5000:]), dtype=bool)
tfPrediction = set(train["ID"][5000:][tfPredictedMask])
tfMeasures = calcPrecRecallFMeasure(tfPrediction, y)
# print((tfNBC.predict(tf_data_features[1000:])))
print("Precision: " + tfMeasures[0].__str__() + "\n")
print("Recall: " + tfMeasures[1].__str__() + "\n")
print("f_measure: " + tfMeasures[2].__str__() + "\n\n")

print("Precision, Recall and F_Measure for tfidf data features\n")
tfidfPredictedMask = np.array(tfidfNBC.predict(tfidf_data_features[5000:]), dtype=bool)
tfidfPrediction = set(train["ID"][5000:][tfidfPredictedMask])
tfidfMeasures = calcPrecRecallFMeasure(tfidfPrediction, y)

print("Precision: " + tfidfMeasures[0].__str__() + "\n")
print("Recall: " + tfidfMeasures[1].__str__() + "\n")
print("f_measure: " + tfidfMeasures[2].__str__() + "\n\n")

print("Precision, Recall and F_Measure for ngram data features\n")
ngramPredictedMask = np.array(ngramNBC.predict(ngram_data_features[5000:]), dtype=bool)
ngramPrediction = set(train["ID"][5000:][ngramPredictedMask])
ngramMeasures = calcPrecRecallFMeasure(ngramPrediction, y)
print("Precision: " + ngramMeasures[0].__str__() + "\n")
print("Recall: " + ngramMeasures[1].__str__() + "\n")
print("f_measure: " + ngramMeasures[2].__str__() + "\n\n")