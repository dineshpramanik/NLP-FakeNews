##Importing necessary Libraries
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score
import pickle

#Reading the Dataset
df = pd.read_csv('train.csv')

#Dropping the rows of null values
df = df.dropna()

messages = df.copy()
messages.reset_index(inplace=True)

ps = PorterStemmer()
lemma = WordNetLemmatizer()

corpus = []
for i in range(0, len(messages)):
    review = (re.sub('[^a-zA-Z]', ' ', messages['title'][i])).lower()
    review = review.split()

    review = [lemma.lemmatize(word, pos='n') for word in review if word not in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)

#Extract features with CountVectorizer
cv= CountVectorizer(max_features=2000, ngram_range=(1,3))
X= cv.fit_transform(corpus).toarray()   #Fit the Data
y= messages['label']

pickle.dump(cv, open('transform.pkl', 'wb'))

#Divide the dataset into Train and Test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

from sklearn.feature_extraction.text import CountVectorizer
count_df = pd.DataFrame(X_train, columns=cv.get_feature_names())

#Multinomial NaiveBayes classifier
from sklearn.naive_bayes import  MultinomialNB
classifier = MultinomialNB()

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

score = accuracy_score(y_test, y_pred)
print('Accuracy: ', np.round(score *100, 2), ' %')

cm = confusion_matrix(y_test, y_pred)
print(cm)


filename = 'fake-news-cv.pkl'
pickle.dump(classifier, open(filename, 'wb'))