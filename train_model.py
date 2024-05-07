import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import set_config
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
import warnings
warnings.simplefilter("ignore")

# Loading the dataset
data = pd.read_csv("Language Detection.csv")

#print(data.head())

X = data["Text"]
y = data["Language"]

le = LabelEncoder()
y = le.fit_transform(y)

print(le.classes_)

data_list = []
for text in X:
    text = re.sub(r'[!@#$(),\n"%^*?\:;~`0-9]', ' ', text)
    text = re.sub(r'[[]]', ' ', text)
    text = text.lower()
    data_list.append(text)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

# creating bag of words using countvectorizer
cv = CountVectorizer()
cv.fit(X_train)
x_train = cv.transform(X_train).toarray()
x_test  = cv.transform(X_test).toarray()

model = MultinomialNB()
# model.fit(x_train, y_train)
# y_pred = model.predict(x_test)
#
# ac = accuracy_score(y_test, y_pred)
# cm = confusion_matrix(y_test, y_pred)
# cr = classification_report(y_test, y_pred)
# print("Accuracy is :",ac)

##Using Sklearn Pipeline
pipe = Pipeline([('vectorizer', cv), ('multinomialNB', model)])
pipe.fit(X_train, y_train)
set_config(display="diagram")

y_pred2 = pipe.predict(X_test)
ac2 = accuracy_score(y_test, y_pred2)
print("Accuracy is :",ac2)

model_version = '0.1.0'
with open(f'trained_model-{model_version}.joblib','wb') as f:
    joblib.dump(pipe, f)

print("Model has been saved !!!")
