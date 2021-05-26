# Sentimental Analysis Of Tweets Using Decision Trees
## Introduction
This project uses two datasets for getting sentiments of many of tweets: -
* [Training Data](https://raw.githubusercontent.com/dD2405/Twitter_Sentiment_Analysis/master/train.csv)
* [Testing Data](https://raw.githubusercontent.com/dD2405/Twitter_Sentiment_Analysis/master/test.csv)
## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install multiple modules. 

Main modules with installation: -

```bash
pip install sklearn
pip install pandas
pip install nltk
pip install numpy
``` 
## Usage
Import necessary files
```python
from sklearn import tree
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import nltk
from nltk import PorterStemmer
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, confusion_matrix,accuracy_score
```
## Pre-Processing tweets: -
```python
def remove_at(text,pattern="@"):
    r = re.findall("@[\w]*",text)
    for i in r:
        text = re.sub(i,"",text)
    return text
def remove_hash(text,pattern="#"):
    r = re.findall("#[\w]*",text)
    for i in r:
        text = re.sub(i,"",text)
    return text
# Removing Punctuations
combined['tweet'] = np.vectorize(remove_at)(combined['tweet'])
combined['tweet'] = np.vectorize(remove_hash)(combined['tweet'])
combined['tweet'] = combined['tweet'].str.replace("[^a-zA-Z#]", " ")
# Tokenizing
combined['tweet'] = combined['tweet'].apply(lambda x: [w.lower() for w in x.split()])
# Eliminating words which are less than 4 words
combined['tweet'] = combined['tweet'].apply(lambda x: [w.lower() for w in x if len(w)>3])
# Stemming words
combined['tweet'] = combined['tweet'].apply(lambda x: " ".join([PorterStemmer().stem(w) for w in x]))
```
## Getting TF-IDF values from pre-processed tweets and splitting data: -
```python
tfidf=TfidfVectorizer(max_df=0.90, min_df=2,max_features=1000,stop_words='english')
tfidf_matrix=tfidf.fit_transform(combined['tweet'])
df_tfidf = pd.DataFrame(tfidf_matrix.todense())
# splitting
train_tfidf_matrix = tfidf_matrix[:31962]
# data in a matrix
train_tfidf_matrix.todense()
x_train, x_test, y_train, y_test = train_test_split(train_tfidf_matrix,train['label'],test_size=0.33,random_state=17)
```

## Fitting Decision Tree on training data and predicting sentiments(0 or 1) using Decision Tree (with accuracy score and confusion matrix)
```python
# creating and pruning Decision Tree
dct = tree.DecisionTreeClassifier(criterion='entropy', random_state=1, max_depth=5)
dct.fit(x_train, y_train)
y_pred = dct.predict(x_test)
tree.plot_tree(dct)
acc = accuracy_score(y_test,y_pred)
cm = confusion_matrix(y_test, y_pred)
f"Accuracy = {acc*100}% \n Confusion Matrix = {cm}"
```
# Results
## Accuracy Score and Confusion Matrix
![alt](https://github.com/yash722/Sentimental-Analysis-Of-Tweets-Using-Decision-Trees/blob/main/Acc_score_and_Conf.png)
## Decision Tree
![alt](https://github.com/yash722/Sentimental-Analysis-Of-Tweets-Using-Decision-Trees/blob/main/Dec_Tree.png)
