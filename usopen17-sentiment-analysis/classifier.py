import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, roc_auc_score
#from nltk.corpus import twitter_samples
#from preprocessReviews import preprocess
import numpy as np

def vectorize_labels(labels):
	return_vect = []
	for label in labels:
		if label == 1:
			return_vect.append([0,1])
		else:
			return_vect.append([1,0])
	return return_vect

# pos = [(sentence,1) for sentence in preprocess(twitter_samples.strings('C:/Users/user/Desktop/PythonPrograms/twitterAnalysis/positive_tweets.json'))]
# neg = [(sentence,0) for sentence in preprocess(twitter_samples.strings('C:/Users/user/Desktop/PythonPrograms/twitterAnalysis/negative_tweets.json'))]

# dataset = pd.DataFrame(list(pos+neg), columns=['Sentence','Class'], index=np.arange(len(pos+neg)))

dataset = pd.read_csv('train.csv')

tfidf_vect = TfidfVectorizer(min_df=1,max_df=0.8,ngram_range=(1,1))
tfidf_vect.fit_transform(dataset['Sentence'])

vect = CountVectorizer()
X = vect.fit_transform(dataset['Sentence'])
y = dataset['Class']

kfold = StratifiedKFold(n_splits=10,shuffle=True)
for train_idx, test_idx in kfold.split(X,y):
	train_X, test_X, train_y, test_y = X[train_idx], X[test_idx], y[train_idx], y[test_idx]
	clf = MultinomialNB()
	clf.fit(train_X,train_y)
	print 'accuracy: %f, mse: %f' % (clf.score(test_X,test_y), mean_squared_error(vectorize_labels(test_y),clf.predict_proba(test_X)))
	print 'auc: %f' % roc_auc_score(test_y,clf.predict(test_X))