import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, roc_auc_score
#from nltk.corpus import twitter_samples
#from preprocessReviews import preprocess
import numpy as np
import pickle

def vectorize_labels(labels):
	return_vect = []
	for label in labels:
		if label == 1:
			return_vect.append([0,1])
		else:
			return_vect.append([1,0])
	return return_vect

def top_class_feats(log_probs,vocab,n):
	sorted_pos = sorted(dict(enumerate(log_probs[1])).items(), key=operator.itemgetter(1), reverse=True)
	sorted_neg = sorted(dict(enumerate(log_probs[0])).items(), key=operator.itemgetter(1), reverse=True)
	pos = [vocab[elem[0]] for elem in sorted_pos[:n]]
	neg = [vocab[elem[0]] for elem in sorted_neg[:n]]
	return pos, neg

# pos = [(sentence,1) for sentence in preprocess(twitter_samples.strings('C:/Users/user/Desktop/PythonPrograms/twitterAnalysis/positive_tweets.json'))]
# neg = [(sentence,0) for sentence in preprocess(twitter_samples.strings('C:/Users/user/Desktop/PythonPrograms/twitterAnalysis/negative_tweets.json'))]

# dataset = pd.DataFrame(list(pos+neg), columns=['Sentence','Class'], index=np.arange(len(pos+neg)))

dataset = pd.read_csv('train.csv')

tfidf_vect = TfidfVectorizer(min_df=1,max_df=0.8,ngram_range=(1,1))
tfidf_vect.fit_transform(dataset['Sentence'])

vect = CountVectorizer(vocabulary=tfidf_vect.vocabulary_)
vect_b = CountVectorizer(binary=True)
X = vect.fit_transform(dataset['Sentence'])
X_b = vect_b.fit_transform(dataset['Sentence'])

vocab_idx = dict(zip(tfidf_vect.vocabulary_.values(),tfidf_vect.vocabulary_.keys()))

y = dataset['Class']

kfold = StratifiedKFold(n_splits=10,shuffle=True)
for train_idx, test_idx in kfold.split(X,y):
	train_X_b, test_X_b, train_y, test_y = X_b[train_idx], X_b[test_idx], y[train_idx], y[test_idx]
	clf_b = BernoulliNB()
	clf_b.fit(train_X_b,train_y)
	print 'bernoulli accuracy: %f, mse: %f' % (clf_b.score(test_X_b,test_y), mean_squared_error(vectorize_labels(test_y),clf_b.predict_proba(test_X_b)))
	print 'bernoulli auc: %f' % roc_auc_score(test_y,clf_b.predict(test_X_b))
	top_pos, top_neg top_class_feats(clf_b.feature_log_prob_,vocab_idx,20)
	print 'most informative positives:'
	print top_pos
	print 'most informative negatives:'
	print top_neg