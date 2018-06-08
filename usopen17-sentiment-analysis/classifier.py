import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, roc_auc_score
import plotly.plotly as py
import plotly.graph_objs as go
#from nltk.corpus import twitter_samples
#from preprocessReviews import preprocess
import numpy as np
import pickle
import operator

def save_bar_graph(vocab,feat_tuples,n,label):
	## This method saves a visualization of top n features of a class with corresponding log probabilities
	data = [go.Bar(x=[vocab[elem[0]] for elem in feat_tuples[:n]],
				   y=[elem[1] for elem in feat_tuples[:n]],
				   text="Log probabilities")]
	layout = go.Layout(title="Log probabilities of %s features" % label, xaxis=dict(title='Feature'), yaxis=dict(title='Log Probability'))
	fig = go.Figure(data=data,layout=layout)
	py.image.save_as(fig,filename='%s-top-features.png' % label)

def vectorize_labels(labels):
	## Convert labels into two dimensional vectors for Mean Squared Error calculation
	return_vect = []
	for label in labels:
		if label == 1:
			return_vect.append([0,1])
		else:
			return_vect.append([1,0])
	return return_vect

def top_class_feats(log_probs,vocab,n):
	## This method returns top n features of a class, given a vector of feature log probabilities and a dictionary of vocabulary indices
	sorted_pos = sorted(dict(enumerate(log_probs[1])).items(), key=operator.itemgetter(1), reverse=True)
	sorted_neg = sorted(dict(enumerate(log_probs[0])).items(), key=operator.itemgetter(1), reverse=True)
	pos = [vocab[elem[0]] for elem in sorted_pos[:n]]
	neg = [vocab[elem[0]] for elem in sorted_neg[:n]]
	save_bar_graph(vocab,sorted_pos,n,'Positive')
	save_bar_graph(vocab,sorted_neg,n,'Negative')
	return pos, neg

## Following three lines show how the training dataset is generated using nltk corpus

# pos = [(sentence,1) for sentence in preprocess(twitter_samples.strings('C:/Users/user/Desktop/PythonPrograms/twitterAnalysis/positive_tweets.json'))]
# neg = [(sentence,0) for sentence in preprocess(twitter_samples.strings('C:/Users/user/Desktop/PythonPrograms/twitterAnalysis/negative_tweets.json'))]

# dataset = pd.DataFrame(list(pos+neg), columns=['Sentence','Class'], index=np.arange(len(pos+neg)))

dataset = pd.read_csv('train.csv')

## To eliminate feature that might bring noise to training data, apply filtering based on minimum and maximum document frequencies
## and n-grams. Different values for n-gram ranges are tested throughout the training period. Optimal performance was with unigrams.
tfidf_vect = TfidfVectorizer(min_df=1,max_df=0.8,ngram_range=(1,1))
tfidf_vect.fit_transform(dataset['Sentence'])

## Obtain a binarized feature matrix for the Bernoulli Naive Bayes classifier using the tfidf filtered vocabulary
vect_b = CountVectorizer(vocabulary=tfidf_vect.vocabulary_,binary=True)
X_b = vect_b.fit_transform(dataset['Sentence'])

## Create a vocabulary dictionary with feature matrix indices and corresponding words
vocab_idx = dict(zip(tfidf_vect.vocabulary_.values(),tfidf_vect.vocabulary_.keys()))

y = dataset['Class']

print 'Starting cross-validation...'
## Declare arrays to store metrics throughout the cross-validation step
ave_auc, ave_precision, ave_mse = [], [], []
kfold = StratifiedKFold(n_splits=10,shuffle=True)
for train_idx, test_idx in kfold.split(X_b,y):
	train_X_b, test_X_b, train_y, test_y = X_b[train_idx], X_b[test_idx], y[train_idx], y[test_idx]
	clf_b = BernoulliNB()
	clf_b.fit(train_X_b,train_y)
	ave_auc.append(roc_auc_score(test_y,clf_b.predict(test_X_b)))
	ave_precision.append(clf_b.score(test_X_b,test_y))
	ave_mse.append(mean_squared_error(vectorize_labels(test_y),clf_b.predict_proba(test_X_b)))
	
## Continuation of a method has been determined after evaluating the following three mean metrics
print "Average Precision: %f" % np.mean(ave_precision)
print "Average MSE: %f" % np.mean(ave_mse)	
print "Average AUC: %f" % np.mean(ave_auc)

## If the training has not been terminated, an actual model is trained and saved for use on the usopen17 tweet dataset
print 'Fitting the model...'
X_train, X_test, y_train, y_test = train_test_split(X_b,y,test_size=0.3)
clf = BernoulliNB()
clf.fit(X_train,y_train)
print 'Precision: %f & MSE: %f & AUC Score: %f' % (clf_b.score(X_test,y_test),mean_squared_error(vectorize_labels(y_test),clf_b.predict_proba(X_test)),roc_auc_score(y_test,clf_b.predict(X_test)))

with open('bernoulli_clf.pickle','wb') as f:
	pickle.dump(clf,f)

## Save CountVectorizer to extract trained features from new data
with open('feature_vectorizer.pickle','wb') as f:
	pickle.dump(vect_b,f)

## Top features of the trained model are printed and visualizations are saved in project directory
top_pos, top_neg = top_class_feats(clf.feature_log_prob_,vocab_idx,20)
print 'most informative positives:'
print top_pos
print 'most informative negatives:'
print top_neg