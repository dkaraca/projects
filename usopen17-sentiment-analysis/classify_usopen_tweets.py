import pandas as pd
import pickle
import plotly.plotly as py
import plotly.graph_objs as go
from nltk import word_tokenize
import numpy as np
from preprocessReviews import preprocess

keywords = ['roger','federer','goat']

def visualize_results(result_df):
	## This function creates bar plots for top positive and negative words using the clasifications made by the trained model in the usopen
	## dataset
	grouped = result_df.groupby('Label')
	pos_words = grouped.get_group(1)['Tweet'].str.split(expand=True).unstack().value_counts()
	neg_words = grouped.get_group(0)['Tweet'].str.split(expand=True).unstack().value_counts()

	data = [go.Bar(x=pos_words.index.values[2:50],y=pos_words.values[2:50],text="Count")]
	layout = go.Layout(title='Top 50 Positive Federer Word Counts',xaxis=dict(title='Words'),yaxis=dict(title='Counts'))
	fig = go.Figure(data=data, layout=layout)
	py.image.save_as(fig,filename='federer-top-positive.png')

	data = [go.Bar(x=neg_words.index.values[2:50],y=neg_words.values[2:50],text="Count")]
	layout = go.Layout(title='Top 50 Negative Federer Word Counts',xaxis=dict(title='Words'),yaxis=dict(title='Counts'))
	fig = go.Figure(data=data, layout=layout)
	py.image.save_as(fig,filename='federer-top-negative.png')

with open('bernoulli_clf.pickle','rb') as f:
	clf = pickle.load(f)

with open('feature_vectorizer.pickle','rb') as f:
	vect_b = pickle.load(f)

data = pd.read_csv('raw_tweets.csv')
filtered_tweets = preprocess(data['Tweet'])

## After the preprocessing step, get the tweets that include any of the specified keywords above. These are domain specific keywords that relate
## to Roger Federer.
federer_tweets = [filtered_tweets[i] for i in range(len(filtered_tweets)) if any(keyword in word_tokenize(filtered_tweets[i]) for keyword in keywords)]
federer_tweets = pd.DataFrame({"Tweet":federer_tweets})

## Transform the raw text in this new data into the Feature Matrix that was fit on the training data.
feature_matrix = vect_b.transform(federer_tweets['Tweet'])

## Classify new data using the trained Bernoulli Naive Bayes classifier
labels = clf.predict(feature_matrix)

federer_tweets['Label'] = labels

## Visualize results for comparison with the training data
visualize_results(federer_tweets)