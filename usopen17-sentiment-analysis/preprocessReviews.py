import re
import pandas as pd
import string
import nltk
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import pickle
import sys
import HTMLParser
import itertools
from nltk.stem import WordNetLemmatizer

reload(sys)
sys.setdefaultencoding('UTF8')

lem = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

html_parser = HTMLParser.HTMLParser()

with open('tweet_texts.pickle','rb') as f:
	revs = pickle.load(f)

processed_tweets = []

for i in range(len(revs)):
	try:
		sentence = html_parser.unescape(revs[i])
		sentence = sentence.decode('utf8').encode('ascii','ignore')
		sentence = ' '.join(re.findall('[A-Z][^A-Z]*',sentence))
		sentence = ''.join(''.join(s)[:2] for _, s in itertools.groupby(sentence))

		# remove all mentions
		processed = re.sub(r'([@?])(\w+)\b',' ',sentence)
		# remove all hashtags
		#processed = re.sub(r'([#?])(\w+)\b',' ',processed)
		processed = processed.replace('#',' ')
		# remove all urls
		processed = re.sub(r'^http\S+', '', processed)
		processed = processed.replace('http',' ')
		# place whitespace next to each punctuation
		processed = re.sub('([.,!?():;-])', r' \1 ', processed)
		# remove extra whitespace
		processed = re.sub('\s{2,}', ' ', processed)
		# remove all punctuation and symbols
		processed = re.sub(r'[^\P{P}a-zA-Z ]+','',processed)
		# word tokenize, stem and transform back into one string and remove retweet mentions
		words = [lem.lemmatize(word.lower()) for word in word_tokenize(processed) if word.isalpha() and len(word) > 1 and word != 'a' and word != 'an' and word != 'the']
		if len(words) > 1:
			words = ' '.join(elem for elem in [word for word in words if len(word) > 1])
			processed_tweets.append(words.strip())

		if len(processed_tweets) % 10000 == 0:
			print(len(processed_tweets))
			print(processed_tweets[-1])

	except Exception, e:
		print revs[i]
		print(e)

pd.DataFrame({'processed': list(set(processed_tweets))}).to_csv('C:/Users/user/Desktop/PythonPrograms/twitterAnalysis/processed_tweets3.csv')