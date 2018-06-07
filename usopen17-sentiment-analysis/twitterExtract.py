import tweepy
from tweepy import OAuthHandler
import time, datetime

consumer_key = 'cons-key'
consumer_secret = 'cons-secret'
access_token = 'acc-token'
access_secret = 'acc-secret'

auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)

api = tweepy.API(auth)

thefile = open('#usopen Tweets with Coordinates.txt', 'w')
tweets = 0

maxID = float("inf")

while tweets<250000 :
	try:
		for tweet in tweepy.Cursor(api.search,
								   q = "#usopen",
								   lang = "en",
								   rpp = 100,
								   max_id = maxID,
								   count = 100).items():
			print tweets
			thefile.write("%d\t%s\t%s\t%s\n" % (tweet.id, tweet.created_at, tweet.coordinates, tweet.text.encode('utf-8')))
			tweets+=1
			if tweet.id<maxID:
				maxID = tweet.id - 1
	except tweepy.TweepError:
		time.sleep(60*15)
		continue


