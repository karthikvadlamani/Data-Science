# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from textblob import TextBlob
from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import re
import plotly
import plotly.graph_objs as go
import numpy as np


compound = []
count = 0

ckey = ''
csecret = ''

atoken = ''
asecret = ''



class listener(StreamListener):
    
    def on_status(self, status):
        global initime
        
        tweet = re.findall("[a-zA-Z]+|[.,!?;']", status.text, flags = re.MULTILINE)
        s = " "
        tweet = s.join(tweet)
        blob = TextBlob(tweet)
        
        global compound
        global count
        global senti
        count+= 1
        senti = 0
        for sen in blob.sentences:
            senti+= sen.sentiment.polarity
        compound.append(senti)

        print(count)
        print(tweet)
        print(senti)
        #print(compound)
   
        if count == 50:
            return False
        else:
            return True
        
        
    def on_error(self, status):
        print(status)
        
    
        
auth = OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)

twitterStream = Stream(auth, listener(count))
twitterStream.filter(track = ["#incredibolt"])

trace2 = go.Scatter(x = np.linspace(0, count, count), y = compound, mode = 'lines+markers', name = 'compound')
plotly.offline.plot([trace2], filename='line-mode.html')
            
            
            
