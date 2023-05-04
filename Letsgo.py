#inspo
#https://medium.com/bitgrit-data-science-publication/sentiment-analysis-on-reddit-tech-news-with-python-cbaddb8e9bb6
#Load libaries
import pandas as pd
import numpy as np

# misc
import datetime as dt
from pprint import pprint
from itertools import chain

# reddit crawler
import praw

# sentiment analysis
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize, RegexpTokenizer # tokenize words
from nltk.corpus import stopwords

# visualization
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10, 8) # default plot size
import seaborn as sns
sns.set(style='whitegrid', palette='Dark2')
from wordcloud import WordCloud

#Dwonloading from nltk
#nltk.download('vader_lexicon') # Dataset of lexicons containing the sentiments of specific texts which powers the Vader Sentiment Analysis
#nltk.download('punkt') # for tokenizer
#nltk.download('stopwords')

#Setting up reddit client
r = praw.Reddit(user_agent='MooseSubstantial7289',
                client_id='P2kPPjuj8KBFvXtTF0GzGg',
                client_secret='v38cMcD6YxH-BzeIAukSNK-lJDJoeA',
                check_for_async=False)

#Selecting subreddit
subreddit = r.subreddit('climbing')

climb = [*subreddit.top(limit=None)] # top posts all time

print(len(climb))
print(climb[0].title)


#data frame with titiles
title = [climb.title for climb in climb]
climb_df = pd.DataFrame({
    "title" : title,
})
climb_df.head()

#Sentiment Analysis with VADER
sid = SentimentIntensityAnalyzer()
res = [*climb_df['title'].apply(sid.polarity_scores)] #applying pos/neg score to titles
sentiment_df = pd.DataFrame.from_records(res) #dictionary =>data frame
climb_df= pd.concat([climb_df,sentiment_df], axis=1, join='inner')
climb_df.head()

#threshold for positive negative
THRESHOLD = 0.2

conditions = [
    (climb_df['compound'] <= -THRESHOLD),
    (climb_df['compound'] > -THRESHOLD) & (climb_df['compound'] < THRESHOLD),
    (climb_df['compound'] >= THRESHOLD),
    ]

values = ["neg", "neu", "pos"]
climb_df['label'] = np.select(conditions, values)

climb_df.head()

#counting up 
climb_df.label.value_counts()
sns.histplot(climb_df.label)