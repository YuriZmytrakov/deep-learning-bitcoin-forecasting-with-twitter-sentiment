# Imports
import pandas as pd
import os
import zipfile
cwd = os.getcwd()
import re
import numpy as np
from sklearn.utils import shuffle
from collections import Counter
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")

# sklearn libraries
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC

from imblearn.over_sampling import SMOTE

# nlp libs
import nltk
nltk.download('stopwords')
nltk.download('punkt') # for stemming
nltk.download('wordnet') # for lemmatizing
nltk.download('omw-1.4')
nltk.download('vader_lexicon')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

print("################## Load data ##################")

# https://www.kaggle.com/utkarshxy/stock-markettweets-lexicon-data - 9917 labelled tweets
stock_data_file = os.getcwd() + "\\Data\\stock_data.csv"
stock_data = pd.read_csv(stock_data_file)
stock_data= stock_data.rename(columns=str.lower)
stock_data.dropna(inplace=True)
print(stock_data)

# https://www.kaggle.com/yash612/stockmarket-sentiment-dataset - 5791 labelled tweets
tweets_data_file = os.getcwd() + "\\Data\\tweets_labelled.csv"
tweets_data = pd.read_csv(tweets_data_file, delimiter=';')
tweets_data.drop(columns=['id','created_at'], inplace=True)
tweets_data.dropna(inplace=True)
print(tweets_data)

# combine
dataset_frames = [stock_data, tweets_data]
dataset_frames_names = ["stock_data", "tweets_data"]
combined_df = result = pd.concat(dataset_frames)

print("################## Definitions ##################")

max_features = 10000
def sentiment_to_label(sentiment):
    if sentiment == "positive" or sentiment == 1:
        return 1
    elif sentiment == "negative" or sentiment == -1 or sentiment == 0:
        return 0
    else:
        return pd.NA

porter_stemmer = PorterStemmer()
def stemming_tokenizer(str_input):
    words = re.sub(r"[^A-Za-z0-9\-]", " ", str_input).lower().split()
    words = [porter_stemmer.stem(word) for word in words]
    return words

lemmer=WordNetLemmatizer()
def lemmatizing_tokenizer(str_input):
    words = re.sub(r"[^A-Za-z0-9\-]", " ", str_input).lower().split()
    words = [lemmer.lemmatize(word) for word in words]
    return words

# From Lab
df_new = combined_df.reset_index(drop=True)
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    """
        text: a string
        return: modified initial string
    """
    
    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text. substitute the matched string in REPLACE_BY_SPACE_RE with space.
    text = BAD_SYMBOLS_RE.sub('', text) # remove symbols which are in BAD_SYMBOLS_RE from text. substitute the matched string in BAD_SYMBOLS_RE with nothing. 
    text = text.replace('x', '')
#    text = re.sub(r'\W+', '', text)
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # remove stopwors from text

    return text

print("################## Cleaning and prep ##################")

df_new['text'] = df_new['text'].apply(clean_text)
df_new['text'] = df_new['text'].str.replace('\d+', '')
df_new['sentiment'] = df_new['sentiment'].apply(sentiment_to_label)
df_new['sentiment'] = pd.to_numeric(df_new['sentiment'])
df_new.dropna(inplace=True)
df_new = shuffle(df_new)

X = df_new['text']
y = [int(x) for x in df_new['sentiment']]

count_vect = CountVectorizer(tokenizer=stemming_tokenizer, stop_words='english', max_features=max_features)
X_binary_counts = count_vect.fit_transform(X)
print("X_binary_counts.shape", X_binary_counts.shape)
print("X_binary_counts[20]:\n%s" %X_binary_counts[20])
print("X_binary_counts.toarray()[0]", X_binary_counts.toarray()[0])
print("len(X_binary_counts.toarray()[0])", len(X_binary_counts.toarray()[0]))
print("max(X_binary_counts.toarray()[0])", len(max(X_binary_counts.toarray(), key=len)))

tfidf_transformer = TfidfTransformer() 

X_tfidf_binary = tfidf_transformer.fit_transform(X_binary_counts)
print("X_tfidf_binary.shape", X_tfidf_binary.shape)
X_all_binary = X_tfidf_binary.toarray()

smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_all_binary, y)
print('SMOTE oversamples dataset shape %s' % Counter(y_res))
print("X_res.shape", X_res.shape)

print("################## Cleaning and prep ##################")
X_all_binary = X_res
y = [int(x) for x in y_res]

print("################## Train classifier ##################")
clf = SVC(C=50, gamma=1, random_state=42).fit(X_all_binary, y)

print("################## Load Bitcoin tweets ##################")
zipped_file_location = cwd = os.getcwd() + "\\Data\\bitcoin_tweets.zip"
unzipped_file_location = os.getcwd() + "\\Data\\"
tweets_file =  os.getcwd() + "\\Data\\bitcoin_tweets.csv"

df_tweets = None
if not os.path.isfile(unzipped_file_location + 'formatted_bitcoin_tweets.csv'):

    # Extract ZIP
    with zipfile.ZipFile(zipped_file_location, 'r') as zip_ref:
        zip_ref.extractall(unzipped_file_location)

    # Read from title file
    df_tweets = pd.read_csv(tweets_file)

    # Remove temp files 
    os.remove(os.path.join(unzipped_file_location,tweets_file))

    # Convert
    df_tweets["date"] = pd.to_datetime(df_tweets["date"], errors="coerce", utc=True)
    df_tweets["date"] = df_tweets["date"].dt.date
    df_tweets = df_tweets.set_index(pd.DatetimeIndex(df_tweets['date']))
    df_tweets.dropna(inplace=True)

    # Remove unrequired columns
    df_tweets = df_tweets[['text']]

    # Save 
    #compression_opts = dict(method='zip', archive_name='formatted_bitcoin_news.zip') 
    df_tweets.to_csv(unzipped_file_location + 'formatted_bitcoin_tweets.csv')
else:
    df_tweets = pd.read_csv(unzipped_file_location + 'formatted_bitcoin_tweets.csv')
    df_tweets = df_tweets.set_index(pd.DatetimeIndex(df_tweets['date']))
    df_tweets = df_tweets[['text']]

print("################## Clean tweets ##################")

df_tweets['text'] = df_tweets['text'].apply(clean_text)
df_tweets['text'] = df_tweets['text'].str.replace('\d+', '')

print("################## Label tweets ##################")

unzipped_file_location = os.getcwd() + "\\Data\\"

print("WARNING - TAKES 24+ HOURS TO COMPLETE!!!") 
if not os.path.isfile(unzipped_file_location + 'formatted_bitcoin_tweets_labelled.csv'):
    # we need to chunk as we don't have nearly enough memory to convert entire sparse matrix to array!
    n = 1000    #chunk row size
    bitcoin_tweet_sentiment_predictions = []
    list_df = np.array_split(df_tweets, n)
    for idx, df_slice in enumerate(list_df):
        print("Processing chunk " + str(idx+1) + "/" +str(n))
        slice_binary_counts = count_vect.transform(df_slice['text'])
        slice_tfidf_binary = tfidf_transformer.transform(slice_binary_counts)
        slice_binary_arr = slice_tfidf_binary.toarray()
        bitcoin_tweet_sentiment_predictions.extend(clf.predict(slice_binary_arr))

    df_tweets['sentiment'] = bitcoin_tweet_sentiment_predictions
    df_tweets['sentiment'] = df_tweets['sentiment'].apply(sentiment_to_label)
    df_tweets['sentiment'] = pd.to_numeric(df_tweets['sentiment'])
    df_tweets.to_csv(unzipped_file_location + 'formatted_bitcoin_tweets_labelled.csv')
else:
    df_tweets = pd.read_csv(unzipped_file_location + 'formatted_bitcoin_tweets_labelled.csv')
    df_tweets = df_tweets.set_index(pd.DatetimeIndex(df_tweets['date']))
    df_tweets = df_tweets[['text','sentiment']]

print("################## Prep historical data ##################")

csv_file =  os.getcwd() + "\\Data\\BTC-USD.csv"

# Read from title file
price_df = pd.read_csv(csv_file)

# Date
price_df['date'] = pd.to_datetime(price_df['date']) 
price_df = price_df.set_index(pd.DatetimeIndex(price_df['date']))

# On our Bitcoin data, we have a column ‘Close’ with the closing price of the day and ‘Open’ with the opening price of the day. 
# We want to get the percentage difference from the closing price with respect to the opening price so we have a variable with that day’s performance. 
# To get this variable we will calculate the logarithmic difference between the close and open price.
price_df['log_diff'] = np.log(price_df['close']) - np.log(price_df['open'])

# Generate our target variable by setting “1” if the performance was positive (log_diff > 0) and “0” if else.
price_df['target'] = [1 if log_diff > 0 else 0 for log_diff in price_df['log_diff']]

# Remove unrequired columns
price_df.drop(columns=['high','low','adj close','volume','date','log_diff'], inplace=True)

# Save 
price_df.columns = map(str.lower, price_df.columns)
price_df.to_csv(os.getcwd() + '\\Data\\formatted_bitcoin_price.csv')

print("Done.")