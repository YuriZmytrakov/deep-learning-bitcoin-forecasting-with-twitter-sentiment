# Imports
import pandas as pd
import os
#import flair
import re
import numpy as np
import seaborn as sns
from sklearn.utils import shuffle
from collections import Counter
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
from datetime import datetime
# sklearn libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
# ML models
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
import re

cwd = os.getcwd()

import nltk
nltk.download('stopwords')
nltk.download('punkt') # for stemming
nltk.download('wordnet') # for lemmatizing
nltk.download('omw-1.4')
nltk.download('vader_lexicon')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

################## Load data ##################
# https://www.kaggle.com/utkarshxy/stock-markettweets-lexicon-data - 9917 labelled tweets
stock_data_file = os.getcwd() + "\\Data\\stock_data.csv"
stock_data = pd.read_csv(stock_data_file)
stock_data= stock_data.rename(columns=str.lower)
stock_data.dropna(inplace=True)

# https://www.kaggle.com/yash612/stockmarket-sentiment-dataset - 5791 labelled tweets
tweets_data_file = os.getcwd() + "\\Data\\tweets_labelled.csv"
tweets_data = pd.read_csv(tweets_data_file, delimiter=';')
tweets_data.drop(columns=['id','created_at'], inplace=True)
tweets_data.dropna(inplace=True)

# combine
dataset_frames = [stock_data, tweets_data]
dataset_frames_names = ["stock_data", "tweets_data"]
combined_df = result = pd.concat(dataset_frames)

################## Definitions ##################

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

################## Cleaning + Prep ##################

df_new['text'] = df_new['text'].apply(clean_text)
df_new['text'] = df_new['text'].str.replace('\d+', '')
df_new['sentiment'] = df_new['sentiment'].apply(sentiment_to_label)
df_new['sentiment'] = pd.to_numeric(df_new['sentiment'])
df_new.dropna(inplace=True)
df_new = shuffle(df_new)

X = df_new['text']
y = [int(x) for x in df_new['sentiment']]

count_vect = CountVectorizer(tokenizer=stemming_tokenizer)
X_binary_counts = count_vect.fit_transform(X)
print("X_binary_counts.shape", X_binary_counts.shape)
print("X_binary_counts[20]:\n%s" %X_binary_counts[20])
print("X_binary_counts.toarray()[0]", X_binary_counts.toarray()[0])
print("len(X_binary_counts.toarray()[0])", len(X_binary_counts.toarray()[0]))
print("max(X_binary_counts.toarray()[0])", max(X_binary_counts.toarray()[0]))

tfidf_transformer = TfidfTransformer()

X_tfidf_binary = tfidf_transformer.fit_transform(X_binary_counts)
print("X_tfidf_binary.shape", X_tfidf_binary.shape)
X_all_binary = X_tfidf_binary.toarray()

smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_all_binary, y)
print('SMOTE oversamples dataset shape %s' % Counter(y_res))
print("X_res.shape", X_res.shape)

################## Split ##################

X_all_binary = X_res
y = [int(x) for x in y_res]

d_test_size = 0.2
i_nTestPts_binary = int(d_test_size*len(X_all_binary))
print("i_nTestPts_binary", i_nTestPts_binary)

b_split_byIndex = False
if b_split_byIndex == True:    
    X_train_binary, X_test_binary, y_train_binary, y_test_binary = X_all_binary[:-i_nTestPts_binary], X_all_binary[-i_nTestPts_binary:], y[:-i_nTestPts_binary], y[-i_nTestPts_binary:]
else:
    X_train_binary, X_test_binary, y_train_binary, y_test_binary = train_test_split(X_all_binary, y, test_size = d_test_size, random_state = 42)

X_train_binary = np.array(X_train_binary)
y_train_binary = np.array(y_train_binary)
y_test_binary = np.array(y_test_binary)
X_test_binary = np.array(X_test_binary)
print(X_train_binary.shape, y_train_binary.shape)
print(X_test_binary.shape, y_test_binary.shape)

################## Model comparison ##################

def RunTextClfModels_v2(gvModelNames, gvModels, gX_all, gy_all, ginFolds=3):
    ''' Get a list of models, data matrix X, target vector Y, and # of folds k
    Return ["RMSE", "R2", "MAE"] values (all values, averaged over k folds)
    '''

    print("RunTextClfModels_v2()")

    t_vMeasures = ["accuracy", "f1-score", "tr_time"]
    kfold_list = ["fold" + str(i) for i in list(range(ginFolds))] 

    # initialize the output dictionary
    output = dict((i,{}) for i in gvModelNames)
    for i in output:
        output[i] = dict((k,[]) for k in kfold_list)
        for k in output[i]:
            output[i][k] = dict((j,[]) for j in t_vMeasures)
    
    kf = KFold(n_splits = ginFolds)
    ctr = 0
    for train_index, test_index in kf.split(gX_all):
        
        # print("train_index:%s, test_index:%s" %(list(train_index), list(test_index)))

        print("--- start fold %d" %(ctr))
    
        # train/test split for k-fold cross-validation
        # X_train, X_test = gX_all.iloc[train_index], gX_all.iloc[test_index]
        # y_train, y_test = gy_all.iloc[train_index], gy_all.iloc[test_index]
        
        X_train, X_test = gX_all[train_index], gX_all[test_index]
        y_train, y_test = gy_all[train_index], gy_all[test_index]
    
        for index, value in enumerate(gvModelNames):
            
            print("--->>> start model %s" %(value))
            
            # model training/prediction
            start = datetime.now()
            model = gvModels[index]
            model.fit(X_train,y_train)
            y_pred = model.predict(X_test)
    
            # store the model outcomes
            output[value]["fold"+str(ctr)]["accuracy"] = np.round(accuracy_score(y_test, y_pred),3)
            output[value]["fold"+str(ctr)]["f1-score"] = np.round(f1_score(y_test, y_pred, average='weighted'),3)
            output[value]["fold"+str(ctr)]["tr_time"] = np.round((datetime.now()-start).total_seconds(),3)
        ctr = ctr + 1
    
    return output

model_names = ["SVC", "MNB", "LogR", "RF", "SGD", "MLP"]
model_lib = [SVC(C=50, gamma=1, random_state=42),  
             MultinomialNB(), 
             LogisticRegression(n_jobs=1, C=1e5), 
             RandomForestClassifier(random_state = 42), 
             SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None),
             MLPClassifier(random_state=42, max_iter=300)]

dic_ress = RunTextClfModels_v2(model_names, model_lib, X_all_binary, np.array(y), ginFolds=5)

pd_ress_bin = pd.DataFrame({(level1_key, level2_key): values
for level1_key, level2_dict in dic_ress.items()
for level2_key, values in level2_dict.items()}).T

pd_ress_bin = pd_ress_bin.reset_index()
pd_ress_bin.rename(columns={'level_0': 'Model', 'level_1': 'Fold'}, inplace=True)

pd_ress_avg = np.round(pd_ress_bin.groupby(["Model"], as_index=False).mean(),3)
pd_ress_std = pd_ress_bin.groupby(["Model"], as_index=False).agg(np.std, ddof=0).round(3).drop("Fold", axis=1)
pd_ress_std.rename(columns={'accuracy': 'accuracy-std', 'f1-score': 'f1-score-std','tr_time': 'tr_time-std'}, inplace=True)

pd_ress_summary = pd.concat([pd_ress_avg, pd_ress_std], axis = 1).T.drop_duplicates().T.sort_index(axis=1, ascending=True)
print(pd_ress_summary)