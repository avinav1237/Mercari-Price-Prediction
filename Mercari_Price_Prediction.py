#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# from wordcloud import WordCloud
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
# from gensim.models import Word2Vec


# ## LOAD THE DATA SET

# In[2]:


data=pd.read_csv('Mercari_df3.csv')
print("shape",data.shape)
print(data.head(5))

data.drop('Unnamed: 0',axis=1,inplace=True)
Data.drop(['price'],axis=1,inplace=True)
X=data.drop('logPrice',axis=1)
y=data['logPrice']
# In[3]:


# data.columns


# # In[4]:


# data.info()


# # ## Now we will check for NULl values as this is an imp part for giving accurate results in the prediction

# # In[5]:


# data.isnull().any()


# # ## category_name,brand_name,item_description have null values

# # In[6]:


# def fill_the_null(data):
#     data.category_name.fillna(value = "Others", inplace = True)
#     data.brand_name.fillna(value = "Not known", inplace = True)
#     data.item_description.fillna(value = "No description given", inplace = True)
#     return data
    


# # In[7]:


# train=fill_the_null(data)


# # In[8]:


# train.isnull().any()


# # ## Loading the test file

# # In[9]:


# test = pd.read_csv('test_stg2.tsv', sep='\t')
# # data= data[1:1000]
# print("shape",np.shape(test))
# test.head(5)


# # In[10]:


# test.info()


# # ## Checking for null values in data

# # In[11]:


# test.isnull().any()


# # ## category_name,brand_name,item_description have null values

# # In[12]:


# test=fill_the_null(test)


# # In[13]:


# test.isnull().any()


# # ## Now there are no null values

# # # univariate data anlysis

# # In[ ]:





# # In[ ]:





# # In[ ]:





# # In[ ]:





# # In[14]:


# train.price.describe()


# # In[15]:


# X=data


# # In[16]:


# fig, ax = plt.subplots(figsize=(14,5))
# plt.title('Price distribution', fontsize=15)
# sns.boxplot(X.price,showfliers=False)
# ax.set_xlabel('Price',fontsize=15)

# plt.show()


# # ## We can see most of the items have the price value between 25-28.And all the items have their price listings between 10-28

# # 

# # In[17]:


# fig, ax = plt.subplots(figsize=(14,8))
# ax.hist(train.price,bins=30,range=[0,200],label="Price")
# plt.title('Price distribution', fontsize=15)
# ax.set_xlabel('Price',fontsize=15)
# ax.set_ylabel('No of items',fontsize=15)

# plt.show()


# # ## It reveals most of the items have their prices between 15-22.but as its a bit skewed so we will use log to see the change in distribution
# # 

# # In[18]:


# #We will add log(price) as a column in our train data
# train["logPrice"] = np.log(train["price"]+1)


# # In[19]:


# train.head()


# # In[20]:


# fig, ax = plt.subplots(figsize=(14,5))
# plt.title('Price distribution', fontsize=15)
# sns.boxplot(train.logPrice,showfliers=False)
# ax.set_xlabel('log(Price+1)',fontsize=15)

# plt.show()


# # ## We have scaled down our 'price' feature logprice. We have added log(Price+1) to it as log(0) is undefined. hence if price for an item is 0, then the item will have no price as defined.

# # In[21]:


# fig, ax = plt.subplots(figsize=(14,8))
# ax.hist(train.logPrice,bins=40,label="Price")
# plt.title('Price distribution', fontsize=15)
# ax.set_xlabel('log(Price+1)',fontsize=15)
# ax.set_ylabel('No of items',fontsize=15)

# plt.show()


# # In[22]:


# sns.distplot(train.logPrice)


# # In[23]:


# train['train_id'].count()


# # ## The 'log(price+1)' feature has the price range spread over 2.4-3.7 for most of the items

# # ## Shipping

# # In[24]:


# print("0: shipping charges paid by seller")
# print("1: shipping charges paid by buyers")
# print("COUNT:\n",train['shipping'].value_counts())
# print("Fraction:\n",train['shipping'].value_counts(normalize=True))


# # In[25]:


# seller_charged = []
# buyer_charged = []
# for i in (range(0,len(train['shipping']))):
#     if train['shipping'][i]==0:
#         seller_charged.append(train['logPrice'][i])
#     else:
#         buyer_charged.append(train['logPrice'][i])


# # In[26]:


# print(len(seller_charged))
# print(len(buyer_charged))


# # In[27]:


# #Ref: https://stackoverflow.com/questions/6871201/plot-two-histograms-at-the-same-time-with-matplotlib
# #Ref: https://stackoverflow.com/questions/28398200/matplotlib-plotting-transparent-histogram-with-non-transparent-edge
# fig, ax = plt.subplots(figsize=(14,8))
# ax.hist(buyer_charged,bins=30,range=[0,8],label="Buyer_charged",color='b',alpha=0.5)
# ax.hist(seller_charged,bins=30,range=[0,8],label="seller_charged",color='r',alpha=0.5)
# plt.title('Price distribution', fontsize=15)
# ax.set_xlabel('log(Price+1)',fontsize=15)
# ax.set_ylabel('No of items',fontsize=15)
# plt.legend()

# plt.show()


# # ## Here, we can see that for items which have lesser price, the shipping had to be paid by the buyer for profit reasons. Also, as the price increases, we can see that the shipping charges have been paid by the seller.. And there is a lot of overlap for items where both buyer and seller have been charged.

# # ## Item category

# # In[28]:


# print("No of unique values in item category is:",train['category_name'].nunique())


# # In[29]:


# print("Top-10 unique category by frequency:\n\n",train['category_name'].value_counts()[:10])


# # In[30]:


def cat_split(row):
    try:
        text = row
        text1, text2, text3 = text.split('/')
        return text1, text2, text3
    except:
        return ("Label not given", "Label not given", "Label not given")


# In[31]:


X['general_cat'], X['subcat_1'], X['subcat_2'] = zip(*X['category_name'].apply(lambda x: cat_split(x)))
# train.head()


# # In[32]:


# test['general_cat'], test['subcat_1'], test['subcat_2'] = zip(*test['category_name'].apply(lambda x: cat_split(x)))


# # In[33]:


# test.head()


# # In[34]:


# print("No of unique values in main category: ",train['general_cat'].nunique())
# print("No of unique values in Sub_category1: ",train['subcat_1'].nunique())
# print("No of unique values in Sub_category2: ",train['subcat_2'].nunique())


# # In[35]:


# fig, ax = plt.subplots(figsize=(20,10))
# sns.countplot(x='general_cat', data=train, ax=ax)
# plt.title('Item count by main_category',fontsize=25)
# plt.ylabel('No of items',fontsize=25)
# plt.xlabel('')
# plt.xticks(rotation=70,fontsize=20)
# plt.yticks(fontsize=20)

# plt.show()


# # ## From this plot, we can conclude that items of women has the maximum number in main category.

# # In[36]:


# sns.boxplot(x=train.general_cat,y=train.price,orient='v')


# # In[37]:


# print("Top-10 subcategory_1:")
# train.subcat_1.value_counts()[:10].plot(kind = 'bar',figsize = (20,10), title="Top_10 subcategory_1",fontsize=20)


# # ## Here, it lists the top 10 items with greatest frequencies in sub-category 1

# # In[38]:


# print("Top-10 subcategory_2:")
# train.subcat_2.value_counts()[:10].plot(kind = 'bar',figsize = (20,10), title="Top_10 subcategory_2",fontsize=20)


# # ## Here, it lists the top 10 items with greatest frequencies in sub-category 2

# # In[39]:


# print("No of unique brands: ",train['brand_name'].nunique())


# # In[40]:


# print("Top-10 brands by frequency of sale:")
# train.brand_name.value_counts()[:10].plot(kind = 'bar',figsize = (20,10), title="Top_10 brand_names",fontsize=20)


# # ## Brand name

# # In[41]:


# print("No of unique brands: ",train['brand_name'].nunique())


# # In[42]:


# print("Top-10 brands by frequency of sale:")
# train.brand_name.value_counts()[:10].plot(kind = 'bar',figsize = (20,10), title="Top_10 brand_names",fontsize=20)


# # ## For most of the items, the brand name has not been listed can be deduced from the plot. Second to it, most number of items have 'Pink' and "Nike" as brand names.

# # ## Item description

# # In[43]:


# def length(description):
#     count = 0
#     for i in description.split():
#         count+=1
#     return count


# # In[44]:


# dec = []
# for i in train['item_description']:
#     temp = []
#     temp.append(i)
#     temp.append(length(str(i)))
#     dec.append(temp)

# print(dec[1])
# print(len(dec))


# # In[45]:


# mydf = pd.DataFrame(dec,columns=['desc','desc_length'])
# print(mydf.head(2))


# # In[46]:


# train['description_len'] = mydf['desc_length']


# # In[47]:


# train.head()


# # In[48]:


# sns.boxplot(x=train.description_len,orient='v')


# # ## The box-plot of decsription length shows that most of the tiems have description length ranging between 15-40 words.

# # In[49]:


# sns.boxplot(y=train.description_len,x=train.shipping,orient='v')


# # In[50]:


# sns.FacetGrid(train,hue='shipping',size=8)    .map(sns.distplot,'description_len')     .add_legend()
# plt.title('distribution plot between status and year')
# plt.ylabel('shipping')


# # ## As length increases, price charged becomes less. Most of the items with lesser description length have more price value.

# # In[51]:


# sns.boxplot(x=train.description_len,y=train.price,orient='v')


# # In[52]:


# print("Top-10 item_descriptions by frequency:")
# train.description_len.value_counts()[:10].plot(kind = 'bar',figsize = (20,10),title="Top_10 item descriptions by frequency",fontsize=20)


# # In[53]:


# wc = WordCloud(max_words=300,width = 1200, height = 900).generate(" ".join(train.item_description.astype(str)))
# plt.figure(figsize = (18, 13))
# plt.imshow(wc)
# plt.axis("off")
# plt.show()


# # ## Basic Feature Engineering and Preprocessing

# # In[54]:


# # https://stackoverflow.com/a/47091490/4084039
import re

def decontracted(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase


# # In[55]:


import nltk
nltk.download('stopwords')


# # In[56]:


stopword=set(stopwords.words("english"))


# # ## Calculating sentiment score on item description as a feature

# # In[57]:


# def clean_html(data):
#     cleanr=re.compile('<.*?>')
#     cleantext=re.sub(cleanr ,' ',data)
#     return cleantext


# # In[ ]:





# # In[58]:


# train["item_description"][100]


# # ## as these needs some cleaning removing special characters

# # In[59]:


# from tqdm import tqdm
# preprocessed_total_train = []
# # tqdm is for printing the status bar
# for sentance in tqdm(train['item_description'].values):
#     sent = decontracted(sentance)
#     sent = sent.replace('\\r', ' ')
#     sent = sent.replace('\\"', ' ')
#     sent = sent.replace('\\n', ' ')
#     sent = re.sub('[^A-Za-z0-9]+', ' ', sent)
#     # https://gist.github.com/sebleier/554280
#     sent = ' '.join(e for e in sent.split() if e.lower() not in stopword)
#     preprocessed_total_train.append(sent.lower().strip())

# # after preprocesing
# preprocessed_total_train[20000]


# # In[60]:


# sno=nltk.stem.SnowballStemmer('english')


# # In[61]:


# print(sno.stem("tasty"))


# # In[62]:


# train["item_description"][1000]


# # In[63]:


# from sklearn.feature_extraction.text import TfidfTransformer
# from sklearn.feature_extraction.text import TfidfVectorizer
# vectorizer = TfidfVectorizer(stop_words='english')


# # In[64]:


# from sklearn.cluster import KMeans


# # In[65]:


# X = vectorizer.fit_transform(train["item_description"])


# # In[66]:


# true_k = 2
# model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
# model.fit(X)


# # In[67]:


# order_centroids = model.cluster_centers_.argsort()[:, ::-1]
# terms = vectorizer.get_feature_names()


# # In[68]:


# import nltk
# from nltk.sentiment.vader import SentimentIntensityAnalyzer

# import nltk
# nltk.download('vader_lexicon')

# sid = SentimentIntensityAnalyzer()

# train_sentiment = []; 
# for sentence in tqdm(preprocessed_total_train):
#     for_sentiment = sentence
#     ss = sid.polarity_scores(for_sentiment)
#     train_sentiment.append(ss)


# # In[69]:


# negative=[]
# neutral=[]
# positive=[]
# for i in train_sentiment:
    
#     for polarity,score in i.items():
#         if(polarity=='neg'):
#             negative.append(score)
#         if(polarity=='neu'):
#             neutral.append(score)
#         if(polarity=='pos'):
#             positive.append(score)
       


# # In[70]:


# train['negative']=negative
# train['neutral']=neutral
# train['positive']=positive


# # In[71]:


# columns = list(train.columns)
# plt.figure(figsize = (10, 10))
# sns.heatmap(train[columns].corr(), annot = True, linewidth = 0.5)
# plt.show()


# # ## We can see that description length has a fair correleation with the price of an item. Hence, we will include this as an additional feature in the features list.

# # ## Also, features of sentiment score share some correlation with the target variable 'Price'. Hence we are including sentiment score on item description as an additional feature

# # In[72]:


# y_train = np.log(train["price"]+1)
# train_ids = train['train_id'].values.astype(np.int32)
# train.drop(['price', 'train_id','logPrice','description_len','negative', 'neutral','positive'], axis=1, inplace=True)


# # In[73]:


# train.head()


# # In[ ]:





# # In[ ]:





# # In[74]:


# train.head()


# # In[75]:


# test_ids = test['test_id'].values.astype(np.int32)
# test.drop(['test_id'], axis=1, inplace=True)


# # ## Splitting the dataset into train and test

# # In[76]:


# X=train 
# y=y_train


# # In[77]:


from sklearn.model_selection import train_test_split
df_train, df_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[78]:


from tqdm import tqdm
preprocessed_test_des = []
# tqdm is for printing the status bar
for sentance in tqdm(df_test['item_description'].values):
    sent = decontracted(sentance)
    sent = sent.replace('\\r', ' ')
    sent = sent.replace('\\"', ' ')
    sent = sent.replace('\\n', ' ')
    sent = re.sub('[^A-Za-z0-9]+', ' ', sent)
    # https://gist.github.com/sebleier/554280
    sent = ' '.join(e for e in sent.split() if e.lower() not in stopword)
    preprocessed_test_des.append(sent.lower().strip())

# after preprocesing
preprocessed_test_des[20000]


# In[79]:


from tqdm import tqdm
preprocessed_train_des = []
# tqdm is for printing the status bar
for sentance in tqdm(df_train['item_description'].values):
    sent = decontracted(sentance)
    sent = sent.replace('\\r', ' ')
    sent = sent.replace('\\"', ' ')
    sent = sent.replace('\\n', ' ')
    sent = re.sub('[^A-Za-z0-9]+', ' ', sent)
    # https://gist.github.com/sebleier/554280
    sent = ' '.join(e for e in sent.split() if e.lower() not in stopword)
    preprocessed_train_des.append(sent.lower().strip())

# after preprocesing
preprocessed_train_des[20000]


# In[80]:


from tqdm import tqdm
# preprocessed_test = []
# # tqdm is for printing the status bar
# for sentance in tqdm(test['item_description'].values):
#     sent = decontracted(sentance)
#     sent = sent.replace('\\r', ' ')
#     sent = sent.replace('\\"', ' ')
#     sent = sent.replace('\\n', ' ')
#     sent = re.sub('[^A-Za-z0-9]+', ' ', sent)
#     # https://gist.github.com/sebleier/554280
#     sent = ' '.join(e for e in sent.split() if e.lower() not in stopword)
#     preprocessed_test.append(sent.lower().strip())

# # after preprocesing
# preprocessed_test[20000]


# In[81]:


# from tqdm import tqdm
# preprocessed_test2 = []
# # tqdm is for printing the status bar
# for sentance in tqdm(test['item_description'].values):
#     sent = decontracted(sentance)
#     sent = sent.replace('\\r', ' ')
#     sent = sent.replace('\\"', ' ')
#     sent = sent.replace('\\n', ' ')
#     sent = re.sub('[^A-Za-z0-9]+', ' ', sent)
#     # https://gist.github.com/sebleier/554280
#     sent = ' '.join(e for e in sent.split() if e.lower() not in stopword)
#     preprocessed_test2.append(sent.lower().strip())

# # after preprocesing
# preprocessed_test2[20000]


# ## Performing one-hot encoding on categorical features

# ## Name

# In[82]:


from sklearn.feature_extraction.text import CountVectorizer


# In[83]:


vectorizer = CountVectorizer(min_df=10)
vectorizer.fit(df_train['name'].values) # fit has to happen only on train data

# we use the fitted CountVectorizer to convert the text to vector
train_name = vectorizer.transform(df_train['name'].values)
test_name = vectorizer.transform(df_test['name'].values)


print("After vectorizations")
print(train_name.shape)
print(test_name.shape)

# print(vectorizer.get_feature_names())
print("="*100)


# In[84]:


vectorizer = CountVectorizer(min_df=10)
vectorizer.fit(df_train['name'].values) # fit has to happen only on train data

# we use the fitted CountVectorizer to convert the text to vector
# submission_name2 = vectorizer.transform(test['name'].values)



# print("After vectorizations")
# print(submission_name2.shape)


# print(vectorizer.get_feature_names())
print("="*100)


# ## BRAND NAME

# In[85]:


vectorizer = CountVectorizer()
vectorizer.fit(df_train['brand_name'].values) # fit has to happen only on train data

# we use the fitted CountVectorizer to convert the text to vector
train_brandname = vectorizer.transform(df_train['brand_name'].values)
test_brandname = vectorizer.transform(df_test['brand_name'].values)


print("After vectorizations")
print(train_brandname.shape)
print(test_brandname.shape)

# print(vectorizer.get_feature_names())
print("="*100)


# In[86]:


vectorizer = CountVectorizer()
vectorizer.fit(df_train['brand_name'].values) # fit has to happen only on train data

# we use the fitted CountVectorizer to convert the text to vector
# submission_brand_name2 = vectorizer.transform(test['brand_name'].values)



# print("After vectorizations")
# print(submission_brand_name2.shape)


# print(vectorizer.get_feature_names())
print("="*100)


# ## General category

# In[87]:


vectorizer = CountVectorizer()
vectorizer.fit(df_train['general_cat'].values) # fit has to happen only on train data

# we use the fitted CountVectorizer to convert the text to vector
train_gen = vectorizer.transform(df_train['general_cat'].values)
test_gen = vectorizer.transform(df_test['general_cat'].values)



print("After vectorizations")
print(train_gen.shape)
print(test_gen.shape)

# print(vectorizer.get_feature_names())
print("="*100)


# In[88]:


vectorizer = CountVectorizer(min_df=10)
vectorizer.fit(df_train['general_cat'].values) # fit has to happen only on train data

# we use the fitted CountVectorizer to convert the text to vector
# submission_general_cat2 = vectorizer.transform(test['general_cat'].values)



# print("After vectorizations")
# print(submission_general_cat2.shape)


# print(vectorizer.get_feature_names())
# print("="*100)


# ## Sub-category 1

# In[89]:


vectorizer = CountVectorizer()
vectorizer.fit(df_train['subcat_1'].values) # fit has to happen only on train data

# we use the fitted CountVectorizer to convert the text to vector
train_subcat1 = vectorizer.transform(df_train['subcat_1'].values)
# we use the fitted CountVectorizer to convert the text to vector
test_subcat1 = vectorizer.transform(df_test['subcat_1'].values)


print("After vectorizations")
print(train_subcat1.shape)
print(test_subcat1.shape)

# print(vectorizer.get_feature_names())
print("="*100)


# In[90]:


vectorizer = CountVectorizer()
vectorizer.fit(df_train['subcat_1'].values) # fit has to happen only on train data

# we use the fitted CountVectorizer to convert the text to vector
# submission_subcat_12 = vectorizer.transform(test['subcat_1'].values)



# print("After vectorizations")
# print(submission_subcat_12.shape)


# print(vectorizer.get_feature_names())
print("="*100)


# ## Sub-category 2

# In[91]:


vectorizer = CountVectorizer()
vectorizer.fit(df_train['subcat_2'].values) # fit has to happen only on train data

# we use the fitted CountVectorizer to convert the text to vector
train_subcat2 = vectorizer.transform(df_train['subcat_2'].values)
test_subcat2 = vectorizer.transform(df_test['subcat_2'].values)


print("After vectorizations")
print(train_subcat2.shape)
print(test_subcat2.shape)

# print(vectorizer.get_feature_names())
print("="*100)


# ## Vectorising text feature 'Item description'

# In[92]:


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=10,max_features=5000)
vectorizer.fit(preprocessed_train_des)
X_train_itemdes = vectorizer.transform(preprocessed_train_des)
X_test_itemdes = vectorizer.transform(preprocessed_test_des)


# In[93]:


print("Shape of train matrix after one hot encodig ",X_train_itemdes.shape)
print("Shape of train matrix after one hot encodig ",X_test_itemdes.shape)


# 

# In[94]:


# from sklearn.feature_extraction.text import TfidfVectorizer
# vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=10,max_features=5000)
# vectorizer.fit(preprocessed_train_des)
# X_submission_itemdes2 = vectorizer.transform(preprocessed_test2)
# print("Shape of train matrix after one hot encodig ",X_submission_itemdes2.shape)


# ## Length of Item description

# In[95]:


X_train_des_wordcount = []
for i in (preprocessed_train_des):
    cnt_words =1
    for j in i:
        if (j==' '):
            cnt_words+=1
    X_train_des_wordcount.append(cnt_words)


# In[96]:


df_train['Number of words in item description']= X_train_des_wordcount


# In[97]:


df_train.head()


# In[98]:


vectorizer = CountVectorizer()
vectorizer.fit(df_train['subcat_2'].values) # fit has to happen only on train data

# we use the fitted CountVectorizer to convert the text to vector
# submission_subcat_22 = vectorizer.transform(test['subcat_2'].values)



# print("After vectorizations")
# print(submission_subcat_22.shape)


# In[99]:


X_test_des_wordcount = []
for i in tqdm(preprocessed_test_des):
    cnt_words =1
    for j in i:
        if (j==' '):
            cnt_words+=1
    X_test_des_wordcount.append(cnt_words)


# In[100]:


df_test['Number of words in item description']= X_test_des_wordcount
len(X_test_des_wordcount)


# In[101]:


df_test.shape


# In[102]:


# X_submission_des_wordcount = []
# for i in tqdm(preprocessed_test2):
#     cnt_words =1
#     for j in i:
#         if (j==' '):
#             cnt_words+=1
#     X_submission_des_wordcount.append(cnt_words)


# In[103]:


# test['Number of words in item description']= X_submission_des_wordcount


# In[104]:


from sklearn.preprocessing import Normalizer
normalizer = Normalizer()


# In[105]:


normalizer.fit(df_train['Number of words in item description'].values.reshape(-1,1))

#X_train_price_norm = normalizer.transform(X_train['price'].values.reshape(-1,1))
X_train_words_des_norm = normalizer.transform(df_train['Number of words in item description'].values.reshape(-1,1))

X_test_words_des_norm = normalizer.transform(df_test['Number of words in item description'].values.reshape(-1,1))

print("After normalizations")
print(X_train_words_des_norm.shape, y_train.shape)

print(X_test_words_des_norm.shape, y_test.shape)


# In[106]:


normalizer = Normalizer()


# In[107]:


normalizer.fit(df_train['Number of words in item description'].values.reshape(-1,1))

#X_train_price_norm = normalizer.transform(X_train['price'].values.reshape(-1,1))
# submission_words_des_norm2 = normalizer.transform(test['Number of words in item description'].values.reshape(-1,1))


# print("After normalizations")
# print(submission_words_des_norm2.shape)


# ## Sentiment score of an item in description

# In[108]:


import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

import nltk
nltk.download('vader_lexicon')

sid = SentimentIntensityAnalyzer()

X_train_sentiment = []; 
for sentence in tqdm(preprocessed_train_des):
    for_sentiment = sentence
    ss = sid.polarity_scores(for_sentiment)
    X_train_sentiment.append(ss)


# In[109]:


negative=[]
neutral=[]
positive=[]

for i in X_train_sentiment:
    
    for polarity,score in i.items():
        if(polarity=='neg'):
            negative.append(score)
        if(polarity=='neu'):
            neutral.append(score)
        if(polarity=='pos'):
            positive.append(score)
        


# In[110]:


df_train['negative']=negative
df_train['neutral']=neutral
df_train['positive']=positive


# In[111]:


import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

import nltk
nltk.download('vader_lexicon')

sid = SentimentIntensityAnalyzer()

X_test_sentiment = []; 
for sentence in tqdm(preprocessed_test_des):
    for_sentiment = sentence
    ss = sid.polarity_scores(for_sentiment)
    X_test_sentiment.append(ss)


# In[112]:


negative=[]
neutral=[]
positive=[]

for i in X_test_sentiment:
    
    for polarity,score in i.items():
        if(polarity=='neg'):
            negative.append(score)
        if(polarity=='neu'):
            neutral.append(score)
        if(polarity=='pos'):
            positive.append(score)
        


# In[113]:


df_test['negative']=negative
df_test['neutral']=neutral
df_test['positive']=positive


# In[114]:


normalizer = Normalizer()

normalizer.fit(df_train['negative'].values.reshape(-1,1))


# In[115]:


X_train_neg_norm = normalizer.transform(df_train['negative'].values.reshape(-1,1))

X_test_neg_norm = normalizer.transform(df_test['negative'].values.reshape(-1,1))

print("After normalizations")
print(X_train_neg_norm.shape, y_train.shape)

print(X_test_neg_norm.shape, y_test.shape)


# In[116]:


normalizer = Normalizer()

normalizer.fit(df_train['neutral'].values.reshape(-1,1))


# In[117]:


X_train_neu_norm = normalizer.transform(df_train['neutral'].values.reshape(-1,1))

X_test_neu_norm = normalizer.transform(df_test['neutral'].values.reshape(-1,1))

print("After normalizations")
print(X_train_neu_norm.shape, y_train.shape)

print(X_test_neu_norm.shape, y_test.shape)


# In[118]:


normalizer = Normalizer()

normalizer.fit(df_train['positive'].values.reshape(-1,1))


# In[119]:


X_train_pos_norm = normalizer.transform(df_train['positive'].values.reshape(-1,1))

X_test_pos_norm = normalizer.transform(df_test['positive'].values.reshape(-1,1))

print("After normalizations")
print(X_train_pos_norm.shape, y_train.shape)

print(X_test_pos_norm.shape, y_test.shape)


# In[120]:


import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

import nltk
nltk.download('vader_lexicon')

sid = SentimentIntensityAnalyzer()

# submission_sentiment = []; 
# for sentence in tqdm(preprocessed_test2):
#     for_sentiment = sentence
#     ss = sid.polarity_scores(for_sentiment)
#     submission_sentiment.append(ss)


# # In[121]:


# negative=[]
# neutral=[]
# positive=[]

# for i in submission_sentiment:
    
#     for polarity,score in i.items():
#         if(polarity=='neg'):
#             negative.append(score)
#         if(polarity=='neu'):
#             neutral.append(score)
#         if(polarity=='pos'):
#             positive.append(score)
        


# # In[122]:


# test['negative']=negative
# test['neutral']=neutral
# test['positive']=positive


# In[123]:


# from sklearn.preprocessing import Normalizer
# normalizer = Normalizer()

# normalizer.fit(df_train['negative'].values.reshape(-1,1))
# sub_neg_norm2 = normalizer.transform(test['negative'].values.reshape(-1,1))
# normalizer.fit(df_train['neutral'].values.reshape(-1,1))
# sub_neu_norm2 = normalizer.transform(test['neutral'].values.reshape(-1,1))
# normalizer.fit(df_train['positive'].values.reshape(-1,1))

# sub_pos_norm2 = normalizer.transform(test['positive'].values.reshape(-1,1))


# ## Merging all features in a matrix

# In[124]:


from scipy.sparse import hstack
X_train = hstack((train_name,train_brandname, train_gen,train_subcat1,train_subcat2,X_train_itemdes,X_train_words_des_norm,X_train_neg_norm,X_train_neu_norm,X_train_pos_norm)).tocsr()


# In[125]:



X_test = hstack((test_name,test_brandname,test_gen,test_subcat1,test_subcat2,X_test_itemdes,X_test_words_des_norm,X_test_neg_norm,X_test_neu_norm,X_test_pos_norm)).tocsr()


# In[126]:


# submission2 = hstack((submission_name2,submission_brand_name2, submission_general_cat2,submission_subcat_12,submission_subcat_22,X_submission_itemdes2,submission_words_des_norm2,sub_neg_norm2,sub_neu_norm2,sub_pos_norm2)).tocsr()


# In[127]:


X_test_itemdes.shape


# ## MODEL PART

# ## LINEAR REGRESSION MODEL

# In[132]:





from sklearn.metrics import mean_squared_error


# In[145]:




# In[146]:




# ## LGBM REGRESSOR

# In[133]:


from sklearn.model_selection import GridSearchCV
# from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


# In[137]:




# lgbm_params={ 'objective': 'regression','boosting_type': 'gbdt','learning_rate': 0.5,'max_depth': 8,'n_estimators': 500,}
# model = LGBMRegressor(**lgbm_params)
# model.fit(X_train, y_train,early_stopping_rounds=100,verbose=True,eval_set = (X_test, y_test))
import pickle
model = pickle.load(open('model.pkl','rb'))


y_pred=model.predict(X_test)
print(y_pred)


