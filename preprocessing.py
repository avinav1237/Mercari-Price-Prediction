import numpy as np
import string
import re
import pandas as pd
from nltk.corpus import stopwords
from scipy.sparse import hstack
import pickle
import scipy.sparse as sps
from scipy.sparse import csr_matrix
# from sklearn im
def fill_the_null(data):
	data.category_name.fillna(value = "Others", inplace = True)
	# data.brand_name.fillna(value = "Not known", inplace = True)
	# data.item_description.fillna(value = "No description given", inplace = True)
	return data
def cat_split(row):
	try:
		text = row
		text1, text2, text3 = text.split('/')
		return text1, text2, text3
	except:
		return ("Label not given", "Label not given", "Label not given")

def decontracted(phrase):
	# specific
	phrase = re.sub(r"won't", "will not", phrase)
	phrase = re.sub(r"can\'t", "can not", phrase)
	# general
	phrase = re.sub(r"n\'t"," not", phrase)
	phrase = re.sub(r"\'re"," are", phrase)
	phrase = re.sub(r"\'s"," is", phrase)
	phrase = re.sub(r"\'d"," would", phrase)
	phrase = re.sub(r"\'ll"," will", phrase)
	phrase = re.sub(r"\'t"," not", phrase)
	phrase = re.sub(r"\'ve"," have", phrase)
	phrase = re.sub(r"\'m"," am", phrase)
	return phrase
def preprocess(train):
	train=fill_the_null(train)
	train['general_cat'], train['subcat_1'], train['subcat_2'] = zip(*train['category_name'].apply(lambda x: cat_split(x)))
	import nltk
	nltk.download('stopwords')
	stopword=set(stopwords.words("english"))
	from tqdm import tqdm
	preprocessed_total_train = []
	for sentance in tqdm(train['item_description'].values):
		sent = decontracted(sentance)
		sent = sent.replace('\\r', ' ')
		sent = sent.replace('\\"', ' ')
		sent = sent.replace('\\n', ' ')
		sent = re.sub('[^A-Za-z0-9]+', ' ', sent)
		# https://gist.github.com/sebleier/554280
		sent = ' '.join(e for e in sent.split() if e.lower() not in stopword)
		preprocessed_total_train.append(sent.lower().strip())
	X_train_des_wordcount = []
	for i in (preprocessed_total_train):
		cnt_words =1
		for j in i:
			if (j==' '):
				cnt_words+=1
		X_train_des_wordcount.append(cnt_words)
	train['Number of words in item description']= X_train_des_wordcount
	from sklearn.feature_extraction.text import TfidfTransformer
	from sklearn.feature_extraction.text import TfidfVectorizer
	vectorizer = TfidfVectorizer(stop_words='english')
	import nltk
	from nltk.sentiment.vader import SentimentIntensityAnalyzer
	import nltk
	nltk.download('vader_lexicon')
	sid = SentimentIntensityAnalyzer()
	train_sentiment = []; 
	for sentence in tqdm(preprocessed_total_train):
		for_sentiment = sentence
		ss = sid.polarity_scores(for_sentiment)
		train_sentiment.append(ss)
	negative=[]
	neutral=[]
	positive=[]
	for i in train_sentiment:
		for polarity,score in i.items():
			if(polarity=='neg'):
				negative.append(score)
			if(polarity=='neu'):
				neutral.append(score)
			if(polarity=='pos'):
				positive.append(score)
	train['negative']=negative
	train['neutral']=neutral
	train['positive']=positive
	model1=pickle.load(open("vector1.pickel", "rb"))
	xn=model1.transform(train['name'].values)
	model2=pickle.load(open("vector2.pickel","rb"))
	xb=model2.transform(train['brand_name'].values)
	model3=pickle.load(open("vector3.pickel","rb"))
	xgn=model3.transform(train['general_cat'].values)
	model4=pickle.load(open("vector4.pickel","rb"))
	xsub=model4.transform(train['subcat_1'].values)
	model5=pickle.load(open("vector5.pickel","rb"))
	xsub1=model5.transform(train['subcat_2'].values)
	model6=pickle.load(open("vector6.pickel","rb"))
	xpr=model6.transform(preprocessed_total_train)
	model8=pickle.load(open("vector8.pickel","rb"))
	xtr=model8.transform(train['Number of words in item description'].values.reshape(-1,1))
	model9=pickle.load(open("vector9.pickel","rb"))
	xnn=model9.transform(train['negative'].values.reshape(-1,1))
	model10=pickle.load(open("vector10.pickel","rb"))
	xnr=model10.transform(train['neutral'].values.reshape(-1,1))
	model11=pickle.load(open("vector11.pickel","rb"))
	xpr=model11.transform(train['positive'].values.reshape(-1,1))
	data=hstack((xn,xb,xgn,xsub,xsub1,xpr,xtr,xnn,xnr,xpr)).tocsr()
	B = csr_matrix(np.zeros([1,4999]), dtype = int)
	C = csr_matrix(np.hstack((data.toarray(), B.toarray())))
	data=C
	return data







