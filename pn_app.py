#import flask
from flask import Flask,render_template
import PyPDF2 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from wordcloud import WordCloud,STOPWORDS
import request
import tablib
import os 
import plotly
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import json
import pickle
import re
import nltk
from nltk.corpus import stopwords
#from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from collections import Counter
from IPython.display import Image
# from werkzeug.utils import secure_filename
from sklearn.feature_extraction.text import TfidfVectorizer,TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import plotly.figure_factory as ff


app = Flask(__name__)
with open('file.bin', 'rb') as file:
	legal_dict = pickle.load(file)
print(file)

legal_dict={k.lower(): v for k,v in legal_dict.items()}
pdf_og = open("CitizenshipAct1955.pdf",'rb')
pdf = open("CitizenshipAct.pdf",'rb')



@app.route('/', methods = ['POST','GET'])
def index():
	global org_text,wordclouds,wordcloud1
	org_text=wordclouds=wordcloud1=0
	if request == 'POST':
		text = request.files['seq']
		text1 =request.files['seq1']
		text.save(secure_filename(f.filename))
		wordclouds = compare_wordcloud(Counter(tokens_org),Counter(tokens_amd))
	return render_template('basic_nik.html', wordclouds=wordclouds, wordcloud1=wordcloud1)

#comparing wordcloud


def compare_wordcloud(dict1,dict2):
	fig=plt.figure(figsize=(20,15)) 
	wc= WordCloud(background_color="white",
			  width=700,height=400,
			  stopwords=stopwords).generate_from_frequencies(dict1)
	wc1= WordCloud(background_color="white",
			   width=700,height=400,
			  stopwords=stopwords).generate_from_frequencies(dict2)
	l=[wc,wc1]
	for i in range(len(l)):
		ax = fig.add_subplot(1,2,i+1)
		plt.imshow(l[i])
		plt.axis("off")
	
	plt.savefig("..static/images/wrdcld.jpg")
	img = "wrdcld.jpg"
	pngfile = Image(filename=img)
	return "Comparing wordclouds"


#function to extract text from pdf

def get_text(file):
	read = PyPDF2.PdfFileReader(file) 
	n=read.numPages 
	text=""
	for i in range(n):
		pageObj = read.getPage(i)
		text += pageObj.extractText()
		text=text.lower()
	return text



#function to replace the legal term in dictionary with their respective meaning


def multiwordReplace(text, wordDic):
	rc = re.compile('|'.join(map(re.escape, wordDic)))
	def translate(match):
		return wordDic[match.group(0)]
	return rc.sub(translate, text)

stopwords=set(stopwords.words('english'))


#text preprocessing


def clean_text(data):    
	wnl = WordNetLemmatizer()
	text = re.sub('[^a-zA-Z]', ' ', data)
	text = text.lower()
	text = text.split()
	text = [wnl.lemmatize(word,pos='v') for word in text if not word in stopwords]
	text = ' '.join(text)
	return text

##replacing legal terms

org = multiwordReplace(get_text(pdf_og),legal_dict).lower()
amd =multiwordReplace(get_text(pdf),legal_dict).lower()

# original documents

lorg = get_text(pdf_og).lower()
lamd =get_text(pdf).lower()



tokens_org =[i for i in word_tokenize(clean_text(org)) if not i in stopwords]
tokens_amd =[j for j in word_tokenize(clean_text(amd)) if not j in stopwords]

org_legal=[k for k in word_tokenize(clean_text(get_text(pdf_og)))]
amd_legal=[k for k in word_tokenize(clean_text(get_text(pdf)))]

legal_terms=list(legal_dict.keys())
org_rel_legal=[x for x in org_legal if x in legal_terms]
amd_rel_legal=[x for x in amd_legal if x in legal_terms]

cv=CountVectorizer()
word_count_vector=cv.fit_transform([clean_text(org),clean_text(amd)])
tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
tfidf_transformer.fit(word_count_vector)
df_idf = pd.DataFrame(tfidf_transformer.idf_, index=cv.get_feature_names(),columns=["idf_weights"])
count_vector=cv.transform([clean_text(org),clean_text(amd)])
 

tf_idf_vector=tfidf_transformer.transform(count_vector)
feature_names = cv.get_feature_names()
 
#get tfidf vector for first document
first_document_vector=tf_idf_vector[0]
 

dfo = pd.DataFrame(first_document_vector.T.todense(), feature_names, columns=["tfidf"])
dfo = dfo.sort_values(by=["tfidf"],ascending=False)#.head()
dfo['word'] = dfo.index

#get tfidf vector for second document
second_document_vector=tf_idf_vector[1]
 

dfa = pd.DataFrame(second_document_vector.T.todense(), feature_names, columns=["tfidf"])
dfa = dfa.sort_values(by=["tfidf"],ascending=False)#.head()
dfa['word'] = dfa.index

c1=Counter(tokens_org)
c2=Counter(tokens_amd)

c1_keys = list(c1.keys())
c2_keys = list(c2.keys())
union=[]
for i in range(len(c1_keys)):
	union.append(c1_keys[i])
for i in range(len(c2_keys)):
	union.append(c2_keys[i])    
	
union=list((set(union)))

for word in union:
	if word not in c1_keys:
		c1[word]=0
	if word not in c2_keys:
		c2[word]=0



def compare_dist_plots():
	global c1
	x1 = list(c1.values())
	group_labels = ['distplot']

	fig1 = [ff.create_distplot([x1],group_labels)]
	graph = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)
	print(graph)				
	return render_template('distall.html',graph1=graph)

def relevant_wordcloud(dict1,dict2):
	fig = plt.figure(figsize=(8,8))
	wc= WordCloud(background_color="white",
			  width=400,height=400,
			  stopwords=stopwords).generate_from_frequencies(dict1)
	wc1= WordCloud(background_color="white",
			   width=400,height=400,
			  stopwords=stopwords).generate_from_frequencies(dict2)
	l=[wc,wc1]
	for i in range(len(l)):
		ax = fig.add_subplot(1,2,i+1)
		plt.imshow(l[i])
		plt.axis("off")
		plt.savefig("..static/images/wrdcld2.jpg")
		img = "wrdcld2.jpg"
		pngfile = Image(filename=img)
	return "Comparing wordclouds"


def create_plot():
	global dfo,dfa
	df1=dfo[0:100]

	data =[
			go.Bar(
		x=df1.word,
	 y=df1.tfidf 
	 )
	]
	graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)

	df2=dfa[0:100]


	data1 = [go.Bar(
		x=df2.word, y=df2.tfidf)
	]
	graphJSON1 = json.dumps(data1, cls=plotly.utils.PlotlyJSONEncoder)

	return render_template('tfidf_plotly.html',graphJSON=graphJSON,graphJSON1=graphJSON1)


@app.route('/uploader')
def home():
	return render_template('home.html'	)#,home = home)
@app.route('/pdf1')
def content():
	return render_template('content.html', text=org)
@app.route('/pdf2')
def content1():
	return render_template('content1.html', text=amd)
@app.route('/wrdcld')
def wrdcldshow():
	return render_template('wrdcld.html', wordclouds="Comparing wordclouds")
@app.route('/tfidf')
def tfidf_plot():
	bar = create_plot()
	return  bar#, plot=bar)
@app.route('/dist_all')
def distPlot_all():
	dist=compare_dist_plots()
	return dist

@app.route('/legal_dict')
def legal_dictionary():
	legal_dictionary = legal_dict
	return render_template('search.html',legal_dictionary=legal_dictionary)

		 

if __name__ == '__main__':
	app.run(debug=True)