#import nltk
import re
#import bs4 as bs
import math
import os
#import numpy as np

#from nltk.corpus import stopwords

from orderedset import OrderedSet

from speech import convert_audio_to_text, silence_based_conversion
from mailing import send
from flask import Flask,render_template,url_for,flash,redirect, request

# convert_audio_to_text("gandhi.wav")
# #silence_based_conversion()
# text = ''
# with open('recognized.txt') as source:
# 	for sent in source:
# 		text += sent

app = Flask(__name__)

HOST_NAME = os.environ.get('OPENSHIFT_APP_DNS', 'localhost')
APP_NAME = os.environ.get('OPENSHIFT_APP_NAME', 'flask')
IP = os.environ.get('OPENSHIFT_PYTHON_IP', '127.0.0.1')
PORT = int(os.environ.get('OPENSHIFT_PYTHON_PORT', 8080))
HOME_DIR = os.environ.get('OPENSHIFT_HOMEDIR', os.getcwd())

#print(text)

def clean(text):

	text = text.lower() # convert all words to lower case, 

	print(text)

	repeat_eliminate = list(text.split())
	
	for i in range (1,len(repeat_eliminate)):
		if(repeat_eliminate[i] == repeat_eliminate[i-1]):
			repeat_eliminate[i-1] = ''
	
	for i in repeat_eliminate:
		if i == '':
			repeat_eliminate.remove(i)

	text = ' '.join(repeat_eliminate)

	print(text)
	
	text = re.sub(r'\s+', ' ', text) # replace or substitute all spaces, tabs, indents (\s) with ' ' (space) 

	text = re.sub(r'\d', ' ', text) # replace all digits by ' '

	text = re.sub(r'[^a-zA-Z. ]+', '', text) # replace all non words (\W) with ' '. (note: small w is for all words. capital W is for all non-words)
		
	return text

	#print(text)

def get_sentence_of_words(text):

	sentence = list() # list of sentences
	words = list() # list of words in each sentence.

	sentence_list = list()

	temp = text.strip().split(".") # temporary list of sentences. 

	for sent in temp:

		words = sent.strip().split(" ") # getting the words in sentences.

		if(len(words) > 0): 
			sentence.append(words) # sentence is a list of lists. (contains a list of sentences in which each sentence is a list of words)

		sentence_list.append(sent)	

	# print(sentence)
	if('' in sentence_list):
		sentence_list.remove('')
		
	return sentence, sentence_list


def vectorize(sentence):

	# set of unique words in the whole document.
	unique_words = OrderedSet() 

	for sent in sentence:
		for word in sent:
			
			unique_words.add(word)

	unique_words = list(unique_words) # converting the set to a list to make it easier to work with it. 

	# a list of lists that contains the vectorized form of each sentence in the document. 
	vector = list()


	# in the vectorized representation, we consider the bag of words (unique words in the text).
	# then, we count the occurence of each word in a sentence and represent it in a vector whose length = length(unique_words)
	# ex: sent1 = "i am a boy"
	#     sent2 = "i am a girl"
	# unique_words = ["i", "am", "a", "boy", "girl"]
	# vector representation of sent1 = [1, 1, 1, 1, 0]
	# vector representation of sent2 = [1, 1, 1, 0, 1]

	for sent in sentence: # iterate for every sentence in the document
		temp_vector = [0] * len(unique_words) # create a temporary vector to calculate the occurence of each word in that sentence. 
		
		for word in sent: # iterate for every word in the sentence. 

			temp_vector[unique_words.index(word)] += 1	

		vector.append(temp_vector) # add the temporary vector to the list of vectors for each sentence (list of lists)

	# print(vector)	


	return vector, unique_words	

# function to calculate the tf scores
def tf(vector, sentence, unique_words):

	termf = list()

	no_of_unique_words = len(unique_words) 

	for i in range(len(sentence)):

		tflist = list()
		sent = sentence[i]
		count = vector[i]

		for word in sent:

			score = float(count[sent.index(word)])/ float(len(sent)) # tf = no. of occurence of a word/ total no. of words in the sentence. 

			tflist.append(score)  

		termf.append(tflist)

	# print(tf)	
	
	return termf	


#function to calculate idf. 
def idf(vector, sentence, unique_words):

	# idf = log(no. of sentences / no. of sentences in which the word appears).

	no_of_sentences = len(sentence)

	indef = list()

	for sent in sentence:
		
		idflist = list()

		for word in sent:

			count = 0 # no. of times the word occurs in the entire text.

			for k in sentence:
				if(word in k):
					count += 1
		

			score = float(math.log(no_of_sentences/float(count))) # caclulating idf scores

			idflist.append(score)

		indef.append(idflist)	

	#print(idf)	

	return indef


# function to calculate the tf-idf scores.
def tf_idf(tf, idf):

	# tf-idf = tf(w) * idf(w)

	tfidf = [[0.1 for j in range(len(tf[i]))] for i in range(len(tf))]

	for i in range(len(tf)):
		for j in range(len(tf[i])):

			tfidf[i][j] = tf[i][j] * float(idf[i][j])

	#print(tfidf)		

	return tfidf

# function to calculate the scores of each sentence.
def sentence_score(tfidf, sentence):

	# score of a sentence = sum of scores of every word in the sentence.

	score = list()

	for i in range(len(sentence)):

		s = 0
		sent = sentence[i]

		for j in range(len(sent)):

			s += tfidf[i][j]
		
		score.append(s)	

	return score	


#function to sort the sentences based on their scores.
def sort_sentences(score, sentence, text):

	# a mapping from score to sentence. i.e, key = scores, value = sentences.
	score_to_sentence = {i:k for i, k in zip(score, sentence)}

	# a mapping from sentence to index.
	sentence_to_index = {i:k for k, i in enumerate(sentence)}

	text = re.sub(r'\s+', ' ', text)
	idx_to_sent = text.split('.')

	# a mapping from index to original sentence in the unprocessed text.
	# this is needed to geet back the original sentence for summary.
	index_to_sentence = {i:k for i, k in enumerate(idx_to_sent)}

	# sort the sentences based on their scores.
	sorted_score_to_sentence = sorted(score_to_sentence, reverse = True)

	#print(sentence_to_index, sorted_score_to_sentence)

	return sorted_score_to_sentence, sentence_to_index, score_to_sentence, index_to_sentence


def get_summary(sorted_score_to_sentence, sentence_to_index, score_to_sentence, index_to_sentence, n):

	# print(sorted_score_to_sentence, sentence_to_index, score_to_sentence)

	summary = ""

	indices = []
	#print(sorted_score_to_sentence, len(sorted_score_to_sentence))
	for i in range(n):

		s = sorted_score_to_sentence[i]

		sentence = score_to_sentence[s]

		# print(sentence, end = '. ')

		ind = sentence_to_index[sentence]

		indices.append(ind)

	indices.sort()

	for i in indices:

		summary += '* ' + index_to_sentence[i] + '.\n'
		print('* ', index_to_sentence[i])

	return summary	
		



@app.route("/")
def home():
	#internal_tab()
	return render_template('Katti.html')

@app.route("/summarizer",methods=['POST','GET'])
def commence():
	
	audio_path = "gandhi.wav"
	if(request.method == 'POST'):
		audio_path = request.form['FileName']
	
	#convert_audio_to_text(audio_path)
	#print(audio_path)				##Here we can put the name of the file or link to the audio file
	silence_based_conversion(audio_path)
	text = ''
	with open('recognized.txt') as source:
		for sent in source:
			text += sent

	t_clean = clean(text)	

	processed_text, sentence_list = get_sentence_of_words(t_clean)
	'''
	from sklearn.feature_extraction.text import CountVectorizer
	from sklearn.feature_extraction.text import TfidfTransformer

	count_vect = CountVectorizer(stop_words = 'english')
	count_vect = count_vect.fit(sentence_list)
	freq_term_matrix = count_vect.transform(sentence_list)

	transformer = TfidfTransformer()
	transformed_weights = transformer.fit_transform(freq_term_matrix)

	print(transformed_weights)


	# print(freq_term_matrix)
	'''
	#sw = stopwords.words('english')

	sentence_to_index = {i:k for k, i in enumerate(sentence_list)}

	# print(sentence_to_index)

	#compression = int(input('enter the percentage of compression (without "%" symbol)'))
	# set the compression percentage. can be customized.
	compression = 35

	vector, unique_words = vectorize(processed_text)

	termf = tf(vector, processed_text, unique_words)

	indf = idf(vector, processed_text, unique_words)

	tfidf = tf_idf(termf, indf)

	scores = sentence_score(tfidf, processed_text)

	sorted_score_to_sentence, sentence_to_index, score_to_sentence, index_to_sentence = sort_sentences(scores, sentence_list, text)

	length_of_summary = math.ceil(len(sorted_score_to_sentence) * compression/100)

	print('number of sentences =', length_of_summary)

	summary = get_summary(sorted_score_to_sentence, sentence_to_index, score_to_sentence, index_to_sentence, length_of_summary)
	f = open('summary.txt', 'w+')
	f.write(summary)
	f.close()

	send()
	#exit(0)
	##Here we should redirect to a pic of us three
	return render_template('home.html')



if __name__ == '__main__':
	app.run(host='0.0.0.0', port=PORT)






