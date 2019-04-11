#import nltk
import re
#import bs4 as bs
import math
#import numpy as np

#from nltk.corpus import stopwords

from orderedset import OrderedSet

from speech import convert_audio_to_text, silence_based_conversion
from mailing import send

convert_audio_to_text()
#silence_based_conversion()
text = ''
with open('recognized.txt') as source:
	for sent in source:
		text += sent

print(text)

def clean(text):

	text = text.lower() # convert all words to lower case, 
	
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

	tf = list()

	no_of_unique_words = len(unique_words) 

	for i in range(len(sentence)):

		tflist = list()
		sent = sentence[i]
		count = vector[i]

		for word in sent:

			score = count[sent.index(word)]/ float(len(sent)) # tf = no. of occurence of a word/ total no. of words in the sentence. 

			tflist.append(score)  

		tf.append(tflist)

	# print(tf)	
	
	return tf	


#function to calculate idf. 
def idf(vector, sentence, unique_words):

	# idf = log(no. of sentences / no. of sentences in which the word appears).

	no_of_sentences = len(sentence)

	idf = list()

	for sent in sentence:
		
		idflist = list()

		for word in sent:

			count = 0 # no. of times the word occurs in the entire text.

			for k in sentence:
				if(word in k):
					count += 1
		

			score = math.log(no_of_sentences/float(count)) # caclulating idf scores

			idflist.append(score)

		idf.append(idflist)	

	#print(idf)	

	return idf


# function to calculate the tf-idf scores.
def tf_idf(tf, idf):

	# tf-idf = tf(w) * idf(w)

	tfidf = [[0 for j in range(len(tf[i]))] for i in range(len(tf))]

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
	score_to_sentence = {i:k for i, k in zip(scores, sentence)}

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
		



if __name__ == '__main__':


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
	compression = 20

	vector, unique_words = vectorize(processed_text)

	tf = tf(vector, processed_text, unique_words)

	idf = idf(vector, processed_text, unique_words)

	tfidf = tf_idf(tf, idf)

	scores = sentence_score(tfidf, processed_text)

	sorted_score_to_sentence, sentence_to_index, score_to_sentence, index_to_sentence = sort_sentences(scores, sentence_list, text)

	length_of_summary = math.ceil(len(sorted_score_to_sentence) * compression/100)

	print('number of sentences =', length_of_summary)

	summary = get_summary(sorted_score_to_sentence, sentence_to_index, score_to_sentence, index_to_sentence, length_of_summary)
	f = open('summary.txt', 'w+')
	f.write(summary)
	f.close()

	send()










