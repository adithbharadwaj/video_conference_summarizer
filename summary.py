import nltk
import re
import bs4 as bs
import math
import numpy as np

#from nltk.corpus import stopwords

from orderedset import OrderedSet

from speech import convert_audio_to_text, silence_based_conversion

'''
text = """  Mohandas Karamchand Gandhi was an Indian activist who was the leader of the 
        Indian independence movement against British rule. Employing nonviolent civil 
        disobedience, Gandhi led India to independence and inspired movements for civil 
        rights and freedom across the world. In India, he is also called 
        Bapu and Gandhi ji. He is unofficially called the Father of the Nation. Born and 
        raised in a Hindu merchant caste family in coastal Gujarat, India, and trained
        in law at the Inner Temple, London, Gandhi first employed nonviolent civil 
        disobedience as an expatriate lawyer in South Africa, in the resident Indian 
        community's struggle for civil rights. After his return to India in 1915, he 
        set about organising peasants, farmers, and urban labourers to protest against 
        excessive land-tax and discrimination. Assuming leadership of the Indian 
        National Congress in 1921, Gandhi led nationwide campaigns for various social 
        causes and for achieving Swaraj or self-rule. Gandhi famously led Indians in 
        challenging the British-imposed salt tax with the 400 km Dandi Salt 
        March in 1930, and later in calling for the British to Quit India in 1942. He 
        was imprisoned for many years, upon many occasions, in both South Africa and 
        India. He lived modestly in a self-sufficient residential community and wore 
        the traditional Indian dhoti and shawl, woven with yarn hand-spun on a charkha. 
        He ate simple vegetarian food, and also undertook long fasts as a means of 
        both self-purification and political protest. Gandhi's vision of an independent 
        India based on religious pluralism, however, was challenged in the early 1940s 
        by a new Muslim nationalism which was demanding a separate Muslim homeland 
        carved out of India. Eventually, in August 1947, Britain granted independence, 
        but the British Indian Empire was partitioned into two dominions, a 
        Hindu-majority India and Muslim-majority Pakistan. As many displaced Hindus, 
        Muslims, and Sikhs made their way to their new lands, religious violence broke 
        out, especially in the Punjab and Bengal. Eschewing the official celebration of 
        independence in Delhi, Gandhi visited the affected areas, attempting to provide 
        solace. In the months following, he undertook several fasts unto death to stop 
        religious violence. The last of these, undertaken on 12 January 1948 when he was 
        78, also had the indirect goal of pressuring India to pay out some cash 
        assets owed to Pakistan. Some Indians thought Gandhi was too accommodating. 
        Among them was Nathuram Godse, a Hindu nationalist, who assassinated Gandhi on 
        30 January 1948 by firing three bullets into his chest. Captured along with 
        many of his co-conspirators and collaborators, Godse and his co-conspirator 
        Narayan Apte were tried, convicted and executed while many of their other 
        accomplices were given prison sentences. Gandhi's birthday, 2 October, is 
        commemorated in India as Gandhi Jayanti, a national holiday, and worldwide as 
        the International Day of Nonviolence. sleep. While all mammals sleep, whales
         cannot afford to become unconscious for long because they may drown. The only way they 
         can sleep is by remaining partially conscious. It is believed that only one hemisphere of
          the whale’s brain sleeps at a time, so they rest but are never completely asleep. 
          They can do so most probably near the surface so that they can come up for air easily. 

           """
'''
#convert_audio_to_text()
silence_based_conversion()
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

		if(len(words) > 1): 
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

	for i in range(n):

		s = sorted_score_to_sentence[i]

		sentence = score_to_sentence[s]

		# print(sentence, end = '. ')

		ind = sentence_to_index[sentence]

		indices.append(ind)

	indices.sort()

	for i in indices:

		summary += index_to_sentence[i] + '.'

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

	compression = int(input('enter the percentage of compression (without "%" symbol)'))

	vector, unique_words = vectorize(processed_text)

	tf = tf(vector, processed_text, unique_words)

	idf = idf(vector, processed_text, unique_words)

	tfidf = tf_idf(tf, idf)

	scores = sentence_score(tfidf, processed_text)

	sorted_score_to_sentence, sentence_to_index, score_to_sentence, index_to_sentence = sort_sentences(scores, sentence_list, text)

	length_of_summary = math.ceil(len(sentence_list) * compression/100)

	print('number of sentences =', length_of_summary)

	print(get_summary(sorted_score_to_sentence, sentence_to_index, score_to_sentence, index_to_sentence, length_of_summary))











