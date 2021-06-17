import os
import operator
from future.utils import iteritems
from itertools import chain
from collections import defaultdict
import numpy as np
import sys
import time

# import custom process display
from util import show_progress

#######################################################
#################### PATH PARSING #####################
#######################################################

# retrieve my home directory
local_file = '../home_directory/home_dir.txt'
with open(local_file, 'r') as file:
	my_home = file.read().rstrip('\r\n')

# set repository folder path
repo_path = my_home + '/github/article_recommender'

# select directories to use
data_directories = {0: '/0704', 
					1: '/2105'
					}

# capture month and year directory to use
d = 0

# set text directory path
read_directory = repo_path + '/data' + '/full_month_cleaned' + data_directories[d]

########################################################
#################### I/O FUNCTIONS #####################
########################################################

def read_file(file):
	"""Returns document read in from given path"""
	read_path = read_directory + '/' + file
	with open(read_path, 'r') as f_in:
		# read in text file
		doc = f_in.read()

		return doc

#######################################################
#################### TF-IDF BUILD #####################
#######################################################

def get_tf_idf_matrix(n_vocab=None):
	"""Returns TF-IDF matrix --> (D x V)"""

#################### DATA PARSING ####################

	start = time.time()
	print('Processing files...')

	# instantiate outer corpus list to hold word indices lists for all articles
	indexed_articles = []
	# set first word index to 0
	i = 0
	# instantiate word index mapping dict
	word2idx = {}
	# instantiate unique word list for corpus
	idx2word = []
	# instantiate word index counting dict for all articles
	word_idx_count = {}
	# instantiate article word index appearance counting dict for corpus
	# this will become Inverse Document Frequency (IDF) array
	word_idx_articles_count = {}

	# iterate through files from given path
	sorted_files = sorted(os.listdir(read_directory))
	# capture total number of articles (1st Dim size of TF-IDF)
	D = len(sorted_files)
	# set starting number of files to process for progress display
	files_left_to_process = len(sorted_files)

	for file in sorted_files:

		# display progress
		show_progress(files_left_to_process, 2)

		# tokenize article
		word_lst = read_file(file).split(' ')
		# instantiate word indies list for one article
		indexed_article = []
		# instantiate article word index appearance counting dict for one article
		word_idx_article_count = {}
		# iterate through words in each article
		for token in word_lst:
			# check that word has not already been assigned an index
			if token not in word2idx:
				# build unique word list if words will be remapped for restricted vocab
				if n_vocab is not None:
					# add word to unique word list
					idx2word.append(token)
				# assign word to index mapping
				word2idx[token] = i
				# increment index
				i += 1

			# capture current word index
			idx = word2idx[token]
			# increment word index counter by 1 or initialize index count at 1 if not there
			word_idx_count[idx] = word_idx_count.get(idx, 0) + 1
			# check that word has not already been seen in article
			if token not in word_idx_article_count:
				# update article word index appearance counter for current article
				word_idx_article_count[idx] = 1

			# add current word index to word indies list for current article
			indexed_article.append(idx)

		# add current article word indies list to outer corpus list
		indexed_articles.append(indexed_article)
		# instantiate default dict
		default_dict = defaultdict(int)
		# increment article word index appearance counter for words found in current article
		for k, v in chain(word_idx_articles_count.items(), word_idx_article_count.items()):
			default_dict[k] += v
			# update article appearance counter
			word_idx_articles_count = dict(default_dict)

		# update progress display
		files_left_to_process += -1


	end = time.time()
	# total_time = (end - start) / 3600
	# print(f'File processing time: {total_time:.0f} hours.')
	total_time = end - start
	print(f'File processing time: {total_time:.0f} seconds.')

#################### LIMIT VOCAB SIZE ####################

	# restrict vocab size
	if n_vocab is not None:

		start = time.time()
		print(f'Restricting vocabulary to {n_vocab:,} words...')

		# sort word index counter in descending order
		# specify second item, counts, from tuple (index, counts) to use as sorting key
		sorted_word_idx_count = sorted(word_idx_count.items(), 
									   key=operator.itemgetter(1), reverse=True)
		# free memory alloted to retired data
		del word_idx_count

		# CONSTRUCT NEW MAPPING

		# instantiate limited vocabulary dict for new word to index mapping
		word2idx_small = {}
		# initialize new indices at 0
		new_idx = 0
		# instantiate mapping dict from old word indices to new ones with limited vocabulary
		idx_new_idx_map = {}
		# iterate through smaller set of old word indices
		for idx, count in sorted_word_idx_count[:n_vocab]:
			# capture word string
			word = idx2word[idx]
			# add word and its new index to mapping dict
			word2idx_small[word] = new_idx
			# map old word index to new word index to use after word reduction
			idx_new_idx_map[idx] = new_idx
			# increment new word index
			new_idx += 1
		# free memory alloted to retired data
		del sorted_word_idx_count
		del idx2word

		# 'UNKNOWN' will be last token, used to replace all infrequents word in articles
		word2idx_small['UNKNOWN'] = new_idx
		# capture infrequent word index
		unknown = new_idx

		# MAP OLD IDX TO NEW IDX

		# drop old word indices from article word index appearance counter for all articles
		word_idx_articles_count = {idx_new_idx_map[idx]: count 
								   for idx, count in word_idx_articles_count.items() 
								   if idx in idx_new_idx_map}
		# add infrequent word alias article occuance count to preserve dimensionality
		word_idx_articles_count[unknown] = len(sorted_files)

		# instantiate new article word indices list
		articles_small = []
		# iterate through nested list holding word indices lists for all articles
		for article in indexed_articles:
			# check that article has at least two words in it from corpus
			if len(article) > 1:
				# replace old word index with new one for words to keep
				# replace old word index with dropped word index for infrequent words
				new_article = [idx_new_idx_map[idx] if idx in idx_new_idx_map else unknown 
								for idx in article]
				# add new article to article word indices list
				articles_small.append(new_article)

		# reassign variables after vocab reduction
		indexed_articles, word2idx = articles_small, word2idx_small

		# free memory alloted to retired data
		del idx_new_idx_map
		del articles_small		
		del word2idx_small

		print('Done.')	
		end = time.time()
		total_time = end - start
		print(f'Vocab reduction time: {total_time:.0f} seconds.')

#################### IDF BUILD ####################

	start = time.time()
	print('Building IDF array...')

	# capture vocabulary size (2nd Dim size of TF-IDF)
	V = len(word2idx)

	# convert article word index appearance counting dict for corpus to numpy array (DF)
	word_idx_articles_count = np.array(list(dict(word_idx_articles_count).values()))
	# get Inverse Document Frequency array (IDF)
	# add 1 to Document Frequency to allow for words not in vocab --> no division by 0
	# use log of result to keep values from getting too large, due to many articles
	word_idx_articles_count = np.log(np.divide(D, (np.add(word_idx_articles_count, 1))))

	print('Done.')
	end = time.time()
	total_time = end - start
	print(f'IDF construction time: {total_time:.2f} seconds.')

#################### TF BUILD ####################

	start = time.time()
	print('Building TF array...')

	# set starting number of files to process
	files_left_to_process = D

	# instantiate D x V matrix (TF)
	tf = np.zeros((D, V))

	# fill Term Frequency matrix with word counts from articles
	# iterate through articles
	for art_idx, article in enumerate(indexed_articles):

		# display progress
		show_progress(files_left_to_process, 2)

		# capture number of words in article
		n_words = len(article)
		# iterate through words in article
		for i in range(len(article)):
			# increment matrix value for occurance of word in article by double indexing
			tf[art_idx, article[i]] += 1
		# convert word count to normalized word frequency for article
		tf[art_idx] = np.divide(tf[art_idx], n_words)

		# update progress display
		files_left_to_process += -1

	end = time.time()
	total_time = end - start
	print(f'TF construction time: {total_time:.2f} seconds.')

#################### COMBINE TF AND IDF ####################

	start = time.time()
	print('Building TF-IDF array...')

	# get Term Frequency-Inverse Document Frequency array (TF-IDF)
	tf_idf = np.multiply(tf, word_idx_articles_count)

	end = time.time()
	total_time = end - start
	print(f'TF-IDF construction time: {total_time:.2f} seconds.')

	# Indices of returning tuple...
	# [0] --> indexed_articles: word indicies lists of all articles (list of lenght-D)
	# [1] --> word2idx: map of word indices to word strings for all articles (dict of lenth-V)
	# [2] --> tf_idf: Term Frequency-Inverse Document Frequency (D x V array)
	return indexed_articles, word2idx, tf_idf


#########################################################
#################### MAIN EXECUTION #####################
#########################################################


if __name__ == '__main__':

	START = time.time()
	N_VOCAB = None

	# genrerate numerical articles, word map, and IF-IDF
	articles, word_map, matrix = get_tf_idf_matrix(N_VOCAB)

	print(articles[0][:20])
	print(list(word_map.items())[:10])
	print(matrix[:5, :5])

	print('# articles: ', len(articles))
	print('# words: ', len(word_map))
	print('matrix dim: ', matrix.shape)

	print('articles KB: ', sys.getsizeof(articles) / 1e3)
	print('word_map MB: ', sys.getsizeof(word_map) / 1e6)
	print('matrix MB: ', matrix.nbytes / 1e6)

	END = time.time()
	# TOTAL_TIME = (END - START) / 3600
	# print(f'Total time: {TOTAL_TIME:.0f} hours.')
	TOTAL_TIME = END - START
	print(f'Total time: {TOTAL_TIME:.0f} seconds.')
