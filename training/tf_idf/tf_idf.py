import operator
import numpy as np
from scipy.sparse import lil_matrix
import sys
import time
from joblib import dump, load
import gc

# import custom process display
from util import show_progress




#######################################################
#################### PATH PARSING #####################
#######################################################




# retrieve my home directory path
local_file = '../home_directory/home_dir.txt'
with open(local_file, 'r') as file:
	my_home = file.read().rstrip('\r\n')

# set RAID 0 array mount path
raid_path = my_home + '/mnt/4T_nvme'
# set joblib directory path
joblib_path = raid_path + '/arxiv_data/joblib_dump'
# set training directory path
train_paths_file = joblib_path + '/tf_idf_data/all_arxiv_paths.joblib'

# set joblib objects path
joblib_objects_path = joblib_path + '/tf_idf_objects'

# set unfiltered vocab object paths
unfiltered_articles_path = joblib_objects_path + '/unfiltered_data/indexed_articles.joblib'
unfiltered_word_map_path = joblib_objects_path + '/unfiltered_data/word2idx.joblib'
unfiltered_words_count_path = joblib_objects_path + '/unfiltered_data/word_idx_count.joblib'
unfiltered_articles_count_path = joblib_objects_path + '/unfiltered_data/word_idx_articles_count.joblib'
unfiltered_dims_path = joblib_objects_path + '/unfiltered_data/dimensions.joblib'

# set filtered vocab object paths
filtered_articles_path = joblib_objects_path + '/filtered_data/indexed_articles.joblib'
filtered_word_map_path = joblib_objects_path + '/filtered_data/word2idx.joblib'
filtered_words_count_path = joblib_objects_path + '/filtered_data/word_idx_count.joblib'
filtered_articles_count_path = joblib_objects_path + '/filtered_data/word_idx_articles_count.joblib'
filtered_dims_path = joblib_objects_path + '/filtered_data/dimensions.joblib'

# set tf-idf object paths
tf_path = joblib_objects_path + '/train_arrays/tf.joblib'
idf_path = joblib_objects_path + '/train_arrays/idf.joblib'
tf_idf_path = joblib_objects_path + '/train_arrays/tf_idf.joblib'




########################################################
#################### I/O FUNCTIONS #####################
########################################################




def read_file(path):
	"""Returns document read in by line"""
	with open(path, 'r') as f_in:
		# read in text file
		doc = f_in.read()

		return doc




######################################################
#################### VOCAB BUILD #####################
######################################################




def get_tf_idf(filepaths, min_df=None, n_vocab=None):
	"""Generates  and saves the following objects:
		- indexed_articles: word index representation for all articles (array of lenth D of lists)
		- word2idx: map of word indices to word strings for all articles (dict of lenth V)
		- word_idx_count: map of word indices to their counts for all articles (dict of lenth V)
		- word_idx_articles_count: count of article appearances ordered by word index (array of lenth V)
		- dimensions: number of articles (D) and vocabulary size (V) in corpus (tuple of size D x V)
		- tf: count of word occurance in corpus (array of size D x V)
		- idf: specificity of words to each document (array of lenth V)
		- tf_idf: collection of document vecotrs defined by words each contains (array of size D x V)
	"""



	#################### DATA PARSING ####################


	start = time.time()
	print('\nProcessing files...')

	# set first word index to 0
	i = 0
	# instantiate word index mapping dict
	word2idx = {}
	# instantiate unique word list for corpus
	idx2word = []
	# instantiate word index counting dict for all articles
	word_idx_count = {}
	# instantiate word index article appearance counting dict
	word_idx_articles_count = {}

	# read in list of file paths to articles
	read_paths = load(filepaths)#[:1000]

	# instantiate corpus array to hold word indices lists for all articles
	indexed_articles = np.zeros((len(read_paths), ), dtype=object)
	# initialize document counter
	doc_idx = 0

	# set starting number of files to process for progress display
	files_left_to_process = len(read_paths)

	for read_path in read_paths:

		# display progress
		show_progress(files_left_to_process)

		# tokenize article while filtering out ones that couldn't be processed
		word_lst = list(filter(None, read_file(read_path).split(' ')))
		# truncate articles over ~50 pages
		word_lst = word_lst[:int(1e4)]
		# append empty files (empty strings) with dummy marker
		if len(word_lst) == 0:
			word_lst = ['EMPTY']

		# instantiate word indies list for one article
		indexed_article = []

		# iterate through words in each article
		for token in word_lst:
			# limit tokens to 20 characters
			token = token[:20]
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
			# build word index counter if words will be remapped for restricted vocab
			if n_vocab is not None:
				# increment word index counter by 1 or initialize count at 1 if not there
				word_idx_count[idx] = word_idx_count.get(idx, 0) + 1

			# build word index article appearance counter to remap words if min docs called
			if idx not in indexed_article:
				# increment word index article counter by 1 or initialize at 1 if needed
				word_idx_articles_count[idx] = word_idx_articles_count.get(idx, 0) + 1

			# add current word index to word indies list for current article
			indexed_article.append(idx)

		# add current article word indies list to outer corpus list
		indexed_articles[doc_idx] = indexed_article

		# update document counter
		doc_idx += 1

		# update progress display
		files_left_to_process += -1

	# reassign article counter to be array of count values indexed by word indices
	word_idx_articles_count = np.array(list(word_idx_articles_count.values()), dtype='uint32')

	# capture total number of articles (1st Dim size of TF-IDF)
	D = len(indexed_articles)
	# capture initial vocabulary size (2nd Dim size of TF-IDF)
	V = len(word2idx)

	print(f'Unfiltered Indexed Article Size: {indexed_articles.nbytes / (2**10)**2:.2f} MiB array')
	print(f'Unfiltered Word Map Size: {sys.getsizeof(word2idx) / (2**10)**3:.2f} GiB dictionary')
	print(f'Unfiltered Word Counter Size: {sys.getsizeof(word_idx_count) / (2**10)**3:.2f} GiB dictionary')
	print(f'Unfiltered Article Counter Size: {word_idx_articles_count.nbytes / (2**10)**2:.2f} MiB array')
	print(f'# articles: {D:,}')
	print(f'Vocab size: {V:,}')

	# add '??' token to end of vocab to replace words in query not used in training
	if min_df is None and n_vocab is None:
		word2idx['??'] = len(word2idx)

	# assign TF-IDF dimensions to tuple
	dimensions = (D, V)

	# save vocab objects
	dump(indexed_articles, unfiltered_articles_path)
	dump(word2idx, unfiltered_word_map_path)
	dump(word_idx_count, unfiltered_words_count_path)
	dump(word_idx_articles_count, unfiltered_articles_count_path)
	dump(dimensions, unfiltered_dims_path)

	# free memory alloted to data to be used later
	del word_idx_count
	gc.collect()

	end = time.time()
	total_time = end - start
	print(f'Text tokenization time: {total_time / 60:.0f} min\n')

	# print([{v: k for k, v in word2idx.items()}[k] for k in indexed_articles[0][:30]], '\n')


	#################### MINIMUM DOC FREQUENCY ####################


	# AMEND INDEXED ARTICLES AND WORD MAPPING FOR MINIMUM DOCUMENT FREQUENCY

	# require word appears in minimum number of documents
	if min_df is not None:

		start = time.time()
		print(f'Removing words appearing in less than {min_df} articles...')

		# set boolean flag for chosen minimum document frequency
		keep_words_flag = word_idx_articles_count > min_df - 1

		# filter words from article appearance counts
		word_idx_articles_count = word_idx_articles_count[keep_words_flag != 0]

		# include dummy index in article counter if words were removed
		if n_vocab is None:
			word_idx_articles_count = np.append(word_idx_articles_count, D)

		# capture new vocab map memory size
		articles_count_mem = word_idx_articles_count.nbytes / (2**10)**2

		# save new article appearance counter
		dump(word_idx_articles_count, filtered_articles_count_path)
		# free memory alloted to data to be used later
		del word_idx_articles_count
		gc.collect()

		# get array of words to be kept
		kept_words = np.array(list(word2idx.keys()), dtype='<U20')[keep_words_flag != 0]
		# create new words to word indices mapping
		word2idx_small = {v: k for k, v in enumerate(kept_words)}

		# free memory alloted to retired data
		del kept_words
		gc.collect()

		# filter removed words from unique word list if limiting vocab to max size
		idx2word = list(word2idx_small.keys())

		# instantiate mapping dict from old word indices to new ones after word reduction
		idx2new_idx_map = {}
		# iterate through smaller set of old word indices
		for word, new_idx in word2idx_small.items():
			# map old word index to new word index to use after word reduction
			idx2new_idx_map[word2idx[word]] = new_idx

		# free memory alloted to retired data
		del word2idx
		gc.collect()

		# capture infrequent word index
		unknown = len(word2idx_small)
		# '?' will be last token, used to replace all infrequent words in articles
		word2idx_small['?'] = unknown

		# capture new vocabulary size (2nd Dim size of TF-IDF)
		V = len(word2idx_small)
		# capture vocab map memory size
		vocab_mem = sys.getsizeof(word2idx_small) / (2**10)**2

		# assign TF-IDF dimensions to tuple
		dimensions = (D, V)
		# save TF-IDF dimensions
		dump(dimensions, filtered_dims_path)

		# add '??' token to end of vocab to replace words in query not used in training
		if n_vocab is None:
			word2idx_small['??'] = len(word2idx_small)

		# save new word mapping
		dump(word2idx_small, filtered_word_map_path)
		# free memory alloted to retired data
		del word2idx_small
		gc.collect()

		# MAP OLD INDICES TO NEW INDICES

		# reload word index counter
		word_idx_count = load(unfiltered_words_count_path)

		if n_vocab is not None:
			# remove words with infrequent article occurance from word index counter
			# map old word indices to new word indice
			word_idx_count = {idx2new_idx_map[item[0]]: item[1] for item, flag in 
							  zip(word_idx_count.items(), keep_words_flag) 
							  if flag == True}

		# capture new word index counter memory size
		word_count_mem = sys.getsizeof(word_idx_count) / (2**10)**2

		# save new word index counter
		dump(word_idx_count, filtered_words_count_path)

		# free memory alloted to data to be used later
		del word_idx_count
		# free memory alloted to retired data
		del keep_words_flag
		gc.collect()

		# set starting number of files to process
		files_left_to_process = D

		# instantiate article counter
		doc_idx = 0

		# iterate through nested list holding word indices lists for all articles
		for article in indexed_articles:

			# display progress
			show_progress(files_left_to_process)

			# replace old word index with new one for words to keep, and
			# replace old word index with dropped word index for infrequent words
			new_article = [idx2new_idx_map[idx] if idx in idx2new_idx_map else unknown 
							for idx in article]
			# replace old article word indices list with new one
			indexed_articles[doc_idx] = new_article

			# update document counter
			doc_idx += 1

			# update progress display
			files_left_to_process += -1

		# free memory alloted to retired data
		del idx2new_idx_map
		gc.collect()

		print(f'Min Doc Indexed Article Size: {indexed_articles.nbytes / (2**10)**2:.2f} MiB array')
		print(f'Min Doc Word Map Size: {vocab_mem:.2f} MiB dictionary')
		print(f'Min Doc Word Counter Size: {word_count_mem:.2f} MiB dictionary')
		print(f'Min Doc Article Counter Size: {articles_count_mem:.2f} MiB array')
		print(f'Vocab size: {V:,}')

		# save amended article word indices list
		dump(indexed_articles, filtered_articles_path)

		end = time.time()
		total_time = end - start
		print(f'Low document frequency filtering time: {total_time / 60:.0f} min\n')

		# print([{v: k for k, v in word2idx_small.items()}[k] for k in indexed_articles[0][:30]], '\n')


	#################### LIMIT VOCAB SIZE ####################


	# AMEND INDEXED ARTICLES AND WORD MAPPING FOR MAXIMUM VOCAB SIZE

	# create upper bound of vocab size for numerical comparison
	vocab_limit = np.inf

	# check if numerical limit was given
	if n_vocab is not None:
		# reduce vocab limit
		vocab_limit = n_vocab

	# check if vocab size has already been reduced enough during min doc frequency step
	if (n_vocab is not None) and (V > vocab_limit):

		start = time.time()
		print(f'Restricting vocabulary to {n_vocab:,} words...')

		# reload word index counter
		if min_df is None:
			word_idx_count = load(unfiltered_words_count_path)
		else:
			word_idx_count = load(filtered_words_count_path)

		# sort word index counter in descending order
		# specify second item, counts, from tuple (index, counts) to use as sorting key
		sorted_word_idx_count = sorted(word_idx_count.items(), 
									   key=operator.itemgetter(1), reverse=True)[:n_vocab]

		# free memory alloted to retired data
		del word_idx_count
		gc.collect()

		# CONSTRUCT NEW MAPPING

		# get frequent word indices
		keep_words_idx = list(dict(sorted_word_idx_count).keys())

		# reload article appearance counter
		if min_df is None:
			word_idx_articles_count = load(unfiltered_articles_count_path)
		else:
			word_idx_articles_count = load(filtered_articles_count_path)

		# remove infrequent words from article appearance counter
		word_idx_articles_count = word_idx_articles_count[keep_words_idx]

		# include dummy index in article counter if words were removed
		word_idx_articles_count = np.append(word_idx_articles_count, D)

		# capture vocab map memory size
		articles_count_mem = word_idx_articles_count.nbytes / (2**10)**2

		# save new article appearance counter
		dump(word_idx_articles_count, filtered_articles_count_path)
		# free memory alloted to retired data
		del word_idx_articles_count
		gc.collect()

		# instantiate limited vocabulary dict for new word to index mapping
		word2idx_small = {}
		# initialize new indices at 0
		new_idx = 0
		# instantiate mapping dict from old word indices to new ones with limited vocabulary
		idx2new_idx_map = {}

		# iterate through smaller set of old word indices
		for idx, count in sorted_word_idx_count:
			# capture word string
			word = idx2word[idx]
			# add word and its new index to mapping dict
			word2idx_small[word] = new_idx
			# map old word index to new word index to use after word reduction
			idx2new_idx_map[idx] = new_idx
			# increment new word index
			new_idx += 1

		# capture new word index counter memory size
		word_count_mem = sys.getsizeof(sorted_word_idx_count) / (2**10)**2

		# free memory alloted to retired data
		del sorted_word_idx_count
		del idx2word
		gc.collect()

		# '?' will be last token, used to replace all infrequents word in articles
		word2idx_small['?'] = new_idx
		# capture infrequent word index
		unknown = new_idx

		# capture new vocabulary size (2nd Dim size of TF-IDF)
		V = len(word2idx_small)
		# capture vocab map memory size
		vocab_mem = sys.getsizeof(word2idx_small) / (2**10)**2

		# assign TF-IDF dimensions to tuple
		dimensions = (D, V)
		# save TF-IDF dimensions
		dump(dimensions, filtered_dims_path)

		# add '??' token to end of vocab to replace words in query not used in training
		word2idx_small['??'] = len(word2idx_small)

		# save new word mapping
		dump(word2idx_small, filtered_word_map_path)
		# free memory alloted to retired data
		del word2idx_small
		gc.collect()

		# MAP OLD INDEX TO NEW INDEX

		# set starting number of files to process
		files_left_to_process = D

		# instantiate article counter
		doc_idx = 0

		# iterate through nested list holding word indices lists for all articles
		for article in indexed_articles:

			# display progress
			show_progress(files_left_to_process)

			# replace old word index with new one for words to keep, and
			# replace old word index with dropped word index for infrequent words
			new_article = [idx2new_idx_map[idx] if idx in idx2new_idx_map else unknown 
							for idx in article]
			# replace old article word indices list with new one
			indexed_articles[doc_idx] = new_article

			# update document counter
			doc_idx += 1

			# update progress display
			files_left_to_process += -1

		# free memory alloted to retired data
		del idx2new_idx_map
		gc.collect()

		print(f'Max Vocab Indexed Article Size: {indexed_articles.nbytes / (2**10)**2:.2f} MiB array')
		print(f'Max Vocab Word Map Size: {vocab_mem:.2f} MiB dictionary')
		print(f'Max Vocab Word Counter Size: {word_count_mem:.2f} MiB dictionary')
		print(f'Max Vocab Article Counter Size: {articles_count_mem:.2f} MiB array')
		print(f'Vocab size: {V:,}')

		# save amended article word indices list
		dump(indexed_articles, filtered_articles_path)

		end = time.time()
		total_time = end - start
		print(f'Vocab reduction time: {total_time / 60:.0f} min\n')

		# print([{v: k for k, v in word2idx_small.items()}[k] for k in indexed_articles[0][:30]], '\n')




	###################################################
	#################### TF BUILD #####################
	###################################################




	start = time.time()
	print('Building TF array...')
	
	# set starting number of files to process
	files_left_to_process = D

	# instantiate D x V  List of List (LIL) sparse matrix (TF)
	tf = lil_matrix((D, V), dtype=np.float32)

	# FILL TERM FREQUENCY MATRIX WITH WORD COUNTS FROM ARTICLES

	# iterate through articles
	for art_idx, article in enumerate(indexed_articles):

		# display progress
		show_progress(files_left_to_process)

		# capture number of words in article
		n_words = len(article)
		# iterate through words in article
		for i in range(n_words):
			# increment matrix value for occurance of word in article by double indexing
			tf[art_idx, article[i]] += 1
		# convert word count to normalized word frequency for article
		tf[art_idx] = np.divide(tf[art_idx], n_words)

		# update progress display
		files_left_to_process += -1

	# free memory alloted to retired data
	del indexed_articles
	gc.collect()

	# Term Frequency (D x V array)
	# convert matrix to Compressed Sparse Column (CSC) matrix
	tf = tf.tocsc()

	# save tf matrix
	dump(tf, tf_path)

	print(f'TF Dims: {tf.shape}')
	# get total size of sparse matrix
	sparse_size = tf.data.nbytes + tf.indices.nbytes + tf.indptr.nbytes
	print(f'TF Matrix Size: {sparse_size / (2**10)**3:.2f} GiB array')

	end = time.time()
	total_time = end - start
	print(f'TF construction time: {total_time / 60:.0f} min\n')
	



	####################################################
	#################### IDF BUILD #####################
	####################################################




	# get Inverse Document Frequency array (IDF)

	start = time.time()
	print('Building IDF array...')

	# load article appearance counter
	if min_df is None and n_vocab is None:
		word_idx_articles_count = load(unfiltered_articles_count_path)
	else:	
		word_idx_articles_count = load(filtered_articles_count_path)		

	# Inverse Document Frequency (1D array of length V)
	# add 1 to Document Frequency to allow for words not in vocab --> no division by 0, and
	# use log of result to keep values from getting too large, due to many articles
	idf = np.log10(np.divide(D, (np.add(word_idx_articles_count, 1, dtype='float32'))))

	# free memory alloted to retired data
	del word_idx_articles_count
	gc.collect()

	# save idf array
	dump(idf, idf_path)

	print('Done.')
	print(f'IDF Dims: {idf.shape}')
	print(f'IDF Size: {idf.nbytes / (2**10)**2:.2f} MiB array')

	end = time.time()
	total_time = end - start
	print(f'IDF construction time: {total_time:.0f} sec\n')




	#######################################################
	#################### TF-IDF BUILD #####################
	#######################################################




	start = time.time()
	print('Building TF-IDF array...')

	# Term Frequency-Inverse Document Frequency (D x V array)
	# get TF-IDF as CSC Matrix
	tf_idf = tf.multiply(idf).tocsc()

	# save tf_idf matrix
	dump(tf_idf, tf_idf_path)

	print('Done.')
	print(f'TF-IDF Dims: {tf_idf.shape}')

	# get total size of sparse matrix
	sparse_size = tf_idf.data.nbytes + tf_idf.indices.nbytes + tf_idf.indptr.nbytes
	print(f'TF-IDF Matrix Size: {sparse_size / (2**10)**3:.2f} GiB array')

	end = time.time()
	total_time = end - start
	print(f'TF-IDF construction time: {total_time:.0f} sec\n')




#########################################################
#################### MAIN EXECUTION #####################
#########################################################




if __name__ == '__main__':

	START = time.time()

	MIN_DF = 2
	N_VOCAB = int(5e6)
	
	# parse text documents, then generate and save vocab and TF-IDF objects
	get_tf_idf(train_paths_file, MIN_DF, N_VOCAB)

	END = time.time()
	TOTAL_TIME = (END - START)
	print(f'Total time: {TOTAL_TIME  / 60:.0f} min\n')
