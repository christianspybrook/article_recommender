import operator
import numpy as np
import sys
import time
from joblib import dump, load

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
train_paths_file = joblib_path + '/tf_idf_data/train_data_paths.joblib'




########################################################
#################### I/O FUNCTIONS #####################
########################################################




def read_file(path):
	"""Returns document read in by line"""
	with open(path, 'r') as f_in:
		# read in text file
		doc = f_in.read()

		return doc




#######################################################
#################### TF-IDF BUILD #####################
#######################################################




def get_tf_idf_matrix(filepaths, n_vocab=None, min_df=None):
	"""Returns (D x V) TF-IDF matrix and associated objects"""


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

	# read in list of file paths to articles
	read_paths = load(filepaths)

	# set starting number of files to process for progress display
	files_left_to_process = len(read_paths)

	for read_path in read_paths:

		# display progress
		show_progress(files_left_to_process)

		# tokenize article while remmoving empty strings
		word_lst = list(filter(None, read_file(read_path).split(' ')))
		# make sure empty files get ignored
		if len(word_lst) > 0:
			# instantiate word indies list for one article
			indexed_article = []

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
				# build word index counter if words will be remapped for restricted vocab
				if n_vocab is not None:
					# increment word index counter by 1 or initialize index count at 1 if not there
					word_idx_count[idx] = word_idx_count.get(idx, 0) + 1

				# add current word index to word indies list for current article
				indexed_article.append(idx)

			# add current article word indies list to outer corpus list
			indexed_articles.append(indexed_article)

			# update progress display
			files_left_to_process += -1

	end = time.time()
	total_time = end - start
	print(f'File processing time: {total_time:.0f} sec\n')

	print([{v: k for k, v in word2idx.items()}[k] for k in indexed_articles[0][:40]], '\n')


#################### TF BUILD ####################


	start = time.time()
	print('Building TF array...')

	# capture total number of articles (1st Dim size of TF-IDF)
	D = len(indexed_articles)
	# capture vocabulary size (2nd Dim size of TF-IDF)
	V = len(word2idx)

	# set starting number of files to process
	files_left_to_process = D

	# instantiate D x V matrix (TF)
	tf = np.zeros((D, V), dtype='float32')

	# FILL Term Frequency MATRIX WITH WORD COUNTS FROM ARTICLES

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

	# get article appearance counts for word indices
	word_idx_articles_count = np.count_nonzero(tf, axis=0).astype(np.int32)

	print(f'TF Dims: {tf.shape}')

	end = time.time()
	total_time = end - start
	print(f'TF construction time: {total_time / 60:.0f} min\n')


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
		# filter words from TF matrix
		tf = tf[:, keep_words_flag != 0]

		# get array of words to be kept
		kept_words = np.array(list(word2idx.keys()))[keep_words_flag != 0]
		# create new words to word indices mapping
		word2idx_small = {v: k for k, v in enumerate(kept_words)}
		# filter removed words from unique word list
		idx2word = list(word2idx_small.keys())

		# free memory alloted to retired data
		del kept_words

		# instantiate mapping dict from old word indices to new ones after word reduction
		idx2new_idx_map = {}
		# iterate through smaller set of old word indices
		for word, new_idx in word2idx_small.items():
			# map old word index to new word index to use after word reduction
			idx2new_idx_map[word2idx[word]] = new_idx

		# free memory alloted to retired data
		del word2idx

		# 'UNKNOWN' will be last token, used to replace all infrequents word in articles
		word2idx_small['UNKNOWN'] = new_idx + 1
		# capture infrequent word index
		unknown = new_idx + 1

		# MAP OLD INDICES TO NEW INDICES

		# instantiate new article word indices list
		articles_small = []
		# iterate through nested list holding word indices lists for all articles
		for article in indexed_articles:
			# replace old word index with new one for words to keep, and
			# replace old word index with dropped word index for infrequent words
			new_article = [idx2new_idx_map[idx] if idx in idx2new_idx_map else unknown 
							for idx in article]
			# add new article to article word indices list
			articles_small.append(new_article)

		if n_vocab is not None:
			# remove words with infrequent article occurance from word index counter
			# map old word indices to new word indice
			word_idx_count = {idx2new_idx_map[item[0]]: item[1] for item, flag in 
							  zip(word_idx_count.items(), keep_words_flag) 
							  if flag == True}

		# free memory alloted to retired data
		del idx2new_idx_map
		del keep_words_flag

		# reassign variables after vocab reduction
		indexed_articles, word2idx = articles_small, word2idx_small
		
		# free memory alloted to retired data
		del articles_small
		del word2idx_small


		print('Done.')
		print(f'TF Dims: {tf.shape}')

		end = time.time()
		total_time = end - start
		print(f'Low document frequency filtering time: {total_time / 60:.0f} min\n')

		print([{v: k for k, v in word2idx.items()}[k] for k in indexed_articles[0][:40]], '\n')


#################### LIMIT VOCAB SIZE ####################


	# create upper bound of vocab size for numerical comparison
	vocab_limit = np.inf

	# check if numerical limit was given
	if n_vocab is not None:
		# reduce vocab limit
		vocab_limit = n_vocab

	# check if vocab size has already been reduced enough during min doc frequency step
	if (n_vocab is not None) & (len(word2idx) > vocab_limit):

		start = time.time()
		print(f'Vocabulary size: {len(word2idx):,} words...')
		print(f'Restricting vocabulary to {n_vocab:,} words...')

		# sort word index counter in descending order
		# specify second item, counts, from tuple (index, counts) to use as sorting key
		sorted_word_idx_count = sorted(word_idx_count.items(), 
									   key=operator.itemgetter(1), reverse=True)

		# free memory alloted to retired data
		del word_idx_count

		# CONSTRUCT NEW MAPPING

		# get frequent word indices
		keep_words_idx = list(dict(sorted_word_idx_count[:n_vocab]).keys())

		# remove infrequent words from corpus arrays
		word_idx_articles_count = word_idx_articles_count[keep_words_idx]
		tf = tf[:, keep_words_idx]

		# instantiate limited vocabulary dict for new word to index mapping
		word2idx_small = {}
		# initialize new indices at 0
		new_idx = 0
		# instantiate mapping dict from old word indices to new ones with limited vocabulary
		idx2new_idx_map = {}

		# iterate through smaller set of old word indices
		for idx, count in sorted_word_idx_count[:n_vocab]:
			# capture word string
			word = idx2word[idx]
			# add word and its new index to mapping dict
			word2idx_small[word] = new_idx
			# map old word index to new word index to use after word reduction
			idx2new_idx_map[idx] = new_idx
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

		# instantiate new article word indices list
		articles_small = []
		# iterate through nested list holding word indices lists for all articles
		for article in indexed_articles:
			# replace old word index with new one for words to keep, and
			# replace old word index with dropped word index for infrequent words
			new_article = [idx2new_idx_map[idx] if idx in idx2new_idx_map else unknown 
							for idx in article]
			# add new article to article word indices list
			articles_small.append(new_article)

		# reassign variables after vocab reduction
		indexed_articles, word2idx = articles_small, word2idx_small

		# free memory alloted to retired data
		del idx2new_idx_map
		del articles_small		
		del word2idx_small

		print('Done.')	
		print(f'TF Dims: {tf.shape}')

		end = time.time()
		total_time = end - start
		print(f'Vocab reduction time: {total_time:.0f} sec\n')

		print([{v: k for k, v in word2idx.items()}[k] for k in indexed_articles[0][:30]], '\n')


#################### IDF BUILD ####################


	# get Inverse Document Frequency array (IDF)

	start = time.time()
	print('Building IDF array...')

	# add 1 to Document Frequency to allow for words not in vocab --> no division by 0, and
	# use log of result to keep values from getting too large, due to many articles
	idf = np.log10(np.divide(D, (np.add(word_idx_articles_count, 1, dtype='float32'))))

	# free memory alloted to retired data
	del word_idx_articles_count

	print('Done.')
	end = time.time()
	total_time = end - start
	print(f'IDF construction time: {total_time:.0f} sec\n')


#################### TF-IDF BUILD ####################


	start = time.time()
	print('Building TF-IDF array...')

	# get Term Frequency-Inverse Document Frequency array (TF-IDF)
	tf_idf = np.multiply(tf, idf)

	print('Done.')
	end = time.time()
	total_time = end - start
	print(f'TF-IDF construction time: {total_time:.0f} sec\n')

	# Indices of returning tuple...
	# [0] --> indexed_articles: word indicies lists of all articles (list of lenght D)
	# [1] --> word2idx: map of word indices to word strings for all articles (dict of lenth V)
	# [2] --> idf: Inverse Document Frequency (1D array of length V)
	# [3] --> tf_idf: Term Frequency-Inverse Document Frequency (D x V array)
	return indexed_articles, word2idx, idf, tf_idf




#########################################################
#################### MAIN EXECUTION #####################
#########################################################




if __name__ == '__main__':

	START = time.time()

	N_VOCAB = int(1e6)
	MIN_DF = 2

	# genrerate indexed articles, word map, IDF, and TF-IDF
	articles, word_map, array, matrix = get_tf_idf_matrix(train_paths_file, N_VOCAB, MIN_DF)
	# print(matrix.dtype)
	# train_norms = np.apply_along_axis(np.linalg.norm, 1, matrix)
	# print(len(train_norms))
	# print((train_norms == 0).sum(0))
	# print(train_norms.dtype)
	# save TF-IDF objects
	dump(articles, joblib_path + '/tf_idf_objects/indexed_articles.joblib')
	dump(word_map, joblib_path + '/tf_idf_objects/word2idx.joblib')
	dump(array, joblib_path + '/tf_idf_objects/idf.joblib')
	dump(matrix, joblib_path + '/tf_idf_objects/tf_idf.joblib')

	print('# articles: ', len(articles))
	print(f'{sys.getsizeof(articles) / (2**10)**2:.2f} MiB indexed array\n')

	print('# words: ', len(word_map))
	print(f'{sys.getsizeof(word_map) / (2**10)**2:.2f} MiB word map\n')

	print('IDF Dim: ', array.shape)
	print(f'{sys.getsizeof(articles) / (2**10)**2:.2f} MiB array\n')

	print('TF-IDF Dim: ', matrix.shape)
	print(f'{matrix.nbytes / (2**10)**3:.2f} GiB array\n')

	END = time.time()
	TOTAL_TIME = (END - START) / 60
	print(f'Total time: {TOTAL_TIME:.0f} min\n')
