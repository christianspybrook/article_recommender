import operator
import numpy as np
from scipy.sparse import lil_matrix
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
train_paths_file = joblib_path + '/tf_idf_data/all_arxiv_paths.joblib'
# set vocab object paths
articles_path = joblib_path + '/tf_idf_objects/indexed_articles.joblib'
word_map_path = joblib_path + '/tf_idf_objects/word2idx.joblib'
articles_count_path = joblib_path + '/tf_idf_objects/word_idx_articles_count.joblib'
# set tf-idf object paths
tf_path = joblib_path + '/tf_idf_objects/tf.joblib'
idf_path = joblib_path + '/tf_idf_objects/idf.joblib'
tf_idf_path = joblib_path + '/tf_idf_objects/tf_idf.joblib'




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




def get_vocab(filepaths, min_df=None, n_vocab=None):
	"""Returns (D x V) TF-IDF sparse matrix and associated objects"""


	#################### DATA PARSING ####################


	start = time.time()
	print('\nProcessing files...')

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
	# instantiate word index article appearance counting dict
	word_idx_articles_count = {}
	# initialize removed word index to indicate all words are being used
	unknown = None

	# read in list of file paths to articles
	read_paths = load(filepaths)#[:1000]

	# set starting number of files to process for progress display
	files_left_to_process = len(read_paths)

	for read_path in read_paths:

		# display progress
		show_progress(files_left_to_process)

		# tokenize article while filtering out ones that couldn't be processed
		word_lst = list(filter(None, read_file(read_path).split(' ')))
		# ignore empty files (empty strings) and articles over ~250 pages
		if len(word_lst) > 0 and len(word_lst) < 1e5:
			# instantiate word indies list for one article
			indexed_article = []

			# iterate through words in each article
			for token in word_lst:
				# limit tokens to 50 characters
				token = token[:50]
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
			indexed_articles.append(indexed_article)

			# update progress display
			files_left_to_process += -1

	# reassign article counter to be array of count values indexed by word indices
	word_idx_articles_count = np.array(list(word_idx_articles_count.values()), dtype='uint32')

	# capture total number of articles (1st Dim size of TF-IDF)
	D = len(indexed_articles)
	# capture initial vocabulary size (2nd Dim size of TF-IDF)
	V = len(word2idx)

	print(f'# articles: {D:,}')
	print(f'Vocab size: {V:,}')

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

		# get array of words to be kept
		kept_words = np.array(list(word2idx.keys()), dtype='<U50')[keep_words_flag != 0]
		# create new words to word indices mapping
		word2idx_small = {v: k for k, v in enumerate(kept_words)}

		# free memory alloted to retired data
		del kept_words

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

		# capture infrequent word index
		unknown = len(word2idx_small)
		# 'UNKNOWN' will be last token, used to replace all infrequent words in articles
		word2idx_small['UNKNOWN'] = unknown

		# MAP OLD INDICES TO NEW INDICES

		# set starting number of files to process
		files_left_to_process = D

		# iterate through nested list holding word indices lists for all articles
		for article in indexed_articles:

			# display progress
			show_progress(files_left_to_process)

			# replace old word index with new one for words to keep, and
			# replace old word index with dropped word index for infrequent words
			new_article = [idx2new_idx_map[idx] if idx in idx2new_idx_map else unknown 
							for idx in article]
			# revove current old article from beginning of article word indices list
			indexed_articles = indexed_articles[1:]
			# add new article to end of article word indices list
			indexed_articles.append(new_article)

			# update progress display
			files_left_to_process += -1

		if n_vocab is not None:
			# remove words with infrequent article occurance from word index counter
			# map old word indices to new word indice
			word_idx_count = {idx2new_idx_map[item[0]]: item[1] for item, flag in 
							  zip(word_idx_count.items(), keep_words_flag) 
							  if flag == True}

		# free memory alloted to retired data
		del idx2new_idx_map
		del keep_words_flag

		# reassign variable after vocab reduction
		word2idx = word2idx_small
		
		# free memory alloted to retired data
		del word2idx_small

		# capture new vocabulary size (2nd Dim size of TF-IDF)
		V = len(word2idx)

		print(f'Vocab size: {V:,}')

		end = time.time()
		total_time = end - start
		print(f'Low document frequency filtering time: {total_time / 60:.0f} min\n')

		# print([{v: k for k, v in word2idx.items()}[k] for k in indexed_articles[0][:30]], '\n')

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

		# capture new vocabulary size (2nd Dim size of TF-IDF)
		V = len(word2idx_small)

		# MAP OLD INDEX TO NEW INDEX

		# set starting number of files to process
		files_left_to_process = D

		# iterate through nested list holding word indices lists for all articles
		for article in indexed_articles:

			# display progress
			show_progress(files_left_to_process)

			# replace old word index with new one for words to keep, and
			# replace old word index with dropped word index for infrequent words
			new_article = [idx2new_idx_map[idx] if idx in idx2new_idx_map else unknown 
							for idx in article]
			# revove current old article from beginning of article word indices list
			indexed_articles = indexed_articles[1:]
			# add new article to end of article word indices list
			indexed_articles.append(new_article)
			
			# update progress display
			files_left_to_process += -1

		# reassign variables after vocab reduction
		word2idx = word2idx_small

		# free memory alloted to retired data
		del idx2new_idx_map
		del word2idx_small

		print(f'Vocab size: {V:,}')

		end = time.time()
		total_time = end - start
		print(f'Vocab reduction time: {total_time / 60:.0f} min\n')

		# print([{v: k for k, v in word2idx.items()}[k] for k in indexed_articles[0][:30]], '\n')

	#################### UNKNOWN WORDS HOUSEKEEPING ####################

	# include dummy index in article counter if words were removed
	if unknown:
		word_idx_articles_count = np.append(word_idx_articles_count, D)

	# add 'UNKNOWN_TOKEN' token to end of vocab to replace words in query not used in training
	word2idx['UNKNOWN_TOKEN'] = len(word2idx)

	#################### SAVE VOCAB PARSING OBJECTS ####################

	# indexed_articles: word index representation for all articles (nested list of lenth D)
	# word2idx: map of word indices to word strings for all articles (dict of lenth V)
	# word_idx_articles_count: count of article appearances ordered by word index (array of lenth V)

	# save vocab objects
	dump(indexed_articles, articles_path)
	dump(word2idx, word_map_path)
	dump(word_idx_articles_count, articles_count_path)

	print(f'Indexed Article Size: {sys.getsizeof(indexed_articles) / (2**10)**2:.2f} MiB list')
	print(f'Word Map Size: {sys.getsizeof(word2idx) / (2**10)**2:.2f} MiB dictionary')
	print(f'Article Counter Size: {word_idx_articles_count.nbytes / (2**10)**2:.2f} MiB array\n')

	return D, V




###################################################
#################### TF BUILD #####################
###################################################




def get_tf(D, V, loadpath):

	start = time.time()
	print('Building TF array...')

	# load indexed articles
	indexed_articles = load(loadpath)		
	
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

	# tf: Term Frequency (D x V array)
	return tf
	



####################################################
#################### IDF BUILD #####################
####################################################




def get_idf(D, loadpath):

	# get Inverse Document Frequency array (IDF)

	start = time.time()
	print('Building IDF array...')

	# load article appearance counter
	word_idx_articles_count = load(loadpath)		

	# add 1 to Document Frequency to allow for words not in vocab --> no division by 0, and
	# use log of result to keep values from getting too large, due to many articles
	idf = np.log10(np.divide(D, (np.add(word_idx_articles_count, 1, dtype='float32'))))

	# save idf array
	dump(idf, idf_path)

	print('Done.')
	print(f'IDF Dims: {idf.shape}')
	print(f'IDF Size: {idf.nbytes / (2**10)**2:.2f} MiB array')

	end = time.time()
	total_time = end - start
	print(f'IDF construction time: {total_time:.0f} sec\n')

	# idf: Inverse Document Frequency (1D array of length V)
	return idf




#######################################################
#################### TF-IDF BUILD #####################
#######################################################




def get_tf_idf(tf_loadpath, idf_loadpath):

	start = time.time()
	print('Building TF-IDF array...')

	tf = load(tf_loadpath)
	idf = load(idf_loadpath)

	# get Term Frequency-Inverse Document Frequency array (TF-IDF) as CSC Matrix
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

	# tf_idf: Term Frequency-Inverse Document Frequency (D x V array)
	return tf_idf




#########################################################
#################### MAIN EXECUTION #####################
#########################################################




if __name__ == '__main__':

	START = time.time()

	MIN_DF = 2
	N_VOCAB = int(5e6)
	
	# parse text documents and save vocab objects
	D, V = get_vocab(train_paths_file, MIN_DF, N_VOCAB)

	# genrerate and save TF, IDF, and TF-IDF objects
	get_tf(D, V, articles_path)
	get_idf(D, articles_count_path)
	get_tf_idf(tf_path, idf_path)

	END = time.time()
	TOTAL_TIME = (END - START)
	print(f'Total time: {TOTAL_TIME  / 60:.0f} min\n')
