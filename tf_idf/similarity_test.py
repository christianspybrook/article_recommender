import numpy as np
import time
from joblib import dump, load
import sys

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

# set test files list path
test_files_path = joblib_path + '/tf_idf_data/test_data_paths.joblib'
# set word to index mapping object path
word_map_path = joblib_path + '/tf_idf_objects/word2idx.joblib'
# set IDF object path
idf_path = joblib_path + '/tf_idf_objects/idf.joblib'
# set TF-IDF object path
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




#########################################################
#################### VECTORIZE DOCS #####################
#########################################################




def vectorize_test_data(paths, word2idx):
	"""Returns vectorized test documents"""

	start = time.time()
	print('Building vectorized test array...')

	# capture dimensionality of test corpus
	D = len(paths)
	files_left_to_process = D
	docs = []

	for path in paths:

		# display progress
		show_progress(files_left_to_process, 10)

		# tokenize article while remmoving empty strings
		doc = list(filter(None, read_file(path).split(' ')))

		# make sure empty files get ignored
		if len(doc) > 0:
			# last index is 1 lower than its length
			unknown_idx = len(word2idx) - 1
			indexed_doc = [word2idx.get(w, unknown_idx) for w in doc]

			docs.append(indexed_doc)

			files_left_to_process -= 1

	end = time.time()
	total_time = end - start
	print(f'Test array construction time: {total_time:.0f} sec\n')

	return docs




###################################################
#################### TF BUILD #####################
###################################################


def get_test_tf(indexed_articles, word2idx):
	start = time.time()
	print('Building TF array...')

	# capture total number of articles (1st Dim size of TF-IDF)
	D = len(indexed_articles)
	# capture vocabulary size (2nd Dim size of TF-IDF)
	V = len(word2idx)

	# set starting number of files to process
	files_left_to_process = D

	# instantiate D x V matrix (TF)
	tf = np.zeros((D, V), dtype='float16')

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

	# remove final two columns from array:
	# - words dropped from training corpus --> 'UNKNOWN'
	# - words in test corpus, but not in training vocab --> 'UNKNOWN_TOKEN'
	tf = tf[:, :-2]
	print(f'TF Dims: {tf.shape}')

	end = time.time()
	total_time = end - start
	print(f'TF construction time: {total_time:.0f} sec\n')

	return tf


#######################################################
#################### TF-IDF BUILD #####################
#######################################################




def get_test_tf_idf(tf, idf):

	start = time.time()
	print('Building TF-IDF array...')

	# get Term Frequency-Inverse Document Frequency array (TF-IDF)
	tf_idf = np.multiply(tf, idf)

	print('Done.')
	end = time.time()
	total_time = end - start
	print(f'TF-IDF construction time: {total_time:.0f} sec\n')

	return tf_idf




############################################################
#################### FIND SIMILARITIES #####################
############################################################




def get_similarities(train_matrix, test_matrix):

	start = time.time()
	print('Building similarities array...')

	n_sim_rows = train_matrix.shape[0]
	n_sim_cols = test_matrix.shape[0]
	num_sims = n_sim_rows * n_sim_cols

	sims = np.zeros((n_sim_rows, n_sim_cols), dtype='float32')

	# calculate L2 norm of each article
	train_norms = np.apply_along_axis(np.linalg.norm, 1, train_matrix)
	test_norms = np.apply_along_axis(np.linalg.norm, 1, test_matrix)

	for train_row_idx, train_row in enumerate(train_matrix):
		for test_row_idx, test_row in enumerate(test_matrix):
			# display progress
			show_progress(num_sims, int(1e3))

			sims[train_row_idx, test_row_idx] = np.dot(train_row, test_row) / (
									train_norms[train_row_idx] * test_norms[test_row_idx])
			num_sims -= 1

	end = time.time()
	total_time = end - start
	print(f'Similarity construction time: {total_time / 60:.0f} min\n')

	return sims




#########################################################
#################### MAIN EXECUTION #####################
#########################################################




if __name__ == '__main__':

	START = time.time()

	print('Loading test file paths...')
	# load test file paths
	test_files = load(test_files_path)
	print('Done.\n')

	# load word to index mapping
	word_map = load(word_map_path)
	# 'UNKNOWN_TOKEN' will be last token, used to replace all word not in training vocab
	word_map['UNKNOWN_TOKEN'] = len(word_map)
	
	# vectorize test articles
	test_data = vectorize_test_data(test_files, word_map)
	# get TF of test data
	test_tf = get_test_tf(test_data, word_map)

	# load training IDF and TF-IDF
	train_idf = load(idf_path)	
	train_tf_idf = load(tf_idf_path)

	# get TF-IDF of test data
	test_tf_idf = get_test_tf_idf(test_tf, train_idf)

	# calculate sorted similarities
	similarities = get_similarities(train_tf_idf, test_tf_idf)

	dump(word_map, joblib_path + '/tf_idf_objects/word2idx.joblib')
	dump(test_data, joblib_path + '/tf_idf_objects/indexed_test_articles.joblib')
	dump(test_tf_idf, joblib_path + '/tf_idf_objects/test_tf_idf.joblib')
	dump(similarities, joblib_path + '/tf_idf_objects/similarities.joblib')

	print('# test articles: ', len(test_data))
	print(f'{sys.getsizeof(test_data) / (2**10)**2:.2f} MiB indexed array\n')

	print('# words: ', len(word_map))
	print(f'{sys.getsizeof(word_map) / (2**10)**2:.2f} MiB word map\n')

	print('Test TF-IDF Dim: ', test_tf_idf.shape)
	print(f'{test_tf_idf.nbytes / (2**10)**3:.2f} GiB array\n')

	print('Similarities Dim: ', similarities.shape)
	print(f'{similarities.nbytes / (2**10)**3:.2f} GiB array\n')

	print(similarities[: 20, : 20])

	END = time.time()
	TOTAL_TIME = (END - START) / 60
	print(f'Total time: {TOTAL_TIME:.0f} min\n')

	print([{v: k for k, v in word_map.items()}[k] for k in test_data[0][:40]])
