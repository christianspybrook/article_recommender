import os
import operator
from future.utils import iteritems

# import custom process display
from util import show_progress

libraries = ['nltk', 'spacy', 'gensim', 'sklearn']

def get_stopwords(library):
	"""Returns list of stopwords from chosen library"""
	if library == 'nltk':
		from nltk.corpus import stopwords
		s_words = stopwords.words('english')
	elif library == 'spacy':
		import spacy
		en = spacy.load('en_core_web_lg')
		s_words = en.Defaults.stop_words
	elif library == 'gensim':
		from gensim.parsing.preprocessing import STOPWORDS
		s_words = STOPWORDS
	elif library == 'sklearn':
		from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
		s_words = ENGLISH_STOP_WORDS

	return s_words

# retrieve my home directory
local_file = '../home_directory/home_dir.txt'
with open(local_file, 'r') as file:
	my_home = file.read().rstrip('\r\n')

# set repository folder path
repo_path = my_home + '/github/article_recommender'

# select directories to clean
data_directories = {0: '/0704', 
					1: '/2105'
					}

# capture month and year directory to clean
d = 1

# map stages of cleaning to separate folders for analysis
text_folders = {0: '/full_month_with_stopwords', 
				1: '/full_month_cleaned', 
				}

# capture cleaning stage to perform
f = 1

# set text directory path
read_directory = repo_path + '/data' + text_folders[f] + data_directories[d]

def read_file(file):
	"""Returns document read in from given path"""
	read_path = read_directory + '/' + file
	with open(read_path, 'r') as f_in:
		# read in text file
		doc = f_in.read()

		return doc

def get_word_idx_and_count():
	"""Returns mappings of article word indices and word count"""

	# set first article index to 0
	i = 0
	# create article index mapping dict
	word2idx = {}
	# instantiate word index counting dict
	word_idx_count = {}

	# iterate through files from given path
	sorted_files = sorted(os.listdir(read_directory))
	# set starting number of files to process
	files_left_to_process = len(sorted_files)
	for file in sorted_files:
		# display progress
		show_progress(files_left_to_process)
		word_lst = read_file(file).split(' ')
		# iterate through words in each article
		for token in word_lst:
			# check that word has not already been indexed
			if token not in word2idx:
				# assign word to index mapping
				word2idx[token] = i
				# increment index
				i += 1

			# capture current word index
			idx = word2idx[token]
			# increment word index counter by 1 or initialize index count at 1 if not there
			word_idx_count[idx] = word_idx_count.get(idx, 0) + 1
		# update progress display
		files_left_to_process += -1

	return word2idx, word_idx_count

def get_top_words(n_words):
	# Returns top n words from word frequency index counter

	# sort word index counter in descending order for n words
	# tell itemgetter to grab second items (index counts) in tuple
	sorted_word_idx_count = sorted(word_idx_count.items(), 
		key=operator.itemgetter(1), reverse=True)[:n_words]
	# get top n words and their frequencies
	top_n = [(idx2word[i], j) for i, j in sorted_word_idx_count]

	return top_n




if __name__ == '__main__':

	# genrerate word to index map and word index count
	word2idx, word_idx_count = get_word_idx_and_count()

	# create index to word map
	idx2word = dict((v, k) for k, v in iteritems(word2idx))
	# get top n words
	top_word_limit = None
	top_words = get_top_words(top_word_limit)
	# display 200 top words, at most
	print(min(top_words[:200], top_words[:top_word_limit]))

	stopword_sums = {}
	# iterate through stopwords from different libraries
	for lib in libraries:
		# get stopwords from library
		stopwords = get_stopwords(lib)
		# calculate occurnces of stopwords in articles from chosen directory
		sum_common_words = sum([j for i, j in top_words if i in stopwords])
		# map library name to stopword occurances
		stopword_sums[lib] = sum_common_words

	# get vocabulary size
	V = len(word2idx)
	print("Vocab size:", V)
	print(stopword_sums)
