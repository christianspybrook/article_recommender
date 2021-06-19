import re
import string
import os
from gensim.parsing.preprocessing import remove_stopwords
from joblib import Parallel, delayed, dump, load

# import custom process display
from util import show_progress




#######################################################
#################### PATH PARSING #####################
#######################################################




# retrieve my home directory
local_file = '../home_directory/home_dir.txt'
with open(local_file, 'r') as file:
	my_home = file.read().rstrip('\r\n')

# set RAID 0 array mount path
raid_path = my_home + '/mnt/4T_nvme'
# set repository folder path
repo_path = my_home + '/github/article_recommender'

# set text directory path
read_directory = raid_path + '/arxiv_data/raw_text_latest' # astro-ph_latest
write_directory = raid_path + '/arxiv_data/clean_text_latest' # clean_astro-ph_latest




########################################################
#################### I/O FUNCTIONS #####################
########################################################




def read_file(path):
	"""Returns document read in by line"""
	with open(path, 'r') as f_in:
		# read in text file
		doc = f_in.readlines()

		return doc

def write_file(path, new_doc):
	"""Writes new file to specified path"""
	with open(path, 'w') as f_out:
		# write new text string to new text file
		for new_line in new_doc:
			f_out.write(new_line)




########################################################
#################### TEXT CLEANING #####################
########################################################




# define ligature mapping to restore words
ligatures = {'': 'ff', 
			 '': 'fi', 
			 '': 'fl', 
			 '': 'ffi'}
# get hexadecimal codes for ligatures
keys = list(ligatures.keys())

# set Regex pattern for additional hexadecimal codes not being replaced
hex_string = ''
pattern = '[' + hex_string + ']'

def remove_hex(string):
	"""Returns text string free of hexadecimal coding"""
	for key in keys:
		regex = re.compile(key)
		match_object = regex.findall(string)
		if len(match_object) != 0:
			string = string.replace(key, ligatures[key])
	string = re.sub(pattern, '', string)

	return string

def keep_alphanumeric(string):
	"""Returns string with only alphanumeric and whitespace type characters"""
	string = re.sub(r'[^A-Za-z0-9\s]+', '', string)

	return string

def downcase(string):
	"""Returns string with lowercase characters"""
	string = string.lower()

	return string

def no_stopwords(string):
	"""Returns string without stopwords"""
	string = remove_stopwords(string)

	return string

def no_short_lines(string):
	"""Returns final cleaned string"""
	if len(string) > 3:
		# remove whitespace and new line character
		string = re.sub('\s+', ' ', string)
		# remove stopwords
		string = no_stopwords(string)

	return string

def clean_file(path_pair):
	"""Reads in a file, cleans its text, and writes a new file"""

	# split file path tuple
	read_path, write_path = path_pair
	# get next file
	doc = read_file(read_path)
	# instantiate text file to return
	new_doc = []
	# iterate through text string
	for line in doc:
		# remove hexadecimal codes and restore English words
		new_string = remove_hex(line)
		# remove all characters except alphanumeric and whitespace
		new_string = keep_alphanumeric(new_string)
		# downcase text
		new_string = downcase(new_string)
		# eliminate short lines created by parsing equations, figures, tables, and page numbers
		if len(new_string) > 3:
			# remove whitespace and new line character
			new_string = no_short_lines(new_string)
			# remove stopwords and add whitespace for end of each line
			new_string = remove_stopwords(new_string) + ' '
			# add filtered string to new string list
			new_doc.append(new_string)
	# save new cleaned file
	write_file(write_path, new_doc)




##########################################################
#################### PARALLELIZATION #####################
##########################################################




def make_chunks(paths, chunksize):
	"""Returns path pairs broken into chunks"""
	chunks = (paths[idx: idx + chunksize] 
			  for idx in range(0, len(paths), chunksize))

	return chunks

def parallel_cleaner(paths, chunksize):
	"""Runs parallel processed cleaned text"""

	# instantiate parallel helper
	executor = Parallel(n_jobs=-1, backend='multiprocessing', prefer="processes")
	# create jobs to distribute execution of test cleaner
	jobs = delayed(clean_file)
	# create task chain
	task_chain = (jobs(chunk[0]) for chunk in make_chunks(paths, chunksize=chunksize))
	# execute parallel jobs
	executor(task_chain)




#########################################################
#################### MAIN EXECUTION #####################
#########################################################




if __name__ == '__main__':

	# instantiate directories list
	paths = []

	# iterate through directories
	for dir in sorted(os.listdir(read_directory)):
		# display cirectory being cleaned
		print(dir)

		# iterate through files from given path
		files_to_clean = sorted(os.listdir(os.path.join(read_directory, dir)))
		# set starting number of files to clean for progress display
		files_left_to_clean = len(files_to_clean)

		for file in files_to_clean:
			# display progress
			show_progress(files_left_to_clean)

			# set file paths to nested folders
			read_path = read_directory + '/' + dir + '/' + file
			write_path = write_directory + '/' + dir + '/' + file
			path_pair = read_path, write_path

			# add both file paths to directories list
			paths.append(path_pair)

			# update progress display
			files_left_to_clean += -1

	# execute parallel text cleaning jobs
	parallel_cleaner(paths, chunksize=1000)
