import re
import string
import os
from gensim.parsing.preprocessing import STOPWORDS

# retrieve my home directory
local_file = '../home_directory/home_dir.txt'
with open(local_file, 'r') as file:
	my_home = file.read().rstrip('\r\n')

# set RAID 0 array mount path
raid_path = my_home + '/mnt/4T_nvme'
# set repository folder path
repo_path = my_home + '/github/article_recommender'

# select directories to clean
data_directories = {0: '/0704', 
					1: '/2105'
					}

# capture month and year directory to clean
d = 1

# map stages of cleaning to separate folders for analysis
text_folders = {0: '/sample_data' + data_directories[d] + '/sample_100_raw_text', 
				1: '/sample_data' + data_directories[d] + '/sample_100_no_hexadecimal', 
				2: '/sample_data' + data_directories[d] + '/sample_100_with_short_lines', 
				3: '/sample_data' + data_directories[d] + '/sample_100_with_new_lines', 
				4: '/sample_data' + data_directories[d] + '/sample_100_with_stopwords', 
				5: '/data' + data_directories[d] + '/full_month_with_stopwords', 
				6: '/sample_data' + data_directories[d] + '/sample_100_cleaned', 
				7: '/data' + data_directories[d] + '/full_month_cleaned'
				}

# capture cleaning stage to perform
f = 7

# set text directory paths
read_directory = raid_path + '/arxiv_data/raw_text_latest' + data_directories[d]
write_directory = repo_path + text_folders[f]

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

def read_file(file):
	"""Returns document read in by line"""
	read_path = read_directory + '/' + file
	with open(read_path, 'r') as f_in:
		# read in text file
		doc = f_in.readlines()

		return doc

def write_new_file(file, new_doc):
	"""Writes new file to specified path"""
	write_path = write_directory + '/' + file
	with open(write_path, 'w') as f_out:
		# write new text string to new text file
		for new_line in new_doc:
			f_out.write(new_line)

def remove_hex(string):
	"""Returns text string free of hexadecimal coding"""
	for key in keys:
		regex = re.compile(key)
		match_object = regex.findall(string)
		if len(match_object) != 0:
			string = string.replace(key, ligatures[key])
	string = re.sub(pattern, '', string)

	return string

def clean_file(file):
	"""Reads in a file, cleans its text, and writes a new file"""
	# get next file
	doc = read_file(file)
	# instantiate text file to return
	new_doc = []
	# iterate through text string
	for line in doc:
		# remove hexadecimal codes and restore English words
		new_string = remove_hex(line)
		# remove all characters except alpha-numeric and whitespace
		new_string = re.sub(r'[^A-Za-z0-9\s]+', '', new_string)
		# downcase text
		new_string = new_string.lower()
		# eliminate short lines created by parsing equations, figures, tables, and page numbers
		if len(new_string) > 3:
			# remove whitespace and new line character
			new_string = re.sub('\s+', ' ', new_string)
			# remove stopwords
			new_string = ' '.join([t for t in new_string.split() if t not in STOPWORDS])
			# add filtered string to new string list
			new_doc.append(new_string)
	# save new cleaned file
	write_new_file(file, new_doc)

def show_progress(num_files):
	"""Prints files left to process"""
	if num_files % 100 == 0:
		print(num_files, end='')
		print('\r', end='')




if __name__ == '__main__':

	# set limit to number of files to clean
	n_files = None
	files_to_clean = sorted(os.listdir(read_directory))[:n_files]
	files_left_to_clean = len(files_to_clean)

	# iterate through and clean first 100 files from given folder
	for file in files_to_clean:
		show_progress(files_left_to_clean)
		clean_file(file)
		files_left_to_clean += -1
