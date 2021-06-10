import re
import string
import os

# retrieve my home directory
local_file = '../home_directory/home_dir.txt'
with open(local_file, 'r') as file:
    my_home = file.read().rstrip('\r\n')

# choose directories to clean
data_directories = ['0704', '2105']

# choose stages of cleaning to analyze
text_folders = ['sample_100_raw_text', 
				'sample_100_no_hexadecimal', 
				'sample_100_with_short_lines', 
				'sample_100_with_new_lines', 
				'sample_100_with_stopwords', 
				'sample_100_cleaned']

# capture month and year directory to clean
d = 1
# capture cleaning stage to perform
f = 4

# set text directory paths
read_directory = my_home + '/mnt/4T_nvme/arxiv_data/raw_text_latest/' + data_directories[d]
write_directory = my_home + '/github/article_recommender/sample_data/' + data_directories[d] + '/' + text_folders[f]

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
	"""Returns read in document"""
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
			# add filtered string to new string list
			new_doc.append(new_string)
	# save new cleaned file
	write_new_file(file, new_doc)


if __name__ == '__main__':

	# iterate through and clean first 100 files from May 2021
	for file in sorted(os.listdir(read_directory))[:100]:
		clean_file(file)


