import chardet
import os

# retrieve my home directory
local_file = '../home_directory/home_dir.txt'
with open(local_file, 'r') as file:
	my_home = file.read().rstrip('\r\n')

# choose directories to clean
data_directories = ['0704', '2105']

# choose stages of cleaning to analyze
text_folder = ['sample_100_raw_text']

# capture month and year directory to clean
d = 0
# capture cleaning stage to perform
f = 0

# set text directory path
read_directory = my_home + '/github/article_recommender/data/sample_data/' + data_directories[d] + '/' + text_folder[f]

def read_file(file):
	"""Returns document read in from given path"""
	read_path = read_directory + '/' + file
	with open(read_path, 'rb') as f_in:
		# read in text file
		doc = f_in.read()

		return doc

def get_encoding(file):
	"""Reads in a file and gets its encoding"""
	# get next file
	doc = read_file(file)
	encoding_dict = chardet.detect(doc)

	return encoding_dict


if __name__ == '__main__':

	# instantiate dict to map file to its encoding
	file_map = {}
	encoding_map = {}
	confidence_map = {}
	# iterate through and get encoding for first 100 files from given folder
	for file in sorted(os.listdir(read_directory))[:100]:
		encoding_dict = get_encoding(file)
		file_map[file] = encoding_dict
	for f, d in file_map.items():
		if (d['encoding'] != 'utf-8') | (d['confidence'] < 0.99):
			print(f, d['encoding'], round(d['confidence'], 2))
	# print(file_map)
	# 	encoding_map[file] = encoding_dict['encoding']
	# 	confidence_map[file] = encoding_dict['confidence']
	# print(set(encoding_map.values()))
	# print(set(confidence_map.values()))
	# for file, encoding in encoding_map.items():
	# 	if encoding == 'Windows-1252':
	# 		print(file, encoding)
	# 	if encoding == 'Windows-1254':
	# 		print(file, encoding)
	# for file, confidence in confidence_map.items():
	# 	if confidence < 0.99:
	# 		print(file, confidence)
