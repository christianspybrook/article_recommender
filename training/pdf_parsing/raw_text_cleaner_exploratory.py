import re
import string
import os
from gensim.parsing.preprocessing import remove_stopwords

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

#################### SAMPLE PATHS ####################

# select directories to clean
data_directories = {0: '/0704', 
					1: '/2105'
					}

# capture month and year directory to clean
d = 1

# map stages of cleaning to separate folders for analysis
text_folders = {0: '/sample_data' + '/sample_100_raw_text' + data_directories[d], 
				1: '/sample_data' + '/sample_100_no_hexadecimal' + data_directories[d], 
				2: '/sample_data' + '/sample_100_with_short_lines' + data_directories[d], 
				3: '/sample_data' + '/sample_100_with_new_lines' + data_directories[d], 
				4: '/sample_data' + '/sample_100_with_stopwords' + data_directories[d], 
				5: '/full_month_with_stopwords' + data_directories[d], 
				6: '/sample_data' + '/sample_100_cleaned' + data_directories[d], 
				7: '/full_month_cleaned' + data_directories[d]
				}

# capture cleaning stage to perform
f = 7

# set text directory paths
read_directory = raid_path + '/arxiv_data/raw_text_latest' + data_directories[d]
write_directory = repo_path + '/data' + text_folders[f]

#################### FULL DATA PATHS ####################

# set text directory path
deep_read_directory = raid_path + '/arxiv_data/astro-ph_latest' # raw_text_latest (arxiv)
deep_write_directory = raid_path + '/arxiv_data/clean_astro-ph_latest' # clean_text_latest (arxiv)




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

def clean_file(read_path, write_path):
	"""Reads in a file, cleans its text, and writes a new file"""
	# get next file
	doc = read_file(read_path)
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
			new_string = remove_stopwords(new_string) + ' '
			# add filtered string to new string list
			new_doc.append(new_string)
	# save new cleaned file
	write_file(write_path, new_doc)




#########################################################
#################### MAIN EXECUTION #####################
#########################################################




if __name__ == '__main__':

	# make option to clean files in nested folders
	DEEP = True

	if DEEP:
		# set path to nested folders to clean whole dataset
		directories = (deep_read_directory, deep_write_directory)

		for dir in sorted(os.listdir(directories[0])):
			# display cirectory being cleaned
			print(dir)

			# iterate through files from given path
			files_to_clean = sorted(os.listdir(os.path.join(directories[0], dir)))
			# set starting number of files to clean for progress display
			files_left_to_clean = len(files_to_clean)

			for file in files_to_clean:
				# display progress
				show_progress(files_left_to_clean)

				# set file paths to nested folders
				read_path = directories[0] + '/' + dir + '/' + file
				write_path = directories[1] + '/' + dir + '/' + file

				clean_file(read_path, write_path)

				# update progress display
				files_left_to_clean += -1

	else:
		# set path to top level folders for analyzing preprocessing methods
		directories = (read_directory, write_directory)

		# set limit to number of files to clean
		n_files = None

		# iterate through files from given path
		files_to_clean = sorted(os.listdir(directories[0]))[:n_files]
		# set starting number of files to clean for progress display
		files_left_to_clean = len(files_to_clean)

		for file in files_to_clean:
			# display progress
			show_progress(files_left_to_clean)

			# set file paths to top level folders
			read_path = directories[0] + '/' + file
			write_path = directories[1] + '/' + file

			clean_file(read_path, write_path)

			# update progress display
			files_left_to_clean += -1
