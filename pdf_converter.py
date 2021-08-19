import os
import requests

import multiprocessing
from argparse import ArgumentParser
from arxiv_public_data.fulltext import convert_directory_parallel
from arxiv_public_data.config import DIR_FULLTEXT, DIR_OUTPUT, DIR_PDFTARS
from arxiv_public_data.s3_bulk_download import call

import re
from gensim.parsing.preprocessing import remove_stopwords

def get_url(article):
	url_header = 'https://arxiv.org/pdf/'
	test_url = url_header + article + '.pdf'

	return test_url	

def get_pdf(url):
	response = requests.get(url)
	temp_file = 'arxiv-data/tarpdfs/temp.pdf'
	with open(temp_file, 'wb') as f:
		f.write(response.content)

def call_converter():
	parser = ArgumentParser(description="""Convert all pdfs contained in a directory and its sub-directories into txt files""")
	parser.add_argument("-N", type=int, default=multiprocessing.cpu_count(), help="OPTIONAL number of CPUs, default "
																				  "all available")
	parser.add_argument("--PLAIN_PDFS", action="store_true",
						help="OPTIONAL, add this if plain pdfs are available in " + DIR_PDFTARS)
	args = parser.parse_args()

	# Convert directory of plain PDFs file
	convert_directory_parallel(DIR_PDFTARS, processes=args.N)
	#  Subprocesss to move the converted text files inside DIR_FULLTEXT, recursively
	call('rsync -rv --remove-source-files --prune-empty-dirs --include="*.txt" --exclude="*.pdf" '
		 '--exclude="{}" --exclude="{}" {} {} '.format(os.path.basename(DIR_FULLTEXT),
													   os.path.basename(DIR_OUTPUT),
													   DIR_PDFTARS + os.sep, DIR_FULLTEXT), 0)




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

def clean_file():
	"""Reads in a file, cleans its text, and writes a new file"""

	with open('arxiv-data/fulltext/temp.txt', 'r') as f_in:
		# read in text file
		doc = f_in.readlines()

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
	with open('arxiv-data/temp/query_clean.txt', 'w') as f_out:
		# write new text string to new text file
		for new_line in new_doc:
			f_out.write(new_line)
