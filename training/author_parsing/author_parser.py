import json
import pandas as pd

# retrieve my home directory
local_file = '../home_directory/home_dir.txt'
with open(local_file, 'r') as file:
	my_home = file.read().rstrip('\r\n')

# set repository folder path
repo_path = my_home + '/github/article_recommender'
# set unfiltered author JSON file path
full_authors_path = repo_path + '/data/authors/authors-parsed.json'
# set filtered author JSON file path
filtered_authors_path = repo_path + '/data/authors/arxiv_authors_parsed_100.json'

def read_json(read_path):
	"""Returns dict from JSON file"""
	with open(read_path, 'r') as f_in:
		# read in JSON file as dict
		data = json.load(f_in)

		return data

def get_arxiv_only_dict(full_authors_filepath):
	"""Returns dict of parsed authors for arxiv folder, only"""

	#  convert full parsed authors JSON file to dictionary
	authors_dict = read_json(full_authors_filepath)
	# convert keys of autors dict to DataFrame
	key_lst = list(authors_dict.keys())
	key_df = pd.DataFrame(key_lst)
	# create boolean mask to extract articles that were from arxiv folder
	key_df['bool'] = key_df[0].str.split('/').apply(len) == 1
	# make folder names become index to allow boolean mapping
	key_df.set_index(0, inplace=True)
	# get integer corresponding to last index of arxiv article
	new_len = key_df['bool'].sum()
	# extract data from articles in arxiv folder, only
	new_dict = dict(list(authors_dict.items())[: new_len])

	return new_dict

def write_json(new_data, write_path):
	"""Writes new JSON file to specified path"""
	with open(write_path, 'w') as f_out:
		# save new JSON dict to new file
		json.dump(new_data, f_out)




if __name__ == '__main__':

	# get dict from file of only articles from arxiv folder
	arxiv_dict = get_arxiv_only_dict(full_authors_path)
	# limit each articles to only 100 authors
	arxiv_dict_100 = {k: v[: 100] for k, v in arxiv_dict.items()}

	# Note: to make DataFrame, use orient='index'

	# store new JSON file with filtered parsed authors
	write_json(arxiv_dict_100, filtered_authors_path)
