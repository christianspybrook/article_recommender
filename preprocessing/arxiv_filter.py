import os
import re
from shutil import copyfile

# retrieve my home directory
local_file = '../home_directory/home_dir.txt'
with open(local_file, 'r') as file:
    my_home = file.read().rstrip('\r\n')
    file.close()

# set text directory path
old_directory = my_home + '/mnt/4T_nvme/arxiv_data/raw_text'
new_directory = my_home + '/mnt/4T_nvme/arxiv_data/raw_text_latest'

# loop through folders sorted by month
for dir in sorted(os.listdir(old_directory)):
	# initialize article number for new folder
	working_article = '0000'
	for file in sorted(os.listdir(os.path.join(old_directory, dir))):
		# split filenames by version number and article number
		date_list = re.split('v|\.', file)
		# extract article number
		article_num = date_list[1]
		# extract version number
		version = int(date_list[2])

		# check if we are on a new batch of articles
		if article_num != working_article:
			# check if we are in first batch of articles
			if working_article == '0000':
				pass
			else:
				# get index of latest version from version list
				latest_idx = version_lst.index(max(version_lst))
				# get latest version of article
				latest_article = batch_lst[latest_idx]
				# get paths to copy latest version
				old_path = old_directory + '/' + dir + '/' + latest_article
				new_path = new_directory + '/' + dir + '/' + latest_article
				# store latest version of article from last batch
				copyfile(old_path, new_path)
			# set new article batch and directory number
			working_article = article_num
			# initialize article batch list
			batch_lst = []
			# initialize article version list
			version_lst = []
		# add article version number to list
		version_lst.append(version)
		# add filename to article batch list
		batch_lst.append(file)

	# get index of latest version from version list
	latest_idx = version_lst.index(max(version_lst))
	# get latest version of article
	latest_article = batch_lst[latest_idx]
	# get paths to copy latest version
	old_path = old_directory + '/' + dir + '/' + latest_article
	new_path = new_directory + '/' + dir + '/' + latest_article
	# store latest version of article from last batch
	copyfile(old_path, new_path)
