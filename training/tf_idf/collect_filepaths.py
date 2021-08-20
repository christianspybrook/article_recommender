import os
import random
from joblib import dump

# set seed for reproducibility
random.seed(27)




#######################################################
#################### PATH PARSING #####################
#######################################################




# retrieve my home directory
local_file = '../home_directory/home_dir.txt'
with open(local_file, 'r') as file:
	my_home = file.read().rstrip('\r\n')

# set RAID 0 array mount path
raid_path = my_home + '/mnt/4T_nvme'

# set text directory paths
arxiv_read_directory = raid_path + '/arxiv_data/clean_text_latest'

# set joblib dump path
data_directory = raid_path + '/arxiv_data/joblib_dump/tf_idf_data'




#############################################################
#################### COLLECT FILE LISTS #####################
#############################################################




# instantiate file path list
all_arxiv_paths = []

# get arxiv folders list
for dir in sorted(os.listdir(arxiv_read_directory)):

	# collect all arxiv files from directories
	all_arxiv_files = sorted(os.listdir(os.path.join(arxiv_read_directory, dir)))

	# iterate through files from given path
	for file in all_arxiv_files:
		# add file path to arxiv list
		all_arxiv_paths.append(arxiv_read_directory + '/' + dir + '/' + file)

# shuffle file paths
random.shuffle(all_arxiv_paths)

# store data and labels for all paths and smaller subset
dump(all_arxiv_paths[:int(1e5)], data_directory + '/arxiv_paths_100k.joblib')
dump(all_arxiv_paths, data_directory + '/all_arxiv_paths.joblib')

print('Done.')
print(f'List of file paths stored in {data_directory} directory.')
