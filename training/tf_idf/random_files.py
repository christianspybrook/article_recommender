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
one_field_read_directory = raid_path + '/arxiv_data/clean_astro-ph_latest'

# set joblib dump path
data_directory = raid_path + '/arxiv_data/joblib_dump/tf_idf_data'




############################################################
#################### STRUCTURE DATASET #####################
############################################################




# set number of training files
train_size = int(1e4)
# set proportionality of minority class files
minority_train_prop = 0.05
# set proportionality of test files
test_prop = 0.01

# calculate number of minority class training files
minority_train_size = int(train_size * minority_train_prop)
# calculate number of test files
test_size = int(train_size * test_prop)
# calculate total number of minority class files
one_field_size = minority_train_size + test_size
# calculate number of majority class files
arxiv_size = train_size - minority_train_size




#############################################################
#################### COLLECT FILE LISTS #####################
#############################################################




# instantiate minority class file path list
all_one_field_paths = []

# get minority class folders list
for dir in sorted(os.listdir(one_field_read_directory)):

	# collect all minority class files
	all_one_field_files = sorted(os.listdir(os.path.join(one_field_read_directory, dir)))
	
	# iterate through files from given path
	for file in all_one_field_files:
		# add file path to minority class list
		all_one_field_paths.append(one_field_read_directory + '/' + dir + '/' + file)

# select random set of minority class files
rand_one_field_files = random.sample(all_one_field_paths, one_field_size)

# instantiate majority file path list
all_arxiv_paths = []

# get majority class folders list
for dir in sorted(os.listdir(arxiv_read_directory)):

	# collect all majority class files
	all_arxiv_files = sorted(os.listdir(os.path.join(arxiv_read_directory, dir)))

	# iterate through files from given path
	for file in all_arxiv_files:
		# add file path to majority class list
		all_arxiv_paths.append(arxiv_read_directory + '/' + dir + '/' + file)

# select random set of majority class files
rand_arxiv_files = random.sample(all_arxiv_paths, arxiv_size)

# select minority class training files
rand_minority_train_files = rand_one_field_files[:-test_size]
# select test files
rand_test_files = rand_one_field_files[-test_size:]

# assign majority class to arxiv files
labeled_arxiv = {k: 0 for k in rand_arxiv_files}
# assign minority class to specific scientific field files
labeled_one_field_train = {k: 1 for k in rand_minority_train_files}
labeled_one_field_test = {k: 1 for k in rand_test_files}

# merge training file paths
labeled_train_paths = labeled_arxiv | labeled_one_field_train
# shuffle training file paths
train_keys = list(labeled_train_paths.keys())
random.shuffle(train_keys)
shuffled_train_paths = {k: labeled_train_paths[k] for k in train_keys}

# store data and labels
dump(list(shuffled_train_paths.keys()), data_directory + '/train_data_paths.joblib')
dump(list(shuffled_train_paths.values()), data_directory + '/train_label_paths.joblib')
dump(list(labeled_one_field_test.keys()), data_directory + '/test_data_paths.joblib')
dump(list(labeled_one_field_test.values()), data_directory + '/test_label_paths.joblib')

print('Done.')
print(f'Data stored in {data_directory} directory.')
