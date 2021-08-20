import numpy as np
from joblib import load
import random

# set seed for reproducibility
random.seed(27)




#######################################################
#################### PATH PARSING #####################
#######################################################




# retrieve my home directory path
local_file = '../home_directory/home_dir.txt'
with open(local_file, 'r') as file:
	my_home = file.read().rstrip('\r\n')

# set RAID 0 array mount path
raid_path = my_home + '/mnt/4T_nvme'
# set joblib directory path
joblib_path = raid_path + '/arxiv_data/joblib_dump'

# set train label list path
train_label_path = joblib_path + '/tf_idf_data/train_label_paths.joblib'
# set test label list path
test_label_path = joblib_path + '/tf_idf_data/test_label_paths.joblib'
# set similarities object path
similarities_path = joblib_path + '/tf_idf_objects/similarities.joblib'

# set training file list path
train_file_paths = joblib_path + '/tf_idf_data/train_data_paths.joblib'
# set test file list path
test_file_paths = joblib_path + '/tf_idf_data/test_data_paths.joblib'




#########################################################
#################### MAIN EXECUTION #####################
#########################################################




if __name__ == '__main__':

	print('Loading data...')
	# load training labels
	train_data = load(train_label_path)
	# load test labels
	test_data = load(test_label_path)
	# load similarity scores
	similarities_matrix = load(similarities_path)
	# load training file paths
	train_paths = load(train_file_paths)
	# load test file paths
	test_paths = load(test_file_paths)
	print('Done.\n')

	print(f'# training files: {len(train_data)}')
	print(f'# test files: {len(test_data)}')
	print(f'Similarity matrix Dim: {similarities_matrix.shape}\n')

	# get average score over whole matrix
	print('All train...')
	mean_sim = np.mean(similarities_matrix)
	print(f'Dimensions: {similarities_matrix.shape}')
	print(f'Average similarity score: {mean_sim}\n')
	mean_sim_rows = np.mean(similarities_matrix, axis=1)
	idx2mean = {idx: score for idx, score in enumerate(mean_sim_rows)}
	sorted_scores = {k: v for k, v in sorted(idx2mean.items(), key=lambda item: item[1], reverse=True)}
	print(f'{len(sorted_scores)} scores...')
	print(list(sorted_scores.items())[: 50], '\n')	

	# get average score where training article was minority class
	print('Minority train...')
	minority_train_idx = np.nonzero(train_data)[0]
	minority_train = similarities_matrix[minority_train_idx]
	mean_minority_train = np.mean(minority_train)
	print(f'Dimensions: {minority_train.shape}')
	print(f'Average similarity score: {mean_minority_train}\n')
	mean_minority_rows = np.mean(minority_train, axis=1)
	idx2mean_minority = {idx: score for idx, score in idx2mean.items() if idx in minority_train_idx}
	minority_sorted_scores = {k: v for k, v in sorted(idx2mean_minority.items(), key=lambda item: item[1], reverse=True)}
	print(f'{len(minority_sorted_scores)} scores...')
	print(list(minority_sorted_scores.items())[: 50], '\n')

	# get average score where training article was majority class
	print('Majority train...')
	majority_train_idx = np.nonzero(np.array(train_data) == 0)[0]
	majority_train = similarities_matrix[majority_train_idx, :]
	mean_majority_train = np.mean(majority_train)
	print(f'Dimensions: {majority_train.shape}')
	print(f'Average similarity score: {mean_majority_train}\n')
	mean_majority_rows = np.mean(majority_train, axis=1)
	idx2mean_majority = {idx: score for idx, score in idx2mean.items() if idx in majority_train_idx}
	majority_sorted_scores = {k: v for k, v in sorted(idx2mean_majority.items(), key=lambda item: item[1], reverse=True)}
	print(f'{len(majority_sorted_scores)} scores...')
	print(list(majority_sorted_scores.items())[: 50], '\n')

	# get average score over whole matrix with classes added
	print('Add classes...\n')
	add_1_sorted = 	[tup + (1,) for tup in list(sorted_scores.items()) if tup in list(minority_sorted_scores.items())]
	add_0_sorted = 	[tup + (0,) for tup in list(sorted_scores.items()) if tup in list(majority_sorted_scores.items())]
	new_sorted = sorted(add_1_sorted + add_0_sorted, key=lambda item: item[1], reverse=True)
	print(f'{len(new_sorted)} scores...')
	print(new_sorted[: 50], '\n')

	# get true positive rate
	tpr = mean_minority_train / (mean_minority_train + mean_majority_train)
	print(f'True Positive Rate: {tpr}\n')

	for tup in new_sorted:
		if tup[2] == 0:
			max_score_majority_idx = tup[0]
			break

	max_score_majority_filename = train_paths[max_score_majority_idx]
	max_score_majority_arxiv_num = max_score_majority_filename.split('/')[-1]
	print(f'Best majority class file: {max_score_majority_arxiv_num}')

	for tup in new_sorted:
		if tup[2] == 1:
			max_score_minority_idx = tup[0]
			break

	max_score_minority_filename = train_paths[max_score_minority_idx]
	max_score_minority_arxiv_num = max_score_minority_filename.split('/')[-1]
	print(f'Best minority class file: {max_score_minority_arxiv_num}\n')

	# pick one file index and get top 5 recommendations
	random_test_idx = random.sample(range(len(test_paths)), 1)[0]
	random_test_filename = test_paths[random_test_idx]
	random_test_arxiv_num = random_test_filename.split('/')[-1]
	print(f'Random minority class file: {random_test_arxiv_num}\n')

	# get top 5 recommended articles
	random_article_scores = similarities_matrix[:, random_test_idx]
	idx2score_random = {idx: score for idx, score in enumerate(random_article_scores)}
	sorted_random = {k: v for k, v in sorted(idx2score_random.items(), key=lambda item: item[1], reverse=True)}
	print(f'Top 10 out of {len(sorted_random)} scores...')
	top_articles = list(sorted_random.items())[: 10]
	print(top_articles, '\n')
	top_articles_idx = [tup[0] for tup in top_articles]
	print(top_articles_idx)
	top_paths = np.array(train_paths)[top_articles_idx]
	print(top_paths)
