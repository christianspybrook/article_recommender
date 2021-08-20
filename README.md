ArXiv Science Article Recommender System
========================================

Project Scope
-------------

This project builds and deploys a recommender system that can be used to source scientific articles with topics closely related to that of a given article. Currently, the input must be in the form of the unique number for an article from the [ArXiv](https://arxiv.org/) website. For example, the article titled ["Attention Is All You Need"](https://arxiv.org/pdf/1706.03762.pdf) has the ArXiv number 1706.03762. This is the value to give to use as input. The application will return 5 URLs to articles, from a random sample of 100,000 ArXiv articles dating back to April 2007, that are most similar to the input article.

Stage 1 - Data Sourcing and Text Extraction
-------------------------------------------

The data was originally downloaded from Google Storage buckets as full PDFs of the articles.
Raw text was then extracted from each PDF, using code that was sourced and adapted from a public repository.

Here you can find the source project whose modules were modified to perform these tasks:

[Dataset Source and Extraction Tools](https://github.com/mattbierbaum/arxiv-public-datasets)

This is where to find my modified code, used to parse the PDFs:

[arxiv_public_data](https://github.com/christianspybrook/article_recommender/tree/master/arxiv_public_data)

Stage 2 - Document Selection and TextPreprocessing
--------------------------------------------------

After the raw text had been extracted from the full PDFs, preprocessing of the data could begin.

As the original dataset contained all of the draft stages of the articles, I created a filter to keep only the final version of each. I filtered out a small set of articles that gave encoding errors, as well.
To decide which stopwords to remove from the text, I wrote a script to identify which library would filter the greatest number of stopwords from a large sample of the dataset. The stopwords used by the Gensim library were chosen.
After some exploratory analysis of different cleaning methods, a final preprocessing pipeline was built. This is the final script used:

[raw_text_cleaner.py](https://github.com/christianspybrook/article_recommender/blob/master/training/pdf_parsing/raw_text_cleaner.py)

Here is where to find all of the preprocessing methods used to clean the text data, before tokenization and embedding :

[pdf_parsing](https://github.com/christianspybrook/article_recommender/tree/master/training/pdf_parsing)

Stage 3 - Tokenization and Embedding
------------------------------------

<!-- Algorithms, Framaeworks, and Libraries Demonstrated:
----------------------------------------------------

1. Laten Dirichlet Allocation
2. Convolutional Neural Network
3. GPU Parallelization
4. Random Forest
5. Tensorflow
6. spaCy
7. Scikit-learn
8. Joblib
9. Dask

Project Workflow:
-----------------

[Data Preprocessing](https://github.com/christianspybrook/eluvio_coding_challenge/blob/master/data_preprocessing/preprocessing.ipynb):  
&nbsp;&nbsp;&nbsp;&nbsp;- [x] Determine Business Objective  
&nbsp;&nbsp;&nbsp;&nbsp;- [x] Reduce Memory Footprint  
&nbsp;&nbsp;&nbsp;&nbsp;- [x] Feature Engineering  
[Topic Modeling](https://github.com/christianspybrook/eluvio_coding_challenge/blob/master/modeling/topic_modeling.ipynb):  
&nbsp;&nbsp;&nbsp;&nbsp;- [x] Text Tokenization Pipeline  
&nbsp;&nbsp;&nbsp;&nbsp;- [x] Latent Dirichlet Allocation  
&nbsp;&nbsp;&nbsp;&nbsp;- [x] Topic Analysis & Visualization  
[Classifier Selection](https://github.com/christianspybrook/eluvio_coding_challenge/blob/master/modeling/classification_model_selection.ipynb):  
&nbsp;&nbsp;&nbsp;&nbsp;- [x] Cross Validation Pipeline  
&nbsp;&nbsp;&nbsp;&nbsp;- [x] Analysis & Model Selection  
[Random Forest Optimization](https://github.com/christianspybrook/eluvio_coding_challenge/blob/master/modeling/rf_classifier.ipynb):  
&nbsp;&nbsp;&nbsp;&nbsp;- [x] Bayesian Hyperparameter Search  
&nbsp;&nbsp;&nbsp;&nbsp;- [x] Analysis & Final Model Selection    
&nbsp;&nbsp;&nbsp;&nbsp;- [x] Test Performance  
Coming Soon:  
&nbsp;&nbsp;&nbsp;&nbsp;- [ ] Neural Network Regression  
&nbsp;&nbsp;&nbsp;&nbsp;- [ ] Out of Memory Modifications Using Dask

In Progress...  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;more coming, but ready for submission as is.
 -->