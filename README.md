ArXiv Science Article Recommender System
========================================

Project Scope
-------------

This project will deploy a recommender system that can be used to source scientific articles with topics closely related to that of a current experiment.

Stage 1 - Data Sourcing and Text Extraction
-------------------------------------------

The data was originally downloaded from Google Storage buckets as full PDFs of the articles.
Raw text was then extracted from each PDF.

This is the source project whose modules were used to perform these tasks:

[Dataset Source](https://github.com/mattbierbaum/arxiv-public-datasets)

Stage 2 - Preprocessing and Tokenization
----------------------------------------

Now that the raw text has been extracted from the full PDFs, preprocessing of the data can begin.

This is a sample selection of the raw text data:

[sample_data](https://github.com/christianspybrook/article_recommender/blob/master/sample_data/raw_sample.csv)

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