# TW-NMF
approach integrates the semantic richness of Word2Vec(W2ve) embeddings with NMF, weighted by TF-IDF scores called TW-NMF
# At a glance:
This code utilizes a combination of TF-IDF Weighted Word2Vec along with Non-Negative Matrix Factorization (NMF) and comparisons are done based on other topic modeling methodologies: Latent Dirichlet Allocation (LDA), Latent Semantic Indexing (LSI) that have been used here along with applying them on different datasets given in dataset folder ( available on Kaggle). This extract the meaningful topics from the dataset using our proposed approch named as TW-NMF. It determines coherence scores for generated topics and visualizes the topic-document mapping.

# Required Libraries: 
pandas, numpy, nltk, sklearn, gensim, matplotlib

# Code Structure
  Data Loading: The script begins by loading a CSV file.
  
  Text Preprocessing: Tokenizes, cleans, and removes stopwords from the text data to prepare it for modeling.
  
  TF-IDF and Word2Vec Embeddings:
    TF-IDF is applied to weigh the importance of words in each document.
    Word2Vec is used to generate semantic embeddings for the words in the corpus.
    Document embeddings are created by combining Word2Vec embeddings with TF-IDF weights.
    
  NMF and Topic Extraction:
    TF-IDF Weighted Word2Vec NMF: NMF is applied to the TF-IDF weighted Word2Vec document embeddings to extract topics.
    Standard NMF: NMF is applied directly to the TF-IDF matrix for comparison.
    LDA and LSI: These models are also trained and their coherence scores are calculated.
    Coherence Score Calculation: Measures the interpretability of topics for all models.
  
  Visualization: 
    PCA is used to reduce dimensionality and visualize the document-topic mapping.

# Output
The code will print out:
Topic-wise Coherence Scores for each model (TF-IDF Weighted Word2Vec NMF, Standard NMF, LDA, LSI).
Overall Coherence Score for each model.
PCA Plot of document-topic mapping (for TF-IDF Weighted Word2Vec NMF).
