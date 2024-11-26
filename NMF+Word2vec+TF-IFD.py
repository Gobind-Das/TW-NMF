import pandas as pd
import numpy as np
from sklearn.decomposition import NMF, PCA
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from gensim.models import Word2Vec, LsiModel, LdaModel
from gensim import corpora
from gensim.models.coherencemodel import CoherenceModel
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords



# Preprocess the Data
def preprocess_text(text):
    text = text.lower()  # Convert text to lowercase
    tokens = word_tokenize(text)  # Tokenize the text
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words and word.isalpha()]  # Remove stop words and punctuation
    return tokens

# Generate TF-IDF weighted Word2Vec document embeddings
def get_document_embedding(doc, tfidf_vocab, tfidf_scores):
    embeddings = []
    for word in doc:
        if word in word2vec_model.wv and word in tfidf_vocab:
            word_idx = np.where(tfidf_vocab == word)[0][0]  # Get the first index if word exists
            embeddings.append(word2vec_model.wv[word] * tfidf_scores[word_idx])
    if embeddings:
        return np.mean(embeddings, axis=0)
    else:
        return np.zeros(word2vec_model.vector_size)



# Extract topics from NMF for Coherence Calculation
def extract_nmf_topics(model, feature_names, no_top_words=15):
    topics = []
    for topic_idx, topic in enumerate(model.components_):
        top_words = [feature_names[i] for i in topic.argsort()[-no_top_words:][::-1]]
        topics.append(top_words)
    return topics

# Generate Topics for TF-IDF Weighted Word2Vec NMF
def display_word2vec_nmf_topics(model, tokenized_docs, no_top_words=15):
    topics = []
    for idx, topic in enumerate(model.components_):
        top_word_indices = topic.argsort()[-no_top_words:][::-1]
        all_words = set(word for doc in tokenized_docs for word in doc)
        topic_words = [list(all_words)[i] for i in top_word_indices if list(all_words)[i] in word2vec_model.wv]
        print(f"\nTF-IDF Weighted Word2Vec NMF Topic {idx + 1}: {', '.join(topic_words)}")
        topics.append(topic_words)
    return topics






# Load and preprocess the CSV Data
df = pd.read_csv('Indian_Domestic_Airline.csv')
documents = df['Review'].dropna().tolist()

# Preprocess each document
tokenized_documents = [preprocess_text(doc) for doc in documents]

# Create Dictionary and Corpus
dictionary = corpora.Dictionary(tokenized_documents)
corpus = [dictionary.doc2bow(doc) for doc in tokenized_documents]

# Train Word2Vec Model
word2vec_model = Word2Vec(sentences=tokenized_documents, window=5, min_count=1)

# Compute TF-IDF weights for Standard NMF
tfidf_vectorizer = TfidfVectorizer(tokenizer=lambda doc: doc, lowercase=False)
tfidf_matrix = tfidf_vectorizer.fit_transform([' '.join(doc) for doc in tokenized_documents])
tfidf_vocab = tfidf_vectorizer.get_feature_names_out()


# Create TF-IDF weighted Word2Vec document embeddings
document_embeddings_list = []
for doc in tokenized_documents:
    tfidf_scores = tfidf_vectorizer.transform([' '.join(doc)]).toarray().flatten()
    embedding = get_document_embedding(doc, tfidf_vocab, tfidf_scores)
    document_embeddings_list.append(embedding)

document_embeddings = np.array(document_embeddings_list)

# Ensure document embeddings are non-negative for NMF
document_embeddings = np.abs(document_embeddings)



num_topics = 5

###############################################################################

# LSI Model and Coherence Calculation
lsi_model = LsiModel(corpus=corpus, num_topics=num_topics, id2word=dictionary)
lsi_coherence_model = CoherenceModel(model=lsi_model, texts=tokenized_documents, dictionary=dictionary, coherence='c_v')
lsi_coherence_topic_scores = lsi_coherence_model.get_coherence_per_topic()
lsi_coherence = lsi_coherence_model.get_coherence()

print("\nLSI Topic-wise Coherence Scores:")
for idx, score in enumerate(lsi_coherence_topic_scores, 1):
    print(f"Topic {idx}: {score:.4f}")
print(f"\nLSI Overall Coherence Score: {lsi_coherence:.4f}")

###############################################################################

# LDA Model and Coherence Calculation
lda_model = LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary, random_state=42)
lda_coherence_model = CoherenceModel(model=lda_model, texts=tokenized_documents, dictionary=dictionary, coherence='c_v')
lda_coherence_topic_scores = lda_coherence_model.get_coherence_per_topic()
lda_coherence = lda_coherence_model.get_coherence()

print("\nLDA Topic-wise Coherence Scores:")
for idx, score in enumerate(lda_coherence_topic_scores, 1):
    print(f"Topic {idx}: {score:.4f}")
print(f"\nLDA Overall Coherence Score: {lda_coherence:.4f}")


###############################################################################

# Apply Standard NMF directly on the TF-IDF matrix (not Word2Vec embeddings)

nmf_standard_model = NMF(n_components=num_topics, max_iter=400, random_state=42)  # Increased max_iter for better convergence
nmf_standard_model.fit(tfidf_matrix)  # Apply NMF on the TF-IDF matrix

# Extract topics from the NMF model
nmf_standard_topics = extract_nmf_topics(nmf_standard_model, tfidf_vectorizer.get_feature_names_out())

# Compute Coherence Score for Standard NMF
nmf_standard_coherence_model = CoherenceModel(topics=nmf_standard_topics, texts=tokenized_documents, dictionary=dictionary, coherence='c_v')
nmf_standard_coherence = nmf_standard_coherence_model.get_coherence()
nmf_standard_coherence_topic_scores = nmf_standard_coherence_model.get_coherence_per_topic()

print("\nStandard NMF Topic-wise Coherence Scores:")
for idx, score in enumerate(nmf_standard_coherence_topic_scores, 1):
    print(f"Topic {idx}: {score:.4f}")
print(f"\nStandard NMF Overall Coherence Score: {nmf_standard_coherence:.4f}")



###############################################################################



# Apply TF-IDF weighted Word2Vec NMF
nmf_word2vec_model = NMF(n_components=num_topics, max_iter=400, random_state=42)  # Increased max_iter for better convergence
nmf_word2vec_model.fit(document_embeddings)


# Display TF-IDF Word2Vec NMF Topics
nmf_word2vec_topics = display_word2vec_nmf_topics(nmf_word2vec_model, tokenized_documents, no_top_words=15)

# Compute Coherence Score for TF-IDF Weighted Word2Vec NMF
nmf_word2vec_coherence_model = CoherenceModel(topics=nmf_word2vec_topics, texts=tokenized_documents, dictionary=dictionary, coherence='c_v')
nmf_word2vec_coherence = nmf_word2vec_coherence_model.get_coherence()
nmf_word2vec_coherence_topic_scores = nmf_word2vec_coherence_model.get_coherence_per_topic()

print("\nTF-IDF Weighted Word2Vec NMF Topic-wise Coherence Scores:")
for idx, score in enumerate(nmf_word2vec_coherence_topic_scores, 1):
    print(f"Topic {idx}: {score:.4f}")
print(f"\nTF-IDF Weighted Word2Vec NMF Overall Coherence Score: {nmf_word2vec_coherence:.4f}")



###############################################################################



# Visualize document-topic mapping using PCA for TF-IDF Weighted Word2Vec NMF
pca = PCA(n_components=2)
document_embeddings_2d = pca.fit_transform(document_embeddings)
plt.figure(figsize=(10, 7))
colormap = plt.get_cmap('Spectral')

# Assign colors for each topic
doc_topics = np.argmax(nmf_word2vec_model.transform(document_embeddings), axis=1)
colors = [colormap(i / num_topics) for i in range(num_topics)]

# Scatter plot for documents, color-coded by topic
for i in range(num_topics):
    plt.scatter(document_embeddings_2d[doc_topics == i, 0],
                document_embeddings_2d[doc_topics == i, 1],
                color=colors[i], label=f"Topic {i+1}", alpha=0.6)

plt.title("PCA Visualization of Document-Topic Mapping (TF-IDF Weighted Word2Vec + NMF)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.savefig('TF-IDF_NMF_PCA.png')
plt.show()
