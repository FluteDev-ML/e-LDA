import pandas as pd
from gensim import corpora, models
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from gensim.models import Word2Vec
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pyLDAvis
import pyLDAvis.gensim_models
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load the data from CSV
output_csv = 'bbc_news_20220307_20240703.csv'
df = pd.read_csv(output_csv)

# Display the first few rows to understand the structure of your data
print("Head of the DataFrame:")
print(df.head())

# Preprocess the text data
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    additional_stopwords = {"'s", "``", "''"}
    stop_words = stop_words.union(additional_stopwords)
    tokens = [token for token in tokens if token not in stop_words and token not in string.punctuation]
    return tokens

df['processed_text'] = df['description'].apply(preprocess_text)

# Train a Word2Vec model on the processed text
word2vec_model = Word2Vec(sentences=df['processed_text'])

# Create a Gensim Dictionary and Corpus
dictionary = corpora.Dictionary(df['processed_text'])
dictionary.filter_extremes(no_below=3, no_above=0.5)
corpus = [dictionary.doc2bow(doc) for doc in df['processed_text']]

# Build and Train Traditional LDA Model
num_topics = 5
lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary)

# Assign Topics to Each Document
topics = []
for text in range(len(df['processed_text'])):
    topic_distribution = lda_model[corpus[text]]
    sorted_topics = sorted(topic_distribution, key=lambda x: x[1], reverse=True)
    topics.append(sorted_topics[0][0])
df['topic'] = topics

# Count Topic Occurrences
topic_counts = df['topic'].value_counts()
print("Topic Counts:")
print(topic_counts)

# Print Traditional LDA Topics
print("\nTraditional LDA Topics:")
for idx, topic in lda_model.print_topics():
    print(f"Topic {idx}: {topic}")

# Save pyLDAvis visualization for Traditional LDA
lda_display = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
pyLDAvis.save_html(lda_display, 'Traditional_LDA.html')
print("Traditional LDA visualization has been saved to Traditional_LDA.html")

# Compute Traditional LDA Topic Coherence
def compute_topic_coherence(topic_vectors, document_vectors):
    coherence_scores = []
    for topic_vector in topic_vectors:
        similarities = cosine_similarity([topic_vector], document_vectors)
        coherence_scores.append(np.mean(similarities))
    overall_coherence = np.mean(coherence_scores)
    return coherence_scores, overall_coherence

# Compute and print coherence for Traditional LDA

# Compute Coherence Score for Each Topic in Traditional LDA
coherence_model = models.CoherenceModel(model=lda_model, corpus=corpus, dictionary=dictionary, coherence='u_mass')
topics_coherence = coherence_model.get_coherence_per_topic()
print("\nTraditional LDA Topic coherence Scores:")
for idx, score in enumerate(topics_coherence):
    print(f"Topic {idx}: {score}")
    
    
    
def compute_coherence(model, corpus, dictionary):
    coherence_model_lda = models.CoherenceModel(model=model, corpus=corpus, dictionary=dictionary, coherence='u_mass')
    coherence_lda = coherence_model_lda.get_coherence()
    return coherence_lda

traditional_coherence_score = compute_coherence(lda_model, corpus, dictionary)
print(f"\nTraditional LDA Coherence Score: {traditional_coherence_score}")

    
    

# Enhance LDA results with embeddings
def get_topic_terms_embedding(topic_terms, word2vec_model):
    topic_vector = np.mean([word2vec_model.wv[term] for term, _ in topic_terms if term in word2vec_model.wv], axis=0)
    return topic_vector

def get_document_embedding(document, word2vec_model):
    word_vectors = [word2vec_model.wv[word] for word in document if word in word2vec_model.wv]
    if len(word_vectors) > 0:
        return np.mean(word_vectors, axis=0)
    else:
        return None

topic_vectors = [get_topic_terms_embedding(lda_model.show_topic(topicid, topn=10), word2vec_model) for topicid in range(num_topics)]

df['document_vector'] = df['processed_text'].apply(lambda doc: get_document_embedding(doc, word2vec_model))
df = df[df['document_vector'].apply(lambda x: x is not None and len(x) > 0)]
document_vectors = np.array(df['document_vector'].tolist())

# Check if the document_vectors array is valid
if document_vectors.size == 0:
    raise ValueError("No valid document vectors found. Ensure your Word2Vec model is correctly trained, and that the documents contain words present in the model's vocabulary.")

# Compute Enhanced LDA Topic Coherence
def compute_topic_coherence(topic_vectors, document_vectors):
    coherence_scores = []
    for topic_vector in topic_vectors:
        similarities = cosine_similarity([topic_vector], document_vectors)
        coherence_scores.append(np.mean(similarities))
    overall_coherence = np.mean(coherence_scores)
    return coherence_scores, overall_coherence

traditional_coherence_scores, traditional_overall_coherence_score = compute_topic_coherence(topic_vectors, document_vectors)

# Assign Enhanced Topics
def assign_enhanced_topic(document_vector, topic_vectors):
    similarities = cosine_similarity([document_vector], topic_vectors)
    assigned_topic = np.argmax(similarities)
    return assigned_topic

df['enhanced_topic'] = df['document_vector'].apply(lambda doc_vector: assign_enhanced_topic(doc_vector, topic_vectors))

# Count Enhanced Topic Occurrences
enhanced_topic_counts = df['enhanced_topic'].value_counts()
print("Enhanced Topic Counts:")
print(enhanced_topic_counts)

# Print Enhanced LDA Topics with Word Embeddings
print("\nEnhanced LDA Topics with Word Embeddings:")
for idx, topic_vector in enumerate(topic_vectors):
    print(f"Enhanced Topic {idx}: {topic_vector}")

# Compute Enhanced LDA Topic Coherence
enhanced_coherence_scores, enhanced_overall_coherence_score = compute_topic_coherence(topic_vectors, document_vectors)

# Print Enhanced Coherence Scores
print("\nEnhanced Topic Coherence Scores:")
for idx, score in enumerate(enhanced_coherence_scores):
    print(f"Enhanced Topic {idx}: {score}")
print(f"\nEnhanced Overall Coherence Score: {enhanced_overall_coherence_score}")

# Save pyLDAvis visualization for Enhanced LDA
enhanced_lda_display = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
pyLDAvis.save_html(enhanced_lda_display, 'Enhanced_LDA.html')
print("Enhanced LDA visualization has been saved to Enhanced_LDA.html")

# PCA for Traditional LDA Visualization
pca = PCA(n_components=2)
traditional_document_embeddings_2d = pca.fit_transform(document_vectors)
traditional_topic_embeddings_2d = pca.transform(topic_vectors)

plt.figure(figsize=(10, 7))
plt.scatter(traditional_document_embeddings_2d[:, 0], traditional_document_embeddings_2d[:, 1], c=df['topic'], cmap='rainbow', alpha=0.7)
plt.scatter(traditional_topic_embeddings_2d[:, 0], traditional_topic_embeddings_2d[:, 1], c='black', marker='X', s=200)
plt.title("Traditional LDA Topic Visualization (PCA)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.colorbar(label="Topic")
plt.savefig('Traditional_LDA_Visualization.png')
plt.show()

# PCA for Enhanced LDA Visualization
enhanced_document_embeddings_2d = pca.fit_transform(document_vectors)
enhanced_topic_embeddings_2d = pca.transform(topic_vectors)

plt.figure(figsize=(10, 7))
plt.scatter(enhanced_document_embeddings_2d[:, 0], enhanced_document_embeddings_2d[:, 1], c=df['enhanced_topic'], cmap='rainbow', alpha=0.7)
plt.scatter(enhanced_topic_embeddings_2d[:, 0], enhanced_topic_embeddings_2d[:, 1], c='black', marker='X', s=200)
plt.title("Enhanced LDA Topic Visualization (PCA)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.colorbar(label="Enhanced Topic")
plt.savefig('Enhanced_LDA_Visualization.png')
plt.show()
