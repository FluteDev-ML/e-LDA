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

######################################################################################################
# FUNCTIONS
######################################################################################################
# Preprocess the text data
def preprocess_text(text):
    # Check if the input is a string
    if isinstance(text, str):
        tokens = word_tokenize(text.lower())
        stop_words = set(stopwords.words('english'))
        additional_stopwords = {"'s", "``", "''"}
        stop_words = stop_words.union(additional_stopwords)
        
        filtered_tokens = []
        for token in tokens:
            # Filter out stopwords and punctuation
            if token not in stop_words and token not in string.punctuation:
                filtered_tokens.append(token)
        
        return filtered_tokens
    else:
        return []  # Return an empty list for non-string inputs (e.g., NaN)


# Calculate Topic Coherence
def compute_topic_coherence(topic_vectors, document_vectors):
    coherence_scores = []
    for topic_vector in topic_vectors:
        similarities = cosine_similarity([topic_vector], document_vectors)
        coherence_scores.append(np.mean(similarities))
    overall_coherence = np.mean(coherence_scores)
    return coherence_scores, overall_coherence

# Calculate Overall Coherence
def compute_coherence(model, corpus, dictionary):
    coherence_model_lda = models.CoherenceModel(model=model, corpus=corpus, dictionary=dictionary, coherence='u_mass')
    coherence_lda = coherence_model_lda.get_coherence()
    return coherence_lda


# Assign Enhanced Topics
def assign_enhanced_topic(document_vector, topic_vectors):
    similarities = cosine_similarity([document_vector], topic_vectors)
    assigned_topic = np.argmax(similarities)
    return assigned_topic


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

######################################################################################################

# Load the data from CSV
output_csv = r'D:\MTECH\3rd Semester\7_ECIR2024_conference\Amazon Reviwe\Amazon.csv'
df = pd.read_csv(output_csv)

# Display the first few rows to understand the structure of your data
print("Head of the DataFrame:")
print(df.head())

# Preprocess the text
df['processed_text'] = [preprocess_text(Review_Text) for Review_Text in df['Review_Text']]

# Train a Word2Vec model on the processed text
word2vec_model = Word2Vec(sentences=df['processed_text'])

print('###################################')
print(df)

print('###################################')
# Create a Gensim Dictionary and Corpus
dictionary = corpora.Dictionary(df['processed_text'])
dictionary.filter_extremes(no_above=0.5)
corpus = [dictionary.doc2bow(doc) for doc in df['processed_text']]

# Build and Train Traditional LDA Model
num_topics = 5

lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary)

# Traditional LDA: Assign Topics to Each Document
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

# Computation of Coherence Score for Each Topic in Traditional LDA
coherence_model = models.CoherenceModel(model=lda_model, corpus=corpus, dictionary=dictionary, coherence='u_mass')
topics_coherence = coherence_model.get_coherence_per_topic()
print("\nTraditional LDA Topic Coherence Scores:")
for idx, score in enumerate(topics_coherence):
    print(f"Topic {idx}: {score}")

traditional_coherence_score = compute_coherence(lda_model, corpus, dictionary)
print(f"\nTraditional LDA Coherence Score: {traditional_coherence_score}")

####################################### Traditional LDA END #######################################

################################### Enhanced LDA ##################################################
# Generate document vectors using Word2Vec embeddings
document_vectors = []
valid_indices = []  # To keep track of valid rows

for idx, doc in enumerate(df['processed_text']):
    vector = get_document_embedding(doc, word2vec_model)
    if vector is not None and len(vector) > 0:
        document_vectors.append(vector)
        valid_indices.append(idx)  # Store the index of rows with valid document vectors

# Filter df to only include rows with valid document vectors
df = df.loc[valid_indices].reset_index(drop=True)

# Ensure that the number of document vectors matches the number of rows in df
if len(document_vectors) == len(df):
    df['document_vector'] = document_vectors
else:
    raise ValueError(f"Length of document_vectors ({len(document_vectors)}) does not match DataFrame ({len(df)})")

# Build Enhanced LDA model using document vectors
enhanced_corpus = [dictionary.doc2bow(doc) for doc in df['processed_text']]
enhanced_lda_model = models.LdaModel(enhanced_corpus, num_topics=num_topics, id2word=dictionary)

# Generate topic vectors for the Enhanced LDA model
enhanced_topic_vectors = []
for topicid in range(num_topics):
    topic_terms = enhanced_lda_model.show_topic(topicid, topn=10)
    topic_embedding = get_topic_terms_embedding(topic_terms, word2vec_model)
    enhanced_topic_vectors.append(topic_embedding)

df['enhanced_topic'] = [assign_enhanced_topic(doc_vector, enhanced_topic_vectors) for doc_vector in df['document_vector']]

# Count Enhanced Topic Occurrences
enhanced_topic_counts = df['enhanced_topic'].value_counts()
print("Enhanced Topic Counts:")
print(enhanced_topic_counts)

# Print Enhanced LDA Topics with Word Embeddings
print("\nEnhanced LDA Topics with Word Embeddings:")
for idx, topic_vector in enumerate(enhanced_topic_vectors):
    print(f"Enhanced Topic {idx}: {topic_vector}")

# Compute Enhanced LDA Topic Coherence
enhanced_coherence_scores, enhanced_overall_coherence_score = compute_topic_coherence(enhanced_topic_vectors, document_vectors)

# Print Enhanced Coherence Scores
print("\nEnhanced Topic Coherence Scores:")
for idx, score in enumerate(enhanced_coherence_scores):
    print(f"Enhanced Topic {idx}: {score}")
print(f"\nEnhanced Overall Coherence Score: {enhanced_overall_coherence_score}")

# Save pyLDAvis visualization for Enhanced LDA
enhanced_lda_display = pyLDAvis.gensim_models.prepare(enhanced_lda_model, enhanced_corpus, dictionary)
pyLDAvis.save_html(enhanced_lda_display, 'Enhanced_LDA.html')
print("Enhanced LDA visualization has been saved to Enhanced_LDA.html")

#################################### END ##################################################

############################### Plotting for Traditional and Enhanced LDA ##################

# PCA for Traditional LDA Visualization
pca = PCA(n_components=2)
traditional_document_embeddings_2d = pca.fit_transform(document_vectors)
traditional_topic_embeddings_2d = pca.transform(enhanced_topic_vectors)

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
enhanced_topic_embeddings_2d = pca.transform(enhanced_topic_vectors)

plt.figure(figsize=(10, 7))
plt.scatter(enhanced_document_embeddings_2d[:, 0], enhanced_document_embeddings_2d[:, 1], c=df['enhanced_topic'], cmap='rainbow', alpha=0.7)
plt.scatter(enhanced_topic_embeddings_2d[:, 0], enhanced_topic_embeddings_2d[:, 1], c='black', marker='X', s=200)
plt.title("Enhanced LDA Topic Visualization (PCA)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.colorbar(label="Enhanced Topic")
plt.savefig('Enhanced_LDA_Visualization.png')
plt.show()
