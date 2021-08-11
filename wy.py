from nltk import word_tokenize, WordNetLemmatizer
from nltk.corpus import stopwords
import string
import re

import pandas as pd 
from gensim.models import Word2Vec
from gensim.models import keyedvectors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from numpy import dot
from numpy.linalg import norm

from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.manifold import TSNE

links_csv = "~/Projects/w2v/data/links.csv"
df = pd.read_csv(links_csv)
df.index += 1

def remove_punct(text):
    text_nopunct = ''
    text_nopunct = re.sub('['+string.punctuation+']', '', text)
    return text_nopunct

def lower_token(tokens): 
    return [w.lower() for w in tokens]  

def remove_stop_words(tokens):
    stoplist = stopwords.words('english')
    return [word for word in tokens if word not in stoplist]

"""
To remove numbers and other silly characters these buggers use in titles
"""

def clean_sentence(val):
    "remove chars that are not letters or numbers, downcase, then remove stop words"
    regex = re.compile('([^\s\w]|_)+')
    sentence = regex.sub('', val).lower()
    "removes 2019 like strings but not 4th. As just th on its own does not make sense"
    sentence = re.sub("\d+", " ", sentence) 
    sentence = sentence.split(" ")
    sentence = " ".join(sentence)
    return sentence

def clean_dataframe(data):
    "drop nans, then apply 'clean_sentence' function to question1 and 2"
    data = data.dropna(how="any")
    
    for col in ['Text_Final']:
        data[col] = data[col].apply(clean_sentence)
    
    return data

df['Text_Clean'] = df['title'].apply(lambda x: remove_punct(x))
df['Text_Clean'] = df['title'].apply(lambda x: clean_sentence(x))
tokens = [word_tokenize(sen) for sen in df.Text_Clean]
lower_tokens = [lower_token(token) for token in tokens]
filtered_words = [remove_stop_words(sen) for sen in lower_tokens]
result = [' '.join(sen) for sen in filtered_words]
df['Text_Final'] = result
df['tokens'] = filtered_words
df.head()
combined_df= df[['title','link','Text_Final', 'tokens']]

data = clean_dataframe(combined_df)

w2v_input = []
#print (combined_df['tokens'][1])
for i in range (1,167):
    w2v_input.append(data['tokens'][i])

## Train the genisim word2vec model with our own custom corpus
model = Word2Vec(w2v_input, min_count=1,vector_size=50,workers=4, window =3, sg = 1)

# print (model.wv['eeg'])

vocab_list = list (model.wv.index_to_key)
# print (vocab_list)

def vectorize(list_of_docs, model):
    """Generate vectors for list of documents using a Word Embedding

    Args:
        list_of_docs: List of documents
        model: Gensim's Word Embedding

    Returns:
        List of document vectors
    """
    features = []

    for tokens in list_of_docs:
        zero_vector = np.zeros(model.vector_size)
        vectors = []
        for token in tokens:
            if token in model.wv:
                try:
                    vectors.append(model.wv[token])
                except KeyError:
                    continue
        if vectors:
            vectors = np.asarray(vectors)
            avg_vec = vectors.mean(axis=0)
            features.append(avg_vec)
        else:
            features.append(zero_vector)
    return features

vectorized_docs = vectorize(w2v_input, model)

def mbkmeans_clusters(
    X, 
    k, 
    mb, 
    print_silhouette_values, 
):
    """Generate clusters and print Silhouette metrics using MBKmeans

    Args:
        X: Matrix of features.
        k: Number of clusters.
        mb: Size of mini-batches.
        print_silhouette_values: Print silhouette values per cluster.

    Returns:
        Trained clustering model and labels based on X.
    """
    km = MiniBatchKMeans(n_clusters=k, batch_size=mb).fit(X)
    print(f"For n_clusters = {k}")
    print(f"Silhouette coefficient: {silhouette_score(X, km.labels_):0.2f}")
    print(f"Inertia:{km.inertia_}")

    if print_silhouette_values:
        sample_silhouette_values = silhouette_samples(X, km.labels_)
        print(f"Silhouette values:")
        silhouette_values = []
        for i in range(k):
            cluster_silhouette_values = sample_silhouette_values[km.labels_ == i]
            silhouette_values.append(
                (
                    i,
                    cluster_silhouette_values.shape[0],
                    cluster_silhouette_values.mean(),
                    cluster_silhouette_values.min(),
                    cluster_silhouette_values.max(),
                )
            )
        silhouette_values = sorted(
            silhouette_values, key=lambda tup: tup[2], reverse=True
        )
        for s in silhouette_values:
            print(
                f"    Cluster {s[0]}: Size:{s[1]} | Avg:{s[2]:.2f} | Min:{s[3]:.2f} | Max: {s[4]:.2f}"
            )
    return km, km.labels_

docs = data['Text_Final'].values

clustering, cluster_labels = mbkmeans_clusters(
	X=vectorized_docs,
    k=7,
    mb=500,
    print_silhouette_values=True,
)
df_clusters = pd.DataFrame({
    "text": docs,
    "tokens": [" ".join(text) for text in w2v_input],
    "cluster": cluster_labels
})

print("Most representative terms per cluster (based on centroids):")
for i in range(7):
    tokens_per_cluster = ""
    most_representative = model.wv.most_similar(positive=[clustering.cluster_centers_[i]], topn=5)
    for t in most_representative:
        tokens_per_cluster += f"{t[0]} "
    print(f"Cluster {i}: {tokens_per_cluster}")

test_cluster = 2
most_representative_docs = np.argsort(
    np.linalg.norm(vectorized_docs - clustering.cluster_centers_[test_cluster], axis=1)
)
for d in most_representative_docs[:3]:
    print(docs[d])
    print("-------------")

def display_closestwords_tsnescatterplot(model, word, size):
    
    arr = np.empty((0,size), dtype='f')
    word_labels = [word]

    close_words = model.wv.similar_by_word(word)

    arr = np.append(arr, model.wv[word_labels], axis=0)
    for wrd_score in close_words:
        wrd_vector = model.wv[wrd_score[0]]
        word_labels.append(wrd_score[0])
        arr = np.append(arr, np.array([wrd_vector]), axis=0)
        
    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(arr)

    x_coords = Y[:, 0]
    y_coords = Y[:, 1]
    plt.scatter(x_coords, y_coords)

    for label, x, y in zip(word_labels, x_coords, y_coords):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.xlim(x_coords.min()+0.00005, x_coords.max()+0.00005)
    plt.ylim(y_coords.min()+0.00005, y_coords.max()+0.00005)
    plt.show()

display_closestwords_tsnescatterplot(model, 'ecg', 50)
