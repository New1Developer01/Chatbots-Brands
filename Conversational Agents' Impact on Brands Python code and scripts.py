# Check if packages are installed, otherwise install them
packages = ['pyLDAvis', 'openpyxl', 'pandas', 'openai']
for package in packages:
    try:
        version(package)
    except:
        !pip install {package}

# Import necessary libraries
import os
import json
import logging
import multiprocessing
from pprint import pprint

# Gensim
import gensim
from gensim.models import CoherenceModel, LdaMulticore, TfidfModel
from gensim.models.wrappers.dtmmodel import DtmModel
from gensim.models import ldaseqmodel
from gensim.corpora import Dictionary
from gensim.utils import simple_preprocess

# NLTK
import nltk
from nltk.corpus import stopwords
from nltk import download

# Spacy
import spacy
from spacy.cli.download import download

# pyLDAvis
import pyLDAvis
import pyLDAvis.gensim_models

# Matplotlib
import matplotlib.pyplot as plt

# OpenAI
import openai

# Additional Libraries
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Enable logging for gensim
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

# Download NLTK stopwords
download('stopwords')

# Initialize Spacy 'en' model
download('en')
nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
nlp.max_length = 2000000  # or even higher

# Define additional stopwords
all_stopwords = sp.Defaults.stop_words | {'study', 'number', 'process', 'base', 'model', 'use', 'system',
                                         'systems', 'bid', 'fig', 'method', 'result'}

# NLTK Stop words
stop_words = stopwords.words('english')
stop_words.extend(['study', 'number', 'process', 'base', 'model', 'use', 'system'])

# Load data from a JSON file
corpus_file = '/path/to/your/corpus.json'
documents_raw = json.load(open(corpus_file, 'r'))
data = [document['text'] for document in documents_raw]

# Define a function to tokenize sentences
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

data_words = list(sent_to_words(data))

# Build bigram and trigram models
bigram = gensim.models.Phrases(data_words, min_count=10, threshold=15)
trigram = gensim.models.Phrases(bigram[data_words], threshold=5)

bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

# Define functions for text preprocessing
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in all_stopwords] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

# Remove Stop Words
data_words_nostops = remove_stopwords(data_words)

# Form Bigrams
data_words_bigrams = make_bigrams(data_words_nostops)

# Do lemmatization keeping only noun, adj, vb, adv
def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

# Remove stopwords again
def remove_stopwords2(texts):
    return [[word for word in doc if word not in all_stopwords] for doc in texts]

lemmatized_stopped = remove_stopwords2(data_lemmatized)

# Create Dictionary and Corpus
id2word = Dictionary(lemmatized_stopped)
texts = lemmatized_stopped
corpus = [id2word.doc2bow(text) for text in texts]

# Build LDA model
chunk_size = len(corpus) * 20 / 200
lda_model = LdaMulticore(corpus=corpus, id2word=id2word, num_topics=8, random_state=100,
                         update_every=3, chunksize=chunk_size, passes=5, alpha='auto',
                         per_word_topics=True)

# Print the Keyword in the N topics
print(lda_model.print_topics())
doc_lda = lda_model[corpus]

# Compute Perplexity
print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # A measure of model quality. Lower is better.

# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=lda_model, texts=lemmatized_stopped, dictionary=id2word, coherence='c_v')
coherencemodel_umass = CoherenceModel(model=lda_model, corpus=corpus, dictionary=id2word, coherence='u_mass')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score C_V: ', coherence_lda)
print('\nCoherence Score U_MASS: ', coherencemodel_umass.get_coherence())

# Visualize the topics using pyLDAvis
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word)

# Get the document distribution
topic_paper = []
for i in range(len(corpus)):
    info = sorted(lda_model.get_document_topics(corpus[i], minimum_probability=0.1), key=lambda x: x[1], reverse=True)
    topic_paper.append(info[0])

# Define a class for DTMcorpus
class DTMcorpus(corpora.textcorpus.TextCorpus):
    def get_texts(self):
        return self.input

    def __len__(self):
        return len(self.input)

# Set up DTM paths and parameters
dtm_home = os.environ.get('DTM_HOME', "dtm-master")
dtm_path = os.path.join(dtm_home, 'bin', 'dtm') if dtm_home else None
# You can also directly specify the path to your DTM executable

# Create an LdaSeqModel
ldaseq = ldaseqmodel.LdaSeqModel(corpus=corpus, id2word=id2word, time_slice=time_seq, num_topics=5)

model = DtmModel(dtm_path, corpus, time_seq, num_topics=7, id2word=corpus.dictionary, initialize_lda=True)

# Choose the number of topics using coherence scores
coherence_values = []
model_list = []
num_passes = 20
chunk_size = len(corpus) * num_passes / 200

for num_topics in range(2, 21):
    print(num_topics)
    model = LdaMulticore(corpus=corpus, id2word=id2word, num_topics=num_topics)
    model_list.append(model)
    coherencemodel_umass = CoherenceModel(model=model, corpus=corpus, dictionary=id2word, coherence='u_mass')
    coherencemodel_cv = CoherenceModel(model=model, corpus=corpus, dictionary=id2word, texts=texts, coherence='c_v')
    coherence_values.append((num_topics, coherencemodel_umass.get_coherence(), coherencemodel_cv.get_coherence()))

results = pd.DataFrame(coherence_values)
results = results.set_axis(['topic', 'umass', 'c_v'], axis=1, inplace=False)
s = pd.Series(results.umass.values, index=results.topic.values)
_ = s.plot()

###############################
#### DYNAMIC TOPIC MODELING ###
###############################
### based on thematic map ###
### 1985-2006 (326 papers), 2007-2010 (150 papers), 2011-2016 (392 papers), and 2017- early 2021 (497 papers).
time_seq = [327, 150, 397, 501]


class DTMcorpus(corpora.textcorpus.TextCorpus):

    def get_texts(self):
        return self.input

    def __len__(self):
        return len(self.input)

corpus = DTMcorpus(data_lemmatized)

ldaseq = ldaseqmodel.LdaSeqModel(corpus=corpus, id2word=id2word, time_slice=time_seq, num_topics=8)


### Example on how to read DTM results by period
document_no = 1 #document 2
topic_no = 1 #topic number 2
time_slice = 0 #time slice 1

model.influences_time[time_slice][document_no][topic_no]


#######################################
#### LAR LANGUAGE MODEL AND EXPERTS ###
#######################################
### based on thematic map ###
# set up OpenAI API key
openai.api_key = "PLACE YOUR openAI API KEY HERE"


# define the expert labels and GPT labels
experts_labels = [
    ['Expert 1 Topic label 1', 'Expert 1 Topic label 2', 'Expert 1 Topic label 3', 'Expert 1 Topic label 4'],
    ['Expert 2 Topic label 1', 'Expert 2 Topic label 2', 'Expert 2 Topic label 3', 'Expert 2 Topic label 4'],
    ['Expert 3 Topic label 1', 'Expert 3 Topic label 2', 'Expert 3 Topic label 3', 'Expert 3 Topic label 4'],
    ['Expert 4 Topic label 1', 'Expert 4 Topic label 2', 'Expert 4 Topic label 3', 'Expert 4 Topic label 4']
]

gpt_labels = ['GPT Topic label 1', 'GPT Topic label 2', 'GPT Topic label 3', 'GPT Topic label 4']

# initialize GPT-3 model
model_engine = "text-embedding-ada-002"


# create a function to get the GPT embeddings
def get_gpt_embeddings(text):
    response = openai.Embedding.create(
        model=model_engine,
        input=[text]
    )
    embedding = np.array(response.data[0].embedding)
    return embedding


experts_embeddings = []
for expert_labels in experts_labels:
    expert_embeddings = []
    for label in expert_labels:
        embedding = get_gpt_embeddings(label)
        expert_embeddings.append(embedding)
    experts_embeddings.append(expert_embeddings)

gpt_embeddings = []
for label in gpt_labels:
    embedding = get_gpt_embeddings(label)
    gpt_embeddings.append(embedding)


# create a function to get the label with the most similarity

def get_most_similar_label(label_embeddings, expert_embeddings, gptLabel):
    global_similarity=[]
    for indx, expert_embedding in enumerate(expert_embeddings):
        similarities = []
        for ind, label_expert in enumerate(expert_embedding):
          expert_similarity = cosine_similarity([label_embeddings], [label_expert])
          similarities.append({'score':expert_similarity[0][0], 'expert_label':experts_labels[indx][ind], 'gpt_label': gptLabel,})
        most_similar_label = np.argmax([d['score'] for d in similarities])
        # most_similar_label=0
        # print(experts_labels[indx][most_similar_label], most_similar_label, gptLabel)
        global_similarity.append({'expert': indx+1, 'similarities':similarities})
        #print(global_similarity)
    return (most_similar_label, global_similarity)

# get the most similar expert label for each GPT label
most_similar_expert_labels = []
similarity_labels = []
for indx, gpt_embedding in enumerate(gpt_embeddings):
    most_similar_label = get_most_similar_label(gpt_embedding, experts_embeddings, gpt_labels[indx])
    most_similar_expert_labels.append(most_similar_label[0])
    list_of_similarities = similarity_labels.append(most_similar_label[1])

# Concatenate all labels
all_labels = []
for label in experts_labels:
    all_labels += label
all_labels += gpt_labels

# Get embeddings for all labels
embeddings = []
for label in all_labels:
    result = openai.Embedding.create(
        engine=model_engine,
        input=[label]
    )
    embeddings.append(result.data[0].embedding)


from scipy import interpolate
from scipy.spatial import ConvexHull
# Use PCA and TSNE for dimensionality reduction
pca = PCA(n_components=48)
pca_result = pca.fit_transform(embeddings)
tsne = TSNE(n_components=2, perplexity=6)
tsne_result = tsne.fit_transform(pca_result)


"DEFINE CLUSTERS AND EXPERT COLORS"
colors = ['#ED4C67', '#F79F1F', '#F7DC6F', '#4B4E6D', '#3B3B98', '#00A8CC', '#06D6A0', '#118AB2']

# Cluster the data
kmeans = KMeans(n_clusters=8, random_state=42).fit(tsne_result)
# Get the centroids of the clusters
centroids = kmeans.cluster_centers_

# Getting points clusters
groups_kmeans = kmeans.predict(tsne_result)

#PALETTE AND MARKERS FOR CLUSTERING
col = ["#F94144", "#F3722C", "#ED4C67", "#F9C74F", "#90BE6D", "#43AA8B", "#577590", "#3B3B98"]
markers = ["D", "v", "^", "s", "p", "*"]


step = 8
predicts = []
for i in range(0, 48, step):
  x=i
  predicts.append(groups_kmeans[x:x+step])



# Plot the clusters
fig, ax = plt.subplots(figsize=(10, 10))
for i, label in enumerate(experts_labels + [gpt_labels]):
    x = tsne_result[i*8:(i+1)*8, 0]
    y = tsne_result[i*8:(i+1)*8, 1]
    info=""
    if(i<5):
      info="Expert " + str(i+1)
    else:
      info="GPT"
    ax.scatter(x, y, label=f"{info}", marker=markers[i])
    # print(np.array(predicts[i]))
    print(i, label,x, y)
    for j, txt in enumerate(label):
        info = txt
        ax.annotate(info, (x[j], y[j]), fontsize=8, c=col[groups_kmeans[j]])

for i, label in enumerate(centroids):
  ax.scatter(label[0], label[1], marker='^', s=200, c=col[i])

  new_center= centroids.astype(int)


ax.legend()
plt.show()