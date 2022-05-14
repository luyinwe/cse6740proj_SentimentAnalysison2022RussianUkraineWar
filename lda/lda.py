# -*- coding: utf-8 -*-
# !pip install pyldavis
# !pip install gensim pyLDAvis==3.2.1
# !python3 -m spacy download en_core_web_sm


import pandas as pd
import numpy as np

import string
import spacy
import nltk
 
import gensim
from gensim import corpora
 
import pyLDAvis
import pyLDAvis.gensim
 
nltk.download('wordnet')
from nltk.corpus import wordnet as wn
nltk.download('stopwords')
from nltk.corpus import stopwords
import spacy.cli
spacy.cli.download("en_core_web_md")
import en_core_web_md
import json
from gensim.models.coherencemodel import CoherenceModel
import warnings
import random
from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import matplotlib.colors as mcolors

warnings.filterwarnings("ignore")

def clean_text(text):
  delete_dict = {sp_char: '' for sp_char in string.punctuation}
  delete_dict[' '] =' '
  table = str.maketrans(delete_dict)
  text1 = text.translate(table)
  textArr= text1.split()
  text2 = ' '.join([w for w in textArr if ( not w.isdigit() and
                                           ( not w.isdigit() and len(w)>3))])
  return text2.lower()

def remove_stopwords(text):
    textArr = text.split(' ')
    rem_text = " ".join([i for i in textArr if i not in stop_words])
    return rem_text

def lemmatization(texts, nlp):
     output = []
     for sent in texts:
        doc = nlp(sent)
        output.append([token.lemma_ for token in doc if token.lemma_.isalnum()])
     return output



# prepare data
f = open('sample_df.csv', 'r')
content = f.read()
data_comment = json.loads(content)

data = {}
data['idx'] = []
data['content'] = []
data['senti'] = []
for k in data_comment['content'].keys():
  data['idx'].append(k)
  data['content'].append(data_comment['content'][k])
  data['senti'].append(data_comment['SA'][k])

data_comment = data

# find idx for different labels
label_idx = {'POS':[], 'NEU':[], 'NEG':[]}
for i in range(len(data['senti'])):
  if data['senti'][i] == 1:
     label_idx['POS'].append(i)
  elif data['senti'][i] == 0:
     label_idx['NEU'].append(i)
  else:
     label_idx['NEG'].append(i)

 
# clean the document and remove punctuation
data_comment['content'] = [clean_text(text) for text in data_comment['content']]


# shuffle and sample(optional) data
idx_pos = list(range(len(label_idx['POS'])))
idx_neu = list(range(len(label_idx['NEU'])))
idx_neg = list(range(len(label_idx['NEG'])))
random.shuffle(idx_pos)
random.shuffle(idx_neu)
random.shuffle(idx_neg)
df_sampled_pos = {}
df_sampled_neu = {}
df_sampled_neg = {}

for k in data_comment.keys():
  df_sampled_pos[k] = list(np.array(data_comment[k])[label_idx['POS']][idx_pos])
  df_sampled_neu[k] = list(np.array(data_comment[k])[label_idx['NEU']][idx_neu])
  df_sampled_neg[k] = list(np.array(data_comment[k])[label_idx['NEG']][idx_neg])



# remove stopwords from the text
stop_words = stopwords.words('english')
df_sampled_pos['content'] = [remove_stopwords(text) for text in df_sampled_pos['content']]
df_sampled_neu['content'] = [remove_stopwords(text) for text in df_sampled_neu['content']]
df_sampled_neg['content'] = [remove_stopwords(text) for text in df_sampled_neg['content']]


# perform Lemmatization
nlp = en_core_web_md.load(disable=['parser', 'ner'])

text_list_pos = df_sampled_pos['content']
text_list_neg = df_sampled_neg['content']
text_list_neu = df_sampled_neu['content']

tokenized_comments_pos = lemmatization(text_list_pos, nlp)
tokenized_comments_neg = lemmatization(text_list_neg, nlp)
tokenized_comments_neu = lemmatization(text_list_neu, nlp)

 
# convert to document term frequency:
dictionary_pos = corpora.Dictionary(tokenized_comments_pos)
doc_term_matrix_pos = [dictionary_pos.doc2bow(rev) for rev in tokenized_comments_pos]
dictionary_neu = corpora.Dictionary(tokenized_comments_neu)
doc_term_matrix_neu = [dictionary_neu.doc2bow(rev) for rev in tokenized_comments_neu]
dictionary_neg = corpora.Dictionary(tokenized_comments_neg)
doc_term_matrix_neg = [dictionary_neg.doc2bow(rev) for rev in tokenized_comments_neg]
 
# Creating the object for LDA model using gensim library
LDA = gensim.models.ldamodel.LdaModel


n_topics = 20
# Build LDA model
lda_model_pos = LDA(corpus=doc_term_matrix_pos, id2word=dictionary_pos,
                num_topics= n_topics, random_state=100,
                chunksize=1000, passes=50, iterations=100)
lda_model_neu = LDA(corpus=doc_term_matrix_neu, id2word=dictionary_neu,
                num_topics= n_topics, random_state=100,
                chunksize=1000, passes=50, iterations=100)
lda_model_neg = LDA(corpus=doc_term_matrix_neg, id2word=dictionary_neg,
                num_topics= n_topics, random_state=100,
                chunksize=1000, passes=50, iterations=100)


# create wordclouds
cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]

cloud = WordCloud(stopwords=stop_words,
                  background_color='white',
                  width=2500,
                  height=1800,
                  max_words=1000,
                  colormap='tab10',
                  color_func=lambda *args, **kwargs: cols[i],
                  prefer_horizontal=1.0)

topics_pos = lda_model_pos.show_topics(num_topics= n_topics, formatted=False)
topics_neu = lda_model_neu.show_topics(num_topics= n_topics, formatted=False)
topics_neg = lda_model_neg.show_topics(num_topics= n_topics, formatted=False)

topics_list_pos = []
topics_list_neu = []
topics_list_neg = []

for i in range(n_topics):
    topics_list_pos += topics_pos[i][1]
    topics_list_neu += topics_neu[i][1]
    topics_list_neg += topics_neg[i][1]

topic_words_pos = dict(topics_list_pos)
topic_words_neu = dict(topics_list_neu)
topic_words_neg = dict(topics_list_neg)

topic_words_total = [topic_words_pos, topic_words_neu, topic_words_neg]

fig, axes = plt.subplots(1, 3, figsize=(10,10), sharex=True, sharey=True)

for i, ax in enumerate(axes.flatten()):
    fig.add_subplot(ax)
    cloud.generate_from_frequencies(topic_words_total[i])
    plt.gca().imshow(cloud)
    plt.gca().axis('off')


plt.subplots_adjust(wspace=0, hspace=0)
plt.axis('off')
plt.margins(x=0, y=0)
plt.tight_layout()
plt.show()