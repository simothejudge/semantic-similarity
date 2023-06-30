"""
source: https://medium.com/@mishra.thedeepak/doc2vec-in-a-simple-way-fa80bfe81104

"""
# like in w2v with gensim, I'm using my specific-domain corpus and the Brown corpus

# !pip install gensim
import os


#Import all the dependencies
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from nltk.corpus import brown
import nltk
nltk.download('brown')

import sys
print(os.getcwd())
import Lemmatizer
import datetime
import numpy as np

# Read the data into a list of strings.
def read_data(filename, read_path):
  """Extract the first file as a list of words."""
  file = open(read_path+filename, encoding="utf-8")
  data = file.read().split("\n")
  file.close()
  print ("total sentences: ", len(data))
  return Lemmatizer.clear_sentences(data)


def compute_sim(model, sents1, sent2):
  print (sent2)
  vec2 = model.infer_vector(sent2.split())
  for sent in sents1:
    print ("sentence: ", sent)
    tokens = sent.split()
    new_vector = model.infer_vector(tokens)
    sim = np.dot(new_vector, vec2)
    #cosine_similarity(new_vector.reshape(-1,1), vec2.reshape(-1,1))
    print (sim)


#print ("Similarity score between words banana, dog", model.wv.n_similarity("human error approach is the best", "financial liabilities can help a company growing"))
def get_sentence (tag):
  return tagged_data[tag]

print(os.getcwd())
path = "txt_output/"
filename ='CleanedText_sentences.txt'

data = read_data(filename, path)

# add the brown corpus
temp_brown = Lemmatizer.clear_sentences_lol(brown.sents())
for i, element in enumerate(temp_brown):
  sent = " ".join(word for word in element)
  temp_brown[i] = sent
data += temp_brown

del temp_brown

print ("total lentgh final length ", len(data))


tagged_data = [TaggedDocument(words=word_tokenize(_d), tags=[str(i)]) for i, _d in enumerate(data)]
print ("Tagged sentences")

max_epochs = 20 #max epoch
vec_size = 128 #dimensionality of sent vect
alpha = 0.025 #learning rate

t1 = datetime.datetime.now()
print ("Time: ", t1)
model = Doc2Vec(size=vec_size,
                alpha=alpha, 
                min_alpha=0.025,
                min_count=1,#Ignores all words with total frequency lower than this
                window=10, #The maximum distance between the current and predicted word within a sentence.
                dm =0, #Define the training algo, here it is set to PV-DBOW
                sample=1e-1,
                epochs=10,
                hs=1, #using hierical softmax
                negative = 5,
                compute_loss=True, #to have estimation of a loss 
                dm_concat=1, #use concatenation of context vectors rather than sum/average
                )
  
model.build_vocab(tagged_data)
t2 = datetime.datetime.now()
print ("model built")
print ("Total time required:", t2-t1)

sents1 = data[:5]
#sent2 = "We need to find the recepie for cooking pasta"
sent2="unfortunately many today aviation safety personnel have little formal education human factor aviation psychology"
compute_sim(model, sents1, sent2)

t1= datetime.datetime.now()
for epoch in range(max_epochs):
  if epoch % 5 == 0:
    compute_sim(model, sents1, sent2)
    
  print('iteration {0}'.format(epoch))
  model.train(tagged_data,
                total_examples=model.corpus_count,
                epochs=model.iter)
  
  # decrease the learning rate
  model.alpha -= 0.0002
  # fix the learning rate, no decay
  model.min_alpha = model.alpha
  if epoch == (max_epochs-1):
       compute_sim(model, sents1, sent2)




print ("model trained")
t2 = datetime.datetime.now()
print ("Total time required:", t2-t1)

model.save("d2v.model")

print("Model Saved")
print ("Similarity score between words accident, incident", model.wv.n_similarity("accident", "incident"))
print ("Similarity score between words cat, dog", model.wv.n_similarity("cat", "dog"))
print ("Similarity score between words banana, dog", model.wv.n_similarity("banana", "dog"))
print ("Similarity score between words banana, accident", model.wv.n_similarity("banana", "accident"))
print ("Similarity score between words banana, apple", model.wv.n_similarity("banana", "apple"))
print ("Similarity score between words dog, animal", model.wv.n_similarity("dog", "animal"))
print ("Similarity score between words dog, wolf", model.wv.n_similarity("dog", "wolf"))
print ("Similarity score between words obstruction, limitation", model.wv.n_similarity("obstruction", "limitation"))