import gensim
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import brown
nltk.download('brown')
import sys
sys.path.append('/Users/simona/Desktop/TESI/FINAL/')
import Lemmatizer

import datetime

# specific domain corpus mixed to brown corpus!
# specific domain model vocabs: 5163 (9236 sentences)
# adding the brown corpus vocabs: 32052
#(total number of sentences: 66576)

#training a Gensim w2v model using both corpus


# Read the data into a list of strings.
def read_data(filename, path):
  """Extract the first file as a list of words."""
  file = open(path+filename, encoding="utf-8")
  data = file.read().split("\n")
  file.close()
  sentences = []
  for sent in data:
    words = sent.split(" ")
    sentences.append(words)
  sentences = Lemmatizer.clear_sentences_lol(sentences)
  return sentences

path = '/Users/simona/Desktop/TESI/FINAL/CORPUS/'
filename ='CleanedText_sentences.txt'


documents = read_data(filename, path)
print ("lenght: ", len(documents), "Adding Brown dict...")

documents += Lemmatizer.clear_sentences_lol(brown.sents())
#documents += brown.sents()

print ("Added\n New lenght: ", len(documents))

#print ("First sentences from specific corpus: \n ", documents[:5], "\nSome sentences from Brown corpus: \n", documents[-5:])

t1 = datetime.datetime.now()
print ("Time: ", t1)

model = gensim.models.Word2Vec(documents, #lista di sentneces?
                               sg=1, #skip-gram over cbow
                               size=128, #lenght of word vector
                               alpha=0.025, #learning rate
                               window=5, #The maximum distance between the target word and its neighboring word
                               negative=10, #skip-gram with negative sampling, with 32 noise words
                               hs=0, #using the softmax with negative sampling
                               batch_words=10, #batch size
                               sample=1e-1, # threshold for configuring which higher-frequency words are randomly downsampled
                               iter=5, #number of epochs
                               sorted_vocab=1, #sorting the vocabulary with respect to high frequency words
                               compute_loss=True, #to have estimation of a loss  
                               min_count=1, #Minimium frequency count of words (the others are ignored)
                               workers=1) # level of parallelization. We would need more power to get more than 1 worker

print("model created")
t2 = datetime.datetime.now()
print ("Total time required:", t2-t1)
print ("Initial Loss value: ", model.get_latest_training_loss())

t1= datetime.datetime.now()
model.train(documents, compute_loss=True, total_examples=len(documents), epochs=model.epochs)
print ("model trained: ", model)
t2 = datetime.datetime.now()
print ("Total time required:", t2-t1)

model.save("gensim_w2v8.bin")

print ("Final Loss value: ", model.get_latest_training_loss())
print ("Similarity score between words accident, incident", model.wv.n_similarity("accident", "incident"))
print ("Similarity score between words accident, incident", model.wv.n_similarity("water", "human"))
print ("Similarity score between words accident, incident", model.wv.n_similarity("pear", "accident"))

#print("model accuracy: ", model.accuracy('questions-words.txt', case_insensitive=True))
#model.wv.save_word2vec_format('prova.txt', binary=False)






