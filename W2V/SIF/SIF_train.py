# we need to use TF and the text8 corpus for this

import gensim
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import brown
nltk.download('brown')
import sys
sys.path.append('/Users/simona/Desktop/TESI/FINAL/')
import Lemmatizer

# specific domain corpus mixed to brown corpus!
# specific domain model vocabs: 5163 (9236 sentences)
# adding the brown corpus vocabs: 32052
#(total number of sentences: 66576)

#training TF W2V WITH FREQUENCY IN TEXT

#error in line101


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

print ("Added\n New lenght: ", len(documents))

print ("First sentences from specific corpus: \n ", documents[:5], "\nSome sentences from Brown corpus: \n", documents[-5:])

model = gensim.models.Word2Vec(documents, #lista di sentneces
                               size=128, #lenght of word vector
                               window=5, #The maximum distance between the target word and its neighboring word
                               min_count=1, #Minimium frequency count of words (the others are ignored)
                               workers=5) # dunno

print("model created")

model.train(documents, total_examples=len(documents), epochs=10)


print ("model trained: ", model)
model.save("SIF_w2v.bin")


