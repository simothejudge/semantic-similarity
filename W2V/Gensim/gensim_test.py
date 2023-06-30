#Relevant Results
# CORPUS DI GENSIM (MIO+BROWN)

import gensim
from nltk import word_tokenize
import os
path = os.getcwd()

import Lemmatize as Lemmatizer
import SHELL_Processing as shell
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from prettytable import PrettyTable
#import operator

model= gensim.models.Word2Vec.load("gensim_w2v.bin")

#testing variables:
test_sent = "The Mmarshaller did not contact the crew, leaving and returning at his place"
shell_tag = "LP"

# function that get the mean of word vectors
def mean_vector(doc):
    # remove out-of-vocabulary words
    doc = [word for word in doc if word in model.wv.vocab]
    return np.mean(model[doc], axis=0)

#check if the model has a representation and for which words
def has_vector_representation(doc):
    """check if at least one word of the document is in the
    word2vec dictionary"""
    return not all(word not in model.wv.vocab for word in doc)

test_data = (word_tokenize(Lemmatizer.clear_text(test_sent)))


print ("Test sentence:", test_sent )

assert has_vector_representation(test_data)
test_vect = mean_vector(test_data)
test_vect = test_vect.reshape(1,-1)

cosine_treshold = 0.5

#found_hf = shell.Human_Factor(None, None, None, None)
key_hfs = dict() #a dictionary to link each hf to a key value (range)
similarity = dict() #key=number of hf, value=similarity
found_hf = None
found_score = 0
factors = shell.get_factors(shell_tag)

for i, hf in enumerate(factors):
    key_hfs[i] = hf
    sents = hf.get_sents()
    for sentence in sents:
        if sentence == " ".join(w for w in test_data):
            found_hf=hf
            found_score = 1
            print("Found corrisponding HF with same sentence:\n", found_hf.print_hf())
        else:
            sent_vect = mean_vector(word_tokenize(Lemmatizer.clear_text(sentence)))
            sent_vect = sent_vect.reshape(1,-1)
            cos_sim = cosine_similarity(test_vect, sent_vect)
            #print("sentence:\t %s \n similarity score: %s\n" %(sentence, cos_sim))
            if hf in similarity:
                if cos_sim>similarity[hf]: #the sentence found is more similar than the previous one
                    similarity[hf]=cos_sim #updating the similarity value for the human factor
            else:
                similarity[hf]=cos_sim

if found_hf == None:
    print("factors and their score: \n")
    t = PrettyTable(['Index', 'HF Name', 'Similarity'])
    for i, hf in enumerate(similarity):
        if similarity[hf]>cosine_treshold:   #mimal requirement to be considered
            t.add_row([i, hf.get_name(), similarity[hf]])
    
    table= t.get_string(sortby="Similarity", reversesort=True, end=5)
    print (table)
    index = input('Select among the listed factors which one is the apporpriate:\n')
    print("Factor selected: ", index)
    found_hf = key_hfs[int(index)]
    found_score = similarity[found_hf]
else:
    if found_score!=1:
        print("No factors found\n Please, be sure to have inserted the correct target sentence and the relative SHEL tag\n If everything is in place, manually insert the HF in the SHEL_docs")

for i,hf in enumerate(factors):
    if hf == found_hf:
        factors[i].add_sent(" ".join(w for w in test_data))

#shell.rewrite_file(factors, shell_tag)
print ("Human Factor associated:\n", found_hf.get_name(), "\t", found_score)


"""
#plot:

x =[]
y = factors

for doc in model.: #look up each doc in model
    x.append(document_vector(model.wv, doc)) #--> error: need at least one array to concatenate, to be fixed
X = np.array(x) #list to array 

np.save('documents_vectors.npy', X)  #np.savetxt('documents_vectors.txt', X)
np.save('labels.npy', y)             #np.savetxt('labels.txt', y)

np.savetxt('documents_vectors.txt', X)
np.savetxt('labels.txt', y)

print (X.shape, len(y))


pca = PCA(n_components=2)
x_pca = pca.fit_transform(X)

plt.figure(1, figsize=(30, 20),)
plt.scatter(x_pca[:, 0], x_pca[:, 1],s=100, c=y, alpha=0.2)

from sklearn.manifold import TSNE
X_tsne = TSNE(n_components=2, verbose=2).fit_transform(X)
plt.figure(1, figsize=(30, 20),)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1],s=100, c=y, alpha=0.2)
plt.show()


"""
