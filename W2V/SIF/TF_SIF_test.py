#Relevant Results


import gensim
from nltk import word_tokenize
import sys
sys.path.append('/Users/simona/Desktop/TESI/FINAL/')
import Lemmatizer
import SHELL_Processing as shell
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import numpy as np
from prettytable import PrettyTable


model = gensim.models.Word2Vec.load("/Users/simona/Desktop/TESI/FINAL/W2V/Gensim/gensim_w2v.bin")

#testing variables:
test_sents = ["Strip are not compliant with ICAO standards",
              "The weather conditions of wind gusts to 21 kts during the landing, have hindered the correct landing on the runway",
              "The contact of the aircraft with the ground occurred ar a distance of 1000m from the runway head."
              "El Cairo is usually subject to high temperature",
              "The weather was particularly humid",
              "The sky was parttially covered, and the level of environmental luminosity was close to minimum",
              "The helicopter flew during night hours, causing spatial disorientation",
              "Bad weather conditions with strong rainstorm"]
shell_tag = "E"
embedding_size = 128

   

#check if the model has a representation and for which words
def has_vector_representation(doc):
    """check if at least one word of the document is in the
    word2vec dictionary"""
    cleaned = (word for word in doc if word in model.wv.vocab)
    #sent = " ".join(word for word in cleaned)
    #print(sent)
    return cleaned

# get the frequency for a word in the shell document
def get_word_frequency(word_text, tag):
    #one option is to get the frequency from our shell_tag document,
    #so that the relevant words (for the context) will have more impact
    count = shell.get_count(tag)
    #print (count, len(count))
    occurrance = 1
    for element in count:
        if element[0]== word_text:
            occurrance = element[1]
    return occurrance/(len(count)-1)

def sentence_to_vec(sentence_list, embedding_size: int, tag):
    sentence_set = []
    a = float = 1e-3
    for sentence in sentence_list:
        vs = np.zeros(embedding_size)  # add all word2vec values into one vector for the sentence
        sentence_length = len(sentence)
        vec_freq = list() #for each word in sent it gets the frequency value
        vec_words = list() #for each word it gets the vector
        for word in sentence.split():
            if word in model.wv.vocab:
                a_value = get_word_frequency(word, tag)  # smooth inverse frequency, SIF
                vec_freq.append(a_value)
                wv = model.wv[word]
                vec_words.append(wv)
                vs = np.add(vs, np.multiply(a_value, wv)) # vs += sif * word_vector
        #vs = np.average(vec_words, axis=embedding_size, weights=vec_freq) 
        vs = np.divide(vs, sentence_length)  # weighted average
        sentence_set.append(vs)  # add to our existing re-calculated set of sentences
    return sentence_set
"""
    # calculate PCA of this sentence set
    pca = PCA(n_components=embedding_size)
    pca.fit(np.array(sentence_set))
    u = pca.components_[0]  # the PCA vector
    u = np.multiply(u, np.transpose(u))  # u x uT

    # pad the vector?  (occurs if we have less sentences than embeddings_size)
    if len(u) < embedding_size:
        for i in range(embedding_size - len(u)):
            u = np.append(u, 0)  # add needed extension for multiplication below

    # resulting sentence vectors, vs = vs -u x uT x vs
    sentence_vecs = []
    for vs in sentence_set:
        sub = np.multiply(u,vs)
        sentence_vecs.append(np.subtract(vs, sub))

    #return sentence_vecs
"""

cleaned_sents=list()
for sent in test_sents:
    #lemmatize
    test_data = (word_tokenize(Lemmatizer.clear_text(sent)))
    #check vocabulary
    test_data = has_vector_representation(test_data)
    sent = " ".join(w for w in test_data)
    cleaned_sents.append(sent)

sents_vect = sentence_to_vec(cleaned_sents,embedding_size, shell_tag)
#cleaned_sents e sents_vect hanno stesso indice
for index, test_vect in enumerate(sents_vect):
    test_vect = test_vect.reshape(1,-1)
    cosine_treshold = 0.2

    key_hfs = dict() #a dictionary to link each hf to a key value (range)
    similarity = dict() #key=number of hf, value=similarity
    found_hf = None
    found_score = 0
    factors = shell.get_factors(shell_tag)

    for i, hf in enumerate(factors):
        key_hfs[i] = hf
        sents = hf.get_sents()
        for sentence in sents:
            if sentence == cleaned_sents[index]:
                found_hf=hf
                found_score = 1
                print("Found corrisponding HF with same sentence:\n", found_hf.print_hf())
            else:
                new_list = list()
                new_list.append(sentence)
                sent_vect_list = sentence_to_vec(new_list,embedding_size, shell_tag)
                sent_vect = sent_vect_list[0].reshape(1,-1)
                cos_sim = cosine_similarity(test_vect, sent_vect)
                #print("sentence:\t %s \n similarity score: %s\n" %(sentence, cos_sim))
                if hf.get_name() in similarity:
                    if cos_sim>similarity[hf.get_name()]: #the sentence found is more similar than the previous one
                        similarity[hf.get_name()]=cos_sim #updating the similarity value for the human factor
                else:
                    similarity[hf.get_name()]=cos_sim

    if found_hf == None:
        print("Target sentence: ", sent)
        print("factors and their score: \n")
        t = PrettyTable(['Index', 'HF Name', 'Similarity'])
        for i, hf in enumerate(similarity):
            if similarity[hf]>cosine_treshold:   #mimal requirement to be considered
                t.add_row([i, hf, similarity[hf]])

        table= t.get_string(sortby="Similarity", reversesort=True, end=5)
        print (table)
        #index = input('Select among the listed factors which one is the apporpriate:\n')
        #print("Factor selected: ", index)
        #found_hf = key_hfs[int(index)]
        #found_score = similarity[found_hf]

    else:
        if found_score!=1:
            print("No factors found\n Please, be sure to have inserted the correct target sentence and the relative SHEL tag\n If everything is in place, manually insert the HF in the SHEL_docs")

    for i,hf in enumerate(factors):
        if hf == found_hf:
            factors[i].add_sent(cleaned_sents[index])

    #shell.rewrite_file(factors, shell_tag)
    #print ("Human Factor associated:\n", found_hf.get_name(), "\t", found_score)

