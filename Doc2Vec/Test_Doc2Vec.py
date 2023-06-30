from gensim.models.doc2vec import Doc2Vec
from nltk.tokenize import word_tokenize
import sys
sys.path.append('/Users/simona/Desktop/TESI/FINAL/')
import SHELL_Processing as shell
import Lemmatizer
from prettytable import PrettyTable


# to measure the similarity scklearn library is considered the most effective. although, there are others cosine calculation that can be used
from sklearn.metrics.pairwise import cosine_similarity


model= Doc2Vec.load("d2v2.model")

#to find the vector of a document which is not in training data
test_sents =[ "The procedure for managing emergency landing of the aircraft was not effective",
              "The NOTAMS were deliveried with delate",
              "There was not appropriate risk analysis for assistance under dangerous conditions",
              "Absence of operative procedure to describe dispacement positions"]
shell_tag = "S"

for test_sent in test_sents:
    
    test_data = word_tokenize(Lemmatizer.clear_text(test_sent))
    print ("Test sentence:", test_sent )


    test_vect = model.infer_vector(test_data) #getting the sent vector
    test_vect = test_vect.reshape(1,-1) # resizing from 128 dim to 2 (need to calculate PCA)
    cosine_treshold = 0.02

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
                sent_vect = model.infer_vector(sentence)
                sent_vect = sent_vect.reshape(1,-1)
                cos_sim = cosine_similarity(test_vect, sent_vect)
                # print("sentence:\t %s \n similarity score: %s\n" %(sentence, cos_sim))
                if hf.get_name() in similarity:
                    if cos_sim>similarity[hf.get_name()]: #the sentence found is more similar than the previous one
                        similarity[hf.get_name()]=cos_sim #updating the similarity value for the human factor                   
                else:
                    similarity[hf.get_name()]=cos_sim

    if found_hf == None:
        print("factors and their score: \n")
        
        """
        for key,value in similarity.items():
            print(key, value)
        """
 
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
            factors[i].add_sent(" ".join(w for w in test_data))

#shell.rewrite_file(factors, shell_tag)
        
    
"""
# to find most similar doc using tags
similar_doc = model.docvecs.most_similar([v1])
print(similar_doc)

id_sents = [doc[0] for doc in similar_doc]
print (id_sents)

"""
