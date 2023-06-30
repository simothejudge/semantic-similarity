# the processor for the SHELL elements, read the shell docs, get the human factors and the sentences related, write new sentences
import os

import gensim
import nltk
from nltk import WordNetLemmatizer
from nltk.corpus import wordnet
import sys
#sys.path.append('/Users/simona/Desktop/TESI/FINAL/')
import Lemmatizer
import collections

#global variable
path = "SHELL_Sentences/SHELL/"
count = [['UNK', -1]] #initialized with unknown
shell_tag = ""
h_factors = []

class Human_Factor():
    def __init__(self, category, name, index, sentences):
        self.category = category
        self.name = name
        self.index = index
        self.sentences = Lemmatizer.clear_sentences(sentences) 
    def get_sents (self):
        return self.sentences
    def get_name(self):
        return "".join(self.name)
    def add_sent (self, sent):
        self.sentences.append(sent)
    def print_hf(self):
        sents_string = "\n".join(sent for sent in self.sentences[1:])
        string = "#"+self.category+"-->"+self.name+"-->"+self.index+"\n"+sents_string+"\n"
        return string

def get_factors(tag):
    # read the file of the the shell tag
    shell_tag = tag
    file = open(path+shell_tag+".txt", encoding="utf-8")
    text = file.read()
    sents = text.split("#")
    sents.pop(0)
    file.close()
    for sent in sents:
        first = sent.splitlines()[0]
        category, name, index = first.split("-->")
        sentences = sent.split("\n")
        sentences[0] = name
        sentences.pop(len(sentences)-1)
        hf = Human_Factor(category, name, index, sentences)
        h_factors.append(hf)
    return (h_factors)

def get_count(shell_tag): #the tag variable should be already set, so no need to pass it
    assert shell_tag != ""
    file = open(path+shell_tag+".txt", encoding="utf-8")
    text = file.read()
    words = Lemmatizer.clear_text_l(text)
    n_words = len(words)
    count.extend(collections.Counter(words).most_common(n_words - 1))
    file.close()
    return (count)

def rewrite_file (new_h_factors, shell_tag):
    assert shell_tag != ""
    file = open(path+shell_tag+".txt", "w+", encoding="utf-8")
    text = file.read()
    new = ""
    for hf in new_h_factors:
        new += "".join(hf.print_hf())
    file.write(new)
    file.close()
    return


# to test this module:
print(os.getcwd())
shell_tag = "LP"
hfs = get_factors(shell_tag)
for hf in hfs: 
    print ("Factor: \n Category: %s \n Name: %s \n Index: %s \n Sentences: %s \n"
           % (hf.category, hf.name, hf.index, hf.sentences))
hfs[3].add_sent("trial for testing")
rewrite_file(hfs, shell_tag)
print(get_count(shell_tag))

