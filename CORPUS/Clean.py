# corpus as a text document with separation of sentences, lemmatization and stopwords
# stopwords = taken from English dictionary and modified basing on our need
import os
import glob
import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

import datetime
write_path = "txt_output/"


# tag converter
def get_wordnet_pos(tag):

    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def clean_sentences(text):

    # copy stopwords
    file = open("CORPUS/StopWords.txt", encoding="utf-8")
    textSW = file.read()
    stopwords = [w for w in textSW.split(" ")]
    # print (stopwords)
    file.close()

    #tokenize
    tokens = nltk.sent_tokenize(text)
    del text #keeping light

    #write a new clean file
    new_file_name = "CleanedText_sentences.txt"
    file = open(write_path+new_file_name,"w+", encoding ="utf-8")
    sents = ""
    lemmatizer = WordNetLemmatizer()

    for token in tokens:
        tagged = nltk.pos_tag(nltk.word_tokenize(token))
        lemmas = [lemmatizer.lemmatize(word[0].lower(), pos = get_wordnet_pos(word[1])) for word in tagged
                         if (word[0].isalpha() and word[0].lower() not in stopwords)]          
        sent = " ".join(w for w in lemmas)
        sents += sent+"\n"

    del stopwords
    
    file.write(sents)
    print('New file "CleanedText_sentences.txt" created')
    file.close()
    del sents
    return new_file_name

def clean_raw_text(text):

    # copy stopwords
    file = open("CORPUS/StopWords.txt", encoding="utf-8")
    textSW = file.read()
    stopwords = [w for w in textSW.split(" ")]
    # print (stopwords)
    file.close()

    #tokenize
    tokens = word_tokenize(text)
    del text #keeping light
    
    tagged = nltk.pos_tag(tokens)
    
    lemmatizer = WordNetLemmatizer()
    lemmas = [lemmatizer.lemmatize(word[0].lower(), pos = get_wordnet_pos(word[1])) for word in tagged
              if (word[0].isalpha() and word[0].lower() not in stopwords)]

    del tagged 
    del stopwords #keeping light
    
    raw_text = " ".join(lemma for lemma in lemmas)
    
    #write a new clean file
    new_file_name = "CleanedText_raw.txt"
    f = open(write_path+new_file_name ,"w", encoding='utf-8')
    f.write(raw_text)
    print('New file "CleanedText_raw.txt" created')
    del raw_text
    f.close()
    return new_file_name

def get_text():
    print("Clensing of the Corpus in raw text...")

    path= r"all_txt_from_MAIN/"
    my_files = glob.glob(path+'*.txt')
    print("List of texts to add to the corpus: ")
    text = ""
    for filename in my_files:
        print("\t- ", filename)
        file = open(filename, encoding="utf-8")
        tt = file.read().lower()
        print('lenght: ', len(tt))
        text += tt
        file.close()

    print("Total lenght: ", len(text))
    return text



"""
t1 = datetime.datetime.now()
print ("Time: ", t1)
clean_raw_text()
t2 = datetime.datetime.now()
print ("Total time required:", t2-t1)
t1 = datetime.datetime.now()
clean_sentences()
t2 = datetime.datetime.now()
print ("Total time required:", t2-t1)
"""

"""
to test this module:
clean_raw_text()
clean_sentences()


"""

text = get_text()
clean_raw_text(text)
clean_sentences(text)

#filename = 'MAIN(book thesis ICAO Draft).txt'
#file = open(filename, encoding="utf-8")
#print(file)