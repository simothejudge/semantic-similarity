from nltk import WordNetLemmatizer
from nltk.corpus import wordnet
import nltk.tokenize
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

import os



# general stopwords, lemmatize
file = open("CORPUS/StopWords.txt", encoding="utf-8")
textSW = file.read()
stopwords = [w for w in textSW.split(" ")]
file.close()
lemmatizer = WordNetLemmatizer()

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

#get list of sentences (strings) and returns cleaned sents
def clear_sentences(sentences):
    for i, line in enumerate(sentences):
        words = [w.lower() for w in nltk.tokenize.word_tokenize(line)
             if w.isalpha() and w not in stopwords]
        tagged = nltk.pos_tag(words)
        lemmas = [lemmatizer.lemmatize(word[0], pos = get_wordnet_pos(word[1]))
              for word in tagged]
        sentences[i] = " ".join(lem for lem in lemmas)
    return sentences

# when sents are organized as lists of list of words (not string)
def clear_sentences_lol(sentences):
    sents = []
    for sent in sentences:
        words = [w.lower() for w in sent
             if w.isalpha() and w not in stopwords]
        tagged = nltk.pos_tag(words)
        lemmas = [lemmatizer.lemmatize(word[0], pos = get_wordnet_pos(word[1])) for word in tagged]
        sents.append(lemmas)
    return sents


# get raw text and returns cleaned text
def clear_text(text):
    words = [w.lower() for w in nltk.tokenize.word_tokenize(text)
             if w.isalpha() and w not in stopwords]
    tagged = nltk.pos_tag(words)
    lemmas = [lemmatizer.lemmatize(word[0], pos = get_wordnet_pos(word[1]))
              for word in tagged]
    text = " ".join(lem for lem in lemmas)
    return text

# get raw text and returns cleaned list of words
def clear_text_l(text):
    words = [w.lower() for w in nltk.tokenize.word_tokenize(text)
             if w.isalpha() and w not in stopwords]
    tagged = nltk.pos_tag(words)
    lemmas = [lemmatizer.lemmatize(word[0], pos = get_wordnet_pos(word[1]))
              for word in tagged]
    return lemmas



"""
# testing this:
text = "Python comes with a library of standard modules, described in a separate document, the Python Library Reference (“Library Reference” hereafter). Some modules are built into the interpreter; these provide access to operations that are not part of the core of the language but are nevertheless built in, either for efficiency or to provide access to operating system primitives such as system calls. The set of such modules is a configuration option which also depends on the underlying platform. For example, the winreg module is only provided on Windows systems. One particular module deserves some attention: sys, which is built into every Python interpreter. The variables sys.ps1 and sys.ps2 define the strings used as primary and secondary prompts:"
print("text to clean: \n", text)
sents = clear_sentences(text.split("."))
print ("Cleaned: \n", sents)
"""

