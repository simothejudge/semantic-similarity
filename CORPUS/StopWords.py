import nltk
from nltk.corpus import stopwords

#write a new clean file
f = open("StopWords.txt","w+")

# some of the classical stopwords are not to be avoided
wanted = {"any", "where", "only", "don't", "are",
                      "been", "do", "wouldn't", "couldn't", "shouldn't", "before",
                      "can", "down", "doing", "hadn't", "below", "too", "having",
                      "does", "was", "after", "weren't", "during", "wasn't",
                      "against", "doesn't", "were", "aren't", "mustn't",
                      "isn't", "has", "haven't", "above", "did", "under", "few",
                      "being", "not", "is", "have", "over", "didn't", "same",
                      "had", "for", "hasn't", "later"}


stop_words = [e for e in stopwords.words('english') if e not in wanted]
stop_words.extend(["at", "to"])
s = " ".join(w for w in stop_words)
del stop_words
f.write(s)
f.close()
