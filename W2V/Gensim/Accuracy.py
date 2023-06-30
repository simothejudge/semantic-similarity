import gensim
from gensim.models import Word2Vec
from gensim.test.utils import datapath
from nltk import word_tokenize
# read the evaluation file, get it at:
# https://word2vec.googlecode.com/svn/trunk/questions-words.txt
questions = 'questions-words2.txt'
evals = open(questions, 'r').readlines()
num_sections = len([l for l in evals if l.startswith(':')])
print('total evaluation sentences: {} '.format(len(evals) - num_sections))

def w2v_model_accuracy(model):

    accuracy = model.accuracy(questions)
    
    sum_corr = len(accuracy[-1]['correct'])
    sum_incorr = len(accuracy[-1]['incorrect'])
    total = sum_corr + sum_incorr
    percent = lambda a: a / total * 100
    
    print('Total sentences: {}, Correct: {:.2f}%, Incorrect: {:.2f}%'.format(total, percent(sum_corr), percent(sum_incorr)))

 
model= gensim.models.Word2Vec.load("gensim_w2v.bin")
#model= gensim.models.Word2Vec.load("GoogleNews-vectors-negative300.bin")
    
# test the model accuracy
w2v_model_accuracy(model)
#google = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
#w2v_model_accuracy(google)

l1 = "love tiger tiger book computer computer plane train telephone television media drug bread cucumber doctor professor student smart smart company stock stock stock stock stock fertility stock stock book bank wood money professor king king king bishop"
l2= "sex cat tiger paper keyboard internet car car communication radio radio abuse butter potato nurse doctor professor student stupid stock market phone cd jaguar egg egg live life library money forest cash cucumber cabbage queen rook rabbi"

l1 = word_tokenize(l1)
l2 = word_tokenize(l2)


l3=[]
for i in range(0,len(l1)):
    l3.append(model.wv.n_similarity(l1[1], l2[i]))

for i, e in enumerate(l3): 
    print(i+1,":  ", e*10)

#test the model similarity
similarities = model.wv.evaluate_word_pairs(datapath('wordsim353.tsv'))
print("similarities eval: ", similarities)
