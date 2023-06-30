import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.test.utils import datapath
from nltk import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
import codecs

#parameters
model_name="d2v2.model"
test_docs="test_docs.txt"
output_file="test_vectors.txt"

#inference hyper-parameters
start_alpha=0.01
infer_epoch=1000

#load model
model = Doc2Vec.load(model_name)
test_docs = [ x.strip().split() for x in codecs.open(test_docs, "r", "utf-8").readlines() ]

#infer test vectors
output = open(output_file, "w")
for d in test_docs:
    output.write( " ".join([str(x) for x in model.infer_vector(d, alpha=start_alpha, steps=infer_epoch)]) + "\n" )
output.flush()
output.close()

questions = 'questions-words2.txt'
evals = open(questions, 'r').readlines()
num_sections = len([l for l in evals if l.startswith(':')])
print('total evaluation sentences: {} '.format(len(evals) - num_sections))

def d2v_model_accuracy(model):

    accuracy = model.accuracy(questions, restrict_vocab=30000)
    
    sum_corr = len(accuracy[-1]['correct'])
    sum_incorr = len(accuracy[-1]['incorrect'])
    total = sum_corr + sum_incorr
    percent = lambda a: a / total * 100
    
    print('Total sentences: {}, Correct: {:.2f}%, Incorrect: {:.2f}%'.format(total, percent(sum_corr), percent(sum_incorr)))

 

# test the model accuracy
#d2v_model_accuracy(model)

l1 = "love tiger tiger book computer computer plane train telephone television media drug bread cucumber doctor professor student smart smart company stock stock stock stock stock fertility stock stock book bank wood money professor king king king bishop"
l2= "sex cat tiger paper keyboard internet car car communication radio radio abuse butter potato nurse doctor professor student stupid stock market phone cd jaguar egg egg live life library money forest cash cucumber cabbage queen rook rabbi"

l1 = word_tokenize(l1)
l2 = word_tokenize(l2)


l3=[]

for i in range(0,len(l1)):
    l3.append(model.n_similarity(l1[1], l2[i]))

for i, e in enumerate(l3): 
    print(i+1,":  ", e*10)

#test the model similarity
similarities = model.evaluate_word_pairs(datapath('wordsim353.tsv'))
print("similarities eval: ", similarities)
