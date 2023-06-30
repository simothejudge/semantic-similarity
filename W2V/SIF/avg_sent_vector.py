#################################################################################
###     Calculate the average vector for all words in every sentence/document ###
###     and use cosine similarity between vectors.                            ###
#################################################################################
#  Copyright 2016 Peter de Vocht
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import numpy as np
from sklearn.decomposition import PCA
import tensorflow as tf
from typing import List
import sys
sys.path.append('/Users/simona/Desktop/TESI/FINAL/')
import Lemmatizer
import SHELL_Processing as shell


#loading the TF model
export_dir = "/Users/simona/Desktop/TESI/FINAL/W2V/log_dir/mymodel.ckpt"

#testing variables:
test_sent = "blablabla"
shell_tag = "LP"

from tensorflow.python.tools import inspect_checkpoint as chkp
with tf.Session(graph=tf.Graph()) as sess:
    chkp.print_tensors_in_checkpoint_file(export_dir, tensor_name='', all_tensors=True)
    #print(tf.saved_model.loader.maybe_saved_model_directory(export_dir))

    new_saver = tf.train.import_meta_graph(export_dir+'.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint(export_dir+'.meta'))
    all_vars = tf.get_collection('vars')
    for v in all_vars:
        v_ = sess.run(v)
        print(v_)



# an embedding word with associated vector
class Word:
    def __init__(self, text):
        self.text = text
        #self.vector = model.


# a sentence, a list of words
class Sentence:
    def __init__(self, word_list):
        self.word_list = word_list

    # return the length of a sentence
    def len(self) -> int:
        return len(self.word_list)


# get the frequency for a word in the shell document
def get_word_frequency(word_text):
    #one option is to get the frequency from our shell_tag document,
    #so that the relevant words (for the context) will have more impact
    count = shell.get_count("LP")
    print (count, len(count))
    occurrance = 0
    for element in count:
        if element[0]== word_text:
            occurrance = element[1]
    return occurrance/(len(count)-1)



#print(get_word_frequency("ciao"))

# A SIMPLE BUT TOUGH TO BEAT BASELINE FOR SENTENCE EMBEDDINGS
# Sanjeev Arora, Yingyu Liang, Tengyu Ma
# Princeton University
# convert a list of sentence with word2vec items into a set of sentence vectors
def sentence_to_vec(sentence_list: List[Sentence], embedding_size: int, a: float=1e-3):
    sentence_set = []
    for sentence in sentence_list:
        vs = np.zeros(embedding_size)  # add all word2vec values into one vector for the sentence
        sentence_length = sentence.len()
        for word in sentence.word_list:
            a_value = a / (a + get_word_frequency(word.text))  # smooth inverse frequency, SIF
            vs = np.add(vs, np.multiply(a_value, word.vector))  # vs += sif * word_vector

        vs = np.divide(vs, sentence_length)  # weighted average
        sentence_set.append(vs)  # add to our existing re-calculated set of sentences

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

    return sentence_vecs



