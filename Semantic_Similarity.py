import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns


module_url = "https://tfhub.dev/google/universal-sentence-encoder/1" #@param ["https://tfhub.dev/google/universal-sentence-encoder/1", "https://tfhub.dev/google/universal-sentence-encoder-large/1"]

# Import the Universal Sentence Encoder's TF Hub module
embed = hub.Module(module_url)


# Compute a representation for each message, showing various lengths supported.
word = "supervision"
sentence = "The controller did not notice the error"
paragraph = (
    "the clearance is corrected by the controller",
    "The pilot did not realize the wrong readback ",
    "After 3 hours, the pilot noticed again the controller")
messages = [word, sentence, paragraph]

# Reduce logging output.
tf.logging.set_verbosity(tf.logging.ERROR)

with tf.Session() as session:
  session.run([tf.global_variables_initializer(), tf.tables_initializer()])
  message_embeddings = session.run(embed(messages))

  for i, message_embedding in enumerate(np.array(message_embeddings).tolist()):
    print("Message: {}".format(messages[i]))
    print("Embedding size: {}".format(len(message_embedding)))
    message_embedding_snippet = ", ".join(
        (str(x) for x in message_embedding[:3]))
    print("Embedding: [{}, ...]\n".format(message_embedding_snippet))
"""

#semantic sim task example
def plot_similarity(labels, features, rotation):
  corr = np.inner(features, features)
  sns.set(font_scale=1.0)
  g = sns.heatmap(
      corr,
      xticklabels=labels,
      yticklabels=labels,
      vmin=0,
      vmax=1,
      cmap="YlOrRd")
  g.set_xticklabels(labels, rotation=rotation)
  g.set_title("Semantic Textual Similarity")
  plt.show()


def run_and_plot(session_, input_tensor_, messages_, encoding_tensor):
  message_embeddings_ = session_.run(
      encoding_tensor, feed_dict={input_tensor_: messages})
  plot_similarity(messages_, message_embeddings_, 90)


messages = [
    "The pilot did not realize the wrong readback",
    "operational supervision",
    "lack of vigilance",
    "supervision",
    "clear",
    "not realizing",
    "the controller did not notice the error",
    "the clearance is corrected by the controller",
    "inadequate control",
    "lack of reports"
]

messages2 = [
    "The pilot did not realize the wrong readback",
    "workspace layout",
    "APU",
    "aircraft model",
    "passenger capacity",
    "suitable layout",
    "weather visibility",
    "horizon undefinable",
    "mirrored sea",
    "visible moon"
]


similarity_input_placeholder = tf.placeholder(tf.string, shape=(None))

similarity_message_encodings = embed(similarity_input_placeholder)

with tf.Session() as session:
  session.run(tf.global_variables_initializer())
  session.run(tf.tables_initializer())
  run_and_plot(session, similarity_input_placeholder, messages2,
               similarity_message_encodings)
"""
