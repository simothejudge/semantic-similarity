# Natural Language Processing for the Identification of Human Factors in Aviation Accident Causes (2018)

## A Semantic Similarity Exploration

Master Thesis in Management and Information Engineering at the Polytechnic of Turin (September 2018). 

The published article can be found here: 
[CIRRELT-2020-36.pdf](https://github.com/simothejudge/SemanticSimilarity/files/11914178/CIRRELT-2020-36.pdf)

## Abstarct
Accidents in aviation are rare events. From them, aviation safety management systems take fast and effective remedy actions by performing the analysis of the root causes of accidents, most of which are proven to be human factors. Since the current standard relies on the manual classification performed by trained staff, there are no technical standards already defined for automated human factors identification. This paper considers this issue, proposing machine learning techniques by leveraging the state-of-the-art technologies of Natural Language Processing. The techniques are then adapted to the SHEL standard accident causality model and tested on a set of real accidents. Computational results show the accuracy and effectiveness of the proposed methodology, which leads to a possible reduction of time and costs by up to 30%.


## The problem
The Goal of the Safety Management System is to minimize accident risk, by drafting safety recommendations. 
This is done following an experience-based approach. So whenever a new accident happens, it is necessary to conduct deep investigations to understand the dynamics of the accident, and to identify which are the causal factors contributing to the accident, called human factors. 
The person responsible for this is the safety manager or investigator. This process is not trivial and this is why an automated system should be used to support the investigator's job. 

So, what happens when an accident occurs? 
All the events are collected in an organized and ordered way by the investigator, in a report.
Then, an automatic model called SHEL tagger is used, to tag each relevant event in the report under 5 possible categories: these are H, S, E, LO, LP. To classify HUMAN factors the HFACS model is used. It’s a highly structured system that identifies Human Factors with a complete and detailed taxonomy, using nanocodes. From these codes, the investigator can directly extract safety recommendations. 
The challenges in this process are the huge number of available Human Factors classes (which are more than 400). 
The HFACS Classification system itself is also very difficult to deal with and requires deep knowledge and experience in the field.

## The proposed solution
After experimenting with different approaches we decided to build an Intelligent System which receives as input the report tagged with shel categories, understands the meaning of the natural language text, and returns as output the closest possible human factors connected to these events. The innovation was to build a system that can “self” learn from each new report processed, increasing its knowledge base and therefore making more accurate hf identifications. For this reason, the first step of the project was to organize all the existing reports with their human factors classification in an initial knowledge base.

The key technologies used for this project are Machine Learning Algorithms and Artificial Neural Networks (in the field of Natural Language Processing). Additionally, we leveraged the Distributional Semantic theory. The basic assumption of this theory is that the semantic meaning of a word is strictly related to its context, meaning that words with similar semantic meanings share a similar context. By combining this theory with NN, we can try to represent words in a machine-friendly way, through vectors, carrying semantic information. This process of multi-dimensional vector representation of unstructured data like text is called Embedding and it is a core foundation of the neural language theory.

We implemented 3 different models based on three different neural networks to be able to evaluate which algorithms, parameters, and data work best for our specific case.


<img width="1129" alt="Screenshot 2023-06-30 at 16 16 56" src="https://github.com/simothejudge/semantic-similarity/assets/37406249/a427bfcd-4cee-452e-abb7-a169f532db67">


## Data Collection and text Cleansing

Neural Language Models are trained over a Corpus: a large amount of raw text including books, articles, papers, online resources like Wikipedia, and Tweets. To ensure adequate accuracy in the Aviation-related field, a Specific Domain Corpus was created and integrated into online available corpora, of more generic topics. In this first step the training corpus was built, mixing generic Corpora like Brown and Text8.
The objective of this step is to build the corpus for training the neural networks. One relevant factor was that we built an aviation-specific domain corpus that we integrated with a generic online corpus to have more data. This increased the accuracy of the learning and the better understanding of Aviation-related terms.  

<img width="996" alt="Screenshot 2023-06-30 at 16 03 43" src="https://github.com/simothejudge/semantic-similarity/assets/37406249/7da03d27-d0f1-48db-8c26-8077569c4d76">


## Word and Sentence Embedding
In these steps, we used - for the first and the second models - the Word2vec paradigm, in order to get a good representation of words, and then we used a weighted average of the word vectors composing the sentence, in order to get a vector representation of a sentence.
For the third model, we directly used Doc2vec: a model that automatically returns the vector representation of the sentence (sentence embedding).
 
 <img width="1110" alt="Screenshot 2023-06-30 at 16 27 06" src="https://github.com/simothejudge/semantic-similarity/assets/37406249/9261012d-8797-4dd7-9371-2004169f26de">


## Sentence Similarity
In the 4th step, the similarity score was obtained by computing the cosine of the angle between the two sentence vectors, considering that similar sentences share a similar direction, because of the distributional semantic hypothesis.
 

## Results
To evaluate these results we used 4 REPORTS tagged and classified with SHELL and HFACS.

The comparison is based on:
	1.	the number of times the correct HF was in the top-5 list. 
	2.	the average position of the correct HF in the top 5 list.
	3.	the number of times the correct HF was in the top-5 but not at the top position (the difference between its score and the top position score).  

<img width="1100" alt="Screenshot 2023-06-30 at 16 28 47" src="https://github.com/simothejudge/semantic-similarity/assets/37406249/4e1a1e48-1ee5-42ac-814c-a86dc1374d23">
 
TF is the one performing better on total %, basically because of the usage of a larger corpus for training. Although, the Gensim model performs better on the other factors. This means that Gensim is slightly worse at getting the HF in the top-5 list, but if the HF is included in the list, it’s more likely that it is identified in the top position. 
In addition, the general accuracy of Gensim is worse by only 2 points although the model is simpler and easier to implement. This is why we recommended the Gensim model for our specific scenario.
 
Overall, the results of our prototype fully meet the goals of the research. We presented an alternative way to support the investigator in identifying human factors by processing unstructured text (language). 
But what is more interesting is that we could leverage the outcome of our solution and use it as a more structured input dataset to train a more sophisticated neural network, that can perform autonomously sentence semantic similarity, without the need for word and sentence embedding. 
