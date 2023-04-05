# Retrieval Method
Introduction to Retrieval methods  in recommender systems and implementations for different approaches using TensorFlow. 


# Table of contents
1. [Introduction](#introduction)
2. [Retrieval Evaluation Metrics](#retrieval-evaluation-metrics)
3. [Resources](#resources)

# Introduction

Retrieval in recommender systems is the process of selecting a subset of items from a large collection that are relevant or interesting for a given user or query. Retrieval models aim to efficiently filter out items that the user is not likely to interact with, and provide a diverse and personalized set of candidates for further ranking or evaluation. Retrieval models can be based on different techniques, such as:

- Content-based filtering: These models use the features or attributes of the items and the user to compute a similarity score between them. For example, if a user likes movies of a certain genre, a content-based retrieval model might suggest other movies of the same genre.
- Collaborative filtering: These models use the interactions or ratings of other users who have similar preferences or behavior to the current user to compute a similarity score between items. For example, if a user and another user have both watched and enjoyed the same movies, a collaborative filtering retrieval model might suggest other movies that the other user has liked but the current user hasn't seen yet.
- Hybrid models: These models combine different techniques and sources of information to provide more accurate and effective retrieval than a single technique. For example, a hybrid retrieval model might use both content-based and collaborative filtering methods to generate suggestions that match both the user's personal taste and the popular trends among similar users.


# Retrieval Evaluation Metrics

Some evaluation metrics for retrieval models are:

- Precision: This metric measures the fraction of retrieved items that are relevant to the query. Precision is high when the retrieval model returns mostly relevant items and avoids irrelevant ones. Precision can be calculated as: precision = (number of relevant items retrieved) / (number of items retrieved).
- Recall: This metric measures the fraction of relevant items that are retrieved by the query. Recall is high when the retrieval model returns most of the relevant items and does not miss any. Recall can be calculated as: recall = (number of relevant items retrieved) / (number of relevant items in the collection).
- F-measure: This metric combines precision and recall into a single score that balances both aspects. F-measure is high when both precision and recall are high.
- Mean average precision (MAP): This metric averages the precision values at different ranks of the retrieved items, giving more weight to the higher ranks. MAP is high when the retrieval model returns relevant items at the top of the ranking and maintains a high precision throughout. MAP can be calculated as: MAP = (average of precision at rank k for all queries) / (number of queries).
- Normalized discounted cumulative gain (NDCG): This metric accounts for the position and relevance of the retrieved items, giving more weight to the higher ranks and higher relevance levels. NDCG is high when the retrieval model returns highly relevant items at the top of the ranking and maintains a high relevance throughout. NDCG can be calculated as: NDCG = (DCG / IDCG), where DCG is the discounted cumulative gain and IDCG is the ideal discounted cumulative gain.

# Resources 
1.  [Evaluating Information Retrieval Models](https://medium.com/@prateekgaurav/evaluating-information-retrieval-models-a-comprehensive-guide-to-performance-metrics-78aadacb73b4)
2.  [Tensorflow Recommenders](https://github.com/tensorflow/recommenders/tree/main/tensorflow_recommenders)

