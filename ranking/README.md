# Ranking Method
Introduction to Learning to rank methods and implementations for different approaches using TensorFlow. 


# Table of contents
1. [Introduction](#introduction)
2. [Types of Ranking Method](#types-of-manking-method)
3. [Learning to Rank](#learning_to_rank)
	* [Pointwise Ranking](#pointwise-ranking)
	* [Pairwise Ranking](#pairwise-ranking)
	* [Listwise Ranking](#listwise-ranking)
4. [Ranking Evaluation Metrics](#ranking-evaluation-metrics)
5. [Resources](#resources)
# Introduction

Ranking methods are techniques that assign a score or a rank to each item in a set based on some criteria or preferences. Ranking methods are widely used in various domains such as information retrieval, machine learning, sports, and social choice.

# Types of Ranking Method
There are different types of ranking methods depending on the input data, the output format, and the objective function. Some common ranking methods are:

- **Vector space models**: These methods compute a vector representation for each item and then measure the similarity between items using a distance metric such as cosine similarity. For example, if you want to rank web pages based on a query, you can use a vector space model to compute the relevance score of each web page as the cosine similarity between its vector and the query vector.
- **Learning to rank**: These methods use machine learning algorithms to learn a scoring function that predicts a score or a rank for each item given some features or attributes. For example, if you want to rank products based on user feedback, you can use a learning to rank method to train a model that predicts the rating of each product based on its features and user profile.
- **Rank aggregation**: These methods combine multiple rankings of the same set of items into a single consensus ranking that minimizes some loss function. For example, if you want to rank candidates based on votes from different sources, you can use a rank aggregation method to find a ranking that best reflects the overall preferences of the voters.
- **Feedback arc set**: These methods find a linear ordering of items that minimizes the number of inconsistencies or conflicts with a given set of pairwise comparisons. For example, if you want to rank teams based on their outcomes in tournaments, you can use a feedback arc set method to find a ranking that minimizes the number of backward edges in the tournament graph.



# Learning to Rank
There are mainly three methods to learn a scoring function that predicts a score or a rank for each item given some features or attributes:
### Pointwise ranking
Pointwise ranking is a method for learning to rank by looking at a single item at a time and predicting its relevance score or class label for a given query. Pointwise ranking models can be trained using regression or classification loss functions, such as mean squared error, cross entropy, or hinge loss. The final ranking is achieved by simply sorting the items by their predicted scores or labels. Some examples of pointwise ranking algorithms are linear regression, logistic regression, and support vector machines¹.
### Pairwise Ranking
Pairwise ranking is a method for learning to rank by comparing pairs of items and predicting which one is more relevant or preferable for a given query. Pairwise ranking models minimize the number of inversions in the ranking, i.e., cases where a less relevant item is ranked higher than a more relevant item. Pairwise ranking models can be trained using different loss functions, such as hinge loss, cross entropy loss, or rank loss. Some examples of pairwise ranking algorithms are RankNet, LambdaRank, and LambdaMART.

Pairwise ranking has some advantages over pointwise ranking, such as:

- It is closer to the nature of ranking than predicting class label or relevance score.
- It can handle different levels of relevance or preference among items, rather than assuming a binary or ordinal scale.
- It can avoid label noise and sparsity by focusing on relative order rather than absolute values.

### Listwise Ranking
Listwise and pairwise ranking are two different approaches for learning to rank, which is a task to automatically construct a ranking model using training data.

Pairwise ranking looks at a pair of items at a time and tries to predict which one is more relevant or preferable for a given query. Pairwise ranking models minimize the number of inversions in the ranking, i.e., cases where a less relevant item is ranked higher than a more relevant item. Some examples of pairwise ranking algorithms are RankNet, LambdaRank, and LambdaMART.

Listwise ranking looks at the entire list of items and tries to predict the optimal ordering for them. Listwise ranking models directly optimize a ranking metric such as NDCG or MAP, or a loss function that captures the properties of the ranking problem. Some examples of listwise ranking algorithms are SoftRank, AdaRank, ListNet, and ListMLE.

Listwise ranking has some advantages over pairwise ranking, such as:

- It can capture the global structure of the ranking list and avoid inconsistencies caused by local comparisons.
- It can optimize the ranking metric of interest directly and avoid the mismatch between the loss function and the evaluation metric.
- It can handle different levels of relevance or preference among items, rather than assuming a binary or ordinal scale.


# Ranking Evaluation Metrics

Ranking methods can be evaluated using different metrics that measure how well they match the true or desired ranking of the items. Some common ranking metrics are:

- Mean average precision (MAP): This metric computes the average precision at each position in the ranking and then averages it over all queries or items. Precision measures how many of the top-ranked items are relevant or correct. MAP is used for binary relevance tasks where each item can be either relevant or not relevant.
- Discounted cumulative gain (DCG): This metric computes the sum of the gains or utilities of each item in the ranking discounted by its position¹. Gain measures how useful or valuable an item is for the user. DCG is used for graded relevance tasks where each item can have different levels of relevance.
- Normalized discounted cumulative gain (NDCG): This metric evaluates the quality of a ranking by measuring how well the relevant items are ranked higher than the irrelevant ones. NDCG is the ratio of DCG and ideal DCG (IDCG). IDCG is the maximum possible DCG for a given set of items and relevance scores. It is obtained by sorting the items by their relevance scores in descending order. NDCG normalizes DCG by IDCG to make it comparable across different sets of items and queries. NDCG ranges from 0 to 1, where 1 means a perfect ranking.

# Resources 
1. [Learning to Rank: A Complete Guide to Ranking using Machine Learning](https://towardsdatascience.com/learning-to-rank-a-complete-guide-to-ranking-using-machine-learning-4c9688d370d4)
2.  [Normalized Discounted Cumulative Gain](https://towardsdatascience.com/normalized-discounted-cumulative-gain-37e6f75090e9)
3. [ML Design Pattern - Ranking](https://www.youtube.com/watch?v=kBYHOum_iUA&t=1651s)
4. [Pointwise vs. Pairwise vs. Listwise Learning to Rank](https://medium.com/@nikhilbd/pointwise-vs-pairwise-vs-listwise-learning-to-rank-80a8fe8fadfd)
5. [Tensorflow Recommenders](https://github.com/tensorflow/recommenders/tree/main/tensorflow_recommenders)
