# Recommender Systems
This repository gives an introduction to recommender systems and the state of the art recommender models. Also, the repository provides different implementations for two-steps recommenders using TensorFlow. 

A two-step recommender system is a type of recommender system that separates the recommendation problem into two stages: retrieval and ranking. In the first stage, a retrieval model selects a subset of items from a large catalog that are relevant to the user query or context. In the second stage, a ranking model reorders the retrieved items according to their predicted utility or preference for the user. A two-step recommender system can improve the efficiency and accuracy of recommendations by reducing the search space and optimizing different objectives at each stage.
![alt text](https://miro.medium.com/v2/resize:fit:720/format:webp/1*v3vWFYmejaHzx1GSLVngaw.jpeg)
# Table of contents
1. [Introduction](#introduction)
2. [Recommender Systems Paradigms](#recommender-systems-paradigms)
	* [Collaborative filtering](#collaborative-filtering)
		* [Model-Based Collaborative Filtering](#model-based-collaborative-filtering)
	* [Content-Based Methods](#content-based-methods)
3. [Recommender Systems Challenges](#recommender-systems-challenges)
4. [Building Scalable Recommender Systems](#building-scalable-recommender-systems)
5. [Recommender System Evaluation Metrics](#recommender-system-evaluation-metrics)
6. [Resources](#resources)
# Introduction

Recommender systems are algorithms that provide suggestions for items that are most relevant to a particular user. They are widely used in various domains, such as e-commerce, entertainment, social media, and online advertising. Recommender systems aim to enhance the user experience by helping them discover new products, content, or services that match their preferences and needs.

# Recommender Systems Paradigms 
There are two main paradigms of recommender systems: collaborative filtering and content-based methods. Collaborative filtering methods rely on the past interactions between users and items, such as ratings, purchases, or clicks, to learn the preferences of users and the characteristics of items. Content-based methods use the features of users and items, such as demographics, keywords, or genres, to recommend items that are similar to what the user has liked before.
## Collaborative Filtering
Collaborative filtering methods can be divided into memory-based and model-based approaches. Memory-based approaches use the user-item interactions matrix to compute the similarity between users or items, and then use these similarities to make predictions or recommendations. For example, user-based collaborative filtering recommends items that are liked by other users who are similar to the target user, while item-based collaborative filtering recommends items that are similar to the items that the target user has liked. Memory-based approaches are simple and intuitive, but they suffer from scalability and sparsity issues.

Model-based approaches use machine learning techniques to learn a latent model from the user-item interactions matrix, and then use this model to make predictions or recommendations. Model-based approaches can overcome some of the limitations of memory-based approaches, but they require more computational resources and may not capture complex patterns in the data.
### Model-Based Collaborative Filtering
1. Matrix factorization is a class of collaborative filtering algorithms used in recommender systems. Matrix factorization algorithms work by decomposing the user-item interaction matrix into the product of two lower dimensionality rectangular matrices, one representing the latent features of users and the other representing the latent features of items. Then, these latent features can be used to estimate the preferences of users for items or to recommend items that are similar to what the user has liked before. Matrix factorization techniques can improve the accuracy and scalability of recommender systems by reducing the dimensionality and sparsity of the user-item matrix. They can also incorporate additional information such as implicit feedback, temporal effects, and confidence levels. Some examples of matrix factorization methods are singular value decomposition (SVD), non-negative matrix factorization (NMF), and probabilistic matrix factorization (PMF).
![alt text](https://developers.google.com/machine-learning/recommendation/images/Matrixfactor.svg)
2. Two-Tower recommender system is a type of hybrid recommender system that combines collaborative filtering and content-based methods. It uses two deep neural networks, one for users and one for items, to embed them into a low-dimensional space. Then, it predicts the ratings or preferences of users for items based on the geometric relationship of the embeddings, such as cosine similarity or dot product. A two-tower recommender system can leverage both the user-item interactions and the features of users and items to make more accurate and personalized recommendations. 

![alt text](https://miro.medium.com/v2/resize:fit:1100/0*bii7baVk5nF6lulx)	
#### Two-Towers VS Matrix Factorization

Two towers and matrix factorization are two different approaches for building recommender systems based on collaborative filtering. They have some similarities and differences, as follows:

Similarities:

- Both methods use low-dimensional embeddings to represent users and items.

- Both methods can learn the embeddings from user-item interactions, such as ratings, clicks, or purchases.

- Both methods can use dot product or cosine similarity to measure the affinity between users and items.

Differences:

- Matrix factorization decomposes the user-item interaction matrix into two matrices, one for users and one for items. Two towers use two separate neural networks, one for users and one for items, to embed them into a common space.

- Matrix factorization assumes that the user-item interaction matrix is sparse and can only use the observed interactions to learn the embeddings. Two towers can use additional features of users and items, such as text, images, or metadata, to enrich the embeddings.

- Matrix factorization treats the recommendation problem as a regression or classification problem, where the goal is to predict the rating or preference of a user for an item. Two towers treat the recommendation problem as a retrieval problem, where the goal is to find the most similar items to a user query.

## Content-based methods
Content-based methods use the features of users and items to build a profile for each user and item, and then use these profiles to recommend items that are similar to the user's profile. For example, a content-based recommender system for movies may use the genres, actors, directors, and plot keywords of movies to build a profile for each movie, and then use the user's ratings or reviews to build a profile for each user. Then, these profiles can be used to measure the similarity between users and movies, and recommend movies that are close to the user's profile. Content-based methods can deal with new items and do not depend on other users' feedback, but they may suffer from overspecialization and limited content analysis.
# Recommender Systems Challenges 
Recommender systems are facing various challenges in their design, implementation, and evaluation. Some of the common challenges are:
- Cold-start: This challenge occurs when a new user or a new item enters the system, and there is not enough information about them to make meaningful recommendations. For example, a new user may not have rated any items yet, or a new item may not have been rated by any users yet. This requires recommender systems to use alternative sources of information, such as user or item features, or to elicit feedback from users or items.
- Data sparsity: This challenge occurs when the user-item interaction matrix is very sparse, meaning that most of the users have rated or interacted with only a few items, or vice versa. This makes it difficult for recommender systems to learn the preferences of users and the characteristics of items, and to provide accurate and personalized recommendations.
- Scalability: This challenge occurs when the number of users or items in the system grows very large, and the recommender system needs to handle a huge amount of data and computation. For example, a recommender system may need to store and process millions of user-item interactions, or to generate and rank thousands of candidate items for each user. This requires recommender systems to use efficient data structures, algorithms, and architectures to ensure fast and reliable performance.
- Diversity: This challenge occurs when the recommender system tends to recommend only popular or similar items to users, and neglects less popular or dissimilar items that may also be relevant or interesting. For example, a recommender system may always recommend blockbuster movies or best-selling books to users, and ignore niche movies or books that may suit their tastes better. This reduces the diversity and novelty of the recommendations, and may lead to user boredom or dissatisfaction.
- Privacy: This challenge occurs when the recommender system collects and uses sensitive or personal information from users, such as their ratings, preferences, behaviors, or profiles. For example, a recommender system may track the browsing history or location of users, or infer their demographics or preferences from their interactions. This may raise privacy concerns among users, who may not want to share their information with the system or with other parties. This requires recommender systems to respect the privacy preferences of users, and to use secure and ethical methods for data collection and processing.


# Building Scalable Recommender Systems
Building scalable recommender systems is a challenging task that requires addressing various aspects, such as data storage, model complexity, computational efficiency, and system architecture. Some possible ways to build scalable recommender systems are:
- Using distributed database systems to store and access large amounts of user-item interaction data and item features. For example, Google uses Bigtable and Spanner to store YouTube video metadata and user activity logs.
- Using parallel and distributed algorithms to train and serve recommender models on multiple machines or clusters. For example, Netflix uses Spark and AWS to train matrix factorization models on millions of users and items.
- Using low-dimensional embeddings to represent users and items, which can reduce the dimensionality and sparsity of the user-item matrix and enable efficient similarity computation. For example, YouTube uses deep neural networks to learn embeddings for users and videos.
- Using two-stage approaches to separate the recommendation problem into retrieval and ranking stages, which can reduce the search space and optimize different objectives. For example, Google uses two-tower models for retrieval and gradient boosted trees for ranking in their ads system.

# Recommender System Evaluation Metrics
Recommender system evaluation metrics are measures that assess how well a recommender system performs in terms of satisfying the user's needs and preferences. There are different types of metrics depending on the type of recommender system, the type of user feedback, and the type of evaluation setting. Some common categories of metrics are:

- Accuracy metrics: These metrics measure how close the predicted ratings or preferences of a recommender system are to the actual ratings or preferences of the users. For example, mean absolute error (MAE), root mean square error (RMSE), and R2 score are accuracy metrics for explicit feedback systems, where users provide numerical ratings for items. Precision, recall, F1-score, and area under the ROC curve (AUC) are accuracy metrics for implicit feedback systems, where users provide binary feedback for items, such as clicks or purchases.
- Ranking metrics: These metrics measure how well a recommender system ranks the items according to their relevance or usefulness for the users. For example, hit ratio (HR), mean reciprocal rank (MRR), mean average precision ([MAP](https://towardsdatascience.com/mean-average-precision-at-k-map-k-clearly-explained-538d8e032d2)), and normalized discounted cumulative gain ([NDCG](https://towardsdatascience.com/demystifying-ndcg-bee3be58cfe0)) are ranking metrics that evaluate how often and how high the relevant items appear in the recommended list.
- Diversity metrics: These metrics measure how diverse or varied the recommended items are in terms of their features or categories. For example, intra-list diversity measures the dissimilarity among the items in a recommended list, while inter-list diversity measures the dissimilarity among the items in different recommended lists for different users. Diversity metrics can capture the trade-off between accuracy and novelty in recommender systems.
- Coverage metrics: These metrics measure how well a recommender system covers the entire set of items or users in its recommendations. For example, catalog coverage measures the percentage of items that are recommended to at least one user, while user coverage measures the percentage of users that receive at least one recommendation. Coverage metrics can capture the trade-off between accuracy and serendipity in recommender systems.

# Resources 
1. [Introduction to recommender systems](https://towardsdatascience.com/introduction-to-recommender-systems-6c66cf15ada)
2.  [Recommender System — Matrix Factorization](https://towardsdatascience.com/recommendation-system-matrix-factorization-d61978660b4b)
3. [On strong convergence of the two-tower model for recommender system]( https://openreview.net/forum?id=Ulj0tR-k7q)
4. [Personalized recommendations - IV (two tower models for retrieval)](https://www.linkedin.com/pulse/personalized-recommendations-iv-two-tower-models-gaurav-chakravorty)
5. [A Survey on the Scalability of Recommender Systems for Social Networks](https://link.springer.com/chapter/10.1007/978-3-319-90059-9_5)
6. [Approaches, Issues and Challenges in Recommender Systems: A Systematic](https://indjst.org/articles/approaches-issues-and-challenges-in-recommender-systems-a-systematic-review)
