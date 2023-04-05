# ref: https://github.com/tensorflow/recommenders/blob/main/tensorflow_recommenders/examples/movielens.py
import collections
import numpy as np
import tensorflow as tf
from typing import Dict, List, Optional, Text, Tuple


def _sample_list(
    feature_lists: Dict[Text, List[tf.Tensor]],
    item_features: List[Text], 
    label_name: Text,
    num_examples_per_list: int,
    random_state: Optional[np.random.RandomState] = None,
    ) ->  Dict[Text, List[tf.Tensor]]:
    """Function for sampling a list example from given feature lists."""
    if random_state is None:
      random_state = np.random.RandomState()

    sampled_indices = random_state.choice(
        range(len(feature_lists[label_name])),
        size=num_examples_per_list,
        replace=False,
    )
    sampled_dict = collections.defaultdict(list)
    for feature in item_features:
        sampled_dict[feature] = [feature_lists[feature][idx] for idx in sampled_indices]
    
    sampled_dict[label_name] = [feature_lists[label_name][idx] for idx in sampled_indices]

    return sampled_dict

def sample_listwise(
    dataset: tf.data.Dataset,
    user_features: List[Text],
    item_features: List[Text],
    label_name: Text,
    num_list_per_user: int = 10,
    num_examples_per_list: int = 10,
    seed: Optional[int] = None,
) -> tf.data.Dataset:
    """Function for converting ratings/clicks dataset to a listwise dataset.
    Args:
        dataset: ratings/clicks dataset.
        user_features: names of user features to be included in the sampled data set,
        item_features: names of user features to be included in the sampled data set,
        label_name: name of feature to be used as label (e.g., ranking or watched or not),
        num_list_per_user:
          An integer representing the number of lists that should be sampled for
          each user in the training dataset.
        num_examples_per_list:
          An integer representing the number of movies to be sampled for each list
          from the list of movies rated by the user.
        seed:
          An integer for creating `np.random.RandomState`.
    Returns:
        A tf.data.Dataset containing list examples.
        Each example contains three keys: "user_id", "movie_title", and
        "user_rating". "user_id" maps to a string tensor that represents the user
        id for the example. "movie_title" maps to a tensor of shape
        [sum(num_example_per_list)] with dtype tf.string. It represents the list
        of candidate movie ids. "user_rating" maps to a tensor of shape
        [sum(num_example_per_list)] with dtype tf.float32. It represents the
        rating of each movie in the candidate list.
    """
    random_state = np.random.RandomState(seed)

    example_lists_by_user = collections.defaultdict(lambda:collections.defaultdict(list))

    movie_title_vocab = set()
    for example in dataset:
        user_features_ = tuple([example[feature].numpy() for feature in user_features])
        user_id = example["user_id"].numpy()
        example_lists_by_user[user_features_]["movie_title"].append(
            example["movie_title"])
        example_lists_by_user[user_features_]["user_rating"].append(
            example["user_rating"])
        movie_title_vocab.add(example["movie_title"].numpy())

    tensor_slices = {label_name: []}
    for feature in user_features:
        tensor_slices[feature] = []
    for feature in item_features:
        tensor_slices[feature] = []

    for user_features_, feature_lists in example_lists_by_user.items():
        for _ in range(num_list_per_user):

          # Drop the user if they don't have enough ratings.
          if len(feature_lists["movie_title"]) < num_examples_per_list:
            continue

          sampled_dict = _sample_list(
              feature_lists, item_features, label_name,
              num_examples_per_list,
              random_state=random_state,
          )
          # tensor_slices.append({"user_id":user_id, "movie_title":sampled_movie_titles, "user_rating":sampled_ratings})
          for i in range(len(user_features)):
              tensor_slices[user_features[i]].append(user_features_[i])
          for key, v in sampled_dict.items():
              tensor_slices[key].append(v)
        

    return tf.data.Dataset.from_tensor_slices(tensor_slices)