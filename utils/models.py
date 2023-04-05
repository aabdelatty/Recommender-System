import tensorflow as tf
import tensorflow_ranking as tfr
import tensorflow_recommenders as tfrs

class RetrievalModel(tfrs.Model):

    def __init__(self, user_tower, movie_tower, metrics):
        super().__init__()
        self.movie_tower = movie_tower
        self.user_tower= user_tower
        self.task = tfrs.tasks.Retrieval(metrics=metrics)

    def compute_loss(self, features, training=False) -> tf.Tensor:
        # We pick out the user features and pass them into the user model.
        user_embeddings = self.user_tower(features)
        # And pick out the movie features and pass them into the movie model,
        # getting embeddings back.
        positive_movie_embeddings = self.movie_tower(features)

        # The task computes the loss and the metrics.
        return self.task(user_embeddings, positive_movie_embeddings)

class RankingModel(tfrs.Model):

  def __init__(self, user_tower, movie_tower, loss, num_examples_per_user):
      super().__init__()
      
      self.user_tower = user_tower

      self.movie_tower = movie_tower
      # Compute predictions.
      self.score_model = tf.keras.Sequential([
        # Learn multiple dense layers.
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        # Make rating predictions in the final layer.
        tf.keras.layers.Dense(1)
      ])

      self.task = tfrs.tasks.Ranking(
        loss=loss,
        metrics=[
          tfr.keras.metrics.NDCGMetric(name="ndcg_metric"),
          tf.keras.metrics.RootMeanSquaredError()
        ]
      )

      self.num_examples_per_user = num_examples_per_user

  def call(self, features):
      # We first convert the id features into embeddings.
      # User embeddings are a [batch_size, embedding_dim] tensor.{"user_id":features["user_id"]}
      user_embeddings = self.user_tower(features)

      # Movie embeddings are a [batch_size, num_movies_in_list, embedding_dim]
      
      movie_embeddings = self.movie_tower(features)
      
      # We want to concatenate user embeddings with movie emebeddings to pass
      # them into the ranking model. To do so, we need to reshape the user
      # embeddings to match the shape of movie embeddings.
      user_embedding_repeated = tf.repeat(
          tf.expand_dims(user_embeddings, 1), [self.num_examples_per_user], axis=1)
      
      # Once reshaped, we concatenate and pass into the dense layers to generate
      # predictions.
      concatenated_embeddings = tf.concat(
          [user_embedding_repeated, movie_embeddings], 2)
      
      return self.score_model(concatenated_embeddings)

  def compute_loss(self, features, training=False):
      labels = features.pop("user_rating")

      scores = self(features)

      return self.task(
          labels=labels,
          predictions=tf.squeeze(scores, axis=-1),
      )