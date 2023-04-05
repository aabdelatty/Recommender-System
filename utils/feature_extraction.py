import tensorflow as tf
import numpy as np

class FeatureExtractionTower(tf.keras.Model):
    def __init__(self, inputs: tf.data.Dataset, 
                       cats_to_embedding: tf.data.Dataset=[],
                       cats_to_hash_embedding: tf.data.Dataset=[], 
                       text_to_embedding: tf.data.Dataset=[], 
                       cont_to_embedding: tf.data.Dataset=[], 
                       cont_to_normalized: tf.data.Dataset=[]):
        super().__init__()
        self.cats_to_embedding = cats_to_embedding 
        self.cats_to_hash_embedding = cats_to_hash_embedding
        self.text_to_embedding = text_to_embedding
        self.cont_to_embedding = cont_to_embedding
        self.cont_to_normalized = cont_to_normalized

        self.preproc_models = dict()
        # Take the input dictionary, pass it through each input layer,
        # and concatenate the result.
        
        for cat_var in self.cats_to_embedding:
            vals = inputs.batch(1_000_000).map(lambda x: x[cat_var])
            unique_vals = np.unique(np.concatenate(list(vals)))
            self.preproc_models[cat_var] = self.build_lookup_model(unique_vals)
        for cat_var in self.cats_to_hash_embedding:
            self.preproc_models[cat_var] = self.build_hashing_model()
        for text_var in self.text_to_embedding:
            vals = inputs.batch(1_000_000).map(lambda x: x[text_var])
            unique_vals = np.unique(np.concatenate(list(vals)))
            self.preproc_models[text_var] = self.build_embedding_extractor_from_text_variable(unique_vals)
        for cont_var in self.cont_to_embedding:
            vals = inputs.batch(1_000_000).map(lambda x: x[cont_var])
            self.preproc_models[cont_var] = self.build_embedding_extractor_from_cont_variable(vals)
        for cont_var in self.cont_to_normalized:
            vals = inputs.batch(1_000_000).map(lambda x: x[cont_var])
            self.preproc_models[cont_var] = self.build_normalizer_for_cont_variable(vals)


    def build_hashing_model(self, output_dim=32, num_bins=100_000):
        input_hashing = tf.keras.layers.Hashing(num_bins=num_bins)
        input_embedding = tf.keras.layers.Embedding(
            input_dim=num_bins,
            output_dim=output_dim)
        return tf.keras.Sequential([input_hashing, input_embedding])
      
    def build_lookup_model(self, input, output_dim=32):
        input_lookup = tf.keras.layers.StringLookup()
        input_lookup.adapt(input)
        input_embedding = tf.keras.layers.Embedding(
            input_dim=input_lookup.vocabulary_size(),
            output_dim=output_dim)
        return tf.keras.Sequential([tf.keras.layers.InputLayer(input_shape=[], dtype=tf.string),input_lookup, input_embedding])

    def build_normalizer_for_cont_variable(self, input):
        normalization_layer = tf.keras.layers.Normalization(axis=None)
        normalization_layer.adapt(input)
        return normalization_layer
      
    def build_embedding_extractor_from_cont_variable(self, input, output_dim=32, num_bins=1000):
        max_var_value = input.numpy().max()
        min_var_value = input.numpy().min()

        var_buckets = np.linspace(min_var_value, max_var_value, num=num_bins)

        var_embedding_model = tf.keras.Sequential([
            tf.keras.layers.Discretization(var_buckets.tolist()),
            tf.keras.layers.Embedding(input_dim=len(var_buckets) + 1, 
                                      output_dim=output_dim)
          ])
        
        return var_embedding_model
        
    def build_embedding_extractor_from_text_variable(self, input, output_dim=32, max_tokens=1000):
        text_vectorizer = tf.keras.layers.TextVectorization(max_tokens=max_tokens)
        text_vectorizer.adapt(input)
        return tf.keras.Sequential([
                    tf.keras.layers.Input(shape=(1,),dtype='string'),
                    text_vectorizer,
                    tf.keras.layers.Embedding(input_dim=max_tokens, 
                                              output_dim=output_dim, 
                                              mask_zero=True),
                    # We average the embedding of individual words to get one embedding vector
                    # per title.
                    tf.keras.layers.GlobalAveragePooling1D(),
                  ])
          
    def call(self, inputs):
        # Take the input dictionary, pass it through each input layer,
        # and concatenate the result.
        out = []
        for cat_var in self.cats_to_embedding:
            out.append(self.preproc_models[cat_var](inputs[cat_var]))
        for cat_var in self.cats_to_hash_embedding:
            out.append(self.preproc_models[cat_var](inputs[cat_var]))
        for text_var in self.text_to_embedding:
            out.append(self.preproc_models[text_var](inputs[text_var]))
        for cont_var in self.cont_to_embedding:
            out.append(self.preproc_models[cont_var](inputs[cont_var]))
        for cont_var in self.cont_to_normalized:
            out.append(tf.reshape(self.preproc_models[cont_var](inputs[cont_var]), (-1, 1)))
        return tf.concat(out, axis=1)