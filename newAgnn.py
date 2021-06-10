# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import tensorflow as tf
from tensorflow import keras

print('TensorFlow version', tf.__version__)


# %%
import pandas as pd

df = pd.read_csv('data/ratings.csv')
df.head()


# %%
# Encoding user IDs and movie IDs
# This is necessary so that the neural network does not infer anything from the IDs

from sklearn.preprocessing import LabelEncoder


class Encoder:
    """Simple class to go between """

    def __init__(self, original):
        encoder = LabelEncoder()
        encoded = encoder.fit_transform(original)
        self.original2encoded = {original: encoded for original, encoded in zip(original, encoded)}
        self.encoded2original = {encoded: original for original, encoded in zip(original, encoded)}

    def original(self, encoded):
        return self.encoded2original[encoded]

    def encoded(self, original):
        return self.original2encoded[original]


userIdMapping = Encoder(df.userId.unique())
movieIdMapping = Encoder(df.movieId.unique())

df.userId = df.userId.map(userIdMapping.original2encoded)
df.movieId = df.movieId.map(movieIdMapping.original2encoded)


# %%
# Splitting to training and testing data

from sklearn.model_selection import train_test_split

X = df[['userId', 'movieId']]
y = df['rating']

# Splitting the data, ensuring 30% is used for testing.
# The data is shuffled to make sure the neural network does not infer anything from the order.
# The random state is also set to 42 so that the results are consistent.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=42)

X_test


# %%
class Recommender(keras.Model):
    def __init__(self, n_users, n_movies):
        super(Recommender, self).__init__()
        # Rule of thumb: 4th root of input for output
        self.users = keras.layers.Embedding(input_dim=n_users, output_dim=n_users**(1/4))
        self.movies = keras.layers.Embedding(input_dim=n_movies, output_dim=n_movies**(1/4))


# %%
from tensorflow.keras.layers import Input, Embedding, Dense, Concatenate, Flatten
from tensorflow.keras import activations, backend, Model

def rootMeanSquaredError(actual, pred): 
    backend.sqrt(backend.square(pred - actual))

def createModel(data: pd.DataFrame, userOutputDim: int, movieOutputDim: int, hiddenLayers: list[int] = []):
    userInput = Input(shape=(1,))
    movieInput = Input(shape=(1,))

    numUsers = data['userId'].nunique()
    numMovies = data['movieId'].nunique()

    # Using average between number of movies and output layer as default hidden layer
    hiddenLayers = [round((numMovies + 1) / 2)] if not hiddenLayers else hiddenLayers

    userEmbedding = Embedding(input_dim=numUsers, output_dim=userOutputDim)(userInput)
    movieEmbedding = Embedding(input_dim=numMovies, output_dim=movieOutputDim)(movieInput)

    # The network must converge down to one 'branch' so we concatenate the two embeddings
    concatenated = Concatenate()([userEmbedding, movieEmbedding])

    # Flattening the 2D data to 1D
    flattened = Flatten()(concatenated)

    # Creating a set of regular layers, with n nodes in each
    denseLayers = flattened
    for layer in hiddenLayers:
        denseLayers = Dense(units=layer, activation=activations.sigmoid)(denseLayers)

    # The final layer is the output which has a single output neuron
    output = Dense(units=1)(denseLayers)
    model = Model(inputs=[userInput, movieInput], outputs=output)
    model.compile(loss=rootMeanSquaredError, metrics="root_mean_squared_error")
    return model

# %%

# Creating an initial model
# TODO: split up 'x' into a list of movie Ids and user Ids (same for validation)
model = createModel(data=df, userOutputDim=50, movieOutputDim=50)
model.fit(x=[], y=y_train, epochs=5, verbose=2, validation_data=([], y_test))
