#!~/.conda/envs/comparison/bin/python
# %%
import pygad
import pygad.kerasga
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import activations, backend, Model
from tensorflow.keras.layers import Input, Embedding, Dense, Concatenate, Flatten
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

print('TensorFlow version', tf.__version__)


# %%

df = pd.read_csv('data/ratings.csv')
df

# %%

# Sampling 50% of the dataset, as the later steps cause
# an ResourceExhaustedError, which means the PC running this does not
# have enough memory to complete this.
df = df.sample(frac=0.1, axis='index', random_state=1)

df


# %%
# Encoding user IDs and movie IDs
# This is necessary so that the neural network does not infer anything from the IDs


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


# Scaling the ratings to values between 0 and 1

npRating = pd.DataFrame({'rating': df['rating']})
npRating = npRating.values.reshape(-1, 1)

scaler = MinMaxScaler()
rating = scaler.fit_transform(npRating)

rating

df['rating'] = rating

df

# %%
# Splitting to training and testing data


X = df[['userId', 'movieId']]
y = df['rating']

# Splitting the data, ensuring 30% is used for testing.
# The data is shuffled to make sure the neural network does not infer anything from the order.
# The random state is also set to 42 so that the results are consistent.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, shuffle=True, random_state=42)  # type: ignore

# Checking the sizes of the training and testing sets
X_train, X_test

# %%


def rootMeanSquaredError(actual, pred):
    backend.sqrt(backend.square(pred - actual))


def createModel(data: pd.DataFrame, userOutputDim: int, movieOutputDim: int, hiddenLayers: list[int] = []):
    userInput = Input(shape=(1,))
    movieInput = Input(shape=(1,))

    numUsers = data['userId'].nunique() + 1
    numMovies = data['movieId'].nunique() + 1

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
        denseLayers = Dense(units=layer, activation=activations.relu)(denseLayers)

    # The final layer is the output which has a single output neuron
    output = Dense(units=1)(denseLayers)
    model = Model(inputs=[userInput, movieInput], outputs=output)
    model.compile(optimizer='Adam', loss='MSE', metrics=['MAE'])
    return model

# %%


X.isna().values.any()

# %%


# Defining types so the linter does not complain
X_train: pd.DataFrame
X_test: pd.DataFrame

# Creating an initial model
model = createModel(data=df, userOutputDim=50, movieOutputDim=50)
model.fit(x=[X_train['userId'], X_train['movieId']], y=y_train,
          epochs=30, verbose=2, validation_data=([X_test['userId'], X_test['movieId']], y_test))

# %%

# Running the neural network on the clusters
# --- Step skipped: The output of the NN in the journal is unclear ---
#     Because of this, the clustering algorithm no longer makes sense.

# %%

print('PyGAD version', pygad.__version__)

# %%

model = createModel(data=df, userOutputDim=50, movieOutputDim=50, hiddenLayers=[50])
agnn = pygad.kerasga.KerasGA(model=model, num_solutions=10)


def fitnessMAE(solution, solutionIndex):
    """Fitness function using MAE"""
    """Based on https://pygad.readthedocs.io/en/latest/README_pygad_kerasga_ReadTheDocs.html#functions-in-the-pygad-kerasga-module"""
    # I don't like that you have to use globals here, but this seems to be
    # the only way, according to the PyGAD documentation.
    global X_train, y_train, model
    modelWeights = pygad.kerasga.model_weights_as_matrix(model=model, weights_vector=solution)
    model.set_weights(modelWeights)
    y_pred = model.predict([X['userId'], X['movieId']])
    mae = keras.losses.MeanAbsoluteError()
    solutionError = mae(y, y_pred).numpy()  # type: ignore
    fitness = 1/solutionError
    return fitness


def callback(instance):
    print('Generation ', instance.generations_completed)
    print('Fitness    ', instance.best_solution()[1])

# %%


# ----- Setting up the genetic algorithm -----
# A lot of assumptions have been made here, as the original journal
# Does not specify a lot of things, such as how many generations the
# algorithm should go on for, the number of generations, mutation percentages
# and how many parents to keep.
geneticAlgorithm = pygad.GA(num_generations=100,
                            num_parents_mating=2,
                            initial_population=agnn.population_weights,
                            parent_selection_type='rank',
                            crossover_type='two_points',
                            mutation_type='random',
                            mutation_percent_genes=10,
                            keep_parents=1,
                            fitness_func=fitnessMAE,
                            on_generation=callback)

geneticAlgorithm.run()
geneticAlgorithm.plot_result()

# %%

# Running the neural network on clustered data

clustered_movies = pd.read_csv('data/clustered_movies.csv')
clustered_movies.drop(clustered_movies.columns[0], axis=1, inplace=True)
clustered_movies

# %%

filtered_ratings = pd.read_csv('data/filtered_ratings.csv')
filtered_ratings

# %%

# Getting the IDs of movies in each cluster


def movie_ids_in_cluster(cluster: pd.Series):
    return cluster.index[cluster.values]


def get_ratings_in_cluster(cluster: pd.Series):
    """Returns a DataFrame of ratings of movies within the cluster"""
    movies_in_cluster = cluster.index[cluster.values].to_numpy(dtype=int)  # type:ignore
    print(movies_in_cluster)
    return filtered_ratings[filtered_ratings['movieId'].isin(movies_in_cluster)]


# for cluster in clustered_movies.itertuples():
clusters = clustered_movies.apply(get_ratings_in_cluster, axis=1).tolist()
clusters[5]

# %%

# Reading the filtered ratings to allocate reviews to each cluster

# for cluster in clusters:
cluster = clusters[0]
print('Cluster empty:', cluster.empty)

# %%

# Splitting the data into features and target
X = cluster[['userId', 'movieId']]
y = cluster['rating']

X

# %%

# Encoding the IDs
userId_encoder = Encoder(X['userId'].unique())
movieId_encoder = Encoder(X['movieId'].unique())
# Assigning new encoded IDs
X.userId = X.userId.map(userId_encoder.original2encoded)
X.movieId = X.movieId.map(movieId_encoder.original2encoded)

# %%

# Splitting dataset into training and testing sets (25% used for testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, shuffle=True, random_state=42)  # type:ignore

# %%

# Creating the model with all the points in the cluster
model = createModel(data=cluster, userOutputDim=50, movieOutputDim=50)
# Training the model with the split data
history = model.fit(x=[X_train['userId'], X_train['movieId']], y=y_train,
                    epochs=30, verbose=2, validation_data=([X_test['userId'], X_test['movieId']], y_test))

# %%

# Plotting the results

plt.plot(history.history['MAE'])
plt.plot(history.history['val_MAE'])
plt.ylabel('Mean Absolute Error (MAE)')
plt.xlabel('Epoch')
plt.legend(['Training data', 'Testing data'])
plt.title('Mean Absolute Error per training epoch')

# %%

# Repeating the above for each cluster

epochs = 5

for cluster in clusters:
    if cluster.empty:
        print('Skipping cluster as it is empty')
        continue
    # Splitting the data into features and target
    X = cluster[['userId', 'movieId']]
    y = cluster['rating']
    # Encoding the IDs
    userId_encoder = Encoder(X['userId'].unique())
    movieId_encoder = Encoder(X['movieId'].unique())
    # Assigning new encoded IDs
    X.userId = X.userId.map(userId_encoder.original2encoded)
    X.movieId = X.movieId.map(movieId_encoder.original2encoded)
    # Splitting dataset into training and testing sets (25% used for testing)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, shuffle=True, random_state=42)  # type:ignore
    # Creating the model with all the points in the cluster
    model = createModel(data=cluster, userOutputDim=50, movieOutputDim=50)
    # Training the model with the split data
    history = model.fit(x=[X_train['userId'], X_train['movieId']], y=y_train,
                        epochs=epochs, verbose=2, validation_data=([X_test['userId'], X_test['movieId']], y_test))
    # Plotting the results
    plt.plot(history.history['MAE'])
    plt.plot(history.history['val_MAE'])
    plt.ylabel('Mean Absolute Error (MAE)')
    plt.xlabel('Epoch')
    plt.legend(['Training data', 'Testing data'])
    plt.title('Mean Absolute Error per training epoch')
    plt.xticks(np.arange(0, epochs, step=1))
    plt.show()

# %%

# Running the above on the whole dataset

# Splitting the data into features and target
X = filtered_ratings[['userId', 'movieId']]
y = filtered_ratings['rating']
# Encoding the IDs
userId_encoder = Encoder(X['userId'].unique())
movieId_encoder = Encoder(X['movieId'].unique())
# Assigning new encoded IDs
X.userId = X.userId.map(userId_encoder.original2encoded)
X.movieId = X.movieId.map(movieId_encoder.original2encoded)
# Splitting dataset into training and testing sets (25% used for testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, shuffle=True, random_state=42)  # type:ignore
# Creating the model with all the points in the cluster
model = createModel(data=filtered_ratings, userOutputDim=50, movieOutputDim=50)
# Training the model with the split data
history = model.fit(x=[X_train['userId'], X_train['movieId']], y=y_train,
                    epochs=epochs, verbose=2, validation_data=([X_test['userId'], X_test['movieId']], y_test))
# Plotting the results
plt.plot(history.history['MAE'])
plt.plot(history.history['val_MAE'])
plt.ylabel('Mean Absolute Error (MAE)')
plt.xlabel('Epoch')
plt.legend(['Training data', 'Testing data'])
plt.title('Mean Absolute Error per training epoch')
plt.xticks(np.arange(0, epochs, step=1))
plt.show()

# %%
