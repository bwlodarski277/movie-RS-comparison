# %% Imports
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Embedding, Dense, Concatenate, Flatten
from tensorflow.keras import activations, backend, Model, losses, metrics, optimizers
import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# %%
# Loading pre-filtered dataset

# This can be loaded in from a pre-filtered dataset as
# it does not need to be calculated every time.
filtered_ratings = pd.read_csv('data/filtered_ratings.csv')
filtered_ratings

# %%
# Loading the genres table

genres_table = pd.read_csv('data/movies.csv')
genres_table

# %%
# Dropping movie title and encoding the genres into columns

encodings = genres_table['genres'].str.split('|').explode()
updated_genres = genres_table.drop(columns=['title', 'genres'])
crosstab = pd.crosstab(encodings.index, encodings)

crosstab

# %%
# Performing Principal Component Analysis on the ratings

# Scaling The dataset first
scaler = MinMaxScaler()
scaled = scaler.fit_transform(crosstab)

# Performing PCA
pca = PCA(n_components=2)
pca_movie_genres = pca.fit_transform(scaled)

# Creating scatterplot of the data
plt.scatter(x=pca_movie_genres[:, 0], y=pca_movie_genres[:, 1])
plt.title('Principal component analyis of the movie genres')
plt.xlabel('Principal component 1')
plt.ylabel('Principal component 2')

# %%
# Performing standard K-means clustering

# Creating an empty array to store the cluster data for later
clusters_standard = []
# Performing K-means with K ranging from 2 to 5
for n_clusters in range(2, 6):
    kmeans = KMeans(n_clusters=n_clusters)
    clusters = kmeans.fit_predict(pca_movie_genres).tolist()
    clusters_standard.append(clusters)
    centroids = kmeans.cluster_centers_
    plt.scatter(x=pca_movie_genres[:, 0], y=pca_movie_genres[:, 1], c=clusters)
    plt.scatter(x=centroids[:, 0], y=centroids[:, 1], marker='^',  # type: ignore
                color='k', edgecolors='w')
    plt.title(f'Standard K-Means with {n_clusters} clusters')
    plt.savefig(f'images/kmeans/{n_clusters}-cluster.png')
    plt.show()

with open('pickle/clusters_standard.pkl', 'wb') as file:
    pickle.dump(clusters_standard, file)

# Clearing memory
# del clusters_standard

# %%
# Performing modified K-means clustering

# Modified K-Means implementation using NumPy


class ModifiedKMeansV2:
    """Modified k-means clustering algorithm. Takes in a number of clusters."""

    def __init__(self, n_clusters: int):
        """n_clusters: number of clusters to create."""

        self.n_clusters: int = n_clusters
        self.centroids = np.empty((n_clusters, 2))
        self.thresholds = np.empty((n_clusters,))

    def _set_centroids(self, data_points: np.ndarray) -> None:
        """Picks random data points as cluster centroids and assigns them."""
        # Get the number of data points
        row_count = data_points.shape[0]
        # Pick n random indexes to select rows as starting centroids
        centroids = np.random.choice(
            row_count, size=self.n_clusters, replace=False)
        # Picking the datapoints as centroids
        centroids = data_points[centroids]
        # Assigning to the Cluster dataclass
        self.centroids = centroids

    def _euclidean_dist(self, t: np.ndarray, c: np.ndarray) -> float:
        """Calculates the Euclidean Distance between datapoint t and centroid c."""
        return np.sqrt(np.sum((t - c) ** 2))

    def _k_means_objective(self, data_points: np.ndarray):
        """Uses the initial objective function to set new clusters for the first iteration of the clustering process."""
        for i, point in enumerate(data_points):
            smallest_dist = np.Infinity
            closest = -1
            for j, centroid in enumerate(self.centroids):
                distance = self._euclidean_dist(point, centroid)
                if distance < smallest_dist:
                    smallest_dist = distance
                    closest = j
                # Adding the distance between cluster centroid and point to list
                self.distances[j, i] = distance
            # Assigning the point to the closest centroid cluster
            self.points[closest, i] = True

    def _calc_threshold(self, distances: np.ndarray):
        return np.mean(distances)

    def _modified_k_means_objective(self, data_points: np.ndarray):
        """The modified objective function which uses thresholds to assign points to the clusters."""
        for i, point in enumerate(data_points):
            assigned = False
            smallest_dist = np.Infinity
            closest = -1
            for j, centroid in enumerate(self.centroids):
                distance = self._euclidean_dist(point, centroid)
                self.distances[j, i] = distance
                if distance < smallest_dist:
                    smallest_dist = distance
                    closest = j
                if distance <= self.thresholds[j]:
                    self.points[j, i] = True
                    assigned = True
            if not assigned:
                self.points[closest, i] = True

    def _new_centroid(self, points: np.ndarray):
        """Calculates the new cluster centroid based on the points in the cluster."""
        return np.mean(points, axis=0)

    def cluster_data(self, iters: int, data_points: np.ndarray):
        """Clusters the dataset using the modified k-means clustering algorithm."""
        # Creating a 2d array which identifies which cluster each point belongs to
        # By default, each point is not assigned to any cluster, so the value is False
        self.points = np.full((self.n_clusters, data_points.shape[0]), False)
        # Setting the distances between each point and cluster centroid
        # By default the values are all set to infinity as the distance
        # at this point has not been measuered yet
        self.distances = np.full((self.n_clusters, data_points.shape[0]), np.Infinity)
        self._set_centroids(data_points)
        self._k_means_objective(data_points)
        for iteration in range(iters):
            old_points = np.copy(self.points)
            # for j, cluster in enumerate(self.clusters):
            for j in range(self.n_clusters):
                # Assigning a new threshold to each cluster
                # By getting the distances between each point in each cluster
                self.thresholds[j] = self._calc_threshold(self.distances[j])
                # Getting the data points in the current cluster
                cluster_points = data_points[self.points[j]]
                # Calculating new centroid based on the average position
                # of the data points in the given cluster
                self.centroids[j] = self._new_centroid(cluster_points)
            # Resetting the values of distances and points assigned to cluster
            self.distances = np.full((self.n_clusters, data_points.shape[0]), np.Infinity)
            self.points = np.full((self.n_clusters, data_points.shape[0]), False)
            self._modified_k_means_objective(data_points)
            if np.array_equal(old_points, self.points):
                print(f'\nFinished in {iteration+1} iterations')
                break
        else:
            # If the iteration limit is reached
            print('\nIteration limit reached')
        return self.points


# Empty array to store the cluster data
clusters_custom = []
# Performing modified K-means with K ranging from 2 to 5
for n_clusters in range(2, 6):
    modified_kmeans = ModifiedKMeansV2(n_clusters=n_clusters)
    clusters = modified_kmeans.cluster_data(iters=100, data_points=pca_movie_genres)
    cluster_points = [pca_movie_genres[cluster] for cluster in clusters]
    clusters_custom.append(modified_kmeans.points)
    centroids = modified_kmeans.centroids
    for cluster in cluster_points:
        plt.scatter(cluster[:, 0], cluster[:, 1])
    plt.scatter(x=centroids[:, 0], y=centroids[:, 1], marker='^',  # type:ignore
                color='k', edgecolors='w')
    plt.title(f'Modified K-Means with {n_clusters} clusters')
    plt.savefig(f'images/modified_kmeans/{n_clusters}-cluster.png')
    plt.show()

with open('pickle/clusters_custom.pkl', 'wb') as file:
    pickle.dump(clusters_custom, file)

# %%
# Creating Keras model


def create_model(data: pd.DataFrame):
    user = Input(shape=(1,))  # Single user
    movie = Input(shape=(1,))  # Single movie
    num_users = data['userId'].nunique()
    num_movies = data['movieId'].nunique()
    # Using average between number of movies and output layer as default hidden layer
    hidden_layer = round((num_movies + 1) / 2)
    # Creating embedding for the users
    users = Embedding(input_dim=num_users, output_dim=50)(user)
    # Creating embedding for movies
    movies = Embedding(input_dim=num_movies, output_dim=50)(movie)
    # The network must converge down to one 'branch' so we concatenate the two embeddings
    users_and_movies = Concatenate()([users, movies])
    # Creating a regular layer, with n nodes
    dense_layer = Dense(units=hidden_layer, activation=activations.relu)(users_and_movies)
    # The final layer is the output which has a single output neuron
    output = Dense(units=1)(dense_layer)
    model = Model(inputs=[user, movie], outputs=output)
    model.compile(optimizer=optimizers.Adam(),
                  loss='MAE',
                  metrics=['MAE'])
    return model

# %% Creating label encoder


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

# %% Running Keras model on standard clusters


def get_standard_ratings(clusters: np.ndarray):
    """Gets the movies that belong to each cluster using standard algorithm"""
    k_max: int = clusters.max() + 1  # type:ignore
    cluster_reviews = np.empty((k_max,), dtype=object)
    for k in range(k_max):
        # Getting list of movies in the cluster
        movies = genres_table[clusters == k]['movieId']
        # Getting list of reviews associate with the cluster
        reviews = filtered_ratings[filtered_ratings['movieId'].isin(movies)]  # type:ignore
        cluster_reviews[k] = reviews
    return cluster_reviews


n_epochs = 10

standard_kmeans_cluster_scores_mae = []
standard_kmeans_cluster_scores_rmse = []


# Iterating over all standard clusters
# Where K is the number of clusters
for k in range(2, 6):
    cluster_reviews = []
    k_idx = k - 2
    # Identifying which cluster each point belongs to
    current_cluster = np.array(clusters_standard[k_idx])
    # Creating array that holds ratings of movies that belong in each cluster
    cluster_reviews = get_standard_ratings(current_cluster)
    print(f'Creating models for standard K-means with {k} clusters')
    cluster_scores_mae = []
    cluster_scores_rmse = []
    for cluster_idx in range(k):
        print(f'Training cluster {cluster_idx+1}')
        model = create_model(data=cluster_reviews[cluster_idx])
        plot_model(model, to_file=f'images/models/standard-{k}.png', show_shapes=True)
        # Splitting dataset into training and testing sets (25% used for testing)
        X = cluster_reviews[cluster_idx][['userId', 'movieId']]
        y = cluster_reviews[cluster_idx]['rating']

        # Encoding IDs
        userId_encoder = Encoder(X['userId'])
        movieId_encoder = Encoder(X['movieId'])
        X['userId'] = X['userId'].map(userId_encoder.original2encoded)
        X['movieId'] = X['movieId'].map(movieId_encoder.original2encoded)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, shuffle=True, random_state=42)  # type:ignore
        # Training the model with the split data
        history = model.fit(x=[X_train['userId'], X_train['movieId']], y=y_train,  # type:ignore
                            epochs=n_epochs, verbose=1,
                            validation_data=([X_test['userId'], X_test['movieId']], y_test))  # type:ignore
        scores = model.evaluate(x=[X_test['userId'], X_test['movieId']], y=y_test)  # type:ignore
        cluster_scores_mae.append(scores[1])
        plt.plot(history.history['MAE'])  # type:ignore
        plt.plot(history.history['val_MAE'])  # type:ignore
        plt.ylabel('Mean Absolute Error (MAE)')
        plt.xlabel('Epoch')
        plt.legend(['Training data', 'Testing data'])
        plt.title(f'Standard K-Means, {k} clusters, # {cluster_idx}')
        plt.savefig(f'images/kmeans/models/{k}-mae-{cluster_idx}.png')
        plt.show()
    k_clusters_mae = np.mean(cluster_scores_mae)
    # k_clusters_rmse = np.mean(cluster_scores_rmse)
    standard_kmeans_cluster_scores_mae.append(k_clusters_mae)
    # standard_kmeans_cluster_scores_rmse.append(k_clusters_rmse)

for k, cluster in zip(range(2, 6), standard_kmeans_cluster_scores_mae):
    print(f'{k} clusters MAE: {cluster}')

# %% Running Keras model on custom K-Means


def get_custom_ratings(cluster: pd.Series):
    """Returns a DataFrame of ratings of movies within the cluster"""
    movies_in_cluster = cluster.index[cluster.values].to_numpy(dtype=int)  # type:ignore
    return filtered_ratings[filtered_ratings['movieId'].isin(movies_in_cluster)]


custom_kmeans_cluster_scores_mae = []
custom_kmeans_cluster_scores_rmse = []

for k in range(2, 6):
    k_idx = k - 2
    current_cluster = pd.DataFrame(
        clusters_custom[k_idx], columns=genres_table['movieId'])  # type:ignore
    clusters = current_cluster.apply(get_custom_ratings, axis=1).tolist()
    custom_cluster_scores_mae = []
    custom_cluster_scores_rmse = []
    print(f'Creating models for modified K-means with {k} clusters')
    for cluster_idx in range(k):
        print(f'Training cluster {cluster_idx+1}')
        cluster = clusters[cluster_idx]
        X = X = cluster[['userId', 'movieId']]
        y = cluster['rating']
        userId_encoder = Encoder(X['userId'])
        movieId_encoder = Encoder(X['movieId'])
        X['userId'] = X['userId'].map(userId_encoder.original2encoded)
        X['movieId'] = X['movieId'].map(movieId_encoder.original2encoded)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, shuffle=True, random_state=42)  # type:ignore
        model = create_model(data=cluster)
        plot_model(model, to_file=f'images/models/modified-{k}.png', show_shapes=True)
        history = model.fit(x=[X_train['userId'], X_train['movieId']], y=y_train,  # type:ignore
                            epochs=n_epochs, verbose=1,
                            validation_data=([X_test['userId'], X_test['movieId']], y_test))  # type:ignore
        scores = model.evaluate(x=[X_test['userId'], X_test['movieId']], y=y_test)  # type:ignore
        custom_cluster_scores_mae.append(scores[1])
        # custom_cluster_scores_rmse.append(scores[2])
        plt.plot(history.history['MAE'])  # type:ignore
        plt.plot(history.history['val_MAE'])  # type:ignore
        plt.ylabel('Mean Absolute Error (MAE)')
        plt.xlabel('Epoch')
        plt.legend(['Training data', 'Testing data'])
        plt.title(f'Modified K-Means, {k} clusters, # {cluster_idx}')
        plt.savefig(f'images/modified_kmeans/models/{k}-mae-{cluster_idx}.png')
        plt.show()
    k_clusters_mae = np.mean(custom_cluster_scores_mae)
    custom_kmeans_cluster_scores_mae.append(k_clusters_mae)

for k, cluster in zip(range(2, 6), custom_kmeans_cluster_scores_mae):
    print(f'{k} clusters MAE: {cluster}')

# %% Metrics

labels = [2, 3, 4, 5]
x = np.arange(len(labels))

width = 0.3

fig, ax = plt.subplots()
fig1 = ax.plot(standard_kmeans_cluster_scores_mae, label='Standard')
fig2 = ax.plot(custom_kmeans_cluster_scores_mae, label='Modified')

ax.set_title('Mean Absolute Error per clustering method')
ax.set_ylabel('Mean Absolute Error (MAE)')
ax.set_xticks(x)
ax.set_xlabel('Number of clusters')
ax.set_xticklabels(labels)
ax.legend()

fig.tight_layout()

plt.savefig('images/modified_vs_custom.png')
plt.show()


# %% Modifying the model from the journal paper

# Adding movie metadata to the neural network

def create_custom_model(data: pd.DataFrame):
    user = Input(shape=(1,))  # Single user
    movie = Input(shape=(1,))  # Single movie
    metadata = Input(shape=(20,))  # Encoded movie genres
    num_users = data['userId'].nunique()
    num_movies = data['movieId'].nunique()
    # Using average between number of movies and output layer as default hidden layer
    hidden_layer = round((num_movies + 1) / 2)
    # Creating embedding for the users
    users = Embedding(input_dim=num_users, output_dim=50)(user)
    # Creating embedding for movies
    movies = Embedding(input_dim=num_movies, output_dim=50)(movie)
    # The network must converge down to one 'branch' so we concatenate the two embeddings
    users_and_movies = Concatenate()([users, movies])
    # We need to flatten the concatenated data so that we can concatenate
    # again, as the number of dimensions needs to match in the items
    # being concatenated.
    users_and_movies = Flatten()(users_and_movies)
    # Including the movie metadata
    metadata_users_movies = Concatenate()([users_and_movies, metadata])
    # Creating a regular layer
    dense_layer = Dense(units=hidden_layer, activation=activations.relu)(metadata_users_movies)
    # The final layer is the output which has a single output neuron
    output = Dense(units=1)(dense_layer)
    model = Model(inputs=[user, movie, metadata], outputs=output)
    model.compile(optimizer=optimizers.Adam(),
                  loss='MAE',
                  metrics=['MAE'])
    return model

# %% Adding features to the ratings


encodings = genres_table['genres'].str.split('|').explode()
crosstab = pd.crosstab(encodings.index, encodings)
encoded = genres_table.drop(columns=['title', 'genres']).join(crosstab)


# %% Training The custom model


def get_with_metadata(cluster: pd.Series):
    """Returns a DataFrame of ratings of movies within the cluster, including metadata"""
    movies_in_cluster = cluster.index[cluster.values].to_numpy(dtype=int)  # type:ignore
    ratings = filtered_ratings[filtered_ratings['movieId'].isin(movies_in_cluster)]
    return ratings.join(encoded.set_index('movieId'), on='movieId')


# %%

custom_model_scores_mae = []
custom_model_scores_rmse = []

for k in range(2, 6):
    k_idx = k - 2
    current_cluster = pd.DataFrame(
        clusters_custom[k_idx], columns=genres_table['movieId'])  # type:ignore
    clusters = current_cluster.apply(get_with_metadata, axis=1).tolist()
    custom_cluster_scores_mae = []
    custom_cluster_scores_rmse = []
    print(f'Creating custom models for modified K-means with {k} clusters')
    for cluster_idx in range(k):
        print(f'Training cluster {cluster_idx+1}')
        cluster = clusters[cluster_idx]
        X = cluster[['userId', 'movieId']]
        X = pd.concat([X, cluster.iloc[:, 3:]], axis=1)
        y = cluster['rating']
        userId_encoder = Encoder(X['userId'])
        movieId_encoder = Encoder(X['movieId'])
        X['userId'] = X['userId'].map(userId_encoder.original2encoded)
        X['movieId'] = X['movieId'].map(movieId_encoder.original2encoded)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, shuffle=True, random_state=42)  # type:ignore
        model = create_custom_model(data=cluster)
        plot_model(model, to_file=f'images/models/custom-{k}.png', show_shapes=True)
        history = model.fit(x=[X_train['userId'], X_train['movieId'], X_train.iloc[:, 2:]],  # type:ignore
                            y=y_train,  # type:ignore
                            epochs=n_epochs, verbose=1,
                            validation_data=([X_test['userId'], X_test['movieId'],  # type:ignore
                                              X_test.iloc[:, 2:]], y_test))  # type:ignore
        scores = model.evaluate(x=[X_test['userId'], X_test['movieId'],  # type:ignore
                                   X_test.iloc[:, 2:]], y=y_test)  # type:ignore
        custom_cluster_scores_mae.append(scores[1])
        plt.plot(history.history['MAE'])  # type:ignore
        plt.plot(history.history['val_MAE'])  # type:ignore
        plt.ylabel('Mean Absolute Error (MAE)')
        plt.xlabel('Epoch')
        plt.legend(['Training data', 'Testing data'])
        plt.title(f'Custom model, {k} clusters, # {cluster_idx}')
        plt.savefig(f'images/custom/{k}-mae-{cluster_idx}.png')
        plt.show()
    k_clusters_mae = np.mean(custom_cluster_scores_mae)
    custom_model_scores_mae.append(k_clusters_mae)

for k, cluster in zip(range(2, 6), custom_model_scores_mae):
    print(f'{k} clusters MAE: {cluster}')

# %% Final plot

labels = [2, 3, 4, 5]
x = np.arange(len(labels))

fig, ax = plt.subplots()
fig1 = ax.plot(standard_kmeans_cluster_scores_mae, label='Standard+AGNN')
fig2 = ax.plot(custom_kmeans_cluster_scores_mae, label='Modified+AGNN')
fig3 = ax.plot(custom_model_scores_mae, label='Custom Model')

ax.set_title('Mean Absolute Error per clustering method')
ax.set_ylabel('Mean Absolute Error (MAE)')
ax.set_xticks(x)
ax.set_xlabel('Number of clusters')
ax.set_xticklabels(labels)
ax.legend()

fig.tight_layout()

plt.savefig('images/final_comparison_mae.png')
plt.show()
