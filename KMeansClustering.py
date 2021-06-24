# %% Imports

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import copy
import numpy as np
import matplotlib.pyplot as plt

# %% Classes


class Cluster:
    """A cluster of datapoints. Contains the centroid location and points associated with the cluster."""
    # The cluster centroid location
    centroid: np.ndarray
    # The threshold value used to check whether a point should be assigned to this cluster.
    threshold: float = 0.0
    # List of data points associated with the cluster.
    points: list[np.ndarray]
    # List of distances between each data point and this cluster.
    distances: list[float]

    def __init__(self, centroid: np.ndarray) -> None:
        self.centroid = centroid
        self.points = []
        self.distances = []

    def __repr__(self) -> str:
        x = self.centroid[0]
        y = self.centroid[1]
        return f'Cluster(centroid=[{x}, {y}])'


class ModifiedKMeans:
    """Modified k-means clustering algorithm. Takes in a number of clusters."""

    def __init__(self, n_clusters: int):
        """n_clusters: number of clusters to create."""

        self.n_clusters: int = n_clusters
        self.clusters: list[Cluster] = []

    def _set_centroids(self, data_points: np.ndarray) -> None:
        """Picks random data points as cluster centroids and assigns them."""
        # Get the number of data points
        row_count = data_points.shape[0]
        # Pick n random indexes to select rows as starting centroids
        centroids = np.random.choice(
            row_count, size=self.n_clusters, replace=False)
        # Picking the datapoints as centroids
        centroids = data_points[centroids, :]
        # Assigning to the Cluster dataclass
        for i, centroid in enumerate(centroids):
            self.clusters.append(Cluster(centroid))
            self.clusters[i].centroid = centroid

    def _euclidean_dist(self, t: np.ndarray, c: np.ndarray) -> float:
        """Calculates the Euclidean Distance between datapoint t and centroid c."""
        square_dist = np.sum((t - c) ** 2)
        return np.sqrt(square_dist)

    def _k_means_objective(self, data_points: np.ndarray):
        """Uses the initial objective function to set new clusters for the first iteration of the clustering process."""
        for point in data_points:
            smallest_dist = np.Infinity
            closest = -1
            for i, cluster in enumerate(self.clusters):
                distance = self._euclidean_dist(point, cluster.centroid)
                if distance < smallest_dist:
                    smallest_dist = distance
                    closest = i
                self.clusters[i].distances.append(distance)
            self.clusters[closest].points.append(point.tolist())

    def _calc_threshold(self, distances: list[float]) -> float:
        return sum(distances) / len(distances)

    def _modified_k_means_objective(self, data_points: np.ndarray):
        """The modified objective function which uses thresholds to assign points to the clusters."""
        for point in data_points:
            assigned = False
            smallest_dist = np.Infinity
            closest = -1
            for i, cluster in enumerate(self.clusters):
                distance = self._euclidean_dist(point, cluster.centroid)
                self.clusters[i].distances.append(distance)
                if distance < smallest_dist:
                    smallest_dist = distance
                    closest = i
                if distance <= cluster.threshold:
                    self.clusters[i].points.append(point.tolist())
                    assigned = True
            if not assigned:
                self.clusters[closest].points.append(point.tolist())

    def _new_centroid(self, points: list[np.ndarray]) -> np.ndarray:
        """Calculates the new cluster centroid based on the points in the cluster."""
        x, y = 0, 0
        for point in points:
            x += point[0]
            y += point[1]
        n_points = len(points)
        x /= n_points
        y /= n_points
        return np.asarray([x, y])

    def cluster_data(self, iters: int, data_points: np.ndarray) -> list[Cluster]:
        """Clusters the dataset using the modified k-means clustering algorithm."""
        self._set_centroids(data_points)
        self._k_means_objective(data_points)
        for iteration in range(iters):
            old_clusters = copy.deepcopy(self.clusters)
            for j, cluster in enumerate(self.clusters):
                # Assigning a new threshold to each cluster
                self.clusters[j].threshold = self._calc_threshold(
                    cluster.distances)
                self.clusters[j].centroid = self._new_centroid(
                    cluster.points)
                self.clusters[j].distances = []
                self.clusters[j].points = []
            self._modified_k_means_objective(data_points)
            for old_cluster, new_cluster in zip(old_clusters, self.clusters):
                if old_cluster.points != new_cluster.points:
                    # Keep calculating new clusters until no datapoint is reassigned
                    break
            else:
                # No datapoint was reassigned, so finish
                print('Finished in {} iterations'.format(iteration+1))
                break
        else:
            # If the iteration limit is reached
            print('Iteration limit reached')
        return self.clusters


# %% Reading the movies file

print('Reading movies CSV')
moviesDf = pd.read_csv('./data/movies.csv')
moviesDf

# %% Encoding genres

# This splits the genres from a single string a list of genres.
encodings = moviesDf['genres'].str.split('|').explode()
crossedDf = pd.crosstab(encodings.index, encodings)
encodedDf = moviesDf.drop(columns=['title', 'genres']).join(crossedDf)
encodedDf

# %%

# encodedDf = pd.get_dummies(genresDf, prefix='genre').groupby(level=0).sum()
# encodedDf

# %%

# pca = PCA(n_components=2)  # type: ignore

# pcaDf = pca.fit_transform(encodedDf)
# pcaDf

# scaler = MinMaxScaler()

# scaledDf = scaler.fit_transform(pcaDf)
# scaledDf

# plt.scatter(scaledDf[:, 0], scaledDf[:, 1])

# %%

# Reading the filtered ratings table

ratings_table = pd.read_csv('data/filtered_ratings.csv')
ratings_table


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

# Clustering using regular KMeans


# kmeans = KMeans(n_clusters=5)
# kmeans.fit_transform(encoded_movie_genres)
# kmeans.cluster_centers_

scaler = MinMaxScaler()

scaled = scaler.fit_transform(crosstab)

pca = PCA(n_components=2)
pca_movie_genres = pca.fit_transform(scaled)
pca_movie_genres

# plt.scatter(x=pca_movie_genres[:, 0], y=pca_movie_genres[:, 1])

# kmeans = KMeans(n_clusters=6)
# pca_clusters = kmeans.fit_predict(pca_movie_genres)

# pca_clusters

# plt.scatter(x=pca_movie_genres[:, 0], y=pca_movie_genres[:, 1], c=pca_clusters)  # type:ignore

# cluster_centers = kmeans.cluster_centers_

# plt.scatter(x=cluster_centers[:, 0], y=cluster_centers[:, 1],
#             marker='^', color='k', edgecolors='w')  # type:ignore

# %%

# 2nd attempt at the modified k-means


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
            print(f'Finished iteration {iteration+1}')
        else:
            # If the iteration limit is reached
            print('\nIteration limit reached')
        return self.points

# %%

# Using the custom clusteirng algorithm


n_clusters = 6

modified_kmeans = ModifiedKMeansV2(n_clusters=n_clusters)

clusters = modified_kmeans.cluster_data(iters=100, data_points=pca_movie_genres)
clusters

# %%

# Getting the points in each cluster
cluster_points = [pca_movie_genres[cluster] for cluster in clusters]
centroids = modified_kmeans.centroids

# %%

for cluster in cluster_points:
    plt.scatter(cluster[:, 0], cluster[:, 1])

plt.scatter(x=centroids[:, 0], y=centroids[:, 1], marker='^',  # type:ignore
            color='k', edgecolors='w')

plt.xlabel('Principal component 1')
plt.ylabel('Principal component 2')

# %%

fig, axs = plt.subplots(n_clusters, figsize=(10, 6 * n_clusters))
for i, cluster in enumerate(cluster_points):
    axs[i].scatter(cluster[:, 0], cluster[:, 1])
    axs[i].scatter(centroids[i, 0], centroids[i, 1], marker='^', color='k', edgecolors='w')
    axs[i].title.set_text(f'Cluster {i+1}')

plt.savefig('images/separate.png', transparent=False)

# %%

# Saving data to csv

df = pd.DataFrame(clusters, columns=genres_table['movieId'])  # type:ignore
df.to_csv('data/clustered_movies.csv')
