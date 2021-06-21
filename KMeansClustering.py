# %% Imports

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import copy
import numpy as np
import seaborn as sns
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
        square_dist = sum((t - c) ** 2)
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

pca = PCA(n_components=2)  # type: ignore

pcaDf = pca.fit_transform(encodedDf)
pcaDf

scaler = MinMaxScaler()

scaledDf = scaler.fit_transform(pcaDf)
scaledDf

plt.scatter(scaledDf[:, 0], scaledDf[:, 1])

# %%
