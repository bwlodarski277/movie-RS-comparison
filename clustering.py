# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from KMeansClustering import Cluster as cluster, ModifiedKMeans
import random
from typing import Any, List, Union
from IPython import get_ipython

# %%
import matplotlib.pyplot as plt
from IPython import get_ipython
from pandas.core.series import Series


# %%
import seaborn as sns
import numpy as np
import multiprocessing as mp
from IPython import get_ipython

# %% [markdown]
#   # AGNN RS model implementation
#   This file implements the Artificial Genetic Neural Network recommender system, using the [AGGN journal](journals/AGNN.pdf).
# %% [markdown]
#   ## 2. Collect dataset
#
#   The rating matrix has been generated using the [ratings_matrix.ipynb](./ratings_matrix.ipynb) script.

# %%

# Reading the ratings matrix

import pandas as pd

ratings_matrix = pd.read_csv('data/ratings_matrix.csv')
# Limiting number of cols. as it would take a week to run.
ratings_matrix = ratings_matrix.iloc[:, :100]
ratings_matrix

# %% [markdown]
#   ## 3. Calculate `sim(om_s, om_t)` using (1);
#
#   ![Similarity function](images/eq_1.png)

# %%

# 3: Calculate sim(om_s, om_t) using (1);


def sim(om_s: Series, om_t: Series) -> Union[float, None]:
    """Generates a similarity value between two online movies."""

    # Merging the two movies so we can find common users who rated both
    temp_om: Any = pd.concat([om_s, om_t], axis=1)

    # Dropping rows with NaNs as this yields movies with common ratings
    common_om = temp_om.dropna()

    # Checking if there are common users
    if common_om.empty:
        # If there are no users, return
        return None

    # Splitting the data again
    om_s = common_om.iloc[:, 0]  # type: ignore
    om_t = common_om.iloc[:, 1]  # type: ignore

    # Getting similarity
    similarity = sum(om_s * om_t) / sum(om_s + om_t)
    return similarity


# %%

# Function to calculate the average similarity for each movie in the matrix.

def avg_sim(batch_matrix: pd.DataFrame):
    """Calculates the average movie similarity."""
    similarities_list = []  # Where the average similarities will be stored.

    # Iterating over every movie
    for om_s_id, om_s in batch_matrix.iteritems():

        if om_s.empty:
            continue

        similarities: List[float] = []
        # Iterating over movies to check similarity against
        for om_t_id, om_t in ratings_matrix.iteritems():

            # Making sure we don't check the similarity of a movie against itself
            if om_s_id == om_t_id:
                continue

            similarity = sim(om_s, om_t)

            # Ignore the movie if there are no common users
            if similarity is None:
                continue

            similarities.append(similarity)

        # Calculating average similarity
        avg_similarity = sum(similarities) / len(ratings_matrix.columns)

        # Appending ID and avg. sim. to list
        similarities_list.append((om_s_id, avg_similarity))
        # print(f"Done movieId {om_s_id}")

    return similarities_list


# Making sure we have access to the original ratings matrix
original_ratings_matrix = ratings_matrix.copy()


# %%
# Dropping the `userId` column, as it is not necessary for this step.

ratings_matrix.drop('userId', axis=1, inplace=True)
ratings_matrix.head()


# %%

similarities_list = []

# Processing similarities in parallel
cpu_count = mp.cpu_count()

batch_matrices = np.array_split(ratings_matrix, cpu_count, axis=1)

pool = mp.Pool(cpu_count)
similarities_list = pool.map(avg_sim, batch_matrices)
similarities_list

print('Completed processing similarities.')


# %%

# Flattening the list (it is currently a list of lists.)
similarities_list = [item for sublist in similarities_list for item in sublist]
print(similarities_list)


# %%

# Sorting the list

similarities_list = sorted(
    similarities_list, key=lambda tupl: tupl[1], reverse=True)
print(similarities_list)


# %%

# Dropping movies with om_s <= Outlier (0.3 in the paper.)
# First step: get movie IDs below or at outlier to drop.

outlier = 0.3
list_to_drop = [item for item in similarities_list if item[1] <= outlier]
print(list_to_drop)


# %%

# Order ratings matrix based on similarity

new_order = [item[0] for item in similarities_list]
print(new_order)


# %%

# Ordering the matrix

# Making sure User ID is always the first column
new_order.insert(0, 'userId')

ordered_matrix = original_ratings_matrix.reindex(
    new_order, axis=1)  # type: ignore
ordered_matrix


# %%

# Dropping columns with similarity lower than threshold

cols_to_drop = [item[0] for item in list_to_drop]

filtered_matrix = ordered_matrix.drop(cols_to_drop, axis=1)

filtered_matrix

# %%

# Creating table from matrix (?)

filtered_table = filtered_matrix.melt(
    id_vars=['userId'], var_name="movieId", value_name="rating").dropna()

filtered_table['userId'] = filtered_table['userId'].astype(int)  # type: ignore
filtered_table['movieId'] = filtered_table['movieId'].astype(int)  # type: ignore

order_col = [i for i in range(filtered_table.shape[0])]

filtered_table.insert(loc=0, column='orderedMovies', value=order_col)

filtered_table


# %%

# Plotting the table

sns.relplot(data=filtered_table, x='movieId', y='userId',
            hue='rating', row_order='orderedMovies')


# %%

# Performing k-means clustering


# Number of clusters
n_clusters = 5
# Converting table to NumPy array
data_points = filtered_table[['userId', 'movieId']].to_numpy()
data_points


k_means = ModifiedKMeans(6)

# Making sure results are reproducible
np.random.seed(seed=1)

clusters: list[Any] = k_means.cluster_data(iters=100, data_points=data_points)
clusters

# %%

colours = ['tab:blue', 'tab:orange', 'tab:green',
           'tab:red', 'tab:purple', 'tab:olive']

fig, axs = plt.subplots(3, 2, figsize=(12, 8))
for i, cluster in enumerate(clusters):
    x = i % 3
    y = i % 2
    points = np.asarray(cluster.points)
    axs[x, y].scatter(points[:, 0], points[:, 1], color=colours[i])
    axs[x, y].scatter(cluster.centroid[0], cluster.centroid[1],
                      marker="x", color=colours[-i])

# fig.delaxes(axs[2, 1])

fig.tight_layout()

plt.show()


# %%

# Get number of rows
row_count = data_points.shape[0]
# Pick n random rows
centroids = np.random.choice(row_count, size=n_clusters, replace=False)
# Select these rows
centroids = data_points[centroids, :]

print('Initial centroids:')
print(centroids)


# %%

# Simple function to calculate ED

def euclidean_dist_sq(t, c):
    return sum((t - c) ** 2)


def euclidean_dist(t, c):
    return np.sqrt(euclidean_dist_sq(t, c))


# %%

# Calculating clusters

# List of of clusters. Each cluster is a list of data points in that cluster.
clusters = [[] for _ in range(n_clusters)]

# List of distances between data points and each cluster.
distances: list[list] = [[] for _ in range(n_clusters)]


def cluster_data():
    # Calculate the ED between each data point and cluster centroids as given in (3)
    for point in data_points:
        smallest_dist = np.Infinity
        closest = -1
        for idx, centroid in enumerate(centroids):
            dist = euclidean_dist_sq(point, centroid)
            if dist < smallest_dist:
                smallest_dist = dist
                closest = idx
            distances[idx].append(dist)
        # Allocate data point to the cluster whose centroid is closest
        clusters[closest].append(point.tolist())

# clusters = np.array(clusters)

# %%


def calc_threshold(distances):
    return sum(distances) / len(distances)

# %%

# Calculate mean position of points assigned to cluster


def new_centroid(cluster):
    x, y = 0, 0
    for item in cluster:
        x += item[0]
        y += item[1]

    x /= len(cluster)
    y /= len(cluster)
    return [x, y]

# %%

# Clustering using modified k-means


thresholds = [0.0 for _ in range(n_clusters)]

cluster_data()

iterations = 0
while iterations < 100:
    iterations += 1

    old_clusters = clusters.copy()

    for j in range(n_clusters):

        # Calculating new thresholds and centroids
        thresholds[j] = calc_threshold(distances[j])
        centroids[j] = new_centroid(clusters[j])

        # print('New centroids:\n', centroids, end='\r')
        # smallest_dist = np.Infinity
        # closest = None

        # for point in data_points:
        #     dist = euclidean_dist_sq(point, centroids[j])
        #     distances[j].append(dist)

        #     if dist <= thresholds[j]:
        #         clusters[j].append(point.tolist())

    distances = [[] for _ in range(n_clusters)]
    clusters = [[] for _ in range(n_clusters)]

    for point in data_points:
        assigned = False

        smallest_dist = np.Infinity
        closest = -1

        for idx, centroid in enumerate(centroids):
            dist = euclidean_dist_sq(point, centroid)
            distances[idx].append(dist)
            if dist < smallest_dist:
                smallest_dist = dist
                closest = idx

            if dist <= thresholds[idx]:
                clusters[idx].append(point.tolist())
                assigned = True

        if not assigned:
            clusters[closest].append(point.tolist())

    if old_clusters == clusters:
        print('No clusters reassigned')
        break
else:
    print('Reached max. iterations')

print('New centroids:\n', centroids)
# %%

# Defining number of nodes in each layer

# The number of neurons in the input layer is equal to the number of
# distinct movies in the dataset after preprocessing.
n_input_layer = len(ratings_matrix.columns)

# The number of neurons in theoutput layer is one.
n_output_layer = 1

# the hidden layer is the mean value of neurons
# in the input and output layers.
n_hidden_layer = round((n_input_layer + n_output_layer) / 2)


# %%


def random_weights(n_neurons: int) -> list[float]:
    """Generates a list of random weights for a neuron layer."""
    return [random.uniform(-1, 1) for _ in range(n_neurons)]


# Maintain a unity value weight for all input layer neurons.
# (Assuming "unity value" means 1)
input_weights = [1 for _ in range(n_input_layer)]

# Generate arbitrary weights within an interval [-1, 1] and assign
# it to the hidden layer neurons and output layer neurons.
hidden_weights = random_weights(n_hidden_layer)
n_output_layer = random_weights(n_output_layer)


# %%

# Error rate function

def error_rate(predicted, actual):
    """Calculates the error rate of the ou"""
    return predicted * (1 - predicted) * (actual - predicted)

# %%
