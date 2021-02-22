# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import multiprocessing as mp
from IPython import get_ipython

# %% [markdown]
# # AGNN RS model implementation
# This file implements the Artificial Genetic Neural Network recommender system, using the [AGGN journal](journals/AGNN.pdf).
# %% [markdown]
# ## 2. Collect dataset
#
# The rating matrix has been generated using the [ratings_matrix.ipynb](./ratings_matrix.ipynb) script.

# %%
# 2: Collect [r]x*y dataset;
import pandas as pd

ratings_matrix = pd.read_csv('data/ratings_matrix.csv')
# Limiting number of cols. as it would take a week to run.
ratings_matrix = ratings_matrix.iloc[:, :100]
ratings_matrix
# %% [markdown]
# ## 3. Calculate `sim(om_s, om_t)` using (1);
#
# ![Similarity function](images/eq_1.png)

# %%
# 3: Calculate sim(om_s, om_t) using (1);


def sim(om_s: pd.DataFrame, om_t: pd.DataFrame) -> float:
    """Generates a similarity value between two online movies."""

    # Merging the two movies so we can find common users who rated both
    temp_om = pd.concat([om_s, om_t], axis=1)

    # Dropping rows with NaNs as this yields movies with common ratings
    common_om = temp_om.dropna()

    # Checking if there are common users
    if common_om.empty:
        # If there are no users, return
        return None

    # Splitting the data again
    om_s = common_om.iloc[:, 0]
    om_t = common_om.iloc[:, 1]

    # Getting similarity
    similarity = sum(om_s * om_t) / sum(om_s + om_t)
    return similarity


# %%
# Dropping the `userId` column, as it is not necessary for this step.
ratings_matrix.drop('userId', axis=1, inplace=True)
ratings_matrix.head()


# %%

def avg_sim(batch_matrix):
    """Calculates the average movie similarity."""
    similarities_list = []  # Where the average similarities will be stored.

    # Iterating over every movie
    for om_s_id, om_s in batch_matrix.iteritems():

        if om_s.empty:
            continue

        similarities = []
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
        print(f"Done movieId {om_s_id}")

    return similarities_list


if __name__ == "__main__":
    cpu_count = mp.cpu_count()

    batch_matrices = np.array_split(ratings_matrix, cpu_count, axis=1)

    pool = mp.Pool(cpu_count)
    similarities_list = pool.map(avg_sim, batch_matrices)
    similarities_list

# %%

# Flattening the list (it is currently a list of lists.)
similarities_list = [item for sublist in similarities_list for item in sublist]
similarities_list

# %%

# Sorting the list
similarities_list = sorted(similarities_list, key=lambda tupl: tupl[1])
similarities_list

# %%

# Dropping movies with om_s <= Outlier (0.3 in the paper.)
outlier = 0.3
similarities_list = [item for item in similarities_list if item[1] > outlier]
similarities_list