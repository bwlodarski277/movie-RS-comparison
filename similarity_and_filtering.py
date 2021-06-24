# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import time
from IPython import get_ipython
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# %% [markdown]
#   # AGNN RS model implementation
#   This file implements the Artificial Genetic Neural Network recommender system, using the [AGGN journal](journals/AGNN.pdf).
# %% [markdown]
#   ## 2. Collect dataset
#
#   The rating matrix has been generated using the [ratings_matrix.ipynb](./ratings_matrix.ipynb) script.

# %%

# Reading the ratings matrix


ratings_matrix = pd.read_csv('data/ratings_matrix.csv')
# Limiting number of cols. as it would take a week to run.
# ratings_matrix = ratings_matrix.iloc[:, :100]
ratings_matrix

# %% [markdown]
#   ## 3. Calculate `sim(om_s, om_t)` using (1);
#
#   ![Similarity function](images/eq_1.png)

# %%

# 3: Calculate sim(om_s, om_t) using (1);


# @numba.njit
def sim(om_s: np.ndarray, om_t: np.ndarray):
    """Generates a similarity value between two online movies."""

    # Merging the two movies so we can find common users who rated both
    temp_om = np.stack((om_s, om_t))

    # Dropping rows with NaNs as this yields movies with common ratings
    # common_om = temp_om.dropna()
    # common_om = temp_om[:, ~np.isnan(temp_om).any(axis=0)]
    common_om = temp_om[:, ~np.any(np.isnan(temp_om), axis=0)]

    # Checking if there are common users
    if common_om.size == 0:
        # If there are no users, return
        return 0.

    # Splitting the data again
    # om_s = common_om.iloc[:, 0]  # type: ignore
    # om_t = common_om.iloc[:, 1]  # type: ignore
    om_s = common_om[0]
    om_t = common_om[1]

    # Getting similarity
    # similarity = sum(om_s * om_t) / sum(om_s + om_t)
    similarity = (om_s * om_t).sum() / (om_s + om_t).sum()
    return similarity


# %%
# Function to calculate the average similarity for each movie in the matrix.


# @numba.njit

def avg_sim(np_matrix: np.ndarray):
    """Calculates the average movie similarity."""
    similarities_list = []  # Where the average similarities will be stored.

    # Iterating over every movie
    for om_s_id, om_s in enumerate(np_matrix.T):

        # if om_s.empty:
        #     continue

        # if om_s.size == 0:
        #     continue

        # if np.all(np.isnan(om_s)):
        #     continue

        start = time.time()
        similarities = []
        # Iterating over movies to check similarity against
        for om_t_id, om_t in enumerate(np_matrix.T):

            # Making sure we don't check the similarity of a movie against itself
            if om_s_id == om_t_id:
                continue

            # Ignore the movie if there are no common users
            # if np.all(np.isnan(om_t)):
            #     continue

            similarity = sim(om_s, om_t)
            similarities.append(similarity)

        similarities_np = np.array(similarities, dtype=np.float32)
        # Calculating average similarity
        avg_similarity = similarities_np.mean()
        time_taken = time.time() - start
        print(
            f'movieId: {om_s_id}\t',
            f'avg: {avg_similarity:.2f}\t'
            f'time: {time_taken:.2f}')

        # Appending ID and avg. sim. to list
        similarities_list.append((om_s_id, avg_similarity))

    return np.array(similarities_list)


# Making sure we have access to the original ratings matrix
original_ratings_matrix = ratings_matrix.copy()


# %%
# Dropping the `userId` column, as it is not necessary for this step.

ratings_matrix.drop('userId', axis=1, inplace=True)
ratings_matrix

# %%

# Processing similarities in parallel

# if __name__ == "__main__":
# cpu_count = mp.cpu_count()

# batch_matrices = np.array_split(ratings_matrix, cpu_count, axis=1)

# pool = mp.Pool(cpu_count)
# similarities_list = pool.map(avg_sim, batch_matrices)
# similarities_list

# %%

np_matrix = ratings_matrix.to_numpy().astype(np.float32)

# multithreaded = False
# if multithreaded:
#     # Getting number of threds on processor
#     cpu_count = mp.cpu_count()

#     # Function for acquiring lock in the function
#     def init(_lock):
#         global lock
#         lock = _lock

#     lock = mp.Lock()
#     pool = mp.Pool(cpu_count, initializer=init, initargs=(lock,))
#     split_similarities_matrix = np.array_split(np_matrix, cpu_count, axis=0)
#     similarities_list = pool.map(avg_sim, split_similarities_matrix)
# else:
similarities_list = avg_sim(np_matrix)

similarities_list

print('\nCompleted processing similarities.')

# %%

# Creating a list containing column names and their respective similarities

column_similarity = np.array(list(zip(ratings_matrix.columns, similarities_list[:, 1])))
column_similarity

np.savetxt('data/similarities_index.csv', similarities_list, fmt='%d,%f', delimiter=',')

np.savetxt('data/similarities.csv', column_similarity,
           fmt='%s,%s', delimiter=',')

# %%

# (not necessary since moving to numpy)
# Flattening the list (it is currently a list of lists.)
# similarities_list = [item for sublist in similarities_list for item in sublist]

# similarities_list

# %%

# Sorting the list

# similarities_list = sorted(
#     similarities_list, key=lambda tupl: tupl[1], reverse=True)
# pprint(similarities_list)

# Using numpy to sort the array
# similarities_order = similarities_list[:, 1].argsort()[::-1]
# sorted_similarities = similarities_list[similarities_order]

# sorted_similarities

# %%

# Dropping movies with om_s <= Outlier (0.3 in the paper.)
# First step: get movie IDs below or at outlier to drop.

# outlier = 0.3
# movies_to_drop = similarities_list[similarities_list[:, 1] < outlier]

# movies_to_drop[:, 0].astype(int)

# list_to_drop = [item for item in similarities_list if item[1] <= outlier]
# print(list_to_drop)

# %%

# Sorting the columns in the matrix

# similarities_order = similarities_list[:, 1].argsort()[::-1]
# similarities_order = column_similarity

# sorted_matrix = ratings_matrix.iloc[:, similarities_order]
# sorted_matrix

# %%

# Sorting the similarities list
# So that we can then sort the columns in the matrix

similarities_order = column_similarity[:, 1].argsort()[::-1]
column_similarity = column_similarity[similarities_order]

column_similarity

# %%

new_column_order = column_similarity[:, 0]

sorted_matrix = ratings_matrix[new_column_order]
sorted_matrix

# %%

# Adding back the user IDs

sorted_matrix.insert(loc=0, column='userId', value=original_ratings_matrix['userId'])
sorted_matrix

# %%

# Dropping columns with similarity lower than threshold

outlier = 0.3  # The journal used this as the threshold value

filtered_column_similarity = column_similarity[column_similarity[:, 1].astype(float) >= outlier]
filtered_column_similarity = np.insert(filtered_column_similarity, 0, 'userId', axis=0)
filtered_matrix = sorted_matrix[filtered_column_similarity[:, 0]]
filtered_matrix

# %%

# Dropping columns with similarity lower than threshold

# cols_to_drop = [item[0] for item in list_to_drop]

# filtered_matrix = ordered_matrix.drop(cols_to_drop, axis=1)

# filtered_matrix

# outlier = 0.3
# # Adding 1 to offset the userId column
# movies_to_drop = similarities_list[similarities_list[:, 1] < outlier] + 1

# movies_to_drop = movies_to_drop[:, 0].astype(int)

# filtered_matrix = sorted_matrix.drop(movies_to_drop, axis=1)
# filtered_matrix

# column_to_id =

# %%

# Plotting the results


get_ipython().run_line_magic('matplotlib', 'inline')

# Excluding the user ID from the plot
plt.matshow(filtered_matrix.drop(labels='userId', inplace=False, axis=1))

plt.xlabel('Movie IDs')
plt.ylabel('User IDs')
bar = plt.colorbar()
bar.set_label('Movie rating (0.5-5 stars)')

# %%

# Creating table from matrix

filtered_table = filtered_matrix.melt(
    id_vars=['userId'], var_name='movieId', value_name='rating').dropna()

filtered_table['userId'] = filtered_table['userId'].astype(int)
filtered_table['movieId'] = filtered_table['movieId'].astype(int)

filtered_table

# %%
# Saving the filtered file to a CSV

filtered_table.to_csv(path_or_buf='data/filtered_ratings.csv', index=False)

# %%
