# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Generating a ratings matrix
# ## Reading data and preparing it
# Here we will read the data and clean it up by dropping unnecessary columns before converting it to a matrix.

# %%
# Reading the data
import pandas as pd

ratings_table = pd.read_csv('data/ratings.csv')
ratings_table.head()


# %%
# Dropping timestamp as it is unnecessary
ratings_table.drop(['timestamp'], axis=1, inplace=True)
ratings_table.head()

# %% [markdown]
# ## Converting to matrix
# Here we will convert the table into a matrix, where each column will be a movie and each row will be a user's rating.

# %%
ratings_matrix = ratings_table.pivot(index='userId', columns='movieId', values='rating')
ratings_matrix

# %% [markdown]
# ## Exporting as CSV
# Self-explanatory. The file will be imported in both RS models.

# %%
ratings_matrix.to_csv('ratings_matrix.csv')


