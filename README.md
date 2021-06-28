# Artificial Genetic Neural Network algorithm

This repository contains a modified version of a novel adaptive genetic neural network (AGNN) model for recommender systems using modified k-means clustering approach by Selvi and Sivasankar ([2019](#Citations)).

## Running the code

### Installation

Please note: These instructions were only tested on Ubuntu 20.04 focal (on the Windows Subsystem for Linux). **This code may not work on Windows**.

Install Python 3.9.1. This code was executed on the 64-bit version.

Next, install the requirements in the `requirements.txt` file. As there are a lot of dependicies in this project, it may be worth using a virtual environment.

Using Pip, the packages can be nstalled by using the following command:

```bash
pip install -r requirements.txt
```

Alternatively, the dependecies may be installed using Conda:

```bash
conda create --name comparison --file requirements.txt
```

_This will install the appropriate version of Python along with all needed modules to run the code._

Then use the following command to activate the environment:

```bash
conda activate comparison
```

## Executing the code

Once the environment is activated, any Python script can be executed using the `python` command instead of the usual `python3` command, because of the conda environment. However, if you do not use Conda, use any command that invokes Python 3.9.1.

The first file that requires to be executed is the `ratings_matrix.py` file, which requires that the MovieLens dataset is present in the `data/` folder.

Then, the `similarity_and_filtering.py` file may be executed.

Lastly, the `final.py` file may be executed.

### Dataset

This research uses the MovieLens small (100K) dataset, which can be downloaded from the [GroupLens](https://grouplens.org/datasets/movielens/) website. Extract the dataset, and place the `ratings.csv` file in the `data/` directory in this directory.

## Citations

Selvi, C., & Sivasankar, E. (2019). A novel adaptive genetic neural network (AGNN) model for recommender systems using modified k-means clustering approach. _Multimedia Tools and Applications, 78_(11), 14303-14330. https://doi.org/10.1007/s11042-018-6790-y
