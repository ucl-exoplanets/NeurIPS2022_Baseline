# Baseline solution for NeurIPS 2022 Ariel Data Challenge

Inside this repo you will find the baseline solution for the Ariel Data Challenge. To run the script you will need access to the training and test data, both of which can be found here. 
There are two ways to run the baseline:

1. via command line:
```
python baseline_MCDropout.py --training PATH/TO/TRAININGDATA/ --test PATH/TO/TESTDATA
```

2. via jupyter notebook, baseline - MCDropout-Public.ipynb

## Description
We trained a neural network to perform a supervised multi-target regression task. The architecture of the network is modified from the CNN network as described in [Yip et al.](https://arxiv.org/abs/2011.11284). 

## Preprocessing Steps
- We used the first 5000 data instances to train the model
- We augmented the data with the observation noise
- Used stellar and planetary radii as additional features
- Standardised both inputs and output

At test time we performed [Monte Carlo Dropout](https://arxiv.org/abs/1506.02142) to provide a mutlivariate distribution for each test example. Samples from the mutlivariate distribution is submitted to the regular track. Quartiles estimates are extracted from the same distribution to submit to the light track.

## Metrics
We have inlcuded the metric we used to compute score for light track and regular track. Please note that the regular could be quite slow. We have used the [POT](https://pythonot.github.io/index.html) python package to compute the Wessestein-2 distance. 

## Things to improve
There are different direction to take from here on, let us summarise the shortcoming of this model:
- The data preprocessing is quite simplitic and could have invested more efforts.
- we have only used 5000 data points, instead of the full dataset
- we didnt train the model with results from the retrieval (QuartilesTable.csv for Light Track and Tracedata.hdf5 for Regular Track), which are the GT for this competition.
- The conditional distribution from MCDropout is very restricted and Gaussian-like
- So far we havent considered the atmospheric targets as a joint distribution
- We have only used stellar radius and planet radius from the auxillary information
- We have not done any hyperparameter tuning 
- the train test split here is not clean, as in, we split the data after we have augmented the data, which results in information leakage to the validation data. There is no leakage to the test data though.
