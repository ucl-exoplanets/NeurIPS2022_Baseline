""" Documents the metric for the Light Track in NeurIPS 2022 competition """

import numpy as np

def scaled_SE(x, x_hat):
    rel_diff = (x- x_hat)/x 
    val = np.square(rel_diff)
    return val

def light_track_metric(x, x_hat, k =1000):
    """
    RMSE based Metric for light track. Compare quartiles between MCMC-based methods and model output"
    x: The reference quartiles generated from a MCMC technique (N, 3, num_targets,)
    x_hat: The quartiles predicted by  ( N, 3, num_targets,)
    k: constant , used to adjust the magnitude of the score. Default = 10
    
    """
    SE_array = scaled_SE(x, x_hat)
    MSE_array = np.mean(SE_array, axis=(1,2))
    RMSE_array = np.sqrt(MSE_array)
    sum_array = np.sum(RMSE_array)
    score = 1000 - k*(sum_array/(x.shape[0]))
    print("score is:",score)
    return score

def load_Quartile_Table(path, order= None):
    """Read quartiles information from Quartiles Table and generate a 3D matrix 

    Args:
        path (string): path to quartiles table
        order (list, optional): order of the parameters, there is a default order if order is not given Defaults to None.

    Returns:
        _type_: quartiles matrix used for calculating the light track metric (N, 3, num_targets)
    """
    import pandas as pd
    quartiles = pd.read_csv(path)
    if order is None:
        targets = ['T','log_H2O', 'log_CO2','log_CH4','log_CO','log_NH3']
    else:
        targets = order
    quartiles_matrix =  np.zeros((len(quartiles), 3, len(targets)))
    for t_idx, t in enumerate(targets):
        for q_idx, q in enumerate(['q1','q2','q3']):
            quartiles_matrix[:,q_idx, t_idx, ] = quartiles.loc[:,t + '_' + q]
    return quartiles_matrix

def get_all_q(df):
    """get the GT values from the test set"""
    all_q_label = ['q1','q2','q3']
    all_q = np.zeros((3, len(df), 6))
    for idx, label in enumerate(all_q_label):
        q_matrix = df.loc[:, df.columns.str.endswith(label)].values
        all_q[idx] = q_matrix
    return all_q