""" Documents the metric for the Regular Track in NeurIPS 2022 competition """
import numpy as np
import ot
import h5py
from tqdm import tqdm
"""
-----Main Functions----
W2 - Wessestein-2 distance (implemented by POT package https://pythonot.github.io/)

"""
def batch_calculate(trace1_matrix, trace1_weights_matrix, trace2_hdf5, id_order = None):
    """Calculate the score for regular track from a (predicted) trace matrix and a Gound Truth hdf5 file

    Args:
        trace1_matrix (_type_): prediction from the model (N X M X 6), assumed to have the same number of trace for every examples
        trace1_weights_matrix: Weight for trace1 matrix (N X M), where N is the number of examples and M is the number of traces. Each row should sum to 1.
        trace2_hdf5 (_type_): GT trace data from a hdf5 file (do not need to open)
        id_order (arr, optional): Order of the planets (N). Defaults to None.

    Returns:
        float: score for regular track
    """
    # hdf5 requires knowledge on the order of the planets (in terms of planet id)
    # If unspecified, it will assume ascending order from 1 to N. 
    trace2 = h5py.File(trace2_hdf5,'r')
    if id_order is None:
        id_order = np.arange(len(trace1_matrix))
    else:
        pass
    # check size of the distribution 
    size = trace1_weights_matrix.shape[1]
    if (size <1000) or (size > 5000):
        print("Distribution size must be within 1000 to 5000")
        return None
    else:
        pass


    all_score = 0
    # ID_order = aux_test_data['planet_ID'].to_numpy()
    for i, val in tqdm(enumerate(id_order)):
        samples1 = trace1_matrix[i]
        samples1_weight = trace1_weights_matrix[i]
        samples2 = trace2[f'Planet_{val}']['tracedata'][:]
        samples2_weight = trace2[f'Planet_{val}']['weights'][:]
        one_score = calculate_w2(samples1, samples2,w1=samples1_weight,w2=samples2_weight,normalise=True)
        all_score += one_score
    overall_mean_score = all_score/len(trace1_matrix)
    print("score is:",overall_mean_score)
    return overall_mean_score


def batch_calculate_from_file(trace1_hdf5, trace2_hdf5):
    """calculate score from two .hdf5 files."""
    # read data from hdf5
    trace1 = h5py.File(trace1_hdf5,'r')
    trace2 = h5py.File(trace2_hdf5,'r')
    trace1_keys = [p for p in trace1.keys()]
    # assume trace1 is coming from participants
    check_distribution(trace1, trace1_keys)

    all_score = 0
    # ID_order = aux_test_data['planet_ID'].to_numpy()
    for val in tqdm(trace1_keys):
        samples1 = trace1[val]['tracedata'][:]
        samples1_weight = trace1[val]['weights'][:]
        samples2 = trace2[val]['tracedata'][:]
        samples2_weight = trace2[val]['weights'][:]
        one_score = calculate_w2(samples1, samples2,w1=samples1_weight,w2=samples2_weight,normalise=True)
        all_score += one_score
    overall_mean_score = all_score/len(trace1_keys)
    print("score is:",overall_mean_score)
    return overall_mean_score


def calculate_w2(trace1, trace2, w1=None, w2=None, normalise = True, bounds_matrix = None):
    """Calculate the Wessestein-2 distnace metric between two multivariate distributions

    Args:
        trace1 (array):  N x D Matrix, where N is the number of points and D is the dimensionality
        trace2 (array):  N x D Matrix, where N is the number of points and D is the dimensionality
        w1 (array, optional):  1D Array of N length. Defaults to None.
        w2 (array, optional):  1D Array of N length. Defaults to None.
        normalise (bool, optional): _description_. Defaults to True.

    Returns:
        scalar: W2 distance between two empirical probability distribution
    """ 
    
    if normalise:
        if bounds_matrix is None:
            bounds_matrix = default_prior_bounds()
        trace1 = normalise_arr(trace1,bounds_matrix )
        trace2 = normalise_arr(trace2,bounds_matrix )
    else:
        pass
    
    # calculate cost matrix
    M = ot.dist(trace1, trace2)

    # assume uniform weight if weights are not given
    if w1 is None:
        a = ot.unif(len(trace1))
    else:
        assert np.isclose(np.sum(w1),1)
        a = w1
    if w2 is None:
        b = ot.unif(len(trace2))
    else:
        assert np.isclose(np.sum(w2),1)
        b = w2
    M /= M.max()
    # numItermax controls the Max number of iteration before the solver "gives up" and return the result, 
    # recommended to use at least 100000 iterations for good results
    W2 = ot.emd2(a, b, M, numItermax=100000)
    # turn into an increasing score
    score = 1000* (1-W2)
    return score

"""General Helper Functions"""

def default_prior_bounds():
    T_range = [0,7000]
    gas1_range = [-12, -1]
    gas2_range = [-12, -1]
    gas3_range = [-12, -1]
    gas4_range = [-12, -1]
    gas5_range = [-12, -1]
    bounds_matrix = np.vstack([T_range,gas1_range,gas2_range,gas3_range,gas4_range,gas5_range])
    return bounds_matrix

def normalise_arr(arr, bounds_matrix):
    norm_arr = (arr - bounds_matrix[:,0])/(bounds_matrix[:,1]- bounds_matrix[:,0])
    norm_arr = restrict_to_prior(norm_arr)
    return norm_arr

def restrict_to_prior(arr,):
    # updates 06-oct, restrict values to default prior range 
    arr[arr<0] = 0
    arr[arr>1] = 1
    return arr

def check_distribution(File, UserPlanets):
    # updates from 06-Oct, perform various checks on the file, some of these changes are noted in submit_format.py from the baseline
    for planet in UserPlanets:
        traces = File[planet]['tracedata']
        weights = File[planet]['weights']
         ## sanity check - samples count should be the same for both
        assert len(traces) == len(weights)
        ## sanity check - weights must be able to sum to one.
        assert np.isclose(np.sum(weights),1)
        ## size of distribution must be between 1000 to 5000
        assert ((len(weights) > 1000) & (len(weights)<5000))
