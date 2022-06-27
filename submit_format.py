import pandas as pd
import h5py
from tqdm import tqdm
import numpy as np 

def to_light_track_format(q1_array, q2_array, q3_array, planet_ID,  columns = None):
    """Helper function to prepare submission file for the light track

    Args:
        q1_array: N x 6 array containing the estimates for 1st Quartiles
        q2_array:  N x 6 array containing the estimates for 2nd Quartiles
        q3_array:  N x 6 array containing the estimates for 3rd Quartiles
        columns: columns for the df. default to none

    Returns:
        Pandas DataFrame object , ID in ascending order
    """
    # create empty array
    LT_submission_df = pd.DataFrame(columns= columns)
    # length should be equal
    assert len(q1_array) == len(q2_array) == len(q3_array)
    targets_label = ['T', 'log_H2O', 'log_CO2','log_CH4','log_CO','log_NH3']
    # create daataframe
    default_quartiles = ['q1','q2','q3']
    default_columns = []
    for c in targets_label:
        for q in default_quartiles:
            default_columns.append(c+q)
    
    if columns is None:
        columns = default_columns
    for i in tqdm(planet_ID):
        quartiles_dict = {}
        quartiles_dict['planet_ID'] = i
        for t_idx, t in enumerate(targets_label):
            quartiles_dict[f'{t}_q1']= q1_array[i, t_idx]
            quartiles_dict[f'{t}_q2'] = q2_array[i, t_idx]
            quartiles_dict[f'{t}_q3']=q3_array[i, t_idx]
        LT_submission_df = pd.concat([LT_submission_df, pd.DataFrame.from_records([quartiles_dict])],axis=0,ignore_index = True)
    LT_submission_df.to_csv("LT_submission.csv",index= False)
    return LT_submission_df


def to_regular_track_format(tracedata_arr, weights_arr, planet_ID, name="RT_submission.hdf5"):
    """convert input into regular track format

    Args:
        tracedata_arr (array): Tracedata array, usually in the form of N x M x 6, where M is the number of tracedata, here we assume tracedata is of equal size. 
        It does not have to be but you will need to craete an alternative function if the size is different. 
        weights_arr (array): Weights array, usually in the form of N x M, here we assumed the number of weights is of equal size, it should have the same size as the tracedata
        planet_ID (array): ID for the example, this is used to correctly evaluate the prediction with the ground truth
        name (str, optional): Defaults to "RT_submission.hdf5".

    Returns:
        _type_: h5py object
    """
    submit_file = name
    RT_submission = h5py.File(submit_file,'w')
    for n in range(len(tracedata_arr)):
        ## sanity check - samples count should be the same for both
        assert len(tracedata_arr[n]) == len(weights_arr[n])
        grp = RT_submission.create_group(f"Planet_{planet_ID[n]}")
        pl_id = grp.attrs['ID'] = planet_ID[n] 
        tracedata = grp.create_dataset('tracedata',data=tracedata_arr[n])
        ## right now it assumes uniform weight
        ## sanity check - weights must be able to sum to one. 
        if np.sum(weights_arr[n])>1:
            weight_adjusted = weights_arr[n]/np.sum(weights_arr[n])
        else:
            weight_adjusted = weights_arr[n]
            
        weights = grp.create_dataset('weights',data=weight_adjusted)
    return RT_submission