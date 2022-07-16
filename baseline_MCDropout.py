
## import libraries
import numpy as np
import pandas as pd
from tensorflow import keras
import h5py
import os
from tqdm import tqdm
from helper import *
from preprocessing import *
from submit_format import *
from MCDropout import MC_Convtrainer
from argparse import ArgumentParser
from metric_light_track import *
from metric_regular_track import *


parser = ArgumentParser()
parser.add_argument( "--training",
                    help="training path")
parser.add_argument( "--test",
                    help="test path")

args = parser.parse_args()
## constants
RJUP = 69911000
MJUP = 1.898e27
RSOL = 696340000

## preprocessing settings
repeat = 10
N = 5000 # train on the first 5000 data instances

# hyperparameters
batch_size= 32
lr= 1e-3
epochs = 10
filters = [32,64,128]
dropout = 0.1
# number of examples to generate in evaluation time (5000 is max for this competition)
N_samples = 5000
threshold = 0.8

#fix seed
np.random.seed(42)
keras.utils.set_random_seed(42)

training_path = args.training
test_path = args.test
training_GT_path = os.path.join(training_path, 'Ground Truth Package')

# read training data
spectral_training_data = h5py.File(os.path.join(training_path,'SpectralData.hdf5'),"r")
aux_training_data = pd.read_csv(os.path.join(training_path,'AuxillaryTable.csv'))
soft_label_data = pd.read_csv(os.path.join(training_GT_path, 'FM_Parameter_Table.csv'))

spec_matrix = to_observed_matrix(spectral_training_data,aux_training_data)

## extract the noise
noise = spec_matrix[:N,:,2]
spectra = spec_matrix[:N,:,1]
wl_channels = len(spec_matrix[0,:,0])
global_mean = np.mean(spectra)
global_std = np.std(spectra)

# additional features
## add Rstar and Rplanet
radii = aux_training_data[['star_radius_m', 'planet_radius_m']]
## we would prefer to use Rsol and Rjup 
radii['star_radius'] = radii['star_radius_m']/RSOL
radii['planet_radius'] = radii['planet_radius_m']/RJUP
radii = radii.drop(['star_radius_m', 'planet_radius_m'],axis=1)
radii = radii.iloc[:N, :]
mean_radii = radii.mean()
stdev = radii.std()

target_labels = ['planet_temp','log_H2O','log_CO2','log_CH4','log_CO','log_NH3']
targets = soft_label_data.iloc[:N][target_labels]
num_targets = targets.shape[1]
targets_mean = targets.mean()
targets_std = targets.std()

# Train valid split
ind = np.random.rand(len(spectra)) < threshold
training_spectra, training_radii,training_targets, training_noise = spectra[ind],radii[ind],targets[ind], noise[ind]
valid_spectra, valid_radii, valid_targets = spectra[~ind],radii[~ind],targets[~ind]


## We will incorporate the noise profile into the observed spectrum by perturbing the spectra and augment to data with these noised spectra
aug_spectra = augment_data_with_noise(training_spectra, training_noise, repeat)
aug_radii = np.tile(training_radii.values,(repeat,1))
aug_targets = np.tile(training_targets.values,(repeat,1))

### standardise ###

# spectrum
std_aug_spectra = standardise(aug_spectra, global_mean, global_std)
std_aug_spectra = std_aug_spectra.reshape(-1, wl_channels)
std_valid_spectra = standardise(valid_spectra, global_mean, global_std)
std_valid_spectra = std_valid_spectra.reshape(-1, wl_channels)

## radius
std_aug_radii= standardise(aug_radii, mean_radii.values.reshape(1,-1), stdev.values.reshape(1,-1))
std_valid_radii= standardise(valid_radii, mean_radii, stdev)

# targets
std_aug_targets = standardise(aug_targets, targets_mean.values.reshape(1,-1), targets_std.values.reshape(1,-1))
std_valid_targets = standardise(valid_targets, targets_mean, targets_std)

## set up our Conv neural network
model = MC_Convtrainer(wl_channels,num_targets,dropout,filters)

## train now 
model.compile(
    optimizer=keras.optimizers.Adam(lr),
    loss='mse',)
model.fit([std_aug_spectra,std_aug_radii], 
          std_aug_targets, 
          validation_data=([std_valid_spectra, std_valid_radii],std_valid_targets),
          batch_size=batch_size, 
          epochs=epochs, 
          shuffle=False,)

# probabilistic inference on the valid set
instances = N_samples
y_pred_valid = np.zeros((instances, len(std_valid_spectra), num_targets ))
for i in tqdm(range(instances)):
    y_pred = model.predict([std_valid_spectra,std_valid_radii])
    y_pred_valid[i] += y_pred


y_pred_valid = y_pred_valid.reshape(-1,num_targets)
y_pred_valid_org = transform_back(y_pred_valid,targets_mean[None, ...], targets_std[None, ...])
y_pred_valid_org = y_pred_valid_org.reshape(instances, len(std_valid_spectra), num_targets)
y_pred_valid_org = np.swapaxes(y_pred_valid_org, 1,0)
q1_pred_valid, q2_pred_valid, q3_pred_valid = np.quantile(y_pred_valid_org, [0.16,0.5,0.84],axis=1)

# evaluate performance on validation data.
## read trace and quartiles table 
GT_trace_path = os.path.join(training_GT_path, 'Tracedata.hdf5')
GT_Quartiles_path = os.path.join(training_GT_path, 'QuartilesTable.csv')
all_qs = load_Quartile_Table(GT_Quartiles_path)

index= np.arange(len(ind))
valid_index = index[~ind]
valid_GT_Quartiles = all_qs[valid_index]
valid_GT_Quartiles = np.swapaxes(valid_GT_Quartiles, 1,0)

### Evaluate ###
valid_q_pred = np.concatenate([q1_pred_valid[None,...], q2_pred_valid[None,...], q3_pred_valid[None,...]],axis=0)

# calculate!
light_track_metric(valid_GT_Quartiles, valid_q_pred, k =1000)

# assuming each distribution produce the same number of trace (N_samples)
trace1_matrix = y_pred_valid_org
# assuming uniform weight, and the weights must sum to 1
trace1_weights_matrix = np.ones((trace1_matrix.shape[0], trace1_matrix.shape[1]))/trace1_matrix.shape[1] 

batch_calculate(trace1_matrix, trace1_weights_matrix, GT_trace_path, id_order = valid_index)

## now we package the prediction for submission to the website.

spec_test_data = h5py.File(os.path.join(test_path,'SpectralData.hdf5'),"r")
aux_test_data = pd.read_csv(os.path.join(test_path,'AuxillaryTable.csv'))

## same preprocessing as before
test_spec_matrix = to_observed_matrix(spec_test_data,aux_test_data )
std_test_spectra = standardise(test_spec_matrix[:,:,1], global_mean, global_std)

test_radii = aux_test_data[['star_radius_m', 'planet_radius_m']]
## we would prefer to use Rsol and Rjup 
test_radii['star_radius'] = test_radii['star_radius_m']/RSOL
test_radii['planet_radius'] = test_radii['planet_radius_m']/RJUP
test_radii = test_radii.drop(['star_radius_m', 'planet_radius_m'],axis=1)

std_test_radii= standardise(test_radii, mean_radii, stdev)


## Inference Time! ##
# instances = N_samples
y_pred_distribution = np.zeros((instances, len(std_test_spectra), num_targets ))
for i in tqdm(range(instances)):
    y_pred = model.predict([std_test_spectra,test_radii])
    y_pred_distribution[i] += y_pred
y_pred_distribution = y_pred_distribution.reshape(-1,num_targets)
y_pred_org = transform_back(y_pred_distribution,targets_mean[None, ...], targets_std[None, ...])
y_pred_org = y_pred_org.reshape(instances, len(std_test_spectra), num_targets)
y_pred_org = np.swapaxes(y_pred_org, 1,0)


## Package!
# extract quartiles estimate for 16th, 50th and 84th percentile.
all_q1_pred, all_q2_pred, all_q3_pred = np.quantile(y_pred_org, [0.16,0.5,0.84],axis=1)
LT_submission = to_light_track_format(all_q1_pred, all_q2_pred, all_q3_pred,)
tracedata = y_pred_org
# weight takes into account the importance of each point in the tracedata. 
# Currently they are all equally weighted and thus I have created a dummy array that sums the contribution to 1
weight = np.ones((y_pred_org.shape[0],y_pred_org.shape[1]))/np.sum(np.ones(y_pred_org.shape[1]) )

RT_submission = to_regular_track_format(y_pred_org, 
                                        weight, 
                                        name="RT_submission.hdf5")

