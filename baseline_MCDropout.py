
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

#hyperparameter settings
batch_size= 32
lr= 1e-3
epochs = 50
filters = [32,64,128]
dropout = 0.1
# number of examples to generate in test time
N_samples = 1000

#fix seed
np.random.seed(42)
keras.utils.set_random_seed(42)

training_path = '../NeurIPS_Competition/competition_script/data_generation/TrainingData2'
training_path = args.training
test_path = '../NeurIPS_Competition/competition_script/data_generation/TestData2'
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

## We will incorporate the noise profile into the observed spectrum by treating the noise as Gaussian noise.
aug_spectra = augment_data_with_noise(spectra, noise, repeat )

## standardise the input using global mean and stdev
std_aug_spectra, global_mean, global_std = transform_data(spectra, aug_arr=aug_spectra)

# additional features
## add Rstar and Rplanet
radii = aux_training_data[['star_radius_m', 'planet_radius_m']]
## we would prefer to use Rsol and Rjup 
radii['star_radius'] = radii['star_radius_m']/RSOL
radii['planet_radius'] = radii['planet_radius_m']/RJUP
radii = radii.drop(['star_radius_m', 'planet_radius_m'],axis=1)
radii = radii.iloc[:N, :]
# standardise
mean_radii = radii.mean()
stdev_radii = radii.std()
std_radii= standardise(radii, mean_radii, stdev_radii)
std_aug_radii = np.tile(std_radii.values,(repeat,1))

## transform some targets into log-scale
target_labels = ['planet_temp','log_H2O','log_CO2','log_CH4','log_CO','log_NH3']
targets = soft_label_data.iloc[:N][target_labels]
num_targets = targets.shape[1]
targets_mean = targets.mean()
targets_std = targets.std()
std_targets = standardise(targets, targets_mean, targets_std)

std_aug_targets = np.tile(std_targets.values,(repeat,1))


## set up our MLP neural network
model = MC_Convtrainer(wl_channels,num_targets,dropout,filters)
ind = np.random.rand(len(std_aug_spectra)) <0.8
x_train_spectra, x_train_radii,y_train = std_aug_spectra[ind],std_aug_radii[ind],std_aug_targets[ind]
x_valid_spectra, x_valid_radii,y_valid = std_aug_spectra[~ind],std_aug_radii[~ind],std_aug_targets[~ind]

## train now 
model.compile(
    optimizer=keras.optimizers.Adam(lr),
    loss='mse',)
model.fit([x_train_spectra,x_train_radii], 
          y_train, 
          validation_data=([x_valid_spectra, x_valid_radii],y_valid),
          batch_size=batch_size, 
          epochs=epochs, 
          shuffle=False,)

## now we test it on the test set. 

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

std_test_radii= standardise(test_radii, mean_radii, stdev_radii)


## Inference Time! ##

instances = N_samples
y_pred_distribution = np.zeros((instances, len(std_test_spectra), num_targets ))
for i in tqdm(range(instances)):
    y_pred = model.predict([std_test_spectra,test_radii])
    y_pred_distribution[i] += y_pred
y_pred_distribution = y_pred_distribution.reshape(-1,num_targets)
## project back to original space
y_pred_org = transform_back(y_pred_distribution,targets_mean[None, ...], targets_std[None, ...])
y_pred_org = y_pred_org.reshape(instances, len(std_test_spectra), num_targets)
y_pred_org = np.swapaxes(y_pred_org, 1,0)


## Package!
# extract quartiles estimate for 25th, 50th and 75th percentile.
all_q1_pred, all_q2_pred, all_q3_pred = np.quantile(y_pred_org, [0.25,0.5,0.75],axis=1)
LT_submission = to_light_track_format(all_q1_pred, all_q2_pred, all_q3_pred,planet_ID = aux_test_data.planet_ID.to_numpy(),)
RT_submission = to_regular_track_format(y_pred_org, 
                                        np.ones((y_pred_org.shape[0],y_pred_org.shape[1])),
                                        planet_ID = aux_test_data.planet_ID.to_numpy(),
                                     name="RT_submission.hdf5")

