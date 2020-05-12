#!/bin/python

'''
This script runs a trained model on a specified dataset, creates a text file
containing the chl-CP predictions against the CP values provided. There are two
types of dataset accepted by the script. These are described in detail in the
README. If a dataset containing chl-ACS or chl-HPLC values is passed in then
plots containing chl-CP against the true values and the relative residuals
are also produced.

Required information:
- The directory containing the model and stats.
- The dataset

Author: Sebastian Graban (seg)
Date: 19/03/2020
'''

from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import SGD
import matplotlib.colors as colors

import argparse
import sys
import glob
import os
import math

import pandas as pd
import numpy
import matplotlib
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.stats import gaussian_kde
from scipy.interpolate import interpn
from mpl_toolkits.axes_grid1 import make_axes_locatable

from pdf_template import PDF

font = {'family' : 'sans-serif',
    'size'   : 16}

# Sets up Latex formatting style for plots.
matplotlib.rc('font', **font)
matplotlib.rcParams['text.latex.preamble'] = [
       r'\usepackage{siunitx}',
       r'\sisetup{detect-all}',
       r'\usepackage{helvet}',
       r'\usepackage{sansmath}',
       r'\sansmath'
]

def prepare_dataset(data,lambda1,lambda2,lambda3,true_chl):
    '''
    Prepare the dataset to set them up to match the datasets used in the training
    of the model.

    :param data: The dataset to run the model on
    :type data: String
    :param lambda1: The wavelength selected for lambda1 nm
    :type lambda1: integer
    :param lambda2: The wavelength selected for lambda2 nm
    :type lambda3: integer
    :param lambda3: The wavelength selected for lambda3 nm
    :type lambda3: integer
    :param true_chl: If the dataset contains true chlorophyll-a values
    :type true_chl: boolean

    :return: The prepared dataset and the test labels associatied with the dataset
    :rtype: Pandas dataframe
    '''

    # Check that the input dataset is in the correct format as specified in the
    # README.
    raw_dataset = pd.read_csv(data, sep=",",  skipinitialspace=True)
    if (raw_dataset.shape[1] == 47 or raw_dataset.shape[1] == 4) and not true_chl:
        raise IOError("Your dataset seems to contain a column containing true chlorophyll-a values, please check the dataset and if this is true run with the true-chl argument.")
    elif raw_dataset.shape[1] != 47 and raw_dataset.shape[1] != 4 and true_chl:
        raise IOError("Your dataset does not have the correct number of columns, did you really want to pass true-chl as an argument?")
    elif (raw_dataset.shape[1] != 46 and raw_dataset.shape[1] != 3) and (true_chl and raw_dataset.shape[1] != 47 and raw_dataset.shape[1] != 4):
        raise IOError("Your dataset does not have the correct number of columns, please check the README for how the input dataset can be formatted")

    # Once the dataset has passed inspection load it in with the correct column
    # names
    column_names = list(range(1,raw_dataset.shape[1]+1))
    raw_dataset = pd.read_csv(data, sep=",",  skipinitialspace=True,names=column_names)
    dataset = raw_dataset.copy()

    if true_chl:
        # Get the labels for the dataset.
        test_labels = dataset.pop(raw_dataset.shape[1])
    else:
        test_labels = None

    dataset.tail()

    #clean it up
    dataset = dataset.dropna()
    #reset index
    dataset = dataset.reset_index()
    dataset.pop('index')

    # Get the location of each wavelength in the dataset
    if dataset.shape[1] == 3:
        lambda1_loc = 1
        lambda2_loc = 2
        lambda3_loc = 3
    elif dataset.shape[1] == 46:
        lambda1_loc = (lambda1 - 618)/2
        lambda2_loc = (lambda2 - 618)/2
        lambda3_loc = (lambda3 - 618)/2

    # Calculate the difference between wavelengths.
    difference_1 = lambda2 - lambda1
    difference_2 = lambda3 - lambda2
    difference_3 = lambda3 - lambda1

    # Get the gradients between the points on the spectral CP graph.
    test_grads = [
    (dataset[lambda2_loc] - dataset[lambda1_loc]) / difference_1,
    (dataset[lambda3_loc] - dataset[lambda2_loc]) / difference_2,
    (dataset[lambda3_loc] - dataset[lambda1_loc]) / difference_3
    ]

    # Add the columns for the gradients between points.
    columns = ["grad1","grad2","grad3"]

    # Create new dataset.
    new_test_data = pd.DataFrame()

    # Go through the dataset adding the gradients.
    for x in range(0,3):
        new_test_data = pd.concat([new_test_data, test_grads[x].rename(columns[x]).reset_index()], axis=1)

    # Rations between point, adding aditional information about the data seems
    # to improve the spread of the results.
    test_ratios_1 = dataset[lambda1_loc]/dataset[lambda2_loc]
    test_ratios_2 = dataset[lambda3_loc]/dataset[lambda2_loc]
    test_ratios_3 = dataset[lambda3_loc]/dataset[lambda1_loc]

    # Add the ratios to the dataset.
    new_test_data = pd.concat([new_test_data, test_ratios_1.rename("ratio_1").reset_index(), test_ratios_2.rename("ratio_2").reset_index(),test_ratios_2.rename("ratio_3").reset_index()], axis=1)

    # Get the abolute particulate beam attenuation values at the specified wavelengths
    beam_attenuation_lambda1 = dataset[lambda1_loc].rename("beam_1").reset_index()
    beam_attenuation_lambda2 = dataset[lambda2_loc].rename("beam_2").reset_index()
    beam_attenuation_lambda3 = dataset[lambda3_loc].rename("beam_3").reset_index()

    # Add the absolute particulate beam attenuation values.
    new_test_data = pd.concat([new_test_data, beam_attenuation_lambda1, beam_attenuation_lambda2, beam_attenuation_lambda3], axis=1)

    # Make sure to remove the index.
    new_test_data.pop('index')

    return new_test_data, test_labels

def norm(x, train_stats):
    '''
    Features are normalised using Z-score normalisation to minimise chance of model
    converging badly.

    :param x: The dataset to normalise.
    :type x: Pandas dataframe
    :param train_stats: The stats on the dataset.
    :type train_stats: Pandas dataframe

    :return: The normalised dataset.
    :rtype: Pandas dataframe
    '''
    return (x - train_stats['mean']) / train_stats['std']

def make_graphs(test_labels,test_predictions,relative_residuals, directory):
    '''
    If true chlorophyll-a values were provided then produce relative_residuals and
    chl-CP against true values plots.

    :param test_labels: The true chlorophyll-a values
    :type test_labels: Pandas Series
    :param test_predictions: The predictions made by the model on the dataset.
    :type test_predictions: Pandas Series
    :param relative_residuals: The relative_residuals of the results of the model.
    :type relative_residuals: Pandas Series
    :param directory: The directory in which to store the results.
    :type directory: String
    '''

    # Get the bias.
    bias = relative_residuals.median()

    # Get the spread using relative standard deviation.
    spread = (relative_residuals.quantile(.84,"higher") - relative_residuals.quantile(.16,"lower"))/2

    # Ensure predictions are in the same format as relative resiudals and labels.
    test_predictions = pd.Series(test_predictions)

    # Create a subplot to show the relative_residuals of the results from the model.
    fig = plt.figure(figsize=(10,30))
    gs = fig.add_gridspec(2, 2,width_ratios=[35,1],wspace=0.05)
    ax1 = fig.add_subplot(gs[0,0])
    ax3 = fig.add_subplot(gs[1,0])
    fig.set_size_inches(15,15)

    # Reset the indexes of the series to ensure they have the same index.
    test_labels.reset_index(inplace=True,drop=True)
    test_labels = test_labels.values
    relative_residuals = relative_residuals.values
    # Create bins to be used in the 2D histogram.
    x_bins = numpy.logspace(numpy.log10(test_labels.min()),numpy.log10(test_labels.max()),90)
    y_bins = numpy.linspace(relative_residuals.min(),relative_residuals.max(),90)

    #Create the scatter graph for relative resiudals.
    ax1.set_xscale('log')
    ax1.set_yscale('linear')
    h2 = ax1.hist2d(test_labels,relative_residuals,bins=[x_bins,y_bins],norm=colors.LogNorm(),cmap="hot_r")
    ax1.set_xlim([0.02 - 0.02/4,9])
    ax1.set_ylim(-3.0,3.0)
    ax1.plot([0, 10], [0,0])
    ax1.set_ylabel(r"$\text{\textbf{Relative Residuals}}$",labelpad=10, usetex=True, weight="bold",size=20)
    ax1.text(0.9, 0.1, "A", transform=ax1.transAxes,
            size=35, weight='bold')
    ax1.set_xticks([0.03,0.1,0.3,1,3])
    ax1.set_xticklabels([0.03,0.1,0.3,1,3])
    ax1.text(0.85, 0.9, r'$\delta = {0:.2f}$'.format(bias), transform=ax1.transAxes,
        size=20, weight='bold', usetex=True)
    ax1.text(0.85, 0.8, r'$\sigma = {0:.2f}$'.format(spread), transform=ax1.transAxes,
        size=20, weight='bold', usetex=True)

    # Ensure only positive predictions are kept, this is needed for the log scales.
    test_predictions = test_predictions.values
    index = numpy.argwhere(test_predictions <= 0)
    test_labels = numpy.delete(test_labels,index)
    test_predictions = numpy.delete(test_predictions,index)

    # Create bins for 2D histogram.
    x_bins = numpy.logspace(numpy.log10(test_labels.min()),numpy.log10(test_labels.max()),90)
    y_bins = numpy.logspace(numpy.log10(test_predictions.min()),numpy.log10(test_predictions.max()),90)

    # Create chl-CP against true chlorophyll-a plot.
    ax3.set_yscale('log')
    ax3.set_xscale('log')
    h = ax3.hist2d(test_labels,test_predictions,bins=[x_bins,y_bins],norm=colors.LogNorm(),cmap="hot_r")
    ax3.set_xlabel(r'$\text{\textbf{chl-ACS}} \text{ \textbf{(mg/m}}^3\text{\textbf{)}}$',labelpad=10, usetex=True, weight="bold", size=20)
    ax3.set_ylabel(r'$\text{\textbf{chl-CP}} \text{ \textbf{(mg/m}}^3\text{\textbf{)}}$',labelpad=10, usetex=True, weight="bold", size=20)
    ax3.set_xlim([0.02 - 0.02/4,9])
    ax3.set_ylim([0.02 - 0.02/4,9])
    ax3.plot([0, 10], [0,10])
    ax3.set_xticks([0.03,0.1,0.3,1,3])
    ax3.set_xticklabels([0.03,0.1,0.3,1,3])
    ax3.set_yticks([0.03,0.1,0.3,1,3])
    ax3.set_yticklabels([0.03,0.1,0.3,1,3])

    ax3.text(0.9, 0.1, "B", transform=ax3.transAxes,
        size=35, weight='bold')

    # Add colorbar
    cax = fig.add_subplot(gs[:,1])

    cb = fig.colorbar(h[3],cax=cax,orientation='vertical')
    cb.set_label(label=r"$\text{\textbf{Density}}$", weight="bold", size=20,usetex=True)

    plt.savefig(os.path.join(directory,"model_results.png"))

def model_run(model_dir,dataset,combined_df,index,lambda1,lambda2,lambda3):
    '''
    Sets up the model and then uses it to predict chl-CP from the provided dataset.

    :param models_dir: The directory containing the model and model stats.
    :type model_dir: String
    :param dataset: The location of the dataset that the model is to be applied to
    :type dataset: String
    :combined_df: The dataframe containing all the predictions made by the models
                  in the ensemble.
    :type combined_df: pandas dataframe
    :param index: The index of the model in the ensemble
    :type index: integer
    :param lambda1: The wavelength selected for lambda1 nm
    :type lambda1: integer
    :param lambda2: The wavelength selected for lambda2 nm
    :type lambda3: integer
    :param lambda3: The wavelength selected for lambda3 nm
    :type lambda3: integer

    :return: The combined dataframe of predictions from the ensemble and the test
             predictions made by the model.
    '''

    # Get the model file.
    model = glob.glob(os.path.join(model_dir,"*.h5"))

    if len(model) != 1:
        raise IOError("Model directory does not contain the correct number of model h5 files")
    else:
        model = model[0]

    # Get the model stats used for normalisation.
    model_stats = os.path.join(model_dir,"train_stats.dat")

    if not os.path.isfile(model_stats):
        raise IOError("Model directory does not contain a stats file")

    # Load the model.
    model = load_model(model)

    # Get the test stats of the training data and format it.
    test_stats = pd.read_csv(model_stats)
    test_stats.set_index('Unnamed: 0', inplace=True)

    # Normalise the validation data.
    normed_test_data = norm(dataset,test_stats)

    # Set up the wavelength labels.
    Wavelengths_train = pd.DataFrame({"minima" : numpy.full((dataset.shape[0]), lambda1), "maxima" : numpy.full((dataset.shape[0]), lambda2),"end_point" : numpy.full((dataset.shape[0]), lambda3)})
    # Add the wavelength labels to the dataset.
    normed_test_data = pd.concat([normed_test_data,Wavelengths_train],axis=1)

    # Predict chlorophyll a for the dataset using the model.
    test_predictions = model.predict(normed_test_data).flatten()

    # Add the test prediction to the combined_df so that we can median the results
    # for an ensemble.
    combined_df["pred_{}".format(index)] = test_predictions

    return combined_df, test_predictions

def run(model_dir,dataset,directory,lambda1,lambda2,lambda3,true_chl,ensemble):
    '''
    Runs the main code.

    :param models_dir: The directory containing the model and model stats.
    :type model_dir: String
    :param dataset: The location of the dataset that the model is to be applied to
    :type dataset: String
    :param directory: The directory in which to store the results
    :type directory: String
    :param lambda1: The wavelength selected for lambda1 nm
    :type lambda1: integer
    :param lambda2: The wavelength selected for lambda2 nm
    :type lambda3: integer
    :param lambda3: The wavelength selected for lambda3 nm
    :type lambda3: integer
    :param true_chl: If the dataset contains true chlorophyll-a values
    :type true_chl: boolean
    :param ensemble: If the model contained is an ensemble or not.
    :type ensemble: boolean
    '''

    # Prepare the dataset to be tested on the model.
    dataset, test_labels = prepare_dataset(dataset,lambda1,lambda2,lambda3,true_chl)

    # Create a combined data frame for storing the results of multiple models
    # from an ensemble.
    combined_df = pd.DataFrame()

    # Index for naming columns in combined dataframe for ensemble.
    index = 0
    # set up standard deviation if an ensemble is used.
    model_sd = None

    # If the model is not an ensemble then simply get the results of the models
    # predictions
    if not ensemble:
        test_predictions = model_run(model_dir,dataset,combined_df,0,lambda1,lambda2,lambda3)[1]
    # If the model is an ensemble then predict chl-CP using each individual model
    # and store the results in the combined dataframe. To achieve a prediction
    # for the overall ensemble take the median of the predictions from all the
    # models.
    else:
        for model in os.listdir(model_dir):
            combined_df = model_run(os.path.join(model_dir,model),dataset,combined_df,index,lambda1,lambda2,lambda3)[0]
            index+=1
        # Calculate the robust standard deviation between the models.
        model_sd = ((combined_df.quantile(.84,axis=1) - combined_df.quantile(.16,axis=1))/2)/math.sqrt(len(os.listdir(model_dir)))
        combined_df["average"] = combined_df.median(axis=1)
        test_predictions = combined_df["average"].tolist()

    output_dataset = pd.DataFrame()
    output_dataset["{} nm".format(lambda1)] = dataset["beam_1"]
    output_dataset["{} nm".format(lambda2)] = dataset["beam_2"]
    output_dataset["{} nm".format(lambda3)] = dataset["beam_3"]
    output_dataset["chl-CP"] = pd.Series(test_predictions)
    if model_sd is not None:
        output_dataset["Uncertainty in predicted chl"] = model_sd

    output_dataset.to_csv(os.path.join(directory,"chl-CP.csv"),index=False)

    # If real chlorophyll-a values were provided in the dataset produce figures
    if test_labels is not None:

        # Calculate the relative residuals of the prediction.
        relative_residuals = (test_predictions - test_labels) / test_labels

        # Get the bias.
        bias = relative_residuals.median()

        # Get the spread using relative standard deviation.
        spread = (relative_residuals.quantile(.84,"higher") - relative_residuals.quantile(.16,"lower"))/2

        print("The relative bias of the model is {}\n The relative robust standard deviation is {}".format(bias,spread))

        # Create the graphs for the predictions and relative_residuals.
        make_graphs(test_labels,test_predictions,relative_residuals, directory)


if __name__ == "__main__":

    # Set up the arguments.
    parser = argparse.ArgumentParser(description="Run a new dataset on a previously trained model")
    parser.add_argument("--model", help="The directory containing all of the information about the model", dest="model_dir", required=True)
    parser.add_argument("--dataset", help="The datasets to be applied to the model", dest="data", required=True)
    parser.add_argument("--out_dir", help="The directory in which to store the results", required=True, dest="directory")
    parser.add_argument("--lambda1", dest="lambda1", action="store", help="Value in nm of lambda1 (for details refer to paper)", type=int, required=True)
    parser.add_argument("--lambda2", dest="lambda2", action="store", help="Value in nm of lambda2 (for details refer to paper)", type=int, required=True)
    parser.add_argument("--lambda3", dest="lambda3", action="store", help="Value in nm of lambda3 (for details refer to paper)", type=int, required=True)
    parser.add_argument("--true-chl", dest="true_chl", action="store_true", help="Does the dataset have a true chlorophyll-a values such as chl-ACS or chl-HPLC as a collumn", default=False)
    parser.add_argument("--ensemble", dest="ensemble", action="store_true", default=False, help="If you intend on running an ensemble of models, please use this argument and ensure that you have a directory with the models inside as described on the github.")

    args = parser.parse_args()

    run(args.model_dir,args.data,args.directory,args.lambda1,args.lambda2,args.lambda3,args.true_chl,args.ensemble)
