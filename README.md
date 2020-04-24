# CMAP

These are the scripts and files required to estimate chlorophyll-a concentrations using particulate beam-attenuation coefficients as described in the paper Graban et al., 2020 (submitted to Optics Express). 

At the moment this repository only contains a script which enables the user to run a
model on a specified dataset. The repository will later be expanded to include
scripts to train new networks as well as re-training existing networks.

## Prerequisites ##

Included within this repository is a requirements.txt files which contains all of
the necessary packages required to run any of the scripts.

To run the model you need to have the latest version of Python installed, links
can be found here: `https://www.python.org/downloads/`

Once you have Python installed you need to ensure all the packages needed by the model are installed.
The easiest way to do this is to manage the packages through conda. Please install "conda" by
following the instructions found here `https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html`

Once you have conda installed, follow these steps to install the packages:

```
conda create -n cmap
conda install -n cmap (all of the packages found in requirements.txt)
conda activate cmap  
```

This should now allow you to run the model scripts.

## Run Model ##

To run the model you need two components: the model and the dataset on
which to run the model. Each of these will be described in detail. It is crucial
that the model and dataset follow the structure described below.

### Model ###

The model directory should contain the model. This should is in the form
of a .h5 file that contains the tuned neural network. The directory should also contain a train_stats.dat file which
contains the stats ("the stats" is pretty vague: can you be more explicit? You can quote equations in the paper) required to normalise the dataset. An example of this can
be found in this repository, containing the model discussed in the paper.

### Dataset ###

There is some flexibility available with the formatting of the dataset to be
provided to the run model script.

The dataset should be in a csv format and contain particulate beam attenuation
coefficients (c_p). The c_p coefficients can either be sampled at every
second wavelength between 620 - 710 nm (resulting in 46 columns in the csv file) or at three
wavelengths, which correspond to the \lambda_1, \lambda_2 and \lambda_3 values (ANY LAMBDA 1,2,3 ??? if so, we need to explain that there is one recommended set of wavelengths and what this is) that you
want the model to run at.

Independent of the format the dataset is provided to the script, you must select a \lambda_1,
\lambda_2 and \lambda_3 which are the three wavelengths from which you want the model
to predict the chl-CP. NOT CLEAR HOW EXACTLY TO SELECT THESE 

The dataset can also have an additional column after the beam attenuation coefficients
containing a chlorophyll-a measurement taken at the same time as the beam attenuation.
If this column is included then the `true-chl` argument should be added when running
the script. This will allow the script to produce figures showing the results of the
model against the chlorophyll-a values provided.

## Script arguments ##

The two main components are the model and the dataset. The
possible arguments for the script are as follows:

 * --dataset The location of the dataset with formatted as described above.
 * --model The directory containing the model and the training stats
 * --directory The directory in which you want to store the results
 * --lambda1 The wavelength at lambda1
 * --lambda2 The wavelength at lambda2
 * --lambda3 The wavelength at lambda3
 * --true_chl If a column exists in the dataset with true chlorophyll-a values
 
 
 ## Example 1: Estimating chl using three c_p wavelengths ##
 1) Enter the cmap conda environment:
 conda activate cmap
 
 2) prepare a csv file as follows:
 0.2, 0.3, 0.2 LET'S USE SOMETHING REALISTIC
 where the three numbers are the values of c_p at \lambda_1=XXX nm, \lambda_2XXX nm and \lambda_3XXX nm
 
 3) Issue the following command:...
 
 
 
