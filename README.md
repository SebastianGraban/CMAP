# CMAP

Chlorophyll-a retrieval using particulate beam attenuation coefficients. At the
moment this repository only contains a script which enables the user to run a
model on a specified dataset. The repository will later be expanded to include
scripts to train new networks as well as re-training existing networks.

## Prerequisites ##

Included within this repository is a requirements.txt files which contains all of
the necessary packages required to run any of the scripts.

To run the scripts ensure you have the latest version of Python installed, links
can be found here: `https://www.python.org/downloads/`

Once you have Python installed you need to ensure all the packages are installed.
The easiest way to do this is to manage the packages through conda. Please
follow the instructions found here `https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html`

Once you have conda installed follow these steps to install the packages:

```
conda create -n cmap
conda install -n cmap (all of the packages found in requirements.txt)
conda activate cmap  
```

This should now allow you to run the scripts.

## Run Model ##

The running of the model requires two components the model and the dataset on
which to run the model. Each of these will be described in detail. It is crucial
that the model and dataset follow the structure described below.

### Model ###

The model directory should contain the model itself. This should be in the form
of a .h5 file. The directory should also contain a train_stats.dat file which
contains the stats required to normalise the dataset. An example of this can
be found in this repository, containing the model discussed in the paper.

### Dataset ###

There is some flexibility available with the formatting of the dataset to be
provided to the run model script.

The dataset should be in a csv format and contain particulate beam attenuation
coefficients. The beam attenuation coefficients can either be sampled at every
second wavelength between 620 - 710 nm (giving 46 columns in the CSV) or at 3
wavelengths, which correspond to the lambda1, lambda2 and lambda3 values that you
want the model to run at.

Whichever dataset format is provided to the script you must select a lamda1,
lambda2 and lambda3 which are the three wavelengths from which you want the model
to predict the chl-CP. For more context about what the reason for this is see the
paper.

The dataset can also have an additional column after the beam attenuation coefficients
containing a chlorophyll-a measurement taken at the same time as the beam attenuation.
If this column is included then the `true-chl` argument should be added when running
the script. This will allow the script to produce figures showing the results of the
model against the chlorophyll-a values provided.

## Example ##

As discussed above the two main components are the model and the dataset. The
possible arguments for the script are as follows:

 * --dataset The location of the dataset with formatted as described above.
 * --model The directory containing the model and the training stats
 * --directory The directory in which you want to store the results
 * --lambda1 The wavelength at lambda1
 * --lambda2 The wavelength at lambda2
 * --lambda3 The wavelength at lambda3
 * --true_chl If a column exists in the dataset with true chlorophyll-a values
