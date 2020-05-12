# CMAP #

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
conda create -n cmap python=3.6
conda install -n cmap (all of the packages found in requirements.txt)
conda install -n cmap -c anaconda tensorflow-gpu
conda install -n cmap -c viascience fpdf
conda activate cmap  
```

This should now allow you to run the model scripts.

Once you have created the environment you need to clone the github into a local
repository to be able to run it.

## Run Model ##

To run the model you need two components: the model and the dataset on
which to run the model. Each of these will be described in detail. It is crucial
that the model and dataset follow the structure described below.

### Model ###

The model directory can come in two forms.

Either the model is a single model in which case it should be made up of a single
.h5 file that contains the tuned neural network. The directory should also contain a train_stats.dat file which
contains the stats required to normalise the dataset. These stats include the mean
and standard deviation for every column used in the training of the model. These
are the minimum required files in a model directory.

- Model_directory
    - model.h5
    - train_stats.dat

The second option is for the model directory to continue multiple models formatted
as explained above. This is the case where the model is an ensemble. The model
directory should then only contain sub directories which are formatted as described
above. If you are using an ensemble then ensure to pass the `--ensemble` argument
when running the script.

- ensemble-directory
    - model1_directory
        - model1.h5
        - train_stats.dat
    - model2_directory
        - model2.h5
        - train_stats.dat

### Dataset ###

There is some flexibility available with the formatting of the dataset to be
provided to the run model script.

The dataset should be in a csv format and contain particulate beam-attenuation
coefficients (c_p). The c_p coefficients can either be sampled at every
second wavelength between 620 - 710 nm (resulting in 46 columns in the csv file) or at three
wavelengths, which correspond to the lambda_1, lambda_2 and lambda_3 values that you
want the model to run at.

Independent of the format the dataset is provided to the script, you must select a lambda_1,
lambda_2 and lambda_3 which are the three wavelengths from which you want the model
to predict the chl-CP.

The ranges for possible lambda values are as follows:

 * 664 <= lambda_1 <= 670
 * 682 <= lambda_2 <= 688
 * 704 <= lambda_3 <= 710

The dataset can also have an additional column after the beam attenuation coefficients
containing a chlorophyll-a measurement taken at the same time as the beam attenuation.
If this column is included then the `--true-chl` argument should be added when running
the script. This will allow the script to produce figures showing the results of the
model against the chlorophyll-a values provided.

## Script arguments ##

The two main components are the model and the dataset. The
possible arguments for the script are as follows:

 * --dataset The location of the dataset with formatted as described above.
 * --model The directory containing the model and the training stats
 * --out_dir The directory in which you want to store the results
 * --lambda1 The wavelength at lambda1
 * --lambda2 The wavelength at lambda2
 * --lambda3 The wavelength at lambda3
 * --true_chl If a column exists in the dataset with true chlorophyll-a values
 * --ensemble If the model is an ensemble add this argument

## Examples ##

### Example 1: Estimating chl using three cp wavelengths ###

 1) Enter the cmap conda environment:
 conda activate cmap

 2) prepare a csv file as follows:
  0.025721, 0.025894, 0.024865 where the three numbers are the values of cp at lambda_1=668 nm, lambda_2=688 nm and lambda_3=710 nm. Or use the example input dataset provided in
`example_input_output/example1/example_1_input.csv`

 3) Issue the following command
  `python run_model_on.py --model ensemble/ --ensemble --dataset example_input_output/example1/example_1_input.csv --out_dir ./ --lambda1 668 --lambda2 688 --lambda3 710`

 4) The output from this should produce the file `chl-CP.csv` which should be the same as the file `example_input_output/example1/example_1_output.csv`

### Example 2: Estimating chl using dataset with cp wavelengths taken at every 2 nm from 620 nm to 710 nm ###

1) Enter the cmap conda environment:
conda activate cmap

2) prepare a csv file as follows:
0.027307, 0.027255, ... , 0.025031, 0.024865
Where there are 46 columns with cp taken at interval of 2 nm within range of 620 nm - 710 nm. Or use the example input dataset provided in
`example_input_output/example2/example_2_input.csv`

3) Issue the following command
`python run_model_on.py --model ensemble/ --ensemble --dataset example_input_output/example2/example_2_input.csv --out_dir ./ --lambda1 668 --lambda2 688 --lambda3 710`

 4) The output from this should produce the file `chl-CP.csv` which should be the same as the file `example_input_output/example2/example_2_output.csv`

### Example 3: Estimating chl using three cp wavelengths with true chl value ###

1) Enter the cmap conda environment:
conda activate cmap

2) prepare a csv file as follows:
0.025721, 0.025894, 0.024865, 0.141172
where the three numbers are the values of cp at lambda_1=668 nm, lambda_2=688 nm and lambda_3=710 nm and the final value is the true chlorophyll-a value. Or use the example input dataset provided in
`example_input_output/example3/example_3_input.csv`

3) Issue the following command
 `python run_model_on.py --model ensemble/ --dataset example_input_output/example3/example_3_input.csv --out_dir example_input_output/example3/ --lambda1 668 --lambda2 688 --lambda3 710 --ensemble --true-chl`

 4) The output from this should produce the file `chl-CP.csv` which should be the same as the file `example_input_output/example3/example_3_output.csv`

### Example 4: Estimating chl using dataset with cp wavelengths taken at every 2 nm from 620 nm to 710 nm with true chl value ###

1) Enter the cmap conda environment:
conda activate cmap

2) prepare a csv file as follows:
0.027307, 0.027255, ... , 0.025031, 0.024865, 0.141172
Where there are 47 columns with cp taken at interval of 2 nm within range of 620 nm - 710 nm and the final column as the true chlorophyll-a value. Or use the example input dataset provided in
`example_input_output/example4/example_4_input.csv`

3) Issue the following command
`python run_model_on.py --model ensemble/ --dataset example_input_output/example3/example_3_input.csv --out_dir example_input_output/example3/ --lambda1 668 --lambda2 688 --lambda3 710 --ensemble --true-chl`

 4) The output from this should produce the file `chl-CP.csv` which should be the same as the file `example_input_output/example4/example_4_output.csv`
