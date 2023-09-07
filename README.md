# Detecting Edgeworth Cycles
Replication package for paper "[Detecting Edgeworth Cycles](https://ssrn.com/abstract=3934367)" by Timothy Holt, Mitsuru Igami, and Simon Scheidegger (2023). We hope that this repository can also serve as a "sandbox" for researchers interested in studying gasoline price data in more detail by providing various tools to assist in the data analysis process.

The scripts in this repository allow the user to:
- Test the various parametric, random forest, and LSTM models that were detailed in the paper.
- Plot random samples of the labeled data. 
- Use the non-parametric machine learning models to classify external data.
- Efficiently parse the [Tankerkoenig](https://tankerkoenig@dev.azure.com/tankerkoenig/tankerkoenig-data/_git/tankerkoenig-data) German retail gasoline price data into easily usable data structures including CSV.

## Cloning the Repository
To run the code, you must first download (clone) this repository:
1. Open a terminal and navigate to your desired directory using   

        cd directory_path
2. Clone the repository using:
        
        git clone https://github.com/tabholt/detecting_edgeworth_cycles.git

    **Note:** This command may not work on Windows, unless you have previously installed git. In this case, you may download this repository from the GitHub webpage and then unzip it in your desired directory. 

3. Navigate into the repository using:

        cd detecting_edgeworth_cycles

     **Note:** You may name this directory anything you like. In this case, update the name after the `cd` command. 

## Package Contents

### User Scripts (`main` directory)
- `plot_sample.py` : plot a random sample of data from a given region
- `run_parametric_models.py` : train and test parametric models
- `run_rf_model.py` : train and test random forest model
- `run_lstm_model.py` : train and test LSTM models
- `nonparamtric_classify_external_data.py` : use pre-trained models to classify external data sets using LSTM or Random Forest models
- `de_rawdata_parse_postal_region.py` : efficiently parse raw [Tankerkoenig](https://tankerkoenig@dev.azure.com/tankerkoenig/tankerkoenig-data/_git/tankerkoenig-data) data into convenient data structures
- `convert_price_window_json_csv.py` : convert price window data structure files from json to csv format and vice-versa

### Framework Code (`main/framework` directory)
- `Estimation_Framework.py` : main code defining the parametric models and feature interfaces
- `RF_Framework.py` : main code defining the random forest models
- `LSTM_Framework.py` : main code defining the LSTM models
- `Label_Class.py` : defines the data structure underpinning single observations, as well as collections of observations
- `Model_Loader.py` : a convenient interface for loading data and splitting into training and test sets
- `model_settings.py` : dictionaries and lists for some important default parameters to make the framework function

### DE Raw Data Parsing Framework Code (`main/framework/de_data_parsing` directory)
- `sta_info_functions.py` : functions to parse station info files (excluding prices)
- `price_db_functions.py` : functions to parse station price files
- `observation_windows_functions.py` : functions to create station window observations of quarterly average daily prices
- `Station_Class.py` : data structure representing a single gas station with all relevant data
- `Timer_Utility.py` : convenient utility for recording timing and performance in Python code
- `parameters.py` : set of default parameters for parsing german raw data

### Data Files (`main/label_databases` direcotry)
- `german_label_db.json` : data from Germany
- `nsw_label_db.json` : data from New South Wales
- `wa_label_db.json` : data from Western Australia
  
**Note:** data files must be unzipped after downloading repository (see Running the Scripts section for details)

### Downloadable Data Files (not included in directory)
- tankerkoenig German raw data (updated daily): 
  - `git clone https://tankerkoenig@dev.azure.com/tankerkoenig/tankerkoenig-data/_git/tankerkoenig-data`
- Detrended labeled data for all DE price windows from Q4-2014 to Q4-2020 (inclusive):
  - `cd labeled_databases`
  - `curl https://drive.switch.ch/index.php/s/pwq1Sw0RssyDuUC/download --output ALL_detrended_price_windows.json` 
  - Only about 10 percent of observations contain human labels. {1: cycling, 0.5: maybe cycling, 0: not cycling}

## Requirements
This code requires Python 3.8 or later, with the following packages and their associated dependencies:
  - matplotlib (3.7.1)
  - numpy (1.24.3)
  - pandas (1.5.3)
  - python (3.10.11)
  - scikit-learn (1.2.2)
  - scipy (1.10.1)
  - seaborn (0.12.2)
  - tensorflow (2.10.0)

**Note:** The code should be broadly compatible with recent versions of the above packages. Specific version numbers are included only for long-term replicability purposes.

The easiest way to create an environment to run the code is using **[Miniconda](https://conda.io/projects/conda/en/stable/user-guide/install/)** or **[Anaconda](https://docs.anaconda.com/free/anaconda/install/)**.

It is recommended to use Miniconda, since this is the simplest and lightest installation but the following setup instructions will work for both Miniconda and Anaconda. Miniconda must be installed using the terminal, while Anaconda has a graphical interface installer. 

## Environment Setup Instructions
1. Download and install **[Miniconda](https://conda.io/projects/conda/en/stable/user-guide/install/)** or **[Anaconda](https://docs.anaconda.com/free/anaconda/install/)**
2. Test your installation using 

        conda list

3. Create a new environment (set of installed packages) from the provided environment yml file using 
        
        conda env create -f replication_conda_environment.yml
    **Note:** you will need to be in the repository main folder for this to work since it requires the `replication_conda_environment.yml` file located in the main folder.
4. Activate new environment using: 
   
        conda activate edgeworth_replication_env

5. (optional) Verify installation was successful using `conda list` and verifying that the packages noted in the requirements section are listed

**Note:** You will need to use `conda activate edgeworth_replication_env` every time you open a new terminal session.

## Running the Scripts
Each of the scripts requires some arguments to be passed to it at run time to define your chosen parameters such as region or type of model to be run. These arguments will be denoted as arg1, arg2, arg3. To run a script you then use:

        python script_name.py arg1 arg2 arg3

replacing `script_name.py` with the name of your script, and the various arg1, arg2, arg3 with your chosen parameter values.

**Note:** You must extract the JSON files from the zip file `detecting_edgeworth_cycles/label_databases/label_data_files.zip` before you can run the scripts.

### Steps to unzip data files
1. Use your favorite unzipping program to extract the contents of the zipped file.
2. Ensure that the files: 
   - `german_label_db.json`
   - `nsw_label_db.json`
   - `wa_label_db.json`

    are in the directory `detecting_edgeworth_cycles/label_databases/`
### Plotting samples of data
To plot $n$ random samples of data from a given region run the script `plot_sample.py` 
- arg1 = region in {wa, nsw, de}
- arg2 = $n$ (positive integer)

**Note:** You will need to close the plot that pops-up in order to see the subsequent plot.

### Running parametric models
To train and test parametric models run the script `run_parametric_models.py`
- arg1 = region in {wa, nsw, de}
- arg2 = method in {PRNR, MIMD, NMC, MBPI, FT0, FT1, FT2, LS0, LS1, LS2, CS0, CS1, WAVY, all} - if 'all' is passed the the a model will be built, trained, and tested for each method.

Once the model has run, results will be printed to the terminal, and saved in a csv log file called `parametric_model_log.csv`. Running multiple models will append new lines onto this log file.

**Correspondences with Methods in Paper**

| Shortcut   | Description                                                     |
| ---------- | --------------------------------------------------------------- |
| PRNR       | **Method 1:** Positive Runs vs. Negative Runs                   |
| MIMD       | **Method 2:** Mean Increase vs. Mean Decrease                   |
| NMC        | **Method 3:** Negative Median Change                            |
| MBPI       | **Method 4:** Many Big Price Increases                          |
| FT0        | **Method 5:** Fourier Transform (maximum value)                 |
| FT1        | alternate Fourier Transform (tallest peak)                      |
| FT2        | alternate Fourier Transform (Herfindahl–Hirschman Index)        |
| LS0        | **Method 6:** Lomb-Scargle Periodogram (maximum value)          |
| LS1        | alternate Lomb-Scargle Periodogram (tallest peak)               |
| LS2        | alternate Lomb-Scargle Periodogram (Herfindahl–Hirschman Index) |
| CS0        | **Method 7:** Cubic Splines (number of roots)                   |
| CS1        | alternate Cubic Splines (integral value)                        |
| WAVY       | number of times detrended price crosses its mean                |


### Running Random Forest models
To train and test Random Forest models run the script `run_rf_model.py`
- arg1 = region in {wa, nsw, de}

Once the model has run, results will be printed to the terminal, and saved in a csv log file called `random_forest_model_log.csv`. Running multiple times will append new lines onto this log file.

### Running LSTM models
To train and test LSTM models run the script `run_lstm_model.py`
- arg1 = region in {wa, nsw, de}
- arg2 = number of training epochs (positive integer)
- arg3 = ensemble model boolean in {0, 1}

A training epoch is a single run through the data set. Model fit will increase with the number of epochs until over-fitting is achieved. For the paper 100 epochs was used, less than 10 is not recommended. 

The ensemble model bool indicates whether to use ensemble LSTM model or basic one. 0 will give basic model, 1 will give ensemble model.

Once the model has run, results will be printed to the terminal, and saved in a csv log file called `lstm_model_log.csv`. Running multiple times will append new lines onto this log file.

