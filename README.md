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
- `convert_price_window_json_csv.py` : convert price window data structure files from JSON to CSV format and vice-versa

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

Once the model has run, results will be printed to the terminal, and saved in a CSV log file called `parametric_model_log.csv`. Running multiple models will append new lines onto this log file.

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

Once the model has run, results will be printed to the terminal, and saved in a CSV log file called `random_forest_model_log.csv`. Running multiple times will append new lines onto this log file.

**Advanced:** To save a model once it has been trained, set variable `save_model = True` in the parameters section of the python script.

### Running LSTM models
To train and test LSTM models run the script `run_lstm_model.py`
- arg1 = region in {wa, nsw, de}
- arg2 = number of training epochs (positive integer)
- arg3 = ensemble model boolean in {0, 1}

A training epoch is a single run through the data set. Model fit will increase with the number of epochs until over-fitting is achieved. For the paper 100 epochs was used, less than 10 is not recommended. 

The ensemble model bool indicates whether to use ensemble LSTM model or basic one. 0 will give basic model, 1 will give ensemble model.

Once the model has run, results will be printed to the terminal, and saved in a CSV log file called `lstm_model_log.csv`. Running multiple times will append new lines onto this log file.

**Advanced:** To save a model once it has been trained, set variable `save_model = True` in the parameters section of the python script.

**Advanced:** Results from Figure 2 - Gains from Additional Data, can be simulated by changing the variable `train_fraction` in the parameters of any of the LSTM, RF, or Parametric models. This will modify the proportion of the data set that is used to train the models.

### Use pre-trained models to classify external data (Advanced)
To use previously trained and saved models to classify a data set contained in a JSON or CSV file:

1. Ensure that the dataset has the same format as the price window files either in CSV or JSON (ie. like `label_databases/german_label_db.json`). Not all data fields need to be present, but there must at least be price series and a unique identifier column for the observations. For example of CSV format, see output of `convert_price_window_json_csv.py`.
2. Modify basic settings in set parameters section of the `nonparamtric_classify_external_data.py` script. You will need to insert:
   - training set hash (see logs) from RF or LSTM model that you previously trained and saved
   - the path and filename to your external data file in JSON or CSV.
   - the type of model in {'rf', 'lstm_basic', 'lstm_ensemble'}
   - the filename where you wish to save the results (either CSV or JSON extensions accepted)
3. Run script using `python nonparamtric_classify_external_data.py`
4. Classification results can be found in the chosen file

**Note:** The performance of the models will generally be negatively affected by biases or other features of the external data that were not also in the training data. Proceed with caution when using this feature. 


### Parse Tankerkoenig DE raw data into convenient data structures
To efficiently parse the [Tankerkoenig](https://tankerkoenig@dev.azure.com/tankerkoenig/tankerkoenig-data/_git/tankerkoenig-data) raw data into easy to use data structures:
1. Download the tankerkoenig data set using:
   - ``git clone https://tankerkoenig@dev.azure.com/tankerkoenig/tankerkoenig-data/_git/tankerkoenig-data``
- Run the script `de_rawdata_parse_postal_region.py`
  - arg1 = postal region in {0...9, all}

German postal region is defined as the region represented by the set of postal codes starting with the specified digit. More information can be found at this [Wikipedia article](https://en.wikipedia.org/wiki/Postal_codes_in_Germany). Passing the argument 'all' will parse all regions from 0 to 9.

The output of this script will be found by default in the folder `de_databases`. There will be three types of file in here:
1. Station Info Database: This is the file that contains only the header data for the various gas stations. No price data is included in this file, to make it small and fast to load. This file is available by default in both JSON as well as serialized Python pkl format. 
2. Price Station Database: These files contain a serialized dictionary of station objects (similar to the station info database) that contains a list of price observations and timestamps of all of the price reports from the given station. These databases are partitioned into postal regions.
3. Price Windows Files: These JSON files contain series of quarterly observations of daily average prices for each station in the given postal region. This 90-day price window data structure is the basis of the Detecting Edgeworth Cycles paper.

**Note:**
- Processing a region should take about 20-50 minutes, thus fully processing all regions will take several hours.
- This step will require at least 8GB of RAM memory, but 16GB or more is recommended.
- The script is preconfigured to expect the folder `tankerkoenig-data` in the main directory. This can be configured using the `framework/de_data_parsing/parameters.py dirs['raw_german_data_folder']` variable.
- The script will ask before overwriting data.

### Convert JSON price window files to CSV and vice-versa
Price window data files as produced by the `de_rawdata_parse_postal_region.py` can be converted into CSV files using the script `convert_price_window_json_csv.py`. Conversion can also be performed on user generated data between CSV and JSON formats, provided the data conforms to the same structure as the base provided data. 

To use the converter run the script `convert_price_window_json_csv.py`
  - arg1 = input_filename (str)

The name of the input file must include a path and be either a JSON or CSV file.

Output:
  - For JSON input, it converts the data to a CSV file with the same name.
  - For CSV input, it converts the data to a JSON file with the same name.