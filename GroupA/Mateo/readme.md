# readme

This markdown goes through the notebook in this folder, it is a summary of the steps taken to preprocess the data and train the model.

## Data loading and initialization

The notebook begins by importing the librairies required for data manipulation, visualization, preprocessing, and model training, model persistence. The main dependencies are `Numpy`, `Pandas`, `Matplotlib`, `Seaborn`, `Scikit-learn`, `LightGBM`, `XGBoost`, and `joblib`.

### Environment Configuration

The notebook is designed to run in Google Colab or VS Code. It first sets the working directory, loads environment variables from a `.env` file, and creates the directory used to save trained models if it does not exist. 

The `.env` file contains the following variables:

- `DATA_FILE`: Path to the linear HRV dataset
- `RESULTS_FILE`: Path to the WPPSI-III results dataset
- `NON_LINEARITY_FILE`: Path to the non-linear HRV dataset
- `WRITE_PATH`: Path to save the preprocessed and merged dataset
- `MODELS_PATH`: Path to save the trained models
- `TARGET_VARIABLE`: Name of the target variable in the WPPSI-III results dataset
- `AFTER_MERGE_TARGET_VARIABLE`: Name of the target variable after merging the datasets
- `WRITE_PATH_PLOTS`: Path to save the plots generated during the analysis

**A template `.env.exemple` file is provided in the repository. Users should create a `.env` file based on this template and fill in the appropriate paths and variable names.**


### Loading HRV data

The `load_data()` function reads the linear and non-linear HRV files using pandas:

- Tabulation is used as the separator
- Some specific characters are replaced to ensure proper parsing of the data
- No header is assumed because the datasets contain two metadata rows at the beginning
- The first row contains the feature names
- The second row contains the units or description of the features
- These two rows are used to create a multi-index for the columns of the DataFrame for this section
- The original metadata rows are dropped from the DataFrame after creating the multi-index
- The dataframe index is reset

The two resulting dataframes are :
- `df_linear`: contains the linear HRV features
- `df_nonlinearity`: contains the non-linear HRV features

Their first rows are displayed to verify that the laoding operation was successful.

### Loading WPPSI-III results data

The WPPSI-III results are loaded separately with ``load_results()``.

This file also contains two header rows:

1. A group or category header
2. The actual feature names

Empty group headers are forward-filled so that each column receives the correct group label. The two rows are then converted into a pandas ``MultiIndex``, removed from the dataset, and the index is reset.

The resulting dataframe is stored in ``df_results``, and its first ten rows are displayed for verification

### Important execution note 

**The notebook expects tab-separated files, despite the ``.csv`` extension used by the project datasets. The paths in ``.env`` must therefore point to the correct local data files. These files contain sensitive health information and must remain local, as specified in the project README.md.**



