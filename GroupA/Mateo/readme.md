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



## Preprocessing and merging the datasets

After loading the three datasets, the notebook prepares them for merging and modelling.

### Target variable selection

The notebook supports four possible WPPSI-III target variables, which can be selected by setting the `TARGET_VARIABLE` and `AFTER_MERGE_TARGET_VARIABLE` in the `.env` file. The available options are:

- ``QIT``: Full-scale IQ score
- ``QIV``: Verbal IQ score
- ``QIP``: Performance IQ score
- ``NCL``: Non-verbal Cognitive Level score

For this analysis, the target variable is set to ``QIT`` (Full-scale IQ score), and the corresponding after-merge target variable is set to ``Note composite_(QIT)``.

### Selecting M0 records

The HRV datasets contain measurements from different recording periods. Only measurements taken at ``M0`` are retained.

The notebook locates the ``Passage`` column in the multi-level column index, converts its values to strings, removes surrounding whitespace, and filters both HRV datasets:

```python
df_linear_m0 = df_linear.loc[passage.eq("M0")].copy()
df_non_linearity_m0 = df_non_linearity.loc[passage.eq("M0")].copy()
```

This ensures that the linear and nonlinear HRV features correspond to the same neonatal recording period

### Removing missing target values

Participants without QTI scores cannot be used for supervised regression. They are therefore removed from the WPPSI-III results dataset:

```python
df_results_clean = df_results.dropna(
    subset=[("Note composite", TARGET_VARIABLE)]
).copy()
```

Only rows with a valid QIT value are retained

### Preparing column names

The HRV and WPPSI datasets initially use pandas ``MultiIndex`` columns. These are converted into single strings to simplify the merge operations.

For example, a column with the levels ``("Feature", "unit")`` becomes ``"Feature_unit"``. 

Columns without a description retain only their main name.

### Merging lienar and non-linear HRV datasets

The two HRV datasets are merged using an inner join on:
- ``Monogramme`` : Unique participant identifier
- ``DDN`` : Date of birth
- ``Passage`` : Recording period

```python
df_merged = pd.merge(
    df_linear_m0_clean,
    df_non_linearity_m0_clean,
    on=["Monogramme", "DDN", "Passage_(-)"],
    how="inner",
)
```

The inner join keeps only participants present in both HRV datasets.

### Merging HRV and WPPSI-III datasets

The merged HRV dataframe is then joined with the cleaned WPPSI results using:

- ``Monogramme`` : Unique participant identifier
- ``DDN`` : Date of birth

Only the participant identifiers and the selected QIT target are retained from the WPPSI dataframe. The final merged dataset is stored in ``df_merged_full`` and saved to the path specified in the `.env` file.

### Removing empty columns and handling missing values

The ``drop_empty()`` function performs the first missing-value cleanup:

1. Removes columns that are entirely empty
2. Converts missing-values markers such as ``"-"`` and ``"NaN"`` to actual ``NaN`` values
3. Removes columns that became completely empty after the conversion
4. Replaces missing values in the remaining columns with the mean of each column

The cleaned dataframe is stored as ``df_merged_full_clean``. The notebook then reports its dimensions and remaining missing values.

