import pandas as pd
from pandas import DataFrame
from pathlib import Path
from numpy import array

def read_dataframes() -> (DataFrame, DataFrame, DataFrame):
    """Read csv files from data folder and return three dataframes as tuple

        Returns:
            Tuple containing 3 dataframes, corresponding to weekly sales, features and stores respectively.
    """

    data_folder = Path("data/")

    # read weekly sales df
    weekly_sales_csv_path = data_folder / "weekly_sales.csv"
    weekly_sales_df = pd.read_csv(weekly_sales_csv_path)

    # read features df
    features_csv_path = data_folder / "features.csv"
    features_df = pd.read_csv(features_csv_path)

    # read stores df
    stores_csv_path = data_folder / "stores.csv"
    stores_df = pd.read_csv(stores_csv_path)

    return weekly_sales_df, features_df, stores_df

def impute_data(df: DataFrame, column: str):
    """Impute data for missing values in dataset given null indexes and column name

        Arguments:
            - df (pd.DataFrame): dataframe with missing values
            - column (str): name of column to be affected

        Affects dataframe in place
    """

    na_values = pd.isna(df)
    na_indexes = na_values[column].astype(int).to_numpy().nonzero()[0]

    for i, impute_index in enumerate(na_indexes):
        na_indexes_index = i
        value_index = impute_index - 1

        while df.at[value_index, column] == 'NaN':
            if value_index < 1:
                value_index = na_indexes[na_indexes_index + 1] - 1
                na_indexes_index += 1
            else:
                value_index -= 1

        df.at[impute_index, column] = df.at[value_index, column]
