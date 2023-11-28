import pandas as pd
import numpy as np
from data_helpers import *

weekly_sales_df, features_df, stores_df = read_dataframes()

# check the features/columns of the other data sets
general_features = features_df.columns
store_features = stores_df.columns

# since Store is common in both, we can merge
merged_features = features_df.merge(stores_df, on='Store')
# print(merged_features)

# check na values
na_values = pd.isna(merged_features)
na_features = na_values.any()
print(f"Features with NA values:\n {na_features}")
print(f"Size of na values: {len(na_values)}")
print(f"Sum of Markdown1 NAs: {sum(na_values['MarkDown1'])}")
print(f"Sum of CPI NAs: {sum(na_values['CPI'])}")
print(f"Sum of Unemployment NAs: {sum(na_values['Unemployment'])}")

# since Markdown columns are NA for over half the dataset, we will drop them

markdowns = [
    'MarkDown1',
    'MarkDown2',
    'MarkDown3',
    'MarkDown4',
    'MarkDown5'
]

merged_features = merged_features.drop(columns=markdowns)
      
# check remaining null values in columns: CPI and unemployment
# do they correspond to the same rows?

na_cpi_int = na_values['CPI'].astype(int)
na_indexes_cpi = na_cpi_int.to_numpy().nonzero()[0]
na_une_int = na_values['Unemployment'].astype(int)
na_indexes_une = na_une_int.to_numpy().nonzero()[0]

assert np.array_equal(na_indexes_cpi, na_indexes_une)

# since they do, we can take a closer look at those rows

na_rows = merged_features.iloc[na_indexes_cpi]
print(f"Missing value weeks: {na_rows['Date'].unique()}")  # missing value weeks

# assert that the na values correspond to final 13 weeks
assert np.array_equal(na_rows['Date'].unique(), merged_features['Date'].unique()[-13:])

# CPI and Unemployment rate are slow-changing values, and can be extrapolated

print(f"First index of missing value CPI: {na_indexes_cpi[0]}")  # first missing value row index, corresponds to 169
print(f"First index of missing value Unemployment: {na_indexes_une[0]}")  # first missing value row index, corresponds to 169

impute_data(merged_features, 'CPI')
impute_data(merged_features, 'Unemployment')

print(merged_features.iloc[na_indexes_cpi])
