# -*- coding: utf-8 -*-
"""
__author__ = "Mahamdi Mohammed"
__copyright__ = "Copyright 2020, PFE"
__license__ = "MIT"
__version__ = "0.2"
__email__ = "fm_mahamdi@esi.dz"
__status__ = "Production"
"""
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import StandardScaler
from pandas.api.types import is_string_dtype, is_numeric_dtype, is_categorical_dtype
from IPython.display import display


# add the headers to the data
def add_header(headers_path, resource_path, input_filename, output_filename):
    headers = []

    # Read the data
    data = pd.read_csv(Path(resource_path, input_filename), sep=',')
    # create the column names
    with open(Path(resource_path, headers_path)) as cols:
        for line_num, col_name in enumerate(cols):
            headers.append(col_name.rstrip())

    # Set the column headers to the names from the Wireshark frame
    data.columns = headers

    # Export the pandas DataFrame as a CSV file
    data.to_csv(Path(resource_path, output_filename), index=False, sep=',')


# display all the data
def display_all(data_frame):
    with pd.option_context("display.max_rows", 1000):
        with pd.option_context("display.max_columns", 1000):
            display(data_frame)


# Change any columns of strings in a panda's dataframe to a column of categorical values.
# This applies the changes inplace.
def str_to_cat(df):
    for n, c in df.items():
        if is_string_dtype(c): df[n] = c.astype('category').cat.as_ordered()


# Get a random sample of n rows from df, without replacement
def get_sample(df, n):
    indexes = sorted(np.random.permutation(len(df))[:n])
    return df.iloc[indexes].copy()


# Fill missing data in a column of df with the median, and add a {name}_na column
#     which specifies if the data was missing.
def fix_missing(df, col, name, na_dict):
    """
    Examples:
    ---------
    >>> df = pd.DataFrame({'col1' : [1, np.NaN, 3], 'col2' : [5, 2, 2]})
    >>> df
       col1 col2
    0     1    5
    1   nan    2
    2     3    2
    >>> fix_missing(df, df['col1'], 'col1', {})
    >>> df
       col1 col2 col1_na
    0     1    5   False
    1     2    2    True
    2     3    2   False
    >>> df = pd.DataFrame({'col1' : [1, np.NaN, 3], 'col2' : [5, 2, 2]})
    >>> df
       col1 col2
    0     1    5
    1   nan    2
    2     3    2
    """
    if is_numeric_dtype(col):
        if pd.isnull(col).sum() or (name in na_dict):
            df[name + '_na'] = pd.isnull(col)
            filler = na_dict[name] if name in na_dict else col.median()
            df[name] = col.fillna(filler)
            na_dict[name] = filler
    return na_dict


def scale_vars(df, mapper):
    # warnings.filterwarnings('ignore', category=sklearn.exceptions.DataConversionWarning)
    # on soustrait aux données leur moyenne empirique # 𝜇 on les divisent par leur écart-type 𝛿 
    if mapper is None:
        map_f = [([n], StandardScaler()) for n in df.columns if is_numeric_dtype(df[n])]
        mapper = DataFrameMapper(map_f).fit(df)
    df[mapper.transformed_names_] = mapper.transform(df)
    return mapper


# Changes the column col from a categorical type to it's integer codes.
def cat_to_int(df, col, name, max_n_cat):
    if not is_numeric_dtype(col) and (max_n_cat is None or len(col.cat.categories) > max_n_cat):
        df[name] = pd.Categorical(col).codes + 1


# takes a  df and splits off the response variable Y
# and changes the df into an entirely numeric df.
# For each column of df (which is not in skip_flds nor in ignore_flds) na values are replaced
# by the median value of the column.
def proc_df(df, y_fld=None, skip_flds=None, ignore_flds=None, do_scale=False, na_dict=None,
            preproc_fn=None, max_n_cat=None, subset=None, mapper=None):
    if not ignore_flds: ignore_flds = []
    if not skip_flds: skip_flds = []
    if subset:
        df = get_sample(df, subset)
    else:
        df = df.copy()
    ignored_flds = df.loc[:, ignore_flds]
    df.drop(ignore_flds, axis=1, inplace=True)
    if preproc_fn: preproc_fn(df)
    if y_fld is None:
        y = None
    else:
        if not is_numeric_dtype(df[y_fld]): df[y_fld] = pd.Categorical(df[y_fld]).codes
        y = df[y_fld].values
        skip_flds += [y_fld]
    df.drop(skip_flds, axis=1, inplace=True)

    if na_dict is None:
        na_dict = {}
    else:
        na_dict = na_dict.copy()
    na_dict_initial = na_dict.copy()
    for n, c in df.items(): na_dict = fix_missing(df, c, n, na_dict)
    if do_scale: mapper = scale_vars(df, mapper)
    for n, c in df.items(): cat_to_int(df, c, n, max_n_cat)
    df = pd.get_dummies(df, dummy_na=True)
    df = pd.concat([ignored_flds, df], axis=1)
    res = [df, y, na_dict]
    if do_scale: res = res + [mapper]
    return res


if __name__ == '__main__':
    input_file_name = 'AWID-ATK-R-Train/Train.csv'
    output_file_name = 'AWID-ATK-R-Train/Train_with_headers.csv'
    resource_dir = Path('./', 'Dataset')
    header_path = 'col_names.txt'
    add_header(header_path, resource_dir, input_file_name, output_file_name)
