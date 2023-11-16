import pandas as pd

def get_one_hot_encoded_dataframe(dataframe):
    """Retrieves the one hot encoded dataframe
     
    :param dataframe: input dataframe
    :return: the associated one hot encoded dataframe"""
    if dataframe is None:
        return None
    return pd.get_dummies(data=dataframe)

def remove_nan_through_mean_imputation(dataframe):
    """Remove NaN (Not a Number) entries through mean imputation
     
    :param dataframe: input dataframe
    :return: the dataframe with  NaN (Not a Number) entries replaced using mean imputation"""
    if dataframe is None:
        return None
    for col in dataframe.columns:
        dataframe[col] = dataframe[col].fillna(dataframe[col].mean())
    return dataframe