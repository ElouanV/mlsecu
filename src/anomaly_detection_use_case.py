from mlsecu import data_exploration_utils, data_preparation_utils
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import numpy as np


def get_list_of_attack_types(dataframe):
    """Retrieves the name of attack types of a pandas dataframe

    :param dataframe: input dataframe
    :return: the name of distinct attack types"""
    if dataframe is None:
        return None
    dataframe = data_preparation_utils.remove_nan_through_mean_imputation(
        data_preparation_utils.get_one_hot_encoded_dataframe(dataframe=dataframe)
    )
    dataframe = data_preparation_utils.remove_nan_through_mean_imputation(
        data_preparation_utils.get_one_hot_encoded_dataframe(dataframe=dataframe)
    )
    return dataframe["attack types"].unique().tolist()


def get_nb_of_attack_types(dataframe):
    """Retrieves the number of distinct attack types of a pandas dataframe

    :param dataframe: input dataframe
    :return: the number of distinct attack types"""
    if dataframe is None:
        return None
    dataframe = data_preparation_utils.remove_nan_through_mean_imputation(
        data_preparation_utils.get_one_hot_encoded_dataframe(dataframe=dataframe)
    )
    return len(get_list_of_attack_types(dataframe=dataframe))


def get_list_of_if_outliers(dataframe, outlier_fraction):
    """Extract the list of outliers according to Isolation Forest algorithm

    :param dataframe: input dataframe
    :param outlier_fraction: rate of outliers to be extracted
    :return: list of outliers according to Isolation Forest algorithm"""
    if dataframe is None:
        return None
    dataframe = data_preparation_utils.remove_nan_through_mean_imputation(
        data_preparation_utils.get_one_hot_encoded_dataframe(dataframe=dataframe)
    )
    isolfst = IsolationForest(random_state=42, contamination=outlier_fraction)
    return np.where(isolfst.fit_predict(dataframe) == -1)[0]


def get_list_of_lof_outliers(dataframe, outlier_fraction):
    """Extract the list of outliers according to Local Outlier Factor algorithm

    :param dataframe: input dataframe
    :param outlier_fraction: rate of outliers to be extracted
    :return: list of outliers according to Local Outlier Factor algorithm"""

    if dataframe is None:
        return None
    dataframe = data_preparation_utils.remove_nan_through_mean_imputation(
        data_preparation_utils.get_one_hot_encoded_dataframe(dataframe=dataframe)
    )
    clf = LocalOutlierFactor(contamination=outlier_fraction)
    return np.where(clf.fit_predict(dataframe) == -1)[0]


def get_list_of_parameters(dataframe):
    """Retrieves the list of parameters of a pandas dataframe

    :param dataframe: input dataframe
    :return: list of parameters"""
    if dataframe is None:
        return None
    return data_exploration_utils.get_column_names(dataframe=dataframe)


def get_nb_of_if_outliers(dataframe, outlier_fraction):
    """Extract the number of outliers according to Isolation Forest algorithm

    :param dataframe: input dataframe
    :param outlier_fraction: rate of outliers to be extracted
    :return: number of outliers according to Isolation Forest algorithm"""
    if dataframe is None:
        return None
    return len(
        get_list_of_if_outliers(dataframe=dataframe, outlier_fraction=outlier_fraction)
    )


def get_nb_of_lof_outliers(dataframe, outlier_fraction):
    """Extract the number of outliers according to Local Outlier Factor algorithm

    :param dataframe: input dataframe
    :param outlier_fraction: rate of outliers to be extracted
    :return: number of outliers according to Local Outlier Factor algorithm"""
    if dataframe is None:
        return None
    return len(
        get_list_of_lof_outliers(dataframe=dataframe, outlier_fraction=outlier_fraction)
    )


def get_nb_of_occurrences(dataframe):
    """Retrieves the number of occurrences of a pandas dataframe

    :param dataframe: input dataframe
    :return: number of occurrences"""
    if dataframe is None:
        return None
    return data_exploration_utils.get_nb_of_rows(dataframe=dataframe)


def get_nb_of_parameters(dataframe):
    """Retrieves the number of parameters of a pandas dataframe

    :param dataframe: input dataframe
    :return: number of parameters"""
    if dataframe is None:
        return None
    return len(get_list_of_parameters(dataframe=dataframe))
