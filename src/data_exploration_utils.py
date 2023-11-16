import pandas as pd
import matplotlib.pyplot as plt


def get_column_names(dataframe):
    """Get the name of columns in the dataframe
     
    :param dataframe: input dataframe
    :return: name of columns"""
    if dataframe is None:
        return None
    return dataframe.columns.tolist()


def get_nb_of_dimensions(dataframe):
    """Retrieves the number of dimensions of a pandas dataframe
     
    :param dataframe: input dataframe
    :return: number of dimensions"""
    if dataframe is None:
        return None
    return dataframe.shape[1]


def get_nb_of_rows(dataframe):
    """Get the number of rows
     
    :param dataframe: input dataframe
    :return: number of rows"""
    if dataframe is None:
        return None
    return dataframe.shape[0]


def get_number_column_names(dataframe):
    """Get the name of numeric columns
     
    :param dataframe: input dataframe
    :return: names of object columns"""
    if dataframe is None:
        return None
    return dataframe.select_dtypes(include=["number"]).columns.values.tolist()


def get_object_column_names(dataframe):
    """Get the name of object columns
     
    :param dataframe: input dataframe
    :return: name of object columns"""
    if dataframe is None:
        return None
    return dataframe.select_dtypes(object).columns.values.tolist()


def get_unique_values(dataframe, column_name):
    """Get the unique values for a given column
     
    :param dataframe: input dataframe
    :param column_name: target column label
    :return: unique values for a given column"""
    if dataframe is None:
        return None
    return dataframe[column_name].unique()


def plot_univariate_histogram(dataframe, column_name):
    """Plot the histogram of a given column

    :param dataframe: input dataframe
    :param column_name: target column label
    :return: None"""
    if dataframe is None:
        return None

    plt.hist(dataframe[column_name])


def plot_univariate_boxplot(dataframe, column_name):
    """Plot the boxplot of a given column

    :param dataframe: input dataframe
    :param column_name: target column label
    :return: None"""
    if dataframe is None:
        return None

    plt.boxplot(dataframe[column_name])


def plot_all_univariate_histogram(dataframe):
    """Plot the histogram of all columns

    :param dataframe: input dataframe
    :return: None"""
    if dataframe is None:
        return None

    for column_name in get_number_column_names(dataframe):
        plot_univariate_histogram(dataframe, column_name)
        plt.show()


def plot_all_univariate_boxplot(dataframe):
    """
    Plot the boxplot of all columns
    :param dataframe:
    :return:
    """
    if dataframe is None:
        return None
    for column_name in get_number_column_names(dataframe):
        plot_univariate_boxplot(dataframe, column_name)
        plt.show()


def plot_bivariate_scatterplot(dataframe, column_name1, column_name2):
    """
    Plot the scatterplot of two given columns
    :param dataframe: input dataframe
    :param column_name1: target column label
    :param column_name2: target column label
    :return: None"""
    if dataframe is None:
        return None
    plt.scatter(dataframe[column_name1], dataframe[column_name2])


def plot_all_bivariate_scatterplot(dataframe):
    """
    Plot the scatterplot of all columns
    :param dataframe: input dataframe
    :return: None"""
    if dataframe is None:
        return None
    for column_name1 in get_number_column_names(dataframe):
        for column_name2 in get_number_column_names(dataframe):
            plot_bivariate_scatterplot(dataframe, column_name1, column_name2)
            plt.show()


