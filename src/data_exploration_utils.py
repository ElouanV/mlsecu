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