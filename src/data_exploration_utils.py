import math
import matplotlib.pyplot as plt
import seaborn as sns


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
    plt.xlabel(column_name)
    plt.ylabel("Frequency")
    plt.title("Histogram of " + column_name)


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

    for column_name in dataframe.columns:
        print(column_name)
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


def univariate_analysis_plots(data):
    num_cols = len(data.columns)
    num_plots_per_row = 5
    num_rows = math.ceil(num_cols / num_plots_per_row)

    fig, axes = plt.subplots(num_rows, num_plots_per_row, figsize=(18, num_rows * 4))
    fig.subplots_adjust(hspace=0.5)

    for i, column in enumerate(data.columns):
        ax = axes[i // num_plots_per_row, i % num_plots_per_row] if num_rows > 1 else axes[i % num_plots_per_row]

        if data[column].dtype == 'object':
            sns.countplot(data[column], ax=ax)
            ax.set_title(f'Countplot of {column}')
            ax.set_xlabel('')
            ax.set_ylabel('Count')
            ax.tick_params(axis='x', rotation=45)
        else:
            sns.histplot(data[column], ax=ax, kde=True)
            ax.set_title(f'Distribution of {column}')
            ax.set_xlabel(column)
            ax.set_ylabel('Frequency')

    # Remove empty subplots
    for i in range(num_cols, num_rows * num_plots_per_row):
        fig.delaxes(axes.flatten()[i])

    plt.tight_layout()
    plt.show()


def boxplot_per_column(data):
    num_cols = len(data.columns)
    num_plots_per_row = 5
    num_rows = math.ceil(num_cols / num_plots_per_row)

    fig, axes = plt.subplots(num_rows, num_plots_per_row, figsize=(18, num_rows * 4))
    fig.subplots_adjust(hspace=0.5)

    for i, column in enumerate(data.columns):
        ax = axes[i // num_plots_per_row, i % num_plots_per_row] if num_rows > 1 else axes[i % num_plots_per_row]

        sns.boxplot(y=data[column], ax=ax)
        ax.set_title(f'Boxplot of {column}')
        ax.set_ylabel(column)
        ax.set_xlabel('')
        ax.tick_params(axis='x', rotation=45)

    # Remove empty subplots
    for i in range(num_cols, num_rows * num_plots_per_row):
        fig.delaxes(axes.flatten()[i])

    plt.tight_layout()
    plt.show()
    return None


def bivariate_analysis(data, target_column):
    num_cols = len(data.columns)
    num_plots_per_row = 5
    num_rows = (num_cols - 1) // num_plots_per_row + 1

    fig, axes = plt.subplots(num_rows, num_plots_per_row, figsize=(18, num_rows * 4))
    fig.subplots_adjust(hspace=0.5)

    target_data = data[target_column]
    other_columns = [col for col in data.columns if col != target_column]

    for i, column in enumerate(other_columns):
        ax = axes[i // num_plots_per_row, i % num_plots_per_row] if num_rows > 1 else axes[i % num_plots_per_row]

        if data[column].dtype == 'object':
            sns.boxplot(x=data[column], y=target_data, ax=ax)
            ax.set_title(f'{target_column} vs {column}')
            ax.set_xlabel(column)
            ax.set_ylabel(target_column)
            ax.tick_params(axis='x', rotation=45)
        else:
            sns.scatterplot(x=data[column], y=target_data, ax=ax)
            ax.set_title(f'{target_column} vs {column}')
            ax.set_xlabel(column)
            ax.set_ylabel(target_column)

    # Remove empty subplots
    for i in range(len(other_columns), num_rows * num_plots_per_row):
        fig.delaxes(axes.flatten()[i])

    plt.tight_layout()
    plt.show()
    return None


def correlation_matrix(data):
    # Calculate the correlation matrix
    corr_matrix = data.corr()

    # Set up the matplotlib figure
    plt.figure(figsize=(10, 8))

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', square=True)
    plt.title('Correlation Matrix')

    # Show plot
    plt.tight_layout()
    plt.show()
    return None
