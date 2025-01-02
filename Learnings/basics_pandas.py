import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def series_pd():
    """
    This function demonstrates the usage of pandas Series.
    It covers various operations like creation, access, manipulation, mathematical operations, 
    analytical operations, function application, sorting, conversion, and plotting.
    """

    # Create a pandas Series with specified data, index, and data type
    series1 = pd.Series(
        data=[10, 20, 30.0, 40, 50],
        index=['a', 'b', 'c', 'd', 'e'],
        dtype='int64',
    )

    # Create another pandas Series with specified data, index, and data type
    series2 = pd.Series(
        data=[1, 2, 3, 4, 5],
        index=['a', 'b', 'c', 'd', 'e'],
        dtype='int64',
    )

    # Print the series
    print(series1)
    print(series2)

    # Access the series data using index label
    print(series1['a'])

    # Access the series data using integer position
    print(series1.iloc[2])

    # Access the series data using a list of indices
    print(series1[['a', 'c', 'e']])

    # Access the series data using a boolean mask
    print(series1[series1 > 25])

    # Add a new element to the series
    series1['f'] = 60

    # Update an existing element in the series
    series1['a'] = 15

    # Delete an element from the series
    del series1['d']

    # Display operations
    print(series2.head(2))  # Display the first 2 elements
    print(series2.tail(2))  # Display the last 2 elements

    # Perform mathematical operations - as scalars
    print(series1 + 10)  # Add 10 to each element
    print(series1 - 5)  # Subtract 5 from each element
    print(series1 * 2)  # Multiply each element by 2
    print(series1 / 2)  # Divide each element by 2
    print(series1 ** 2)  # Square each element

    # Perform mathematical operations - as vectors
    print(series1 + series2)  # Add corresponding elements
    print(series1 - series2)  # Subtract corresponding elements
    print(series1 * series2)  # Multiply corresponding elements
    print(series1 / series2)  # Divide corresponding elements
    print(series1 ** series2)  # Raise each element to the power of corresponding element

    # Perform the analytical operations on the series
    print(series1.sum())  # Sum of all elements
    print(series1.mean())  # Mean of all elements
    print(series1.median())  # Median of all elements
    print(series1.mode())  # Mode of all elements
    print(series1.std())  # Standard deviation of all elements
    print(series1.var())  # Variance of all elements
    print(series1.min())  # Minimum value
    print(series1.max())  # Maximum value
    print(series1.idxmin())  # Index label of minimum value
    print(series1.idxmax())  # Index label of maximum value
    print(series1.cumsum())  # Cumulative sum
    print(series1.cumprod())  # Cumulative product
    print(series1.diff())  # First discrete difference
    print(series1.pct_change())  # Percentage change

    # Apply a function to the series
    def square(x):
        """
        This function squares a number.
        :param x: The number to be squared.
        :return: The squared number.
        """
        return x ** 2

    print(series1.apply(square))  # Apply the square function to each element

    # Sort the series in decreasing order
    print(series1.sort_index())  # Sort by index label
    print(series1.sort_values(ascending=False, inplace=True))  # Sort by value in descending order

    # Convert the series to a dictionary or list format
    series_dict = series1.to_dict()  # Convert to dictionary
    series_li = series1.index.to_list()  # Convert index to list
    df = pd.DataFrame(series1, columns=['value'])  # Convert to DataFrame
    print(df)
    print(series_li)
    print(series_dict)

    # Plot the series
    series1.plot()  # Line plot
    plt.show()

    # Plot a bar chart of the series
    series1.plot.bar()  # Bar plot
    plt.show()

    # Plot a pie chart of the series
    series1.plot.pie()  # Pie plot
    plt.show()

    # Plot a histogram of the series
    series1.plot.hist()  # Histogram plot
    plt.show()
    
def dataframe_pd(df: pd.DataFrame):
    """
    This function demonstrates the usage of pandas DataFrame.
    It covers various operations like creation, access, manipulation, conversion, and plotting.

    Parameters:
    df (pandas.DataFrame): The input DataFrame.

    Returns:
    None
    """

    # Print the first 2 rows of the DataFrame
    print(df.head(2))

    # Print the last 2 rows of the DataFrame
    print(df.tail(2))

    # Get information about the shape, data types, and other details of the DataFrame
    print(df.shape)
    print(df.dtypes)
    print(df.info())

    # Access a specific column of the DataFrame
    print(df['Name'])

    # Access a specific row of the DataFrame using its index label
    print(df.loc[1])

    # Access a specific cell of the DataFrame using its index label and column name
    print(df.loc[1, 'Name'])
    print(df['Name'].iloc[2])

    # Access multiple columns of the DataFrame
    print(df[['Name', 'Age']])

    # Access multiple rows of the DataFrame using a list of index labels
    print(df.loc[[0, 2, 4]])

    # Change the index of the DataFrame to a specific column
    df.set_index('Name', inplace=True)
    print(df)

    # Add a new column to the DataFrame
    df['Job Title'] = ['Manager', 'Senior Manager', 'Accountant', 'Analyst', 'HR Manager']
    print(df)

    # Update a specific cell in the DataFrame using its index label and column name
    df.loc[0, 'Age'] = 26
    print(df)

    # Delete a specific row from the DataFrame using its index label
    df.drop(index=0, inplace=True)
    print(df)

    # Delete a specific column from the DataFrame using its column name
    df.drop(columns='Job Title', inplace=True)
    print(df)

    # Filter rows of the DataFrame based on a specific condition
    print(df[df['Age'] > 30])

    # iterate over dataframe columns and values
    for col, value in df.items():
        print(f'Column: {col}, Value: {value}')
    for col, value in df['Age'].items():
        print(f'Index: {col}, Value: {value}')
    for row in df.iterrows():
        print(f'row: {row}')

    # Plot the DataFrame
    df.plot()
    df['Age'].plot.bar()
    df.hist()
    plt.tight_layout()
    plt.show()

def dataframe_stats(df: pd.DataFrame):
    """
    This function performs various statistical operations on a DataFrame.

    Parameters:
    df (pandas.DataFrame): The input DataFrame on which statistical operations will be performed.

    Returns:
    None
    """

    # count the values for each column in the DataFrame
    print(df.count())

    # get the unique values for a specific column in the DataFrame
    print(df['City'].unique())

    # get the number of unique values for a specific column in the DataFrame
    print(df['City'].nunique())

    # get the frequency of each unique value in a specific column of the DataFrame
    print(df['City'].value_counts())

    # drop duplicate rows from the DataFrame
    print(df.drop_duplicates())

    # fill missing values in the DataFrame with a specific value
    print(df.fillna('Unknown'))

    # fill missing values in the DataFrame with the mean of the respective column
    print(df.fillna(df['Age'].mean()))

    # sort the DataFrame by a specific column in ascending order
    print(df.sort_values('Age'))

    # perform vectorized mathematical operations on a specific column of the DataFrame
    print(df['Age'] + 5)  # add 5 to each value in the 'Age' column
    print(df['Salary'] * 1.1)  # multiply each value in the 'Salary' column by 1.1
    print(df['Age'] ** 2)  # square each value in the 'Age' column
    print(df['Salary'].apply(lambda x: x * 1.1))  # apply a lambda function to each value in the 'Salary' column

    # perform statistical operations on a specific column of the DataFrame
    print(df['Age'].describe())  # get summary statistics for the 'Age' column
    print(df['Age'].mean())  # get the mean of the 'Age' column
    print(df['Age'].std())  # get the standard deviation of the 'Age' column
    print(df['Age'].max())  # get the maximum value in the 'Age' column
    print(df['Age'].min())  # get the minimum value in the 'Age' column
    print(df['Age'].sum())  # get the sum of all values in the 'Age' column
    print(df['Age'].count())  # get the count of non-null values in the 'Age' column
    print(df['Age'].quantile(0.5))  # get the median of the 'Age' column
    print(df['Age'].mode())  # get the mode of the 'Age' column
    print(df['Age'].median())  # get the median of the 'Age' column
    print(df['Age'].var())  # get the variance of the 'Age' column
    print(df['Age'].std())  # get the standard deviation of the 'Age' column
    print(df['Age'].skew())  # get the skewness of the 'Age' column
    print(df['Age'].kurt())  # get the kurtosis of the 'Age' column

    # perform aggregation operations on the DataFrame
    print(df.groupby('Department')['Salary'].sum())  # get the sum of salaries for each department
    print(df.groupby('Department')['Salary'].mean())  # get the mean salary for each department

    # apply numpy functions to pandas columns
    print(df['Age'].apply(np.sqrt))  # apply the square root function to each value in the 'Age' column
    print(df['Salary'].apply(np.log))  # apply the natural logarithm function to each value in the 'Salary' column

    # transpose the DataFrame
    print(df.T)  # transpose the DataFrame

def dataframe_merge(df: pd.DataFrame):

    data2 = {
        'ID': [4, 5, 6, 7, 8],
        'Name': ['Frank', 'Grace', 'Henry', 'Ivy', 'Jack'],
        'Age': [28, 32, 30, 26, 29],
        'City': ['Chicago', 'San Francisco', 'Los Angeles', 'New York', 'Houston'],
        'Salary': [4500, 5000, 5000, 5000, 10000],
        'Department': ['Sales', 'Marketing', 'Finance', 'Finance', 'Sales'],
        'Gender': ['Female', 'Male', 'Male', 'Female', 'Female'],
        'Marital Status': ['Single', 'Married', 'Married', 'Single', 'Single'],
    }

    df2 = pd.DataFrame(data2)
    
    # merge the DataFrames based on the 'ID' column using an inner join (default)
    df_i = pd.merge(df, df2, on='ID')
    print(df_i)

    # merge the DataFrames based on the 'ID' column using an outer join
    df_o = pd.merge(df, df2, on='ID', how='outer')
    print(df_o)

    # merge the DataFrames based on the 'ID' column using an left join
    df_l = pd.merge(df, df2, on='ID', how='left')
    print(df_l)

    # merge the DataFrames based on the 'ID' column using an right join
    df_r = pd.merge(df, df2, on='ID', how='right')
    print(df_r)


def main():

    # series_pd()

    data = {
        'ID': [1, 2, 3, 4, 5],
        'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
        'Age': [25, 30, 28, 32, 27],
        'City': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'San Francisco'],
        'Salary': [50000, 60000, 45000, 55000, 48000],
        'Department': ['Sales', 'Marketing', 'Finance', 'Finance', 'Sales'],
        'Gender': ['Female', 'Male', 'Male', 'Female', 'Female'],
        'Marital Status': ['Single', 'Married', 'Married', 'Single', 'Single'],
    }

    df = pd.DataFrame(data)

    dataframe_pd(df)
    dataframe_stats(df)
    dataframe_merge(df)

if __name__ == '__main__':
    main()