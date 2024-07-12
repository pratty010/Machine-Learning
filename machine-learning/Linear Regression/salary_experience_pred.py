import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def load_data(filename: str) -> pd.DataFrame:
    """
    Load a dataset from a CSV file into a pandas DataFrame.

    Parameters:
    filename (str): The name of the CSV file.

    Returns:
    pandas.DataFrame: The loaded dataset.

    Example:
    >>> df = load_data("data.csv")
    """
    # Read the CSV file into a pandas DataFrame
    return pd.read_csv(filename)

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess a dataset by removing an unnecessary column.

    Parameters:
    df (pandas.DataFrame): The input dataset containing a column named 'Unnamed: 0'.

    Returns:
    pandas.DataFrame: The preprocessed dataset with the 'Unnamed: 0' column removed.

    Example:
    >>> df = pd.DataFrame({'Salary': [50000, 60000, 70000], 'YearsExperience': [2, 3, 4], 'Unnamed: 0': [1, 2, 3]})
    >>> preprocess_data(df)
       Salary  YearsExperience
    0   50000               2
    1   60000               3
    2   70000               4
    """
    # print(df)
    # print(df.columns)

    df.drop('Unnamed: 0', inplace=True, axis=1)

    return df

def train_model(df: pd.DataFrame) -> None:
    """
    Train a linear regression model on a dataset and evaluate its performance.

    Parameters:
    df (pandas.DataFrame): The dataset containing features and target variable.
        The target variable should be named 'Salary', and the features should be
        all other columns in the DataFrame.

    Returns:
    None: The function prints the mean squared error and coefficient of determination (R^2)
        of the trained model. It also displays a scatter plot of the actual and predicted salaries.
    """
    # Separate the features (X) and the target variable (y)
    X = df.drop('Salary', axis=1)
    y = df['Salary']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Create a linear regression model
    model = LinearRegression()

    # Train the model using the training sets
    model.fit(X_train, y_train)

    # Make predictions using the testing set
    y_pred = model.predict(X_test)

    # Calculate the mean squared error - the optimum cost function
    mse = np.mean((y_pred - y_test) ** 2)
    print("Mean squared error:", mse)

    # Calculate the coefficient of determination (R^2) - predict accuracy
    r_squared = model.score(X_test, y_test)
    print("Coefficient of determination (R^2):", r_squared)

    # print `weight` and `bias` parameters for the model
    print("Weight/Coefficients: ", model.coef_)
    print("Intercept: ", model.intercept_)

    y_fit = model.intercept_ + model.coef_ * df.YearsExperience

    # Display a scatter plot of the actual and predicted salaries(best fit line)
    plt.scatter(df.YearsExperience, df.Salary, color='blue' )
    plt.plot(X_test['YearsExperience'], y_pred, 'r')
    # plt.plot(df.YearsExperience, y_fit, 'r')
    plt.tight_layout()
    plt.show()

def main():
    """
    Main function to execute the entire movie dataset analysis pipeline.
    Example of Basic Linear Regression models.
    """

    # Load the dataset
    dataset = 'machine-learning/data/Salary_Experience_LR/Salary_dataset.csv'
    df = load_data(dataset)

    # Preprocess the data. Cleaning the dataset.
    final_df = preprocess_data(df)

    # Train the model and display results.
    train_model(final_df)

if __name__ == "__main__":
    main()