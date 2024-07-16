import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler

def read_file(filename: str) -> pd.DataFrame:
    """
    Read a CSV file into a pandas DataFrame.

    Parameters:
    filename (str): The name of the CSV file.

    Returns:
    pandas.DataFrame: The loaded dataset.
    """

    return pd.read_csv(filename)

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the wine quality dataset by checking for missing values, understanding the distribution of numerical variables,
    and performing initial feature analysis.

    Parameters:
    df (pd.DataFrame): The input dataset containing wine quality data.

    Returns:
    pd.DataFrame: The preprocessed dataset.
    """

    # Check for the NA values in the dataset - none found.
    print(f"Number of NA values column wise: \n{df.isnull().sum()}")
    print(f"Percentage of NA values column wise: \n{df.isnull().sum() / df.shape[0] * 100}")

    # # Understand the distribution of numerical variables
    # print(df.describe())
    # print(df.describe(include=['object']))

    # # check for each volatility feature
    # print(df["fixed acidity"].unique()) # useful to convert into 0-1 range as others (minmax scalar)
    # print(df["volatile acidity"].unique())

    # # check for the components of the wine
    # print(df["citric acid"].unique())
    # print(df["residual sugar"].unique()) # useful to convert into 0-1 range as others (minmax scalar)
    # print(df["chlorides"].unique())
    # print(df["free sulfur dioxide"].unique()) # useful to convert into 0-1 range as others (/10)
    # print(df["total sulfur dioxide"].unique()) # useful to convert into 0-1 range as others (/100)
    # print(df["sulphates"].unique())
    # print(df["alcohol"].unique()) # useful to convert into 0-1 range as others (/10)

    # # check for the properties of the wine
    # print(df["density"].unique())
    # print(df["pH"].unique()) # useful to convert into 0-1 range as others (minmax scalar)

    df.to_excel("machine-learning/data/Wine Quality/processed_data/cleared_data.xlsx", index = False)

    return df 

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform feature engineering on the wine quality dataset. This includes normalizing certain features, scaling others,
    and checking for correlation between features and the target variable.

    Parameters:
    df (pd.DataFrame): The input dataset containing wine quality data.

    Returns:
    pd.DataFrame: The processed dataset after feature engineering.
    """

    # shows that `alcohol`, `citric acid`, `sulphates` have high correlation with the target variable `quality`.
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
    plt.show()

    # Normalize the fixed acidity feature using MinMaxScaler.
    scaler = MinMaxScaler()

    scaler.fit(df[["fixed acidity", "residual sugar", "pH"]])
    df[["fixed acidity", "residual sugar", "pH"]] = scaler.transform(df[["fixed acidity", "residual sugar", "pH"]])

    # Scale the quality feature to 0-1 range.
    scaler = MaxAbsScaler()

    scaler.fit(df[["quality"]])
    df[["quality"]] = scaler.transform(df[["quality"]])

    # Scale down other features to 0-1 range.
    df[["free sulfur dioxide", "alcohol"]] = df[["free sulfur dioxide", "alcohol"]] / 10
    df["total sulfur dioxide"] = df["total sulfur dioxide"] / 100

    # check for correlation after scaling
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
    plt.show()

    # Save the final DataFrame to an Excel file.
    df.to_excel("machine-learning/data/Wine Quality/processed_data/final_data.xlsx", index=False)

    return df

def only_coorelated_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function filters out the features from the input DataFrame that have a correlation coefficient with the target variable 
    'quality' higher than 0.3. It then creates a new DataFrame that includes only these selected features.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the wine quality data. It should have a column named 'quality' and other features.

    Returns:
    pd.DataFrame: A new DataFrame that includes only the selected features. It also displays a heatmap of the correlation matrix for visualization.
    """
    
    # Filter out the features that have a correlation coefficient with the target variable higher than 0.3.
    features_to_keep = df.corr().loc[df.corr()['quality'] > 0.2].index.tolist()

    # Filter the DataFrame to include only the selected features.
    df = df[features_to_keep]

    # Display a heatmap of the correlation matrix for visualization
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
    plt.show()

    return df

def train_model(df: pd.DataFrame) -> None:
    """
    This function trains a linear regression model on the given dataset and evaluates its performance using mean squared error (MSE)
    and coefficient of determination (R^2). It also calculates and prints the weights and bias of the trained model.

    Parameters:
    df (pd.DataFrame): A pandas DataFrame containing the dataset. The DataFrame should have a column named 'quality' as the target variable,
                       and other columns as features.

    Returns:
    None: The function prints the mean squared error, coefficient of determination, weights, sum of weights, and bias of the trained model.
    """

    # Separate the features (X) and the target variable (y)
    X = df.drop('quality', axis=1)
    y = df['quality']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Create a linear regression model
    model = LinearRegression(n_jobs=-1)

    # Train the model using the training sets
    model.fit(X_train, y_train)

    # Make predictions using the testing set
    y_pred = model.predict(X_test)

    # Calculate the mean squared error
    mse = np.mean((y_pred - y_test) ** 2)
    print("Mean squared error:", mse)

    # Calculate the coefficient of determination (R^2)
    r_squared = model.score(X_test, y_test)
    print("Coefficient of determination (R^2):", r_squared)

    # calculate the `weights` and `bias` parameters for the model
    weights = model.coef_
    bias = model.intercept_

    # print("Weights:", weights)
    # print("Sum of weights:", np.sum(weights))
    # print("Bias:", bias)

    return r_squared

def main():
    """
    The main function orchestrates the entire wine quality prediction process. It reads the dataset, preprocesses it,
    performs feature engineering, trains multiple models, and calculates the average accuracy.

    Parameters:
    None

    Returns:
    None

    Note: Weak results. Not a good model to use for predicting wine quality.
    """
    
    # Read the dataset
    file = "machine-learning/data/Wine Quality/winequality-red.csv"

    data = read_file(file)

    # Preprocess the dataset by checking for missing values, understanding the distribution of numerical variables,
    # and performing initial feature analysis.
    cleared_data = preprocess_data(data)

    # Perform feature engineering on the dataset, including normalizing certain features, scaling others,
    # and checking for correlation between features and the target variable.
    final_data = feature_engineering(cleared_data)

    # Filter out the features from the dataset that have a correlation coefficient with the target variable 
    # 'quality' higher than 0.3.
    coor_data = only_coorelated_features(final_data)

    # Train multiple models and calculate the average accuracy
    avg_accuracy = 0

    for i in range(100):
        print("-----------------------------------------")
        print(f"Run : {i + 1}")
        print("-----------------------------------------")
        # Train a linear regression model on the entire dataset
        acc = train_model(final_data)
        # Train a linear regression model on the dataset with only the selected correlated features
        # acc = train_model(coor_data)
        print("-----------------------------------------\n")
        avg_accuracy += acc
        time.sleep(1)
        
    print(f"Average Accuracy: {avg_accuracy / 100}")

if __name__ == "__main__":
    main()