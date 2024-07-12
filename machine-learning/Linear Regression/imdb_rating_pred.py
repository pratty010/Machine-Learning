import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pyreadr
import time

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def load_data(filename: str) -> pd.DataFrame:
    """
    Load a dataset from an RData file and convert it into a pandas DataFrame.

    Parameters:
    filename (str): The path to the RData file containing the dataset.

    Returns:
    pd.DataFrame: A pandas DataFrame containing the loaded dataset.
    """
    data = pyreadr.read_r(filename)
    df = data["movies"]

    # Read the R file into a pandas DataFrame
    return df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess a movie dataset by performing initial exploration, handling missing values, 
    understanding the distribution of numerical variables, setting the index to 'title', 
    and selecting relevant features for the model.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the movie dataset.

    Returns:
    pd.DataFrame: The preprocessed DataFrame ready for further analysis or modeling.
    """

    # Initial exploration of the data
    print(df)
    print(df.columns)
    # print(df.columns.value_counts())

    # Check for the NA values in the dataset
    print(f"Number of NA values column wise: \n{df.isnull().sum()}")
    print(f"Percentage of NA values column wise: \n{df.isnull().sum() / df.shape[0] * 100}")

    # Drop the rows with NA values
    df.dropna(inplace=True)
    print(f"Percentage of NA values column wise: \n{df.isnull().sum() / df.shape[0] * 100}")

    # Understand the distribution of numerical variables
    print(df.describe())
    print(df.describe(include=['object']))
    # print(df.skew())

    # Set index to 'title'
    df.set_index('title', inplace=True)

    # Save the DataFrame to an Excel file for further analysis
    df.to_excel("machine-learning/data/Movie_Data_MLR/data.xlsx")

    # Drop features that may not have the best correlation with the target variable - `imdb_rating`
    print(df["title_type"].unique()) 
    # dummies = pd.get_dummies(df["title_type"])
    # dummies_corr = pd.concat([dummies, df["imdb_rating"]], axis=1)

    print(df["genre"].unique())
    # dummies = pd.get_dummies(df["genre"])
    # dummies_corr = pd.concat([dummies, df["imdb_rating"]], axis=1)

    # Not useful for movie ratings as this is just age criteria. May be more useful for sales or views parameter.
    print(df["mpaa_rating"].unique()) 

    # Drop features with too many unique values with no relation with type of movie made.
    # May have edge cases for some outstanding movies production houses.
    print(df["studio"].unique())

    # sns.heatmap(dummies_corr.corr(), annot=True, cmap="seismic")
    # plt.show()

    # Too many unique values in this feature. Will drop it for now. 
    # There are the `best_actor_win`, `best_actress_win`, `best_director_win`, `best_pic_nom`, `best_pic_win` 
    # that will provide better features to estimate the rating.
    print(len(df["director"].unique())) # Useful for director's reputation
    print(len(df["actor1"].unique())) # Useful for actor's reputation
    print(len(df["actor2"].unique())) # Useful for actor's reputation
    print(len(df["actor3"].unique())) # Useful for actor's reputation
    print(len(df["actor4"].unique())) # Useful for actor's reputation
    print(len(df["actor5"].unique())) # Useful for actor's reputation

    # Drop useless features from the dataset
    features_drop = ["title_type", "genre", "studio", "runtime", "mpaa_rating","director", "actor1", "actor2", "actor3", "actor4", "actor5", "imdb_url", "rt_url", 'thtr_rel_year', 'thtr_rel_month', 'thtr_rel_day', 'dvd_rel_year', 'dvd_rel_month', 'dvd_rel_day']
    df.drop(features_drop, axis=1, inplace=True)

    # Save the cleaned DataFrame to an Excel file
    df.to_excel("machine-learning/data/Movie_Data_MLR/cleaned_data.xlsx")

    return df

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform feature engineering on a movie dataset by scaling down numerical variables, 
    converting categorical variables to numerical, and removing irrelevant features.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the movie dataset.

    Returns:
    pd.DataFrame: The DataFrame after feature engineering, ready for model training.
    """

    ## Scale down the imdb_rating column
    # df["imdb_rating"] = df["imdb_rating"] / 10

    # scale down the imdb_num_votes column
    print(df['imdb_num_votes'].describe())
    df["imdb_num_votes"] = df["imdb_num_votes"] / 10000

    # scale down critic_score and audience_score column
    df["critics_score"] = df["critics_score"] / 100
    df["audience_score"] = df["audience_score"] / 100

    # covert categorical variables to numerical
    critics_rating_mapping = {
        "Rotten" : -1,
        "Fresh" : 0,
        "Certified Fresh" : 1,
    }

    audience_rating_mapping = {
        "Upright" : 1,
        "Spilled" : 0,
    }

    binary_mapping = {
        "no" : 0,
        "yes" : 1,
    }

    df["critics_rating"] = df["critics_rating"].map(critics_rating_mapping)
    df["audience_rating"] = df["audience_rating"].map(audience_rating_mapping)

    for col in ['best_pic_nom', 'best_pic_win', 'best_actor_win', 'best_actress_win', 'best_dir_win', 'top200_box']:
        df[col] = df[col].cat.rename_categories(binary_mapping)

    # save final data to excel file
    df.to_excel("machine-learning/data/Movie_Data_MLR/cleaned_data.xlsx")

    # correlation matrix heatmap
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
    # plt.show()

    # can drop all `best` features as they have very low correlation with the target variable. 
    # This improves the model's performance. Can ignore.
    features_drop = ['best_pic_nom', 'best_pic_win', 'best_actor_win', 'best_actress_win', 'best_dir_win', 'top200_box']
    df.drop(features_drop, axis=1, inplace=True)

    # correlation matrix heatmap
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
    # plt.show()

    return df

def train_model(df: pd.DataFrame) -> None:
    """
    Train a linear regression model on a movie dataset and evaluate its performance.

    Parameters:
    df (pd.DataFrame): A pandas DataFrame containing the movie dataset with features and target variable.

    Returns:
    None: The function prints the mean squared error (MSE) and coefficient of determination (R^2) of the trained model.
    """

    # Separate the features (X) and the target variable (y)
    X = df.drop('imdb_rating', axis=1)
    y = df['imdb_rating']

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
    Main function to execute the entire movie dataset analysis pipeline.
    Example of Multiple Linear Regression models.
    """

    # Load the dataset
    dataset = 'machine-learning/data/Movie_Data_MLR/movies.RData'
    df = load_data(dataset)

    # Preprocess the dataset
    pros_df = preprocess_data(df)

    # Perform feature engineering on the dataset
    final_df = feature_engineering(pros_df)

    # Train multiple models and calculate the average accuracy
    avg_accuracy = 0

    for i in range(100):
        print("-----------------------------------------")
        print(f"Run : {i + 1}")
        print("-----------------------------------------")
        acc = train_model(final_df)
        print("-----------------------------------------\n")
        avg_accuracy += acc
        time.sleep(1)
        
    print(f"Average Accuracy: {avg_accuracy / 100}")
        

if __name__ == '__main__':
    main()