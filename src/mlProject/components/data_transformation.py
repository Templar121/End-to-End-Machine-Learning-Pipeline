import os
from mlProject import logger
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from mlProject.entity.config_entity import DataTransformationConfig

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    
    ## Note: You can add different data transformation techniques such as Scaler, PCA and all
    #You can perform all kinds of EDA in ML cycle here before passing this data to the model

    # I am only adding train_test_spliting cz this data is already cleaned up


    def train_test_spliting(self):
        data = pd.read_csv(self.config.data_path)

        # Split the data into training and test sets. (0.75, 0.25) split.
        train, test = train_test_split(data)

        train.to_csv(os.path.join(self.config.root_dir, "train.csv"),index = False)
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"),index = False)

        logger.info("Splited data into training and test sets")
        logger.info(train.shape)
        logger.info(test.shape)

        print(train.shape)
        print(test.shape)
        
        
    def feature_transformation(self):
        # Load train and test data
        train = pd.read_csv(os.path.join(self.config.root_dir, "train.csv"))
        test = pd.read_csv(os.path.join(self.config.root_dir, "test.csv"))

        # Dynamically infer feature columns (excluding the target column 'quality')
        target_column = 'quality'
        feature_columns = [col for col in train.columns if col != target_column]

        X_train = train[feature_columns]
        X_test = test[feature_columns]
        y_train = train[target_column]
        y_test = test[target_column]

        # Define the pipeline
        feature_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('poly', PolynomialFeatures(degree=2, include_bias=False))
        ])

        # Transform features
        X_train_transformed = feature_pipeline.fit_transform(X_train)
        X_test_transformed = feature_pipeline.transform(X_test)

        # Convert to DataFrames
        train_transformed_df = pd.DataFrame(X_train_transformed)
        train_transformed_df[target_column] = y_train.values

        test_transformed_df = pd.DataFrame(X_test_transformed)
        test_transformed_df[target_column] = y_test.values

        # Save transformed data
        train_path = os.path.join(self.config.root_dir, "train_transformed.csv")
        test_path = os.path.join(self.config.root_dir, "test_transformed.csv")
        train_transformed_df.to_csv(train_path, index=False)
        test_transformed_df.to_csv(test_path, index=False)

        logger.info("Saved transformed train and test datasets to CSV")
        print("X_train_transformed shape:", X_train_transformed.shape)
        print("X_test_transformed shape:", X_test_transformed.shape)
