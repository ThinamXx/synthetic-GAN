"""
Data preprocessing module for synthetic data generation.
Handles missing values, normalization, and categorical encoding.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import pickle
import warnings

warnings.filterwarnings("ignore")


class DataPreprocessor:
    """
    Comprehensive data preprocessor for mixed tabular data.
    Handles missing values, normalization, and categorical encoding.
    """

    def __init__(self, config):
        self.config = config
        self.numerical_columns = []
        self.categorical_columns = []
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
        self.column_info = {}

    def identify_column_types(self, df):
        """Identify numerical and categorical columns"""
        self.numerical_columns = []
        self.categorical_columns = []

        for col in df.columns:
            if df[col].dtype in ["int64", "float64"]:
                self.numerical_columns.append(col)
            else:
                self.categorical_columns.append(col)

        print(
            f"Identified {len(self.numerical_columns)} numerical and {len(self.categorical_columns)} categorical columns"
        )

    def handle_missing_values(self, df, fit=True):
        """Handle missing values with appropriate strategies"""
        df_processed = df.copy()

        if fit:
            for col in self.numerical_columns:
                if df_processed[col].isnull().any():
                    imputer = SimpleImputer(strategy="median")
                    imputer.fit(df_processed[[col]])
                    self.imputers[col] = imputer

            for col in self.categorical_columns:
                if df_processed[col].isnull().any():
                    # Use most frequent for categorical.
                    imputer = SimpleImputer(strategy="most_frequent")
                    imputer.fit(df_processed[[col]])
                    self.imputers[col] = imputer

        # Apply imputation.
        for col in self.numerical_columns:
            if col in self.imputers:
                df_processed[col] = (
                    self.imputers[col].transform(df_processed[[col]]).flatten()
                )

        for col in self.categorical_columns:
            if col in self.imputers:
                df_processed[col] = (
                    self.imputers[col].transform(df_processed[[col]]).flatten()
                )

        return df_processed

    def detect_and_handle_outliers(self, df, fit=True):
        """Detect and cap outliers using IQR method"""
        df_processed = df.copy()

        if fit:
            self.outlier_bounds = {}

        for col in self.numerical_columns:
            if fit:
                Q1 = df_processed[col].quantile(0.25)
                Q3 = df_processed[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                self.outlier_bounds[col] = (lower_bound, upper_bound)
            else:
                lower_bound, upper_bound = self.outlier_bounds[col]

            # Cap outliers.
            df_processed[col] = np.clip(df_processed[col], lower_bound, upper_bound)

        return df_processed

    def encode_categorical_variables(self, df, fit=True):
        """Encode categorical variables"""
        df_processed = df.copy()

        for col in self.categorical_columns:
            if fit:
                # Fit label encoder.
                encoder = LabelEncoder()
                # Handle case where column might have unseen categories.
                unique_values = df_processed[col].dropna().astype(str).unique()
                encoder.fit(unique_values)
                self.encoders[col] = encoder

                self.column_info[col] = {
                    "original_dtype": df[col].dtype,
                    "unique_values": unique_values.tolist(),
                }

            non_null_mask = df_processed[col].notna()
            if non_null_mask.any():
                # Handle unseen categories by mapping to most frequent.
                str_values = df_processed[col].astype(str)
                encoded_values = np.full(len(df_processed), 0)

                for i, val in enumerate(str_values):
                    if pd.notna(df_processed[col].iloc[i]):
                        try:
                            encoded_values[i] = self.encoders[col].transform([val])[0]
                        except ValueError:
                            encoded_values[i] = 0

                df_processed[col] = encoded_values

        return df_processed

    def normalize_numerical_features(self, df, fit=True):
        """Normalize numerical features"""
        df_processed = df.copy()

        for col in self.numerical_columns:
            if fit:
                scaler = StandardScaler()
                scaler.fit(df_processed[[col]])
                self.scalers[col] = scaler

            df_processed[col] = (
                self.scalers[col].transform(df_processed[[col]]).flatten()
            )

        return df_processed

    def fit_transform(self, df):
        """Fit preprocessor and transform data"""
        print("Starting data preprocessing...")

        self.identify_column_types(df)

        # Handle missing values.
        if self.config["handle_missing"]:
            print("Handling missing values...")
            df = self.handle_missing_values(df, fit=True)

        # Handle outliers.
        if self.config["outlier_detection"]:
            print("Detecting and handling outliers...")
            df = self.detect_and_handle_outliers(df, fit=True)

        # Encode categorical variables.
        if self.config["encode_categorical"]:
            print("Encoding categorical variables...")
            df = self.encode_categorical_variables(df, fit=True)

        # Normalize numerical features.
        if self.config["normalize_numerical"]:
            print("Normalizing numerical features...")
            df = self.normalize_numerical_features(df, fit=True)

        print("Preprocessing completed!")
        return df

    def transform(self, df):
        """Transform new data using fitted preprocessor"""
        # Handle missing values.
        if self.config["handle_missing"]:
            df = self.handle_missing_values(df, fit=False)

        # Handle outliers.
        if self.config["outlier_detection"]:
            df = self.detect_and_handle_outliers(df, fit=False)

        # Encode categorical variables.
        if self.config["encode_categorical"]:
            df = self.encode_categorical_variables(df, fit=False)

        # Normalize numerical features.
        if self.config["normalize_numerical"]:
            df = self.normalize_numerical_features(df, fit=False)

        return df

    def inverse_transform(self, df):
        """Inverse transform to original data format"""
        df_processed = df.copy()

        # Denormalize numerical features.
        if self.config["normalize_numerical"]:
            for col in self.numerical_columns:
                if col in self.scalers:
                    df_processed[col] = (
                        self.scalers[col]
                        .inverse_transform(df_processed[[col]])
                        .flatten()
                    )

        # Decode categorical variables.
        if self.config["encode_categorical"]:
            for col in self.categorical_columns:
                if col in self.encoders:
                    # Round to nearest integer for encoded values.
                    encoded_values = np.round(df_processed[col]).astype(int)
                    # Clip to valid range.
                    max_class = len(self.encoders[col].classes_) - 1
                    encoded_values = np.clip(encoded_values, 0, max_class)

                    try:
                        decoded_values = self.encoders[col].inverse_transform(
                            encoded_values
                        )
                        df_processed[col] = decoded_values
                    except ValueError:
                        df_processed[col] = self.encoders[col].classes_[0]

        return df_processed

    def save_preprocessor(self, filepath):
        """Save the fitted preprocessor"""
        preprocessor_data = {
            "numerical_columns": self.numerical_columns,
            "categorical_columns": self.categorical_columns,
            "scalers": self.scalers,
            "encoders": self.encoders,
            "imputers": self.imputers,
            "column_info": self.column_info,
            "config": self.config,
        }

        if hasattr(self, "outlier_bounds"):
            preprocessor_data["outlier_bounds"] = self.outlier_bounds

        with open(filepath, "wb") as f:
            pickle.dump(preprocessor_data, f)

    def load_preprocessor(self, filepath):
        """Load a fitted preprocessor"""
        with open(filepath, "rb") as f:
            preprocessor_data = pickle.load(f)

        self.numerical_columns = preprocessor_data["numerical_columns"]
        self.categorical_columns = preprocessor_data["categorical_columns"]
        self.scalers = preprocessor_data["scalers"]
        self.encoders = preprocessor_data["encoders"]
        self.imputers = preprocessor_data["imputers"]
        self.column_info = preprocessor_data["column_info"]
        self.config = preprocessor_data["config"]

        if "outlier_bounds" in preprocessor_data:
            self.outlier_bounds = preprocessor_data["outlier_bounds"]
