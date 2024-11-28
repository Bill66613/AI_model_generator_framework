import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

class DatasetHandler:
    def __init__(self, file_path):
        """
        Initialize the DatasetHandler with the given dataset file path.
        Args:
            file_path (str): Path to the dataset file.
        """
        self.file_path = file_path
        self.data = None
        self.X = None
        self.y = None
        self.columns = None
        self.task_type = None  # "classification" or "regression"
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None

    def load_data(self):
        """Load the dataset from the file path."""
        try:
            self.data = pd.read_csv(self.file_path)
            print(f"Dataset loaded with shape: {self.data.shape}")
        except Exception as e:
            print(f"Error loading dataset: {e}")

    def detect_task_type(self, target_column):
        """Detect whether the task is classification or regression based on the target column."""
        unique_values = self.data[target_column].nunique()
        if unique_values < 20:  # Threshold to classify as categorical
            self.task_type = "classification"
        else:
            self.task_type = "regression"
        print(f"Detected task type: {self.task_type}")

    def preprocess(self, target_column):
        """
        Preprocess the dataset based on task type.
        Args:
            target_column (str): The name of the target column.
        """
        self.detect_task_type(target_column)

        # Split features and target
        self.y = self.data[target_column]
        self.X = self.data.drop(columns=[target_column])

        if self.task_type == "classification":
            # Encode target labels for classification
            self.y = LabelEncoder().fit_transform(self.y)
        elif self.task_type == "regression":
            # Ensure the target column is numerical for regression
            self.y = pd.to_numeric(self.y)

        # Scale features
        scaler = StandardScaler()
        self.X = scaler.fit_transform(self.X)

        # Split into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        print("Preprocessing complete.")

    def get_splits(self):
        """Retrieve the training and testing splits."""
        return self.X_train, self.X_test, self.y_train, self.y_test

    def apply_sliding_window(self, window_size, step_size):
        """Apply sliding window to sequential data."""
        if self.data is None:
            raise ValueError("No data loaded. Please load a dataset first.")

        data = self.data.values
        num_samples = (len(data) - window_size) // step_size + 1
        windows = [
            data[i * step_size:i * step_size + window_size]
            for i in range(num_samples)
        ]
        return np.array(windows)

    def normalize_data(self):
        """Normalize dataset features."""
        if self.data is None:
            raise ValueError("No data loaded. Please load a dataset first.")

        scaler = StandardScaler()
        self.data.iloc[:, :-1] = scaler.fit_transform(self.data.iloc[:, :-1])
        self.X = self.data.iloc[:, :-1].values
        self.y = self.data.iloc[:, -1].values
