"""Data loading and preprocessing module."""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple
from sklearn.model_selection import train_test_split

# Import SMOTE if available
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False


class DataLoader:
    """
    Handles data loading and preprocessing for the pipeline.

    Responsibilities:
    - Load datasets from CSV files
    - Handle missing values
    - Split into train/test sets
    - Apply SMOTE for class balancing
    - Save preprocessed data
    """

    def __init__(self, config: Dict[str, Any], output_dir: Path, logger: logging.Logger):
        """
        Initialize data loader.

        Args:
            config: Pipeline configuration
            output_dir: Output directory for results
            logger: Logger instance
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.logger = logger

        # Extract preprocessing config
        self.preproc_config = config.get('data', {}).get('preprocessing', {})
        self.random_seed = config.get('project', {}).get('random_seed', 42)

        # Storage for loaded data
        self.datasets = {}
        self.preprocessed_data = {}

    def load_all_datasets(self) -> Dict[str, Dict[str, Any]]:
        """
        Load and preprocess all datasets from config.

        Returns:
            Dictionary mapping dataset names to their preprocessed data:
            {
                'dataset_name': {
                    'raw': DataFrame,
                    'X_train': DataFrame,
                    'X_test': DataFrame,
                    'y_train': Series,
                    'y_test': Series,
                    'target_column': str,
                    'metadata': dict
                }
            }
        """
        datasets_config = self.config.get('data', {}).get('datasets', [])

        if not datasets_config:
            raise ValueError("No datasets found in configuration")

        self.logger.info(f"Loading {len(datasets_config)} dataset(s)...")

        for dataset_config in datasets_config:
            dataset_name = self._load_dataset(dataset_config)
            self.logger.info(f"  [OK] Loaded: {dataset_name}")

        self.logger.info(f"Successfully loaded {len(self.datasets)} dataset(s)")

        return self.preprocessed_data

    def _load_dataset(self, dataset_config: Dict[str, Any]) -> str:
        """
        Load and preprocess a single dataset.

        Args:
            dataset_config: Dataset configuration from config file

        Returns:
            Dataset name
        """
        # Extract config
        file_path = Path(dataset_config['path'])
        dataset_name = dataset_config.get('name', file_path.stem)
        target_column = dataset_config['target_column']

        self.logger.info(f"Loading dataset: {dataset_name}")
        self.logger.info(f"  File: {file_path}")
        self.logger.info(f"  Target column: {target_column}")

        # Load CSV
        try:
            df = pd.read_csv(file_path)
            self.logger.info(f"  Shape: {df.shape}")
        except Exception as e:
            raise RuntimeError(f"Failed to load {file_path}: {str(e)}")

        # Validate target column exists
        if target_column not in df.columns:
            raise ValueError(
                f"Target column '{target_column}' not found in {dataset_name}. "
                f"Available columns: {df.columns.tolist()}"
            )

        # Store raw data
        self.datasets[dataset_name] = df.copy()

        # Preprocess
        preprocessed = self._preprocess_dataset(df, target_column, dataset_name)
        self.preprocessed_data[dataset_name] = preprocessed

        # Save preprocessed data
        self._save_preprocessed_data(dataset_name, preprocessed)

        return dataset_name

    def _preprocess_dataset(
        self,
        df: pd.DataFrame,
        target_column: str,
        dataset_name: str
    ) -> Dict[str, Any]:
        """
        Preprocess a dataset (handle missing, encode categoricals, split, SMOTE).

        Args:
            df: Raw dataframe
            target_column: Name of target column
            dataset_name: Dataset name for logging

        Returns:
            Dictionary with preprocessed data
        """
        # Handle missing values
        if self.preproc_config.get('handle_missing', True):
            df = self._handle_missing_values(df, dataset_name)

        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Encode categorical variables
        X = self._encode_categorical_features(X, dataset_name)

        # Get class distribution before split
        class_dist_before = y.value_counts().to_dict()
        self.logger.info(f"  Class distribution: {class_dist_before}")

        # Train/test split
        test_size = self.preproc_config.get('test_size', 0.3)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=self.random_seed,
            stratify=y if len(y.unique()) > 1 else None
        )

        self.logger.info(f"  Train/test split: {len(X_train)}/{len(X_test)} (test_size={test_size})")

        # Apply SMOTE if enabled
        if self.preproc_config.get('smote', True):
            X_train, y_train = self._apply_smote(X_train, y_train, dataset_name)

        # Create metadata
        metadata = {
            'original_shape': df.shape,
            'train_shape': X_train.shape,
            'test_shape': X_test.shape,
            'target_column': target_column,
            'class_distribution_before': class_dist_before,
            'class_distribution_after': y_train.value_counts().to_dict(),
            'test_size': test_size,
            'smote_applied': self.preproc_config.get('smote', True),
            'random_seed': self.random_seed
        }

        return {
            'raw': df,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'target_column': target_column,
            'metadata': metadata
        }

    def _encode_categorical_features(self, X: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """
        Encode categorical features using LabelEncoder.

        Args:
            X: Feature dataframe
            dataset_name: Dataset name for logging

        Returns:
            DataFrame with encoded categorical features
        """
        from sklearn.preprocessing import LabelEncoder

        # Identify categorical columns (object dtype or categorical dtype)
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

        if not categorical_cols:
            self.logger.debug(f"  No categorical features to encode")
            return X

        self.logger.info(f"  Encoding {len(categorical_cols)} categorical feature(s): {categorical_cols}")

        X_encoded = X.copy()

        for col in categorical_cols:
            try:
                le = LabelEncoder()
                X_encoded[col] = le.fit_transform(X[col].astype(str))
                self.logger.debug(f"    Encoded '{col}': {len(le.classes_)} unique values")
            except Exception as e:
                self.logger.warning(f"    Failed to encode '{col}': {str(e)}")
                # Keep original column if encoding fails
                continue

        return X_encoded

    def _handle_missing_values(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """Handle missing values in dataset."""
        missing_count = df.isnull().sum().sum()

        if missing_count == 0:
            self.logger.info(f"  No missing values found")
            return df

        self.logger.info(f"  Handling {missing_count} missing values...")

        # Simple strategy: drop rows with any missing values
        # Can be made more sophisticated later
        df_clean = df.dropna()
        dropped_rows = len(df) - len(df_clean)

        if dropped_rows > 0:
            self.logger.info(f"  Dropped {dropped_rows} rows with missing values")

        return df_clean

    def _apply_smote(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        dataset_name: str
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Apply SMOTE for class balancing."""
        if not SMOTE_AVAILABLE:
            self.logger.warning("  SMOTE not available - install imbalanced-learn")
            return X_train, y_train

        # Check if balancing is needed
        class_counts = y_train.value_counts()
        if len(class_counts) < 2:
            self.logger.warning("  SMOTE skipped: only one class present")
            return X_train, y_train

        minority_class_count = class_counts.min()
        k_neighbors = self.preproc_config.get('smote_k_neighbors', 5)

        # Adjust k_neighbors if minority class is too small
        if minority_class_count <= k_neighbors:
            k_neighbors = max(1, minority_class_count - 1)
            self.logger.warning(
                f"  Adjusted SMOTE k_neighbors to {k_neighbors} "
                f"(minority class has {minority_class_count} samples)"
            )

        try:
            smote = SMOTE(
                random_state=self.random_seed,
                k_neighbors=k_neighbors
            )
            X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

            self.logger.info(
                f"  SMOTE applied: {len(X_train)} -> {len(X_resampled)} samples"
            )

            # Convert back to DataFrame/Series
            X_resampled = pd.DataFrame(X_resampled, columns=X_train.columns)
            y_resampled = pd.Series(y_resampled, name=y_train.name)

            return X_resampled, y_resampled

        except Exception as e:
            self.logger.error(f"  SMOTE failed: {str(e)}")
            self.logger.warning("  Continuing without SMOTE...")
            return X_train, y_train

    def _save_preprocessed_data(self, dataset_name: str, preprocessed: Dict[str, Any]):
        """Save preprocessed data to disk."""
        # Create dataset-specific directory
        dataset_dir = self.output_dir / 'data' / 'preprocessed' / dataset_name.replace(' ', '_')
        dataset_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Save train/test splits
            train_df = pd.concat([preprocessed['X_train'], preprocessed['y_train']], axis=1)
            test_df = pd.concat([preprocessed['X_test'], preprocessed['y_test']], axis=1)

            train_path = dataset_dir / 'train.csv'
            test_path = dataset_dir / 'test.csv'

            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)

            self.logger.debug(f"  Saved train data: {train_path}")
            self.logger.debug(f"  Saved test data: {test_path}")

            # Save metadata
            import json
            metadata_path = dataset_dir / 'metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(preprocessed['metadata'], f, indent=2)

            self.logger.debug(f"  Saved metadata: {metadata_path}")

        except Exception as e:
            self.logger.error(f"  Failed to save preprocessed data: {str(e)}")
