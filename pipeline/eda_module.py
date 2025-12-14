"""Exploratory Data Analysis module."""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any

# Import result manager
from result_manager import ResultManager


class EDAModule:
    """
    Handles Exploratory Data Analysis for the pipeline.

    Responsibilities:
    - Generate dataset profiles (using ydata-profiling if available)
    - Compute summary statistics
    - Analyze target distribution
    - Compute correlations
    - Generate visualizations
    - Feature importance analysis (RF + LR)
    """

    def __init__(
        self,
        config: Dict[str, Any],
        result_manager: ResultManager,
        logger: logging.Logger
    ):
        """
        Initialize EDA module.

        Args:
            config: Pipeline configuration
            result_manager: Result manager instance
            logger: Logger instance
        """
        self.config = config
        self.result_mgr = result_manager
        self.logger = logger

        # Extract EDA config
        self.eda_config = config.get('analysis', {}).get('eda', {})

    def run_eda_for_all_datasets(
        self,
        datasets: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Run EDA for all datasets.

        Args:
            datasets: Dictionary of preprocessed datasets

        Returns:
            Dictionary with EDA results for each dataset
        """
        self.logger.info(f"Running EDA for {len(datasets)} dataset(s)...")

        results = {}

        for dataset_name, dataset_data in datasets.items():
            self.logger.info(f"  Analyzing: {dataset_name}")

            try:
                eda_result = self._run_eda_for_dataset(dataset_name, dataset_data)
                results[dataset_name] = eda_result
                self.logger.info(f"  [OK] EDA completed for {dataset_name}")

            except Exception as e:
                self.logger.error(
                    f"  [ERROR] EDA failed for {dataset_name}: {str(e)}",
                    exc_info=True
                )
                results[dataset_name] = {'status': 'failed', 'error': str(e)}

        return results

    def _run_eda_for_dataset(
        self,
        dataset_name: str,
        dataset_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run EDA for a single dataset.

        Args:
            dataset_name: Name of the dataset
            dataset_data: Preprocessed dataset data

        Returns:
            Dictionary with EDA results
        """
        # Get the raw dataframe
        df = dataset_data['raw']
        target_column = dataset_data['target_column']

        results = {'status': 'completed', 'artifacts': []}

        # 1. Summary statistics
        if self.eda_config.get('generate_profile', True):
            self._generate_summary_stats(df, dataset_name, target_column, results)

        # 2. Target distribution
        self._analyze_target_distribution(df, dataset_name, target_column, results)

        # 3. Feature importance (if enabled)
        if self.eda_config.get('feature_importance', True):
            self._compute_feature_importance(dataset_data, dataset_name, results)

        return results

    def _generate_summary_stats(
        self,
        df: pd.DataFrame,
        dataset_name: str,
        target_column: str,
        results: Dict[str, Any]
    ):
        """Generate and save summary statistics."""
        try:
            # Basic describe
            describe_df = df.describe(include='all').T
            describe_df['dtype'] = df.dtypes

            # Save using result manager
            safe_name = self.result_mgr.sanitize_name(dataset_name)
            self.result_mgr.save_eda_summary(
                dataset_name=dataset_name,
                summary_stats=describe_df
            )

            self.logger.debug(f"    Saved summary statistics")
            results['artifacts'].append('summary_stats')

        except Exception as e:
            self.logger.warning(f"    Failed to generate summary stats: {str(e)}")

    def _analyze_target_distribution(
        self,
        df: pd.DataFrame,
        dataset_name: str,
        target_column: str,
        results: Dict[str, Any]
    ):
        """Analyze and save target distribution."""
        try:
            target_dist = df[target_column].value_counts().to_frame('count')
            target_dist['percentage'] = (target_dist['count'] / len(df) * 100).round(2)

            # Save to results/eda/
            safe_name = self.result_mgr.sanitize_name(dataset_name)
            output_path = self.result_mgr.eda_dir / f"{safe_name}_target_distribution.csv"
            target_dist.to_csv(output_path)

            self.logger.debug(f"    Saved target distribution")
            results['artifacts'].append('target_distribution')

        except Exception as e:
            self.logger.warning(f"    Failed to analyze target distribution: {str(e)}")

    def _compute_feature_importance(
        self,
        dataset_data: Dict[str, Any],
        dataset_name: str,
        results: Dict[str, Any]
    ):
        """Compute feature importance using RF and LR."""
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.linear_model import LogisticRegression
            import warnings
            warnings.filterwarnings('ignore')

            X_train = dataset_data['X_train']
            y_train = dataset_data['y_train']

            self.logger.debug(f"    Computing feature importance...")

            # Random Forest importance
            rf = RandomForestClassifier(
                n_estimators=100,
                random_state=self.config['project']['random_seed'],
                n_jobs=-1
            )
            rf.fit(X_train, y_train)

            rf_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': rf.feature_importances_
            }).sort_values('importance', ascending=False)

            # Logistic Regression coefficients
            lr = LogisticRegression(
                max_iter=1000,
                random_state=self.config['project']['random_seed']
            )
            lr.fit(X_train, y_train)

            lr_importance = pd.DataFrame({
                'feature': X_train.columns,
                'coefficient': np.abs(lr.coef_[0])
            }).sort_values('coefficient', ascending=False)

            # Merge and save
            merged = rf_importance.merge(
                lr_importance,
                on='feature',
                how='outer'
            )

            safe_name = self.result_mgr.sanitize_name(dataset_name)
            output_path = self.result_mgr.eda_dir / f"{safe_name}_feature_importance_merged.csv"
            merged.to_csv(output_path, index=False)

            # Also save individual files
            rf_path = self.result_mgr.eda_dir / f"{safe_name}_feature_importance_rf.csv"
            lr_path = self.result_mgr.eda_dir / f"{safe_name}_feature_importance_lr.csv"

            rf_importance.to_csv(rf_path, index=False)
            lr_importance.to_csv(lr_path, index=False)

            self.logger.debug(f"    Saved feature importance analysis")
            results['artifacts'].append('feature_importance')

        except Exception as e:
            self.logger.warning(f"    Failed to compute feature importance: {str(e)}")
