"""Model training and evaluation module."""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
import pickle
import warnings
warnings.filterwarnings('ignore')

# Import models and metrics
from models import MODELS
from metrics import compute_metrics
from result_manager import ResultManager


class ModelTrainer:
    """
    Handles model training and evaluation for the pipeline.

    Responsibilities:
    - Load model groups from configuration
    - Train models on each dataset
    - Evaluate using multiple metrics
    - Save trained models and results
    - Generate comparison tables
    """

    def __init__(
        self,
        config: Dict[str, Any],
        result_manager: ResultManager,
        logger: logging.Logger
    ):
        """
        Initialize model trainer.

        Args:
            config: Pipeline configuration
            result_manager: Result manager instance
            logger: Logger instance
        """
        self.config = config
        self.result_mgr = result_manager
        self.logger = logger

        # Extract model config
        self.model_config = config.get('models', {})
        self.selected_groups = self.model_config.get('groups', [])
        self.random_seed = config.get('project', {}).get('random_seed', 42)

        # Storage for trained models and results
        self.trained_models = {}
        self.evaluation_results = {}

    def train_all_models(
        self,
        datasets: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Train all selected models on all datasets.

        Args:
            datasets: Dictionary of preprocessed datasets

        Returns:
            Dictionary with training results:
            {
                'dataset_name': {
                    'group_name': {
                        'model_name': {
                            'metrics': {...},
                            'model_path': 'path/to/model.pkl'
                        }
                    }
                }
            }
        """
        self.logger.info(f"Training models for {len(datasets)} dataset(s)...")
        self.logger.info(f"Selected model groups: {self.selected_groups}")

        results = {}

        for dataset_name, dataset_data in datasets.items():
            self.logger.info(f"  Training models for: {dataset_name}")

            try:
                dataset_results = self._train_models_for_dataset(
                    dataset_name,
                    dataset_data
                )
                results[dataset_name] = dataset_results

                # Count successful models
                total_models = sum(len(models) for models in dataset_results.values())
                self.logger.info(f"  [OK] Trained {total_models} model(s) for {dataset_name}")

            except Exception as e:
                self.logger.error(
                    f"  [ERROR] Model training failed for {dataset_name}: {str(e)}",
                    exc_info=True
                )
                results[dataset_name] = {'status': 'failed', 'error': str(e)}

        # Save summary
        self._save_training_summary(results)

        self.evaluation_results = results
        return results

    def _train_models_for_dataset(
        self,
        dataset_name: str,
        dataset_data: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Train all selected models for a single dataset.

        Args:
            dataset_name: Name of the dataset
            dataset_data: Preprocessed dataset data

        Returns:
            Dictionary with results for each model group
        """
        X_train = dataset_data['X_train']
        X_test = dataset_data['X_test']
        y_train = dataset_data['y_train']
        y_test = dataset_data['y_test']

        results = {}

        for group_name in self.selected_groups:
            self.logger.info(f"    Training group: {group_name}")

            # Get models for this group
            if group_name not in MODELS:
                self.logger.warning(f"      Group '{group_name}' not found in MODELS")
                continue

            group_models = MODELS[group_name]
            group_results = {}

            for model_name, model_builder in group_models.items():
                try:
                    # Train and evaluate model
                    model_result = self._train_single_model(
                        dataset_name=dataset_name,
                        group_name=group_name,
                        model_name=model_name,
                        model_builder=model_builder,
                        X_train=X_train,
                        X_test=X_test,
                        y_train=y_train,
                        y_test=y_test
                    )

                    group_results[model_name] = model_result

                    # Log key metric
                    auc = model_result['metrics'].get('AUC', 0)
                    self.logger.info(f"      [OK] {model_name}: AUC={auc:.4f}")

                except Exception as e:
                    self.logger.error(f"      [ERROR] {model_name}: {str(e)}")
                    group_results[model_name] = {
                        'status': 'failed',
                        'error': str(e)
                    }

            results[group_name] = group_results

        return results

    def _train_single_model(
        self,
        dataset_name: str,
        group_name: str,
        model_name: str,
        model_builder: callable,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series
    ) -> Dict[str, Any]:
        """
        Train and evaluate a single model.

        Args:
            dataset_name: Dataset name
            group_name: Model group name
            model_name: Model name
            model_builder: Function that builds the model pipeline
            X_train, X_test, y_train, y_test: Train/test splits

        Returns:
            Dictionary with model metrics and saved path
        """
        # Build model pipeline
        pipeline = model_builder()

        # Train model
        pipeline.fit(X_train, y_train)

        # Get predictions
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

        # Compute metrics
        metrics = compute_metrics(y_train, y_test, y_pred_proba)

        # Try to save model (may fail due to pickle issues with lambda functions)
        model_path = None
        try:
            model_path = self._save_model(
                pipeline=pipeline,
                dataset_name=dataset_name,
                group_name=group_name,
                model_name=model_name
            )
        except Exception as e:
            # Log warning but don't fail - models will be kept in memory
            self.logger.debug(f"        Could not save model to disk: {str(e)}")

        # Store model in memory for later use (SHAP, etc.)
        if dataset_name not in self.trained_models:
            self.trained_models[dataset_name] = {}
        if group_name not in self.trained_models[dataset_name]:
            self.trained_models[dataset_name][group_name] = {}
        self.trained_models[dataset_name][group_name][model_name] = pipeline

        return {
            'metrics': metrics,
            'model_path': str(model_path) if model_path else None,
            'model_object': pipeline,  # Keep reference for later stages
            'status': 'completed'
        }

    def _save_model(
        self,
        pipeline: Any,
        dataset_name: str,
        group_name: str,
        model_name: str
    ) -> Path:
        """
        Save trained model to disk.

        Args:
            pipeline: Trained model pipeline
            dataset_name: Dataset name
            group_name: Model group name
            model_name: Model name

        Returns:
            Path to saved model file
        """
        # Create models directory
        safe_ds = self.result_mgr.sanitize_name(dataset_name)
        models_dir = self.result_mgr.base_dir / 'models' / safe_ds / group_name
        models_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = models_dir / f"{model_name}.pkl"

        with open(model_path, 'wb') as f:
            pickle.dump(pipeline, f)

        return model_path

    def _save_training_summary(self, results: Dict[str, Any]):
        """
        Save training summary and comparison tables.

        Args:
            results: Training results for all datasets
        """
        try:
            # Create summary DataFrame
            rows = []

            for dataset_name, dataset_results in results.items():
                if 'error' in dataset_results:
                    continue

                for group_name, group_results in dataset_results.items():
                    for model_name, model_result in group_results.items():
                        if model_result.get('status') == 'failed':
                            continue

                        metrics = model_result.get('metrics', {})
                        row = {
                            'dataset': dataset_name,
                            'group': group_name,
                            'model': model_name,
                            **metrics
                        }
                        rows.append(row)

            if not rows:
                self.logger.warning("No successful model results to save")
                return

            summary_df = pd.DataFrame(rows)

            # Save complete summary
            summary_path = self.result_mgr.base_dir / 'model_training_summary.csv'
            summary_df.to_csv(summary_path, index=False)
            self.logger.info(f"Saved training summary: {summary_path}")

            # Save per-dataset summaries
            for dataset_name in summary_df['dataset'].unique():
                dataset_df = summary_df[summary_df['dataset'] == dataset_name]
                safe_ds = self.result_mgr.sanitize_name(dataset_name)
                dataset_path = self.result_mgr.base_dir / f'model_summary_{safe_ds}.csv'
                dataset_df.to_csv(dataset_path, index=False)

        except Exception as e:
            self.logger.error(f"Failed to save training summary: {str(e)}")

    def get_best_models(self, metric: str = 'AUC') -> Dict[str, Any]:
        """
        Get best model for each dataset based on specified metric.

        Args:
            metric: Metric to use for selection (default: AUC)

        Returns:
            Dictionary mapping dataset names to best model info
        """
        best_models = {}

        for dataset_name, dataset_results in self.evaluation_results.items():
            if 'error' in dataset_results:
                continue

            best_score = -float('inf')
            best_model_info = None

            for group_name, group_results in dataset_results.items():
                for model_name, model_result in group_results.items():
                    if model_result.get('status') == 'failed':
                        continue

                    score = model_result.get('metrics', {}).get(metric, -float('inf'))

                    if score > best_score:
                        best_score = score
                        best_model_info = {
                            'group': group_name,
                            'model': model_name,
                            'metrics': model_result['metrics'],
                            'model_path': model_result['model_path'],
                            'model_object': model_result.get('model_object')
                        }

            if best_model_info:
                best_models[dataset_name] = best_model_info

        return best_models
