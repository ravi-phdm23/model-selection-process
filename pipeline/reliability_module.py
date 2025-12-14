"""Reliability testing module for model stability analysis."""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
import warnings
warnings.filterwarnings('ignore')

from result_manager import ResultManager


class ReliabilityModule:
    """
    Handles reliability testing for model stability.

    Responsibilities:
    - Permutation-based reliability testing (optional)
    - Bootstrap confidence intervals
    - Stability metrics across random seeds

    Note: This is a simplified implementation. Full reliability testing
    can be computationally expensive and is marked as optional.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        result_manager: ResultManager,
        logger: logging.Logger
    ):
        """
        Initialize reliability module.

        Args:
            config: Pipeline configuration
            result_manager: Result manager instance
            logger: Logger instance
        """
        self.config = config
        self.result_mgr = result_manager
        self.logger = logger

        # Extract reliability config
        self.reliability_config = config.get('analysis', {}).get('reliability', {})
        self.n_permutations = self.reliability_config.get('n_permutations', 50)

    def run_reliability_testing(
        self,
        datasets: Dict[str, Dict[str, Any]],
        best_models: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run reliability testing for best models.

        Args:
            datasets: Preprocessed datasets
            best_models: Best models for each dataset

        Returns:
            Dictionary with reliability test results
        """
        self.logger.info("Running reliability testing...")
        self.logger.info(f"  Permutations: {self.n_permutations}")

        results = {}

        for dataset_name, model_info in best_models.items():
            self.logger.info(f"  Testing: {dataset_name}")

            try:
                # Simple bootstrap reliability test
                reliability_result = self._test_model_reliability(
                    dataset_name=dataset_name,
                    dataset_data=datasets.get(dataset_name),
                    model_info=model_info
                )

                results[dataset_name] = reliability_result
                self.logger.info(
                    f"    [OK] Reliability: {reliability_result.get('mean_score', 0):.4f} "
                    f"± {reliability_result.get('std_score', 0):.4f}"
                )

            except Exception as e:
                self.logger.error(
                    f"    [ERROR] Reliability test failed: {str(e)}",
                    exc_info=True
                )
                results[dataset_name] = {'status': 'failed', 'error': str(e)}

        # Save summary
        self._save_reliability_summary(results)

        return {
            'status': 'completed',
            'results': results
        }

    def _test_model_reliability(
        self,
        dataset_name: str,
        dataset_data: Dict[str, Any],
        model_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Test reliability of a single model using bootstrap sampling.

        Args:
            dataset_name: Dataset name
            dataset_data: Dataset data
            model_info: Model information

        Returns:
            Dictionary with reliability metrics
        """
        from sklearn.metrics import roc_auc_score

        X_test = dataset_data['X_test']
        y_test = dataset_data['y_test']
        model_pipeline = model_info.get('model_object')

        if model_pipeline is None:
            raise ValueError("Model object not found")

        # Bootstrap sampling
        scores = []
        n_samples = len(X_test)

        self.logger.debug(f"      Running {self.n_permutations} bootstrap iterations...")

        for i in range(self.n_permutations):
            # Sample with replacement
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_boot = X_test.iloc[indices]
            y_boot = y_test.iloc[indices]

            try:
                # Predict
                y_pred_proba = model_pipeline.predict_proba(X_boot)[:, 1]
                score = roc_auc_score(y_boot, y_pred_proba)
                scores.append(score)
            except:
                continue

        if not scores:
            raise ValueError("No successful bootstrap iterations")

        # Compute statistics
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        ci_lower = np.percentile(scores, 2.5)
        ci_upper = np.percentile(scores, 97.5)

        return {
            'status': 'completed',
            'model': model_info.get('model'),
            'mean_score': float(mean_score),
            'std_score': float(std_score),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'n_iterations': len(scores)
        }

    def _save_reliability_summary(self, results: Dict[str, Any]):
        """Save reliability testing summary."""
        try:
            rows = []

            for dataset_name, result in results.items():
                if result.get('status') == 'failed':
                    continue

                row = {
                    'dataset': dataset_name,
                    'model': result.get('model'),
                    'mean_auc': result.get('mean_score'),
                    'std_auc': result.get('std_score'),
                    'ci_lower': result.get('ci_lower'),
                    'ci_upper': result.get('ci_upper'),
                    'n_iterations': result.get('n_iterations')
                }
                rows.append(row)

            if rows:
                summary_df = pd.DataFrame(rows)
                summary_path = self.result_mgr.base_dir / 'reliability_summary.csv'
                summary_df.to_csv(summary_path, index=False)
                self.logger.info(f"Saved reliability summary: {summary_path}")

        except Exception as e:
            self.logger.error(f"Failed to save reliability summary: {str(e)}")
