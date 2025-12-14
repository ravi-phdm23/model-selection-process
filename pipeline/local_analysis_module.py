"""Local instance analysis module for row-specific explanations."""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import warnings
warnings.filterwarnings('ignore')

# Import SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from result_manager import ResultManager


class LocalAnalysisModule:
    """
    Handles local (instance-level) SHAP analysis.

    Responsibilities:
    - Generate row-specific SHAP waterfall plots
    - Create individual prediction explanations
    - Save local analyses for LaTeX integration
    - Support both specified instances and random sampling
    """

    def __init__(
        self,
        config: Dict[str, Any],
        result_manager: ResultManager,
        logger: logging.Logger
    ):
        """
        Initialize local analysis module.

        Args:
            config: Pipeline configuration
            result_manager: Result manager instance
            logger: Logger instance
        """
        self.config = config
        self.result_mgr = result_manager
        self.logger = logger

        # Extract local analysis config
        self.local_config = config.get('analysis', {}).get('local_shap', {})
        self.instances = self.local_config.get('instances', [])
        self.n_random = self.local_config.get('n_random', 5)  # Random instances if none specified

        if not SHAP_AVAILABLE:
            self.logger.warning("SHAP library not available - install shap package")

    def run_local_analysis(
        self,
        datasets: Dict[str, Dict[str, Any]],
        best_models: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run local SHAP analysis for best models.

        Args:
            datasets: Preprocessed datasets
            best_models: Best models for each dataset

        Returns:
            Dictionary with local analysis results
        """
        if not SHAP_AVAILABLE:
            self.logger.error("SHAP library not available")
            return {'status': 'skipped', 'reason': 'SHAP not installed'}

        self.logger.info("Running local SHAP analysis...")

        results = {}

        for dataset_name, model_info in best_models.items():
            self.logger.info(f"  Analyzing: {dataset_name}")

            try:
                # Get dataset and model
                dataset_data = datasets.get(dataset_name)
                if dataset_data is None:
                    self.logger.warning(f"    Dataset not found: {dataset_name}")
                    continue

                model_pipeline = model_info.get('model_object')
                if model_pipeline is None:
                    self.logger.warning(f"    Model object not found for {dataset_name}")
                    continue

                # Run local analysis
                local_result = self._analyze_instances(
                    dataset_name=dataset_name,
                    dataset_data=dataset_data,
                    model_pipeline=model_pipeline,
                    model_name=model_info.get('model', 'unknown')
                )

                results[dataset_name] = local_result
                self.logger.info(
                    f"    [OK] Analyzed {local_result.get('num_instances', 0)} instance(s)"
                )

            except Exception as e:
                self.logger.error(
                    f"    [ERROR] Local analysis failed: {str(e)}",
                    exc_info=True
                )
                results[dataset_name] = {'status': 'failed', 'error': str(e)}

        return {
            'status': 'completed',
            'results': results
        }

    def _analyze_instances(
        self,
        dataset_name: str,
        dataset_data: Dict[str, Any],
        model_pipeline: Any,
        model_name: str
    ) -> Dict[str, Any]:
        """
        Analyze specific instances for a dataset.

        Args:
            dataset_name: Dataset name
            dataset_data: Dataset data
            model_pipeline: Trained model pipeline
            model_name: Model name

        Returns:
            Dictionary with local analysis results
        """
        X_train = dataset_data['X_train']
        X_test = dataset_data['X_test']
        y_test = dataset_data['y_test']

        # Determine which instances to analyze
        if self.instances:
            # Use specified instances
            instance_indices = [i for i in self.instances if i < len(X_test)]
        else:
            # Random sample
            n_sample = min(self.n_random, len(X_test))
            instance_indices = np.random.choice(len(X_test), size=n_sample, replace=False)

        self.logger.debug(f"      Analyzing {len(instance_indices)} instance(s)")

        # Create SHAP explainer
        # Extract final estimator if using sklearn Pipeline
        model = model_pipeline
        if hasattr(model_pipeline, 'steps'):
            # It's a Pipeline - get the last step (the actual model)
            model = model_pipeline.steps[-1][1]
            # If it's CalibratedClassifierCV, get the base estimator
            if hasattr(model, 'estimator'):
                model = model.estimator

        try:
            explainer = shap.TreeExplainer(model)
            self.logger.debug(f"      Using TreeExplainer")
        except:
            sample_size = min(100, len(X_train))
            background = shap.sample(X_train, sample_size)

            # Use the pipeline's predict_proba for explanation
            def predict_fn(X):
                return model_pipeline.predict_proba(X)[:, 1]

            explainer = shap.Explainer(predict_fn, background)
            self.logger.debug(f"      Using Explainer with {sample_size} background samples")

        analyses = []

        for idx in instance_indices:
            try:
                analysis = self._analyze_single_instance(
                    dataset_name=dataset_name,
                    model_name=model_name,
                    explainer=explainer,
                    X_test=X_test,
                    y_test=y_test,
                    model_pipeline=model_pipeline,
                    instance_idx=int(idx)
                )
                analyses.append(analysis)

            except Exception as e:
                self.logger.warning(f"        Failed to analyze instance {idx}: {str(e)}")

        return {
            'status': 'completed',
            'model': model_name,
            'num_instances': len(analyses),
            'analyses': analyses
        }

    def _analyze_single_instance(
        self,
        dataset_name: str,
        model_name: str,
        explainer: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_pipeline: Any,
        instance_idx: int
    ) -> Dict[str, Any]:
        """
        Analyze a single instance.

        Args:
            dataset_name: Dataset name
            model_name: Model name
            explainer: SHAP explainer
            X_test: Test features
            y_test: Test labels
            model_pipeline: Trained model
            instance_idx: Instance index

        Returns:
            Dictionary with instance analysis
        """
        # Get instance
        instance = X_test.iloc[[instance_idx]]
        true_label = int(y_test.iloc[instance_idx])

        # Predict
        pred_proba = model_pipeline.predict_proba(instance)[0, 1]
        pred_label = int(pred_proba > 0.5)

        # Compute SHAP values
        shap_values = explainer.shap_values(instance)

        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        # Create waterfall plot
        waterfall_path = self._create_waterfall_plot(
            dataset_name=dataset_name,
            model_name=model_name,
            instance_idx=instance_idx,
            explainer=explainer,
            shap_values=shap_values[0] if len(shap_values.shape) > 1 else shap_values,
            instance_data=instance.iloc[0].values,
            feature_names=X_test.columns.tolist()
        )

        # Create analysis record
        analysis = {
            'dataset': dataset_name,
            'model': model_name,
            'instance_idx': instance_idx,
            'true_label': true_label,
            'pred_label': pred_label,
            'pred_proba': float(pred_proba),
            'waterfall_plot': str(waterfall_path) if waterfall_path else None,
            'feature_values': instance.iloc[0].to_dict(),
            'shap_values': {
                feature: float(val)
                for feature, val in zip(X_test.columns, shap_values[0] if len(shap_values.shape) > 1 else shap_values)
            }
        }

        # Save to JSON
        self._save_local_analysis(dataset_name, instance_idx, analysis)

        return analysis

    def _create_waterfall_plot(
        self,
        dataset_name: str,
        model_name: str,
        instance_idx: int,
        explainer: Any,
        shap_values: np.ndarray,
        instance_data: np.ndarray,
        feature_names: List[str]
    ) -> Optional[Path]:
        """Create and save waterfall plot for instance."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        try:
            safe_ds = self.result_mgr.sanitize_name(dataset_name)
            safe_model = self.result_mgr.sanitize_name(model_name)

            plots_dir = self.result_mgr.figures_dir / 'shap' / safe_ds
            plots_dir.mkdir(parents=True, exist_ok=True)

            # Create explanation object
            base_value = explainer.expected_value
            if isinstance(base_value, np.ndarray):
                base_value = base_value[1] if len(base_value) > 1 else base_value[0]

            explanation = shap.Explanation(
                values=shap_values,
                base_values=base_value,
                data=instance_data,
                feature_names=feature_names
            )

            # Create waterfall plot
            fig = plt.figure(figsize=(10, 6))
            shap.waterfall_plot(explanation, max_display=15, show=False)
            plt.tight_layout()

            plot_path = plots_dir / f'{safe_model}_waterfall_row{instance_idx}.png'
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()

            return plot_path

        except Exception as e:
            self.logger.warning(f"        Failed to create waterfall plot: {str(e)}")
            plt.close('all')
            return None

    def _save_local_analysis(
        self,
        dataset_name: str,
        instance_idx: int,
        analysis: Dict[str, Any]
    ):
        """Save local analysis to JSON."""
        try:
            safe_ds = self.result_mgr.sanitize_name(dataset_name)

            local_dir = self.result_mgr.local_analyses_dir
            local_dir.mkdir(parents=True, exist_ok=True)

            json_path = local_dir / f'{safe_ds}_row{instance_idx}.json'

            with open(json_path, 'w') as f:
                json.dump(analysis, f, indent=2)

            self.logger.debug(f"        Saved local analysis: {json_path}")

        except Exception as e:
            self.logger.warning(f"        Failed to save local analysis: {str(e)}")
