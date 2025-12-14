"""SHAP analysis module for model interpretability."""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Import SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from result_manager import ResultManager


class SHAPModule:
    """
    Handles SHAP analysis for model interpretability.

    Responsibilities:
    - Compute SHAP values for best models
    - Generate global SHAP visualizations
    - Save SHAP values and plots
    - Support multiple plot types (bar, beeswarm, waterfall)
    """

    def __init__(
        self,
        config: Dict[str, Any],
        result_manager: ResultManager,
        logger: logging.Logger
    ):
        """
        Initialize SHAP module.

        Args:
            config: Pipeline configuration
            result_manager: Result manager instance
            logger: Logger instance
        """
        self.config = config
        self.result_mgr = result_manager
        self.logger = logger

        # Extract SHAP config
        self.shap_config = config.get('analysis', {}).get('shap', {})
        self.plot_types = self.shap_config.get('plot_types', ['bar', 'dot'])
        self.max_display = self.shap_config.get('max_display', 20)

        if not SHAP_AVAILABLE:
            self.logger.warning("SHAP library not available - install shap package")

    def run_shap_analysis(
        self,
        datasets: Dict[str, Dict[str, Any]],
        best_models: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run SHAP analysis for best models on each dataset.

        Args:
            datasets: Preprocessed datasets
            best_models: Best models for each dataset

        Returns:
            Dictionary with SHAP analysis results
        """
        if not SHAP_AVAILABLE:
            self.logger.error("SHAP library not available")
            return {'status': 'skipped', 'reason': 'SHAP not installed'}

        self.logger.info("Running SHAP analysis...")

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

                # Run SHAP analysis
                shap_result = self._analyze_model(
                    dataset_name=dataset_name,
                    dataset_data=dataset_data,
                    model_pipeline=model_pipeline,
                    model_name=model_info.get('model', 'unknown')
                )

                results[dataset_name] = shap_result
                self.logger.info(f"    [OK] SHAP analysis completed")

            except Exception as e:
                self.logger.error(
                    f"    [ERROR] SHAP analysis failed: {str(e)}",
                    exc_info=True
                )
                results[dataset_name] = {'status': 'failed', 'error': str(e)}

        return {
            'status': 'completed',
            'results': results
        }

    def _analyze_model(
        self,
        dataset_name: str,
        dataset_data: Dict[str, Any],
        model_pipeline: Any,
        model_name: str
    ) -> Dict[str, Any]:
        """
        Perform SHAP analysis for a single model.

        Args:
            dataset_name: Dataset name
            dataset_data: Dataset data
            model_pipeline: Trained model pipeline
            model_name: Model name

        Returns:
            Dictionary with SHAP results
        """
        X_train = dataset_data['X_train']
        X_test = dataset_data['X_test']

        # Create SHAP explainer
        self.logger.debug(f"      Creating SHAP explainer...")

        # Extract final estimator if using sklearn Pipeline
        model = model_pipeline
        if hasattr(model_pipeline, 'steps'):
            # It's a Pipeline - get the last step (the actual model)
            model = model_pipeline.steps[-1][1]
            # If it's CalibratedClassifierCV, get the base estimator
            if hasattr(model, 'estimator'):
                model = model.estimator

        # Use TreeExplainer for tree-based models, KernelExplainer otherwise
        try:
            # Try TreeExplainer first (faster for tree models)
            explainer = shap.TreeExplainer(model)
            self.logger.debug(f"      Using TreeExplainer")
        except:
            # Fall back to Explainer with sample (works for any model)
            sample_size = min(100, len(X_train))
            background = shap.sample(X_train, sample_size)

            # Use the pipeline's predict_proba for explanation
            def predict_fn(X):
                return model_pipeline.predict_proba(X)[:, 1]

            explainer = shap.Explainer(predict_fn, background)
            self.logger.debug(f"      Using Explainer with {sample_size} background samples")

        # Compute SHAP values
        self.logger.debug(f"      Computing SHAP values...")
        shap_values = explainer.shap_values(X_test)

        # For binary classification, get positive class SHAP values
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_values = shap_values[1]

        # Create explanation object
        explanation = shap.Explanation(
            values=shap_values,
            base_values=explainer.expected_value if hasattr(explainer, 'expected_value') else 0,
            data=X_test.values,
            feature_names=X_test.columns.tolist()
        )

        # Save SHAP values
        self._save_shap_values(dataset_name, model_name, explanation, X_test)

        # Generate plots
        plots_generated = self._generate_plots(
            dataset_name=dataset_name,
            model_name=model_name,
            explanation=explanation,
            X_test=X_test
        )

        return {
            'status': 'completed',
            'model': model_name,
            'num_instances': len(X_test),
            'plots_generated': plots_generated
        }

    def _save_shap_values(
        self,
        dataset_name: str,
        model_name: str,
        explanation: Any,
        X_test: pd.DataFrame
    ):
        """Save SHAP values to CSV."""
        try:
            safe_ds = self.result_mgr.sanitize_name(dataset_name)
            safe_model = self.result_mgr.sanitize_name(model_name)

            # Create SHAP directory
            shap_dir = self.result_mgr.base_dir / 'shap' / safe_ds
            shap_dir.mkdir(parents=True, exist_ok=True)

            # Save SHAP values as CSV
            shap_df = pd.DataFrame(
                explanation.values,
                columns=[f"{col}_shap" for col in X_test.columns]
            )
            shap_path = shap_dir / f'{safe_model}_shap_values.csv'
            shap_df.to_csv(shap_path, index=False)

            self.logger.debug(f"        Saved SHAP values: {shap_path}")

        except Exception as e:
            self.logger.warning(f"        Failed to save SHAP values: {str(e)}")

    def _generate_plots(
        self,
        dataset_name: str,
        model_name: str,
        explanation: Any,
        X_test: pd.DataFrame
    ) -> List[str]:
        """Generate and save SHAP plots."""
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt

        safe_ds = self.result_mgr.sanitize_name(dataset_name)
        safe_model = self.result_mgr.sanitize_name(model_name)

        plots_dir = self.result_mgr.figures_dir / 'shap' / safe_ds
        plots_dir.mkdir(parents=True, exist_ok=True)

        plots_generated = []

        for plot_type in self.plot_types:
            try:
                fig, ax = plt.subplots(figsize=(10, 6))

                if plot_type == 'bar':
                    shap.summary_plot(
                        explanation.values,
                        X_test,
                        plot_type='bar',
                        max_display=self.max_display,
                        show=False
                    )
                    plot_path = plots_dir / f'{safe_model}_shap_bar.png'

                elif plot_type in ['dot', 'beeswarm']:
                    shap.summary_plot(
                        explanation.values,
                        X_test,
                        plot_type='dot',
                        max_display=self.max_display,
                        show=False
                    )
                    plot_path = plots_dir / f'{safe_model}_shap_beeswarm.png'

                elif plot_type == 'waterfall':
                    # Waterfall for first instance
                    shap.waterfall_plot(
                        shap.Explanation(
                            values=explanation.values[0],
                            base_values=explanation.base_values[0] if isinstance(explanation.base_values, np.ndarray) else explanation.base_values,
                            data=explanation.data[0],
                            feature_names=explanation.feature_names
                        ),
                        max_display=self.max_display,
                        show=False
                    )
                    plot_path = plots_dir / f'{safe_model}_shap_waterfall.png'

                else:
                    self.logger.warning(f"        Unknown plot type: {plot_type}")
                    plt.close(fig)
                    continue

                plt.tight_layout()
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                plt.close()

                plots_generated.append(str(plot_path))
                self.logger.debug(f"        Generated {plot_type} plot: {plot_path}")

            except Exception as e:
                self.logger.warning(f"        Failed to generate {plot_type} plot: {str(e)}")
                plt.close('all')

        return plots_generated

    def compute_feature_importance(
        self,
        datasets: Dict[str, Dict[str, Any]],
        best_models: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Compute SHAP-based feature importance for best models.

        Args:
            datasets: Preprocessed datasets
            best_models: Best models for each dataset

        Returns:
            DataFrame with feature importance rankings
        """
        if not SHAP_AVAILABLE:
            return pd.DataFrame()

        importance_data = []

        for dataset_name, model_info in best_models.items():
            try:
                dataset_data = datasets.get(dataset_name)
                model_pipeline = model_info.get('model_object')

                if dataset_data is None or model_pipeline is None:
                    continue

                X_test = dataset_data['X_test']

                # Extract final estimator if using sklearn Pipeline
                model = model_pipeline
                if hasattr(model_pipeline, 'steps'):
                    model = model_pipeline.steps[-1][1]
                    if hasattr(model, 'estimator'):
                        model = model.estimator

                # Compute SHAP values (simplified)
                try:
                    explainer = shap.TreeExplainer(model)
                except:
                    continue

                shap_values = explainer.shap_values(X_test)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]

                # Mean absolute SHAP values
                mean_shap = np.abs(shap_values).mean(axis=0)

                for feature, importance in zip(X_test.columns, mean_shap):
                    importance_data.append({
                        'dataset': dataset_name,
                        'model': model_info.get('model'),
                        'feature': feature,
                        'shap_importance': importance
                    })

            except Exception as e:
                self.logger.debug(f"Could not compute importance for {dataset_name}: {str(e)}")

        if importance_data:
            return pd.DataFrame(importance_data)
        return pd.DataFrame()
