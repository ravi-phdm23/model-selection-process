"""Report generation module - integrates with existing report_manager.py."""

import logging
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

from result_manager import ResultManager
from report_manager import ReportManager


class ReportGenerationModule:
    """
    Handles LaTeX report generation for the pipeline.

    Responsibilities:
    - Integrate with existing ReportManager
    - Generate comprehensive LaTeX reports
    - Optionally compile to PDF
    - Include all analysis results (EDA, models, SHAP, local)
    """

    def __init__(
        self,
        config: Dict[str, Any],
        result_manager: ResultManager,
        logger: logging.Logger
    ):
        """
        Initialize report generation module.

        Args:
            config: Pipeline configuration
            result_manager: Result manager instance
            logger: Logger instance
        """
        self.config = config
        self.result_mgr = result_manager
        self.logger = logger

        # Extract report config
        self.report_config = config.get('report', {}).get('latex', {})
        self.compile_pdf = self.report_config.get('compile_pdf', False)
        self.include_narratives = self.report_config.get('include_narratives', False)

        # Initialize ReportManager
        self.report_manager = ReportManager(base_dir=result_manager.base_dir.parent)

    def generate_report(
        self,
        pipeline_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate comprehensive LaTeX report.

        Args:
            pipeline_results: Results from all pipeline stages

        Returns:
            Dictionary with report generation results
        """
        self.logger.info("Generating LaTeX report...")

        try:
            # Prepare manifest data
            manifest = self._prepare_manifest(pipeline_results)

            # Generate markdown report (simplified for CLI)
            markdown_report = self._generate_markdown_summary(pipeline_results)

            # Generate LaTeX content
            self.logger.info("  Creating LaTeX document...")
            project_name = self.config.get('project', {}).get('name', 'Model Selection Pipeline')
            latex_content = self.report_manager.generate_latex_report(
                markdown_report=markdown_report,
                manifest=manifest,
                title=project_name,
                author="Automated Pipeline"
            )

            # Save LaTeX content to file
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            reports_dir = self.result_mgr.base_dir.parent / 'reports'
            reports_dir.mkdir(parents=True, exist_ok=True)

            latex_path = reports_dir / f'report_{timestamp}.tex'
            with open(latex_path, 'w', encoding='utf-8') as f:
                f.write(latex_content)

            self.logger.info(f"  LaTeX report saved: {latex_path}")

            # Compile to PDF if enabled
            pdf_path = None
            if self.compile_pdf:
                self.logger.info("  Compiling PDF...")
                try:
                    pdf_path = self._compile_latex_to_pdf(latex_path)
                    if pdf_path:
                        self.logger.info(f"  PDF compiled: {pdf_path}")
                    else:
                        self.logger.warning("  PDF compilation failed (pdflatex not available)")
                except Exception as e:
                    self.logger.warning(f"  PDF compilation failed: {str(e)}")

            return {
                'status': 'completed',
                'latex_path': str(latex_path),
                'pdf_path': str(pdf_path) if pdf_path else None
            }

        except Exception as e:
            self.logger.error(f"Report generation failed: {str(e)}", exc_info=True)
            return {
                'status': 'failed',
                'error': str(e)
            }

    def _generate_markdown_summary(self, pipeline_results: Dict[str, Any]) -> str:
        """
        Generate a simple markdown summary for the report.

        Args:
            pipeline_results: Results from all pipeline stages

        Returns:
            Markdown string
        """
        lines = []
        lines.append("# Executive Summary\n")

        # Data summary
        data_results = pipeline_results.get('data', {})
        if data_results:
            lines.append("## Data Overview\n")
            lines.append(f"- Datasets: {data_results.get('num_datasets', 0)}")
            lines.append(f"- Total training samples: {data_results.get('total_train_samples', 0)}")
            lines.append(f"- Total test samples: {data_results.get('total_test_samples', 0)}\n")

        # Model summary
        models_results = pipeline_results.get('models', {})
        if models_results:
            lines.append("## Model Training\n")
            lines.append(f"- Total models trained: {models_results.get('total_models', 0)}\n")

            best_models = models_results.get('best_models', {})
            if best_models:
                lines.append("### Best Models by Dataset\n")
                for dataset_name, model_info in best_models.items():
                    model_name = f"{model_info.get('group')}/{model_info.get('model')}"
                    metrics = model_info.get('metrics', {})
                    auc = metrics.get('AUC', 0)
                    lines.append(f"- **{dataset_name}**: {model_name} (AUC={auc:.4f})")
                lines.append("")

        # Analysis summary
        shap_results = pipeline_results.get('shap', {})
        if shap_results and shap_results.get('status') == 'completed':
            lines.append("## SHAP Analysis\n")
            num_datasets = len(shap_results.get('results', {}))
            lines.append(f"- SHAP analysis completed for {num_datasets} dataset(s)\n")

        return '\n'.join(lines)

    def _prepare_manifest(self, pipeline_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare manifest data for report generation.

        Args:
            pipeline_results: Results from all pipeline stages

        Returns:
            Manifest dictionary for ReportManager
        """
        manifest = {
            'project_name': self.config.get('project', {}).get('name', 'Model Selection Pipeline'),
            'datasets': [],
            'models': {},
            'best_models': {},
            'shap_plots': {},
            'local_analyses': []
        }

        # Extract dataset information
        data_results = pipeline_results.get('data', {})
        if data_results:
            manifest['num_datasets'] = data_results.get('num_datasets', 0)
            manifest['total_train_samples'] = data_results.get('total_train_samples', 0)
            manifest['total_test_samples'] = data_results.get('total_test_samples', 0)

        # Extract model training results
        models_results = pipeline_results.get('models', {})
        if models_results:
            training_results = models_results.get('training_results', {})
            best_models = models_results.get('best_models', {})

            # Organize by dataset
            for dataset_name, dataset_results in training_results.items():
                if 'error' in dataset_results:
                    continue

                # Add dataset to list
                manifest['datasets'].append(dataset_name)

                # Add models for this dataset
                models_list = []
                for group_name, group_results in dataset_results.items():
                    for model_name, model_result in group_results.items():
                        if model_result.get('status') == 'completed':
                            models_list.append({
                                'group': group_name,
                                'name': model_name,
                                'metrics': model_result.get('metrics', {})
                            })

                manifest['models'][dataset_name] = models_list

                # Add best model
                if dataset_name in best_models:
                    manifest['best_models'][dataset_name] = best_models[dataset_name]

        # Extract SHAP results
        shap_results = pipeline_results.get('shap', {})
        if shap_results and shap_results.get('status') == 'completed':
            for dataset_name, dataset_shap in shap_results.get('results', {}).items():
                if dataset_shap.get('status') == 'completed':
                    plots = dataset_shap.get('plots_generated', [])
                    manifest['shap_plots'][dataset_name] = plots

        # Extract local analyses
        local_results = pipeline_results.get('local', {})
        if local_results and local_results.get('status') == 'completed':
            for dataset_name, dataset_local in local_results.get('results', {}).items():
                if dataset_local.get('status') == 'completed':
                    analyses = dataset_local.get('analyses', [])
                    manifest['local_analyses'].extend(analyses)

        return manifest

    def generate_summary_report(self, pipeline_results: Dict[str, Any]) -> Path:
        """
        Generate a simple text summary report.

        Args:
            pipeline_results: Results from all pipeline stages

        Returns:
            Path to summary report
        """
        try:
            summary_lines = []
            summary_lines.append("=" * 80)
            summary_lines.append("MODEL SELECTION PIPELINE - EXECUTION SUMMARY")
            summary_lines.append("=" * 80)
            summary_lines.append("")

            # Project info
            project_name = self.config.get('project', {}).get('name', 'Unknown')
            summary_lines.append(f"Project: {project_name}")
            summary_lines.append("")

            # Data summary
            data_results = pipeline_results.get('data', {})
            if data_results:
                summary_lines.append("DATA:")
                summary_lines.append(f"  Datasets: {data_results.get('num_datasets', 0)}")
                summary_lines.append(f"  Total train samples: {data_results.get('total_train_samples', 0)}")
                summary_lines.append(f"  Total test samples: {data_results.get('total_test_samples', 0)}")
                summary_lines.append("")

            # Model summary
            models_results = pipeline_results.get('models', {})
            if models_results:
                summary_lines.append("MODELS:")
                summary_lines.append(f"  Total models trained: {models_results.get('total_models', 0)}")
                summary_lines.append("")

                best_models = models_results.get('best_models', {})
                if best_models:
                    summary_lines.append("BEST MODELS:")
                    for dataset_name, model_info in best_models.items():
                        model_name = f"{model_info.get('group')}/{model_info.get('model')}"
                        metrics = model_info.get('metrics', {})
                        auc = metrics.get('AUC', 0)
                        summary_lines.append(f"  {dataset_name}: {model_name} (AUC={auc:.4f})")
                summary_lines.append("")

            # SHAP summary
            shap_results = pipeline_results.get('shap', {})
            if shap_results and shap_results.get('status') == 'completed':
                num_datasets = len(shap_results.get('results', {}))
                summary_lines.append(f"SHAP ANALYSIS: Completed for {num_datasets} dataset(s)")
                summary_lines.append("")

            # Reliability summary
            reliability_results = pipeline_results.get('reliability', {})
            if reliability_results and reliability_results.get('status') == 'completed':
                num_datasets = len(reliability_results.get('results', {}))
                summary_lines.append(f"RELIABILITY TESTING: Completed for {num_datasets} dataset(s)")
                summary_lines.append("")

            # Local analysis summary
            local_results = pipeline_results.get('local', {})
            if local_results and local_results.get('status') == 'completed':
                total_instances = sum(
                    r.get('num_instances', 0)
                    for r in local_results.get('results', {}).values()
                    if isinstance(r, dict)
                )
                summary_lines.append(f"LOCAL ANALYSIS: {total_instances} instance(s) analyzed")
                summary_lines.append("")

            summary_lines.append("=" * 80)

            # Save summary
            summary_path = self.result_mgr.base_dir.parent / 'PIPELINE_SUMMARY.txt'
            with open(summary_path, 'w') as f:
                f.write('\n'.join(summary_lines))

            self.logger.info(f"Summary report saved: {summary_path}")
            return summary_path

        except Exception as e:
            self.logger.error(f"Failed to generate summary report: {str(e)}")
            return None

    def _compile_latex_to_pdf(self, latex_path: Path) -> Optional[Path]:
        """
        Compile LaTeX file to PDF using pdflatex.

        Args:
            latex_path: Path to LaTeX file

        Returns:
            Path to generated PDF file, or None if compilation failed
        """
        import subprocess
        import shutil

        # Check if pdflatex is available
        if not shutil.which('pdflatex'):
            self.logger.warning("  pdflatex not found in PATH")
            return None

        try:
            # Run pdflatex twice for proper references/TOC
            latex_dir = latex_path.parent
            latex_file = latex_path.name

            self.logger.debug(f"  Running pdflatex (pass 1/2)...")
            result1 = subprocess.run(
                ['pdflatex', '-interaction=nonstopmode', latex_file],
                cwd=str(latex_dir),
                capture_output=True,
                timeout=60
            )

            self.logger.debug(f"  Running pdflatex (pass 2/2)...")
            result2 = subprocess.run(
                ['pdflatex', '-interaction=nonstopmode', latex_file],
                cwd=str(latex_dir),
                capture_output=True,
                timeout=60
            )

            # Check if PDF was generated
            pdf_path = latex_path.with_suffix('.pdf')

            # Give filesystem a moment to sync
            import time
            time.sleep(0.5)

            if pdf_path.exists():
                self.logger.debug(f"  PDF generated successfully: {pdf_path}")

                # Clean up auxiliary files
                for ext in ['.aux', '.log', '.out', '.toc']:
                    aux_file = latex_path.with_suffix(ext)
                    if aux_file.exists():
                        try:
                            aux_file.unlink()
                        except:
                            pass  # Ignore cleanup errors

                return pdf_path
            else:
                self.logger.warning(f"  PDF file not found at: {pdf_path}")
                self.logger.debug(f"  pdflatex exit codes: {result1.returncode}, {result2.returncode}")
                return None

        except subprocess.TimeoutExpired:
            self.logger.warning("  pdflatex compilation timed out")
            return None
        except subprocess.CalledProcessError as e:
            self.logger.warning(f"  pdflatex failed with exit code {e.returncode}")
            return None
        except Exception as e:
            self.logger.warning(f"  PDF compilation error: {str(e)}")
            return None
