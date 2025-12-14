"""Pipeline orchestrator - coordinates all pipeline stages."""

import logging
import shutil
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

# Import pipeline modules
from result_manager import ResultManager
from pipeline.data_loader import DataLoader
from pipeline.eda_module import EDAModule
from pipeline.model_trainer import ModelTrainer
from pipeline.benchmark_module import BenchmarkModule
from pipeline.shap_module import SHAPModule
from pipeline.reliability_module import ReliabilityModule
from pipeline.local_analysis_module import LocalAnalysisModule
from pipeline.report_generation_module import ReportGenerationModule


class PipelineOrchestrator:
    """
    Main orchestrator that coordinates all pipeline stages.

    Manages the execution flow, checkpointing, and result aggregation.
    """

    # Define pipeline stages in execution order
    STAGES = [
        'init',           # Initialization
        'data',           # Data loading and preprocessing
        'eda',            # Exploratory Data Analysis
        'models',         # Model training and benchmarking
        'shap',           # SHAP analysis
        'reliability',    # Reliability testing
        'local',          # Local instance analysis
        'report',         # Report generation
        'finalize'        # Finalization and cleanup
    ]

    def __init__(
        self,
        config: Dict[str, Any],
        output_dir: Path,
        logger: logging.Logger,
        resume: bool = False
    ):
        """
        Initialize pipeline orchestrator.

        Args:
            config: Validated configuration dictionary
            output_dir: Output directory for results
            logger: Logger instance
            resume: Whether to resume from checkpoint
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.logger = logger
        self.resume = resume

        # Pipeline state
        self.results = {}
        self.completed_stages = set()
        self.current_stage = None

        # Create subdirectories
        self._create_directories()

        # Save config to output directory
        self._save_config()

        # Initialize pipeline modules
        self._initialize_modules()

        # Load checkpoint if resuming
        if resume:
            self._load_checkpoint()

    def _initialize_modules(self):
        """Initialize pipeline modules (ResultManager, DataLoader, etc.)."""
        # Initialize ResultManager
        self.result_mgr = ResultManager(base_dir=self.output_dir / "results")
        self.logger.debug("Initialized ResultManager")

        # Initialize DataLoader
        self.data_loader = DataLoader(
            config=self.config,
            output_dir=self.output_dir,
            logger=self.logger
        )
        self.logger.debug("Initialized DataLoader")

        # Initialize EDAModule
        self.eda_module = EDAModule(
            config=self.config,
            result_manager=self.result_mgr,
            logger=self.logger
        )
        self.logger.debug("Initialized EDAModule")

        # Initialize ModelTrainer
        self.model_trainer = ModelTrainer(
            config=self.config,
            result_manager=self.result_mgr,
            logger=self.logger
        )
        self.logger.debug("Initialized ModelTrainer")

        # Initialize BenchmarkModule
        self.benchmark_module = BenchmarkModule(
            config=self.config,
            result_manager=self.result_mgr,
            logger=self.logger
        )
        self.logger.debug("Initialized BenchmarkModule")

        # Initialize SHAPModule
        self.shap_module = SHAPModule(
            config=self.config,
            result_manager=self.result_mgr,
            logger=self.logger
        )
        self.logger.debug("Initialized SHAPModule")

        # Initialize ReliabilityModule
        self.reliability_module = ReliabilityModule(
            config=self.config,
            result_manager=self.result_mgr,
            logger=self.logger
        )
        self.logger.debug("Initialized ReliabilityModule")

        # Initialize LocalAnalysisModule
        self.local_analysis_module = LocalAnalysisModule(
            config=self.config,
            result_manager=self.result_mgr,
            logger=self.logger
        )
        self.logger.debug("Initialized LocalAnalysisModule")

        # Initialize ReportGenerationModule
        self.report_generation_module = ReportGenerationModule(
            config=self.config,
            result_manager=self.result_mgr,
            logger=self.logger
        )
        self.logger.debug("Initialized ReportGenerationModule")

        # Storage for loaded datasets and training results (shared across stages)
        self.loaded_datasets = None
        self.training_results = None
        self.best_models = None

    def _create_directories(self):
        """Create output subdirectories."""
        subdirs = [
            'data/preprocessed',
            'eda',
            'models',
            'shap',
            'reliability',
            'local_analyses',
            'reports',
            'checkpoints',
            'logs'
        ]

        for subdir in subdirs:
            (self.output_dir / subdir).mkdir(parents=True, exist_ok=True)

        self.logger.debug(f"Created output directories in {self.output_dir}")

    def _save_config(self):
        """Save configuration to output directory."""
        config_copy = self.output_dir / 'config_used.yaml'

        with open(config_copy, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)

        self.logger.info(f"Configuration saved to {config_copy}")

    def _load_checkpoint(self):
        """Load checkpoint to resume execution."""
        checkpoint_dir = self.output_dir / 'checkpoints'
        checkpoints = list(checkpoint_dir.glob('checkpoint_*.txt'))

        if not checkpoints:
            self.logger.warning("No checkpoints found, starting from beginning")
            return

        # Get latest checkpoint
        latest_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)

        try:
            with open(latest_checkpoint, 'r') as f:
                content = f.read()
                # Simple text format: just stage names, one per line
                self.completed_stages = set(content.strip().split('\n'))

            self.logger.info(
                f"Resumed from checkpoint: {latest_checkpoint.name}"
            )
            self.logger.info(
                f"Completed stages: {', '.join(sorted(self.completed_stages))}"
            )

        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {str(e)}")

    def _save_checkpoint(self, stage: str):
        """Save checkpoint after completing a stage."""
        self.completed_stages.add(stage)

        checkpoint_file = (
            self.output_dir / 'checkpoints' /
            f'checkpoint_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
        )

        try:
            with open(checkpoint_file, 'w') as f:
                f.write('\n'.join(sorted(self.completed_stages)))

            self.logger.debug(f"Checkpoint saved: {checkpoint_file.name}")

        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {str(e)}")

    def run(self, stages: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run the pipeline.

        Args:
            stages: Optional list of specific stages to run.
                   If None, runs all enabled stages.

        Returns:
            Dictionary with pipeline results and summary
        """
        # Determine which stages to run
        if stages:
            # Validate stage names
            invalid_stages = [s for s in stages if s not in self.STAGES]
            if invalid_stages:
                raise ValueError(
                    f"Invalid stage names: {', '.join(invalid_stages)}. "
                    f"Valid stages: {', '.join(self.STAGES)}"
                )
            stages_to_run = stages
        else:
            stages_to_run = self.STAGES

        self.logger.info(f"Pipeline will run {len(stages_to_run)} stages")

        # Execute each stage
        for stage in stages_to_run:
            # Skip if already completed (and resuming)
            if self.resume and stage in self.completed_stages:
                self.logger.info(f"Stage '{stage}' already completed, skipping")
                continue

            # Skip if disabled in config
            if not self._is_stage_enabled(stage):
                self.logger.info(f"Stage '{stage}' is disabled in config, skipping")
                continue

            # Run the stage
            self.current_stage = stage
            self._run_stage(stage)

            # Save checkpoint
            self._save_checkpoint(stage)

        # Collect and return results
        return self._collect_results()

    def _is_stage_enabled(self, stage: str) -> bool:
        """Check if a stage is enabled in configuration."""
        # 'init', 'data', 'models', 'finalize' are always enabled
        always_enabled = ['init', 'data', 'models', 'finalize']

        if stage in always_enabled:
            return True

        # Check config for optional stages
        stage_config_map = {
            'eda': self.config.get('analysis', {}).get('eda', {}).get('enabled', True),
            'shap': self.config.get('analysis', {}).get('shap', {}).get('enabled', True),
            'reliability': self.config.get('analysis', {}).get('reliability', {}).get('enabled', False),
            'local': self.config.get('analysis', {}).get('local_shap', {}).get('enabled', False),
            'report': self.config.get('report', {}).get('latex', {}).get('enabled', True),
        }

        return stage_config_map.get(stage, True)

    def _run_stage(self, stage: str):
        """
        Run a single pipeline stage.

        Args:
            stage: Stage name
        """
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info(f"Stage: {stage.upper()}")
        self.logger.info("=" * 80)

        try:
            # Map stage name to handler method
            handler_map = {
                'init': self._stage_init,
                'data': self._stage_data,
                'eda': self._stage_eda,
                'models': self._stage_models,
                'shap': self._stage_shap,
                'reliability': self._stage_reliability,
                'local': self._stage_local,
                'report': self._stage_report,
                'finalize': self._stage_finalize
            }

            handler = handler_map.get(stage)
            if handler is None:
                raise ValueError(f"No handler for stage: {stage}")

            # Execute stage
            stage_result = handler()

            # Store result
            self.results[stage] = stage_result

            self.logger.info(f"[OK] Stage '{stage}' completed successfully")

        except Exception as e:
            self.logger.error(f"[ERROR] Stage '{stage}' failed: {str(e)}", exc_info=True)
            raise

    # =========================================================================
    # STAGE HANDLERS (Placeholders for Phase 1, will implement in later phases)
    # =========================================================================

    def _stage_init(self) -> Dict[str, Any]:
        """Stage 1: Initialization."""
        self.logger.info("Initializing pipeline...")

        # Validate dependencies
        self._check_dependencies()

        return {'status': 'initialized'}

    def _check_dependencies(self):
        """Check for required dependencies."""
        # Check for pdflatex if PDF compilation is enabled
        if self.config.get('report', {}).get('latex', {}).get('compile_pdf', False):
            if not shutil.which('pdflatex'):
                self.logger.warning(
                    "pdflatex not found - PDF compilation will be skipped. "
                    "Install TeX Live or MiKTeX to enable PDF generation."
                )
                self.config['report']['latex']['compile_pdf'] = False

    def _stage_data(self) -> Dict[str, Any]:
        """Stage 2: Data loading and preprocessing."""
        self.logger.info("Loading and preprocessing data...")

        # Load all datasets from config
        self.loaded_datasets = self.data_loader.load_all_datasets()

        # Summary
        num_datasets = len(self.loaded_datasets)
        total_train_samples = sum(
            len(d['X_train']) for d in self.loaded_datasets.values()
        )
        total_test_samples = sum(
            len(d['X_test']) for d in self.loaded_datasets.values()
        )

        self.logger.info(f"Data loading complete:")
        self.logger.info(f"  Datasets: {num_datasets}")
        self.logger.info(f"  Total train samples: {total_train_samples}")
        self.logger.info(f"  Total test samples: {total_test_samples}")

        return {
            'status': 'completed',
            'num_datasets': num_datasets,
            'total_train_samples': total_train_samples,
            'total_test_samples': total_test_samples
        }

    def _stage_eda(self) -> Dict[str, Any]:
        """Stage 3: Exploratory Data Analysis."""
        self.logger.info("Running EDA...")

        if self.loaded_datasets is None:
            raise RuntimeError("Data must be loaded before running EDA")

        # Run EDA for all datasets
        eda_results = self.eda_module.run_eda_for_all_datasets(self.loaded_datasets)

        # Summary
        successful = sum(
            1 for r in eda_results.values()
            if r.get('status') == 'completed'
        )
        failed = len(eda_results) - successful

        self.logger.info(f"EDA complete: {successful} successful, {failed} failed")

        return {
            'status': 'completed',
            'successful': successful,
            'failed': failed,
            'details': eda_results
        }

    def _stage_models(self) -> Dict[str, Any]:
        """Stage 4: Model training and benchmarking."""
        self.logger.info("Training and benchmarking models...")

        if self.loaded_datasets is None:
            raise RuntimeError("Data must be loaded before training models")

        # Train all models
        self.training_results = self.model_trainer.train_all_models(self.loaded_datasets)

        # Run benchmarking
        benchmark_results = self.benchmark_module.run_benchmarking(self.training_results)

        # Get best models summary
        primary_metric = self.config.get('models', {}).get('benchmark', {}).get('primary_metric', 'AUC')
        self.best_models = self.model_trainer.get_best_models(metric=primary_metric)

        # Summary
        total_models = 0
        for dataset_results in self.training_results.values():
            if 'error' not in dataset_results:
                for group_results in dataset_results.values():
                    total_models += len(group_results)

        self.logger.info(f"Model training complete:")
        self.logger.info(f"  Total models trained: {total_models}")
        self.logger.info(f"  Best models identified: {len(self.best_models)}")

        return {
            'status': 'completed',
            'total_models': total_models,
            'training_results': self.training_results,
            'benchmark_results': benchmark_results,
            'best_models': self.best_models
        }

    def _stage_shap(self) -> Dict[str, Any]:
        """Stage 5: SHAP analysis."""
        self.logger.info("Computing SHAP values...")

        if self.loaded_datasets is None or self.best_models is None:
            raise RuntimeError("Data and models must be available before SHAP analysis")

        # Run SHAP analysis
        shap_results = self.shap_module.run_shap_analysis(
            datasets=self.loaded_datasets,
            best_models=self.best_models
        )

        # Compute feature importance
        importance_df = self.shap_module.compute_feature_importance(
            datasets=self.loaded_datasets,
            best_models=self.best_models
        )

        num_datasets = len([r for r in shap_results.get('results', {}).values() if r.get('status') == 'completed'])
        self.logger.info(f"SHAP analysis complete: {num_datasets} dataset(s)")

        return {
            'status': shap_results.get('status', 'completed'),
            'results': shap_results.get('results', {}),
            'feature_importance': importance_df
        }

    def _stage_reliability(self) -> Dict[str, Any]:
        """Stage 6: Reliability testing."""
        self.logger.info("Running reliability tests...")

        if self.loaded_datasets is None or self.best_models is None:
            raise RuntimeError("Data and models must be available before reliability testing")

        # Run reliability testing
        reliability_results = self.reliability_module.run_reliability_testing(
            datasets=self.loaded_datasets,
            best_models=self.best_models
        )

        num_tested = len([r for r in reliability_results.get('results', {}).values() if r.get('status') == 'completed'])
        self.logger.info(f"Reliability testing complete: {num_tested} model(s)")

        return reliability_results

    def _stage_local(self) -> Dict[str, Any]:
        """Stage 7: Local instance analysis."""
        self.logger.info("Analyzing local instances...")

        if self.loaded_datasets is None or self.best_models is None:
            raise RuntimeError("Data and models must be available before local analysis")

        # Run local analysis
        local_results = self.local_analysis_module.run_local_analysis(
            datasets=self.loaded_datasets,
            best_models=self.best_models
        )

        total_instances = sum(
            r.get('num_instances', 0)
            for r in local_results.get('results', {}).values()
            if isinstance(r, dict) and r.get('status') == 'completed'
        )
        self.logger.info(f"Local analysis complete: {total_instances} instance(s)")

        return local_results

    def _stage_report(self) -> Dict[str, Any]:
        """Stage 8: Report generation."""
        self.logger.info("Generating reports...")

        # Generate LaTeX report
        report_results = self.report_generation_module.generate_report(
            pipeline_results=self.results
        )

        # Generate text summary
        summary_path = self.report_generation_module.generate_summary_report(
            pipeline_results=self.results
        )

        if report_results.get('status') == 'completed':
            self.logger.info(f"Report generation complete:")
            self.logger.info(f"  LaTeX: {report_results.get('latex_path')}")
            if report_results.get('pdf_path'):
                self.logger.info(f"  PDF: {report_results.get('pdf_path')}")
            if summary_path:
                self.logger.info(f"  Summary: {summary_path}")

        return report_results

    def _stage_finalize(self) -> Dict[str, Any]:
        """Stage 9: Finalization."""
        self.logger.info("Finalizing pipeline...")

        # Generate summary
        summary = self._generate_summary()

        # Save summary to file
        summary_file = self.output_dir / 'summary.txt'
        with open(summary_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("Model Selection Pipeline - Execution Summary\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Output Directory: {self.output_dir}\n\n")
            f.write("Completed Stages:\n")
            for stage in sorted(self.completed_stages):
                f.write(f"  [OK] {stage}\n")
            f.write("\n")
            f.write("=" * 80 + "\n")

        self.logger.info(f"Summary saved to {summary_file}")

        return summary

    def _generate_summary(self) -> Dict[str, Any]:
        """Generate pipeline execution summary."""
        return {
            'total_stages': len(self.STAGES),
            'completed_stages': len(self.completed_stages),
            'output_directory': str(self.output_dir),
            'timestamp': datetime.now().isoformat()
        }

    def _collect_results(self) -> Dict[str, Any]:
        """Collect and format final results."""
        return {
            'results': self.results,
            'summary': self._generate_summary(),
            'output_dir': str(self.output_dir),
            'report_pdf': None  # Will be set in Phase 5
        }
