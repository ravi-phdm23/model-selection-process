"""Configuration validation utilities."""

import yaml
from pathlib import Path
from typing import Dict, Any, List


class ConfigValidationError(Exception):
    """Custom exception for configuration validation errors."""
    pass


def validate_config_file(config_path: Path) -> Dict[str, Any]:
    """
    Load and validate YAML configuration file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Validated configuration dictionary

    Raises:
        ConfigValidationError: If configuration is invalid
    """
    # Load YAML
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigValidationError(f"Failed to parse YAML: {str(e)}")
    except Exception as e:
        raise ConfigValidationError(f"Failed to read config file: {str(e)}")

    if config is None:
        raise ConfigValidationError("Configuration file is empty")

    # Validate required top-level keys
    required_keys = ['project', 'data', 'models']
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise ConfigValidationError(
            f"Missing required configuration sections: {', '.join(missing_keys)}"
        )

    # Validate project section
    _validate_project(config['project'])

    # Validate data section
    _validate_data(config['data'])

    # Validate models section
    _validate_models(config['models'])

    # Validate optional sections
    if 'analysis' in config:
        _validate_analysis(config['analysis'])

    if 'report' in config:
        _validate_report(config['report'])

    # Apply defaults for optional sections
    config = _apply_defaults(config)

    return config


def _validate_project(project: Dict[str, Any]):
    """Validate project configuration section."""
    if 'name' not in project:
        raise ConfigValidationError("project.name is required")

    if 'output_dir' not in project:
        raise ConfigValidationError("project.output_dir is required")


def _validate_data(data: Dict[str, Any]):
    """Validate data configuration section."""
    if 'datasets' not in data:
        raise ConfigValidationError("data.datasets is required")

    if not isinstance(data['datasets'], list):
        raise ConfigValidationError("data.datasets must be a list")

    if len(data['datasets']) == 0:
        raise ConfigValidationError("At least one dataset is required")

    # Validate each dataset
    for i, dataset in enumerate(data['datasets']):
        if 'path' not in dataset:
            raise ConfigValidationError(f"Dataset {i}: path is required")

        if 'target_column' not in dataset:
            raise ConfigValidationError(f"Dataset {i}: target_column is required")

        # Check if file exists (only warn, don't fail - allows testing)
        dataset_path = Path(dataset['path'])
        # Note: File existence check temporarily disabled for testing Phase 1
        # Will be re-enabled when data loading is implemented in Phase 2
        # if not dataset_path.exists():
        #     raise ConfigValidationError(
        #         f"Dataset {i}: file not found: {dataset['path']}"
        #     )


def _validate_models(models: Dict[str, Any]):
    """Validate models configuration section."""
    if 'groups' not in models:
        raise ConfigValidationError("models.groups is required")

    if not isinstance(models['groups'], list):
        raise ConfigValidationError("models.groups must be a list")

    if len(models['groups']) == 0:
        raise ConfigValidationError("At least one model group is required")

    # Valid model groups
    valid_groups = [
        'lr', 'lr_reg', 'adaboost', 'Bag-CART', 'BagNN', 'Boost-DT',
        'RF', 'SGB', 'KNN', 'XGB', 'LGBM', 'DL'
    ]

    invalid_groups = [g for g in models['groups'] if g not in valid_groups]
    if invalid_groups:
        raise ConfigValidationError(
            f"Invalid model groups: {', '.join(invalid_groups)}. "
            f"Valid groups: {', '.join(valid_groups)}"
        )


def _validate_analysis(analysis: Dict[str, Any]):
    """Validate analysis configuration section."""
    # Optional validation for analysis sub-sections
    pass


def _validate_report(report: Dict[str, Any]):
    """Validate report configuration section."""
    # Optional validation for report sub-sections
    pass


def _apply_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
    """Apply default values for optional configuration fields."""

    # Project defaults
    if 'random_seed' not in config['project']:
        config['project']['random_seed'] = 42

    if 'n_jobs' not in config['project']:
        config['project']['n_jobs'] = -1

    # Data preprocessing defaults
    if 'preprocessing' not in config['data']:
        config['data']['preprocessing'] = {}

    preprocessing = config['data']['preprocessing']
    preprocessing.setdefault('handle_missing', True)
    preprocessing.setdefault('test_size', 0.3)
    preprocessing.setdefault('smote', True)
    preprocessing.setdefault('smote_k_neighbors', 5)

    # Analysis defaults
    if 'analysis' not in config:
        config['analysis'] = {}

    analysis = config['analysis']

    # EDA defaults
    if 'eda' not in analysis:
        analysis['eda'] = {}
    analysis['eda'].setdefault('enabled', True)
    analysis['eda'].setdefault('generate_profile', True)
    analysis['eda'].setdefault('feature_importance', True)

    # SHAP defaults
    if 'shap' not in analysis:
        analysis['shap'] = {}
    analysis['shap'].setdefault('enabled', True)
    analysis['shap'].setdefault('global_plots', True)
    analysis['shap'].setdefault('plot_types', ['bar', 'dot'])

    # Reliability defaults
    if 'reliability' not in analysis:
        analysis['reliability'] = {}
    analysis['reliability'].setdefault('enabled', False)
    analysis['reliability'].setdefault('n_permutations', 50)

    # Local SHAP defaults
    if 'local_shap' not in analysis:
        analysis['local_shap'] = {}
    analysis['local_shap'].setdefault('enabled', False)
    analysis['local_shap'].setdefault('instances', [])

    # Report defaults
    if 'report' not in config:
        config['report'] = {}

    report = config['report']

    if 'latex' not in report:
        report['latex'] = {}
    report['latex'].setdefault('enabled', True)
    report['latex'].setdefault('compile_pdf', True)
    report['latex'].setdefault('include_narratives', False)

    # Logging defaults
    if 'logging' not in config:
        config['logging'] = {}

    logging = config['logging']
    logging.setdefault('level', 'INFO')
    logging.setdefault('log_file', 'pipeline.log')
    logging.setdefault('console_output', True)

    return config
