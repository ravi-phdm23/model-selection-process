#!/usr/bin/env python3
"""
Model Selection Pipeline - CLI Entry Point

Run the complete model selection and analysis workflow from a YAML configuration file.

Usage:
    python model_pipeline.py --config config.yaml
    python model_pipeline.py --config config.yaml --resume
    python model_pipeline.py --config config.yaml --stages eda,models,report
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from pipeline.orchestrator import PipelineOrchestrator
from utils.logger import setup_logger
from utils.validators import validate_config_file


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Model Selection Pipeline - Automated ML workflow',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline
  python model_pipeline.py --config config.yaml

  # Resume from checkpoint
  python model_pipeline.py --config config.yaml --resume

  # Run specific stages only
  python model_pipeline.py --config config.yaml --stages eda,models,report

  # Dry run (validate config without execution)
  python model_pipeline.py --config config.yaml --dry-run

For more information, see README_CLI.md
        """
    )

    parser.add_argument(
        '--config', '-c',
        type=str,
        required=True,
        help='Path to YAML configuration file'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Override output directory (overrides config)'
    )

    parser.add_argument(
        '--resume', '-r',
        action='store_true',
        help='Resume from last checkpoint'
    )

    parser.add_argument(
        '--stages', '-s',
        type=str,
        help='Comma-separated stages to run (e.g., "eda,models,report")'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Validate configuration without running pipeline'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging (DEBUG level)'
    )

    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Minimal output (WARNING level only)'
    )

    parser.add_argument(
        '--no-pdf',
        action='store_true',
        help='Skip PDF compilation (LaTeX only)'
    )

    parser.add_argument(
        '--parallel',
        type=int,
        help='Max parallel jobs (overrides config, -1 for all cores)'
    )

    return parser.parse_args()


def main():
    """Main entry point for the pipeline."""
    print("=" * 80)
    print("Model Selection Pipeline - CLI")
    print("=" * 80)
    print()

    # Parse arguments
    args = parse_arguments()

    # Validate config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[ERROR] Configuration file not found: {args.config}")
        sys.exit(1)

    print(f"Loading configuration: {config_path}")

    # Validate configuration
    try:
        config = validate_config_file(config_path)
        print("[OK] Configuration validated successfully")
    except Exception as e:
        print(f"[ERROR] Configuration validation failed:")
        print(f"   {str(e)}")
        sys.exit(1)

    # Apply command-line overrides
    if args.output:
        config['project']['output_dir'] = args.output
    if args.parallel is not None:
        config['project']['n_jobs'] = args.parallel
    if args.no_pdf:
        config['report']['latex']['compile_pdf'] = False

    # Determine log level
    if args.verbose:
        log_level = 'DEBUG'
    elif args.quiet:
        log_level = 'WARNING'
    else:
        log_level = config.get('logging', {}).get('level', 'INFO')

    # Setup output directory
    output_dir = Path(config['project']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    log_file = output_dir / config.get('logging', {}).get('log_file', 'pipeline.log')
    logger = setup_logger(
        log_file=log_file,
        level=log_level,
        console_output=config.get('logging', {}).get('console_output', True)
    )

    logger.info("=" * 80)
    logger.info("Model Selection Pipeline Started")
    logger.info("=" * 80)
    logger.info(f"Configuration: {config_path}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)

    # Dry run mode
    if args.dry_run:
        print("\n[OK] Dry run completed successfully")
        print("   Configuration is valid and ready to run")
        logger.info("Dry run completed - configuration valid")
        return 0

    # Parse stages if specified
    stages_to_run = None
    if args.stages:
        stages_to_run = [s.strip() for s in args.stages.split(',')]
        logger.info(f"Running specific stages: {stages_to_run}")

    # Initialize orchestrator
    try:
        orchestrator = PipelineOrchestrator(
            config=config,
            output_dir=output_dir,
            logger=logger,
            resume=args.resume
        )
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {str(e)}", exc_info=True)
        print(f"\n[ERROR] Failed to initialize pipeline: {str(e)}")
        return 1

    # Run pipeline
    try:
        print("\n[RUNNING] Starting pipeline execution...\n")
        results = orchestrator.run(stages=stages_to_run)

        print("\n" + "=" * 80)
        print("[SUCCESS] Pipeline completed successfully!")
        print("=" * 80)
        print(f"\nResults saved to: {output_dir}")

        # Print summary
        if results.get('summary'):
            print("\nSummary:")
            for key, value in results['summary'].items():
                print(f"   {key}: {value}")

        # Print report location
        if results.get('report_pdf'):
            print(f"\nFinal report: {results['report_pdf']}")

        logger.info("=" * 80)
        logger.info("Pipeline completed successfully")
        logger.info("=" * 80)

        return 0

    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        logger.warning("Pipeline interrupted by user (Ctrl+C)")
        print(f"   Resume with: python model_pipeline.py --config {args.config} --resume")
        return 130

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        print(f"\n[ERROR] Pipeline failed: {str(e)}")
        print(f"   Check log file for details: {log_file}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
