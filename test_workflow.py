"""
Automated Test Workflow for Model Selection Process

This script automates the entire workflow from data upload to local SHAP analysis
using configuration from test_config.yaml.

Usage:
    python test_workflow.py
    python test_workflow.py --config custom_config.yaml
"""

import yaml
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import argparse
from io import StringIO
import warnings
warnings.filterwarnings('ignore')

# Import required modules from the project
try:
    from models import MODELS
    from metrics import calculate_metrics
    from shap_analysis import (
        shap_rank_stability, 
        model_randomization_sanity, 
        summarize_reliability,
        get_local_shap_explanation
    )
    from llm_explain import get_llm_explanation
    print("✓ Successfully imported project modules")
except ImportError as e:
    print(f"✗ Error importing modules: {e}")
    print("Make sure you're running this from the project root directory")
    sys.exit(1)

# Import scikit-learn and other dependencies
try:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, f1_score, recall_score
    import shap
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    from dotenv import load_dotenv
    import os
    print("✓ Successfully imported dependencies")
except ImportError as e:
    print(f"✗ Error importing dependencies: {e}")
    sys.exit(1)


class WorkflowRunner:
    """Manages the entire test workflow"""
    
    def __init__(self, config_path: str = "test_config.yaml"):
        """Initialize the workflow runner with configuration"""
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.results = {}
        self.benchmark_results = []
        self.global_shap_dfs = {}
        self.reliability_results = {}
        self.reliability_ratios = {}
        
        # Setup output directory
        self.output_dir = Path(self.config['output']['results_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir = self.output_dir / "figures"
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"✓ Initialized workflow with config: {config_path}")
        print(f"✓ Output directory: {self.output_dir}")
    
    def _load_config(self) -> dict:
        """Load configuration from YAML file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def step_1_load_datasets(self):
        """Step 1: Load datasets"""
        print("\n" + "="*60)
        print("STEP 1: Loading Datasets")
        print("="*60)
        
        self.datasets = {}
        self.target_column = self.config['target_column']
        
        for ds_config in self.config['datasets']:
            ds_path = Path(ds_config['path'])
            ds_name = ds_config['name']
            
            if not ds_path.exists():
                print(f"✗ Dataset not found: {ds_path}")
                continue
            
            try:
                df = pd.read_csv(ds_path)
                self.datasets[ds_name] = df
                print(f"✓ Loaded {ds_name}: {df.shape[0]} rows, {df.shape[1]} columns")
                
                # Verify target column exists
                if self.target_column not in df.columns:
                    print(f"  ⚠ Warning: Target column '{self.target_column}' not found in {ds_name}")
                    print(f"  Available columns: {', '.join(df.columns.tolist())}")
                else:
                    print(f"  Target column: {self.target_column}")
                    print(f"  Class distribution: {df[self.target_column].value_counts().to_dict()}")
                
            except Exception as e:
                print(f"✗ Error loading {ds_name}: {e}")
        
        if not self.datasets:
            raise ValueError("No datasets loaded successfully")
        
        print(f"\n✓ Step 1 Complete: {len(self.datasets)} dataset(s) loaded")
        return True
    
    def step_2_select_models(self):
        """Step 2: Select model groups"""
        print("\n" + "="*60)
        print("STEP 2: Selecting Models")
        print("="*60)
        
        self.selected_groups = self.config['model_groups']
        self.selected_models = []
        
        for group in self.selected_groups:
            if group in MODELS:
                model_names = list(MODELS[group].keys())
                self.selected_models.extend(model_names)
                print(f"✓ Group '{group}': {len(model_names)} models")
                for name in model_names:
                    print(f"  - {name}")
            else:
                print(f"✗ Unknown model group: {group}")
        
        print(f"\n✓ Step 2 Complete: {len(self.selected_models)} models selected")
        return True
    
    def step_3_run_experiment(self):
        """Step 3: Run experiment (train and evaluate models)"""
        print("\n" + "="*60)
        print("STEP 3: Running Experiment")
        print("="*60)
        
        for ds_name, df in self.datasets.items():
            print(f"\n--- Processing dataset: {ds_name} ---")
            
            if self.target_column not in df.columns:
                print(f"✗ Skipping {ds_name}: target column not found")
                continue
            
            # Split features and target
            X = df.drop(columns=[self.target_column])
            y = df[self.target_column]
            
            # Train-test split
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, 
                    stratify=y if y.nunique() > 1 else None
                )
                print(f"✓ Train-test split: {len(X_train)} train, {len(X_test)} test")
            except Exception as e:
                print(f"✗ Error in train-test split: {e}")
                continue
            
            # Initialize results structure
            self.results[ds_name] = {
                'models': {},
                'data': {
                    'X_train': X_train,
                    'X_test': X_test,
                    'y_train': y_train,
                    'y_test': y_test
                }
            }
            
            # Train models
            for group in self.selected_groups:
                if group not in MODELS:
                    continue
                
                print(f"\n  Model Group: {group}")
                self.results[ds_name]['models'][group] = {}
                
                for model_name, model_builder in MODELS[group].items():
                    try:
                        # Build and train model
                        model = model_builder()
                        model.fit(X_train, y_train)
                        
                        # Evaluate
                        y_pred = model.predict(X_test)
                        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
                        
                        metrics = calculate_metrics(y_test, y_pred, y_proba)
                        
                        # Store model
                        self.results[ds_name]['models'][group][model_name] = model
                        
                        print(f"    ✓ {model_name}: AUC={metrics.get('AUC', 0):.4f}, F1={metrics.get('F1', 0):.4f}")
                        
                    except Exception as e:
                        print(f"    ✗ {model_name}: {str(e)[:100]}")
        
        print(f"\n✓ Step 3 Complete: Models trained on {len(self.results)} dataset(s)")
        return True
    
    def step_4_benchmark_analysis(self):
        """Step 4: Find benchmark models (best performing)"""
        print("\n" + "="*60)
        print("STEP 4: Benchmark Analysis")
        print("="*60)
        
        # Collect all model scores
        model_scores = {}  # {group: {model: [auc1, auc2, ...]}}
        
        for ds_name, ds_results in self.results.items():
            for group, models in ds_results['models'].items():
                if group not in model_scores:
                    model_scores[group] = {}
                
                for model_name, model in models.items():
                    if model is None:
                        continue
                    
                    try:
                        X_test = ds_results['data']['X_test']
                        y_test = ds_results['data']['y_test']
                        
                        y_pred = model.predict(X_test)
                        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
                        
                        metrics = calculate_metrics(y_test, y_pred, y_proba)
                        auc = metrics.get('AUC', 0)
                        
                        if model_name not in model_scores[group]:
                            model_scores[group][model_name] = []
                        model_scores[group][model_name].append(auc)
                    except Exception as e:
                        print(f"✗ Error evaluating {model_name} on {ds_name}: {e}")
        
        # Find best model per group
        benchmark_models = {}
        for group, models in model_scores.items():
            avg_scores = {name: np.mean(scores) for name, scores in models.items()}
            best_model = max(avg_scores, key=avg_scores.get)
            benchmark_models[group] = best_model
            print(f"✓ {group}: {best_model} (avg AUC: {avg_scores[best_model]:.4f})")
        
        # Build benchmark results table
        for ds_name, ds_results in self.results.items():
            for group, benchmark_name in benchmark_models.items():
                if group not in ds_results['models']:
                    continue
                
                model = ds_results['models'][group].get(benchmark_name)
                if model is None:
                    continue
                
                try:
                    X_test = ds_results['data']['X_test']
                    y_test = ds_results['data']['y_test']
                    
                    y_pred = model.predict(X_test)
                    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
                    
                    metrics = calculate_metrics(y_test, y_pred, y_proba)
                    
                    self.benchmark_results.append({
                        'Dataset': ds_name,
                        'Model Group': group,
                        'Benchmark Model': benchmark_name,
                        **metrics
                    })
                except Exception as e:
                    print(f"✗ Error: {e}")
        
        # Save benchmark results
        if self.benchmark_results:
            df_bench = pd.DataFrame(self.benchmark_results)
            output_path = self.output_dir / "benchmark_results.csv"
            df_bench.to_csv(output_path, index=False)
            print(f"\n✓ Benchmark results saved: {output_path}")
            print(f"\n{df_bench.to_string()}")
        
        print(f"\n✓ Step 4 Complete: {len(benchmark_models)} benchmark models identified")
        return True
    
    def step_5_global_shap(self):
        """Step 5: Global SHAP analysis"""
        if not self.config['global_shap']['enabled']:
            print("\n⊘ Step 5 Skipped: Global SHAP disabled in config")
            return False
        
        print("\n" + "="*60)
        print("STEP 5: Global SHAP Analysis")
        print("="*60)
        
        use_stable = self.config['global_shap']['use_stable_shap']
        trials = self.config['global_shap']['stable_shap_trials']
        bg_size = self.config['global_shap']['stable_shap_bg_size']
        
        print(f"Mode: {'Stable SHAP' if use_stable else 'Standard SHAP'}")
        if use_stable:
            print(f"Trials: {trials}, Background size: {bg_size}")
        
        # Find benchmark model for each dataset
        benchmark_map = {}
        for result in self.benchmark_results:
            ds = result['Dataset']
            if ds not in benchmark_map:
                # Pick first benchmark for this dataset
                benchmark_map[ds] = {
                    'group': result['Model Group'],
                    'name': result['Benchmark Model']
                }
        
        for ds_name, bench_info in benchmark_map.items():
            print(f"\n--- {ds_name} ---")
            print(f"  Model: {bench_info['name']}")
            
            try:
                model = self.results[ds_name]['models'][bench_info['group']][bench_info['name']]
                X_train = self.results[ds_name]['data']['X_train']
                X_test = self.results[ds_name]['data']['X_test']
                
                if use_stable:
                    # Stable SHAP with rank stability
                    rank_df = shap_rank_stability(
                        model, X_train, X_test,
                        n_bg_samples=bg_size,
                        n_trials=trials
                    )
                    self.global_shap_dfs[ds_name] = rank_df
                    
                    # Save to CSV
                    safe_name = ds_name.replace(' ', '_').replace('.csv', '')
                    csv_path = self.output_dir / f"shap_stability_{safe_name}.csv"
                    rank_df.to_csv(csv_path, index=False)
                    print(f"  ✓ Saved rank stability: {csv_path}")
                    
                    # Generate plots
                    top_features = rank_df.nsmallest(10, 'avg_rank')
                    print(f"  ✓ Top 10 features by avg_rank:")
                    for _, row in top_features.iterrows():
                        print(f"    - {row['feature']}: avg_rank={row['avg_rank']:.2f}, std_rank={row['std_rank']:.2f}")
                else:
                    # Standard single-shot SHAP
                    print("  Standard SHAP (single-shot) - not implemented in this test script")
                
            except Exception as e:
                print(f"  ✗ Error: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"\n✓ Step 5 Complete: Global SHAP for {len(self.global_shap_dfs)} dataset(s)")
        return True
    
    def step_5_5_reliability_test(self):
        """Step 5.5: Reliability test (rank stability + sanity check)"""
        if not self.config['reliability_test']['enabled']:
            print("\n⊘ Step 5.5 Skipped: Reliability test disabled in config")
            return False
        
        print("\n" + "="*60)
        print("STEP 5.5: Reliability Test")
        print("="*60)
        
        n_trials = self.config['reliability_test']['n_trials']
        n_bg = self.config['reliability_test']['n_bg']
        
        print(f"Configuration: {n_trials} trials, {n_bg} background samples")
        
        # Find benchmark model for each dataset
        benchmark_map = {}
        for result in self.benchmark_results:
            ds = result['Dataset']
            if ds not in benchmark_map:
                benchmark_map[ds] = {
                    'group': result['Model Group'],
                    'name': result['Benchmark Model']
                }
        
        for ds_name, bench_info in benchmark_map.items():
            print(f"\n--- {ds_name} ---")
            
            try:
                model = self.results[ds_name]['models'][bench_info['group']][bench_info['name']]
                X_train = self.results[ds_name]['data']['X_train']
                X_test = self.results[ds_name]['data']['X_test']
                y_train = self.results[ds_name]['data']['y_train']
                
                # Rank stability
                print("  Running rank stability analysis...")
                rank_df = shap_rank_stability(model, X_train, X_test, n_bg_samples=n_bg, n_trials=n_trials)
                
                # Sanity check
                print("  Running randomization sanity check...")
                sanity_ratio = model_randomization_sanity(model, X_train, X_test, y_train, n_bg_samples=n_bg)
                
                # Summary
                summary_text = summarize_reliability(rank_df, sanity_ratio, n_trials, n_bg)
                
                # Store results
                safe_name = ds_name.replace(' ', '_').replace('.csv', '')
                self.reliability_results[safe_name] = rank_df
                self.reliability_ratios[safe_name] = sanity_ratio
                
                # Save outputs
                csv_path = self.output_dir / f"reliability_{safe_name}.csv"
                txt_path = self.output_dir / f"reliability_{safe_name}.txt"
                
                rank_df.to_csv(csv_path, index=False)
                with open(txt_path, 'w') as f:
                    f.write(f"Dataset: {ds_name}\n")
                    f.write(f"Sanity Ratio: {sanity_ratio:.3f}\n\n")
                    f.write(summary_text)
                
                print(f"  ✓ Sanity ratio: {sanity_ratio:.3f}")
                print(f"  ✓ Saved: {csv_path}")
                print(f"  ✓ Saved: {txt_path}")
                
            except Exception as e:
                print(f"  ✗ Error: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"\n✓ Step 5.5 Complete: Reliability test for {len(self.reliability_results)} dataset(s)")
        return True
    
    def step_6_local_analysis(self):
        """Step 6: Local SHAP analysis for a single row"""
        if not self.config['local_analysis']['enabled']:
            print("\n⊘ Step 6 Skipped: Local analysis disabled in config")
            return False
        
        print("\n" + "="*60)
        print("STEP 6: Local SHAP Analysis")
        print("="*60)
        
        ds_name = self.config['local_analysis']['dataset']
        row_index = self.config['local_analysis']['row_index']
        
        print(f"Dataset: {ds_name}")
        print(f"Row index: {row_index}")
        
        # Check if dataset exists
        if ds_name not in self.datasets:
            print(f"✗ Dataset '{ds_name}' not found in loaded datasets")
            return False
        
        df = self.datasets[ds_name]
        if row_index >= len(df):
            print(f"✗ Row index {row_index} out of range (max: {len(df)-1})")
            return False
        
        # Find benchmark model
        bench_info = None
        for result in self.benchmark_results:
            if result['Dataset'] == ds_name:
                bench_info = {
                    'group': result['Model Group'],
                    'name': result['Benchmark Model']
                }
                break
        
        if not bench_info:
            print(f"✗ No benchmark model found for {ds_name}")
            return False
        
        print(f"Model: {bench_info['name']}")
        
        try:
            # Get model and data
            model = self.results[ds_name]['models'][bench_info['group']][bench_info['name']]
            X_train = self.results[ds_name]['data']['X_train']
            
            # Get instance
            instance = df.iloc[[row_index]]
            instance_features = instance.drop(columns=[self.target_column])
            actual_target = instance[self.target_column].values[0]
            
            # Get prediction
            pred_proba = model.predict_proba(instance_features)[0]
            prob_class_1 = pred_proba[1]
            
            print(f"\n--- Analysis Results ---")
            print(f"Actual target: {actual_target}")
            print(f"Predicted probability (Class 1): {prob_class_1:.4f}")
            
            # Get SHAP explanation
            print("\nGenerating SHAP explanation...")
            explanation = get_local_shap_explanation(model, X_train, instance_features)
            
            # Create waterfall plot
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.waterfall_plot(explanation, max_display=10, show=False)
            
            safe_ds = ds_name.replace(' ', '_').replace('.csv', '')
            safe_model = bench_info['name'].replace(' ', '_').replace('/', '_')
            wf_path = self.figures_dir / f"shap_{safe_ds}_{safe_model}_waterfall_row{row_index}.png"
            fig.savefig(wf_path, bbox_inches='tight', dpi=150)
            plt.close(fig)
            print(f"✓ Waterfall plot saved: {wf_path}")
            
            # Get feature reliability if available
            feature_reliability = None
            safe_name = ds_name.replace(' ', '_').replace('.csv', '')
            
            if safe_name in self.reliability_results:
                stab_df = self.reliability_results[safe_name]
                sanity_ratio = self.reliability_ratios.get(safe_name)
                
                feature_reliability = {}
                for feat in explanation.feature_names[:10]:
                    if feat in stab_df['feature'].values:
                        feat_row = stab_df[stab_df['feature'] == feat].iloc[0]
                        feature_reliability[feat] = {
                            'avg_rank': feat_row['avg_rank'],
                            'std_rank': feat_row['std_rank']
                        }
                
                if sanity_ratio is not None:
                    feature_reliability['_sanity_ratio'] = sanity_ratio
                
                print(f"✓ Feature reliability data loaded (sanity ratio: {sanity_ratio:.3f})")
            else:
                print("⚠ Feature reliability data not available (Step 5.5 may not have run)")
            
            # Get AI explanation
            print("\nGenerating AI explanation...")
            load_dotenv()  # Load API key
            commentary, error, reliability_metrics = get_llm_explanation(
                explanation,
                actual_target,
                prob_class_1,
                feature_reliability=feature_reliability
            )
            
            if error:
                print(f"✗ AI explanation error: {error}")
            else:
                print("\n--- AI Generated Explanation ---")
                print(commentary)
                
                if reliability_metrics:
                    print("\n--- Reliability Metrics ---")
                    print(f"Reliability Score: {reliability_metrics['reliability_score']:.3f}")
                    print(f"Classification: {reliability_metrics['reliability_bucket']}")
                    print(f"Sanity Ratio: {reliability_metrics['sanity_ratio']:.3f}")
                    print(f"Explanatory Robustness: {reliability_metrics['explanatory_robustness']:.3f}")
                
                # Save explanation
                txt_path = self.output_dir / f"explanation_{safe_ds}_row{row_index}.txt"
                with open(txt_path, 'w') as f:
                    f.write(f"Dataset: {ds_name}\n")
                    f.write(f"Row Index: {row_index}\n")
                    f.write(f"Model: {bench_info['name']}\n")
                    f.write(f"Actual Target: {actual_target}\n")
                    f.write(f"Predicted Probability (Class 1): {prob_class_1:.4f}\n")
                    f.write("\n" + "="*60 + "\n")
                    f.write("AI EXPLANATION\n")
                    f.write("="*60 + "\n\n")
                    f.write(commentary)
                    
                    if reliability_metrics:
                        f.write("\n\n" + "="*60 + "\n")
                        f.write("RELIABILITY METRICS\n")
                        f.write("="*60 + "\n")
                        f.write(f"Reliability Score: {reliability_metrics['reliability_score']:.3f}\n")
                        f.write(f"Classification: {reliability_metrics['reliability_bucket']}\n")
                        f.write(f"Sanity Ratio: {reliability_metrics['sanity_ratio']:.3f}\n")
                        f.write(f"Explanatory Robustness: {reliability_metrics['explanatory_robustness']:.3f}\n")
                
                print(f"\n✓ Explanation saved: {txt_path}")
            
        except Exception as e:
            print(f"✗ Error in local analysis: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        print(f"\n✓ Step 6 Complete: Local analysis for row {row_index}")
        return True
    
    def run_full_workflow(self):
        """Run the complete workflow"""
        print("\n" + "╔" + "="*58 + "╗")
        print("║" + " "*15 + "AUTOMATED TEST WORKFLOW" + " "*20 + "║")
        print("╚" + "="*58 + "╝")
        
        steps = [
            ("Step 1: Load Datasets", self.step_1_load_datasets),
            ("Step 2: Select Models", self.step_2_select_models),
            ("Step 3: Run Experiment", self.step_3_run_experiment),
            ("Step 4: Benchmark Analysis", self.step_4_benchmark_analysis),
            ("Step 5: Global SHAP", self.step_5_global_shap),
            ("Step 5.5: Reliability Test", self.step_5_5_reliability_test),
            ("Step 6: Local Analysis", self.step_6_local_analysis),
        ]
        
        results = {}
        for step_name, step_func in steps:
            try:
                success = step_func()
                results[step_name] = "✓ Passed" if success else "⊘ Skipped"
            except Exception as e:
                results[step_name] = f"✗ Failed: {str(e)[:50]}"
                print(f"\n✗ {step_name} failed: {e}")
                import traceback
                traceback.print_exc()
                
                # Ask if we should continue
                print("\nContinue with remaining steps? (y/n): ", end='')
                try:
                    response = input().strip().lower()
                    if response != 'y':
                        break
                except:
                    break
        
        # Summary
        print("\n" + "="*60)
        print("WORKFLOW SUMMARY")
        print("="*60)
        for step_name, result in results.items():
            print(f"{step_name}: {result}")
        
        print(f"\nResults saved to: {self.output_dir.absolute()}")
        print("="*60)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Run automated test workflow')
    parser.add_argument('--config', default='test_config.yaml', 
                       help='Path to configuration YAML file')
    args = parser.parse_args()
    
    try:
        runner = WorkflowRunner(config_path=args.config)
        runner.run_full_workflow()
    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
