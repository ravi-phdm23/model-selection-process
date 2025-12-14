# Copilot instructions — Model Selection Pipeline

This repository is an automated model selection + analysis pipeline with a Streamlit UI. Keep guidance short and specific so an AI agent can be productive immediately.

**Overview**
- **Purpose:** Runs data ingestion → preprocessing → model training → evaluation → reports (PDF/HTML). The CLI entry point is [model_pipeline.py](model_pipeline.py). The interactive UI is [streamlit_app.py](streamlit_app.py).
- **Key outputs:** per-run folder under `project.output_dir` (configured in YAML). Look in outputs like `output_*/reports`, `output_*/models`, and `output_*/results`.

**How to run (developer workflows)**
- **CLI pipeline:** `python model_pipeline.py --config config/<yaml>` (supports `--resume`, `--dry-run`, `--stages`). See [README_CLI.md](README_CLI.md).
- **Streamlit UI:** `streamlit run streamlit_app.py` (app uses `MODELS` from [models.py](models.py) and `run_experiment()` for execution).
- **Unit tests / quick checks:** run tests under `test_*` files (no single test framework enforced; run with `pytest` if installed).

**Architecture & big-picture notes**
- The pipeline is organized as: CLI/orchestrator → `pipeline/` orchestrator modules → workers that call helpers in `models.py`, `metrics.py`, `shap_analysis.py`, `report_manager.py`.
- `models.py` exposes a `MODELS` registry (dict of groups → builder functions) and a `run_experiment()` helper used by the Streamlit UI. Add new model builders here and register them in `MODELS`.
- Deep/optional models are loaded conditionally: `deep_models.py`, XGBoost (`xgboost`), LightGBM (`lightgbm`), and imbalanced-learn (`imblearn`) are optional and guarded with try/except. If you add features that require these libs, also update `requirements.txt`.
- SHAP and LLM-based explanations live in `shap_analysis.py` and `llm_explain.py`; the UI imports `get_llm_explanation` and SHAP helpers if available.

**Repo-specific conventions & patterns**
- Feature engineering/preprocessing lives in `models.py`'s `_preprocessor()` and is wrapped into sklearn Pipelines. Builders return full pipelines (preprocessor + calibrated classifier).
- SMOTE is applied in-pipeline using `imblearn.pipeline.Pipeline` to avoid CV leakage. Follow the same pattern: never apply SMOTE outside fold/pipeline.
- Calibration: many classifiers are wrapped with `CalibratedClassifierCV(...)` — maintain consistency when adding probabilistic models.
- Results structure: `run_experiment()` returns nested dicts with keys `metrics`, `models`, `data`, and `error`. Tests, UI, and report generators expect this shape.

**Editing guidance (practical examples)**
- To add a model: implement a builder function that returns a sklearn-style pipeline, then register it under `MODELS` in [models.py](models.py). Example builders: `build_lr_lbfgs()` and `make_rf(...)`.
- To enable optional dependency features, add safe try/except import and a graceful fallback message (see how `models.py` handles `xgboost`, `lightgbm`, and `imblearn`).
- To change CLI behavior, edit [model_pipeline.py](model_pipeline.py) or the orchestrator in `pipeline/orchestrator.py` (imported by the CLI).

**Dependencies & integration points**
- Primary runtime: `streamlit`, `pandas`, `numpy`, `scikit-learn`. Optional (enable only if needed): `xgboost`, `lightgbm`, `torch` (via `deep_models.py`), `imbalanced-learn`, `shap`, `openai`.
- Central managers: `result_manager.get_result_manager()` (file saving), `report_manager.py` (report assembly). Use these helpers instead of writing files directly to output directories.

**Tests, debugging & quick checks**
- Use `python model_pipeline.py --config test_config.yaml --dry-run` to validate configs without running heavy jobs.
- For Streamlit UI errors, run `streamlit run streamlit_app.py` and check console logs; many imports are optional and will warn when missing.
- Logs go to `project.output_dir` as configured; `utils/logger.py` is used for logger setup.

**When to ask the maintainer**
- If you need a new persistent artifact location or change the `MODELS` output contract (the nested results dict), ask before modifying `report_manager.py` or downstream report templates.

If any section is unclear or you'd like more examples (e.g., adding a new `Torch` model or enabling XGBoost), tell me which part and I will expand with exact diffs.
