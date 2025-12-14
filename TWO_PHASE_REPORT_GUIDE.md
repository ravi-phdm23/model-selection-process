# Two-Phase Results Section Generation Guide

## Overview

The report generation system has been redesigned to follow a **two-phase approach** that gives you control, speed, and cost efficiency:

- **Phase 1: Assemble Artifacts** - Instant organization of all analysis outputs (no LLM, no cost)
- **Phase 2: Add Narratives** - Optional AI-generated commentary (uses LLM API)

This guide explains how to use both phases effectively.

---

## Phase 1: Assemble Results Artifacts

### What It Does

Phase 1 instantly organizes all your analysis outputs into a structured **Results section** formatted for academic publication:

- Creates proper LaTeX tables with `booktabs` package
- Arranges figures in academic layouts (grids, side-by-side)
- Structures content into 5 subsections:
  1. **Exploratory Data Analysis** (per dataset)
  2. **Model Performance Comparison** (across all datasets)
  3. **Feature Importance Analysis** (Global SHAP, per dataset)
  4. **Model Reliability and Stability** (per dataset)
  5. **Local SHAP Analysis Examples**

### Key Features

✅ **Instant execution** - No LLM calls, no waiting
✅ **Zero cost** - No API charges
✅ **Academic quality** - Matches top AI/Finance journal formatting
✅ **Complete structure** - All tables and figures included
✅ **Ready to use** - Can download and compile immediately

### How to Use in Streamlit

1. Navigate to the **📝 Report Generation** tab
2. Review available artifacts in the summary panel
3. Click **"📦 Assemble Artifacts"**
4. Preview the results in three views:
   - **Markdown Preview**: Human-readable format
   - **LaTeX Source**: Academic paper format
   - **Artifact Summary**: Breakdown of tables/figures
5. Download either format:
   - `results_artifacts_YYYYMMDD_HHMMSS.md`
   - `results_artifacts_YYYYMMDD_HHMMSS.tex`

### Output Structure

```latex
\section{Results}

\subsection{Exploratory Data Analysis}
\subsubsection{Dataset 1}
%%PLACEHOLDER_EDA_dataset1%%
[Summary statistics table]
[Distribution figures]

\subsubsection{Dataset 2}
%%PLACEHOLDER_EDA_dataset2%%
[Summary statistics table]
[Distribution figures]

\subsection{Model Performance Comparison}
%%PLACEHOLDER_BENCHMARK%%
[Performance table with resizebox]
[Comparison charts in 3x2 grid]

\subsection{Feature Importance Analysis}
\subsubsection{Dataset 1}
%%PLACEHOLDER_SHAP_dataset1%%
[Feature ranking table]
[Bar + Dot plots side-by-side]

\subsection{Model Reliability and Stability}
\subsubsection{Dataset 1}
%%PLACEHOLDER_RELIABILITY_dataset1%%
[Batch reliability table]
[Diagnostics table]

\subsection{Local SHAP Analysis Examples}
%%PLACEHOLDER_LOCAL%%
[Waterfall plots for example predictions]
```

### Placeholder System

Phase 1 inserts **placeholder markers** like `%%PLACEHOLDER_EDA_german_credit%%` where narratives can be added in Phase 2. These markers:

- Identify the location for commentary
- Specify the type (EDA, SHAP, etc.)
- Include dataset name for context
- Can be removed if narratives not needed

---

## Phase 2: Add AI-Generated Narratives (Optional)

### What It Does

Phase 2 generates **concise academic commentary** (100-180 words per subsection) that:

- Interprets the results shown in tables/figures
- Highlights key patterns and insights
- Provides domain-specific analysis (credit risk context)
- Uses academic language suitable for publication

### Key Features

✅ **Toggle on/off** - Enable only when you want commentary
✅ **Cost control** - Pay only for what you use
✅ **Concise output** - 100-180 words per narrative (not 400-800)
✅ **Editable** - Review and modify each narrative before download
✅ **Multiple providers** - OpenAI (GPT-4, GPT-3.5) or Anthropic (Claude)

### How to Use in Streamlit

1. After assembling artifacts, check **"Enable AI Commentary"**
2. Configure LLM settings:
   - **Provider**: OpenAI or Anthropic
   - **Model**: GPT-4, GPT-3.5-turbo, Claude Sonnet, etc.
   - **API Key**: Your provider API key
   - **Temperature**: 0.0-1.0 (recommend 0.3 for academic writing)
3. Click **"🎨 Generate Narratives"**
4. Review generated narratives (expandable section)
5. **Edit any narrative** using the text areas
6. Download the complete version:
   - `results_complete_YYYYMMDD_HHMMSS.md`
   - `results_complete_YYYYMMDD_HHMMSS.tex`

### Narrative Types

Each narrative type has a specific focus:

#### EDA Narratives (100-150 words)
- Dataset characteristics (features, observations, target distribution)
- Notable patterns (class imbalance, correlations)
- Data quality considerations

#### Benchmark Narratives (120-180 words)
- Best-performing models
- Model family comparisons
- Performance-complexity trade-offs
- Surprising results

#### SHAP Narratives (100-150 words)
- Most influential features
- Feature importance patterns
- Feature stability analysis
- Domain relevance (credit risk)

#### Reliability Narratives (100-150 words)
- Batch reliability metrics
- Diagnostic test interpretation (KS, Mann-Whitney, ROC AUC)
- Model trustworthiness assessment
- Deployment implications

#### Local SHAP Narratives (80-120 words)
- Purpose of local explanations
- Waterfall plot interpretation
- Example case insights
- Value for transparency

### Important Notes

⚠️ **DO NOT EXPLAIN METHODOLOGY** - Narratives focus only on interpreting results, not explaining methods (that's for your Methods section)

⚠️ **CONCISE LENGTH** - Each narrative is ~100-180 words, not full paragraphs (this saves tokens and cost)

⚠️ **EDITABLE** - You can modify any narrative before final download

---

## Cost Comparison

### Old Single-Phase Approach
- Generated full paper (Introduction, Methods, Results, Conclusion)
- Multiple long prompts (400-800 words each)
- Many LLM calls (10-20+ depending on datasets)
- **Estimated cost**: $2-5 per full report with GPT-4

### New Two-Phase Approach

**Phase 1 Only** (artifacts):
- Cost: **$0** (no LLM calls)
- Time: **Instant** (<5 seconds)
- Output: Complete Results section with all tables/figures

**Phase 1 + Phase 2** (with narratives):
- Cost: **~$0.50-1.50** per report with GPT-4 (concise narratives)
- Time: **30-60 seconds** (depends on # of subsections)
- Output: Complete Results section with commentary

**Savings**: 60-70% cost reduction while maintaining quality

---

## Academic LaTeX Formatting

The system generates LaTeX code matching top journal standards:

### Tables
```latex
\begin{table}[H]
\centering
\caption{Model Performance Comparison}
\label{tab:benchmark_results}
\resizebox{\textwidth}{!}{
\begin{tabular}{lcccccc}
\toprule
Model & Dataset & AUC-ROC & Accuracy & Precision & Recall & F1-Score \\
\midrule
XGBoost & German Credit & 0.7821 & 0.7400 & 0.6522 & 0.5769 & 0.6122 \\
Random Forest & German Credit & 0.7654 & 0.7200 & 0.6190 & 0.5385 & 0.5758 \\
\bottomrule
\end{tabular}
}
\end{table}
```

### Figures (Single)
```latex
\begin{figure}[H]
\centering
\includegraphics[width=0.8\textwidth]{path/to/figure.png}
\caption{Target distribution for German Credit dataset}
\label{fig:target_dist_german_credit}
\end{figure}
```

### Figures (Side-by-Side SHAP)
```latex
\begin{figure}[H]
\centering
\begin{tabular}{cc}
\includegraphics[width=0.48\textwidth]{bar_plot.png} &
\includegraphics[width=0.48\textwidth]{dot_plot.png} \\
(a) Bar Plot & (b) Dot Plot
\end{tabular}
\caption{SHAP feature importance for German Credit}
\label{fig:shap_german_credit}
\end{figure}
```

### Figures (Grid Layout)
```latex
\begin{figure}[H]
\centering
\begin{tabular}{ccc}
\includegraphics[width=0.32\textwidth]{auc.png} &
\includegraphics[width=0.32\textwidth]{accuracy.png} &
\includegraphics[width=0.32\textwidth]{precision.png} \\
(a) AUC-ROC & (b) Accuracy & (c) Precision \\
\includegraphics[width=0.32\textwidth]{recall.png} &
\includegraphics[width=0.32\textwidth]{f1.png} &
\includegraphics[width=0.32\textwidth]{ks.png} \\
(d) Recall & (e) F1-Score & (f) KS Statistic
\end{tabular}
\caption{Performance comparison across metrics}
\label{fig:benchmark_comparison}
\end{figure}
```

---

## Best Practices

### When to Use Phase 1 Only
- You want to write your own narrative commentary
- You're on a tight budget
- You need results quickly for a draft
- You prefer manual interpretation

### When to Use Phase 1 + Phase 2
- You want AI assistance with interpretation
- You need a complete section quickly
- You want concise academic commentary as a starting point
- You plan to edit narratives before final submission

### Workflow Recommendation

1. **Initial Draft**: Use Phase 1 only
2. **Review**: Check that all expected artifacts are present
3. **Compile**: Test LaTeX compilation
4. **Commentary**: Enable Phase 2 to generate narratives
5. **Edit**: Customize narratives to match your writing style
6. **Finalize**: Download complete version

---

## Troubleshooting

### "No artifacts found"
- Complete Steps 1-6 in the Research Lab tab first
- Check that analyses have generated output files in `results/` folder

### "LLM returned empty"
- Verify API key is correct
- Check API credits/quota
- Try reducing temperature or changing model

### "Placeholder not replaced"
- This means Phase 2 failed for that subsection
- Check the "failed" list in the UI
- Can manually add text where placeholder appears

### LaTeX compilation errors
- Ensure you have `booktabs` package: `\usepackage{booktabs}`
- Use `\usepackage{float}` for `[H]` placement
- Use `\usepackage{graphicx}` for figures
- Check that image paths exist

---

## File Structure

```
results/
├── eda/
│   ├── german_credit_summary_stats.csv
│   ├── german_credit_target_dist.csv
│   └── ...
├── figures/
│   ├── eda_german_credit_target_dist.png
│   ├── shap_german_credit_XGBoost_bar.png
│   └── ...
├── reliability/
│   ├── german_credit_rank_stability.csv
│   ├── german_credit_diagnostics.json
│   └── ...
└── benchmark/
    └── benchmark_results.csv

reports/
├── results_artifacts_20240115_143022.md
├── results_artifacts_20240115_143022.tex
├── results_complete_20240115_143530.md
└── results_complete_20240115_143530.tex
```

---

## API Reference

### `assemble_results_artifacts()`
```python
def assemble_results_artifacts(
    result_manager: ResultManager,
    include_placeholders: bool = True
) -> Dict[str, Any]
```

**Returns:**
- `latex`: LaTeX string for Results section
- `markdown`: Markdown preview
- `placeholders`: List of placeholder dicts
- `artifact_summary`: Counts of tables/figures per subsection

### `generate_results_narratives()`
```python
def generate_results_narratives(
    assembled_result: Dict[str, Any],
    result_manager: ResultManager,
    llm_config: Dict[str, Any]
) -> Dict[str, Any]
```

**LLM Config:**
```python
{
    'provider': 'openai',  # or 'anthropic'
    'api_key': 'sk-...',
    'model': 'gpt-4',
    'temperature': 0.3,
    'max_tokens': 500
}
```

**Returns:**
- `narratives`: Dict mapping placeholder → narrative text
- `success`: List of successful generations
- `failed`: List of failed generations
- `latex_with_narratives`: Complete LaTeX
- `markdown_with_narratives`: Complete markdown

---

## Support

For questions or issues:
1. Check artifact summary to ensure analyses completed
2. Review Streamlit console for error messages
3. Test LaTeX compilation with minimal document first
4. Verify API keys and credits for Phase 2

---

**Last Updated**: January 2024
**Version**: 2.0 (Two-Phase System)
