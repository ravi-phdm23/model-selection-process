import os
import numbers
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
import shap # For the Explanation object type hint
import numpy as np
from typing import Dict, Union, Tuple

# Load environment variables from .env file
load_dotenv()

# LLM Configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "huggingface").lower()  # Default to huggingface
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")
HF_MODEL_NAME = os.getenv("HF_MODEL_NAME", "google/gemma-2-2b-it")
AI_COMMENTARY_FEATURE_COUNT = 10 # Number of features to send to the LLM

# --- Reliability Score Computation Functions ---

def compute_feature_weights(avg_ranks: Dict[str, float], std_ranks: Dict[str, float]) -> Dict[str, float]:
    """
    Compute combined weight for each feature based on rank position and stability.
    
    Args:
        avg_ranks: Dictionary of {feature_name: avg_rank_i}
        std_ranks: Dictionary of {feature_name: std_rank_i}
    
    Returns:
        Dictionary of {feature_name: W_i} where W_i = W_rank_i * W_stab_i
    """
    weights = {}
    for feature in avg_ranks.keys():
        avg_rank = avg_ranks.get(feature, 999)
        std_rank = std_ranks.get(feature, 999)
        
        # Rank weight: features with lower avg_rank (higher importance) get higher weight
        W_rank = 1.0 / (1.0 + avg_rank)
        
        # Stability weight: features with lower std_rank (more stable) get higher weight
        W_stab = 1.0 / (1.0 + std_rank)
        
        # Combined weight
        weights[feature] = W_rank * W_stab
    
    return weights


def compute_explanatory_robustness(W: Dict[str, float]) -> float:
    """
    Compute explanatory robustness as the mean of feature weights.
    
    Args:
        W: Dictionary of {feature_name: W_i}
    
    Returns:
        ER: Mean of all W_i values
    """
    if not W:
        return 0.0
    return float(np.mean(list(W.values())))


def compute_signal_weight(sanity_ratio: float) -> float:
    """
    Compute signal weight from sanity ratio, capped at 3.
    
    Args:
        sanity_ratio: Ratio of real feature importance to random noise baseline
    
    Returns:
        W_signal: Normalized weight in [0, 1]
    """
    if sanity_ratio is None or np.isnan(sanity_ratio):
        return 0.0
    
    # Cap at 3 and normalize
    capped_ratio = min(sanity_ratio, 3.0)
    W_signal = capped_ratio / 3.0
    
    return W_signal


def compute_reliability_score(ER: float, sanity_ratio: float) -> float:
    """
    Compute final reliability score combining explanatory robustness and signal quality.
    
    Args:
        ER: Explanatory robustness (mean feature weight)
        sanity_ratio: Ratio of real feature importance to random noise
    
    Returns:
        Reliability score in [0, 1]
    """
    W_signal = compute_signal_weight(sanity_ratio)
    score = ER * W_signal
    
    # Ensure result is in [0, 1]
    return max(0.0, min(1.0, score))


def get_reliability_bucket(score: float) -> str:
    """
    Classify reliability score into interpretable buckets.
    
    Args:
        score: Reliability score in [0, 1]
    
    Returns:
        Reliability bucket label
    """
    # Use the same fallback thresholds as the app's unified bucket helper
    # (these are used when batch quantiles aren't available).
    if score is None or (isinstance(score, float) and np.isnan(score)):
        return "Unknown"
    if score >= 0.60:
        return "Reliable"
    elif score >= 0.40:
        return "Moderately Reliable"
    elif score >= 0.20:
        return "Questionable"
    else:
        return "Unreliable"


# --- Helper functions adapted from your shap_app.py ---

def _format_feature_value(value):
    """Formats feature values for the prompt."""
    if isinstance(value, numbers.Number):
        # Check if it's an integer or float
        if isinstance(value, float) and not value.is_integer():
             return f"{float(value):.4f}"
        return str(value)
    return str(value)

def _summarize_top_features(top_features_df):
    """Converts the top features DataFrame into a numbered list string."""
    lines = []
    for idx, (_, row) in enumerate(top_features_df.iterrows(), start=1):
        feature_name = str(row["Feature"])
        feature_value = _format_feature_value(row["Value"])
        shap_value = _format_feature_value(row["SHAP Value"])
        lines.append(f"{idx}. {feature_name} (value={feature_value}, shap={shap_value})")
    return "\n".join(lines)

@st.cache_resource
def _get_openai_client():
    """
    Initializes and caches the OpenAI client.
    Reads the API key from the .env file.
    """
    # Try to load again, just in case
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # --- NEW, more helpful error ---
        cwd = os.getcwd()
        env_path = os.path.join(cwd, '.env')
        file_exists = os.path.exists(env_path)

        error_message = (
            "**OpenAI API key missing.** `os.getenv('OPENAI_API_KEY')` returned `None`.\n\n"
            "**Please check the following:**\n\n"
            f"1. **File Location:** Is your `.env` file in this exact folder?\n   `{env_path}`\n   (File exists: **{file_exists}**)\n\n"
            "2. **File Format:** Does your `.env` file contain a line *without quotes*?\n   **Correct:** `OPENAI_API_KEY=sk-yourkey...`\n   **Incorrect:** `OPENAI_API_KEY=\"sk-yourkey...\"`\n\n"
            "3. **Restart App:** Have you **restarted** your Streamlit app (e.g., `streamlit run streamlit_app.py`) after creating/editing the `.env` file?"
        )
        return None, error_message
        # --- END NEW ERROR ---
    try:
        client = OpenAI(api_key=api_key)
        return client, None
    except Exception as exc:
        return None, f"Unable to initialize OpenAI client: {exc}"

@st.cache_resource
def _get_huggingface_client():
    """
    Initializes and caches the Hugging Face Inference API client.
    Reads the API token from the .env file.
    """
    try:
        from huggingface_hub import InferenceClient
    except ImportError:
        return None, "Hugging Face Hub library not installed. Run: pip install huggingface_hub"

    load_dotenv()

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        cwd = os.getcwd()
        env_path = os.path.join(cwd, '.env')
        file_exists = os.path.exists(env_path)

        error_message = (
            "**Hugging Face token missing.** `os.getenv('HF_TOKEN')` returned `None`.\n\n"
            "**Please check the following:**\n\n"
            f"1. **File Location:** Is your `.env` file in this exact folder?\n   `{env_path}`\n   (File exists: **{file_exists}**)\n\n"
            "2. **File Format:** Does your `.env` file contain a line *without quotes*?\n   **Correct:** `HF_TOKEN=hf_yourtoken...`\n   **Incorrect:** `HF_TOKEN=\"hf_yourtoken...\"`\n\n"
            "3. **Restart App:** Have you **restarted** your Streamlit app after creating/editing the `.env` file?"
        )
        return None, error_message

    try:
        client = InferenceClient(token=hf_token)
        return client, None
    except Exception as exc:
        return None, f"Unable to initialize Hugging Face client: {exc}"

def _generate_ai_commentary(
    top_features_df,
    prediction_class,
    prediction_confidence,
    actual_label_text,
    feature_reliability=None,
    reliability_score=None,
    reliability_bucket=None,
    sanity_ratio=None,
    enable_row_reliability=False
):
    """
    Sends the prompt to the LLM API (OpenAI or Hugging Face) and returns the text response.

    Args:
        top_features_df: DataFrame with top contributing features
        prediction_class: Predicted class (0 or 1)
        prediction_confidence: Confidence of the prediction
        actual_label_text: Actual class label as text
        feature_reliability: Optional dict with feature reliability metrics (avg_rank, std_rank)
        reliability_score: Overall reliability score [0, 1]
        reliability_bucket: Classification (Reliable/Moderately Reliable/Questionable/Unreliable)
        sanity_ratio: Sanity ratio value for reliability assessment
        enable_row_reliability: Boolean flag to enable row-specific reliability analysis in prompt
    """
    if top_features_df.empty:
        return None, "No feature contributions available for commentary."

    # Get the appropriate client based on provider
    if LLM_PROVIDER == "openai":
        client, client_error = _get_openai_client()
    else:  # Default to huggingface
        client, client_error = _get_huggingface_client()

    if client is None:
        return None, client_error
        
    feature_summary = _summarize_top_features(top_features_df)
    
    # Determine predicted class text
    pred_class_text = f"Class {prediction_class} (Confidence: {prediction_confidence:.2%})"

    # Build reliability context based on mode
    reliability_context = ""
    global_reliability_context = ""
    
    if feature_reliability:
        sanity_ratio = feature_reliability.get('_sanity_ratio', None)
        
        # Global reliability context (SHAP Reliability Test - always available if feature_reliability exists)
        if sanity_ratio is not None:
            global_reliability_context = "\n\nSHAP Reliability Test Results:\n"
            global_reliability_context += f"- Sanity Ratio: {sanity_ratio:.3f} (real features vs random baseline)\n"
            global_reliability_context += "  (>2.0 = strong signal, 1.5-2.0 = moderate, <1.5 = weak signal)\n"
        
        # Row-specific reliability context (only if enabled)
        if enable_row_reliability:
            reliability_context = "\n\nRow-Specific Reliability Metrics:\n"
            
            for feat_name, metrics in feature_reliability.items():
                if feat_name == '_sanity_ratio':
                    continue
                if isinstance(metrics, dict):
                    avg_rank = metrics.get('avg_rank', 'N/A')
                    std_rank = metrics.get('std_rank', 'N/A')
                    reliability_context += f"- {feat_name}: avg_rank={avg_rank}, std_rank={std_rank}\n"
            
            if reliability_score is not None and reliability_bucket is not None:
                reliability_context += f"\nRow Reliability Score: {reliability_score:.3f}\n"
                reliability_context += f"Reliability Classification: {reliability_bucket}\n"

    # Build system message
    system_message = (
        "You are a machine-learning analyst who explains SHAP feature contributions "
        "with rigorous, critical reasoning inspired by David Deutsch. "
        "Your job is not to sound confident—it is to expose which parts of the explanation "
        "are genuinely supported by evidence and which are fragile. "
        "Distinguish prediction from explanation, and explanation from mere correlation."
    )
    
    # Build user prompt based on mode
    if enable_row_reliability and reliability_score is not None and reliability_bucket is not None:
        # Mode: Row-specific reliability enabled (3 paragraphs)
        user_prompt = (
            "Model prediction summary:\n"
            f"- Model Prediction: {pred_class_text}\n"
            f"- Actual Value: {actual_label_text}\n\n"
            f"Top {len(top_features_df)} contributing features (highest impact first):\n"
            f"{feature_summary}\n"
            f"{global_reliability_context}"
            f"{reliability_context}\n\n"
            "Write three short paragraphs (under 260 words total). Do NOT label them as 'Paragraph 1', 'Paragraph 2', etc.\n\n"
            "First paragraph: Explain *why* the model arrived at this probability. "
            "Describe how each major feature (by SHAP value) pushed the prediction up or down. "
            "Focus on causal structure, not storytelling.\n\n"
            "Second paragraph: Compare the prediction with the actual outcome. "
            "If prediction matches actual, explain how the feature contributions form a coherent explanation. "
            "If it differs, identify which feature patterns or interactions may have misled the model.\n\n"
            "Third paragraph: Evaluate the *quality* of this explanation using:\n"
            f"  - Row Reliability Score: {reliability_score:.3f} (computed from feature weights and signal quality)\n"
            f"  - Reliability Classification: **{reliability_bucket}**\n"
            f"  - Sanity Ratio: {sanity_ratio:.3f}\n\n"
            "Interpret these metrics:\n"
            "  • avg_rank shows expected explanatory position of each feature across random trials\n"
            "  • std_rank measures stability: low (<2) = hard-to-vary (robust), high (>3) = easy-to-vary (fragile)\n"
            "  • sanity_ratio tests signal vs noise: >2 is good, <1.5 is weak\n"
            "  • reliability_score combines all metrics into [0,1] scale\n\n"
            f"Based on the {reliability_bucket} classification, state decisively whether this explanation should be TRUSTED, "
            "interpreted CAUTIOUSLY, or treated as UNRELIABLE. Justify using the stability metrics—not intuition. "
            "Be explicit about what the evidence rules out and what remains uncertain."
        )
    else:
        # Mode: Row-specific reliability disabled (2 paragraphs)
        user_prompt = (
            "Model prediction summary:\n"
            f"- Model Prediction: {pred_class_text}\n"
            f"- Actual Value: {actual_label_text}\n\n"
            f"Top {len(top_features_df)} contributing features (highest impact first):\n"
            f"{feature_summary}\n"
            f"{global_reliability_context}\n\n"
            "Write two short paragraphs (under 180 words total). Do NOT label them as 'Paragraph 1', 'Paragraph 2', etc.\n\n"
            "First paragraph: Explain *why* the model arrived at this probability. "
            "Describe how each major feature (by SHAP value) pushed the prediction up or down. "
            "Focus on causal structure, not storytelling."
        )
        
        # Add sanity ratio context if available
        if sanity_ratio is not None:
            user_prompt += (
                " Reference the Sanity Ratio to comment on overall signal quality (whether real features "
                "outperform random baseline).\n\n"
            )
        else:
            user_prompt += "\n\n"
        
        user_prompt += (
            "Second paragraph: Compare the prediction with the actual outcome. "
            "If prediction matches actual, explain how the feature contributions form a coherent explanation. "
            "If it differs, identify which feature patterns or interactions may have misled the model."
        )
        
        # Add note about SHAP Reliability Test if available
        if sanity_ratio is not None:
            user_prompt += (
                f" Note that the Sanity Ratio of {sanity_ratio:.3f} indicates "
                f"{'strong' if sanity_ratio > 2.0 else 'moderate' if sanity_ratio > 1.5 else 'weak'} "
                "signal quality in the feature explanations."
            )
    
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt}
    ]

    # Retry logic for transient failures
    max_retries = 2
    retry_count = 0

    while retry_count <= max_retries:
        try:
            if LLM_PROVIDER == "openai":
                # OpenAI API call
                response = client.chat.completions.create(
                    model=OPENAI_MODEL_NAME,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=400,
                    timeout=30.0,
                )
                content = response.choices[0].message.content.strip()
            else:
                # Hugging Face API call using chat_completion
                response = client.chat_completion(
                    messages=messages,
                    model=HF_MODEL_NAME,
                    max_tokens=400,
                    temperature=0.3,
                )

                # Handle different response formats
                if hasattr(response, 'choices') and len(response.choices) > 0:
                    content = response.choices[0].message.content.strip()
                elif isinstance(response, str):
                    content = response.strip()
                else:
                    raise ValueError(f"Unexpected response format: {type(response)}")

            return content, None

        except StopIteration as exc:
            # StopIteration is a known issue with some HF models - retry
            retry_count += 1
            if retry_count > max_retries:
                error_msg = f"{LLM_PROVIDER.upper()} StopIteration error after {max_retries} retries. Try a different model."
                return None, error_msg
            continue

        except Exception as exc:
            retry_count += 1
            error_msg = f"{LLM_PROVIDER.upper()} request failed: {type(exc).__name__}: {str(exc)}"

            if retry_count > max_retries:
                return None, error_msg
            # For other exceptions, don't retry
            if not isinstance(exc, (ConnectionError, TimeoutError)):
                return None, error_msg
            continue

    return None, f"{LLM_PROVIDER.upper()} request failed after {max_retries} retries"

# --- Main public function to be called by streamlit_app.py ---

def get_llm_explanation(
    explanation: shap.Explanation, 
    actual_target: any, 
    prediction_prob_class_1: float, 
    feature_reliability: dict = None,
    enable_row_reliability: bool = False
) -> Tuple[Union[str, None], Union[str, None], Union[Dict, None]]:
    """
    Takes a SHAP Explanation object and generates a natural language explanation with reliability metrics.
    
    Args:
        explanation: SHAP Explanation object
        actual_target: Actual target value
        prediction_prob_class_1: Predicted probability for class 1
        feature_reliability: Optional dict with reliability metrics for features
                            Format: {feature_name: {'avg_rank': X, 'std_rank': Y}, '_sanity_ratio': Z}
        enable_row_reliability: Boolean to enable row-specific reliability analysis in the explanation
    
    Returns:
        Tuple of (commentary, error, reliability_metrics):
            - commentary: str or None - The explanation text
            - error: str or None - Error message if failed
            - reliability_metrics: dict or None - Computed reliability scores and metrics
    """
    
    try:
        # 1. Convert SHAP Explanation to the required DataFrame format
        contributions_df = pd.DataFrame({
            'Feature': explanation.feature_names,
            'Value': explanation.data,
            'SHAP Value': explanation.values
        })
        
        # 2. Sort by impact
        contributions_df['Absolute Impact'] = np.abs(contributions_df['SHAP Value'])
        contributions_df = contributions_df.sort_values(by='Absolute Impact', ascending=False).drop(columns=['Absolute Impact'])
        
        # 3. Get top features for the prompt
        top_features_for_ai = contributions_df.head(AI_COMMENTARY_FEATURE_COUNT)
        
        # 4. Determine prediction class and confidence
        prediction_class = 1 if prediction_prob_class_1 >= 0.5 else 0
        prediction_confidence = prediction_prob_class_1 if prediction_class == 1 else 1 - prediction_prob_class_1
        
        # 5. Format actual target
        actual_label_text = f"Class {actual_target}"

        # 6. Compute reliability score if metrics available
        reliability_score = None
        reliability_bucket = None
        reliability_metrics_output = None
        
        if feature_reliability:
            # Extract avg_rank and std_rank for features
            avg_ranks = {}
            std_ranks = {}
            
            for feat_name, metrics in feature_reliability.items():
                if feat_name != '_sanity_ratio' and isinstance(metrics, dict):
                    avg_ranks[feat_name] = metrics.get('avg_rank', 999)
                    std_ranks[feat_name] = metrics.get('std_rank', 999)
            
            sanity_ratio = feature_reliability.get('_sanity_ratio', None)
            
            if avg_ranks and std_ranks and sanity_ratio is not None:
                # Compute feature weights
                feature_weights = compute_feature_weights(avg_ranks, std_ranks)
                
                # Compute explanatory robustness
                ER = compute_explanatory_robustness(feature_weights)
                
                # Compute final reliability score
                reliability_score = compute_reliability_score(ER, sanity_ratio)
                
                # Get reliability bucket
                reliability_bucket = get_reliability_bucket(reliability_score)
                
                # Build output metrics
                reliability_metrics_output = {
                    'reliability_score': reliability_score,
                    'reliability_bucket': reliability_bucket,
                    'explanatory_robustness': ER,
                    'sanity_ratio': sanity_ratio,
                    'feature_weights': feature_weights,
                    'per_feature_metrics': {
                        feat: {
                            'avg_rank': avg_ranks.get(feat),
                            'std_rank': std_ranks.get(feat),
                            'weight': feature_weights.get(feat)
                        }
                        for feat in avg_ranks.keys()
                    }
                }

        # 7. Call the generator
        sanity_ratio_val = feature_reliability.get('_sanity_ratio') if feature_reliability else None
        commentary, error = _generate_ai_commentary(
            top_features_for_ai,
            prediction_class=prediction_class,
            prediction_confidence=prediction_confidence,
            actual_label_text=actual_label_text,
            feature_reliability=feature_reliability,
            reliability_score=reliability_score,
            reliability_bucket=reliability_bucket,
            sanity_ratio=sanity_ratio_val,
            enable_row_reliability=enable_row_reliability
        )
        
        if error:
            return None, error, None
        else:
            return commentary, None, reliability_metrics_output

    except Exception as e:
        return None, f"An error occurred preparing data for the LLM: {e}", None

