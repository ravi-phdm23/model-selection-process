"""
Test the llm_explain module with Hugging Face
"""
import os
import sys
from dotenv import load_dotenv

# Load environment and set provider to huggingface
load_dotenv()
os.environ['LLM_PROVIDER'] = 'huggingface'

# Now import llm_explain (after setting env var)
import llm_explain

print("=" * 60)
print("TESTING llm_explain.py WITH HUGGING FACE")
print("=" * 60)

print(f"\nLLM Provider: {llm_explain.LLM_PROVIDER}")
print(f"HF Model: {llm_explain.HF_MODEL_NAME}")

# Test the client initialization
print("\nTesting _get_huggingface_client()...")
client, error = llm_explain._get_huggingface_client()
if error:
    print(f"ERROR: {error}")
    sys.exit(1)
else:
    print("[OK] Client initialized successfully")

# Test commentary generation with minimal data
print("\nTesting _generate_ai_commentary()...")
import pandas as pd

test_df = pd.DataFrame({
    'Feature': ['age', 'income', 'credit_score'],
    'Value': [45, 75000, 720],
    'SHAP Value': [0.234, 0.189, 0.156]
})

commentary, error = llm_explain._generate_ai_commentary(
    top_features_df=test_df,
    prediction_class=1,
    prediction_confidence=0.85,
    actual_label_text="Class 1",
    feature_reliability=None,
    reliability_score=None,
    reliability_bucket=None,
    sanity_ratio=None
)

if error:
    print(f"[FAIL] ERROR: {error}")
else:
    print("[OK] Commentary generated successfully!")
    print(f"\nCommentary:\n{commentary}\n")

print("=" * 60)
print("TEST COMPLETE")
print("=" * 60)
