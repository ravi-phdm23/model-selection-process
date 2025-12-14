"""
Test script for Hugging Face LLM integration
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("=" * 60)
print("HUGGING FACE INTEGRATION TEST")
print("=" * 60)

# Test 1: Check environment variables
print("\n1. Checking environment variables...")
hf_token = os.getenv("HF_TOKEN")
hf_model = os.getenv("HF_MODEL_NAME")
llm_provider = os.getenv("LLM_PROVIDER")

print(f"   LLM_PROVIDER: {llm_provider}")
print(f"   HF_MODEL_NAME: {hf_model}")
print(f"   HF_TOKEN: {'✓ Found' if hf_token else '✗ Missing'}")
if hf_token:
    print(f"   Token starts with: {hf_token[:10]}...")

# Test 2: Check if huggingface_hub is installed
print("\n2. Checking huggingface_hub installation...")
try:
    from huggingface_hub import InferenceClient
    print("   ✓ huggingface_hub is installed")
except ImportError as e:
    print(f"   ✗ huggingface_hub NOT installed")
    print(f"   Error: {e}")
    print("\n   To install, run:")
    print("   pip install huggingface_hub")
    exit(1)

# Test 3: Initialize client
print("\n3. Initializing Hugging Face client...")
try:
    client = InferenceClient(token=hf_token)
    print("   ✓ Client initialized successfully")
except Exception as e:
    print(f"   ✗ Failed to initialize client")
    print(f"   Error: {e}")
    exit(1)

# Test 4: Test simple text generation
print("\n4. Testing text generation with simple prompt...")
test_prompt = "Hello! Please respond with a brief greeting."
print(f"   Prompt: {test_prompt}")
print(f"   Model: {hf_model}")

try:
    response = client.text_generation(
        prompt=test_prompt,
        model=hf_model,
        max_new_tokens=50,
        temperature=0.3,
        return_full_text=False,
    )
    print("\n   ✓ Generation successful!")
    print(f"   Response: {response}")
except Exception as e:
    print(f"\n   ✗ Generation failed")
    print(f"   Error type: {type(e).__name__}")
    print(f"   Error message: {str(e)}")

    # Try to provide more details
    if hasattr(e, 'response'):
        print(f"   HTTP Status: {e.response.status_code if hasattr(e.response, 'status_code') else 'N/A'}")
        print(f"   Response text: {e.response.text if hasattr(e.response, 'text') else 'N/A'}")

# Test 5: Test with SHAP-style prompt (similar to actual use case)
print("\n5. Testing with SHAP explanation prompt...")
shap_prompt = """System: You are a machine-learning analyst who explains SHAP feature contributions.

User: Model prediction summary:
- Model Prediction: Class 1 (Confidence: 85%)
- Actual Value: Class 1

Top 3 contributing features:
1. age (value=45, shap=0.234)
2. income (value=75000, shap=0.189)
3. credit_score (value=720, shap=0.156)

Write a brief explanation (2-3 sentences) of why the model made this prediction.