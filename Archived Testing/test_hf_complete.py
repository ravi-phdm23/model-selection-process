# Test completion for SHAP prompt
import os
from dotenv import load_dotenv

load_dotenv()

try:
    from huggingface_hub import InferenceClient
    
    hf_token = os.getenv("HF_TOKEN")
    hf_model = os.getenv("HF_MODEL_NAME", "google/gemma-2-2b-it")
    
    print("Testing HF with SHAP prompt...")
    print(f"Model: {hf_model}")
    
    client = InferenceClient(token=hf_token)
    
    shap_prompt = """You are explaining a model prediction. The model predicted Class 1 with 85% confidence (actual was Class 1). Top features: age=45 (shap=0.234), income=75000 (shap=0.189). Explain briefly why."""
    
    response = client.text_generation(
        prompt=shap_prompt,
        model=hf_model,
        max_new_tokens=100,
        temperature=0.3,
        return_full_text=False,
    )
    
    print("SUCCESS!")
    print(f"Response: {response}")
    
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
