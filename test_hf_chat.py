# Test HF with chat completion
import os
from dotenv import load_dotenv

load_dotenv()

try:
    from huggingface_hub import InferenceClient
    
    hf_token = os.getenv("HF_TOKEN")
    hf_model = os.getenv("HF_MODEL_NAME", "google/gemma-2-2b-it")
    
    print("Testing HF with chat completion...")
    print(f"Model: {hf_model}")
    
    client = InferenceClient(token=hf_token)
    
    messages = [
        {"role": "system", "content": "You are a helpful machine learning analyst."},
        {"role": "user", "content": "The model predicted Class 1 with 85% confidence. Top features: age=45, income=75000. Explain briefly why the model made this prediction."}
    ]
    
    response = client.chat_completion(
        messages=messages,
        model=hf_model,
        max_tokens=200,
        temperature=0.3,
    )
    
    print("SUCCESS!")
    print(f"Response: {response.choices[0].message.content}")
    
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
