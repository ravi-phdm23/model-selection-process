"""
Test multiple Hugging Face models to find the best one
"""
import os
from dotenv import load_dotenv

load_dotenv()

try:
    from huggingface_hub import InferenceClient
except ImportError:
    print("ERROR: huggingface_hub not installed. Run: pip install huggingface_hub")
    exit(1)

hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    print("ERROR: HF_TOKEN not found in .env file")
    exit(1)

client = InferenceClient(token=hf_token)

# Test different models
test_models = [
    "google/gemma-2-2b-it",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "microsoft/Phi-3-mini-4k-instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "HuggingFaceH4/zephyr-7b-beta",
    "mistralai/Mistral-7B-Instruct-v0.1",
]

test_messages = [
    {"role": "user", "content": "Explain briefly: the model predicted Class 1 with 85% confidence. Top features: age=45, income=75000. Why?"}
]

print("=" * 70)
print("TESTING HUGGING FACE MODELS")
print("=" * 70)

results = []

for model in test_models:
    print(f"\nTesting: {model}")
    print("-" * 70)

    try:
        response = client.chat_completion(
            messages=test_messages,
            model=model,
            max_tokens=150,
            temperature=0.3,
        )

        content = response.choices[0].message.content
        print(f"[SUCCESS] Response length: {len(content)} chars")
        print(f"Preview: {content[:200]}...")
        results.append({
            "model": model,
            "status": "SUCCESS",
            "response": content[:200]
        })

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        print(f"[FAILED] {error_msg}")
        results.append({
            "model": model,
            "status": "FAILED",
            "error": error_msg
        })

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

successful = [r for r in results if r["status"] == "SUCCESS"]
failed = [r for r in results if r["status"] == "FAILED"]

print(f"\nSuccessful: {len(successful)}/{len(test_models)}")
print(f"Failed: {len(failed)}/{len(test_models)}")

if successful:
    print("\n[RECOMMENDED MODELS]")
    for r in successful:
        print(f"  - {r['model']}")

    print(f"\n[BEST CHOICE]: {successful[0]['model']}")
    print("\nUpdate your .env file with:")
    print(f'HF_MODEL_NAME = "{successful[0]["model"]}"')
else:
    print("\n[WARNING] No models worked! Consider using OpenAI instead.")
    print("Update .env: LLM_PROVIDER = \"openai\"")
