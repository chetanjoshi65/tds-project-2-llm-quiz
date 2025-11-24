"""
Find which Gemini models are available with your API key
"""

import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

print("=" * 70)
print("Available Gemini Models")
print("=" * 70)

models_to_test = []

for model in genai.list_models():
    if 'generateContent' in model.supported_generation_methods:
        print(f"\n✅ Model: {model.name}")
        print(f"   Display Name: {model.display_name}")
        print(f"   Description: {model.description[:100] if model.description else 'N/A'}")
        models_to_test.append(model.name)

print("\n" + "=" * 70)
print("Testing Models")
print("=" * 70)

for model_name in models_to_test[:3]:  # Test first 3 models
    print(f"\nTesting: {model_name}")
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content("Say 'Hello!'")
        print(f"✅ WORKS: {model_name}")
        print(f"   Response: {response.text[:50]}")
        break  # Stop after finding first working model
    except Exception as e:
        print(f"❌ FAILED: {model_name}")
        print(f"   Error: {str(e)[:100]}")

print("\n" + "=" * 70)
print(f"Recommended model to use in main.py:")
if models_to_test:
    print(f'llm = genai.GenerativeModel("{models_to_test[0]}")')
else:
    print("❌ No models available with this API key")
print("=" * 70)