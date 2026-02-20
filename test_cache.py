import os
import requests
import time
from dotenv import load_dotenv

load_dotenv()

URL = os.environ.get("URL", "https://cursor.gursimanoor.com/v1/chat/completions")
# Replace with a long dummy text block to ensure it hits the 1024 token minimum for caching
LONG_SYSTEM = "You are a helpful programming assistant. " * 300 

payload = {
    "model": "claude-3-5-sonnet-20241022",
    "messages": [
        {"role": "system", "content": LONG_SYSTEM},
        {"role": "user", "content": "What is 2+2?"}
    ],
    "stream": False
}

def make_request():
    print("Sending request...")
    api_key = os.environ.get('GATEWAY_API_KEY', '')
    if not api_key:
        print("Warning: GATEWAY_API_KEY not found in env, using dummy key")
        api_key = "dummy"

    resp = requests.post(URL, json=payload, headers={"Authorization": f"Bearer {api_key}"})
    if resp.status_code != 200:
        print(f"Error: {resp.status_code} {resp.text}")
        return None
        
    j = resp.json()
    usage = j.get("usage", {})
    
    print("\n--- Request Completed ---")
    print(f"Output: {j.get('choices')[0].get('message').get('content')}")
    print(f"Total Tokens: {usage.get('total_tokens')}")
    print(f"Input Tokens: {usage.get('prompt_tokens')}")
    print(f"Output Tokens: {usage.get('completion_tokens')}")
    
    # Extract Anthropic-specific caching stats from the payload
    # Note: Gateway maps anthropic usage differently in standard vs openAI format
    if "prompt_tokens_details" in usage:
        print("\nCache Details:")
        print(f"  Cached Tokens Matched:   {usage['prompt_tokens_details'].get('cached_tokens', 0)}")
    elif "cache_creation_input_tokens" in usage:
        print("\nCache Details:")
        print(f"  Tokens Written to Cache: {usage.get('cache_creation_input_tokens', 0)}")
        print(f"  Tokens Read from Cache:  {usage.get('cache_read_input_tokens', 0)}")
    else:
        print("\nCache Details:")
        print("  (No cache details found in standard payload)")
        
    return usage

print("=== RUN 1 (Should create cache) ===")
make_request()

print("\nWaiting 2 seconds...")
time.sleep(2)

print("\n=== RUN 2 (Should Read from cache) ===")
make_request()
