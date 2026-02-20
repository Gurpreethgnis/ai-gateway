import requests
import time

API_KEY = "bv7CZqbgqeWkS67+3Yl/N1bBWku8T7QPfZyNxfUKqMdzQG+ZTpwN0OXAZwORt4IL"
URL = "https://cursor.gursimanoor.com/v1/chat/completions"

large_system_prompt = "You are a helpful coding assistant. " * 300

payload = {
    "model": "claude-3-5-sonnet-20241022",
    "messages": [
        {"role": "system", "content": large_system_prompt},
        {"role": "user", "content": "Hi, what is 2+2?"}
    ],
    "max_tokens": 50,
    "stream": False
}

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

print("=== TEST 1: First Request (Should WRITE to Anthropic Cache) ===")
try:
    resp1 = requests.post(URL, json=payload, headers=headers)
    resp1.raise_for_status()
    data1 = resp1.json()
    print("Success! Response Choice:")
    print(data1["choices"][0]["message"]["content"])
    print("\nUSAGE STATS:")
    usage1 = data1.get("usage", {})
    print(f"- Total Tokens: {usage1.get('total_tokens')}")
    print(f"- Input Tokens Billed: {usage1.get('prompt_tokens')}")
    if "prompt_tokens_details" in usage1:
        print(f"- Cached Tokens: {usage1['prompt_tokens_details'].get('cached_tokens')}")
except Exception as e:
    print(f"Failed: {e}")
    if hasattr(e, 'response') and e.response is not None:
        print(e.response.text)

print("\n=== Waiting 3 seconds... ===")
time.sleep(3)

print("\n=== TEST 2: Second Request (Should READ from Anthropic Cache) ===")
try:
    resp2 = requests.post(URL, json=payload, headers=headers)
    resp2.raise_for_status()
    data2 = resp2.json()
    print("Success! Response Choice:")
    print(data2["choices"][0]["message"]["content"])
    print("\nUSAGE STATS:")
    usage2 = data2.get("usage", {})
    print(f"- Total Tokens: {usage2.get('total_tokens')}")
    print(f"- Input Tokens Billed: {usage2.get('prompt_tokens')}")
    if "prompt_tokens_details" in usage2:
        print(f"- Cached Tokens: {usage2['prompt_tokens_details'].get('cached_tokens')}")
except Exception as e:
    print(f"Failed: {e}")
    if hasattr(e, 'response') and e.response is not None:
        print(e.response.text)
