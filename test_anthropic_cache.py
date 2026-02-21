import os
import httpx
import time
import json

# Configuration
GATEWAY_URL = "https://cursor.gursimanoor.com/v1/chat/completions"
API_KEY = os.getenv("GATEWAY_API_KEY")

if not API_KEY:
    print("Error: GATEWAY_API_KEY environment variable not set.")
    exit(1)

# A large context string (~2000 tokens)
LARGE_CONTEXT = "Repeat this sentence 100 times: The quick brown fox jumps over the lazy dog. " * 50

payload = {
    "model": "claude-3-haiku-20240307",
    "messages": [
        {"role": "user", "content": LARGE_CONTEXT + "\n\nExplain the sentence above."}
    ],
    "max_tokens": 100,
    "stream": False
}

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

def run_test():
    print("--- FIRST REQUEST (Cache Miss) ---")
    t0 = time.time()
    resp1 = httpx.post(GATEWAY_URL, json=payload, headers=headers, timeout=60.0)
    t1 = time.time()
    
    if resp1.status_code != 200:
        print(f"Error: {resp1.status_code} - {resp1.text}")
        return

    data1 = resp1.json()
    usage1 = data1.get("usage", {})
    print(f"Time: {t1-t0:.2f}s")
    print(f"Usage: {json.dumps(usage1, indent=2)}")
    
    # Wait a bit for Anthropic to persist the cache (usually immediate but good to wait)
    print("\nWaiting 5 seconds...")
    time.sleep(5)
    
    print("\n--- SECOND REQUEST (Expected Cache Hit) ---")
    t2 = time.time()
    resp2 = httpx.post(GATEWAY_URL, json=payload, headers=headers, timeout=60.0)
    t3 = time.time()
    
    if resp2.status_code != 200:
        print(f"Error: {resp2.status_code} - {resp2.text}")
        return

    data2 = resp2.json()
    usage2 = data2.get("usage", {})
    print(f"Time: {t3-t2:.2f}s")
    print(f"Usage: {json.dumps(usage2, indent=2)}")
    
    cached = usage2.get("prompt_tokens_details", {}).get("cached_tokens", 0)
    if cached > 0:
        print(f"\n✅ SUCCESS: {cached} tokens were read from cache!")
    else:
        print("\n❌ FAILURE: No tokens were read from cache.")

if __name__ == "__main__":
    run_test()
