#!/usr/bin/env python3
"""
Test AI Gateway cost savings implementation.
Usage: python test_gateway.py
"""
import os
import sys
import json
import time

try:
    import requests
except ImportError:
    print("❌ requests library not found. Install with: pip install requests")
    sys.exit(1)

# Configuration
GATEWAY_URL = os.getenv("GATEWAY_URL", "https://your-gateway.railway.app")
GATEWAY_API_KEY = os.getenv("GATEWAY_API_KEY")

if not GATEWAY_API_KEY:
    print("❌ GATEWAY_API_KEY not set!")
    print("Set it: export GATEWAY_API_KEY='your-key-here'")
    print("Or:     $env:GATEWAY_API_KEY = 'your-key-here'  (PowerShell)")
    sys.exit(1)

# Long system prompt to trigger constitution injection (>1024 chars)
LONG_SYSTEM_PROMPT = """
You are an expert software engineer working on a production system.
Follow these guidelines:
1. Write clean, maintainable code with proper error handling
2. Use type hints and documentation
3. Follow language-specific best practices
4. Prefer composition over inheritance
5. Keep functions focused and single-purpose
6. Optimize for readability and maintainability
7. Consider edge cases and error conditions
8. Write comprehensive tests for critical paths
9. Use descriptive variable and function names
10. Follow the principle of least surprise
11. Document complex logic and business rules
12. Use consistent formatting and style
13. Avoid premature optimization
14. Consider performance implications
15. Think about security from the start
16. Make code reviewable and understandable
17. Use version control best practices
18. Write self-documenting code when possible
19. Add comments only when necessary to explain why, not what
20. Keep dependencies minimal and well-justified
This system prompt is intentionally long to trigger Anthropic prompt caching 
with the platform constitution blocks for maximum cost savings.
""".strip()

def send_request(request_num):
    """Send a chat completion request."""
    url = f"{GATEWAY_URL}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GATEWAY_API_KEY}",
        "Content-Type": "application/json",
    }
    
    payload = {
        "model": "claude-sonnet-4-0",
        "messages": [
            {"role": "user", "content": "Say 'Hello from AI Gateway cost savings test!'"}
        ],
        "system": LONG_SYSTEM_PROMPT,
        "max_tokens": 100
    }
    
    print(f"\n[TEST {request_num}] Sending request {'(cache MISS expected)' if request_num == 1 else '(cache HIT expected for constitution)'}...")
    start_time = time.time()
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Success! ({elapsed:.2f}s)")
            print(f"Response: {data['choices'][0]['message']['content']}")
            
            if 'usage' in data:
                usage = data['usage']
                print(f"Usage: input={usage.get('input_tokens')} output={usage.get('output_tokens')}")
                if 'cache_read_input_tokens' in usage:
                    print(f"Cache hits: {usage.get('cache_read_input_tokens')} tokens read from cache!")
            
            return data
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return None

def check_metrics():
    """Check gateway metrics."""
    print("\n" + "=" * 60)
    print("Checking Metrics for Savings...")
    print("=" * 60)
    
    try:
        response = requests.get(f"{GATEWAY_URL}/metrics", timeout=10)
        if response.status_code == 200:
            metrics = response.text
            
            # Extract relevant metrics
            print("\n📊 Token Savings Metrics:")
            for line in metrics.split('\n'):
                if 'gateway_tokens_saved_total' in line and not line.startswith('#'):
                    print(f"  {line}")
            
            print("\n📊 Cache Hit Metrics:")
            for line in metrics.split('\n'):
                if 'gateway_cache_hits_total' in line and not line.startswith('#'):
                    print(f"  {line}")
            
            print("\n📊 Prompt Cache Metrics:")
            for line in metrics.split('\n'):
                if 'gateway_prompt_cache_tokens' in line and not line.startswith('#'):
                    print(f"  {line}")
        else:
            print(f"⚠️  Could not fetch metrics (status {response.status_code})")
    except Exception as e:
        print(f"⚠️  Could not fetch metrics: {e}")

def main():
    print("=" * 60)
    print("AI Gateway Cost Savings Test")
    print("=" * 60)
    print(f"\nGateway URL: {GATEWAY_URL}")
    print(f"System prompt length: {len(LONG_SYSTEM_PROMPT)} chars")
    
    # Test 1: First request
    response1 = send_request(1)
    if not response1:
        print("\n❌ First request failed. Check your GATEWAY_URL and GATEWAY_API_KEY")
        return
    
    time.sleep(2)
    
    # Test 2: Identical request (should hit prompt cache)
    response2 = send_request(2)
    
    # Check metrics
    check_metrics()
    
    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)
    print("\n📋 What to look for:")
    print("1. Second request should be faster (prompt cache hit)")
    print("2. Look for cache_read_input_tokens > 0 in usage on 2nd request")
    print("3. Check Railway logs for 'Gateway reduction' messages")
    print("4. Metrics should show increasing token savings")
    print("\n💰 Expected savings: 60-80% on cached prompts!\n")

if __name__ == "__main__":
    main()
